import argparse  # Added to convert dict hyperparameters back to Namespace when loading from checkpoint
import logging
import os


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # Added for saving predictions into a CSV file
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import torchvision
from torch.optim import AdamW
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from utils import setup_logging


class SimplePyramid(nn.Module):
    """Simple Feature Pyramid Network for multi-scale features (P3-P6)"""
    
    def __init__(self, in_ch=768, out_ch=256):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        
        # Lateral connections to reduce channels
        self.lateral_conv = nn.Conv2d(in_ch, out_ch, 1)
        
        # Pyramid levels
        self.p3_conv = nn.Conv2d(out_ch, out_ch, 3, padding=1)  # 74x74
        self.p4_conv = nn.Conv2d(out_ch, out_ch, 3, padding=1)  # 37x37
        self.p5_conv = nn.Conv2d(out_ch, out_ch, 3, padding=1)  # 18x18
        self.p6_conv = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)  # 9x9
        
    def forward(self, f14):
        """
        Args:
            f14: Feature tensor from ViT backbone [B, 768, 37, 37]
        Returns:
            Dict of pyramid features {P3, P4, P5, P6}
        """
        # Reduce channels
        f14_reduced = self.lateral_conv(f14)  # [B, 256, 37, 37]
        
        # P4 (same as backbone output)
        p4 = self.p4_conv(f14_reduced)  # [B, 256, 37, 37]
        
        # P3 (upsample)
        p3 = F.interpolate(f14_reduced, scale_factor=2, mode='bilinear', align_corners=False)
        p3 = self.p3_conv(p3)  # [B, 256, 74, 74]
        
        # P5 (downsample)
        p5 = F.max_pool2d(f14_reduced, kernel_size=2)
        p5 = self.p5_conv(p5)  # [B, 256, 18, 18]
        
        # P6 (further downsample)
        p6 = self.p6_conv(p5)  # [B, 256, 9, 9]
        
        return {
            'P3': p3,
            'P4': p4, 
            'P5': p5,
            'P6': p6
        }


class RetinaHead(nn.Module):
    """RetinaNet detection head for lesion detection"""
    
    def __init__(self, in_ch=256, num_classes=1, num_anchors=9, dropout_rate=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.dropout_rate = dropout_rate
        
        # Shared convolution towers with dropout
        self.cls_tower = self._make_tower(in_ch, 4)
        self.bbox_tower = self._make_tower(in_ch, 4)
        
        # Add dropout before final predictions
        self.cls_dropout = nn.Dropout2d(dropout_rate)
        self.bbox_dropout = nn.Dropout2d(dropout_rate)
        
        # Classification head
        self.cls_logits = nn.Conv2d(in_ch, num_anchors * num_classes, 3, padding=1)
        
        # Regression head
        self.bbox_pred = nn.Conv2d(in_ch, num_anchors * 4, 3, padding=1)
        
        # Initialize weights
        self._init_weights()
        
    def _make_tower(self, in_ch, num_layers):
        """Build one of the two parallel conv towers used for cls / bbox heads.

        Each block: Conv (no bias) → GroupNorm → ReLU → Dropout. GroupNorm keeps the
        activation scale under control even when batch-size is very small,
        which is common in 3-D medical imaging. Dropout helps prevent overfitting.
        """

        layers = []
        for i in range(num_layers):
            layers.extend([
                nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False),
                nn.GroupNorm(32, in_ch),
                nn.ReLU(inplace=True)
            ])
            # Add dropout after each layer except the last one (to avoid too much regularization)
            if i < num_layers - 1:
                layers.append(nn.Dropout2d(self.dropout_rate * 0.5))  # Use half dropout rate in intermediate layers
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        # Initialize classification head with bias for rare positive samples
        # Less aggressive initialization for medical imaging (assume 5% positive instead of 1%)
        nn.init.constant_(self.cls_logits.bias, -np.log((1 - 0.05) / 0.05))
        
        # Initialize other layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Tighter initialisation to reduce the risk of overflow in the
                # very first iterations when One-Cycle peaks the LR.
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None and m != self.cls_logits:
                    nn.init.zeros_(m.bias)
    
    def forward(self, features):
        """
        Args:
            features: Dict of pyramid features {P3, P4, P5, P6}
        Returns:
            cls_logits: List of classification predictions
            bbox_preds: List of bbox regression predictions
        """
        cls_logits = []
        bbox_preds = []
        
        for level, feat in features.items():
            # Classification branch
            cls_feat = self.cls_tower(feat)
            cls_feat = self.cls_dropout(cls_feat)
            cls_out = self.cls_logits(cls_feat)
            cls_logits.append(cls_out)
            
            # Regression branch  
            bbox_feat = self.bbox_tower(feat)
            bbox_feat = self.bbox_dropout(bbox_feat)
            bbox_out = self.bbox_pred(bbox_feat)
            bbox_preds.append(bbox_out)
            
        return cls_logits, bbox_preds


class AnchorGenerator:
    """Generate anchors for RetinaNet"""
    
    def __init__(self, sizes=[16, 32, 64, 128], ratios=[0.5, 1.0, 2.0], scales=[1.0, 1.26, 1.587]):
        self.sizes = sizes
        self.ratios = ratios
        self.scales = scales
        self.num_anchors = len(ratios) * len(scales)  # 9 anchors per location
        
    def generate_anchors_for_level(self, feat_h, feat_w, stride, base_size, device='cpu'):
        """Generate anchors for one pyramid level"""
        # Create anchor templates
        anchors = []
        for ratio in self.ratios:
            for scale in self.scales:
                # Compute anchor width and height
                area = (base_size * scale) ** 2
                w = np.sqrt(area / ratio)
                h = w * ratio
                
                # Store as [x_ctr, y_ctr, w, h] format
                anchors.append([0, 0, w, h])
        
        anchors = torch.tensor(anchors, device=device, dtype=torch.float32)  # [9, 4]
        
        # Generate grid of anchor centers
        shift_x = torch.arange(0, feat_w, device=device, dtype=torch.float32) * stride + stride / 2.0
        shift_y = torch.arange(0, feat_h, device=device, dtype=torch.float32) * stride + stride / 2.0
        
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing='ij')
        shifts = torch.stack([shift_x.flatten(), shift_y.flatten()], dim=1)  # [H*W, 2]
        
        # Broadcast anchors to all positions
        A = anchors.size(0)  # 9
        K = shifts.size(0)   # H*W
        
        # Expand anchors and shifts
        anchors = anchors.view(1, A, 4).expand(K, A, 4).clone()  # Clone to avoid memory sharing
        shifts = shifts.view(K, 1, 2).expand(K, A, 2)
        
        # Apply shifts to anchor centers
        anchors[:, :, :2] += shifts
        
        return anchors.reshape(-1, 4)  # [H*W*9, 4]
    
    def generate_anchors(self, feat_dict, img_size=518):
        """Generate anchors for all pyramid levels"""
        anchors_per_level = []
        
        # Calculate actual strides based on feature map sizes
        # For 518x518 input with DINOv2 ViT-B/14 (patch size 14)
        level_info = {
            'P3': {'expected_size': 74, 'base_size_idx': 0},  # 16
            'P4': {'expected_size': 37, 'base_size_idx': 1},  # 32  
            'P5': {'expected_size': 18, 'base_size_idx': 2},  # 64
            'P6': {'expected_size': 9,  'base_size_idx': 3},  # 128
        }
        
        for level, feat in feat_dict.items():
            B, C, H, W = feat.shape
            
            # Calculate actual stride based on feature map size
            stride = img_size / H
            
            # Get appropriate base anchor size for this level
            if level in level_info:
                base_size_idx = level_info[level]['base_size_idx']
                if base_size_idx < len(self.sizes):
                    base_size = self.sizes[base_size_idx]
                else:
                    base_size = self.sizes[-1]  # Use largest size if index out of bounds
            else:
                base_size = self.sizes[0]  # Default fallback
            
            level_anchors = self.generate_anchors_for_level(H, W, stride, base_size, feat.device)
            anchors_per_level.append(level_anchors)
            
        return anchors_per_level


class ViTLesionDetector(nn.Module):
    """Complete ViT-based lesion detector for medical imaging"""
    
    def __init__(self, args, num_classes=1, freeze_backbone=True, dropout_rate=0.1):
        super().__init__()
        
        # Load DINOv2 ViT-B/14 backbone
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        if args.backbone_checkpoint is not None:
            state_dict = torch.load(
                args.backbone_checkpoint, map_location=torch.device("cpu")
            )

            # Remove the "backbone." prefix from all keys to match expected structure
            new_state_dict = {}
            for k, v in state_dict["teacher"].items():
                new_key = k.replace("backbone.", "", 1)
                new_state_dict[new_key] = v

            load_result = self.backbone.load_state_dict(new_state_dict, strict=False)
            logging.info(
                f"Loaded Backbone DINO State dict from {args.backbone_checkpoint} with result: {load_result}"
            )

        else:
            logging.info(
                "Loaded the baseline facebook DINO model: facebookresearch/dinov2 (dinov2_vitb14)"
            )
            
        # Check backbone initialization
        for name, param in self.backbone.named_parameters():
            if torch.isnan(param).any():
                print(f"WARNING: NaN values detected in backbone parameter {name}")
        
        # Remove classification head if it exists
        if hasattr(self.backbone, 'head'):
            self.backbone.head = nn.Identity()
        
        # Feature pyramid neck
        self.neck = SimplePyramid(in_ch=768, out_ch=256)
        
        # Check neck initialization
        for name, param in self.neck.named_parameters():
            if torch.isnan(param).any():
                print(f"WARNING: NaN values detected in neck parameter {name}")
        
        # Detection head with dropout
        self.head = RetinaHead(in_ch=256, num_classes=num_classes, num_anchors=9, dropout_rate=dropout_rate)
        
        # Check head initialization
        for name, param in self.head.named_parameters():
            if torch.isnan(param).any():
                print(f"WARNING: NaN values detected in head parameter {name}")
        
        # Anchor generator
        self.anchor_generator = AnchorGenerator()
        
        # Freeze backbone if specified
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()
        
        self.num_classes = num_classes
        
    def forward(self, x):
        """
        Forward pass: backbone → neck → head
        
        Args:
            x: Input tensor [B, 3, H, W]
            
        Returns:
            cls_logits: List of classification predictions per pyramid level
            bbox_preds: List of bbox regression predictions per pyramid level
            pyramid_features: Dict of pyramid features for visualization/analysis
        """
        # Extract patch token features from backbone
        if self.backbone.training:
            # During training, allow gradients through backbone if not frozen
            features_dict = self.backbone.forward_features(x)
        else:
            # During inference, use no_grad for efficiency
            with torch.no_grad():
                features_dict = self.backbone.forward_features(x)
        
        # Extract patch tokens (not CLS token)
        patch_tokens = features_dict['x_norm_patchtokens']
        
        # Reshape to spatial format (B, C, H, W)
        B, L, C = patch_tokens.shape
        H = W = int(np.sqrt(L))
        backbone_feat = patch_tokens.reshape(B, H, W, C).permute(0, 3, 1, 2)
        
        # Pass through neck
        pyramid_features = self.neck(backbone_feat)
        
        # Pass through head
        cls_logits, bbox_preds = self.head(pyramid_features)

        # Sanitize potential NaNs/Infs to keep training going
        def _sanitize(t):
            return torch.nan_to_num(t, nan=0.0, posinf=1e4, neginf=-1e4)

        for i in range(len(cls_logits)):
            if torch.isnan(cls_logits[i]).any():
                print(f"WARNING: NaN values detected in cls_logits level {i}. Replacing with zeros.")
                cls_logits[i] = _sanitize(cls_logits[i])

        for i in range(len(bbox_preds)):
            if torch.isnan(bbox_preds[i]).any():
                print(f"WARNING: NaN values detected in bbox_preds level {i}. Replacing with zeros.")
                bbox_preds[i] = _sanitize(bbox_preds[i])

        return cls_logits, bbox_preds, pyramid_features
    
    def get_anchors(self, pyramid_features):
        """Generate anchors for current pyramid features"""
        return self.anchor_generator.generate_anchors(pyramid_features)
    
    def get_parameter_count(self):
        """Get detailed parameter counts"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': frozen_params,
            'trainable_ratio': trainable_params / total_params
        }


class LesionDetectionModule(pl.LightningModule):
    """
    PyTorch Lightning module for training ViT-based lesion detector.
    
    Features:
    - Single-class lesion detection (binary classification)
    - Anchor-based detection with multi-scale features
    - Medical imaging optimized loss functions
    - Comprehensive metrics tracking
    - Efficient transfer learning (frozen backbone)
    """
    
    def __init__(
        self,
        args,
        freeze_backbone: bool = True,
        weight_decay: float = 5e-4,  # Increased from 1e-5 for better regularization
        max_epochs: int = 50,
        pred_output_dir: str = "/autofs/space/crater_001/projects/breast_cancer_dbt/detection_deit_test",
        dropout_rate: float = 0.15,  # Dropout rate for regularization
        label_smoothing: float = 0.05,  # Label smoothing for regularization
        pred_mode: str = "always_top5",  # 'threshold' or 'always_top5'
        conf_threshold: float = 0.6,      # Score cut-off used when exporting predictions
        save_to_csv: bool = False,        # Enable CSV collection during `trainer.predict()`
        csv_filename: str = "predictions.csv",
        **kwargs
    ):
        # `super().__init__()` called once below (after args post-processing)
        super().__init__()
        # If a checkpoint stores `args` as a plain dict (the default behaviour
        # of `save_hyperparameters`), convert it back to a Namespace so that
        # attribute access (e.g. `args.output_dir`) continues to work.
        if isinstance(args, dict):
            args = argparse.Namespace(**args)

        self.save_hyperparameters(args)
        setup_logging(args.output_dir)
        # ------------------------------------------------------------------
        # Retrieve (and fall back) all tuneable hyper-parameters from *args*
        # ------------------------------------------------------------------

        # Regularisation & optimisation
        self.dropout_rate    = getattr(args, "dropout_rate", dropout_rate)
        self.label_smoothing = getattr(args, "label_smoothing", label_smoothing)

        # Classification ↔ bbox loss weighting
        self.cls_bbox_ratio = getattr(args, "cls_bbox_ratio", None)

        # Focal-loss hyper-parameters
        self.focal_alpha  = getattr(args, "focal_alpha", 0.9)
        self.focal_gamma  = getattr(args, "focal_gamma", 1.5)

        # Anchor assignment parameters
        self.iou_threshold     = getattr(args, "iou_threshold", 0.5)
        self.neg_iou_threshold = getattr(args, "neg_iou_threshold", 0.4)
        self.neg_pos_ratio     = getattr(args, "neg_pos_ratio", 3)

        # Non-maximum suppression
        self.nms_threshold = getattr(args, "nms_threshold", 0.05)

        # Loss-weight configuration
        if self.cls_bbox_ratio is not None:
            # Interpret ratio as *classification : bbox*.  Regression weight
            # kept at 1 so that tuning can freely up-/down-weight the cls term.
            self.cls_loss_weight  = float(self.cls_bbox_ratio)
            self.bbox_loss_weight = 1.0
        else:
            self.cls_loss_weight  = 1.0
            self.bbox_loss_weight = 1.0
        
        # Model with dropout
        self.model = ViTLesionDetector(
            args,
            num_classes=1,          # Always 1 for lesion detection
            freeze_backbone=True,   # Always freeze backbone for transfer learning
            dropout_rate=self.dropout_rate,
        )
        
        # ------------------------------------------------------------------
        # Loss functions
        # ------------------------------------------------------------------
        self.cls_loss_fn  = self.focal_loss
        self.bbox_loss_fn = torch.nn.SmoothL1Loss(reduction='none')
        
        # Anchor-level metrics kept for quick sanity check (optional)
        self.train_accuracy = torchmetrics.Accuracy(task='binary')
        self.val_accuracy = torchmetrics.Accuracy(task='binary')
        self.val_precision = torchmetrics.Precision(task='binary')
        self.val_recall = torchmetrics.Recall(task='binary')
        self.val_f1 = torchmetrics.F1Score(task='binary')

        # Detection metric – mean Average Precision (mAP)
        # Uses torchmetrics implementation which accumulates over the epoch.
        # Default configuration computes COCO-style AP @[0.5:0.95]. For
        # simplicity we rely on that default; you can pass explicit
        # iou_thresholds=[0.5] if you only care about AP50.
        self.val_map = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
        
        # Storage for FROC evaluation
        self.val_predictions = []  # List of dicts with boxes, scores, image_id
        self.val_ground_truth = []  # List of dicts with boxes, image_id
        
        # ------------------------------------------------------------------
        # IoU thresholds & negative-sampling strategy – already set above
        # ------------------------------------------------------------------

        # When an image has *no* positives we still need some negatives – cap them by this value.
        self.max_negatives = 2000

        # Confidence score used during inference / NMS filtering (overridable)
        self.conf_threshold = conf_threshold

        # Prediction output directory
        self.pred_output_dir = pred_output_dir
        os.makedirs(self.pred_output_dir, exist_ok=True)
        
        # Prediction mode ('threshold' | 'always_top5')
        assert pred_mode in {"threshold", "always_top5"}, "pred_mode must be 'threshold' or 'always_top5'"
        self.pred_mode = pred_mode
        
        # ------------------------------------------------------------------
        # NMS configuration – threshold set above; rest of the comments kept
        # ------------------------------------------------------------------
        
        # ------------------------------------------------------------------
        # Prediction-to-CSV configuration
        # ------------------------------------------------------------------
        self.save_to_csv  = save_to_csv
        self.csv_filename = csv_filename
        if self.save_to_csv:
            # Accumulate one row per *predicted* bounding box
            self.pred_csv_rows: list[dict[str, float | str]] = []
        
    def focal_loss(self, pred, target):
        """
        Focal loss for better handling of class imbalance.
        Modified to handle extreme imbalance better with label smoothing.
        
        Args:
            pred: [N] predicted logits
            target: [N] target labels (0 or 1)
            
        Returns:
            loss: scalar focal loss
        """
        # Apply label smoothing for regularization
        if self.label_smoothing > 0:
            # Smooth positive labels: 1 -> (1 - label_smoothing)
            # Smooth negative labels: 0 -> label_smoothing
            target_smooth = target * (1 - self.label_smoothing) + (1 - target) * self.label_smoothing
        else:
            target_smooth = target
        
        # Convert logits to probabilities
        pred_prob = torch.sigmoid(pred)
        
        # Clamp probabilities to prevent numerical issues
        pred_prob = torch.clamp(pred_prob, min=1e-7, max=1.0 - 1e-7)
        
        # Compute p_t (probability of true class) using smoothed targets
        p_t = target_smooth * pred_prob + (1 - target_smooth) * (1 - pred_prob)
        
        # Compute alpha weighting (use original targets for alpha, not smoothed)
        alpha_factor = target * self.focal_alpha + (1 - target) * (1 - self.focal_alpha)
        
        # Compute focal modulating factor with less aggressive scaling
        modulating_factor = (1.0 - p_t) ** self.focal_gamma
        
        # Compute standard binary cross entropy with smoothed targets
        bce_loss = F.binary_cross_entropy_with_logits(pred, target_smooth, reduction='none')
        
        # Apply focal loss weighting
        focal_loss = alpha_factor * modulating_factor * bce_loss
        
        # Additional scaling for positive samples to ensure they don't get too small
        positive_mask = target > 0.5  # Use original targets for positive mask
        if positive_mask.sum() > 0:
            # Give positive samples extra weight to combat extreme imbalance
            focal_loss[positive_mask] = focal_loss[positive_mask] * 2.0
        
        return focal_loss.mean()

    def forward(self, x):
        """Forward pass through the model"""
        return self.model(x)
    
    def compute_iou(self, box1, box2):
        """
        Compute IoU between two sets of boxes.
        
        Args:
            box1: [N, 4] tensor (x1, y1, x2, y2)
            box2: [M, 4] tensor (x1, y1, x2, y2)
            
        Returns:
            iou: [N, M] tensor of IoU values
        """
        # Compute intersection
        x1 = torch.max(box1[:, None, 0], box2[None, :, 0])  # [N, M]
        y1 = torch.max(box1[:, None, 1], box2[None, :, 1])  # [N, M]
        x2 = torch.min(box1[:, None, 2], box2[None, :, 2])  # [N, M]
        y2 = torch.min(box1[:, None, 3], box2[None, :, 3])  # [N, M]
        
        intersection = torch.clamp(x2 - x1, 0) * torch.clamp(y2 - y1, 0)
        
        # Compute areas
        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N]
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M]
        
        union = area1[:, None] + area2[None, :] - intersection  # [N, M]
        
        iou = intersection / (union + 1e-6)
        return iou
    
    def encode_bbox(self, gt_boxes, anchors):
        """
        Encode ground truth boxes relative to anchors.
        
        Args:
            gt_boxes: [N, 4] ground truth boxes (x1, y1, x2, y2) 
            anchors: [N, 4] anchor boxes (x_ctr, y_ctr, w, h) - same number as gt_boxes
            
        Returns:
            encoded: [N, 4] encoded boxes (dx, dy, dw, dh)
        """
        # Ensure inputs have the same number of boxes
        assert gt_boxes.shape[0] == anchors.shape[0], f"gt_boxes ({gt_boxes.shape[0]}) and anchors ({anchors.shape[0]}) must have same length"
        
        # Convert gt_boxes to center format
        gt_w = gt_boxes[:, 2] - gt_boxes[:, 0]
        gt_h = gt_boxes[:, 3] - gt_boxes[:, 1]
        gt_x = gt_boxes[:, 0] + gt_w / 2
        gt_y = gt_boxes[:, 1] + gt_h / 2
        
        # Extract anchor parameters
        anchor_x, anchor_y, anchor_w, anchor_h = anchors[:, 0], anchors[:, 1], anchors[:, 2], anchors[:, 3]
        
        # Add small epsilon to prevent division by zero
        anchor_w = torch.clamp(anchor_w, min=1e-6)
        anchor_h = torch.clamp(anchor_h, min=1e-6)
        gt_w = torch.clamp(gt_w, min=1e-6)
        gt_h = torch.clamp(gt_h, min=1e-6)
        
        # Encode
        dx = (gt_x - anchor_x) / anchor_w
        dy = (gt_y - anchor_y) / anchor_h
        dw = torch.log(gt_w / anchor_w)
        dh = torch.log(gt_h / anchor_h)
        
        return torch.stack([dx, dy, dw, dh], dim=1).to(dtype=torch.float32)
    
    def decode_bbox(self, bbox_preds, anchors):
        """
        Decode bbox predictions to absolute coordinates.
        
        Args:
            bbox_preds: [N, 4] bbox predictions (dx, dy, dw, dh)
            anchors: [N, 4] anchor boxes (x_ctr, y_ctr, w, h)
            
        Returns:
            decoded: [N, 4] decoded boxes (x1, y1, x2, y2)
        """
        # Extract anchor parameters
        anchor_x, anchor_y, anchor_w, anchor_h = anchors[:, 0], anchors[:, 1], anchors[:, 2], anchors[:, 3]
        
        # Decode predictions
        pred_x = bbox_preds[:, 0] * anchor_w + anchor_x
        pred_y = bbox_preds[:, 1] * anchor_h + anchor_y
        pred_w = torch.exp(bbox_preds[:, 2]) * anchor_w
        pred_h = torch.exp(bbox_preds[:, 3]) * anchor_h
        
        # Convert to corner format
        x1 = pred_x - pred_w / 2
        y1 = pred_y - pred_h / 2
        x2 = pred_x + pred_w / 2
        y2 = pred_y + pred_h / 2
        
        return torch.stack([x1, y1, x2, y2], dim=1)
    
    def assign_targets(self, gt_boxes, gt_labels, anchors_per_level):
        """
        Assign ground truth to anchors using IoU matching.
        """
        # Handle single sample case
        if len(gt_boxes.shape) == 2:
            gt_boxes = gt_boxes.unsqueeze(0)
            gt_labels = gt_labels.unsqueeze(0)
        
        B = gt_boxes.shape[0]
        
        cls_targets = []
        bbox_targets = []
        valid_masks = []
        
        # Process each pyramid level
        for level_idx, anchors in enumerate(anchors_per_level):
            num_anchors = anchors.shape[0]
            
            # Convert anchors to corner format for IoU computation
            anchor_x1 = anchors[:, 0] - anchors[:, 2] / 2
            anchor_y1 = anchors[:, 1] - anchors[:, 3] / 2
            anchor_x2 = anchors[:, 0] + anchors[:, 2] / 2
            anchor_y2 = anchors[:, 1] + anchors[:, 3] / 2
            anchor_boxes = torch.stack([anchor_x1, anchor_y1, anchor_x2, anchor_y2], dim=1)
            
            # Initialize batch tensors
            batch_cls_target = torch.zeros(B, num_anchors, device=anchors.device)
            batch_bbox_target = torch.zeros(B, num_anchors, 4, device=anchors.device, dtype=torch.float32)
            batch_valid_mask = torch.zeros(B, num_anchors, dtype=torch.bool, device=anchors.device)
            
            # Process each sample in batch
            for b in range(B):
                sample_boxes = gt_boxes[b]
                
                if sample_boxes.shape[0] == 0:
                    num_keep = min(num_anchors, self.max_negatives)
                    keep_idx = torch.randperm(num_anchors, device=anchors.device)[:num_keep]
                    batch_cls_target[b, keep_idx] = 0.0
                    batch_valid_mask[b, keep_idx] = True
                    continue
                
                # Ensure gt_boxes are float32
                sample_boxes = sample_boxes.to(dtype=torch.float32)
                
                # Compute IoU between anchors and ground truth
                ious = self.compute_iou(anchor_boxes, sample_boxes)  # [num_anchors, num_gt]
                
                # Find best matching ground truth for each anchor
                max_iou, best_gt_idx = torch.max(ious, dim=1)  # [num_anchors]
                
                # Positive samples (IoU > threshold)
                positive_mask = max_iou > self.iou_threshold
                batch_cls_target[b, positive_mask] = 1.0
                batch_valid_mask[b, positive_mask] = True
                
                # Negative samples
                negative_mask = max_iou < self.neg_iou_threshold
                
                if negative_mask.any():
                    neg_indices = negative_mask.nonzero(as_tuple=False).squeeze(-1)
                    pos_count = positive_mask.sum().item()
                    
                    if pos_count > 0:
                        max_negs = min(len(neg_indices), pos_count * self.neg_pos_ratio)
                    else:
                        max_negs = min(len(neg_indices), self.max_negatives)
                    
                    if max_negs > 0:
                        perm = torch.randperm(len(neg_indices), device=anchors.device)
                        keep_neg_indices = neg_indices[perm[:max_negs]]
                        batch_cls_target[b, keep_neg_indices] = 0.0
                        batch_valid_mask[b, keep_neg_indices] = True
                
                # Encode bbox regression targets for positive anchors
                if positive_mask.sum() > 0:
                    positive_anchors = anchors[positive_mask]
                    positive_gt_boxes = sample_boxes[best_gt_idx[positive_mask]]
                    encoded_boxes = self.encode_bbox(positive_gt_boxes, positive_anchors)
                    batch_bbox_target[b, positive_mask] = encoded_boxes
            
            cls_targets.append(batch_cls_target)
            bbox_targets.append(batch_bbox_target)
            valid_masks.append(batch_valid_mask)
        
        return cls_targets, bbox_targets, valid_masks
    
    def compute_loss(self, cls_logits, bbox_preds, cls_targets, bbox_targets, valid_masks):
        """
        Compute detection loss (classification + regression).
        
        Args:
            cls_logits: List of classification predictions per level
            bbox_preds: List of bbox predictions per level  
            cls_targets: List of classification targets per level [B, num_anchors]
            bbox_targets: List of bbox targets per level [B, num_anchors, 4]
            valid_masks: List of valid masks per level [B, num_anchors]
            
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of individual losses
        """
        cls_losses = []
        bbox_losses = []
        total_valid = 0
        total_positive = 0
        
        for cls_out, bbox_out, cls_tgt, bbox_tgt, valid_mask in zip(
            cls_logits, bbox_preds, cls_targets, bbox_targets, valid_masks
        ):
            # Reshape predictions to match targets
            B, A, H, W = cls_out.shape  # [B, 9, H, W]
            cls_out_flat = cls_out.permute(0, 2, 3, 1).reshape(B, -1)  # [B, H*W*9]
            bbox_out_flat = bbox_out.permute(0, 2, 3, 1).reshape(B, -1, 4)  # [B, H*W*9, 4]
            
            # Process each sample in the batch
            for b in range(B):
                # Get valid indices for this sample
                valid_indices = valid_mask[b].nonzero().squeeze(-1)
                
                if len(valid_indices) > 0:
                    total_valid += len(valid_indices)
                    
                    # Classification loss (only on valid anchors)
                    cls_loss = self.cls_loss_fn(
                        cls_out_flat[b][valid_indices], 
                        cls_tgt[b][valid_indices]
                    )
                    cls_losses.append(cls_loss)
                    
                    # Regression loss (only on positive anchors)
                    positive_mask = (cls_tgt[b] > 0.5)[valid_indices]
                    positive_indices = positive_mask.nonzero().squeeze(-1)
                    
                    if len(positive_indices) > 0:
                        total_positive += len(positive_indices)
                        bbox_loss = self.bbox_loss_fn(
                            bbox_out_flat[b][valid_indices][positive_indices], 
                            bbox_tgt[b][valid_indices][positive_indices]
                        ).mean()
                        bbox_losses.append(bbox_loss)
        
        # Combine losses with proper normalization
        cls_loss_total = torch.stack(cls_losses).mean() if cls_losses else torch.tensor(0.0, device=self.device)
        bbox_loss_total = torch.stack(bbox_losses).mean() if bbox_losses else torch.tensor(0.0, device=self.device)
        
        # Remove aggressive scaling - let the loss weights handle the balance
        # The bbox loss scaling was making bbox loss dominate too much
        
        total_loss = (
            self.cls_loss_weight * cls_loss_total + 
            self.bbox_loss_weight * bbox_loss_total
        )
        
        loss_dict = {
            'total_loss': total_loss,
            'cls_loss': cls_loss_total,
            'bbox_loss': bbox_loss_total,
            'total_valid': total_valid,
            'total_positive': total_positive,
            'positive_ratio': total_positive / total_valid if total_valid > 0 else 0
        }
        
        return total_loss, loss_dict
    
    def training_step(self, batch, batch_idx):
        """Training step"""
        # Extract images and targets from batch dictionary
        images = batch['img']  # [B, C, H, W]
        targets = batch['target']  # Dictionary of target lists/tensors
        
        # Forward pass
        cls_logits, bbox_preds, pyramid_features = self(images)
        
        # Generate anchors
        anchors_per_level = self.model.get_anchors(pyramid_features)
        
        # Get boxes and labels from target dictionary (now lists!)
        gt_boxes_list = targets['boxes']  # List of tensors
        gt_labels_list = targets['labels']  # List of tensors
        
        # FIXED: Convert list format to padded tensor format for assign_targets
        batch_size = images.shape[0]
        
        # Find max number of boxes in this batch
        max_boxes = max(len(boxes) for boxes in gt_boxes_list)
        if max_boxes == 0:
            max_boxes = 1  # Ensure at least 1 to avoid empty tensors
        
        # Create padded tensors
        padded_gt_boxes = torch.zeros(batch_size, max_boxes, 4, device=images.device, dtype=torch.float32)
        padded_gt_labels = torch.zeros(batch_size, max_boxes, device=images.device, dtype=torch.float32)
        
        # Fill padded tensors
        for i, (boxes, labels) in enumerate(zip(gt_boxes_list, gt_labels_list)):
            if len(boxes) > 0:
                padded_gt_boxes[i, :len(boxes)] = boxes
                padded_gt_labels[i, :len(labels)] = labels
        
        # Process all samples in batch at once
        cls_targets, bbox_targets, valid_masks = self.assign_targets(
            padded_gt_boxes, padded_gt_labels, anchors_per_level
        )
        
        # Compute loss
        loss, loss_dict = self.compute_loss(
            cls_logits, bbox_preds, 
            cls_targets, bbox_targets, valid_masks
        )
        # Print anchor assignment statistics
        total_positive = loss_dict['total_positive']
        total_valid = loss_dict['total_valid']
        total_negative = total_valid - total_positive
        positive_ratio = loss_dict['positive_ratio']
        
        # Check how many samples have zero positive anchors across all levels
        samples_with_zero_positives = 0
        for b in range(batch_size):
            sample_total_positives = 0
            for cls_tgt in cls_targets:
                sample_total_positives += (cls_tgt[b] > 0.5).sum().item()
            if sample_total_positives == 0:
                samples_with_zero_positives += 1
                # Print detailed warning for samples with zero positives across ALL levels
                gt_boxes_tensor = gt_boxes_list[b]
                if len(gt_boxes_tensor) > 0:
                    # print(f"WARNING: Training sample {b} has NO POSITIVE ANCHORS across ALL pyramid levels!")
                    # print(f"  - GT boxes: {gt_boxes_tensor}")
                    # print(f"  - IoU threshold: {self.iou_threshold}")
                    pass
        
        # print(f"Anchor assignments - Positive: {total_positive}, Negative: {total_negative}, Total valid: {total_valid}")
        # print(f"Positive ratio: {positive_ratio:.4f} ({positive_ratio*100:.2f}%)")
        # if samples_with_zero_positives > 0:
        #     print(f"WARNING: {samples_with_zero_positives}/{batch_size} training samples have ZERO positive anchors across ALL levels!")
        
        # Add debugging output every 10 steps
        #if batch_idx % 10 == 0:
        #    print(f"\n=== Training Step {batch_idx} Debug Info ===")
        #    print(f"Batch size: {batch_size}")
        #    print(f"Max boxes: {max_boxes}")
        #    
#
        #    
        #    # Print ground truth box statistics
        #    for i in range(min(2, batch_size)):
        #        sample_boxes = gt_boxes_list[i]
        #        if len(sample_boxes) > 0:
        #            print(f"Sample {i} GT boxes: {sample_boxes}")
        #        else:
        #            print(f"Sample {i}: No valid GT boxes")
        #    
        #    print(f"Loss components - cls: {loss_dict['cls_loss']:.4f}, bbox: {loss_dict['bbox_loss']:.4f}, total: {loss:.4f}")
        #    print("=" * 50)

        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log('train_cls_loss', loss_dict['cls_loss'], on_step=True, on_epoch=True, batch_size=batch_size)
        self.log('train_bbox_loss', loss_dict['bbox_loss'], on_step=True, on_epoch=True, batch_size=batch_size)
        
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step"""
        # Extract images and targets from batch dictionary
        images = batch['img']  # [B, C, H, W]
        targets = batch['target']  # Dictionary of target lists/tensors
        
        # Forward pass
        cls_logits, bbox_preds, pyramid_features = self(images)
        
        # Generate anchors
        anchors_per_level = self.model.get_anchors(pyramid_features)
        
        # Get boxes and labels from target dictionary (now lists!)
        gt_boxes_list = targets['boxes']  # List of tensors
        gt_labels_list = targets['labels']  # List of tensors
        
        # FIXED: Convert list format to padded tensor format for assign_targets
        batch_size = images.shape[0]
        
        # Find max number of boxes in this batch
        max_boxes = max(len(boxes) for boxes in gt_boxes_list)
        if max_boxes == 0:
            max_boxes = 1  # Ensure at least 1 to avoid empty tensors
        
        # Create padded tensors
        padded_gt_boxes = torch.zeros(batch_size, max_boxes, 4, device=images.device, dtype=torch.float32)
        padded_gt_labels = torch.zeros(batch_size, max_boxes, device=images.device, dtype=torch.float32)
        
        # Fill padded tensors
        for i, (boxes, labels) in enumerate(zip(gt_boxes_list, gt_labels_list)):
            if len(boxes) > 0:
                padded_gt_boxes[i, :len(boxes)] = boxes
                padded_gt_labels[i, :len(labels)] = labels
        
        
        # Process all samples in batch at once
        cls_targets, bbox_targets, valid_masks = self.assign_targets(
            padded_gt_boxes, padded_gt_labels, anchors_per_level
        )
        
        # Compute loss
        loss, loss_dict = self.compute_loss(
            cls_logits, bbox_preds,
            cls_targets, bbox_targets, valid_masks
        )
        
        # Print anchor assignment statistics
        #total_positive = loss_dict['total_positive']
        #total_valid = loss_dict['total_valid']
        #total_negative = total_valid - total_positive
        #positive_ratio = loss_dict['positive_ratio']
        #
        ## Check how many samples have zero positive anchors across all levels
        #samples_with_zero_positives = 0
        #for b in range(batch_size):
        #    sample_total_positives = 0
        #    for cls_tgt in cls_targets:
        #        sample_total_positives += (cls_tgt[b] > 0.5).sum().item()
        #    if sample_total_positives == 0:
        #        samples_with_zero_positives += 1
        #        # Print detailed warning for samples with zero positives across ALL levels
        #        gt_boxes_tensor = gt_boxes_list[b]
        #        if len(gt_boxes_tensor) > 0:
        #            # print(f"WARNING: Validation sample {b} has NO POSITIVE ANCHORS across ALL pyramid levels!")
        #            # print(f"  - GT boxes: {gt_boxes_tensor}")
        #            # print(f"  - IoU threshold: {self.iou_threshold}")
        #            pass
        #
        # print(f"VAL Anchor assignments - Positive: {total_positive}, Negative: {total_negative}, Total valid: {total_valid}")
        # print(f"VAL Positive ratio: {positive_ratio:.4f} ({positive_ratio*100:.2f}%)")
        # if samples_with_zero_positives > 0:
        #     print(f"WARNING: {samples_with_zero_positives}/{batch_size} validation samples have ZERO positive anchors across ALL levels!")

        # Update mAP metric (keep the list format for mAP)
        batch_predictions = []
        batch_targets_dict = []

        B = images.size(0)

        for b in range(B):
            boxes_pred_all = []
            scores_pred_all = []
            labels_pred_all = []

            # Gather predictions **once** across all pyramid levels using the same
            # logic as predict_step so that visualisation and metrics match.
            boxes_cat, scores_cat = self._gather_image_predictions(
                cls_logits,
                bbox_preds,
                anchors_per_level,
                img_idx=b,
                conf_threshold=1e-6,              # include everything
                nms_threshold=self.nms_threshold,
                topk = 5 if self.pred_mode == "always_top5" else None
            )

            if boxes_cat.numel() > 0:
                labels_cat = torch.zeros(boxes_cat.shape[0], dtype=torch.long, device=boxes_cat.device)
            else:
                scores_cat = torch.empty((0,), device=images.device)
                labels_cat = torch.empty((0,), dtype=torch.long, device=images.device)

            batch_predictions.append({
                "boxes": boxes_cat.detach(),
                "scores": scores_cat.detach(),
                "labels": labels_cat.detach(),
            })

            # Ground-truth dict (use original list format)
            gt_boxes_b = gt_boxes_list[b]  # Already a tensor
            gt_labels_b = gt_labels_list[b]  # Already a tensor
            if gt_boxes_b.numel() == 0:
                gt_boxes_b = torch.empty((0, 4), device=images.device)
                gt_labels_b = torch.empty((0,), dtype=torch.long, device=images.device)
            batch_targets_dict.append(
                {"boxes": gt_boxes_b.detach(), "labels": gt_labels_b.long().detach()}
            )

        # Update torchmetrics MAP
        self.val_map.update(batch_predictions, batch_targets_dict)

        # Store predictions and ground truth for FROC evaluation
        for b in range(B):
            # Get predictions for this image
            pred_dict = batch_predictions[b]
            pred_boxes = pred_dict["boxes"].cpu().numpy()
            pred_scores = pred_dict["scores"].cpu().numpy()
            
            # Store predictions with unique image ID
            image_id = f"{batch_idx}_{b}"
            if len(pred_boxes) > 0:
                self.val_predictions.append({
                    'boxes': pred_boxes,
                    'scores': pred_scores,
                    'image_id': image_id
                })
            
            # Store ground-truth information.  We now add **every** image so that later
            # metrics know the true number of images that were processed.  Images
            # without lesions will simply have an empty ``boxes`` array which is
            # perfectly acceptable.
            gt_boxes = gt_boxes_list[b].cpu().numpy()

            self.val_ground_truth.append({
                'boxes': gt_boxes,  # may be empty (shape: (0, 4)) for negative images
                'image_id': image_id
            })

        # Log metrics
        batch_size = images.size(0)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log('val_cls_loss', loss_dict['cls_loss'], on_epoch=True, batch_size=batch_size)
        self.log('val_bbox_loss', loss_dict['bbox_loss'], on_epoch=True, batch_size=batch_size)
        
        # Quick prediction check every 5 validation batches
        #if batch_idx % 5 == 0:
        #    self._debug_predictions(images, cls_logits, bbox_preds, anchors_per_level, gt_boxes_list, batch_idx)
        
        return loss


    
    def _debug_predictions(self, images, cls_logits, bbox_preds, anchors_per_level, gt_boxes_list, batch_idx):
        """Quick debug function to print prediction statistics"""
        with torch.no_grad():
            # Look at first image only
            if len(images) > 0:
                # Get best prediction across all levels for first image
                best_conf = -1
                best_coords = None
                
                for level_idx, (cls_logits_level, bbox_preds_level, anchors_level) in enumerate(
                    zip(cls_logits, bbox_preds, anchors_per_level)
                ):
                    cls_pred = cls_logits_level[0]  # [A, H, W]
                    A, H, W = cls_pred.shape
                    cls_probs = torch.sigmoid(cls_pred).view(A, -1).max(dim=0)[0]  # [H*W]
                    max_conf, max_idx = torch.max(cls_probs, dim=0)
                    
                    if max_conf > best_conf:
                        best_conf = max_conf
                        # Convert to spatial coordinates
                        h_idx = max_idx // W
                        w_idx = max_idx % W
                        anchor_idx = cls_pred.view(A, -1).max(dim=1)[1][cls_pred.view(A, -1).max(dim=1)[0].argmax()]
                        
                        # Get anchor center (approximately)
                        if level_idx == 0:  # P3 level
                            stride = 7.0
                        elif level_idx == 1:  # P4 level  
                            stride = 14.0
                        elif level_idx == 2:  # P5 level
                            stride = 28.8
                        else:  # P6 level
                            stride = 57.6
                            
                        center_x = w_idx * stride + stride / 2
                        center_y = h_idx * stride + stride / 2
                        best_coords = (center_x.item(), center_y.item())
                
                # Print debug info - UPDATED for list format
                print(f"\n--- Val Batch {batch_idx} Prediction Debug ---")
                
                # gt_boxes_list[0] is now a tensor directly (not a list of padded tensors)
                gt_boxes_tensor = gt_boxes_list[0]  # tensor([N, 4]) or tensor([0, 4]) if empty
                
                # Check if there are any valid ground truth boxes
                if len(gt_boxes_tensor) > 0 and gt_boxes_tensor[0].sum() > 0:
                    # Take the first valid box
                    gt_box = gt_boxes_tensor[0]
                    gt_center_x = (gt_box[0] + gt_box[2]) / 2
                    gt_center_y = (gt_box[1] + gt_box[3]) / 2
                    print(f"GT center: ({gt_center_x:.1f}, {gt_center_y:.1f})")
                    
                    # If there are multiple boxes, show count
                    if len(gt_boxes_tensor) > 1:
                        print(f"Total GT boxes in image: {len(gt_boxes_tensor)}")
                        # Show centers of all boxes
                        for i, box in enumerate(gt_boxes_tensor):
                            if box.sum() > 0:  # Only show non-zero boxes
                                center_x = (box[0] + box[2]) / 2
                                center_y = (box[1] + box[3]) / 2
                                print(f"  GT box {i+1} center: ({center_x:.1f}, {center_y:.1f})")
                else:
                    print("No valid GT boxes")
                    
                if best_coords:
                    print(f"Best pred center: ({best_coords[0]:.1f}, {best_coords[1]:.1f}), conf: {best_conf:.3f}")
                else:
                    print("No prediction found")
                print(f"Image size: {images.shape[-2:]} (HxW)")
                print("-" * 40)

    
    def test_step(self, batch, batch_idx):
        """Test step"""
        return self.validation_step(batch, batch_idx)
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler with regularization"""
        # Only optimize trainable parameters
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        
        # AdamW optimizer with higher weight decay for regularization
        optimizer = AdamW(
            trainable_params,
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,  # Now 5e-4 instead of 1e-5
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Cosine decay scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.epochs,
            eta_min=1e-6
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }
    
    def predict_step(self, batch, batch_idx):
        """Prediction step for inference and visualization"""
        # If CSV export mode is enabled, we bypass all visualisation logic
        # and *only* gather predictions to be saved later in
        # ``on_predict_epoch_end``.
        if getattr(self, "save_to_csv", False) and self.save_to_csv:
            images  = batch["img"]   # (B,C,H,W)
            targets = batch["target"]

            # Forward & anchors as usual
            cls_logits, bbox_preds, pyramid_features = self(images)
            anchors_per_level = self.model.get_anchors(pyramid_features)

            batch_size = images.size(0)
            for i in range(batch_size):
                boxes_cat, scores_cat = self._gather_image_predictions(
                    cls_logits,
                    bbox_preds,
                    anchors_per_level,
                    img_idx=i,
                    conf_threshold=self.conf_threshold,
                    nms_threshold=self.nms_threshold,
                    topk=None,
                )

                # Meta-data with safe fall-backs (robust against missing keys or
                # lists that are shorter than the current batch index).

                patient_id = targets.get("patient_id", "unknown")[i]
                study_uid  = targets.get("study_uid",  "unknown")[i]
                view       = targets.get("view",      "unknown")[i]

                slice_val = int(targets.get("slice", -1)[i].item())

                for box, score in zip(boxes_cat.cpu(), scores_cat.cpu()):
                    self.pred_csv_rows.append({
                        "patient_id": patient_id,
                        "study_uid": study_uid,
                        "view": view,
                        "slice": slice_val,
                        "x1": float(box[0]),
                        "y1": float(box[1]),
                        "x2": float(box[2]),
                        "y2": float(box[3]),
                        "score": float(score),
                    })

            # Nothing to return for CSV mode; Lightning is fine with None
            return None
        # Extract images and targets from batch dictionary
        images = batch['img']  # [B, C, H, W]
        targets = batch['target']  # Dictionary of target lists/tensors
        
        # Forward pass
        cls_logits, bbox_preds, pyramid_features = self(images)
        
        # Generate anchors
        anchors_per_level = self.model.get_anchors(pyramid_features)
        
        # Get boxes and labels from target dictionary - NOW LISTS!
        gt_boxes_list = targets['boxes']  # List of tensors [tensor([N1, 4]), tensor([N2, 4]), ...]
        gt_labels_list = targets['labels']  # List of tensors [tensor([N1]), tensor([N2]), ...]
        
        # Process each image in the batch
        batch_size = images.shape[0]
        for i in range(batch_size):
            # ----------------------------------------------------------
            # Aggregate predictions from *all* levels, perform a *global* NMS,
            # then keep the top-5 boxes ranked by confidence.
            # ----------------------------------------------------------
            boxes_all = []
            scores_all = []

            for cls_logits_level, bbox_preds_level, anchors_level in zip(
                cls_logits, bbox_preds, anchors_per_level
            ):
                cls_pred = cls_logits_level[i]  # [A, H, W]
                bbox_pred = bbox_preds_level[i]  # [A*4, H, W]

                A, H, W = cls_pred.shape

                cls_pred_flat = cls_pred.view(A, -1).transpose(0, 1)  # [H*W, A]
                bbox_pred_flat = (
                    bbox_pred.view(A * 4, -1)
                    .transpose(0, 1)
                    .view(-1, A, 4)
                )  # [H*W, A, 4]

                probs = torch.sigmoid(cls_pred_flat)  # [H*W, A]

                # Mask by confidence threshold first
                conf_mask = probs > self.conf_threshold
                if conf_mask.any():
                    idx_flat = conf_mask.nonzero(as_tuple=False)
                    # idx_flat: [N_keep, 2] -> (loc_idx, anchor_idx)
                    loc_idx = idx_flat[:, 0]
                    anchor_idx = idx_flat[:, 1]

                    scores_sel = probs[loc_idx, anchor_idx]
                    anchors_sel = anchors_level[loc_idx * A + anchor_idx]
                    deltas_sel = bbox_pred_flat[loc_idx, anchor_idx]

                    # Decode
                    pred_x = deltas_sel[:, 0] * anchors_sel[:, 2] + anchors_sel[:, 0]
                    pred_y = deltas_sel[:, 1] * anchors_sel[:, 3] + anchors_sel[:, 1]
                    pred_w = torch.exp(deltas_sel[:, 2]) * anchors_sel[:, 2]
                    pred_h = torch.exp(deltas_sel[:, 3]) * anchors_sel[:, 3]

                    x1 = torch.clamp(pred_x - pred_w / 2, 0, images.shape[-1])
                    y1 = torch.clamp(pred_y - pred_h / 2, 0, images.shape[-2])
                    x2 = torch.clamp(pred_x + pred_w / 2, 0, images.shape[-1])
                    y2 = torch.clamp(pred_y + pred_h / 2, 0, images.shape[-2])

                    boxes_all.append(torch.stack([x1, y1, x2, y2], dim=1))
                    scores_all.append(scores_sel)

            if boxes_all:
                boxes_cat = torch.cat(boxes_all, dim=0)
                scores_cat = torch.cat(scores_all, dim=0)

                # Global NMS with more aggressive threshold
                keep = torchvision.ops.nms(boxes_cat, scores_cat, self.nms_threshold)
                boxes_cat = boxes_cat[keep]
                scores_cat = scores_cat[keep]

                # Keep top-5 by confidence
                if scores_cat.numel() > 0:
                    topk = torch.argsort(scores_cat, descending=True)[:5]
                    top_boxes = boxes_cat[topk]
                    top_scores = scores_cat[topk]
                else:
                    top_boxes = torch.empty((0, 4), device=images.device)
                    top_scores = torch.empty((0,), device=images.device)
            else:
                top_boxes = torch.empty((0, 4), device=images.device)
                top_scores = torch.empty((0,), device=images.device)

            # ------------------------------------------------------
            # Fallback: if no box survives the threshold + NMS and
            # ``pred_mode`` is 'always_top5', gather the *highest* scoring
            # anchors irrespective of the original confidence threshold so
            # that we always visualise something.
            # ------------------------------------------------------
            if top_boxes.numel() == 0 and self.pred_mode == "always_top5":
                boxes_all_low = []
                scores_all_low = []

                for cls_logits_level, bbox_preds_level, anchors_level in zip(
                    cls_logits, bbox_preds, anchors_per_level
                ):
                    cls_pred = cls_logits_level[i]
                    bbox_pred = bbox_preds_level[i]

                    A, H, W = cls_pred.shape
                    cls_pred_flat = cls_pred.view(A, -1).transpose(0, 1)
                    bbox_pred_flat = (
                        bbox_pred.view(A * 4, -1)
                        .transpose(0, 1)
                        .view(-1, A, 4)
                    )

                    probs = torch.sigmoid(cls_pred_flat)

                    # Flatten everything and take top K (e.g., 300) to avoid huge tensors
                    scores_flat = probs.flatten()
                    if scores_flat.numel() == 0:
                        continue
                    k_low = min(300, scores_flat.numel())
                    topk_scores, topk_idx_flat = torch.topk(scores_flat, k=k_low)

                    loc_idx = topk_idx_flat // A
                    anchor_idx = topk_idx_flat % A

                    anchors_sel = anchors_level[loc_idx * A + anchor_idx]
                    deltas_sel = bbox_pred_flat[loc_idx, anchor_idx]

                    # Decode
                    pred_x = deltas_sel[:, 0] * anchors_sel[:, 2] + anchors_sel[:, 0]
                    pred_y = deltas_sel[:, 1] * anchors_sel[:, 3] + anchors_sel[:, 1]
                    pred_w = torch.exp(deltas_sel[:, 2]) * anchors_sel[:, 2]
                    pred_h = torch.exp(deltas_sel[:, 3]) * anchors_sel[:, 3]

                    x1 = torch.clamp(pred_x - pred_w / 2, min=0)
                    y1 = torch.clamp(pred_y - pred_h / 2, min=0)
                    x2 = torch.clamp(pred_x + pred_w / 2, min=0)
                    y2 = torch.clamp(pred_y + pred_h / 2, min=0)

                    boxes_all_low.append(torch.stack([x1, y1, x2, y2], dim=1))
                    scores_all_low.append(topk_scores)

                if boxes_all_low:
                    boxes_cat_low = torch.cat(boxes_all_low, dim=0)
                    scores_cat_low = torch.cat(scores_all_low, dim=0)

                    # NMS with more aggressive threshold
                    keep_low = torchvision.ops.nms(boxes_cat_low, scores_cat_low, self.nms_threshold)
                    boxes_cat_low = boxes_cat_low[keep_low]
                    scores_cat_low = scores_cat_low[keep_low]

                    topk_final = torch.argsort(scores_cat_low, descending=True)[:5]
                    top_boxes = boxes_cat_low[topk_final]
                    top_scores = scores_cat_low[topk_final]

            # Create visualization
            plt.figure(figsize=(12, 12))
            
            # Plot image (handle both grayscale and RGB)
            img = images[i].cpu().numpy()
            if img.shape[0] == 3:  # RGB
                img = img.transpose(1, 2, 0)
                img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0, 1]
                plt.imshow(img)
            else:  # Grayscale
                img = img[0]  # Take first channel
                img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0, 1]
                plt.imshow(img, cmap='gray')
            
            # UPDATED: Plot ground truth boxes in green - using list format
            gt_boxes_tensor = gt_boxes_list[i]  # Get tensor for this image: tensor([N, 4])
            for gt_box in gt_boxes_tensor:
                if gt_box.sum() > 0:  # Only plot non-zero boxes
                    x1, y1, x2, y2 = gt_box.cpu().numpy()
                    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color='green', linewidth=3)
                    plt.gca().add_patch(rect)
            
            # Evaluate predictions according to challenge rule and plot with appropriate styling
            colors = ["red", "orange", "yellow", "blue", "purple"]
            for rank, (box, score) in enumerate(zip(top_boxes.cpu(), top_scores.cpu()), 1):
                x1, y1, x2, y2 = box.numpy()
                color = colors[(rank - 1) % len(colors)]
                
                # Evaluate if this prediction is a true positive
                is_true_positive = self._is_true_positive(box, gt_boxes_tensor.cpu().numpy())
                
                # Draw rectangle with solid line for TP, dashed line for FP
                linestyle = '-' if is_true_positive else '--'
                rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color=color, 
                                linewidth=2, linestyle=linestyle)
                plt.gca().add_patch(rect)
                
                # Annotate with rank, confidence, and TP/FP status
                tp_fp_label = "TP" if is_true_positive else "FP"
                plt.text(
                    x1,
                    y1 - 10,
                    f"#{rank}: {score:.3f} ({tp_fp_label})",
                    color=color,
                    fontsize=11,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                )
            
            # Add legend and title
            plt.plot([], [], color='green', linewidth=3, label='Ground Truth')
            # Add legend entries for each rank actually plotted
            for rank, color in enumerate(colors[: len(top_boxes)], 1):
                plt.plot([], [], color=color, linewidth=3, label=f'Pred #{rank}')
            # Add legend entries for line styles
            plt.plot([], [], color='black', linewidth=2, linestyle='-', label='True Positive')
            plt.plot([], [], color='black', linewidth=2, linestyle='--', label='False Positive')
            plt.legend(loc='upper right')
            plt.title(f'Lesion Detection - Batch {batch_idx}, Image {i}')
            
            # Remove axes for cleaner visualization
            plt.axis('off')
            
            # Retrieve metadata lists (if present) and safely index them. If a
            # particular list is shorter than the current batch index we fall
            # back to a sensible default value to avoid ``IndexError``.

            def _safe_get(key: str, default_value):
                value = targets.get(key, default_value)
                if isinstance(value, list):
                    return value[i] if i < len(value) else (default_value[0] if isinstance(default_value, list) else default_value)
                return value

            patient_id = _safe_get("patient_id", ["unknown"])
            study_uid  = _safe_get("study_uid",  ["unknown"])
            view       = _safe_get("view",      ["unknown"])

            # Handle slice numbers (still potentially a list of ints)
            slice_info = _safe_get("slice", [[0]])
            # Convert to a *single* integer as stored by the dataset.
            if torch.is_tensor(slice_info):
                # Tensor can be 0-D or 1-D; handle both.
                slice_val = int(slice_info.item()) if slice_info.numel() > 0 else -1
            elif isinstance(slice_info, list):
                # List of ints – take the first element (dataset always stores length-1 list).
                slice_val = int(slice_info[0]) if len(slice_info) > 0 else -1
            else:
                # Already a scalar (int/str/etc.)
                slice_val = int(slice_info)
            
            # UPDATED: Save figure with improved filename for multiple slices
            filename = f'pred_p{patient_id}_s{study_uid}_v{view}_sl{slice_val}_b{batch_idx}_i{i}.png'
            filepath = os.path.join(self.pred_output_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight', pad_inches=0.1)
            plt.close()
            
            # UPDATED: Print more informative message
            num_gt_boxes = len(gt_boxes_tensor)
            print(f"Saved prediction visualization: {filename} (GT boxes: {num_gt_boxes})")
        
        return None

    
    def get_parameter_count(self):
        """Get detailed parameter counts"""
        return self.model.get_parameter_count()

    def _is_true_positive(self, pred_box, gt_boxes):
        """
        Check if a predicted box is a true positive against any ground truth box.
        
        Args:
            pred_box: [4] predicted box (x1, y1, x2, y2)
            gt_boxes: [N, 4] ground truth boxes (x1, y1, x2, y2)
            
        Returns:
            bool: True if prediction is a true positive
        """
        if len(gt_boxes) == 0:
            return False
            
        pred_center_x = (pred_box[0] + pred_box[2]) / 2
        pred_center_y = (pred_box[1] + pred_box[3]) / 2
        
        for gt_box in gt_boxes:
            if np.sum(gt_box) == 0:  # Skip zero boxes
                continue
                
            gt_center_x = (gt_box[0] + gt_box[2]) / 2
            gt_center_y = (gt_box[1] + gt_box[3]) / 2
            
            # Calculate distance between centers
            distance = np.sqrt((pred_center_x - gt_center_x)**2 + (pred_center_y - gt_center_y)**2)
            
            # Calculate diagonal length of gt box
            gt_width = gt_box[2] - gt_box[0]
            gt_height = gt_box[3] - gt_box[1]
            diagonal_length = np.sqrt(gt_width**2 + gt_height**2)
            
            # Threshold is half of max(diagonal_length, 20_pixels)
            threshold = max(diagonal_length / 2, 20.0)
            
            # Check if this prediction matches this ground truth
            if distance < threshold:
                return True
                
        return False

    def _compute_froc(self, predictions_with_tp, n_images, n_gt_boxes, evaluation_fps):
        """
        Compute FROC curve and return sensitivity at specified FP rates.
        
        Args:
            predictions_with_tp: List of dicts with 'score', 'tp', 'image_id'
            n_images: Total number of images
            n_gt_boxes: Total number of ground truth boxes
            evaluation_fps: Tuple of FP rates to evaluate at
            
        Returns:
            List of sensitivity values at the specified FP rates
        """
        if n_gt_boxes == 0:
            return [0.0] * len(evaluation_fps)
            
        # Sort predictions by score (descending)
        predictions_with_tp.sort(key=lambda x: x['score'], reverse=True)
        
        # Get unique thresholds
        thresholds = [pred['score'] for pred in predictions_with_tp]
        thresholds = [max(thresholds) + 1.0] + thresholds + [min(thresholds) - 1.0]
        thresholds = sorted(set(thresholds), reverse=True)
        
        tpr = []
        fps = []
        
        for th in thresholds:
            # Count TPs and FPs at this threshold
            n_tps = 0
            n_fps = 0
            detected_gt_boxes = set()  # Track which GT boxes have been detected
            
            for pred in predictions_with_tp:
                if pred['score'] >= th:
                    if pred['tp']:
                        # Only count as TP if this GT box hasn't been detected yet
                        gt_key = (pred['image_id'], pred['gt_box_id'])
                        if gt_key not in detected_gt_boxes:
                            n_tps += 1
                            detected_gt_boxes.add(gt_key)
                    else:
                        n_fps += 1
            
            tpr_th = n_tps / n_gt_boxes
            fps_th = n_fps / n_images
            
            tpr.append(tpr_th)
            fps.append(fps_th)
            
            if fps_th > max(evaluation_fps):
                break
        
        # Interpolate sensitivity at desired FP rates
        return [np.interp(x, fps, tpr) for x in evaluation_fps]

    # --------------------------------------------------------------
    # Epoch-level metric logging
    # --------------------------------------------------------------
    def on_validation_epoch_end(self):
        """Compute and log mean Average Precision once per epoch."""
        metrics = self.val_map.compute()  # dict with multiple entries

        # Log a few key numbers as standalone scalars so Lightning is happy.
        # '.log' expects tensors, not dictionaries.
        self.log("val_map", metrics["map"], prog_bar=True, sync_dist=True)
        self.log("val_map_50", metrics["map_50"], prog_bar=False, sync_dist=True)

        # Reset metric state for next epoch
        self.val_map.reset()
        
        # Compute FROC-based sensitivity metric
        if len(self.val_predictions) > 0 and len(self.val_ground_truth) > 0:
            mean_sensitivity = self._compute_mean_sensitivity()
            self.log("val_mean_sensitivity_1_5_fps", mean_sensitivity, prog_bar=True, sync_dist=True)
            print(f"Mean sensitivity: {mean_sensitivity}")
        else:
            self.log("val_mean_sensitivity_1_5_fps", 0.0, prog_bar=True, sync_dist=True)
        
        # Reset FROC data for next epoch
        self.val_predictions = []
        self.val_ground_truth = []

    def _compute_mean_sensitivity(self):
        """
        Compute mean sensitivity across 1-5 false positives per image.
        
        Returns:
            float: Mean sensitivity value
        """
        # Build lookup for ground truth boxes by image_id
        gt_lookup = {}
        total_gt_boxes = 0
        for gt_data in self.val_ground_truth:
            image_id = gt_data['image_id']
            gt_boxes = gt_data['boxes']
            gt_lookup[image_id] = gt_boxes
            total_gt_boxes += len(gt_boxes)
        
        # Process all predictions and mark TPs/FPs
        predictions_with_tp = []
        for pred_data in self.val_predictions:
            image_id = pred_data['image_id']
            pred_boxes = pred_data['boxes']
            pred_scores = pred_data['scores']
            
            # Get ground truth for this image
            gt_boxes = gt_lookup.get(image_id, np.array([]).reshape(0, 4))
            
            # Track which GT boxes have been matched (to avoid double counting)
            matched_gt_indices = set()
            
            # Sort predictions by score (descending) for this image
            sorted_indices = np.argsort(pred_scores)[::-1]
            
            for idx in sorted_indices:
                pred_box = pred_boxes[idx]
                score = pred_scores[idx]
                
                # Determine if this prediction matches a GT box that has **not yet**
                # been detected.  If the same lesion was already hit by a higher-
                # scoring box we simply ignore this prediction (it is neither TP
                # nor FP) – exactly the behaviour of the external evaluation you
                # provided.

                is_tp = False
                matched_gt_idx = -1

                if len(gt_boxes) > 0:
                    for gt_idx, gt_box in enumerate(gt_boxes):
                        if gt_idx in matched_gt_indices:
                            continue  # this lesion already counted

                        if self._is_true_positive(pred_box, gt_box.reshape(1, -1)):
                            is_tp = True
                            matched_gt_idx = gt_idx
                            matched_gt_indices.add(gt_idx)
                            break

                # Only append the prediction if it is a *new* TP or an FP.  Duplicate
                # hits on an already-detected GT are discarded so they do not become
                # false positives later in the FROC calculation.
                if is_tp or matched_gt_idx == -1:
                    predictions_with_tp.append({
                        'score': float(score),
                        'tp': is_tp,
                        'image_id': image_id,
                        'gt_box_id': matched_gt_idx if is_tp else -1
                    })
        
        # Compute FROC curve and sensitivity at 1-5 FPs per image
        n_images = len(set(gt_lookup.keys()) | set(pred_data['image_id'] for pred_data in self.val_predictions))
        evaluation_fps = (1.0, 2.0, 3.0, 4.0, 5.0)
        
        sensitivities = self._compute_froc(
            predictions_with_tp, 
            n_images, 
            total_gt_boxes, 
            evaluation_fps
        )
        
        # Return mean sensitivity
        mean_sensitivity = np.mean(sensitivities)
        
        # Log individual sensitivities for debugging
        for i, (fps, sens) in enumerate(zip(evaluation_fps, sensitivities)):
            self.log(f"val_sensitivity_at_{int(fps)}_fps", sens, prog_bar=False, sync_dist=True)
        
        return mean_sensitivity

    # ------------------------------------------------------------------
    # Re-usable helper: collect predictions for *one* image using exactly
    # the same rules that are later used in predict_step visualisations
    # (confidence cutoff, global NMS across levels, optional top-K).
    # ------------------------------------------------------------------
    def _gather_image_predictions(
        self,
        cls_logits,
        bbox_preds,
        anchors_per_level,
        img_idx: int,
        conf_threshold: float | None = None,
        nms_threshold: float = 0.3,
        topk: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (boxes, scores) tensors for a single image index.

        Args:
            cls_logits / bbox_preds: lists per pyramid level (same as forward output)
            anchors_per_level: list of anchor tensors matching the levels
            img_idx: which sample in the batch to process
            conf_threshold: score cut-off; defaults to ``self.conf_threshold``
            nms_threshold: IoU used for NMS across *all* levels combined
            topk: if given, keep only the *topk* highest-scoring boxes
        Returns:
            boxes_cat (N,4) and scores_cat (N,) – both on the *same device*
        """

        if conf_threshold is None:
            conf_threshold = self.conf_threshold

        device = cls_logits[0].device

        boxes_all: list[torch.Tensor] = []
        scores_all: list[torch.Tensor] = []

        for cls_lvl, box_lvl, anchors_lvl in zip(cls_logits, bbox_preds, anchors_per_level):
            A, H, W = cls_lvl.shape[1], cls_lvl.shape[2], cls_lvl.shape[3]

            cls_pred = cls_lvl[img_idx]              # (A, H, W)
            bbox_pred = box_lvl[img_idx]             # (A*4, H, W)

            cls_pred_flat = cls_pred.view(A, -1).transpose(0, 1)       # (H*W, A)
            bbox_pred_flat = (
                bbox_pred.view(A * 4, -1).transpose(0, 1).view(-1, A, 4)
            )                                                       # (H*W, A, 4)

            probs = torch.sigmoid(cls_pred_flat)                     # (H*W, A)

            # Apply confidence threshold per-anchor
            conf_mask = probs > conf_threshold
            if not conf_mask.any():
                continue

            idx_flat = conf_mask.nonzero(as_tuple=False)
            loc_idx = idx_flat[:, 0]
            anchor_idx = idx_flat[:, 1]

            scores_sel = probs[loc_idx, anchor_idx]
            anchors_sel = anchors_lvl[loc_idx * A + anchor_idx]
            deltas_sel = bbox_pred_flat[loc_idx, anchor_idx]

            # Decode centre-format deltas into corner coords
            pred_x = deltas_sel[:, 0] * anchors_sel[:, 2] + anchors_sel[:, 0]
            pred_y = deltas_sel[:, 1] * anchors_sel[:, 3] + anchors_sel[:, 1]
            pred_w = torch.exp(deltas_sel[:, 2]) * anchors_sel[:, 2]
            pred_h = torch.exp(deltas_sel[:, 3]) * anchors_sel[:, 3]

            x1 = torch.clamp(pred_x - pred_w / 2, min=0)
            y1 = torch.clamp(pred_y - pred_h / 2, min=0)
            x2 = torch.clamp(pred_x + pred_w / 2, min=0)
            y2 = torch.clamp(pred_y + pred_h / 2, min=0)

            boxes_all.append(torch.stack([x1, y1, x2, y2], dim=1))
            scores_all.append(scores_sel)

        if not boxes_all:
            return torch.empty(0, 4, device=device), torch.empty(0, device=device)

        boxes_cat = torch.cat(boxes_all, dim=0)
        scores_cat = torch.cat(scores_all, dim=0)

        keep = torchvision.ops.nms(boxes_cat, scores_cat, nms_threshold)
        boxes_cat = boxes_cat[keep]
        scores_cat = scores_cat[keep]

        # Optional top-K selection (highest scores)
        if topk is not None and scores_cat.numel() > topk:
            top_idx = torch.argsort(scores_cat, descending=True)[:topk]
            boxes_cat = boxes_cat[top_idx]
            scores_cat = scores_cat[top_idx]

        return boxes_cat, scores_cat

    # --------------------------------------------------------------
    # Prediction-only hook: write out accumulated rows into a single
    # CSV file once the predict loop has finished.
    # --------------------------------------------------------------
    def on_predict_epoch_end(self, results=None):  # type: ignore[override]
        if getattr(self, "save_to_csv", False) and self.save_to_csv and hasattr(self, "pred_csv_rows") and len(self.pred_csv_rows) > 0:
            df = pd.DataFrame(self.pred_csv_rows)
            csv_path = os.path.join(self.pred_output_dir, self.csv_filename)
            df.to_csv(csv_path, index=False)
            logging.info(f"Saved {len(df)} predictions to {csv_path}")

