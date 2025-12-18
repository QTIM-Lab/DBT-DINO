import os
from typing import Hashable, Mapping, Sequence

import monai
import numpy as np
import pandas as pd
import torch
from monai.data import Dataset, PersistentDataset
from monai.transforms import (
    Compose,
    LoadImaged,
    MapTransform,
    RandAdjustContrastd,
    RandGaussianNoised,
    ScaleIntensityd,
)


class CreateDetectionTargetd(MapTransform):
    def __init__(self, keys, im_size=518, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.im_size = im_size

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> dict[Hashable, torch.Tensor]:
        d = dict(data)
        
        # Handle both list and tensor cases
        boxes = d['boxes']
        labels = d['labels']
        
        # Convert to tensors if they're lists
        if isinstance(boxes, list):
            if len(boxes) > 0:
                boxes = torch.tensor(boxes, dtype=torch.float32)
                labels = torch.tensor(labels, dtype=torch.float32)
            else:
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.zeros((0,), dtype=torch.float32)
        
        # Scale bounding boxes if needed
        if len(boxes) > 0 and self.im_size != 518:
            scale_factor = self.im_size / 518
            boxes = boxes * scale_factor
        
        d['target'] = {
            'boxes': boxes,
            'labels': labels,
            'patient_id': d['PatientID'],
            'study_uid': d.get('StudyUID', ''),
            'view': d.get('View', ''),
            'slice': d['SliceNumbers'],
        }
        
        return d


class RandCoarseDropoutBBoxAwared(MapTransform):
    """
    Randomly coarse dropout regions in the image while avoiding overlap with bounding boxes.
    
    This transform generates random rectangular regions for dropout but filters out any
    regions that would overlap with the provided bounding boxes to preserve lesion areas.
    
    Args:
        keys: keys of the corresponding items to be transformed.
        holes: number of regions to dropout, if `max_holes` is not None, use this arg as the minimum number.
        spatial_size: spatial size of the regions to dropout, if `max_spatial_size` is not None, use this arg
            as the minimum spatial size to randomly select size for every region.
        dropout_holes: if `True`, dropout the regions of holes and fill value, if `False`, keep the holes and
            dropout the outside and fill value. default to `True`.
        fill_value: target value to fill the dropout regions, if providing a number, will use it as constant
            value to fill all the regions. if providing a tuple for the `min` and `max`, will randomly select
            value for every pixel / voxel from the range `[min, max)`. if None, will compute the `min` and `max`
            value of input image then randomly select value to fill, default to None.
        max_holes: if not None, define the maximum number to randomly select the expected number of regions.
        max_spatial_size: if not None, define the maximum spatial size to randomly select size for every region.
        prob: probability of applying the transform.
        bbox_key: key to access bounding box information in the data dict.
        allow_missing_keys: don't raise exception if key is missing.
    """
    
    def __init__(
        self,
        keys,
        holes: int,
        spatial_size: Sequence[int] | int,
        dropout_holes: bool = True,
        fill_value: tuple[float, float] | float | None = None,
        max_holes: int | None = None,
        max_spatial_size: Sequence[int] | int | None = None,
        prob: float = 0.1,
        bbox_key: str = "target",
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.holes = holes
        self.spatial_size = spatial_size if isinstance(spatial_size, (list, tuple)) else [spatial_size, spatial_size]
        self.dropout_holes = dropout_holes
        self.max_holes = max_holes
        self.max_spatial_size = max_spatial_size if max_spatial_size is None else (
            max_spatial_size if isinstance(max_spatial_size, (list, tuple)) else [max_spatial_size, max_spatial_size]
        )
        self.prob = prob
        self.bbox_key = bbox_key
        
        if isinstance(fill_value, (tuple, list)):
            if len(fill_value) != 2:
                raise ValueError("fill value should contain 2 numbers if providing the `min` and `max`.")
        self.fill_value = fill_value
    
    def _boxes_overlap(self, box1, box2):
        """Check if two bounding boxes overlap.
        
        Args:
            box1: [x1, y1, x2, y2] format
            box2: [x1, y1, x2, y2] format
        
        Returns:
            bool: True if boxes overlap
        """
        return not (box1[2] <= box2[0] or box2[2] <= box1[0] or box1[3] <= box2[1] or box2[3] <= box1[1])
    
    def _generate_hole_coords(self, img_shape, bboxes):
        """Generate hole coordinates that don't overlap with bounding boxes.
        
        Args:
            img_shape: Shape of the image (H, W)
            bboxes: List of bounding boxes in [x1, y1, x2, y2] format
        
        Returns:
            List of coordinate tuples for numpy array indexing
        """
        if np.random.rand() >= self.prob:
            return []
        
        # Determine number of holes
        num_holes = self.holes
        if self.max_holes is not None:
            num_holes = np.random.randint(self.holes, self.max_holes + 1)
        
        hole_coords = []
        
        # Generate all holes randomly
        for _ in range(num_holes):
            # Determine hole size
            hole_h = self.spatial_size[0]
            hole_w = self.spatial_size[1] if len(self.spatial_size) > 1 else self.spatial_size[0]
            
            if self.max_spatial_size is not None:
                hole_h = np.random.randint(hole_h, self.max_spatial_size[0] + 1)
                hole_w = np.random.randint(hole_w, self.max_spatial_size[1] + 1)
            
            # Handle negative values (use image dimensions)
            if hole_h <= 0:
                hole_h = img_shape[0] + hole_h
            if hole_w <= 0:
                hole_w = img_shape[1] + hole_w
            
            # Ensure hole size doesn't exceed image dimensions
            hole_h = min(hole_h, img_shape[0])
            hole_w = min(hole_w, img_shape[1])
            
            # Generate random position
            if hole_h >= img_shape[0]:
                y1 = 0
                y2 = img_shape[0]
            else:
                y1 = np.random.randint(0, img_shape[0] - hole_h + 1)
                y2 = y1 + hole_h
            
            if hole_w >= img_shape[1]:
                x1 = 0
                x2 = img_shape[1]
            else:
                x1 = np.random.randint(0, img_shape[1] - hole_w + 1)
                x2 = x1 + hole_w
            
            # Check if this hole overlaps with any bounding box
            hole_bbox = [x1, y1, x2, y2]
            overlap_found = False
            
            for bbox in bboxes:
                if self._boxes_overlap(hole_bbox, bbox):
                    overlap_found = True
                    break
            
            # Only add hole if it doesn't overlap with any bounding box
            if not overlap_found:
                hole_coords.append((slice(None), slice(y1, y2), slice(x1, x2)))
        
        return hole_coords
    
    def _apply_dropout(self, img: np.ndarray, hole_coords):
        """Apply dropout to the specified coordinates."""
        if len(hole_coords) == 0:
            return img
        
        fill_value = (img.min(), img.max()) if self.fill_value is None else self.fill_value
        
        if self.dropout_holes:
            # Dropout the hole regions
            img_copy = img.copy()
            for h in hole_coords:
                if isinstance(fill_value, (tuple, list)):
                    img_copy[h] = np.random.uniform(fill_value[0], fill_value[1], size=img_copy[h].shape)
                else:
                    img_copy[h] = fill_value
            return img_copy
        else:
            # Keep the holes and dropout everything else
            if isinstance(fill_value, (tuple, list)):
                ret = np.random.uniform(fill_value[0], fill_value[1], size=img.shape).astype(img.dtype)
            else:
                ret = np.full_like(img, fill_value)
            for h in hole_coords:
                ret[h] = img[h]
            return ret
    
    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> dict[Hashable, torch.Tensor]:
        d = dict(data)
        
        # Extract bounding boxes if available
        bboxes = []
        if self.bbox_key in d and 'boxes' in d[self.bbox_key]:
            bbox_tensor = d[self.bbox_key]['boxes']
            # Convert tensor to list of lists for easier processing
            if bbox_tensor.numel() > 0:
                bboxes = bbox_tensor.cpu().numpy().tolist()
                # Ensure we have a list of boxes, even if there's only one
                if len(bbox_tensor.shape) == 1:
                    bboxes = [bboxes]
        
        for key in self.key_iterator(d):
            img = d[key]
            
            # Handle torch tensors
            if isinstance(img, torch.Tensor):
                img_np = img.cpu().numpy()
                
                # Get image shape (assuming format is [C, H, W])
                img_shape = img_np.shape[-2:]  # Get H, W
                
                # Generate hole coordinates
                hole_coords = self._generate_hole_coords(img_shape, bboxes)
                
                # Apply dropout
                img_np = self._apply_dropout(img_np, hole_coords)
                
                # Convert back to tensor
                d[key] = torch.from_numpy(img_np).to(img.device)
            else:
                # Handle numpy arrays directly
                img_shape = img.shape[-2:]  # Get H, W
                hole_coords = self._generate_hole_coords(img_shape, bboxes)
                d[key] = self._apply_dropout(img, hole_coords)
        
        return d


# -------------------------
# Dedicated horizontal-flip that always keeps image and bbox in sync.
# -------------------------


class RandHorizontalFlipBBoxd(MapTransform):
    """Randomly perform a left-right (horizontal) flip on the image together with its
    associated bounding boxes.

    This transform is functionally similar to ``RandFlipBoundingBoxd`` with
    ``spatial_axis=1``, but it is provided explicitly for clarity when you only
    want horizontal flips.  The bounding boxes are assumed to be in ``[x1, y1,
    x2, y2]`` format where *x2* is the *exclusive* right boundary (matching the
    convention used when boxes are created via ``CreateDetectionTargetd``).
    """

    def __init__(
        self,
        keys,
        prob: float = 0.5,
        bbox_key: str = "target",
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.prob = prob
        self.bbox_key = bbox_key

    def _flip_boxes(self, boxes: torch.Tensor, img_width: int, img_height: int) -> torch.Tensor:
        """Flip boxes horizontally.

        Args:
            boxes: ``(..., 4)`` tensor of ``[x1, y1, x2, y2]`` where ``x2`` is
                   *exclusive*.
            img_width: width *W* of the image that the boxes refer to.
            img_height: height *H* of the image that the boxes refer to.
        Returns:
            Tensor with the same shape containing the flipped coordinates.
        """
        single_box = boxes.ndim == 1
        if single_box:
            boxes = boxes.unsqueeze(0)  # shape (1,4)

        x1, y1, x2, y2 = boxes.unbind(-1)
        new_x1 = img_width - x2
        new_x2 = img_width - x1
        flipped = torch.stack((new_x1, y1, new_x2, y2), dim=-1)

        # Clamp to valid range just in case numerical errors push us outside.
        flipped = torch.clamp(
            flipped,
            min=boxes.new_tensor([0, 0, 0, 0]),
            max=boxes.new_tensor([img_width, img_height, img_width, img_height]),
        )

        if single_box:
            flipped = flipped.squeeze(0)
        return flipped

    def __call__(self, data: Mapping[Hashable, torch.Tensor]):
        d = dict(data)

        # Determine upfront whether we flip this sample
        if torch.rand(1).item() >= self.prob:
            return d

        # Flip all requested image keys left-right (width dimension = -1)
        for key in self.key_iterator(d):
            img = d[key]
            d[key] = torch.flip(img, dims=[-1])

        # Update bounding boxes so they stay aligned
        if self.bbox_key in d and "boxes" in d[self.bbox_key]:
            img_h, img_w = d[self.first_key(d)].shape[-2:]
            boxes = d[self.bbox_key]["boxes"]
            d[self.bbox_key]["boxes"] = self._flip_boxes(boxes, img_w, img_h)

        return d


# -------------------------
# Dedicated vertical-flip that always keeps image and bbox in sync.
# -------------------------


class RandVerticalFlipBBoxd(MapTransform):
    """Randomly perform an up-down (vertical) flip on the image together with its
    associated bounding boxes.

    This transform mirrors ``RandHorizontalFlipBBoxd`` but operates along the
    height axis (``spatial_axis=0``). The bounding boxes are assumed to be in
    ``[x1, y1, x2, y2]`` format where *y2* is the *exclusive* bottom boundary
    (matching the convention used when boxes are created via
    ``CreateDetectionTargetd``).
    """

    def __init__(
        self,
        keys,
        prob: float = 0.5,
        bbox_key: str = "target",
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.prob = prob
        self.bbox_key = bbox_key

    def _flip_boxes(self, boxes: torch.Tensor, img_width: int, img_height: int) -> torch.Tensor:
        """Flip boxes vertically.

        Args:
            boxes: ``(..., 4)`` tensor of ``[x1, y1, x2, y2]`` where ``y2`` is
                   *exclusive*.
            img_width: width *W* of the image that the boxes refer to.
            img_height: height *H* of the image that the boxes refer to.
        Returns:
            Tensor with the same shape containing the flipped coordinates.
        """
        single_box = boxes.ndim == 1
        if single_box:
            boxes = boxes.unsqueeze(0)  # shape (1,4)

        x1, y1, x2, y2 = boxes.unbind(-1)
        new_y1 = img_height - y2
        new_y2 = img_height - y1
        flipped = torch.stack((x1, new_y1, x2, new_y2), dim=-1)

        # Clamp to valid range just in case numerical errors push us outside.
        flipped = torch.clamp(
            flipped,
            min=boxes.new_tensor([0, 0, 0, 0]),
            max=boxes.new_tensor([img_width, img_height, img_width, img_height]),
        )

        if single_box:
            flipped = flipped.squeeze(0)
        return flipped

    def __call__(self, data: Mapping[Hashable, torch.Tensor]):
        d = dict(data)

        # Determine upfront whether we flip this sample
        if torch.rand(1).item() >= self.prob:
            return d

        # Flip all requested image keys up-down (height dimension = -2)
        for key in self.key_iterator(d):
            img = d[key]
            d[key] = torch.flip(img, dims=[-2])

        # Update bounding boxes so they stay aligned
        if self.bbox_key in d and "boxes" in d[self.bbox_key]:
            img_h, img_w = d[self.first_key(d)].shape[-2:]
            boxes = d[self.bbox_key]["boxes"]
            d[self.bbox_key]["boxes"] = self._flip_boxes(boxes, img_w, img_h)

        return d


# -------------------------
# Random zoom-in transform that adjusts bounding boxes accordingly.
# -------------------------

class RandZoomInBBoxd(MapTransform):
    """Randomly zoom an image *in* **or** *out* while keeping the associated bounding
    boxes consistent.

    The transform supports two modes controlled by the sampled ``zoom`` factor:

    * ``zoom > 1`` – *zoom in*: a smaller crop is taken and up-scaled back to the
      original size.
    * ``zoom < 1`` – *zoom out*: the image is down-scaled and pasted onto a canvas
      of the original size (with configurable ``pad_value``).

    Additional constraint: if ``bbox_size_limits`` is provided, the transform is
    only accepted when **all** resulting bounding boxes have widths and heights
    within the specified ranges. If no valid zoom is found after several
    attempts, the sample is returned unchanged.

    Bounding boxes use ``[x1, y1, x2, y2]`` where *x2*/*y2* denote **exclusive**
    right/bottom boundaries (as produced by ``CreateDetectionTargetd``).
    """

    def __init__(
        self,
        keys,
        zoom_range: tuple[float, float] = (0.8, 1.5),
        prob: float = 0.5,
        bbox_key: str = "target",
        allow_missing_keys: bool = False,
        mode: str | int = "bilinear",
        align_corners: bool | None = False,
        bbox_size_limits: tuple[float, float, float, float] | None = (15.0, 206.0, 9.0, 182.0),
        pad_value: float = 0.0,
    ) -> None:
        """Create the transform.

        Args:
            keys: Keys of the images to apply the transform to.
            zoom_range: ``(min_zoom, max_zoom)`` to sample ``zoom`` from. Values
                must be ``> 0``. ``zoom > 1`` zooms *in*, ``zoom < 1`` zooms *out*.
            prob: Probability of performing the transform.
            bbox_key: Dict key containing the detection target with ``"boxes"``.
            bbox_size_limits: ``(min_w, max_w, min_h, max_h)``. Bounding boxes
                outside this range invalidate the sampled transform. ``None``
                disables the check.
            pad_value: Constant used to fill the background when zooming out.
            mode, align_corners: Forwarded to ``F.interpolate``.
        """
        super().__init__(keys, allow_missing_keys)

        if zoom_range[0] <= 0 or zoom_range[1] < zoom_range[0]:
            raise ValueError("zoom_range must be (min>0, max>=min)")

        self.zoom_range = zoom_range
        self.prob = prob
        self.bbox_key = bbox_key
        self.mode = mode
        self.align_corners = align_corners
        self.bbox_size_limits = bbox_size_limits
        self.pad_value = pad_value

    # -------------------------------------------------------------
    # Helper methods for zoom-in (crop + resize)
    # -------------------------------------------------------------
    def _get_crop_params(
        self,
        boxes: torch.Tensor | None,
        img_h: int,
        img_w: int,
        zoom: float,
    ) -> tuple[int, int] | None:
        new_h = int(round(img_h / zoom))
        new_w = int(round(img_w / zoom))

        if boxes is None or boxes.numel() == 0:
            max_y = img_h - new_h
            max_x = img_w - new_w
            crop_y = torch.randint(0, max(max_y, 1), (1,)).item() if max_y > 0 else 0
            crop_x = torch.randint(0, max(max_x, 1), (1,)).item() if max_x > 0 else 0
            return crop_y, crop_x

        # Union of all boxes
        min_xy = boxes[:, :2].min(dim=0).values
        max_xy = boxes[:, 2:].max(dim=0).values
        minx, miny = min_xy.tolist()
        maxx, maxy = max_xy.tolist()

        # Crop must encompass the union
        if (maxx - minx) > new_w or (maxy - miny) > new_h:
            return None

        crop_x_min = max(0, int(maxx - new_w))
        crop_x_max = min(int(minx), img_w - new_w)
        crop_y_min = max(0, int(maxy - new_h))
        crop_y_max = min(int(miny), img_h - new_h)

        if crop_x_min > crop_x_max or crop_y_min > crop_y_max:
            return None

        crop_x = torch.randint(crop_x_min, crop_x_max + 1, (1,)).item() if crop_x_min < crop_x_max else crop_x_min
        crop_y = torch.randint(crop_y_min, crop_y_max + 1, (1,)).item() if crop_y_min < crop_y_max else crop_y_min
        return crop_y, crop_x

    def _apply_zoom_in(self, img: torch.Tensor, crop_y: int, crop_x: int, new_h: int, new_w: int, orig_h: int, orig_w: int) -> torch.Tensor:
        img = img[..., crop_y : crop_y + new_h, crop_x : crop_x + new_w]
        img = torch.nn.functional.interpolate(
            img.unsqueeze(0),
            size=(orig_h, orig_w),
            mode=self.mode,
            align_corners=self.align_corners if isinstance(self.mode, str) and self.mode in {"bilinear", "bicubic", "trilinear"} else None,
        ).squeeze(0)
        return img

    # -------------------------------------------------------------
    # Helper methods for zoom-out (shrink + pad)
    # -------------------------------------------------------------
    def _get_pad_params(self, img_h: int, img_w: int, new_h: int, new_w: int):
        max_y = img_h - new_h
        max_x = img_w - new_w
        pad_y = torch.randint(0, max(max_y, 1), (1,)).item() if max_y > 0 else 0
        pad_x = torch.randint(0, max(max_x, 1), (1,)).item() if max_x > 0 else 0
        return pad_y, pad_x

    def _apply_zoom_out(self, img: torch.Tensor, pad_y: int, pad_x: int, new_h: int, new_w: int, orig_h: int, orig_w: int) -> torch.Tensor:
        img_ds = torch.nn.functional.interpolate(
            img.unsqueeze(0),
            size=(new_h, new_w),
            mode=self.mode,
            align_corners=self.align_corners if isinstance(self.mode, str) and self.mode in {"bilinear", "bicubic", "trilinear"} else None,
        ).squeeze(0)

        canvas = img.new_full(img.shape, self.pad_value)
        canvas[..., pad_y : pad_y + new_h, pad_x : pad_x + new_w] = img_ds
        return canvas

    # -------------------------------------------------------------
    # Main entry-point
    # -------------------------------------------------------------
    def __call__(self, data: Mapping[Hashable, torch.Tensor]):
        d = dict(data)

        if torch.rand(1).item() >= self.prob:
            return d

        img_h, img_w = d[self.first_key(d)].shape[-2:]

        boxes: torch.Tensor | None = None
        if self.bbox_key in d and "boxes" in d[self.bbox_key]:
            boxes = d[self.bbox_key]["boxes"].clone()

        check_bbox_sizes = (
            boxes is not None and boxes.numel() > 0 and self.bbox_size_limits is not None
        )
        if check_bbox_sizes:
            min_w, max_w, min_h, max_h = self.bbox_size_limits

        # Try multiple times to satisfy all constraints
        for _ in range(15):
            zoom = torch.empty(1).uniform_(self.zoom_range[0], self.zoom_range[1]).item()

            # Skip identity (rare, but clearer semantics)
            if abs(zoom - 1.0) < 1e-3:
                continue

            if zoom > 1.0:
                # ------------------------------ ZOOM IN ------------------------------
                new_h = int(round(img_h / zoom))
                new_w = int(round(img_w / zoom))

                crop_params = self._get_crop_params(boxes, img_h, img_w, zoom)
                if crop_params is None:
                    continue
                crop_y, crop_x = crop_params

                if check_bbox_sizes:
                    pred_boxes = boxes.clone()
                    pred_boxes[:, 0] = (pred_boxes[:, 0] - crop_x) * zoom
                    pred_boxes[:, 2] = (pred_boxes[:, 2] - crop_x) * zoom
                    pred_boxes[:, 1] = (pred_boxes[:, 1] - crop_y) * zoom
                    pred_boxes[:, 3] = (pred_boxes[:, 3] - crop_y) * zoom

                    widths = pred_boxes[:, 2] - pred_boxes[:, 0]
                    heights = pred_boxes[:, 3] - pred_boxes[:, 1]
                    if (
                        widths.min() < min_w
                        or widths.max() > max_w
                        or heights.min() < min_h
                        or heights.max() > max_h
                    ):
                        continue

                # Apply to all images
                for key in self.key_iterator(d):
                    d[key] = self._apply_zoom_in(d[key], crop_y, crop_x, new_h, new_w, img_h, img_w)

                if boxes is not None:
                    boxes[:, 0] = (boxes[:, 0] - crop_x) * zoom
                    boxes[:, 2] = (boxes[:, 2] - crop_x) * zoom
                    boxes[:, 1] = (boxes[:, 1] - crop_y) * zoom
                    boxes[:, 3] = (boxes[:, 3] - crop_y) * zoom

                    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, img_w)
                    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, img_h)

                    d[self.bbox_key]["boxes"] = boxes

                return d

            else:
                # ------------------------------ ZOOM OUT -----------------------------
                new_h = int(round(img_h * zoom))
                new_w = int(round(img_w * zoom))

                if new_h <= 0 or new_w <= 0:
                    continue

                pad_y, pad_x = self._get_pad_params(img_h, img_w, new_h, new_w)

                if check_bbox_sizes:
                    pred_boxes = boxes.clone()
                    pred_boxes[:, [0, 2]] = pred_boxes[:, [0, 2]] * zoom + pad_x
                    pred_boxes[:, [1, 3]] = pred_boxes[:, [1, 3]] * zoom + pad_y

                    widths = pred_boxes[:, 2] - pred_boxes[:, 0]
                    heights = pred_boxes[:, 3] - pred_boxes[:, 1]
                    if (
                        widths.min() < min_w
                        or widths.max() > max_w
                        or heights.min() < min_h
                        or heights.max() > max_h
                    ):
                        continue

                for key in self.key_iterator(d):
                    d[key] = self._apply_zoom_out(d[key], pad_y, pad_x, new_h, new_w, img_h, img_w)

                if boxes is not None:
                    boxes[:, [0, 2]] = boxes[:, [0, 2]] * zoom + pad_x
                    boxes[:, [1, 3]] = boxes[:, [1, 3]] * zoom + pad_y

                    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, img_w)
                    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, img_h)

                    d[self.bbox_key]["boxes"] = boxes

                return d

        # No valid zoom found – return unmodified
        return d


class CorruptedImageFlipd(MapTransform):
    """Flip known corrupted images horizontally *and* vertically.

    Instead of matching full filenames, we only check if the basename **contains**
    any of the provided substrings (e.g. ``"DBT-P01461_DBT-S00251_lcc"``). This
    is more robust to differences in slice numbers or suffixes.
    """

    def __init__(
        self,
        keys,
        corrupted_keywords: list[str],
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        # store as tuple for fast membership via any(keyword in filename for keyword in self._keywords)
        self._keywords = tuple(corrupted_keywords)

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> dict[Hashable, torch.Tensor]:
        d = dict(data)

        filename = d.get("filename", "")
        if not isinstance(filename, str):
            return d

        # Flip if the basename contains any corrupted keyword
        if any(kw in filename for kw in self._keywords):
            for key in self.key_iterator(d):
                img = d[key]
                if isinstance(img, (torch.Tensor, monai.data.meta_tensor.MetaTensor)):
                    d[key] = torch.flip(img, dims=[-2, -1])  # vertical (-2) and horizontal (-1)

        return d


# Helper function to group slices based on proximity
def _group_slices_by_proximity(group, volume_slices):
    """
    Group bounding boxes by slice proximity.
    Only combine boxes if they are within 0.25*VolumeSlices of each other.
    """
    if len(group) == 1:
        return [group]
    
    # Calculate the slice distance threshold
    slice_threshold = 0.25 * volume_slices
    
    # Sort by slice number
    sorted_group = group.sort_values('Slice').reset_index(drop=True)
    
    # Create slice groups using a simple clustering approach
    slice_groups = []
    used_indices = set()
    
    for i in range(len(sorted_group)):
        if i in used_indices:
            continue
            
        # Start a new group with the current slice
        current_group = [sorted_group.iloc[i]]
        used_indices.add(i)
        current_slice = sorted_group.iloc[i]['Slice']
        
        # Find all other slices within the threshold
        for j in range(i + 1, len(sorted_group)):
            if j in used_indices:
                continue
                
            slice_diff = abs(sorted_group.iloc[j]['Slice'] - current_slice)
            if slice_diff <= slice_threshold:
                current_group.append(sorted_group.iloc[j])
                used_indices.add(j)
        
        # Add this group to the list
        slice_groups.append(pd.DataFrame(current_group))
    
    return slice_groups

def _create_detection_dict_with_slices(group):
    # Get the first row for image information
    first_row = group.iloc[0]
    
    # Create list of bounding boxes (convert to x1, y1, x2, y2 format)
    boxes = []
    slice_numbers = []
    for _, row in group.iterrows():
        boxes.append([row['X'], row['Y'], row['X'] + row['Width'], row['Y'] + row['Height']])
        slice_numbers.append(row['Slice'])
    
    # Convert to tensors directly
    boxes_tensor = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
    labels_tensor = torch.ones(len(boxes), dtype=torch.float32) if boxes else torch.zeros((0,), dtype=torch.float32)
    
    return {
        'img': first_row['FilePath'],
        'PatientID': first_row['PatientID'],
        'StudyUID': first_row.get('StudyUID', ''),
        'View': first_row.get('View', ''),
        'Slice': first_row['Slice'],  # Primary slice (first one)
        'SliceNumbers': slice_numbers,  # All slice numbers for the boxes
        'VolumeSlices': first_row.get('VolumeSlices', 1),
        'SliceThreshold': 0.25 * first_row.get('VolumeSlices', 1),
        'boxes': boxes_tensor,  # Now a tensor [N, 4]
        'labels': labels_tensor,  # Now a tensor [N]
        'filename': os.path.basename(first_row['FilePath']),
    }

def get_datasets_detection_deit(
    csv_path,
    data_dir,
    persistent_cache=True,
    im_size=518,
    val_size=0.2,
    random_state=42
):
    """
    Create train, validation, and test datasets for breast cancer detection.
    
    Args:
        csv_path: Path to CSV file with detection data (from cropping_preprocessing.py)
        data_dir: Directory containing preprocessed .npy image files
        persistent_cache: Whether to use persistent cache for faster loading
        im_size: Target image size for the model (518 default)
        val_size: Not used when split column exists (kept for backward compatibility)
        random_state: Not used when split column exists (kept for backward compatibility)
    
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    
    # Load the detection data
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} detections from {csv_path}")
    
    # Check if files exist and filter
    len_df = len(df)
    df = df[df['FilePath'].apply(lambda x: os.path.exists(x))]
    print(f"Dropped {len_df - len(df)} rows because their file path does not exist")
    
    # Check if 'split' column exists
    if 'split' not in df.columns:
        raise ValueError("The dataframe must contain a 'split' column to determine train/val/test splits")
    
    # Get unique split values
    unique_splits = df['split'].unique()
    print(f"Found splits: {unique_splits}")
    
    # Filter data by split
    train_df = df[df['split'] == 'train']
    val_df = df[df['split'].isin(['val', 'validation'])]  # Support both 'val' and 'validation'
    test_df = df[df['split'] == 'test']
    
    print(f"Split sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    
    # Create data lists for train, val, and test
    train_data = []
    val_data = []
    test_data = []
    
    # Group by StudyUID and View for each split, then by slice proximity
    for split_df, data_list, split_name in [(train_df, train_data, 'train'), (val_df, val_data, 'val'), (test_df, test_data, 'test')]:
        if len(split_df) > 0:
            grouped = split_df.groupby(['StudyUID', 'View'])
            start_len = len(data_list)
            
            split_examples = []  # Track examples of slice grouping for this split
            
            for _, group in grouped:
                # Get VolumeSlices for this group (should be same for all rows)
                volume_slices = group.iloc[0].get('VolumeSlices', 1)
                
                # Group by slice proximity within this StudyUID/View group
                slice_groups = _group_slices_by_proximity(group, volume_slices)
                
                # Track if this study/view was split into multiple slice groups
                if len(slice_groups) > 1:
                    slice_ranges = []
                    for sg in slice_groups:
                        slices = sg['Slice'].tolist()
                        slice_ranges.append(f"{min(slices)}-{max(slices)}")
                    split_examples.append({
                        'StudyUID': group.iloc[0]['StudyUID'],
                        'View': group.iloc[0]['View'],
                        'VolumeSlices': volume_slices,
                        'OriginalBoxes': len(group),
                        'SliceGroups': len(slice_groups),
                        'SliceRanges': slice_ranges
                    })
                
                # Create a data entry for each slice group
                for slice_group in slice_groups:
                    data_list.append(_create_detection_dict_with_slices(slice_group))
            
            # Print statistics
            num_groups = len(data_list) - start_len
            total_boxes = sum(len(d['boxes']) for d in data_list[-num_groups:])
            unique_images = len(set(d['StudyUID'] + '_' + d['View'] for d in data_list[-num_groups:]))
            
            print(f"\n{split_name.upper()}:")
            print(f"  {len(split_df)} detections -> {num_groups} slice groups from {unique_images} unique images")
            print(f"  Total boxes: {total_boxes}")
            
            # Show examples of studies that were split due to slice distance
            if split_examples:
                print(f"  {len(split_examples)} studies split into multiple slice groups:")
                for i, ex in enumerate(split_examples):  # Show first 3 examples
                    print(f"    {i+1}. {ex['StudyUID']}-{ex['View']}: {ex['OriginalBoxes']} boxes -> {ex['SliceGroups']} groups (slices: {', '.join(ex['SliceRanges'])})")
            else:
                print("  No studies were split (all boxes within slice proximity threshold)")

    
    print(f"Found {len(train_data)} training detections, {len(val_data)} validation detections, and {len(test_data)} test detections")
    
    # List of known corrupted files that need flipping
    corrupted_files = [
        "DBT-P01461_DBT-S00251_lcc",
        "DBT-P01461_DBT-S00251_lmlo",
        "DBT-P02471_DBT-S03894_lcc",
        "DBT-P02471_DBT-S03894_lmlo",
        "DBT-P02510_DBT-S04417_lcc",
        "DBT-P02510_DBT-S04417_lmlo",
        "DBT-P03176_DBT-S03730_lcc",
        "DBT-P03176_DBT-S03730_lmlo",
        "DBT-P01150_DBT-S04884_lmlo",
        "DBT-P03027_DBT-S03974_lcc",
        "DBT-P03027_DBT-S03974_lmlo",
        "DBT-P00715_DBT-S01349_lcc",
        "DBT-P00715_DBT-S01349_lmlo",
    ]
    
    # Define base transforms for both train and val
    base_transforms = [
        LoadImaged(keys=["img"]),  # Use MONAI's standard loader
        ScaleIntensityd(keys=["img"]),  # Normalize to [0, 1] range
        CorruptedImageFlipd(keys=["img"], corrupted_keywords=corrupted_files),  # Fix corrupted images
        CreateDetectionTargetd(keys=["img"], im_size=im_size),
    ]
    
    # Define validation and test transforms (no augmentation)
    val_test_transforms = Compose(base_transforms)
    
    # Define training transforms with augmentation
    train_transforms = Compose([
        *base_transforms,
        RandHorizontalFlipBBoxd(keys=["img"], prob=0.5),
        RandVerticalFlipBBoxd(keys=["img"], prob=0.5),
        RandZoomInBBoxd(keys=["img"], zoom_range=(0.8, 1.5), prob=0.5),
        RandCoarseDropoutBBoxAwared(
            keys=["img"],
            holes=15,  # Number of holes to attempt
            spatial_size=(20, 20),  # Size of each hole
            max_holes=20,  # Maximum number of holes
            max_spatial_size=(40, 40),  # Maximum hole size
            prob=0.3,
            fill_value=0,  # Random values between 0 and 1
            bbox_key="target",
        ),
        # Random contrast adjustment (30% probability)
        RandAdjustContrastd(
            keys=["img"],
            prob=0.3,
            gamma=(0.8, 1.2),  # Subtle contrast changes
        ),
        # Random Gaussian noise (30% probability)
        RandGaussianNoised(
            keys=["img"],
            prob=0.3,
            mean=0.0,
            std=0.05,  # Small noise level
        ),
    ])
    
    if persistent_cache:
        tmpdir = "/vast/scratch/fd881_persistent_cache/detection_deit_baseline"
        print(f"Creating Persistent Dataset at: {tmpdir}")
        train_ds = PersistentDataset(
            data=train_data, transform=train_transforms, cache_dir=tmpdir
        )
        
        val_ds = PersistentDataset(
            data=val_data, transform=val_test_transforms, cache_dir=tmpdir
        )
        
        test_ds = PersistentDataset(
            data=test_data, transform=val_test_transforms, cache_dir=tmpdir
        )
    else:
        train_ds = Dataset(data=train_data, transform=train_transforms)
        val_ds = Dataset(data=val_data, transform=val_test_transforms)
        test_ds = Dataset(data=test_data, transform=val_test_transforms)
    
    return train_ds, val_ds, test_ds

def get_datasets_detection_deit_all_slices(
    csv_path,
    data_dir,
    persistent_cache=True,
    im_size=518,
    val_size=0.2,
    random_state=42,
):
    """Create *un-grouped* train/val/test datasets for breast-cancer detection.

    Every **row** of ``csv_path`` becomes one sample – no grouping across slices or
    studies is performed. The CSV must contain the following columns produced by
    ``cropping_preprocessing.py`` (at minimum)::

        FilePath, PatientID, X, Y, Width, Height, Slice, split

    Optional (but recommended) columns that will be propagated if present::

        StudyUID, View, VolumeSlices

    Notes
    -----
    • The function mirrors ``get_datasets_detection_deit`` but deliberately
      *omits* any grouping logic and any data-augmentation transforms that act
      on the bounding boxes.  Only the *base* transforms are applied so that
      bounding boxes remain untouched.
    • The *data_dir* argument is retained for API compatibility but is **not**
      used (the ``FilePath`` column is assumed to contain absolute paths).

    Returns
    -------
    tuple[Dataset | PersistentDataset, ...]
        ``(train_ds, val_ds, test_ds)``
    """

    # ---------------------------------------------------------------------
    # 1) Read CSV and basic filtering
    # ---------------------------------------------------------------------
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} detections from {csv_path}")

    # Drop rows whose image file does not exist
    initial_len = len(df)
    df = df[df["FilePath"].apply(os.path.exists)]
    if len(df) < initial_len:
        print(f"Dropped {initial_len - len(df)} rows because their file path does not exist")

    # Sort the df by patient_id, study_uid, view, slice
    df = df.sort_values(['PatientID', 'StudyUID', 'View', 'Slice'])

    # ---------------------------------------------------------------------
    # 2) Prepare per-row dictionaries (no bounding boxes)
    # ---------------------------------------------------------------------
    def _row_to_detection_dict(row):
        """Convert dataframe row into MONAI input dict (image only)."""
        return {
            "img": row["FilePath"],
            "PatientID": row["PatientID"],
            "StudyUID": row.get("StudyUID", ""),
            "View": row.get("View", ""),
            "SliceNumbers": [row.get("Slice", -1)],  # Changed to list format
            "VolumeSlices": row.get("VolumeSlices", 1),
            "filename": os.path.basename(row["FilePath"]),
            "target": {
                "patient_id": row["PatientID"],
                "study_uid": row.get("StudyUID", ""),
                "view": row.get("View", ""),
                "slice": row.get("Slice", -1),
                "boxes": torch.zeros((0, 4), dtype=torch.float32),  # Empty boxes tensor
                "labels": torch.zeros((0,), dtype=torch.float32)   # Empty labels tensor
            }
        }

    train_data = [
        _row_to_detection_dict(r)
        for _, r in df[df["split"] == "train"].iterrows()
    ]
    val_data = [
        _row_to_detection_dict(r)
        for _, r in df[df["split"].isin(["val", "validation"])].iterrows()
    ]
    test_data = [
        _row_to_detection_dict(r)
        for _, r in df[df["split"] == "test"].iterrows()
    ]


    print(
        f"Created {len(train_data)} training, {len(val_data)} validation and {len(test_data)} test samples (no grouping)"
    )

    # ---------------------------------------------------------------------
    # 3) Define transforms – *base* only (no bbox / target creation)
    # ---------------------------------------------------------------------
    corrupted_files = [
        "DBT-P01461_DBT-S00251_lcc",
        "DBT-P01461_DBT-S00251_lmlo",
        "DBT-P02471_DBT-S03894_lcc",
        "DBT-P02471_DBT-S03894_lmlo",
        "DBT-P02510_DBT-S04417_lcc",
        "DBT-P02510_DBT-S04417_lmlo",
        "DBT-P03176_DBT-S03730_lcc",
        "DBT-P03176_DBT-S03730_lmlo",
        "DBT-P01150_DBT-S04884_lmlo",
        "DBT-P03027_DBT-S03974_lcc",
        "DBT-P03027_DBT-S03974_lmlo",
        "DBT-P00715_DBT-S01349_lcc",
        "DBT-P00715_DBT-S01349_lmlo",
    ]

    base_transforms = Compose(
        [
            LoadImaged(keys=["img"]),
            ScaleIntensityd(keys=["img"]),
            CorruptedImageFlipd(keys=["img"], corrupted_keywords=corrupted_files),
        ]
    )

    # For this inference-oriented dataset we use identical transforms for all splits
    train_transforms = base_transforms
    val_test_transforms = base_transforms

    # ---------------------------------------------------------------------
    # 4) Wrap with (Persistent)Dataset
    # ---------------------------------------------------------------------
    if persistent_cache:
        tmpdir = "/vast/scratch/fd881_persistent_cache/detection_deit_all_slices"
        print(f"Creating Persistent Dataset at: {tmpdir}")
        train_ds = PersistentDataset(data=train_data, transform=train_transforms, cache_dir=tmpdir)
        val_ds = PersistentDataset(data=val_data, transform=val_test_transforms, cache_dir=tmpdir)
        test_ds = PersistentDataset(data=test_data, transform=val_test_transforms, cache_dir=tmpdir)
    else:
        train_ds = Dataset(data=train_data, transform=train_transforms)
        val_ds = Dataset(data=val_data, transform=val_test_transforms)
        test_ds = Dataset(data=test_data, transform=val_test_transforms)

    return train_ds, val_ds, test_ds