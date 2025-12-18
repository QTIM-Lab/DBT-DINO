import pytorch_lightning as pl
import torch
from monai.data import DataLoader
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from utils import pad_collate_fn


class OptunaPruning(PyTorchLightningPruningCallback, pl.Callback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def objective(trial, args, train_dataset, val_dataset, model_class):
    """
    Optuna objective function for hyperparameter optimization.
    
    Args:
        trial: Optuna trial object
        args: Arguments namespace
        train_dataset: Training dataset
        val_dataset: Validation dataset
        model_class: Lightning module class to optimize
    """
    # Only tune the learning rate
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    #weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True)
    #attention_type = trial.suggest_categorical("attention_type", ["gated", "standard"])
    use_ms = False #trial.suggest_categorical("use_ms", [True, False])

    # Update args with trial parameter
    args.lr = lr
    args.use_ms = use_ms
    #args.attention_type = attention_type
    #args.weight_decay = weight_decay

    # print the modified args
    print(f"lr: {lr}, use_ms: {use_ms}, attention_type: {args.attention_type}, weight_decay: {args.weight_decay} for the optuna trial")
    
    # Create dataloaders with fixed batch size
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        collate_fn=pad_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        collate_fn=pad_collate_fn,
    )

    # Initialize model
    model = model_class(args)

    # Set up callbacks including the pruning callback
    logger = TensorBoardLogger(args.log_dir, name=f"{args.model}_trial_{trial.number}")
    pruning_callback = OptunaPruning(trial, monitor="val_loss")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        filename="{epoch:02d}-{val_loss:.2f}",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor, pruning_callback],
        precision="16-mixed",
        accumulate_grad_batches=args.accum_iter,
        num_sanity_val_steps=2,
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    # Return the best validation loss
    return trainer.callback_metrics["val_loss"].item()



def risk_objective(trial, args, train_loader, val_loader, model_name):
    """
    Optuna objective function for hyperparameter optimization.

    Args:
        trial: Optuna trial object
        args: Arguments namespace
        train_dataset: Training dataset
        val_dataset: Validation dataset
        model: Lightning module to optimize
    """

    # Only tune the learning rate
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    #use_ms = trial.suggest_categorical("use_ms", [True, False])

    # Update args with trial parameter
    args.lr = lr
    args.use_ms = False
    print("Running Optuna optimization")
    model = model_name(args=args, max_followup=args.max_followup, embed_type=args.embedding_type, stats=args.stats)

    # Set up callbacks including the pruning callback
    logger = TensorBoardLogger(args.log_dir, name=f"{args.model}_trial_{trial.number}")
    pruning_callback = OptunaPruning(trial, monitor="val_loss")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        filename="{epoch:02d}-{val_loss:.2f}",
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor, pruning_callback],
        accumulate_grad_batches=args.accum_iter,
        num_sanity_val_steps=2,
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    # Return the best validation loss
    return trainer.callback_metrics["val_loss"].item()

# Custom collate function for variable number of boxes per image
def detection_collate_fn(batch):
    """
    Custom collate function that handles variable numbers of boxes per image.
    Uses lists instead of padding to avoid fake boxes.
    """
    # Stack images normally (all same size)
    images = torch.stack([item['img'] for item in batch])
    
    # Keep boxes and labels as lists (no padding)
    targets = {
        'boxes': [item['target']['boxes'] for item in batch],  # List of tensors [N_i, 4]
        'labels': [item['target']['labels'] for item in batch],  # List of tensors [N_i]
        'patient_id': [item['target']['patient_id'] for item in batch],
        'study_uid': [item['target']['study_uid'] for item in batch],
        'view': [item['target']['view'] for item in batch],
        'slice': [item['target']['slice'] for item in batch],
    }
    
    return {'img': images, 'target': targets}

def detection_deit_objective(trial, args, train_dataset, val_dataset, model_class):
    """
    Optuna objective function for hyperparameter optimization.
    
    Args:
        trial: Optuna trial object
        args: Arguments namespace
        train_dataset: Training dataset
        val_dataset: Validation dataset
        model_class: Lightning module class to optimize
    """
    # --------------------------------------------------------------
    # Hyper-parameters to tune for Detection-DEiT model
    # --------------------------------------------------------------
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.1)

    dropout_rate = trial.suggest_categorical("dropout_rate", [0.0, 0.05, 0.1, 0.15])

    cls_bbox_ratio = trial.suggest_float("cls_bbox_ratio", 0.1, 10.0, log=True)

    focal_alpha = trial.suggest_float("focal_alpha", 0.25, 0.95)
    focal_gamma = trial.suggest_float("focal_gamma", 0.5, 2.0)

    iou_threshold = trial.suggest_float("iou_threshold", 0.4, 0.6)
    neg_iou_threshold = trial.suggest_float("neg_iou_threshold", 0.2, 0.5)

    neg_pos_ratio = trial.suggest_categorical("neg_pos_ratio", [1, 2, 3, 4, 5])

    nms_threshold = trial.suggest_float("nms_threshold", 0.03, 0.3)

    # --------------------------------------------------------------
    # Write suggestions back to args so the LightningModule can read
    # them in its __init__
    # --------------------------------------------------------------
    args.lr = lr
    args.weight_decay = weight_decay
    args.label_smoothing = label_smoothing
    args.dropout_rate = dropout_rate
    args.cls_bbox_ratio = cls_bbox_ratio
    args.focal_alpha = focal_alpha
    args.focal_gamma = focal_gamma
    args.iou_threshold = iou_threshold
    args.neg_iou_threshold = neg_iou_threshold
    args.neg_pos_ratio = neg_pos_ratio
    args.nms_threshold = nms_threshold

    print({
        "lr": lr,
        "weight_decay": weight_decay,
        "label_smoothing": label_smoothing,
        "dropout_rate": dropout_rate,
        "cls_bbox_ratio": cls_bbox_ratio,
        "focal_alpha": focal_alpha,
        "focal_gamma": focal_gamma,
        "iou_threshold": iou_threshold,
        "neg_iou_threshold": neg_iou_threshold,
        "neg_pos_ratio": neg_pos_ratio,
        "nms_threshold": nms_threshold,
    })

    # Create dataloaders with fixed batch size
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        collate_fn=detection_collate_fn,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        collate_fn=detection_collate_fn,
        drop_last=False,
    )

    # Initialize model
    model = model_class(args)

    # Set up callbacks including the pruning callback
    logger = TensorBoardLogger(args.log_dir, name=f"{args.model}_trial_{trial.number}")
    pruning_callback = OptunaPruning(trial, monitor="val_mean_sensitivity_1_5_fps")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_mean_sensitivity_1_5_fps",
        save_top_k=1,
        mode="max",
        filename="{epoch:02d}-{val_mean_sensitivity_1_5_fps:.4f}",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor, pruning_callback],
        precision="16-mixed",
        accumulate_grad_batches=args.accum_iter,
        num_sanity_val_steps=2,
        check_val_every_n_epoch=2,
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    # Return the best validation mean sensitivity
    return trainer.callback_metrics["val_mean_sensitivity_1_5_fps"].item()