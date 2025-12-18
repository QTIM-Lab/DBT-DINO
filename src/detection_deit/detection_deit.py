import logging
import os
from pathlib import Path

import optuna
import pytorch_lightning as pl
import torch
from monai.data import DataLoader
from pycrumbs import tracked
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger

from data import get_datasets_detection_deit
from models.detection_deit_model import LesionDetectionModule
from utils import (
    detection_deit_objective,
    get_args_parser,
    setup_logging,
)


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

@tracked(directory_parameter="output_dir", include_timestamp=True)
def main(args, output_dir):
    args.output_dir = output_dir
    if args.log_dir is None:
        args.log_dir = output_dir

    # Set up logging
    setup_logging(output_dir)

    # generate random seed
    if args.seed is None:
        args.seed = torch.randint(0, 2**32 - 1, (1,)).item()
    # Set random seed and save it to the log file
    logging.info(f"Random seed: {args.seed}")

    pl.seed_everything(args.seed)
    torch.set_float32_matmul_precision("medium") # needed?

    dataset_train, dataset_val, dataset_test = get_datasets_detection_deit(
        args.csv_path, args.data_dir, persistent_cache=False, im_size=518
    )

    if args.use_optuna:
        # Create Optuna study with persistent storage
        if not args.db_folder:
            args.db_folder = output_dir
        storage_path = os.path.join(args.db_folder, "optuna.db")
        if not os.path.exists(args.db_folder):
            os.makedirs(args.db_folder)
        storage = optuna.storages.RDBStorage(
            url=f"sqlite:///{storage_path}",
            engine_kwargs={"connect_args": {"timeout": 30}},
        )

        study = optuna.create_study(
            study_name=args.study_name,
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5, n_warmup_steps=15, interval_steps=3
            ),
            storage=storage,  # Use the RDBStorage instance
            load_if_exists=True,
        )

        # Run optimization
        study.optimize(
            lambda trial: detection_deit_objective(
                trial,
                args,
                dataset_train,
                dataset_val,
                LesionDetectionModule,
            ),
            n_trials=args.n_trials,
            timeout=None,
        )

        # Print optimization results
        logging.info(f"Number of finished trials: {len(study.trials)}")
        logging.info("Best trial:")
        trial = study.best_trial
        logging.info(f"  Value: {trial.value}")
        logging.info("  Params: ")
        for key, value in trial.params.items():
            logging.info(f"    {key}: {value}")

        # Update args with best parameter
        args.lr = trial.params["lr"]
        args.weight_decay = trial.params["weight_decay"]
        args.label_smoothing = trial.params["label_smoothing"]
        args.dropout_rate = trial.params["dropout_rate"]
        args.cls_bbox_ratio = trial.params["cls_bbox_ratio"]
        args.focal_alpha = trial.params["focal_alpha"]
        args.focal_gamma = trial.params["focal_gamma"]
        args.iou_threshold = trial.params["iou_threshold"]
        args.neg_iou_threshold = trial.params["neg_iou_threshold"]
        args.neg_pos_ratio = trial.params["neg_pos_ratio"]
        args.nms_threshold = trial.params["nms_threshold"]


    # Create dataloaders
    data_loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        shuffle=True,
        collate_fn=detection_collate_fn,
    )
    data_loader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        collate_fn=detection_collate_fn,
    )
    data_loader_test = DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        collate_fn=detection_collate_fn,
    )

    # Train final model with best parameters
    model = LesionDetectionModule(args, pred_output_dir=os.path.join(output_dir, "predictions_before_train"))
    logger = TensorBoardLogger(args.log_dir, name=args.model)
    checkpoint_callback = ModelCheckpoint(
        monitor='val_mean_sensitivity_1_5_fps',  # Monitor mean sensitivity (challenge metric)
        mode='max',  # Higher is better
        save_top_k=1,  # Save only the best model
        save_last=True,  # Also save the last checkpoint
        filename="best-{epoch:02d}-{val_mean_sensitivity_1_5_fps:.4f}",
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    early_stopping = EarlyStopping(
        monitor='val_mean_sensitivity_1_5_fps',
        patience=10,
        mode='max',
        verbose=True
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor, early_stopping],
        accumulate_grad_batches=args.accum_iter,
        num_sanity_val_steps=2,
        precision="16-mixed",
        check_val_every_n_epoch=2,
    )

    # Run predictions on training set before training to look at transforms
    #logging.info("Running predictions on training set...")
    #trainer.predict(model, data_loader_train)
    #logging.info(f"Predictions saved to: {os.path.join(output_dir, 'predictions_before_train')}")


    trainer.fit(model, data_loader_train, data_loader_val)

    # Load the best checkpoint
    best_model_path = checkpoint_callback.best_model_path
    logging.info(f"Loading best model from: {best_model_path}")
    
    # Run predictions on training set
    logging.info("Running predictions on training set...")
    train_model = LesionDetectionModule.load_from_checkpoint(
        best_model_path,
        args=args,
        pred_output_dir=os.path.join(output_dir, "predictions_train")
    )
    train_model.eval()
    trainer.predict(train_model, data_loader_train)
    logging.info(f"Predictions saved to: {os.path.join(output_dir, 'predictions_train')}")

    # Run predictions on validation set
    logging.info("Running predictions on validation set...")
    val_model = LesionDetectionModule.load_from_checkpoint(
        best_model_path,
        args=args,
        pred_output_dir=os.path.join(output_dir, "predictions_val")
    )
    val_model.eval()
    trainer.predict(val_model, data_loader_val)
    logging.info(f"Predictions saved to: {os.path.join(output_dir, 'predictions_val')}")

    # Run predictions on test set
    logging.info("Running predictions on test set...")
    test_model = LesionDetectionModule.load_from_checkpoint(
        best_model_path,
        args=args,
        pred_output_dir=os.path.join(output_dir, "predictions_test")
    )
    test_model.eval()
    trainer.predict(test_model, data_loader_test)
    logging.info(f"Predictions saved to: {os.path.join(output_dir, 'predictions_test')}")

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = "./model_output"
    main(args, Path(output_dir))
