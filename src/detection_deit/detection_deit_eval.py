import logging
import os
from pathlib import Path

import pytorch_lightning as pl
import torch
from monai.data import DataLoader
from monai.data.utils import list_data_collate
from pycrumbs import tracked
from pytorch_lightning.loggers import TensorBoardLogger

from data import (
    get_datasets_detection_deit,
    get_datasets_detection_deit_all_slices,
)
from models.detection_deit_model import LesionDetectionModule
from utils import (
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

    # ------------------------------------------------------------------
    # Select the appropriate dataset implementation
    # ------------------------------------------------------------------
    if getattr(args, "all_slices", False):
        dataset_train, dataset_val, dataset_test = get_datasets_detection_deit_all_slices(
            args.csv_path,
            args.data_dir,
            persistent_cache=False,
            im_size=518,
        )
        logging.info("Using *all-slices* dataset – every slice is treated as an individual sample.")
    else:
        dataset_train, dataset_val, dataset_test = get_datasets_detection_deit(
            args.csv_path,
            args.data_dir,
            persistent_cache=False,
            im_size=518,
        )
        logging.info("Using *grouped* detection dataset with bounding-box targets.")

    #if args.use_optuna:
    #    # Create Optuna study with persistent storage
    #    if not args.db_folder:
    #        args.db_folder = output_dir
    #    storage_path = os.path.join(args.db_folder, "optuna.db")
    #    if not os.path.exists(args.db_folder):
    #        os.makedirs(args.db_folder)
    #    storage = optuna.storages.RDBStorage(
    #        url=f"sqlite:///{storage_path}",
    #        engine_kwargs={"connect_args": {"timeout": 30}},
    #    )
    #    # Load the study from the database
    #    study = optuna.load_study(
    #        study_name=args.study_name,
    #        storage=storage, 
    #    )
#
    #    logging.info(f"Loaded study from {storage_path}")
    #    logging.info(f"Number of finished trials: {len(study.trials)}")
#
    #    # If no trial number is provided, use the best trial
    #    if args.trial_number is None:
    #        logging.info("No trial number provided, using best trial")
    #        args.trial_number = study.best_trial.number
#
#
    #    # Get dataframe with all trials and save for future reference
    #    df_trials = study.trials_dataframe()
    #    df_trials.to_csv(os.path.join(output_dir, "trials.csv"))
#
#
    #    # Use values from the selected trial (specified via args.trial_number)
    #    try:
    #        trial_row = df_trials[df_trials["number"] == args.trial_number].iloc[0]
    #    except IndexError:
    #        raise ValueError(f"Trial number {args.trial_number} not found in Optuna study.")
#
    #    logging.info(f"Loading parameters from trial #{args.trial_number} (value={trial_row['value']})")
#
    #    # Update args with parameters stored in the dataframe. Columns follow the pattern 'params_<param_name>'.
    #    args.lr = float(trial_row["params_lr"])
    #    args.weight_decay = float(trial_row["params_weight_decay"])
    #    args.label_smoothing = float(trial_row["params_label_smoothing"])
    #    args.dropout_rate = float(trial_row["params_dropout_rate"])
    #    args.cls_bbox_ratio = float(trial_row["params_cls_bbox_ratio"])
    #    args.focal_alpha = float(trial_row["params_focal_alpha"])
    #    args.focal_gamma = float(trial_row["params_focal_gamma"])
    #    args.iou_threshold = float(trial_row["params_iou_threshold"])
    #    args.neg_iou_threshold = float(trial_row["params_neg_iou_threshold"])
    #    args.neg_pos_ratio = int(trial_row["params_neg_pos_ratio"])
    #    args.nms_threshold = float(trial_row["params_nms_threshold"])
#
    #    # Log parameters for transparency
    #    logging.info("Using params from trial: ")
    #    for col in trial_row.index:
    #        if col.startswith("params_"):
    #            logging.info(f"    {col.replace('params_', '')}: {trial_row[col]}")
#

    # Create dataloaders
    if getattr(args, "all_slices", False):
        collate_fn = list_data_collate
    else:
        collate_fn = detection_collate_fn

    data_loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        collate_fn=collate_fn,
    )
    data_loader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        collate_fn=collate_fn,
    )
    data_loader_test = DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        collate_fn=collate_fn,
    )

    if args.checkpoint_folder is None:
        raise ValueError("Checkpoint folder is required for evaluation")
    else:
        best_model_path = os.path.join(args.checkpoint_folder, os.listdir(args.checkpoint_folder)[0])
        logging.info(f"Loading best model from: {best_model_path}")

    # Train final model with best parameters
    logger = TensorBoardLogger(args.log_dir, name=args.model)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        devices="auto",
        logger=logger,
        accumulate_grad_batches=args.accum_iter,
        num_sanity_val_steps=2,
        log_every_n_steps=10,
        precision="16-mixed",
    )

    # Run predictions on training set
    logging.info("Running predictions on training set...")
    train_model = LesionDetectionModule.load_from_checkpoint(
        best_model_path,
        pred_output_dir=os.path.join(output_dir, "predictions_train"),
        save_to_csv=True,
        csv_filename="train_predictions.csv",
        conf_threshold=0.6,
    )
    train_model.eval()
    trainer.predict(train_model, data_loader_train)
    logging.info(f"Predictions saved to: {os.path.join(output_dir, 'predictions_train')}")

    del train_model

    # Run predictions on validation set
    logging.info("Running predictions on validation set...")
    val_model = LesionDetectionModule.load_from_checkpoint(
        best_model_path,
        pred_output_dir=os.path.join(output_dir, "predictions_val"),
        save_to_csv=True,
        csv_filename="val_predictions.csv",
        conf_threshold=0.6,
    )
    val_model.eval()
    #trainer.validate(val_model, data_loader_val)
    trainer.predict(val_model, data_loader_val)
    logging.info(f"Predictions saved to: {os.path.join(output_dir, 'predictions_val')}")

    del val_model

    # Run predictions on test set
    logging.info("Running predictions on test set...")
    test_model = LesionDetectionModule.load_from_checkpoint(
        best_model_path,
        pred_output_dir=os.path.join(output_dir, "predictions_test"),
        save_to_csv=True,
        csv_filename="test_predictions.csv",
        conf_threshold=0.6,
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
