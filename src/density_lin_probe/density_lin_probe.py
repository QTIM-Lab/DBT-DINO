import logging
import os
from pathlib import Path

import optuna
import pytorch_lightning as pl
import torch
from monai.data import DataLoader
from pycrumbs import tracked
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from data import EmbeddingDataset, get_datasets
from models.baseline_feature_model import DenseNet121
from models.linear_probing_model import LinearProbingModel
from utils import (
    evaluate_model,
    extract_and_save_embeddings,
    get_args_parser,
    objective,
    setup_logging,
)


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

    # Check if embeddings already exist
    train_embedding_dir = os.path.join(args.embedding_dir, "train", "embeddings")
    val_embedding_dir = os.path.join(args.embedding_dir, "val", "embeddings")
    test_embedding_dir = os.path.join(args.embedding_dir, "test", "embeddings")

    embeddings_exist = (
            os.path.exists(train_embedding_dir)
            and os.path.exists(val_embedding_dir)
            and os.path.exists(test_embedding_dir)
            and len(os.listdir(train_embedding_dir)) > 0
            and len(os.listdir(val_embedding_dir)) > 0
            and len(os.listdir(test_embedding_dir)) > 0
    )

    if not embeddings_exist:
        logging.info("No existing embeddings found. Extracting embeddings...")

        # Load datasets
        if args.model == "vit_baseline" or args.model == "densenet_121":
            dataset_train, dataset_val, dataset_test = get_datasets(
                args.csv_path, args.data_dir, persistent_cache=False, im_size=224
            )
        else:
            dataset_train, dataset_val, dataset_test = get_datasets(
                args.csv_path, args.data_dir, persistent_cache=True 
            )

        # Create dataloaders
        data_loader_train = DataLoader(
            dataset_train,
            batch_size=1,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )
        data_loader_val = DataLoader(
            dataset_val,
            batch_size=1,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )
        data_loader_test = DataLoader(
            dataset_test,
            batch_size=1,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )

        if args.model == "dino_dbt":
            # Load DINO model
            dino_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")

            if args.backbone_checkpoint is not None:
                state_dict = torch.load(
                    args.backbone_checkpoint, map_location=torch.device("cpu")
                )

                # Remove the "backbone." prefix from all keys to match expected structure
                new_state_dict = {}
                for k, v in state_dict["teacher"].items():
                    new_key = k.replace("backbone.", "", 1)
                    new_state_dict[new_key] = v

                load_result = dino_model.load_state_dict(new_state_dict, strict=False)
                logging.info(
                    f"Loaded Backbone DINO State dict from {args.backbone_checkpoint} with result: {load_result}"
                )

            else:
                logging.info(
                    "Loaded the baseline facebook DINO model: facebookresearch/dinov2 (dinov2_vitb14)"
                )
            embedding_model = dino_model

        elif args.model == "densenet_121":
            embedding_model = DenseNet121()
            logging.info(f"Loaded the baseline DenseNet model")

        else:
            raise ValueError(f"Unsupported model: {args.model}. Choose from 'dino_dbt', 'vit_baseline' or 'densenet_121'.")
            
        # Extract and save embeddings
        logging.info("Extracting and saving embeddings...")
        extract_and_save_embeddings(
            embedding_model,
            data_loader_train,
            os.path.join(args.embedding_dir, "train"),
            args.model,
        )
        extract_and_save_embeddings(
            embedding_model,
            data_loader_val,
            os.path.join(args.embedding_dir, "val"),
            args.model,
        )
        extract_and_save_embeddings(
            embedding_model,
            data_loader_test,
            os.path.join(args.embedding_dir, "test"),
            args.model,
        )
    else:
        logging.info("Found existing embeddings. Skipping extraction step...")
        print("Found existing embeddings. Skipping extraction step...")

    # Create embedding datasets and dataloaders
    train_embedding_dataset = EmbeddingDataset(
        task=args.task,
        embedding_dir=os.path.join(args.embedding_dir, "train"),
        csv_path=args.csv_path,
        split="train",
        embeds_to_load=args.embedding_type,
        stats_to_load=args.stats,
        fraction=args.dataset_fraction,
    )
    val_embedding_dataset = EmbeddingDataset(
        task=args.task,
        embedding_dir=os.path.join(args.embedding_dir, "val"),
        csv_path=args.csv_path,
        split="val",
        embeds_to_load=args.embedding_type,
        stats_to_load=args.stats,
    )

    if args.use_optuna:
        # Create Optuna study with persistent storage
        storage_path = os.path.join(output_dir, "optuna.db")
        storage = optuna.storages.RDBStorage(
            url=f"sqlite:///{storage_path}",
            engine_kwargs={"connect_args": {"timeout": 30}},
        )

        study = optuna.create_study(
            study_name=args.study_name,
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5, n_warmup_steps=15, interval_steps=3
            ),
            storage=storage,  # Use the RDBStorage instance
            load_if_exists=True,
        )

        # Run optimization
        study.optimize(
            lambda trial: objective(
                trial,
                args,
                train_embedding_dataset,
                val_embedding_dataset,
                LinearProbingModel,
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

    # Create dataloaders with final parameters
    train_embedding_loader = DataLoader(
        train_embedding_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
    )
    val_embedding_loader = DataLoader(
        val_embedding_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
    )

    # Train final model with best parameters
    model = LinearProbingModel(args)

    logger = TensorBoardLogger(args.log_dir, name=args.model)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=3,
        mode="min",
        filename="{epoch:02d}-{val_loss:.2f}",
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        # accelerator="gpu",
        devices="auto",
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        accumulate_grad_batches=args.accum_iter,
        num_sanity_val_steps=2,
        log_every_n_steps=10,
    )

    trainer.fit(model, train_embedding_loader, val_embedding_loader)

    # Load the best checkpoint based on validation loss
    best_model_path = checkpoint_callback.best_model_path
    logging.info(f"Loading best model from: {best_model_path}")
    model = LinearProbingModel.load_from_checkpoint(best_model_path, args=args)

    # Create test dataloader
    test_embedding_dataset = EmbeddingDataset(
        embedding_dir=os.path.join(args.embedding_dir, "test"),
        csv_path=args.csv_path,
        split="test",
        embeds_to_load=args.embedding_type,
        stats_to_load=args.stats,
        task=args.task
    )

    test_embedding_loader = DataLoader(
        test_embedding_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    # Evaluate model and save metrics
    evaluate_model(trainer, model, test_embedding_loader, output_dir)


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = "./model_output"
    main(args, Path(output_dir))
