import logging
import os
from pathlib import Path

import optuna
import pytorch_lightning as pl
import torch
from pycrumbs import tracked
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from src.data import EmbeddingDataset
from src.data.linear_probing_data import get_datasets_risk, get_datasets_risk_inference
from src.models.linear_probing_model import Cumulative_Risk_Model, LinearProbingModel
from src.utils import (
    get_args_parser,
    risk_objective,
    setup_logging,
)


@tracked(directory_parameter="output_dir", include_timestamp=True)
def main(args, output_dir):
    args.output_dir = output_dir
    if args.log_dir is None:
        args.log_dir = output_dir

    # Set up logging
    setup_logging(output_dir, "dino_linear_probing")

    # generate random seed
    if args.seed is None:
        args.seed = torch.randint(0, 2**32 - 1, (1,)).item()
    # Set random seed and save it to the log file
    logging.info(f"Random seed: {args.seed}")

    pl.seed_everything(args.seed)
    torch.set_float32_matmul_precision("medium") # needed?

    print(f"Running in {args.mode} mode with task '{args.task}'.")

    if args.task == 'risk' or args.task == 'overall_risk':
        train_embedding_dir_cancer = os.path.join(args.embedding_dir_cancer, "train", "embeddings")
        val_embedding_dir_cancer = os.path.join(args.embedding_dir_cancer, "val", "embeddings")
        train_embedding_dir_healthy = os.path.join(args.embedding_dir_healthy, "train", "embeddings")
        val_embedding_dir_healthy = os.path.join(args.embedding_dir_healthy, "val", "embeddings")
        embeddings_exist = (
                os.path.exists(train_embedding_dir_cancer)
                and os.path.exists(val_embedding_dir_cancer)
                and os.path.exists(train_embedding_dir_healthy)
                and os.path.exists(val_embedding_dir_healthy)
                and len(os.listdir(train_embedding_dir_cancer)) > 0
                and len(os.listdir(val_embedding_dir_cancer)) > 0
                and len(os.listdir(train_embedding_dir_healthy)) > 0
                and len(os.listdir(val_embedding_dir_healthy)) > 0
        )
        embeddings_exist = True

    if not embeddings_exist:
        raise ValueError("Embeddings do not exist. Please extract embeddings first for risk task.")
    else:
        logging.info("Found existing embeddings. Continuing with training...")

    if args.task == 'risk' and args.mode == 'train':
        print("Loading datasets for risk prediction task...")
        train_cancer_embed_dataset = EmbeddingDataset(
            task=args.task,
            embedding_dir=os.path.join(args.embedding_dir_cancer, args.model, "train"),
            csv_path=args.csv_path,
            split="train",
            cohort="cancer",
            embeds_to_load=args.embedding_type,
            stats_to_load=args.stats,
            breast_specific=args.breast_specific,
            overall_risk=args.task == 'overall_risk',
            include_6_months=args.include_6_months,
        )

        train_healthy_embed_dataset = EmbeddingDataset(
            task=args.task,
            embedding_dir=os.path.join(args.embedding_dir_healthy, args.model, "train"),
            csv_path=args.csv_path,
            split="train",
            cohort="healthy",
            embeds_to_load=args.embedding_type,
            stats_to_load=args.stats,
            breast_specific=args.breast_specific,
            overall_risk=args.task == 'overall_risk',
        )

        val_cancer_embed_dataset = EmbeddingDataset(
            task=args.task,
            embedding_dir=os.path.join(args.embedding_dir_cancer, args.model, "val"),
            csv_path=args.csv_path,
            split="val",
            cohort="cancer",
            embeds_to_load=args.embedding_type,
            stats_to_load=args.stats,
            breast_specific=args.breast_specific,
            overall_risk=args.task == 'overall_risk',
        )

        val_healthy_embed_dataset = EmbeddingDataset(
            task=args.task,
            embedding_dir=os.path.join(args.embedding_dir_healthy, args.model, "val"),
            csv_path=args.csv_path,
            split="val",
            cohort="healthy",
            embeds_to_load=args.embedding_type,
            stats_to_load=args.stats,
            breast_specific=args.breast_specific,
            overall_risk=args.task == 'overall_risk',
        )

        test_cancer_embed_dataset = EmbeddingDataset(
            task=args.task,
            embedding_dir=os.path.join(args.embedding_dir_cancer, args.model, "test"),
            csv_path=args.csv_path,
            split='test',
            cohort='cancer',
            embeds_to_load=args.embedding_type,
            stats_to_load=args.stats,
            breast_specific=args.breast_specific,
            overall_risk=args.task == 'overall_risk',
        )

        test_healthy_embed_dataset = EmbeddingDataset(
            task=args.task,
            embedding_dir=os.path.join(args.embedding_dir_healthy, args.model, "test"),
            csv_path=args.csv_path,
            split='test',
            cohort='healthy',
            embeds_to_load=args.embedding_type,
            stats_to_load=args.stats,
            breast_specific=args.breast_specific,
            overall_risk=args.task == 'overall_risk',
        )

        # print lengths of datasets
        print(f"Length of train_cancer_embed_dataset: {len(train_cancer_embed_dataset)}")
        print(f"Length of train_healthy_embed_dataset: {len(train_healthy_embed_dataset)}")
        print(f"Length of val_cancer_embed_dataset: {len(val_cancer_embed_dataset)}")
        print(f"Length of val_healthy_embed_dataset: {len(val_healthy_embed_dataset)}")
        print(f"Length of test_cancer_embed_dataset: {len(test_cancer_embed_dataset)}")
        print(f"Length of test_healthy_embed_dataset: {len(test_healthy_embed_dataset)}")

    elif args.task == 'risk' and args.mode == 'test':
        test_cancer_embed_dataset = EmbeddingDataset(
            task=args.task,
            embedding_dir=os.path.join(args.embedding_dir_cancer, args.model, "test"),
            csv_path=args.csv_path,
            split='test',
            cohort='cancer',
            embeds_to_load=args.embedding_type,
            stats_to_load=args.stats,
            breast_specific=args.breast_specific,
            overall_risk=args.task == 'overall_risk',
        )

        test_healthy_embed_dataset = EmbeddingDataset(
            task=args.task,
            embedding_dir=os.path.join(args.embedding_dir_healthy, args.model, "test"),
            csv_path=args.csv_path,
            split='test',
            cohort='healthy',
            embeds_to_load=args.embedding_type,
            stats_to_load=args.stats,
            breast_specific=args.breast_specific,
            overall_risk=args.task == 'overall_risk',
        )

        print(f"Length of test_cancer_embed_dataset: {len(test_cancer_embed_dataset)}")
        print(f"Length of test_healthy_embed_dataset: {len(test_healthy_embed_dataset)}")

        test_embedding_loader = get_datasets_risk_inference(test_cancer_embed_dataset, test_healthy_embed_dataset, args)

    if args.task == 'risk' and args.mode == 'train':
        # Create dataloaders with sampler
        train_embedding_loader, val_embedding_loader, val_embedding_loader_balanced, test_embedding_loader = get_datasets_risk(
            train_cancer_embed_dataset,
            train_healthy_embed_dataset,
            val_cancer_embed_dataset,
            val_healthy_embed_dataset,
            test_cancer_embed_dataset,
            test_healthy_embed_dataset,
            args
        )

        combined_val_loader = [val_embedding_loader, val_embedding_loader_balanced]

        if args.use_optuna:
            # Create Optuna study with persistent storage
            if args.db_folder is not None:
                storage_path = os.path.join(args.db_folder, "optuna.db")
            else:
                storage_path = os.path.join(output_dir, "optuna.db")
            storage = optuna.storages.RDBStorage(
                url=f"sqlite:///{storage_path}",
                engine_kwargs={"connect_args": {"timeout": 30}},
            )

            study = optuna.create_study(
                study_name=args.study_name,
                direction="minimize",
                pruner=optuna.pruners.MedianPruner(
                    n_startup_trials=5, n_warmup_steps=5, interval_steps=3
                ),
                storage=storage,  # Use the RDBStorage instance
                load_if_exists=True,
            )

            # Run optimization
            study.optimize(
                lambda trial: risk_objective(
                    trial,
                    args,
                    train_embedding_loader, # change to balanced training dataset?
                    val_embedding_loader_balanced,
                    Cumulative_Risk_Model
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


        if args.task == 'risk':
            # Define linear probing model to train
            model = Cumulative_Risk_Model(embed_type=args.embedding_type,
                                          stats=args.stats,
                                          args=args,
                                          max_followup=args.max_followup,
                                          )

        elif args.task == 'overall_risk':
            model = LinearProbingModel(args)


    if args.task == 'risk':

        logger = TensorBoardLogger(args.log_dir, name=args.model)
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            save_top_k=3,
            mode="min",
            filename="{epoch:02d}-{val_loss:.2f}",
        )

        lr_monitor = LearningRateMonitor(logging_interval="step")

        checkpoint_avg_auroc = ModelCheckpoint(
            monitor="val_avg_auroc",
            save_top_k=3,
            mode="max",
            filename="{epoch:02d}-{val_avg_auroc:.2f}",
        )

        # Initialize trainer with specific device settings
        trainer = pl.Trainer(
            max_epochs=args.epochs,  # infinite number of epochs, stop once val loss converges
            accelerator="gpu",
            devices=1,
            logger=logger,
            callbacks=[checkpoint_callback, checkpoint_avg_auroc, lr_monitor],
            accumulate_grad_batches=args.accum_iter,
            num_sanity_val_steps=-1,
            log_every_n_steps=10,
            enable_progress_bar=True,
        )

        if args.mode == 'train':
            if args.validation == 'balanced':
                trainer.fit(model, train_embedding_loader, val_embedding_loader_balanced)
            elif args.validation == 'imbalanced':
                trainer.fit(model, train_embedding_loader, val_embedding_loader)
            elif args.validation == 'combined':
                trainer.fit(model, train_embedding_loader, combined_val_loader)
            # Load the best model from the checkpoint
            best_model = Cumulative_Risk_Model.load_from_checkpoint(
                checkpoint_callback.best_model_path,
                map_location="cuda",  # or "cpu" if you want to load on CPU
                embed_type=args.embedding_type,
                stats=args.stats,
                args=args,
                max_followup=args.max_followup
            )
            trainer.test(best_model, dataloaders=test_embedding_loader)

        elif args.mode == 'test':

            model_path = args.checkpoint_path if args.checkpoint_path else checkpoint_callback.best_model_path
            # Load the best model from the checkpoint
            best_model = Cumulative_Risk_Model.load_from_checkpoint(
                #checkpoint_callback.best_model_path,
                checkpoint_path=model_path,
                map_location="cuda",  # or "cpu" if you want to load on CPU
                embed_type=args.embedding_type,
                stats=args.stats,
                args=args,
                max_followup=args.max_followup
            )
            trainer.test(best_model, dataloaders=test_embedding_loader)

        #best_model = Cumulative_Risk_Model.load_from_checkpoint(checkpoint_callback.best_model_path)

        #evaluate_model(trainer, best_model, test_embedding_loader, output_dir)


    elif args.task == 'overall_risk':
        # Initialize trainer with specific device settings
        trainer = pl.Trainer(
            max_epochs=args.epochs,
            #accelerator="gpu",
            devices="auto",
            logger=logger,
            callbacks=[checkpoint_callback, lr_monitor],
            accumulate_grad_batches=args.accum_iter,
            num_sanity_val_steps=2,
            log_every_n_steps=10,
            precision="16-mixed",
        )

        if args.validation == 'balanced':
            trainer.fit(model, train_embedding_loader, val_embedding_loader_balanced)
        elif args.validation == 'imbalanced':
            trainer.fit(model, train_embedding_loader, val_embedding_loader)
        elif args.validation == 'combined':
            trainer.fit(model, train_embedding_loader, combined_val_loader)


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = "./model_output"
    main(args, Path(output_dir))
