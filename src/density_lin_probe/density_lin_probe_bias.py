import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from monai.data import DataLoader
from pycrumbs import tracked
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import ConcatDataset


from models.linear_probing_model import LinearProbingModel
from utils import (
    evaluate_model,
    get_args_parser,
    setup_logging,
)


class EmbeddingDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        task,
        embedding_dir,
        csv_path,
        embeds_to_load: list = ["patch"],
        stats_to_load: list = ["mean"],
        split="train",
        cohort="healthy",
        fraction=1.0,
        num_samples=None,
        breast_specific=False,
        max_followup=5,
        overall_risk=False,
        mask_gap_years=False,
    ):

        if not all(embed in ['patch', 'cls'] for embed in embeds_to_load):
            raise ValueError("embedding_type must be a list containing 'patch' and/or 'cls'")
        
        valid_stats = ["mean", "std", "min", "max"]
        if not all(stat in valid_stats for stat in stats_to_load):
            raise ValueError(
                f"stats_to_load must be a list containing any of: {', '.join(valid_stats)}"
            )

        self.embedding_dir = embedding_dir
        self.embeds_to_load = embeds_to_load
        self.stats_to_load = stats_to_load

        self.task = task
        self.max_followup = max_followup
        self.overall_risk = overall_risk
        self.mask_gap_years = mask_gap_years



        self.df = pd.read_csv(csv_path, dtype=str)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)

        self.df["patient_id"] = self.df["ANON_StudyID"].str.split("_").str[0]


        if task == 'density':
            # Process labels and IDs
            self.df["breast_density"] = (
                self.df["breast_density"].map({"a": 0, "b": 1, "c": 2, "d": 3}).astype(int)
            )

            # Keep only rows with existing embeddings for all requested statistics
            len_df = len(self.df)
            self.df = self.df[
                self.df["patient_id"].apply(
                    lambda pid: all(
                        os.path.exists(
                            os.path.join(
                                embedding_dir,
                                "embeddings",
                                f"{pid}_bilateral_{embed}_{stat}.npy",
                                #f"{pid}_{embed}.npy",
                            )
                        )
                        for embed in embeds_to_load # Loop over embedding types
                        for stat in stats_to_load # Loop over statistics
                    )
                )
            ]
            print(f"Embedding directory: {embedding_dir}")
            print(f"Number of studies in {split} set after checking for missing embeddings: {self.df['patient_id'].nunique()}")
            print(f"Dropped {len_df - len(self.df)} rows from {cohort} {split} due to missing embedding files")

            # Group by patient
            self.df = (
                self.df.groupby("patient_id")
                .agg({"breast_density": "first", "ANON_SeriesID": "first"})
                .reset_index()
            )


            #print(f"Dropped {len_df - len(self.df)} rows due to missing embedding files")

            #print(f"Using {cohort} {split} split with {self.df['patient_id'].nunique()} patients and {self.df['ANON_StudyID'].nunique()} studies.")
        else:
            raise ValueError(f"Task {task} not supported")
        
        print(f"Loading embeddings {', '.join(embeds_to_load)} and statistics {', '.join(stats_to_load)}")


        if fraction < 1.0 and task == 'risk':
            num_patients = max(1, int(fraction * len(self.df)))
            self.df['years_to_last_follow_up'] = self.df['years_to_last_follow_up'].astype(int)
            self.df = self.df[self.df['years_to_last_follow_up'] >= 4]
            self.df = self.df.sample(n=num_patients, random_state=42)
            print(f"Selected {len(self.df)} studies ({fraction:.1%} of total in {cohort} {split} set)")
        
        elif fraction < 1.0 and task == 'density':
            num_patients = max(1, int(fraction * len(self.df)))
            self.df = self.df.sample(n=num_patients, random_state=42)
            print(f"Selected {len(self.df)} patients ({fraction:.1%} of total in {cohort} {split} set)")

        if num_samples is not None:
            self.df = self.df.sample(n=num_samples, random_state=42)
            print(f"Selected {len(self.df)} patients ({num_samples} of total in {cohort} {split} set)")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        if self.task == 'density':
            patient_id = row["patient_id"]

        if self.task == 'density':
            # Load requested statistics
            embeddings = {}
            for stat in self.stats_to_load:
                embedding_path = os.path.join(
                    self.embedding_dir,
                    "embeddings",
                    f"{patient_id}_bilateral_{self.embeds_to_load[0]}_{stat}.npy"
                )
                embeddings[stat] = torch.from_numpy(np.load(embedding_path))
                
            # If only one statistic is requested, return it directly
            # Otherwise concatenate embeddings vertically
            embedding_data = (
                embeddings[self.stats_to_load[0]]
                if len(self.stats_to_load) == 1
                else torch.cat([embeddings[stat] for stat in self.stats_to_load], dim=0)
            )

            return {
                "embedding": embedding_data,
                "label": torch.tensor(row["breast_density"]),
                "patient_id": patient_id
            }
        
        else:
            raise ValueError(f"Task {self.task} not supported")

            
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

    embeddings_exist = (
            os.path.exists(train_embedding_dir)
            and os.path.exists(val_embedding_dir)
            and len(os.listdir(train_embedding_dir)) > 0
            and len(os.listdir(val_embedding_dir)) > 0
    )

    if not embeddings_exist:
        raise ValueError("No existing embeddings found. This script is not meant to be run without embeddings.")


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
        fraction=args.dataset_fraction,
    )

    test_embedding_dataset = EmbeddingDataset(
        embedding_dir=os.path.join(args.embedding_dir, "test"),
        csv_path=args.csv_path,
        split="test",
        embeds_to_load=args.embedding_type,
        stats_to_load=args.stats,
        task=args.task,
        fraction=args.dataset_fraction,
    )

    combined_dataset = ConcatDataset([train_embedding_dataset, val_embedding_dataset, test_embedding_dataset])

    test_embedding_loader = DataLoader(
        combined_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
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
        devices="auto",
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        accumulate_grad_batches=args.accum_iter,
        num_sanity_val_steps=2,
        log_every_n_steps=10,
    )

    # Load the best checkpoint based on validation loss
    best_model_path = args.model_path #checkpoint_callback.best_model_path
    logging.info(f"Loading best model from: {best_model_path}")
    model = LinearProbingModel.load_from_checkpoint(best_model_path, args=args)

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
