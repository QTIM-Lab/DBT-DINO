import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import numpy as np
import pandas as pd
import torch
from src.data.transforms import (
    RepeatChanneld,
)
from monai.data import Dataset, PersistentDataset, DataLoader
from torch.utils.data import ConcatDataset
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    Resized,
)

from utils.highdcm_utils import HighdicomMultiframeImageReaderd
from src.utils.train_utils import define_sampler

class EmbeddingDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        task,
        embedding_dir,
        csv_path,
        embeds_to_load: list = ["patch"],
        stats_to_load: list = ["mean"],
        split="train",
        cohort="cancer",
        fraction=1.0,
        num_samples=None,
        breast_specific=False,
        max_followup=5,
        overall_risk=False,
        mask_gap_years=False,
        include_6_months=False,
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


        # Load and filter dataframe
        if task == 'density':
            self.df = pd.read_csv(csv_path)
            self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        elif task == 'risk' or task == 'overall_risk':
            self.df = pd.read_csv(csv_path, dtype=str)
            if include_6_months:
                df_6_months = pd.read_csv(r"6_months_cases.csv", dtype=str)
                df_6_months['ANON_StudyID'] = df_6_months['ANON_StudyID'].apply(lambda x: '6M' + x[3:])
                self.df = pd.concat([self.df, df_6_months], ignore_index=True)
            self.df = self.df[self.df["split"] == split].reset_index(drop=True)

        self.df["patient_id"] = self.df["ANON_SeriesID"].str[:6]

        #if task == 'overall_risk':
        #    self.df['baseline_risk'] = self.df['baseline_risk'].astype(int)

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
                                f"{pid}_{embed}_{stat}.npy",
                                #f"{pid}_{embed}.npy",
                            )
                        )
                        for embed in embeds_to_load # Loop over embedding types
                        for stat in stats_to_load # Loop over statistics
                    )
                )
            ]
            print(f"Embedding directory: {embedding_dir}")
            print(f"Number of patients in {split} set after checking for missing embeddings: {len(self.df)}")
            print(f"Dropped {len_df - len(self.df)} rows from {cohort} {split} due to missing embedding files")

            # Group by patient
            self.df = (
                self.df.groupby("patient_id")
                .agg({"breast_density": "first", "ANON_SeriesID": "first"})
                .reset_index()
            )

        elif task == 'overall_risk':

            # Process labels and IDs
            self.df["baseline_risk"] = (
                self.df["baseline_risk"].astype(int)
            )
            self.df["study_id"] = self.df["ANON_StudyID"]

            # Keep only rows with existing embeddings for all requested statistics
            len_df = len(self.df)
            print(len_df)
            self.df = self.df[
                self.df["study_id"].apply(
                    lambda sid: all(
                        os.path.exists(
                            os.path.join(
                                embedding_dir,
                                "embeddings",
                                f"{sid}_bilateral_{embed}_{stat}.npy",
                            )
                        )
                        for embed in embeds_to_load  # Loop over embedding types
                        for stat in stats_to_load  # Loop over statistics
                    )
                )
            ]
            print(embedding_dir)
            print(len(self.df))
            print(f"Dropped {len_df - len(self.df)} rows from {cohort} {split} due to missing embedding files")

            # Group by study and keep only first entry for the label
            self.df = (
                self.df.groupby("study_id")
                .agg({"baseline_risk": "first", "ANON_StudyID": "first"})
                .reset_index()
            )


        elif task == 'risk':
            self.df = self.df[self.df['cohort'] == cohort].reset_index(drop=True)
            self.df["patient_id"] = self.df["Patient ID"]
            self.df["study_id"] = self.df["ANON_StudyID"]
            self.df["breast_density"] = self.df["breast_density"]


            if not breast_specific:
                # Keep first row entry for each study_id
                self.df = self.df.groupby('study_id').first().reset_index()

            # Keep only rows with existing embeddings for all requested statistics
            len_df = len(self.df)
            self.df = self.df[
                self.df["study_id"].apply(
                    lambda sid: all(
                        os.path.exists(
                            os.path.join(
                                embedding_dir,
                                "embeddings",
                                f"{sid}_bilateral_{embed}_{stat}.npy",
                            )
                        )
                        for embed in embeds_to_load # Loop over embedding types
                        for stat in stats_to_load # Loop over statistics
                    )
                )
            ]


            print(f"Dropped {len_df - len(self.df)} rows due to missing embedding files")

            print(f"Using {cohort} {split} split with {self.df['EMPI'].nunique()} patients and {self.df['ANON_StudyID'].nunique()} studies.")

        print(f"Loading embeddings {', '.join(embeds_to_load)} and statistics {', '.join(stats_to_load)}")


        if fraction < 1.0 and task == 'risk':
            num_patients = max(1, int(fraction * len(self.df)))
            self.df['years_to_last_follow_up'] = self.df['years_to_last_follow_up'].astype(int)
            self.df = self.df[self.df['years_to_last_follow_up'] >= 4]
            self.df = self.df.sample(n=num_patients, random_state=42)
            print(f"Selected {self.df['ANON_StudyID'].nunique()} studies ({fraction:.1%} of total in {cohort} {split} set)")
        
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
        if self.task == 'risk' or self.task == 'overall_risk':
            study_id = row["ANON_StudyID"]

        if self.task == 'density':
            # Load requested statistics
            embeddings = {}
            for stat in self.stats_to_load:
                embedding_path = os.path.join(
                    self.embedding_dir,
                    "embeddings",
                    f"{patient_id}_{self.embeds_to_load[0]}_{stat}.npy"
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

        elif self.task == 'overall_risk':
            # Load requested statistics
            embeddings = {}
            for stat in self.stats_to_load:
                embedding_path = os.path.join(
                    self.embedding_dir,
                    "embeddings",
                    f"{study_id}_bilateral_{self.embeds_to_load[0]}_{self.stats_to_load[0]}.npy"
                )
                embeddings[stat] = torch.from_numpy(np.load(embedding_path))
            # If only one statistic is requested, return it directly
            # Otherwise return a dictionary of statistics
            embedding_data = (
                embeddings[self.stats_to_load[0]]
                if len(self.stats_to_load) == 1
                else embeddings
            )

            return {
                "embedding": embedding_data,
                "label": row["baseline_risk"],
                "study_id": study_id
            }

        elif self.task == 'risk':
            embeddings = {}
            for embed in self.embeds_to_load:
                embeddings[embed] = {}
                for stat in self.stats_to_load:
                    embedding_path = os.path.join(
                        self.embedding_dir,
                        "embeddings",
                        f"{study_id}_bilateral_{embed}_{stat}.npy",
                    )
                    embeddings[embed][stat] = torch.from_numpy(np.load(embedding_path))

            if self.overall_risk:
                return {
                    "embedding": embeddings,
                    "label": row["baseline_risk"],
                    "study_id": study_id
                }

            else:
                return {
                    "embedding": embeddings,
                    "label": row["y_label"],
                    "label_l": row["y_label_l"],
                    "label_r": row["y_label_r"],
                    "mask": row["y_mask"],
                    "mask_l": row["y_mask_l"],
                    "mask_r": row["y_mask_r"],
                    "patient_id": row["patient_id"],
                    "study_id": study_id,
                    "breast_density": row["breast_density"]
                }

def get_datasets(
    csv_path,
    data_dir,
    persistent_cache=True,
    im_size=518,
):
    df = pd.read_csv(
        csv_path, dtype={"ANON_SeriesID": str, "breast_density": str, "split": str}
    )

    # make new column in df that contains the full path to the image, this means data_dir + image_name + .dcm
    df["file_path"] = df["ANON_SeriesID"].apply(
        lambda x: os.path.join(data_dir, f"{x}.dcm")
    )

    # check if the file path exists, if not drop the row
    len_df = len(df)
    df = df[df["file_path"].apply(lambda x: os.path.exists(x))]
    print(f"Dropped {len_df - len(df)} rows because their file path does not exist")

    # Sanity Check: Drop rows where the report description does not contain the word "screen"
    len_df = len(df)
    df = df[df["Report_Description"].str.contains("screen", case=False, na=False)]
    print(
        f"Dropped {len_df - len(df)} rows because they do not contain the word 'screen' in the report description"
    )

    # HOTFIX: Drop broken patients: P_2175 and P_4285
    len_df = len(df)
    df = df[df["ANON_PatientID"] != "P_2175"]
    df = df[df["ANON_PatientID"] != "P_4285"]
    print(f"Dropped {len_df - len(df)} rows for patients P_2175 and P_4285 (broken data)")

    # Replace the letters a, b, c, d with 0, 1, 2, 3
    df["breast_density"] = df["breast_density"].replace(
        {"a": 0, "b": 1, "c": 2, "d": 3}
    )

    print("Dataset Classes:")
    print(df["breast_density"].value_counts().sort_index())

    # Define views we want
    VIEWS = ["LCC", "RCC", "LMLO", "RMLO"]

    # Create lists for train and val data
    train_data = []
    val_data = []
    test_data = []

    # Group by patient ID
    for patient_id, patient_df in df.groupby("ANON_PatientID"):
        # Create dict with views and density initialized to None
        patient_dict = {view: None for view in VIEWS}
        patient_dict["breast_density"] = None

        # Fill in data from patient's rows
        split = None
        for _, row in patient_df.iterrows():
            view = row["View"]
            if view in VIEWS:
                patient_dict[view] = row["file_path"]
                patient_dict["breast_density"] = row["breast_density"]
                # HOTFIX: We are currently saving the ANON_SeriesID in the dataset, but it should be the ANON_PatientID
                # This is currently still not fixed so that the persistent dataset does not have to be reconstructed
                patient_dict["ANON_SeriesID"] = row["ANON_SeriesID"]
                split = row["split"]

        # Add to appropriate list if all views present
        if all(patient_dict[view] for view in VIEWS):
            if split == "train":
                train_data.append(patient_dict)
            elif split == "val":
                val_data.append(patient_dict)
            elif split == "test":
                test_data.append(patient_dict)

    print(
        f"Found {len(train_data)} training patients and {len(val_data)} validation patients and {len(test_data)} test patients with all {len(VIEWS)} views"
    )

    transforms = [
        HighdicomMultiframeImageReaderd(is_dbt=True, keys=VIEWS),
        EnsureChannelFirstd(keys=VIEWS),
        Resized(
            spatial_size=(-1, im_size, im_size), keys=VIEWS
        ),
        RepeatChanneld(keys=VIEWS),
    ]


    transforms = Compose(transforms)

    if persistent_cache:
        tmpdir = os.getenv('TMPDIR')
        print(f"Creating Persistent Dataset at: {tmpdir}")
        train_ds = PersistentDataset(
            data=train_data, transform=transforms, cache_dir=tmpdir
        )

        val_ds = PersistentDataset(
            data=val_data, transform=transforms, cache_dir=tmpdir
        )

        test_ds = PersistentDataset(
            data=test_data, transform=transforms, cache_dir=tmpdir
        )

    else:
        train_ds = Dataset(data=train_data, transform=transforms)

        val_ds = Dataset(data=val_data, transform=transforms)

        test_ds = Dataset(data=test_data, transform=transforms)

    return train_ds, val_ds, test_ds



def get_datasets_risk(train_cancer_embed_dataset, train_healthy_embed_dataset, val_cancer_embed_dataset,
                      val_healthy_embed_dataset, test_cancer_embed_dataset, test_healthy_embed_dataset,
                      args):
    """
    Get train, val and test datasets using sampler to compensate for class imbalance between cancer and healthy cases.
    """

    # Combine datasets
    train_dataset = ConcatDataset([train_cancer_embed_dataset, train_healthy_embed_dataset])
    val_dataset = ConcatDataset([val_cancer_embed_dataset, val_healthy_embed_dataset])
    test_dataset = ConcatDataset([test_cancer_embed_dataset, test_healthy_embed_dataset])

    # Define number of samples as the length of the healthy training dataset
    num_samples_train = len(train_healthy_embed_dataset)
    #num_samples_train = len(train_cancer_embed_dataset)*2

    # Create Weighted Sampler for training set
    train_sampler = define_sampler(train_cancer_embed_dataset, train_healthy_embed_dataset, num_samples_train)

    # Define fixed sample for validation set
    indices = torch.randperm(len(val_healthy_embed_dataset))[:len(val_cancer_embed_dataset)]
    #healthy_val_subset = Subset(val_healthy_embed_dataset, indices)
    #val_subset = ConcatDataset([val_cancer_embed_dataset, healthy_val_subset])

    frac = len(val_cancer_embed_dataset) / len(val_healthy_embed_dataset)

    healthy_val_subset = EmbeddingDataset(
        task=args.task,
        embedding_dir=os.path.join(args.embedding_dir_healthy, args.model, "val"),
        csv_path=args.csv_path,
        split="val",
        cohort="healthy",
        embeds_to_load=args.embedding_type,
        stats_to_load=args.stats,
        breast_specific=args.breast_specific,
        overall_risk=args.task == 'overall_risk',
        fraction=frac,
    )

    val_subset = ConcatDataset([val_cancer_embed_dataset, healthy_val_subset])

    # Shuffle indices
    shuffled_idx = torch.randperm(len(val_subset)).tolist()

    val_subset = Dataset(data=[val_subset[i] for i in shuffled_idx])

    print(f"Training dataset positive cases: {len(train_cancer_embed_dataset)} samples")
    print(f"Training dataset negative cases: {len(train_healthy_embed_dataset)} samples")
    print(f"Size of training set after random weighted sampling: {num_samples_train} samples")
    print(f"Validation dataset positive cases: {len(val_cancer_embed_dataset)} samples")
    print(f"Validation dataset negative cases: {len(healthy_val_subset)} samples")
    print(f"Size of validation set (balanced): {len(val_subset)} samples")
    print(f"Test dataset: {len(test_dataset)} samples")

    # Training DataLoader with sampler
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    print(f"Training DataLoader: {len(train_data_loader)} batches")

    # Validation DataLoader non-balanced
    val_data_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_data_loader_balanced = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"Balanced validation dataloader: {len(val_data_loader_balanced)} batches")

    print(f"Imbalanced Validation DataLoader: {len(val_data_loader)} batches")

    # Test DataLoader
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
    )

    print(f"Test DataLoader: {len(test_data_loader)} batches")

    return train_data_loader, val_data_loader, val_data_loader_balanced, test_data_loader

def get_datasets_risk_inference(test_cancer_embed_dataset, test_healthy_embed_dataset, args):
    """
    Get test datasets for inference.
    """

    # Combine datasets
    test_dataset = ConcatDataset([test_cancer_embed_dataset, test_healthy_embed_dataset])

    # Test DataLoader
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
    )

    print(f"Test DataLoader: {len(test_data_loader)} batches")

    return test_data_loader
