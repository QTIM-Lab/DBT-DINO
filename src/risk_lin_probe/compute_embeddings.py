"""
INFO: use pyenv breast_cancer_dbt_manon to run this script.
"""


import argparse
import logging
import os
import sys
from typing import Hashable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from dcmtrain.dataset import StreamingDataset
from dcmtrain.identifier import RPSDICOMSeriesIdentifier
from dcmtrain.image_reader import RPSDICOMWebSeriesReader
from monai.data import Dataset, MetaTensor
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    Flip,
    LoadImaged,
    MapTransform,
    Orientationd,
    Resized,
)
from rps_client.client import RPSClient
from torch.utils.data import DataLoader
from tqdm import tqdm

#from transformers import AutoModel
from utils.highdcm_utils import HighdicomMultiframeImageReaderd


def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "save_embeddings.log")

    # Configure logging to write to both file and console
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    logging.info(f"Logging to {log_file}")


class CustomOrientationd(MapTransform):
    def __init__(self, keys: Sequence[str], allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.flipper = Flip(spatial_axis=1)  # Flip along the horizontal axis

    def __call__(
        self, data: Mapping[Hashable, torch.Tensor]
    ) -> dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            img = d[key]
            if "00200020" in img.meta and "F" in img.meta["00200020"]["Value"][1]:
                # Flip the image horizontally
                img = self.flipper(img)
            d[key] = img
        return d


class FlipCCd(MapTransform):
    def __init__(self, keys: Sequence[str], allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)

    def __call__(
            self, data: Mapping[Hashable, torch.Tensor]
    ) -> dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            img = d[key]
            if img.meta['patient_orientation'].value[0] == 'A' and img.meta['patient_orientation'].value[1] == 'R':
                # Flip the image horizontally
                print(img.meta['patient_orientation'].value)
                img = torch.flip(img, dims=[3])
            d[key] = img
        return d


class CustomWindowd(MapTransform):
    def __init__(self, keys: Sequence[str], allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)

    def _get_window_center(self, img) -> torch.Tensor:
        return torch.tensor(
            img.meta["52009229"]["Value"][0]["00289132"]["Value"][0]["00281050"][
                "Value"
            ][0],
            dtype=torch.float32,
        )

    def _get_window_width(self, img) -> torch.Tensor:
        return torch.tensor(
            img.meta["52009229"]["Value"][0]["00289132"]["Value"][0]["00281051"][
                "Value"
            ][0],
            dtype=torch.float32,
        )

    def rescale_window(self, img, center, width):
        window_min = center - (width / 2)
        window_max = center + (width / 2)

        # Apply windowing
        windowed_image = torch.clamp(img, window_min, window_max)

        # Rescale between 0 and 1
        windowed_image = (windowed_image - window_min) / (window_max - window_min)

        # Clip values outside the range [0, 1] (just in case)
        windowed_image = torch.clamp(windowed_image, 0, 1)

        return windowed_image

    def __call__(
        self, data: Mapping[Hashable, torch.Tensor]
    ) -> dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            img = d[key]
            try:
                center = self._get_window_center(img)
                width = self._get_window_width(img)
            except Exception as e:
                print(f"Error in windowing: {e}")
                center = 512
                width = 512
                print(f"Setting center and width to {center} and {width}")

            d[key] = self.rescale_window(img, center, width)
        return d


class RepeatChanneld(MapTransform):
    def __init__(self, keys: Sequence[str], allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)

    def __call__(
        self, data: Mapping[Hashable, torch.Tensor]
    ) -> dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            img = d[key]
            # Expand the single channel to create a pseudo RGB image
            if len(img.shape) == 3 and img.shape[0] == 1:
                img = img.expand(3, -1, -1)
            elif len(img.shape) == 4 and img.shape[0] == 1:
                img = img.expand(3, -1, -1, -1)
            else:
                raise ValueError(f"Unexpected shape: {img.shape}")
            d[key] = img
        return d


class CustomTransformDataset(Dataset):
    def __getitem__(self, index):
        data = super().__getitem__(index)
        identifier, transform = data['img']
        # Apply the specified transform to the image using the identifier
        data['image'] = transform({'img': identifier})['img']
        return data


def get_class_weights(df_train, label_col='birads_label', enable_weights=True):
    label_counts = df_train[label_col].value_counts().sort_index()

    # Total number of samples
    total_samples = len(df_train)
    # Number of classes
    num_classes = len(label_counts)

    if enable_weights:
        # Calculate class weights
        class_weights_dict = {label: total_samples / (num_classes * count) for label, count in label_counts.items()}
        # Convert to tensor and sort by label index
        class_weights = torch.tensor([class_weights_dict[label] for label in sorted(label_counts.index)],
                                     dtype=torch.float32)
        print(f"Using class weights: {class_weights_dict}")
    else:
        # Equal weights for all classes
        class_weights = torch.ones(num_classes, dtype=torch.float32)
        print("Using equal class weights.")

    return class_weights

def get_datasets_streaming(
        csv_path,
        cohort,
        token_0,
        token_1,
        output_dir,
        num_workers,
        all_views_bilat=False,
        all_views_unilat=False
):

    df = pd.read_csv(csv_path, dtype=str)

    # Convert all entries in Series Instance UID to str
    df['Series Instance UID'] = df['Series Instance UID'].astype(str)

    if cohort == 'cancer':
        df = df[df['cohort'] == cohort]
        token = token_1
    elif cohort == 'healthy':
        df = df[df['cohort'] == cohort]
        token = token_0
    else:
        raise ValueError("Cohort must be either 'cancer' or 'healthy'.")

    # HOTFIX: Remove studies that are already in the output directory from the df at this point
    if os.path.exists(output_dir):
        import glob
        existing_study_ids = set()
        
        logging.info("Scanning for existing embeddings...")
        # Use glob for faster recursive file search
        npy_files = glob.glob(os.path.join(output_dir, '**', '*.npy'), recursive=True)
        
        for file_path in npy_files:
            # Get just the filename without path
            filename = os.path.basename(file_path)
            # Get the study ID from the file name (first part before first underscore)
            study_id = filename.split('_')[0]
            existing_study_ids.add(study_id)
        
        # Remove rows from df where ANON_StudyID exists in the output directory
        if existing_study_ids:
            initial_count = len(df)
            df = df[~df['ANON_StudyID'].isin(existing_study_ids)]
            removed_count = initial_count - len(df)
            logging.info(f"Found {len(npy_files)} .npy files with {len(existing_study_ids)} unique study IDs")
            logging.info(f"Removed {removed_count} rows with existing embeddings. Remaining rows: {len(df)}")
        else:
            logging.info("No existing embeddings found in output directory")
    # Initialize RPS clients for different cohorts
    client = RPSClient(token=token)

    # Create readers for different clients
    reader = RPSDICOMWebSeriesReader(client=client)

    # Create lists for train and val data
    train_data = []
    val_data = []
    test_data = []

    if all_views_bilat:

        views = ['RCC', 'LCC', 'RMLO', 'LMLO']

        # Remove studies with missing views
        df = df[df['all_views_bilateral'] == 'True']

        # Define transforms
        transforms = Compose([
            HighdicomMultiframeImageReaderd(keys=views, client=client, is_dbt=True),
            EnsureChannelFirstd(keys=views),
            Resized(spatial_size=(-1, 518, 518), keys=views),
            RepeatChanneld(keys=views)
        ])

        # Group by study ID
        for study_id, study_df in df.groupby("ANON_StudyID"):
            # Create dict with views and label initialized to None
            study_dict = {view: None for view in views}
            study_dict["label"] = None
            study_dict["mask"] = None

            # Fill in data from patient's rows
            split = None
            for _, row in study_df.iterrows():
                view = row['View']
                if view in views:
                    # Store identifier data as dictionary instead of RPSDICOMSeriesIdentifier object
                    identifier_data = {
                        'mrn': row['Patient ID'],
                        'accession': row['Accession_Number'],
                        'series_instance_uid': row['Series Instance UID'],
                        'site': row['Issuer of Patient ID']
                    }
                    study_dict[view] = identifier_data
                study_dict["label"] = row["y_label"]
                study_dict["mask"] = row["y_mask"]
                study_dict["ANON_SeriesID"] = row["ANON_SeriesID"]
                study_dict["ANON_StudyID"] = row["ANON_StudyID"]
                split = row["split"]

            # Add to appropriate list if all views present
            if all(study_dict[view] for view in views):
                if split == "train":
                    train_data.append(study_dict)
                elif split == "val":
                    val_data.append(study_dict)
                elif split == "test":
                    test_data.append(study_dict)

    elif all_views_unilat:
        df = df[df['all_views_unilateral'] == 'True']

        # Group by study ID
        for study_id, study_df in df.groupby("ANON_StudyID"):
            for _, study_df_unilat in study_df.groupby("Laterality"):
                # Get laterality
                laterality = study_df_unilat['Laterality'].iloc[0]
                # Define views
                if laterality == 'Right':
                    views = ['RCC', 'RMLO']
                elif laterality == 'Left':
                    views = ['LCC', 'LMLO']
                else:
                    raise ValueError(f"Unexpected laterality: {laterality}")

                # Define transforms
                transforms = Compose([
                    HighdicomMultiframeImageReaderd(keys=views, client=client, is_dbt=True),
                    EnsureChannelFirstd(keys=views),
                    Resized(spatial_size=(-1, 518, 518), keys=views),
                    RepeatChanneld(keys=views)
                ])

                # Create dict with views and label initialized to None
                study_dict = {view: None for view in views}
                study_dict["Laterality"] = laterality
                study_dict["label"] = None
                study_dict["mask"] = None

                # Fill in data from patient's rows
                split = None
                for _, row in study_df_unilat.iterrows():
                    view = row['View']
                    if view in views:
                        # Store identifier data as dictionary instead of RPSDICOMSeriesIdentifier object
                        identifier_data = {
                            'mrn': row['Patient ID'],
                            'accession': row['Accession_Number'],
                            'series_instance_uid': row['Series Instance UID'],
                            'site': row['Issuer of Patient ID']
                        }
                        study_dict[view] = identifier_data
                    study_dict["label"] = row["y_label"]
                    study_dict["mask"] = row["y_mask"]
                    study_dict["ANON_SeriesID"] = row["ANON_SeriesID"]
                    study_dict["ANON_StudyID"] = row["ANON_StudyID"]
                    split = row["split"]

                # Add to appropriate list if all views present
                if all(study_dict[view] for view in views):
                    if split == "train":
                        train_data.append(study_dict)
                    elif split == "val":
                        val_data.append(study_dict)
                    elif split == "test":
                        test_data.append(study_dict)

    else:
        # Group by patient ID
        for patient_id, patient_df in df.groupby("EMPI"):
            # Create empty dict
            patient_dict = {}
            patient_dict["label"] = None
            patient_dict["mask"] = None

            # Fill in data from patient's rows
            split = None
            for _, row in patient_df.iterrows():
                # Store identifier data as dictionary instead of RPSDICOMSeriesIdentifier object
                identifier_data = {
                    'mrn': row['Patient ID'],
                    'accession': row['Accession_Number'],
                    'series_instance_uid': row['Series Instance UID'],
                    'site': row['Issuer of Patient ID']
                }
                patient_dict["img"] = identifier_data
                patient_dict["label"] = row["y_label"]
                patient_dict["mask"] = row["y_mask"]
                patient_dict["ANON_SeriesID"] = row["ANON_SeriesID"]
                patient_dict["ANON_StudyID"] = row["ANON_StudyID"]
                split = row["split"]

            # Add to appropriate list if all views present
            if split == "train":
                train_data.append(patient_dict)
            elif split == "val":
                val_data.append(patient_dict)
            elif split == "test":
                test_data.append(patient_dict)

    print(
        f"Found {len(train_data)} training datapoints, {len(val_data)} validation datapoints and {len(test_data)} test datapoints."
        f"Number of studies in train_data: {len(set([d['ANON_StudyID'] for d in train_data]))}"
        f"Number of studies in val_data: {len(set([d['ANON_StudyID'] for d in val_data]))}"
        f"Number of studies in test_data: {len(set([d['ANON_StudyID'] for d in test_data]))}"
    )

    len_train_data = len(train_data)
    len_val_data = len(val_data)
    len_test_data = len(test_data)

    train_ds = StreamingDataset(
        data=train_data,
        transform=transforms,
        workers=num_workers,
        queue_size=20,
        log_dir=os.path.join(output_dir, 'train_log'),
        repeat=False,
    )

    val_ds = StreamingDataset(
        data=val_data,
        transform=transforms,
        workers=num_workers,
        queue_size=20,
        log_dir=os.path.join(output_dir, 'val_log'),
        repeat=False,
    )

    test_ds = StreamingDataset(
        data=test_data,
        transform=transforms,
        workers=num_workers,
        queue_size=20,
        log_dir=os.path.join(output_dir, 'test_log'),
        repeat=False,
    )

    return train_ds, val_ds, test_ds, len_train_data, len_val_data, len_test_data


def get_datasets_rps_web(
        csv_path,
        token_0,
        token_1,
        cohort,
        all_views=False
):
    df = pd.read_csv(csv_path, dtype=str)

    # Extract rows and get token for correct cohort (BC_label = 'False' for healthy, 'True' for cancer)
    if cohort == 'cancer':
        df = df[df['cohort'] == cohort]
        token = token_1
    elif cohort == 'healthy':
        df = df[df['cohort'] == cohort]
        token = token_0
    else:
        raise ValueError("Cohort must be either 'cancer' or 'healthy'.")

    # Convert all entries in Series Instance UID to str
    df['Series Instance UID'] = df['Series Instance UID'].astype(str)

    df_train = df[df['split'] == 'train']
    df_val = df[df['split'] == 'val']
    df_test = df[df['split'] == 'test']

    # Initialize RPS client
    client = RPSClient(token=token)

    # Create reader
    reader = RPSDICOMWebSeriesReader(client=client)

    # Define views
    views = ["LCC", "RCC", "LMLO", "RMLO"]

    # Create lists for train and val data
    train_data = []
    val_data = []
    test_data = []

    if all_views:
        # Define transforms
        transforms = Compose([
            LoadImaged(reader=reader, keys=views),  # Use RPSReader to load images
            EnsureChannelFirstd(keys=views),
            Orientationd(keys=views, axcodes='LPS'),
            CustomOrientationd(keys=views),
            Resized(spatial_size=(-1, 518, 518), keys=views),
            CustomWindowd(keys=views),
            RepeatChanneld(keys=views)
        ])

        # Group by EMPI and Accession_Number
        for _, sub_df in df.groupby(["EMPI", "Accession_Number"]):
            # Create dict with views and label initialized to None
            study_dict = {view: None for view in views}
            study_dict["label"] = None
            study_dict["mask"] = None

            # Fill in data from study sub-dataframe
            split = None
            for _, row in sub_df.iterrows():
                view = row['View']
                if view in views:
                    identifier = RPSDICOMSeriesIdentifier(
                        mrn=row['Patient ID'],
                        accession=row['Accession_Number'],
                        series_instance_uid=row['Series Instance UID'],
                        site=row['Issuer of Patient ID'],
                    )
                    study_dict[view] = identifier
                study_dict["label"] = row["y_label"]
                study_dict["mask"] = row["y_mask"]
                study_dict["ANON_StudyID"] = row["ANON_StudyID"]
                split = row["split"]

            # Add to appropriate list if all views present
            if all(study_dict[view] for view in views):
                if split == "train":
                    train_data.append(study_dict)
                elif split == "val":
                    val_data.append(study_dict)
                elif split == "test":
                    test_data.append(study_dict)

    else:
        transforms = Compose([
            LoadImaged(reader=reader, keys=['img']),  # Use RPSReader to load images
            EnsureChannelFirstd(keys=['img']),
            Orientationd(keys=['img'], axcodes='LPS'),
            CustomOrientationd(keys=['img']),
            Resized(spatial_size=(-1, 518, 518), keys=['img']),
            CustomWindowd(keys=['img']),
            RepeatChanneld(keys=['img'])
        ])

        for _, row in df_train.iterrows():
            identifier = RPSDICOMSeriesIdentifier(
                mrn=row['Patient ID'],
                accession=row['Accession_Number'],
                series_instance_uid=row['Series Instance UID'],
                site=row['Issuer of Patient ID'],
            )
            train_data.append({
                "img": identifier,
                "label": row['y_label'],
                "mask": row['y_mask'],  # y_mask is a vector
                "ANON_SeriesID": row['ANON_SeriesID'],
                "ANON_StudyID": row['ANON_StudyID']
            })

        for _, row in df_val.iterrows():
            identifier = RPSDICOMSeriesIdentifier(
                mrn=row['Patient ID'],
                accession=row['Accession_Number'],
                series_instance_uid=row['Series Instance UID'],
                site=row['Issuer of Patient ID'],
            )
            val_data.append({
                "img": identifier,
                "label": row['y_label'], # y_label is ground truth vector
                "mask": row['y_mask'],  # y_mask is a vector
                "ANON_SeriesID": row['ANON_SeriesID'],
                "ANON_StudyID": row['ANON_StudyID']
            })

        for _, row in df_test.iterrows():
            identifier = RPSDICOMSeriesIdentifier(
                mrn=row['Patient ID'],
                accession=row['Accession_Number'],
                series_instance_uid=row['Series Instance UID'],
                site=row['Issuer of Patient ID'],
            )
            test_data.append({
                "img": identifier,
                "label": row['y_label'], # y_label is ground truth vector
                "mask": row['y_mask'],  # y_mask is a vector
                "ANON_SeriesID": row['ANON_SeriesID'],
                "ANON_StudyID": row['ANON_StudyID']
            })

    print(
        f"Found {len(train_data)} training datapoints, {len(val_data)} validation datapoints and {len(test_data)} test datapoints."
    )

    len_train_data = len(train_data)
    len_val_data = len(val_data)
    len_test_data = len(test_data)

    train_ds = Dataset(data=train_data, transform=transforms)
    val_ds = Dataset(data=val_data, transform=transforms)
    test_ds = Dataset(data=test_data, transform=transforms)

    return train_ds, val_ds, test_ds, len_train_data, len_val_data, len_test_data


def custom_collate_fn(batch):
    # Remove metadata if tensors are MetaTensor
    batch = [{k: v.as_tensor() if isinstance(v, MetaTensor) else v for k, v in item.items()} for item in batch]

    # Check that batch contains exactly one element
    if len(batch) != 1:
        raise ValueError("Batch should contain exactly one element.")

    img = batch[0]['img']
    label = batch[0]['label']

    # Check that the img is 4D (C, H, L, W)
    if img.ndim != 4:
        raise ValueError(f"Image tensor must be 4-dimensional (C, H, L, W). Received: {img.shape}")

    # Assuming shape (C, H, L, W)
    C, H, L, W = img.shape

    # Split along H dimension to get 2D slices and concatenate along the batch dimension
    imgs_split = [img[:, h, :, :] for h in range(H)]
    new_batch_imgs = torch.stack(imgs_split, dim=0)

    # Create a new batch with the same label repeated for each slice
    new_batch_labels = [label] * H

    return {'img': new_batch_imgs, 'label': new_batch_labels}


def _get_stat_files(embeddings_dir, patient_id, token_type, lat):
    """Function to get file paths for all statistics of a given token type."""
    return {
        stat: os.path.join(embeddings_dir, f"{patient_id}_{lat}_{token_type}_{stat}.npy")
        for stat in ['mean', 'std', 'min', 'max']
    }


def _get_embeddings(model, im, device):
    """Internal function to get embeddings from model."""
    # If input has 5 dimensions (batch dimension), squeeze it
    if len(im.shape) == 5:
        im = im.squeeze(0)
    # permute from [C, Z, 518, 518] to [Z, C, 518, 518]
    im = im.permute(1, 0, 2, 3)
    #logging.info(f"Input shape: {im.shape}")
    chunks = torch.split(im, 75, dim=0) # drop to 40 (70, 60 before)
    for i, chunk in enumerate(chunks):
        #logging.debug(f"Processing chunk {i} of shape {chunk.shape}")
        chunk_feat = model.forward_features(chunk.to(device))
        if i == 0:
            patch_tokens = chunk_feat["x_norm_patchtokens"]
            cls_tokens = chunk_feat["x_norm_clstoken"]
        else:
            patch_tokens = torch.cat(
                (patch_tokens, chunk_feat["x_norm_patchtokens"]), dim=0
            )
            cls_tokens = torch.cat((cls_tokens, chunk_feat["x_norm_clstoken"]), dim=0)
    return patch_tokens, cls_tokens


def _calculate_stats(tokens, is_patch_token=True):
    """Function to calculate statistics for tokens."""
    dims = (0, 1) if is_patch_token else (0,)
    return {
        'mean': tokens.mean(dim=dims),
        'std': tokens.std(dim=dims),
        'min': tokens.min(dim=0)[0].min(dim=0)[0] if is_patch_token else tokens.min(dim=0)[0],
        'max': tokens.max(dim=0)[0].max(dim=0)[0] if is_patch_token else tokens.max(dim=0)[0],
        'median': tokens.median(dim=0)[0].median(dim=0)[0] if is_patch_token else tokens.median(dim=0)[0],
        'percentile_25': tokens.quantile(0.25, dim=0)[0] if is_patch_token else tokens.quantile(0.25, dim=0),
        'percentile_75': tokens.quantile(0.75, dim=0)[0] if is_patch_token else tokens.quantile(0.75, dim=0),
    }


def _save_embeddings(stats_dict, file_paths):
    """Function to save statistics to corresponding files."""
    for stat, file_path in file_paths.items():
        np.save(file_path, stats_dict[stat].cpu().numpy())


def extract_and_save_embeddings(model, dataset, device, num_workers, pin_mem, embeddings_dir, len_dataset, bilateral: True,
                                unilateral: False, streaming: False):
    """Function to extract and save embeddings from a model."""
    logging.info(f"Using device: {device}")

    model.to(device)
    model.eval()

    if streaming:
        embeddings_exist = 0 # Counter to log number of iterations since last new embeddings saved
        max_iters = 200 # Maximum number of iterations to skip if embeddings already exist
        #with dataset.active():
        dataset.start()
        with torch.no_grad():
            #for batch in tqdm(dataset, desc="Extracting embeddings"):
            for study in dataset:#.take(len_dataset):
                study_id = study['ANON_StudyID']

                if bilateral:
                    # Get file paths for both token types
                    patch_files = _get_stat_files(embeddings_dir, study_id, "patch", lat='bilateral')
                    cls_files = _get_stat_files(embeddings_dir, study_id, "cls", lat='bilateral')

                    # Check if all files exist
                    if all(os.path.exists(f) for f in patch_files.values()) and \
                            all(os.path.exists(f) for f in cls_files.values()):
                        print(f"Skipping series {study_id} - embeddings already exist")
                        logging.info("Skipping series %s - embeddings already exist" % study_id)
                        continue

                    views = ["RCC", "LCC", "RMLO", "LMLO"]

                    # Get embeddings for all views
                    patch_embeddings = {}
                    cls_embeddings = {}
                    for view in views:
                        img = study[view]
                        logging.info(f"Processing view {view} of shape {img.shape}")
                        patch_tokens, cls_tokens = _get_embeddings(model, img, device)

                        # Calculate statistics for both token types
                        patch_embeddings[view] = _calculate_stats(patch_tokens, is_patch_token=True)
                        cls_embeddings[view] = _calculate_stats(cls_tokens, is_patch_token=False)


                    # Concatenate embeddings from all views
                    combined_patch_stats = {
                        stat: torch.cat([v[stat] for v in patch_embeddings.values()], dim=0)
                        for stat in ['mean', 'std', 'min', 'max']
                    }

                    combined_cls_stats = {
                        stat: torch.cat([v[stat] for v in cls_embeddings.values()], dim=0)
                        for stat in ['mean', 'std', 'min', 'max']
                    }

                    # Save all statistics
                    _save_embeddings(combined_patch_stats, patch_files)
                    _save_embeddings(combined_cls_stats, cls_files)

                elif unilateral:

                    lat = study['Laterality']

                    # Get file paths for both token types
                    patch_files = _get_stat_files(embeddings_dir, study_id, "patch", lat=lat)
                    cls_files = _get_stat_files(embeddings_dir, study_id, "cls", lat=lat)

                    # Check if all files exist
                    if all(os.path.exists(f) for f in patch_files.values()) and \
                            all(os.path.exists(f) for f in cls_files.values()):
                        print(f"Skipping series {study_id} - embeddings already exist")
                        logging.info("Skipping series %s - embeddings already exist" % study_id)
                        continue

                    if lat == 'Right':
                        views = ["RCC", "RMLO"]
                    elif lat == 'Left':
                        views = ["LCC", "LMLO"]
                    else:
                        raise ValueError(f"Unexpected laterality: {lat}")

                    # Get embeddings for all views
                    patch_embeddings = {}
                    cls_embeddings = {}
                    for view in views:
                        img = study[view]
                        patch_tokens, cls_tokens = _get_embeddings(model, img, device)

                        # Calculate statistics for both token types
                        patch_embeddings[view] = _calculate_stats(patch_tokens, is_patch_token=True)
                        cls_embeddings[view] = _calculate_stats(cls_tokens, is_patch_token=False)

                    # Concatenate embeddings from all views
                    combined_patch_stats = {
                        stat: torch.cat([v[stat] for v in patch_embeddings.values()], dim=0)
                        for stat in ['mean', 'std', 'min', 'max']
                    }

                    combined_cls_stats = {
                        stat: torch.cat([v[stat] for v in cls_embeddings.values()], dim=0)
                        for stat in ['mean', 'std', 'min', 'max']
                    }

                    # Save all statistics
                    _save_embeddings(combined_patch_stats, patch_files)
                    _save_embeddings(combined_cls_stats, cls_files)

            dataset.stop()
    else:
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=num_workers,
            pin_memory=pin_mem,
            drop_last=False,
        )

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting embeddings"):
                # HOTFIX: Get patient_id first to check if embeddings exist
                img = batch['img']
                label = batch['label'][0]
                mask = batch['mask'][0]
                series_id = batch['ANON_SeriesID'][0]

                # Get file paths for both token types
                patch_files = _get_stat_files(embeddings_dir, series_id, "patch")
                cls_files = _get_stat_files(embeddings_dir, series_id, "cls")

                # Check if all files exist
                if all(os.path.exists(f) for f in patch_files.values()) and \
                        all(os.path.exists(f) for f in cls_files.values()):
                    logging.info(f"Skipping series {series_id} - embeddings already exist")
                    continue

                # Get embeddings
                patch_tokens, cls_tokens = _get_embeddings(model, img, device)

                # Calculate statistics for both token types
                patch_embeddings = _calculate_stats(patch_tokens, is_patch_token=True)
                cls_embeddings = _calculate_stats(cls_tokens, is_patch_token=False)

                # Save all statistics
                _save_embeddings(patch_embeddings, patch_files)
                _save_embeddings(cls_embeddings, cls_files)

                logging.info(f"Saved patch embeddings to {', '.join(patch_files.values())}")
                logging.info(f"Saved cls embeddings to {', '.join(cls_files.values())}")


def run_full_pipeline(csv_path, token_0, token_1, cohort, pin_mem, num_workers, model_name, output_dir, streaming, all_views_bilat, all_views_unilat):

    # Set up logging
    setup_logging(output_dir)

    # Define destination directories for train and val embeddings
    save_dir_train = os.path.join(output_dir, cohort, model_name, 'train')
    save_dir_val = os.path.join(output_dir, cohort, model_name, 'val')
    logging.info("Saving embeddings to: %s and %s" % (save_dir_train, save_dir_val))

    # fix the seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA Available:", torch.cuda.is_available())
    print("Current Device:", torch.cuda.current_device())
    print("Device Name:", torch.cuda.get_device_name(0))

    # Check if embeddings already exist
    if args.train:
        train_embedding_dir = os.path.join(output_dir, cohort, model_name, "train", "embeddings")
        embeddings_exist = (
                os.path.exists(train_embedding_dir)
                and len(os.listdir(train_embedding_dir)) > 0
        )
        if not embeddings_exist:
            os.makedirs(train_embedding_dir, exist_ok=True)
            logging.info("No existing embeddings found for subset train. Extracting embeddings...")

    if args.val:
        val_embedding_dir = os.path.join(output_dir, cohort, model_name, "val", "embeddings")
        embeddings_exist = (os.path.exists(val_embedding_dir)
                            and len(os.listdir(val_embedding_dir)) > 0
                            )
        if not embeddings_exist:
            os.makedirs(val_embedding_dir, exist_ok=True)
            logging.info("No existing embeddings found for subset val. Extracting embeddings...")

    if args.test:
        test_embedding_dir = os.path.join(output_dir, cohort, model_name, "test", "embeddings")
        embeddings_exist = (os.path.exists(test_embedding_dir)
                            and len(os.listdir(test_embedding_dir)) > 0
                            )
        if not embeddings_exist:
            os.makedirs(test_embedding_dir, exist_ok=True)
            logging.info("No existing embeddings found for subset test. Extracting embeddings...")


    # Create dataset objects
    if streaming:
        dataset_train, dataset_val, dataset_test, len_train_data, len_val_data, len_test_data = get_datasets_streaming(
            csv_path,
            cohort,
            token_0,
            token_1,
            output_dir,
            num_workers,
            all_views_bilat=all_views_bilat,
            all_views_unilat=all_views_unilat
        )
    else:
        dataset_train, dataset_val, dataset_test, len_train_data, len_val_data, len_test_data = get_datasets_rps_web(
            csv_path,
            token_0,
            token_1,
            cohort=cohort,
            all_views=all_views_bilat,
        )

    # Load DINOv2 model
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')

    if model_name == "dbtdino":
        logging.info(f"Loading backbone checkpoint from {args.backbone_checkpoint}")

        # Checkpoint file for ImageNet dino pre-trained on dbt:
        state_dict = torch.load(args.backbone_checkpoint, map_location='cpu')
        ## Define new state dict to match with the dinov2_vitb14 model loaded above
        new_state_dict = {}
        for k, v in state_dict['teacher'].items():
            if not k.startswith('dino_head'):  # Exclude dino_head weights
                new_key = k.replace('backbone.', '', 1)  # Remove only the first occurrence of "backbone."
                new_state_dict[new_key] = v
        msg = model.load_state_dict(new_state_dict) #don't use strict=False
        logging.info(msg)
        del new_state_dict # Delete new_state_dict to free up memory
    elif model_name == 'dinov2_vitb14':
        logging.info(f"Using default DINOv2 ViT-B/14 model")
        pass
    else:
        raise ValueError("Model not recognized. Please choose from 'dinov2_vitb14' or 'dbtdino'.")

    print("Model = %s" % str(model_name))

    logging.info("Extracting and saving embeddings...")
    # Compute and save embeddings to disk
    if args.train:
        extract_and_save_embeddings(
            model,
            dataset_train,
            device,
            num_workers,
            pin_mem,
            train_embedding_dir,
            len_train_data,
            bilateral=all_views_bilat,
            unilateral=all_views_unilat,
            streaming=streaming
        )

    if args.val:
        extract_and_save_embeddings(
            model,
            dataset_val,
            device,
            num_workers,
            pin_mem,
            val_embedding_dir,
            len_val_data,
            bilateral=all_views_bilat,
            unilateral=all_views_unilat,
            streaming=streaming
        )

    if args.test:
        extract_and_save_embeddings(
            model,
            dataset_test,
            device,
            num_workers,
            pin_mem,
            test_embedding_dir,
            len_test_data,
            bilateral=all_views_bilat,
            unilateral=all_views_unilat,
            streaming=streaming
        )


    print(f"Finished extracting features to: {save_dir_train} and {save_dir_val}. Exiting...")
    sys.exit(0)


def main(args):

    run_full_pipeline(args.csv_path, args.token_0, args.token_1, args.cohort, args.pin_mem, args.num_workers,
                      args.model_name, args.output_dir, args.streaming, args.all_views_bilat, args.all_views_unilat)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Compute DINOv2 embeddings on DBT images.")
    parser.add_argument('--csv_path', help='path to csv definition file of the dataset')
    parser.add_argument('--token_0', help='RPSClient token for cancer positive samples')
    parser.add_argument('--token_1', help='RPSClient token for cancer negative samples')
    parser.add_argument('--cohort', help='Cohort to extract embeddings for: cancer or healthy')
    parser.add_argument('--train', action='store_true', help='Extract embeddings for training set')
    parser.add_argument('--val', action='store_true', help='Extract embeddings for validation set')
    parser.add_argument('--test', action='store_true', help='Extract embeddings for test set')
    parser.add_argument('--pin_mem', action='store_true', help='Pin memory in DataLoader')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers in DataLoader')
    parser.add_argument('--model_name', help='Model to use for extracting embeddings')
    parser.add_argument('--output_dir', help='Output directory to save embeddings')
    parser.add_argument('--streaming', action='store_true', help='Use streaming dataset')
    parser.add_argument('--all_views_bilat', action='store_true', help='Compute and save concatenated embeddings for all views')
    parser.add_argument('--all_views_unilat', action='store_true', help='Compute and save concatenated embeddings for all views')
    parser.add_argument('--backbone_checkpoint', default="path/to/backbone_checkpoint.pth", 
                       help='Path to the backbone checkpoint file (default: ImageNet DINO checkpoint)')
    parser.add_argument('--gpu_id', type=int, default=None, help='GPU ID to use (e.g., 0, 1, 2). If not specified, uses default device.')
    args = parser.parse_args()

    main(args)





