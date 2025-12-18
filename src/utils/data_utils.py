import logging
import os

import numpy as np
import torch
from monai.transforms import Resize
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm


def _get_embeddings(model, im, device, model_type):
    """Get embeddings from model."""
    # If input has 5 dimensions (batch dimension), squeeze it
    if len(im.shape) == 5:
        im = im.squeeze(0)
    # permute to [3, 1, Z, 518, 518]
    im = im.permute(1, 0, 2, 3)
    logging.info(f"Input shape: {im.shape}")
    chunks = torch.split(im, 75, dim=0)

    if model_type == "dino_dbt":
        for i, chunk in enumerate(chunks):
            logging.debug(f"Processing chunk {i} of shape {chunk.shape}")
            chunk_feat = model.forward_features(chunk.to(device))
            if i == 0:
                patch_tokens = chunk_feat["x_norm_patchtokens"]
                cls_tokens = chunk_feat["x_norm_clstoken"]
            else:
                patch_tokens = torch.cat(
                    (patch_tokens, chunk_feat["x_norm_patchtokens"]), dim=0
                )
                cls_tokens = torch.cat((cls_tokens, chunk_feat["x_norm_clstoken"]), dim=0)

    elif model_type == "vit_baseline":
        for i, chunk in enumerate(chunks):
            logging.debug(f"Processing chunk {i} of shape {chunk.shape}")
            chunk_cls_tokens, chunk_patch_tokens  = model(chunk.to(device))
            if i == 0:
                patch_tokens = chunk_patch_tokens
                cls_tokens = chunk_cls_tokens
            else:
                patch_tokens = torch.cat(
                    (patch_tokens, chunk_patch_tokens), dim=0
                )
                cls_tokens = torch.cat((cls_tokens, chunk_cls_tokens), dim=0)

    elif model_type == "densenet_121":
        for i, chunk in enumerate(chunks):
            logging.debug(f"Processing chunk {i} of shape {chunk.shape}")
            chunk_feat = model(chunk.to(device))
            if i == 0:
                patch_tokens = chunk_feat
                cls_tokens = chunk_feat
            else:
                patch_tokens = torch.cat((patch_tokens, chunk_feat), dim=0)
                cls_tokens = torch.cat((cls_tokens, chunk_feat), dim=0)
    else: 
        raise ValueError(f"Unsupported model type: {model_type}. Choose from 'dino_dbt' or 'vit_baseline'.")
    
    return patch_tokens, cls_tokens


def _calculate_stats(tokens, is_patch_token=True, model_type="dino_dbt"):
    """Calculate statistics for tokens."""
    if model_type == "densenet_121":
        # For DenseNet, treat as CLS token and ensure proper dimensions
        if len(tokens.shape) == 1:
            tokens = tokens.unsqueeze(0)
        is_patch_token = False
    
    dims = (0, 1) if is_patch_token else (0,)
    
    return {
        "mean": tokens.mean(dim=dims),
        "std": tokens.std(dim=dims),
        "min": tokens.min(dim=0)[0].min(dim=0)[0]
        if is_patch_token
        else tokens.min(dim=0)[0],
        "max": tokens.max(dim=0)[0].max(dim=0)[0]
        if is_patch_token
        else tokens.max(dim=0)[0],
    }


def _get_stat_files(embeddings_dir, patient_id, token_type):
    """Get file paths for all statistics of a given token type."""
    return {
        stat: os.path.join(embeddings_dir, f"{patient_id}_{token_type}_{stat}.npy")
        for stat in ["mean", "std", "min", "max"]
    }


def _save_embeddings(stats_dict, file_paths):
    """Save statistics to corresponding files."""
    for stat, file_path in file_paths.items():
        np.save(file_path, stats_dict[stat].cpu().numpy())


def extract_and_save_embeddings(model, dataloader, embedding_dir, model_type):
    """Extract and save embeddings from a model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    model.to(device)
    model.eval()
    os.makedirs(embedding_dir, exist_ok=True)
    embeddings_dir = os.path.join(embedding_dir, "embeddings")
    os.makedirs(embeddings_dir, exist_ok=True)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            patient_id = batch["ANON_SeriesID"][0][:6]

            patch_files = _get_stat_files(embeddings_dir, patient_id, "patch")
            cls_files = _get_stat_files(embeddings_dir, patient_id, "cls")

            if all(os.path.exists(f) for f in patch_files.values()) and all(
                os.path.exists(f) for f in cls_files.values()
            ):
                logging.info(
                    f"Skipping patient {patient_id} - embeddings already exist"
                )
                continue

            views = ["RCC", "LCC", "RMLO", "LMLO"]

            patch_embeddings = {}
            cls_embeddings = {}
            for view in views:
                imgs = batch[view]
                
                patch_tokens, cls_tokens = _get_embeddings(model, imgs, device, model_type)
                
                patch_embeddings[view] = _calculate_stats(
                    patch_tokens, is_patch_token=True, model_type=model_type
                )
                cls_embeddings[view] = _calculate_stats(
                    cls_tokens, is_patch_token=False, model_type=model_type
                )
            
            combined_patch_stats = {
                stat: torch.cat([v[stat] for v in patch_embeddings.values()], dim=0)
                for stat in ["mean", "std", "min", "max"]
            }

            combined_cls_stats = {
                stat: torch.cat([v[stat] for v in cls_embeddings.values()], dim=0)
                for stat in ["mean", "std", "min", "max"]
            }

            _save_embeddings(combined_patch_stats, patch_files)
            _save_embeddings(combined_cls_stats, cls_files)

            logging.info(f"Saved patch embeddings to {', '.join(patch_files.values())}")
            logging.info(f"Saved cls embeddings to {', '.join(cls_files.values())}")



def extract_and_save_embeddings_detection(model, dataloader, embedding_dir):
    """Extract and save embeddings from a model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    model.to(device)
    model.eval()
    os.makedirs(embedding_dir, exist_ok=True)
    embeddings_dir = os.path.join(embedding_dir, "embeddings")
    os.makedirs(embeddings_dir, exist_ok=True)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):

            patient_id = batch["AnonID"]
            patch_files = os.path.join(embeddings_dir, patient_id[0])

            if os.path.exists(patch_files):
                logging.info(
                    f"Skipping patient {patient_id} - embeddings already exist"
                )
                continue

            patch_embeddings = model.forward_ms(batch["img"].to(device))
            patch_embeddings = torch.cat(patch_embeddings, dim=1)

            # remove the first dimension
            patch_embeddings = patch_embeddings.squeeze(0)
            
            np.save(patch_files, patch_embeddings.cpu().numpy())

            logging.info(f"Saved patch embeddings to {patch_files}")

def extract_and_save_embeddings_mil_detection(model, dataloader, embedding_dir):
    """Extract and save embeddings from a model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    model.to(device)
    model.eval()
    os.makedirs(embedding_dir, exist_ok=True)
    embeddings_dir = os.path.join(embedding_dir, "embeddings")
    os.makedirs(embeddings_dir, exist_ok=True)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            patient_id = batch["AnonID"]
            patch_files = os.path.join(embeddings_dir, patient_id[0])

            if os.path.exists(patch_files):
                logging.info(
                    f"Skipping patient {patient_id} - embeddings already exist"
                )
                continue

            im = batch["img"]
            if len(im.shape) == 5:
                im = im.squeeze(0)
            # permute to [3, 1, Z, 518, 518]
            im = im.permute(1, 0, 2, 3)
            logging.info(f"Input shape: {im.shape}")
            chunks = torch.split(im, 100, dim=0)
            for i, chunk in enumerate(chunks):
                logging.debug(f"Processing chunk {i} of shape {chunk.shape}")
                chunk_feat = model.forward_ms(chunk.to(device))
                chunk_feat = torch.cat(chunk_feat, dim=1)

                if i == 0:
                    patch_embeddings = chunk_feat
                else:
                    patch_embeddings = torch.cat(
                        (patch_embeddings, chunk_feat), dim=0
                    )

            # Move embeddings to CPU and convert to regular tensor if it's a MetaTensor
            patch_embeddings = patch_embeddings.cpu()
            if hasattr(patch_embeddings, 'as_tensor'):
                patch_embeddings = patch_embeddings.as_tensor()

            # shape of patch_embeddings is (slice_dim, feat_dim, 37, 37)
            # Permute to (slice_dim, 37, 37, feat_dim)
            patch_embeddings = patch_embeddings.permute(0, 2, 3, 1)

            # TEMP FIX, Take 32 linearly spaced slices from the first dimension
            print(f"WARNING: Taking 32 linearly spaced slices from the first dimension. This is a TEMP FIX.")
            
            # Handle cases with fewer than 32 slices by replicating slices
            
            num_slices = patch_embeddings.shape[0]
            if num_slices < 32:
                print(f"WARNING: Found {num_slices} slices, replicating to reach 32 slices")
                # Calculate how many times we need to repeat the slices
                repeat_factor = 32 // num_slices + (1 if 32 % num_slices != 0 else 0)
                # Repeat the slices
                patch_embeddings = patch_embeddings.repeat(repeat_factor, 1, 1, 1)
                # Take exactly 32 slices
                patch_embeddings = patch_embeddings[:32]
            else:
                # If we have more than 32 slices, take 32 linearly spaced slices
                indices = torch.linspace(0, num_slices - 1, 32, dtype=torch.long)
                patch_embeddings = patch_embeddings.index_select(dim=0, index=indices)

            # Flatten the patch_embeddings to get (slice_dim * 37 * 37, feat_dim)
            patch_embeddings = patch_embeddings.reshape(patch_embeddings.shape[0] * 37 * 37, -1)

            # Convert to float16
            patch_embeddings = patch_embeddings.numpy().astype(np.float16)

            np.save(patch_files, patch_embeddings)

            logging.info(f"Saved patch embeddings with shape {patch_embeddings.shape} to {patch_files}")


def extract_and_save_embeddings_mil_detection_cls(model, dataloader, embedding_dir):
    """Extract and save embeddings from a model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    model.to(device)
    model.eval()
    os.makedirs(embedding_dir, exist_ok=True)
    embeddings_dir = os.path.join(embedding_dir, "embeddings")
    os.makedirs(embeddings_dir, exist_ok=True)

    patch_size = 252
    resize = Resize(spatial_size=(-1, 1008, 1008))

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            patient_id = batch["AnonID"]
            patch_files = os.path.join(embeddings_dir, patient_id[0])

            if os.path.exists(patch_files):
                logging.info(
                    f"Skipping patient {patient_id} - embeddings already exist"
                )
                continue

            im = batch["img"].as_tensor()
            if len(im.shape) == 5:
                im = im.squeeze(0)
            logging.info(f"Input shape: {im.shape}")


            im_resized = resize(im)
            logging.info(f"Resized shape: {im_resized.shape}")

            # Create mask from middle slice
            # Get the middle slice from the slice dimension (dim=1)
            middle_slice_idx = im_resized.shape[1] // 2
            middle_slice = im_resized[0, middle_slice_idx]  # Take first channel of middle slice
            mask = middle_slice > 0.01

            # Get dimensions and calculate patch counts
            channels, num_slices, height, width = im_resized.shape
            patches_h, patches_w = height // patch_size, width // patch_size
            overlap_threshold = 0.1

            # Extract patches with sufficient mask overlap
            patches_list = []  # Temporary list to collect patches
            for slice_idx in range(num_slices):
                for i in range(patches_h):
                    for j in range(patches_w):
                        y_start, x_start = i * patch_size, j * patch_size
                        patch = im_resized[:, slice_idx, y_start:y_start+patch_size, x_start:x_start+patch_size]
                        
                        patch_mask = mask[y_start:y_start+patch_size, x_start:x_start+patch_size]
                        if patch_mask.sum() / (patch_size * patch_size) >= overlap_threshold:
                            patches_list.append(patch)
            all_patches = torch.stack(patches_list) if patches_list else torch.tensor([])
            logging.info(f"All patches shape: {all_patches.shape}")
            chunks = torch.split(all_patches, 200, dim=0)
            for i, chunk in enumerate(chunks):
                logging.debug(f"Processing chunk {i} of shape {chunk.shape}")
                chunk_feat = model(chunk.to(device))

                if i == 0:
                    cls_embeddings = chunk_feat
                else:
                    cls_embeddings = torch.cat(
                        (cls_embeddings, chunk_feat), dim=0
                    )
            # Move embeddings to CPU and convert to regular tensor if it's a MetaTensor
            cls_embeddings = cls_embeddings.cpu()
            if hasattr(cls_embeddings, 'as_tensor'):
                cls_embeddings = cls_embeddings.as_tensor()

            # shape of cls_embeddings is (num_patches, feat_dim)

            # Convert to float16
            cls_embeddings = cls_embeddings.numpy().astype(np.float16)

            np.save(patch_files, cls_embeddings)

            logging.info(f"Saved cls embeddings with shape {cls_embeddings.shape} to {patch_files}")

def pad_collate_fn(batch):
    """
    Pads the 'embedding' tensors in the batch to the same number of patches.
    Returns a dict with padded embeddings, stacked labels, and patient_ids.
    """
    embeddings = [item["embedding"] for item in batch]
    labels = torch.stack([item["label"] for item in batch])
    patient_ids = [item["patient_id"] for item in batch]

    # Pad all to the max number of patches
    padded_embeddings = pad_sequence(embeddings, batch_first=True, padding_value=0.0)
    # padded_embeddings shape: [batch_size, max_num_patches, embedding_dim]

    return {
        "embedding": padded_embeddings,
        "label": labels,
        "patient_id": patient_ids,
    }