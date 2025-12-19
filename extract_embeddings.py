"""
Minimal Example: Extract Embeddings with DBT-DINO

This script demonstrates how to use the DBT-DINO backbone to extract embeddings from DICOM images.

DBT-DINO is built on DINOv2 ViT-B/14 and trained on Digital Breast Tomosynthesis (DBT) images.

Requirements:
    - torch
    - numpy
    - monai
    - highdicom
    - pydicom
    See requirements.txt for full dependencies
"""

import os
import sys
import torch
import numpy as np

# Add src to path to import utilities
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from monai.transforms import Compose, EnsureChannelFirstd, Resized
from data.transforms import RepeatChanneld
from utils.highdcm_utils import HighdicomMultiframeImageReaderd


def _download_checkpoint(url, save_path):
    """
    Download checkpoint with progress bar and error handling.
    
    Args:
        url: URL to download from
        save_path: Path to save the downloaded file
    """
    import urllib.request
    import sys
    
    try:
        def reporthook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(downloaded * 100.0 / total_size, 100.0) if total_size > 0 else 0
            bar_length = 40
            filled = int(bar_length * percent / 100)
            bar = '=' * filled + '-' * (bar_length - filled)
            sys.stdout.write(f'\r[{bar}] {percent:.1f}% ({downloaded/(1024**2):.1f}MB/{total_size/(1024**2):.1f}MB)')
            sys.stdout.flush()
        
        print(f"Downloading model from {url} ...")
        urllib.request.urlretrieve(url, save_path, reporthook)
        print("\n✓ Download completed")
        
    except Exception as e:
        if os.path.exists(save_path):
            os.remove(save_path)  # Clean up partial download
        raise RuntimeError(f"Failed to download checkpoint: {e}")


def load_dbtdino_model(checkpoint_path="dbt_dino.pth", device="cpu"):
    """
    Load the DBT-DINO model from checkpoint.
    
    Args:
        checkpoint_path: Path to the DBT-DINO checkpoint file
        device: Device to load the model on ('cpu' or 'cuda')
    
    Returns:
        model: Loaded DBT-DINO model in eval mode
    """
    print(f"Loading DBT-DINO model on {device}...")
    
    # Load base DINOv2 ViT-B/14 model from torch hub
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    
    # Load DBT-DINO checkpoint weights
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint file not found at {checkpoint_path}")
        url = "https://zenodo.org/records/17981813/files/dbt_dino.pth"
        _download_checkpoint(url, checkpoint_path)
    
    try:
        state_dict = torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint from {checkpoint_path}: {e}")
        
    # Extract teacher weights and remove 'backbone.' prefix
    new_state_dict = {}
    for k, v in state_dict['teacher'].items():
        if not k.startswith('dino_head'):
            new_key = k.replace('backbone.', '', 1)
            new_state_dict[new_key] = v
    
    # Load weights into model
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    
    print("✓ Model loaded successfully")
    return model


def get_transforms(views, target_size=518):
    """
    
    This pipeline:
    1. Reads multiframe DICOM with highdicom (applies VOI transformations)
    2. Ensures channel is first dimension
    3. Resizes to target size while preserving depth dimension
    4. Repeats channel to create pseudo-RGB
    
    Args:
        views: List of view names (e.g., ['RCC', 'LCC', 'RMLO', 'LMLO'])
        target_size: Target spatial size (default: 518)
    
    Returns:
        Composed transform pipeline
    """
    transforms = Compose([
        HighdicomMultiframeImageReaderd(is_dbt=True, keys=views),
        EnsureChannelFirstd(keys=views),
        Resized(spatial_size=(-1, target_size, target_size), keys=views),
        RepeatChanneld(keys=views),
    ])
    return transforms


def preprocess_dicom(dicom_path, view_name='RCC', target_size=518):
    """
    Preprocess a single DICOM file.
    
    Args:
        dicom_path: Path to the DICOM file
        view_name: View name for the image (e.g., 'RCC', 'LCC', 'RMLO', 'LMLO')
        target_size: Target size for resizing (default: 518)
    
    Returns:
        Preprocessed image tensor of shape [C, Z, H, W] where:
        - C = 3 (pseudo-RGB channels)
        - Z = number of slices
        - H, W = target_size
    """
    # Create transform pipeline
    transform = get_transforms([view_name], target_size=target_size)
    
    # Create data dictionary
    data = {view_name: dicom_path}
    
    # Apply transforms
    transformed = transform(data)
    
    # Extract the image tensor
    img_tensor = transformed[view_name]
    
    return img_tensor


def extract_embeddings_from_dbt(model, img_tensor, device="cpu", chunk_size=75):
    """
    Extract embeddings from a DBT volume.
    
    This function:
    1. Permutes the tensor to [num_slices, 3, H, W]
    2. Processes slices in chunks of 75 (default)
    3. Extracts patch and CLS tokens for each slice
    4. Returns all embeddings
    
    Args:
        model: DBT-DINO model
        img_tensor: Preprocessed image tensor of shape [3, num_slices, H, W]
        device: Device to run inference on
        chunk_size: Number of slices to process at once (default: 75, as in paper)
    
    Returns:
        patch_tokens: Patch token embeddings of shape [num_slices, num_patches, embed_dim]
        cls_tokens: CLS token embeddings of shape [num_slices, embed_dim]
    """
    # If input has 5 dimensions (batch dimension), squeeze it
    if len(img_tensor.shape) == 5:
        img_tensor = img_tensor.squeeze(0)
    
    # Permute from [C, Z, H, W] to [Z, C, H, W]
    img_tensor = img_tensor.permute(1, 0, 2, 3)
    
    # Split into chunks to avoid memory issues (75 slices at a time, as in paper)
    chunks = torch.split(img_tensor, chunk_size, dim=0)
    
    patch_tokens_list = []
    cls_tokens_list = []
    
    with torch.no_grad():
        for chunk in chunks:
            chunk = chunk.to(device)
            
            # Forward pass through the model
            features = model.forward_features(chunk)
            
            # Extract and store tokens
            patch_tokens_list.append(features["x_norm_patchtokens"])
            cls_tokens_list.append(features["x_norm_clstoken"])
    
    # Concatenate all chunks
    patch_tokens = torch.cat(patch_tokens_list, dim=0)
    cls_tokens = torch.cat(cls_tokens_list, dim=0)
    
    return patch_tokens, cls_tokens


def compute_embedding_statistics(tokens, is_patch_token=True):
    """
    Compute statistics across slices.
    
    This matches the approach used in src/utils/data_utils.py for aggregating
    embeddings from multiple slices in DBT volumes.
    
    Args:
        tokens: Embedding tokens, shape [num_slices, num_patches, embed_dim] for patch tokens
                or [num_slices, embed_dim] for CLS tokens
        is_patch_token: Whether these are patch tokens (True) or CLS tokens (False)
    
    Returns:
        Dictionary containing mean, std, min, and max statistics
    """
    dims = (0, 1) if is_patch_token else (0,)
    
    stats = {
        'mean': tokens.mean(dim=dims),
        'std': tokens.std(dim=dims),
        'min': tokens.min(dim=0)[0].min(dim=0)[0] if is_patch_token else tokens.min(dim=0)[0],
        'max': tokens.max(dim=0)[0].max(dim=0)[0] if is_patch_token else tokens.max(dim=0)[0],
    }
    
    return stats


# ============================================================================
# Example Usage - EXACT pipeline from density prediction task in paper
# ============================================================================

def example_single_dicom():
    """
    Example: Extract embeddings from a single DICOM file.
    """
    print("=" * 70)
    print("EXAMPLE: Extracting embeddings from DBT DICOM file")
    print("=" * 70)
    
    # Configuration
    CHECKPOINT_PATH = "dbt_dino.pth"
    DICOM_PATH = "path/to/your/dbt_image.dcm"  # Path to your DICOM file
    VIEW_NAME = "RCC"  # View name: 'RCC', 'LCC', 'RMLO', or 'LMLO'
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Step 1: Load the model
    print("\n[1/4] Loading DBT-DINO model...")
    model = load_dbtdino_model(CHECKPOINT_PATH, device=DEVICE)
    
    # Step 2: Preprocess DICOM using EXACT transforms from paper
    print(f"[2/4] Preprocessing DICOM file with view '{VIEW_NAME}'...")
    img_tensor = preprocess_dicom(DICOM_PATH, view_name=VIEW_NAME, target_size=518)
    print(f"      Preprocessed shape: {img_tensor.shape}")  # [C=3, Z=num_slices, H=518, W=518]
    
    # Step 3: Extract embeddings
    print("[3/4] Extracting embeddings from all slices...")
    patch_tokens, cls_tokens = extract_embeddings_from_dbt(
        model, img_tensor, device=DEVICE, chunk_size=75
    )
    
    print("\n=== Extracted Embeddings ===")
    print(f"CLS tokens shape: {cls_tokens.shape}")      # [num_slices, 768]
    print(f"Patch tokens shape: {patch_tokens.shape}")  # [num_slices, 1369, 768]
    print(f"Number of slices processed: {cls_tokens.shape[0]}")
    print(f"Embedding dimension: {cls_tokens.shape[1]}")
    
    # Step 4: Compute statistics across slices (as done in the paper)
    print("\n[4/4] Computing statistics across slices (as in paper)...")
    patch_stats = compute_embedding_statistics(patch_tokens, is_patch_token=True)
    cls_stats = compute_embedding_statistics(cls_tokens, is_patch_token=False)
    
    print("\n=== Aggregated Statistics (mean, std, min, max) ===")
    print(f"Patch token statistics shape: {patch_stats['mean'].shape}")  # [768]
    print(f"CLS token statistics shape: {cls_stats['mean'].shape}")      # [768]
    
    # Step 5: Save embeddings (optional)
    print("\n=== Saving Embeddings ===")
    output_dir = "embeddings"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save aggregated statistics (as used in paper for density prediction)
    np.save(f"{output_dir}/patch_mean.npy", patch_stats['mean'].cpu().numpy())
    np.save(f"{output_dir}/patch_std.npy", patch_stats['std'].cpu().numpy())
    np.save(f"{output_dir}/patch_min.npy", patch_stats['min'].cpu().numpy())
    np.save(f"{output_dir}/patch_max.npy", patch_stats['max'].cpu().numpy())
    
    np.save(f"{output_dir}/cls_mean.npy", cls_stats['mean'].cpu().numpy())
    np.save(f"{output_dir}/cls_std.npy", cls_stats['std'].cpu().numpy())
    np.save(f"{output_dir}/cls_min.npy", cls_stats['min'].cpu().numpy())
    np.save(f"{output_dir}/cls_max.npy", cls_stats['max'].cpu().numpy())
    
    print(f"✓ Saved aggregated embeddings to '{output_dir}/'")
    print("\n✓ Embeddings extracted successfully using paper's pipeline!")


def example_multi_view():
    """
    Example: Extract embeddings from all 4 views of a patient.
    This matches the approach used in the paper's density prediction task.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE: Extracting embeddings from all 4 views")
    print("Matches paper's density prediction pipeline")
    print("=" * 70)
    
    # Configuration
    CHECKPOINT_PATH = "dbt_dino.pth"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Paths to your 4-view DICOM files
    dicom_paths = {
        'RCC': 'path/to/RCC.dcm',
        'LCC': 'path/to/LCC.dcm',
        'RMLO': 'path/to/RMLO.dcm',
        'LMLO': 'path/to/LMLO.dcm',
    }
    
    # Step 1: Load the model
    print("\n[1/3] Loading DBT-DINO model...")
    model = load_dbtdino_model(CHECKPOINT_PATH, device=DEVICE)
    
    # Step 2: Process all views using EXACT transforms from paper
    print("[2/3] Processing all 4 views with EXACT paper transforms...")
    transform = get_transforms(list(dicom_paths.keys()), target_size=518)
    transformed = transform(dicom_paths)
    
    # Step 3: Extract embeddings for each view and concatenate
    print("[3/3] Extracting and aggregating embeddings from all views...")
    
    all_patch_stats = []
    all_cls_stats = []
    
    for view_name in ['RCC', 'LCC', 'RMLO', 'LMLO']:
        print(f"    Processing {view_name}...")
        img_tensor = transformed[view_name]
        
        # Extract embeddings
        patch_tokens, cls_tokens = extract_embeddings_from_dbt(
            model, img_tensor, device=DEVICE, chunk_size=75
        )
        
        # Compute statistics for this view
        patch_stats = compute_embedding_statistics(patch_tokens, is_patch_token=True)
        cls_stats = compute_embedding_statistics(cls_tokens, is_patch_token=False)
        
        all_patch_stats.append(patch_stats)
        all_cls_stats.append(cls_stats)
    
    # Concatenate statistics from all views (as done in paper)
    combined_patch_stats = {
        stat: torch.cat([view_stats[stat] for view_stats in all_patch_stats], dim=0)
        for stat in ['mean', 'std', 'min', 'max']
    }
    
    combined_cls_stats = {
        stat: torch.cat([view_stats[stat] for view_stats in all_cls_stats], dim=0)
        for stat in ['mean', 'std', 'min', 'max']
    }
    
    print("\n=== Combined Embeddings from All Views ===")
    print(f"Combined patch statistics shape: {combined_patch_stats['mean'].shape}")  # [4*768]
    print(f"Combined CLS statistics shape: {combined_cls_stats['mean'].shape}")      # [4*768]
    
    # Save combined embeddings
    output_dir = "embeddings"
    os.makedirs(output_dir, exist_ok=True)
    patient_id = "patient_001"  # Replace with actual patient ID
    
    for stat in ['mean', 'std', 'min', 'max']:
        np.save(f"{output_dir}/{patient_id}_patch_{stat}.npy", 
                combined_patch_stats[stat].cpu().numpy())
        np.save(f"{output_dir}/{patient_id}_cls_{stat}.npy", 
                combined_cls_stats[stat].cpu().numpy())
    
    print(f"\n✓ Saved combined embeddings to '{output_dir}/'")
    print("✓ This matches the format used in the paper's density prediction task!")


if __name__ == "__main__":
    # Run examples
    # Uncomment the example you want to run:
    
    # For a single DICOM file
    # example_single_dicom()
    
    # For all 4 views (as in paper's density prediction)
    # example_multi_view()
    
    print("\n" + "=" * 70)
    print("To run an example, uncomment the appropriate line in __main__")
    print("These examples use the EXACT preprocessing from the paper.")
    print("=" * 70)

