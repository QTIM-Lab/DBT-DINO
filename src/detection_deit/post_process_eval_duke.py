import argparse
import json
import logging
import multiprocessing as mp
import os
from pathlib import Path
from typing import Dict, List, NamedTuple, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pycrumbs import tracked

matplotlib.use('Agg')  # Use non-interactive backend

# Ground truth boxes path
df_boxes = pd.read_csv("path/to/groundtruthboxes/all_slices_boxes.csv")

def setup_logging(output_dir: str, debug: bool = False):
    """
    Set up logging configuration for the evaluation process.
    
    Args:
        output_dir: Directory to save log files
        debug: If True, set logging level to DEBUG
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set logging level
    log_level = logging.DEBUG if debug else logging.INFO
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    simple_formatter = logging.Formatter('%(message)s')
    
    # Clear any existing handlers
    logging.getLogger().handlers.clear()
    
    # Set up root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # File handler for detailed logs
    log_file = os.path.join(output_dir, 'evaluation.log')
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    # Console handler for user-friendly output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    return logger

def calculate_iou(box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1, box2: Tuples of (x1, y1, x2, y2) coordinates
        
    Returns:
        IoU value between 0 and 1
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection coordinates
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)
    
    # Check if there's no intersection
    if x1_inter >= x2_inter or y1_inter >= y2_inter:
        return 0.0
    
    # Calculate intersection area
    intersection_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    # Calculate union area
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - intersection_area
    
    # Calculate IoU
    if union_area == 0:
        return 0.0
    
    return intersection_area / union_area


def calculate_iosib(box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:
    """
    Calculate Intersection over Smaller Intersecting Box (IoSIB) between two bounding boxes.
    
    Args:
        box1, box2: Tuples of (x1, y1, x2, y2) coordinates
        
    Returns:
        IoSIB value between 0 and 1
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection coordinates
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)
    
    # Check if there's no intersection
    if x1_inter >= x2_inter or y1_inter >= y2_inter:
        return 0.0
    
    # Calculate intersection area
    intersection_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    # Calculate areas of both boxes
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # Calculate IoSIB (intersection over smaller box)
    smaller_area = min(box1_area, box2_area)
    
    if smaller_area == 0:
        return 0.0
    
    return intersection_area / smaller_area


def combine_predictions_into_3d_candidates(
    df_preds: pd.DataFrame,
    score_threshold: float = 0.85,
    iosib_threshold: float = 0.65,
    slice_proximity_factor: float = 0.5,
    cum_prob_threshold: float = 0.95,
    k_max: int = 8,
    ensure_one: bool = True,
    debug: bool = False
) -> pd.DataFrame:
    """
    Combine 2D bounding box predictions into 3D candidates based on slice proximity, 
    score threshold, and IoSIB overlap.
    
    Args:
        df_preds: DataFrame with predictions
        score_threshold: Minimum score threshold (default 0.85)
        iosib_threshold: Minimum IoSIB threshold (default 0.65)
        slice_proximity_factor: Maximum slice distance as fraction of total slices (default 0.5)
        cum_prob_threshold: Cumulative softmax probability threshold per volume (default 0.80)
        k_max: Maximum number of candidates to keep per volume (default 4)
        ensure_one: If True, make sure at least one candidate per volume is kept (keeps the highest prob)
        debug: If True, print detailed processing information (default False)
    Returns:
        DataFrame with 3D candidates
    """
    # Filter by score threshold
    high_score_preds = df_preds[df_preds['score'] >= score_threshold].copy()
    
    logger = logging.getLogger(__name__)
    
    if len(high_score_preds) == 0:
        logger.debug(f"No predictions found with score >= {score_threshold}")
        return pd.DataFrame(columns=df_preds.columns)
    
    logger.debug(f"Processing {len(high_score_preds)} high-score predictions (score >= {score_threshold})")
    
    three_d_candidates = []
    
    # Group by volume
    volume_groups = high_score_preds.groupby(['patient_id', 'study_uid', 'view'])
    
    for (patient_id, study_uid, view), volume_df in volume_groups:
        logger.debug(f"Processing volume {patient_id}/{study_uid}/{view} with {len(volume_df)} predictions")
        
        # Calculate total number of slices in this volume
        total_slices = volume_df['slice'].max() - volume_df['slice'].min() + 1
        max_slice_distance = int(total_slices * slice_proximity_factor)
        
        logger.debug(f"  Total slices: {total_slices}, Max slice distance: {max_slice_distance}")
        
        # Track which predictions have been assigned to candidates
        assigned_predictions = set()
        
        # Convert to list for easier processing
        predictions_list = volume_df.to_dict('records')
        
        # For each prediction, try to form a 3D candidate
        for i, pred in enumerate(predictions_list):
            if i in assigned_predictions:
                continue
                
            # Start a new candidate with this prediction
            candidate_predictions = [i]
            candidate_boxes = [(pred['x1'], pred['y1'], pred['x2'], pred['y2'])]
            candidate_slices = [pred['slice']]
            candidate_scores = [pred['score']]
            
            # Find all other predictions that should be part of this candidate
            for j, other_pred in enumerate(predictions_list):
                if j == i or j in assigned_predictions:
                    continue
                
                # Check slice proximity
                slice_diff = abs(pred['slice'] - other_pred['slice'])
                if slice_diff > max_slice_distance:
                    continue
                
                # Check IoSIB with any prediction already in the candidate
                pred_box = (pred['x1'], pred['y1'], pred['x2'], pred['y2'])
                other_box = (other_pred['x1'], other_pred['y1'], other_pred['x2'], other_pred['y2'])
                
                iosib = calculate_iosib(pred_box, other_box)
                
                if iosib >= iosib_threshold:
                    candidate_predictions.append(j)
                    candidate_boxes.append(other_box)
                    candidate_slices.append(other_pred['slice'])
                    candidate_scores.append(other_pred['score'])
            
            # Now check if any remaining predictions should be added based on IoSIB with any box in candidate
            for j, other_pred in enumerate(predictions_list):
                if j in candidate_predictions or j in assigned_predictions:
                    continue
                
                # Check slice proximity with any slice in candidate
                min_slice_diff = min(abs(s - other_pred['slice']) for s in candidate_slices)
                if min_slice_diff > max_slice_distance:
                    continue
                
                # Check IoSIB with any box in the candidate
                other_box = (other_pred['x1'], other_pred['y1'], other_pred['x2'], other_pred['y2'])
                
                max_iosib = 0.0
                for candidate_box in candidate_boxes:
                    iosib = calculate_iosib(candidate_box, other_box)
                    max_iosib = max(max_iosib, iosib)
                
                if max_iosib >= iosib_threshold:
                    candidate_predictions.append(j)
                    candidate_boxes.append(other_box)
                    candidate_slices.append(other_pred['slice'])
                    candidate_scores.append(other_pred['score'])
            
            # Calculate depth (number of unique slices)
            unique_slices = len(set(candidate_slices))
            
            # Skip candidates with depth 1 (single slice)
            if unique_slices == 1:
                logger.debug(f"  Skipping single-slice candidate")
                continue
            
            # Mark all predictions in this candidate as assigned
            assigned_predictions.update(candidate_predictions)
            
            # Calculate 3D bounding box
            min_x = min(box[0] for box in candidate_boxes)
            min_y = min(box[1] for box in candidate_boxes)
            max_x = max(box[2] for box in candidate_boxes)
            max_y = max(box[3] for box in candidate_boxes)
            
            min_slice = min(candidate_slices)
            max_slice = max(candidate_slices)
            center_slice = int(np.median(candidate_slices))
            
            # Calculate score as average of top 10 scores
            sorted_scores = sorted(candidate_scores, reverse=True)
            top_scores = sorted_scores[:min(10, len(sorted_scores))]
            avg_score = np.mean(top_scores)
            
            logger.debug(f"  Created 3D candidate: {len(candidate_predictions)} predictions, "
                      f"depth {unique_slices}, score {avg_score:.3f}")
            
            three_d_candidates.append({
                'patient_id': patient_id,
                'study_uid': study_uid,
                'view': view,
                'slice': center_slice,
                'x1': min_x,
                'y1': min_y,
                'x2': max_x,
                'y2': max_y,
                'score': avg_score,
                'depth': unique_slices,
                'min_slice': min_slice,
                'max_slice': max_slice,
                'num_predictions': len(candidate_predictions),
                'top_10_avg_score': avg_score
            })
    
    result_df = pd.DataFrame(three_d_candidates)

    # Filter by min/max width and height based on ground truth
    min_width, max_width = 15.0, 206.0
    min_height, max_height = 9.0, 182.0
    if len(result_df) > 0:
        widths = result_df['x2'] - result_df['x1']
        heights = result_df['y2'] - result_df['y1']
        size_mask = (widths >= min_width) & (widths <= max_width) & (heights >= min_height) & (heights <= max_height)
        n_before = len(result_df)
        result_df = result_df[size_mask].reset_index(drop=True)
        n_after = len(result_df)
        logger.debug(f"Filtered candidates by width/height: {n_before} -> {n_after}")

    # ------------------------------------------------------------------
    # Per-volume cumulative probability filtering + top-k cap
    # ------------------------------------------------------------------
    if len(result_df) > 0:
        logger.debug("Applying per-volume cumulative probability filtering and top-k cap...")
        filtered_indices = []
        for (pid, sid, view), grp in result_df.groupby(['patient_id', 'study_uid', 'view']):
            scores = grp['score'].values.astype(np.float32)
            exp_scores = np.exp(scores - np.max(scores))
            probs = exp_scores / np.sum(exp_scores)
            sorted_idx = np.argsort(probs)[::-1]
            cum_prob = 0.0
            kept = []
            for idx_local in sorted_idx:
                global_idx = grp.index.values[idx_local]
                kept.append(global_idx)
                cum_prob += probs[idx_local]
                if len(kept) >= k_max or cum_prob >= cum_prob_threshold:
                    break

            # Safety: ensure at least one kept
            if ensure_one and len(kept) == 0:
                kept = [grp.index.values[sorted_idx[0]]]
            filtered_indices.extend(kept)
        n_before_softmax = len(result_df)
        result_df = result_df.loc[filtered_indices].reset_index(drop=True)
        logger.debug(f"Per-volume cumulative-prob filter: {n_before_softmax} -> {len(result_df)} candidates")

    if len(result_df) > 0:
        logger.debug(f"\nGenerated {len(result_df)} 3D candidates from {len(high_score_preds)} high-score predictions")
        logger.debug(f"Average depth: {result_df['depth'].mean():.1f}")
        logger.debug(f"Average number of predictions per candidate: {result_df['num_predictions'].mean():.1f}")
        # Print depth distribution
        depth_counts = result_df['depth'].value_counts().sort_index()
        logger.debug(f"Depth distribution: {dict(depth_counts)}")
    else:
        logger.debug("No 3D candidates generated")
    return result_df


def convert_predictions_to_challenge_format(df_preds: pd.DataFrame) -> pd.DataFrame:
    """
    Convert predictions from (x1,y1,x2,y2) format to challenge format (X,Y,Width,Height,Z,Depth)
    """
    df_challenge = df_preds.copy()
    
    # Rename columns to match challenge format
    df_challenge = df_challenge.rename(columns={
        'study_uid': 'StudyUID',
        'view': 'View',
        'score': 'Score'
    })
    
    # Convert bounding box format
    df_challenge['X'] = df_challenge['x1']  # Top-left X
    df_challenge['Y'] = df_challenge['y1']  # Top-left Y
    df_challenge['Width'] = df_challenge['x2'] - df_challenge['x1']  # Width
    df_challenge['Height'] = df_challenge['y2'] - df_challenge['y1']  # Height
    df_challenge['Z'] = df_challenge['slice']  # Slice number as Z coordinate
    df_challenge['Depth'] = 1  # Assume depth of 1 slice
    
    # Keep only columns needed for evaluation
    columns_to_keep = ['StudyUID', 'View', 'Score', 'X', 'Y', 'Width', 'Height', 'Z', 'Depth']
    df_challenge = df_challenge[columns_to_keep]
    
    return df_challenge


def filter_boxes_to_prediction_volumes(df_boxes: pd.DataFrame, df_preds: pd.DataFrame) -> pd.DataFrame:
    """
    Filter boxes to only include volumes that have predictions
    """
    # Get unique volumes from predictions
    pred_volumes = df_preds[['study_uid', 'view']].drop_duplicates()
    pred_volumes = pred_volumes.rename(columns={'study_uid': 'StudyUID', 'view': 'View'})
    
    # Rename columns in boxes if needed
    if 'study_uid' in df_boxes.columns:
        df_boxes = df_boxes.rename(columns={'study_uid': 'StudyUID', 'view': 'View'})
    
    # Filter boxes to only include volumes with predictions
    df_boxes_filtered = df_boxes.merge(pred_volumes, on=['StudyUID', 'View'], how='inner')
    
    return df_boxes_filtered


def evaluate(
    df_labels: pd.DataFrame,
    df_boxes: pd.DataFrame,
    df_pred: pd.DataFrame,
    return_froc_curve: bool = False,
) -> Tuple[Dict[str, float], Tuple[List[float], List[float]]]:
    """Evaluate predictions"""
    
    df_labels = df_labels.reset_index().set_index(["StudyUID", "View"]).sort_index()
    df_boxes = df_boxes.reset_index().set_index(["StudyUID", "View"]).sort_index()
    df_pred = df_pred.reset_index().set_index(["StudyUID", "View"]).sort_index()

    df_pred["TP"] = 0
    df_pred["GTID"] = -1

    thresholds = [df_pred["Score"].max() + 1.0]

    # find true positive predictions and assign detected ground truth box ID
    for box_pred in df_pred.itertuples():
        if box_pred.Index not in df_boxes.index:
            continue

        df_boxes_view = df_boxes.loc[[box_pred.Index]]
        view_slice_offset = df_boxes.loc[[box_pred.Index], "VolumeSlices"].iloc[0] / 4
        tp_boxes = [
            b
            for b in df_boxes_view.itertuples()
            if _is_tp(box_pred, b, slice_offset=view_slice_offset)
        ]
        if len(tp_boxes) > 1:
            # find the nearest GT box
            tp_distances = [_distance(box_pred, b) for b in tp_boxes]
            tp_boxes = [tp_boxes[np.argmin(tp_distances)]]
        if len(tp_boxes) > 0:
            tp_i = tp_boxes[0].index
            df_pred.loc[df_pred["index"] == box_pred.index, ("TP", "GTID")] = (1, tp_i)
            thresholds.append(box_pred.Score)

    thresholds.append(df_pred["Score"].min() - 1.0)

    # compute sensitivity at 2 FPs/volume on all cases
    evaluation_fps_all = (2.0,)
    tpr_all, _ = _froc(
        df_pred=df_pred,
        thresholds=thresholds,
        n_volumes=len(df_labels),
        n_boxes=len(df_boxes),
        evaluation_fps=evaluation_fps_all,
        return_full_curve=False,
    )
    result = {f"sensitivity_at_2_fps_all": tpr_all[0]}

    # compute mean sensitivity at 1, 2, 3, 4 FPs/volume on positive cases
    df_pred = df_pred[df_pred.index.isin(df_boxes.index)]
    df_labels = df_labels[df_labels.index.isin(df_boxes.index)]
    evaluation_fps_positive = (1.0, 2.0, 3.0, 4.0)
    tpr_positive, froc_curve_data = _froc(
        df_pred=df_pred,
        thresholds=thresholds,
        n_volumes=len(df_labels),
        n_boxes=len(df_boxes),
        evaluation_fps=evaluation_fps_positive,
        return_full_curve=return_froc_curve,
    )

    result.update(
        dict(
            (f"sensitivity_at_{int(x)}_fps_positive", y)
            for x, y in zip(evaluation_fps_positive, tpr_positive)
        )
    )
    result.update({"mean_sensitivity_positive": np.mean(tpr_positive)})

    if return_froc_curve:
        return result, froc_curve_data
    else:
        return result, ([], [])


def _froc(
    df_pred: pd.DataFrame,
    thresholds: List[float],
    n_volumes: int,
    n_boxes: int,
    evaluation_fps: tuple,
    return_full_curve: bool = False,
) -> Tuple[List[float], Tuple[List[float], List[float]]]:
    tpr = []
    fps = []
    for th in sorted(thresholds, reverse=True):
        df_th = df_pred.loc[df_pred["Score"] >= th]
        df_th_unique_tp = df_th.reset_index().drop_duplicates(
            subset=["StudyUID", "View", "TP", "GTID"]
        )
        n_tps_th = float(sum(df_th_unique_tp["TP"]))
        tpr_th = n_tps_th / n_boxes
        n_fps_th = float(len(df_th[df_th["TP"] == 0]))
        fps_th = n_fps_th / n_volumes
        tpr.append(tpr_th)
        fps.append(fps_th)
        if fps_th > max(evaluation_fps):
            break
    
    interpolated_tpr = [np.interp(x, fps, tpr) for x in evaluation_fps]
    
    if return_full_curve:
        return interpolated_tpr, (fps, tpr)
    else:
        return interpolated_tpr, ([], [])


def _is_tp(
    box_pred: NamedTuple, box_true: NamedTuple, slice_offset: int, min_dist: int = 24
) -> bool:
    pred_y = box_pred.Y + box_pred.Height / 2
    pred_x = box_pred.X + box_pred.Width / 2
    pred_z = box_pred.Z + box_pred.Depth / 2
    true_y = box_true.Y + box_true.Height / 2
    true_x = box_true.X + box_true.Width / 2
    true_z = box_true.Slice
    # 2D distance between true and predicted center points
    dist = np.linalg.norm((pred_x - true_x, pred_y - true_y))
    # compute radius based on true box size
    dist_threshold = np.sqrt(box_true.Width ** 2 + box_true.Height ** 2) / 2.0
    dist_threshold = max(dist_threshold, min_dist)
    slice_diff = np.abs(pred_z - true_z)
    # TP if predicted center within radius and slice within slice offset
    return dist <= dist_threshold and slice_diff <= slice_offset


def _distance(box_pred: NamedTuple, box_true: NamedTuple) -> float:
    pred_y = box_pred.Y + box_pred.Height / 2
    pred_x = box_pred.X + box_pred.Width / 2
    pred_z = box_pred.Z + box_pred.Depth / 2
    true_y = box_true.Y + box_true.Height / 2
    true_x = box_true.X + box_true.Width / 2
    true_z = box_true.Slice
    return np.linalg.norm((pred_x - true_x, pred_y - true_y, pred_z - true_z))


def plot_froc_curve(fps: List[float], tpr: List[float], output_dir: str, filename_prefix: str = "froc_curve") -> None:
    """
    Plot and save FROC curve as SVG.
    
    Args:
        fps: False positives per volume
        tpr: True positive rate (sensitivity)
        output_dir: Directory to save the plot
        filename_prefix: Prefix for the output filename
    """
    plt.figure(figsize=(10, 8))
    plt.plot(fps, tpr, 'b-', linewidth=2, label='FROC Curve')
    
    # Add markers at evaluation points
    evaluation_fps = [1.0, 2.0, 3.0, 4.0]
    evaluation_tpr = [np.interp(x, fps, tpr) for x in evaluation_fps]
    
    for i, (fp, tp) in enumerate(zip(evaluation_fps, evaluation_tpr)):
        plt.plot(fp, tp, 'ro', markersize=8)
        plt.annotate(f'{int(fp)} FP/vol\n{tp:.3f}', 
                    xy=(fp, tp), xytext=(10, 10), 
                    textcoords='offset points', 
                    fontsize=10, ha='left',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    plt.xlabel('False Positives per Volume', fontsize=14)
    plt.ylabel('Sensitivity (True Positive Rate)', fontsize=14)
    plt.title('Free-Response Operating Characteristic (FROC) Curve', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Set axis limits
    plt.xlim(0, max(5.0, max(fps) if fps else 5.0))
    plt.ylim(0, 1.05)
    
    # Save as SVG
    svg_path = os.path.join(output_dir, f"{filename_prefix}.svg")
    plt.savefig(svg_path, format='svg', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    logger = logging.getLogger(__name__)
    logger.info(f"FROC curve plot saved to: {svg_path}")


def save_froc_data_csv(fps: List[float], tpr: List[float], output_dir: str, filename_prefix: str = "froc_curve_data") -> None:
    """
    Save FROC curve plotting data as CSV - contains all points needed to recreate the curve.
    
    Args:
        fps: False positives per volume (complete curve data)
        tpr: True positive rate/sensitivity (complete curve data)
        output_dir: Directory to save the CSV
        filename_prefix: Prefix for the output filename
    """
    froc_df = pd.DataFrame({
        'fps_per_volume': fps,
        'sensitivity': tpr
    })
    
    # Sort by fps for clean plotting
    froc_df = froc_df.sort_values('fps_per_volume').reset_index(drop=True)
    
    csv_path = os.path.join(output_dir, f"{filename_prefix}.csv")
    froc_df.to_csv(csv_path, index=False)
    
    logger = logging.getLogger(__name__)
    logger.info(f"FROC curve plotting data saved to: {csv_path}")
    logger.info(f"  - {len(froc_df)} data points for complete curve reconstruction")


def save_froc_curve_results(fps: List[float], tpr: List[float], output_dir: str) -> None:
    """
    Save both FROC curve plot and data.
    
    Args:
        fps: False positives per volume
        tpr: True positive rate (sensitivity) 
        output_dir: Directory to save outputs
    """
    if not fps or not tpr:
        logger = logging.getLogger(__name__)
        logger.warning("No FROC curve data available to save")
        return
    
    plot_froc_curve(fps, tpr, output_dir)
    save_froc_data_csv(fps, tpr, output_dir)


def load_predictions_from_folder(folder_path: str, split: str) -> pd.DataFrame:
    """
    Load predictions from a folder structure.
    
    Args:
        folder_path: Path to the folder containing predictions
        split: Either 'val' or 'test' to specify which split to load
        
    Returns:
        DataFrame with predictions
    """
    if split == 'val':
        csv_path = os.path.join(folder_path, 'predictions_val', 'val_predictions.csv')
    elif split == 'test':
        csv_path = os.path.join(folder_path, 'predictions_test', 'test_predictions.csv')
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'val' or 'test'")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Predictions file not found: {csv_path}")
    
    return pd.read_csv(csv_path)


def _evaluate_parameter_combination(params_and_data):
    """
    Helper function to evaluate a single parameter combination.
    Used for multiprocessing.
    
    Args:
        params_and_data: Tuple of (score_th, iosib_th, df_preds_val_pickle, df_boxes_pickle)
        
    Returns:
        Tuple of (score_th, iosib_th, mean_sens)
    """
    score_th, iosib_th, df_preds_val, df_boxes_global = params_and_data
    
    # Generate 3D candidates with current parameters (no debug prints during grid search)
    df_3d = combine_predictions_into_3d_candidates(
        df_preds_val,
        score_threshold=score_th,
        iosib_threshold=iosib_th,
        slice_proximity_factor=0.5,
        cum_prob_threshold=0.95,
        k_max=8,
        debug=False
    )
    
    if len(df_3d) == 0:
        mean_sens = 0.0
    else:
        # Convert to challenge format and evaluate
        df_preds_challenge = convert_predictions_to_challenge_format(df_3d)
        df_boxes_filtered = filter_boxes_to_prediction_volumes(df_boxes_global, df_preds_val)
        df_labels = df_boxes_filtered[['StudyUID', 'View']].drop_duplicates()
        df_labels['Label'] = 1
        
        if len(df_boxes_filtered) == 0:
            mean_sens = 0.0
        else:
            res, _ = evaluate(df_labels, df_boxes_filtered, df_preds_challenge, return_froc_curve=False)
            mean_sens = res['mean_sensitivity_positive']
    
    return (score_th, iosib_th, mean_sens)


def grid_search_on_validation_set(df_preds_val: pd.DataFrame, n_processes: int = None) -> Tuple[float, float]:
    """
    Perform parallel grid search to find best parameters on validation set.
    
    Args:
        df_preds_val: Validation predictions DataFrame
        n_processes: Number of parallel processes to use (default: CPU count)
        
    Returns:
        Tuple of (best_score_threshold, best_iosib_threshold)
    """
    logger = logging.getLogger(__name__)
    
    logger.info("\n" + "="*80)
    logger.info("STAGE 1: PARALLEL GRID SEARCH ON VALIDATION SET")
    logger.info("="*80)
    
    # Grid Parameters
    score_thresholds = [0.75, 0.85, 0.95]
    iosib_thresholds = [0.6, 0.70, 0.80]
    
    # Create all parameter combinations
    param_combinations = [(score_th, iosib_th) for score_th in score_thresholds 
                         for iosib_th in iosib_thresholds]
    
    logger.info(f"Testing {len(param_combinations)} parameter combinations...")
    logger.info(f"Score thresholds: {score_thresholds}")
    logger.info(f"IoSIB thresholds: {iosib_thresholds}")
    
    # Determine number of processes
    if n_processes is None:
        n_processes = min(mp.cpu_count(), len(param_combinations))
    
    logger.info(f"Using {n_processes} parallel processes")
    
    # Prepare data for parallel processing
    # Each worker needs the dataframes
    params_and_data = [(score_th, iosib_th, df_preds_val, df_boxes) 
                       for score_th, iosib_th in param_combinations]
    
    logger.info("\nRunning parallel grid search...")
    logger.info(f"{'Progress':<10} {'Score Thresh':<12} {'IoSIB Thresh':<12} {'Mean Sensitivity':<18}")
    logger.info("-" * 55)
    
    # Run parallel evaluation
    with mp.Pool(processes=n_processes) as pool:
        results = pool.map(_evaluate_parameter_combination, params_and_data)
    
    # Process results and find best
    best_sens = -1
    best_setting = None
    all_results = []
    
    for i, (score_th, iosib_th, mean_sens) in enumerate(results):
        all_results.append((score_th, iosib_th, mean_sens))
        progress = f"{i+1}/{len(results)}"
        logger.info(f"{progress:<10} {score_th:<12.2f} {iosib_th:<12.2f} {mean_sens:<18.4f}")
        
        if mean_sens > best_sens:
            best_sens = mean_sens
            best_setting = (score_th, iosib_th)
    
    logger.info(f"\nGrid search completed!")
    logger.info(f"Best parameters found:")
    logger.info(f"  Score threshold: {best_setting[0]}")
    logger.info(f"  IoSIB threshold: {best_setting[1]}")
    logger.info(f"  Mean sensitivity: {best_sens:.4f}")
    
    # Show top 5 parameter combinations
    sorted_results = sorted(all_results, key=lambda x: x[2], reverse=True)
    logger.info(f"\nTop 5 parameter combinations:")
    logger.info(f"{'Rank':<5} {'Score Thresh':<12} {'IoSIB Thresh':<12} {'Mean Sensitivity':<18}")
    logger.info("-" * 50)
    for i, (score_th, iosib_th, mean_sens) in enumerate(sorted_results[:5]):
        logger.info(f"{i+1:<5} {score_th:<12.2f} {iosib_th:<12.2f} {mean_sens:<18.4f}")
    
    return best_setting


def evaluate_on_test_set(df_preds_test: pd.DataFrame, score_threshold: float, iosib_threshold: float, debug: bool = False) -> Tuple[Dict[str, float], Tuple[List[float], List[float]], pd.DataFrame]:
    """
    Evaluate on test set using the best parameters found on validation set.
    
    Args:
        df_preds_test: Test predictions DataFrame
        score_threshold: Best score threshold from validation
        iosib_threshold: Best IoSIB threshold from validation
        debug: If True, print detailed processing information
        
    Returns:
        Tuple of (evaluation results dict, froc curve data, final predictions dataframe)
    """
    logger = logging.getLogger(__name__)
    
    logger.info("\n" + "="*80)
    logger.info("STAGE 2: EVALUATION ON TEST SET")
    logger.info("="*80)
    
    logger.info(f"Using parameters from validation grid search:")
    logger.info(f"  Score threshold: {score_threshold}")
    logger.info(f"  IoSIB threshold: {iosib_threshold}")
    
    # Generate 3D candidates with best parameters
    df_3d = combine_predictions_into_3d_candidates(
        df_preds_test,
        score_threshold=score_threshold,
        iosib_threshold=iosib_threshold,
        slice_proximity_factor=0.5,
        cum_prob_threshold=0.95,
        k_max=8,
        debug=debug
    )
    
    if len(df_3d) == 0:
        logger.warning("No 3D candidates generated on test set!")
        return {}
    
    # Convert to challenge format and evaluate
    df_preds_challenge = convert_predictions_to_challenge_format(df_3d)
    df_boxes_filtered = filter_boxes_to_prediction_volumes(df_boxes, df_preds_test)
    df_labels = df_boxes_filtered[['StudyUID', 'View']].drop_duplicates()
    df_labels['Label'] = 1
    
    logger.info(f"\nEvaluating on test set:")
    logger.info(f"  - {len(df_labels)} volumes with ground truth")
    logger.info(f"  - {len(df_boxes_filtered)} ground truth boxes")
    logger.info(f"  - {len(df_preds_challenge)} 3D candidate predictions")
    
    # Run evaluation with FROC curve data
    results, froc_curve_data = evaluate(df_labels, df_boxes_filtered, df_preds_challenge, return_froc_curve=True)
    
    # Print results
    logger.info("\n" + "="*50)
    logger.info("TEST SET EVALUATION RESULTS")
    logger.info("="*50)
    
    # Print sensitivity metrics for positive cases
    logger.info("\nSensitivity on Positive Cases:")
    logger.info(f"  At 1 FP/volume:  {results['sensitivity_at_1_fps_positive']:.4f}")
    logger.info(f"  At 2 FPs/volume: {results['sensitivity_at_2_fps_positive']:.4f}")
    logger.info(f"  At 3 FPs/volume: {results['sensitivity_at_3_fps_positive']:.4f}")
    logger.info(f"  At 4 FPs/volume: {results['sensitivity_at_4_fps_positive']:.4f}")
    logger.info(f"  Mean Sensitivity: {results['mean_sensitivity_positive']:.4f}")
    
    # Print sensitivity at 2 FPs for all cases
    logger.info(f"\nSensitivity at 2 FPs/volume (All Cases): {results['sensitivity_at_2_fps_all']:.4f}")
    
    return results, froc_curve_data, df_preds_challenge

@tracked(directory_parameter="output_dir")
def main(predictions_folder: str, output_dir: str, n_processes: int = None, debug: bool = False):
    """
    Main function to run the 2-stage evaluation process.
    
    Args:
        predictions_folder: Path to folder containing predictions_val/ and predictions_test/ subdirectories
        output_dir: Directory to save log files and outputs
        n_processes: Number of parallel processes to use for grid search
        debug: If True, print detailed processing information
    """
    # Setup logging first
    logger = setup_logging(output_dir, debug)
    
    logger.info("Starting 2-stage evaluation process...")
    logger.info(f"Predictions folder: {predictions_folder}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Ground truth boxes: {len(df_boxes)} boxes loaded")
    
    # Stage 1: Load validation predictions and run grid search
    logger.info("\nLoading validation predictions...")
    df_preds_val = load_predictions_from_folder(predictions_folder, 'val')
    logger.info(f"Loaded {len(df_preds_val)} validation predictions")
    
    best_score_threshold, best_iosib_threshold = grid_search_on_validation_set(df_preds_val, n_processes)
    
    # Stage 2: Load test predictions and evaluate with best parameters
    logger.info("\nLoading test predictions...")
    df_preds_test = load_predictions_from_folder(predictions_folder, 'test')
    logger.info(f"Loaded {len(df_preds_test)} test predictions")
    
    test_results, froc_curve_data, df_preds_challenge = evaluate_on_test_set(df_preds_test, best_score_threshold, best_iosib_threshold, debug)
    
    # Save FROC curve plot and data
    fps, tpr = froc_curve_data
    if fps and tpr:
        logger.info("\nSaving FROC curve results...")
        save_froc_curve_results(fps, tpr, output_dir)
    
    # Save results to JSON file for reproducibility
    results_dict = {
        'validation_best_parameters': {
            'score_threshold': best_score_threshold,
            'iosib_threshold': best_iosib_threshold
        },
        'test_results': test_results,
        'experiment_metadata': {
            'predictions_folder': predictions_folder,
            'n_validation_predictions': len(df_preds_val),
            'n_test_predictions': len(df_preds_test),
            'n_ground_truth_boxes': len(df_boxes),
            'debug_mode': debug
        }
    }
    
    results_file = os.path.join(output_dir, 'evaluation_results.json')
    with open(results_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    logger.info(f"\nResults saved to: {results_file}")
    
    # Save final predictions and ground truth boxes for later visualization
    logger.info("\nSaving data for visualization...")
    
    # Save final aggregated predictions (3D candidates in challenge format)
    final_preds_file = os.path.join(output_dir, 'final_predictions_3d_candidates.csv')
    df_preds_challenge.to_csv(final_preds_file, index=False)
    logger.info(f"Final 3D candidate predictions saved to: {final_preds_file}")
    
    # Save full ground truth boxes
    gt_boxes_file = os.path.join(output_dir, 'ground_truth_boxes.csv')
    df_boxes.to_csv(gt_boxes_file, index=False)
    logger.info(f"Ground truth boxes saved to: {gt_boxes_file}")
    
    # Also save filtered ground truth boxes (only for volumes with predictions)
    df_boxes_filtered = filter_boxes_to_prediction_volumes(df_boxes, df_preds_test)
    gt_boxes_filtered_file = os.path.join(output_dir, 'ground_truth_boxes_filtered.csv')
    df_boxes_filtered.to_csv(gt_boxes_filtered_file, index=False)
    logger.info(f"Filtered ground truth boxes saved to: {gt_boxes_filtered_file}")
    
    logger.info("\n" + "="*80)
    logger.info("2-STAGE EVALUATION COMPLETED SUCCESSFULLY!")
    logger.info("="*80)
    
    return test_results


if __name__ == "__main__":
    # Multiprocessing protection
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(
        description="2-stage evaluation: Grid search on validation set, then evaluate on test set using 3D candidates.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "predictions_folder",
        type=str,
        help="Path to folder containing predictions_val/ and predictions_test/ subdirectories with CSV files.",
    )

    parser.add_argument(
        "output_dir",
        type=str,
        help="Path to output directory.",
    )
    
    parser.add_argument(
        "--n_processes",
        type=int,
        default=6,
        help="Number of parallel processes to use for grid search (default: CPU count).",
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode to show detailed processing information.",
    )

    args = parser.parse_args()

    # Verify the predictions folder exists and has the required structure
    if not os.path.exists(args.predictions_folder):
        raise FileNotFoundError(f"Predictions folder not found: {args.predictions_folder}")
    
    val_folder = os.path.join(args.predictions_folder, 'predictions_val')
    test_folder = os.path.join(args.predictions_folder, 'predictions_test')
    
    if not os.path.exists(val_folder):
        raise FileNotFoundError(f"Validation predictions folder not found: {val_folder}")
    
    if not os.path.exists(test_folder):
        raise FileNotFoundError(f"Test predictions folder not found: {test_folder}")
    
    # Run the main evaluation
    main(args.predictions_folder, Path(args.output_dir), args.n_processes, args.debug)