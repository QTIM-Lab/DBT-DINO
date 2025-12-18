import json
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from lifelines import KaplanMeierFitter
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
    roc_curve,
)


def evaluate_model(trainer, model, test_loader, output_dir):
    """
    Evaluate the model and save metrics to a JSON file.
    """
    logging.info("Starting model evaluation...")

    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run test set evaluation
    test_results = trainer.test(model, test_loader)[0]

    # Log results
    logging.info("Test results summary:")
    logging.info(f"  Accuracy: {test_results.get('test_acc', 'N/A')}")
    logging.info(f"  F1 Score: {test_results.get('test_f1', 'N/A')}")
    logging.info(f"  AUROC: {test_results.get('test_auroc', 'N/A')}")

    return test_results


def evaluate_model_risk(trainer, model, test_loader, output_dir):
    breakpoint()
    test_results = trainer.test(model, test_loader)[0]
    breakpoint()
    return test_results



def save_test_outputs(predictions_df, confusion_matrix, metrics, output_dir):
    """
    Save test outputs to files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save predictions
    pred_path = output_dir / "predictions.csv"
    predictions_df.to_csv(pred_path, index=False)
    logging.info(f"Saved predictions to {pred_path}")

    # Save confusion matrix
    cm_path = output_dir / "confusion_matrix.npy"
    np.save(cm_path, confusion_matrix)
    cm_csv_path = output_dir / "confusion_matrix.csv"
    pd.DataFrame(confusion_matrix).to_csv(cm_csv_path)
    logging.info(f"Saved confusion matrix to {cm_path} and {cm_csv_path}")

    # Save metrics
    metrics_path = output_dir / "test_metrics.json"
    with open(metrics_path, "w") as f:
        metrics_json = {
            k: float(v) if isinstance(v, (np.floating, np.integer)) else v
            for k, v in metrics.items()
        }
        json.dump(metrics_json, f, indent=4)
    logging.info(f"Saved metrics to {metrics_path}")


def save_test_outputs_risk(metrics, output_dir):
    # Save metrics
    metrics_path = output_dir / "test_metrics.json"
    with open(metrics_path, "w") as f:
        metrics_json = {
            k: float(v) if isinstance(v, (np.floating, np.integer)) else v
            for k, v in metrics.items()
        }
        json.dump(metrics_json, f, indent=4)
    logging.info(f"Saved metrics to {metrics_path}")


def process_test_outputs(preds_np, labels_np, probs_np, all_patient_ids, output_dir):
    """
    Process and save test outputs.

    Args:
        preds_np: Numpy array of class predictions (shape: [n_samples])
        labels_np: Numpy array of true labels (shape: [n_samples])
        probs_np: Numpy array of prediction probabilities (shape: [n_samples, n_classes])
        all_patient_ids: List of patient IDs
        output_dir: Directory to save the outputs
    """
    # Compute metrics
    metrics = {}

    # Compute AUROC using probabilities
    auroc = roc_auc_score(labels_np, probs_np, multi_class="ovr")
    metrics["test_auroc"] = auroc

    # Generate classification report using class predictions
    class_report = classification_report(labels_np, preds_np, output_dict=True)
    metrics["classification_report"] = class_report

    # Compute confusion matrix using class predictions
    cm = confusion_matrix(labels_np, preds_np)

    # Create predictions DataFrame
    predictions_df = pd.DataFrame(
        {
            "patient_id": all_patient_ids,
            "true_label": labels_np,
            "predicted_label": preds_np,
        }
    )
    # Add probability columns for each class
    for i in range(probs_np.shape[1]):
        predictions_df[f"prob_class_{i}"] = probs_np[:, i]

    # Save all outputs
    save_test_outputs(predictions_df, cm, metrics, output_dir)

    # Save raw predictions to numpy files
    output_dir = Path(output_dir)
    np.save(output_dir / "raw_predictions.npy", preds_np)
    np.save(output_dir / "raw_probabilities.npy", probs_np)
    np.save(output_dir / "raw_labels.npy", labels_np)
    
    # Also save as CSV for easier viewing
    raw_predictions_df = pd.DataFrame({
        'patient_id': all_patient_ids,
        'true_label': labels_np,
        'predicted_label': preds_np,
        **{f'prob_class_{i}': probs_np[:, i] for i in range(probs_np.shape[1])}
    })
    raw_predictions_df.to_csv(output_dir / "raw_predictions.csv", index=False)

    return metrics


def process_test_outputs_risk(labels_np, preds_np, all_patient_ids, all_study_ids, output_dir):

    # Compute metrics
    metrics = {}

    # Compute MSE and MAE
    mse = mean_squared_error(labels_np, preds_np)
    mae = mean_absolute_error(labels_np, preds_np)

    # Compute AUC ROC per year
    auroc_per_year = [roc_auc_score(labels_np[:, i], preds_np[:, i]) for i in range(labels_np.shape[1])]

    # For each year, compute the f1 score and accuracy
    for i in range(labels_np.shape[1]):
        f1_score = np.mean(2 * np.sum(labels_np[:, i] * preds_np[:, i]) / np.sum(labels_np[:, i] + preds_np[:, i]))
        accuracy = np.mean(np.equal(labels_np[:, i], preds_np[:, i]))
        metrics[f"test_f1_score_year_{i}"] = f1_score
        metrics[f"test_accuracy_year_{i}"] = accuracy

    # Threshold predictions at 0.5 and compute accuracy, F1 score, true positive rate, true negative rate, false positive rate, false negative rate
    preds_thres = (preds_np > 0.5).astype(int)
    accuracy = np.mean(np.equal(labels_np, preds_thres))
    precision = np.sum(labels_np * preds_thres) / np.sum(preds_thres)
    recall = np.sum(labels_np * preds_thres) / np.sum(labels_np)
    f1_score = np.mean(2 * np.sum(labels_np * preds_thres, axis=0) / np.sum(labels_np + preds_thres, axis=0))
    true_positive_rate = np.sum(labels_np * preds_thres) / np.sum(labels_np)
    true_negative_rate = np.sum((1 - labels_np) * (1 - preds_thres)) / np.sum(1 - labels_np)
    false_positive_rate = np.sum((1 - labels_np) * preds_thres) / np.sum(1 - labels_np)
    false_negative_rate = np.sum(labels_np * (1 - preds_thres)) / np.sum(labels_np)

    # Add to metrics dict
    metrics["test_mse"] = mse
    metrics["test_mae"] = mae
    metrics["test_auroc_per_year"] = auroc_per_year
    metrics["test_accuracy"] = accuracy
    metrics["test_precision"] = precision
    metrics["test_recall"] = recall
    metrics["test_f1_score"] = f1_score
    metrics["test_TPR"] = true_positive_rate
    metrics["test_TNR"] = true_negative_rate
    metrics["test_FPR"] = false_positive_rate
    metrics["test_FNR"] = false_negative_rate


    # Save all outputs
    save_test_outputs_risk(metrics, output_dir)

    # Compute confusion matrix for each year and save heatmaps
    compute_cm_heatmap(labels_np, preds_np, output_dir)

    return metrics


def compute_cm_heatmap(labels_np, preds_np, output_dir, threshold=0.5):
    """
    Compute confusion matrix for each year individually and save heatmap.
    """

    # Define figure
    fig, axes = plt.subplots(1, labels_np.shape[1], figsize=(10 * labels_np.shape[1], 10))

    for i in range(labels_np.shape[1]):
        # Binarize predictions
        preds_np[:, i] = (preds_np[:, i] > threshold).astype(int)
        # Compute confusion matrix
        cm = confusion_matrix(labels_np[:, i], preds_np[:, i])
        # Plot heatmap
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i])
        axes[i].set_title(f"Year {i + 1}")
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("True")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "yearly_confusion_matrices_heatmap.png"))


def compute_auroc_per_year(y_label, y_pred, y_mask):
    """
    Compute AUROC for each year while defining negative cases as samples that are negative across all five years.
    """

    # Identify negatives across all years
    neg_mask = np.all(y_label == 0, axis=1)
    neg_labels = y_label[neg_mask]

    auroc_per_year = {
        'year1': None,
        'year2': None,
        'year3': None,
        'year4': None,
        'year5': None
    }
    avg_auroc = []

    for yr in range(5):
        # select positive cases for current year
        pos_mask = y_label[:, yr] == 1
        pos_labels = y_label[pos_mask]
        valid_idx = pos_mask | neg_mask

        # exclude years with missing information
        missing_data_mask = y_mask[:, yr] == 0

        # Remove missing data & keep valid entries
        valid_idx = valid_idx & ~missing_data_mask

        y_valid = y_label[valid_idx, yr]
        y_pred_valid = y_pred[valid_idx, yr]

        # Convert to tensor
        y_valid = torch.tensor(y_valid, dtype=torch.float64)
        y_pred_valid = torch.tensor(y_pred_valid, dtype=torch.float64)

        auroc_value = roc_auc_score(y_valid, y_pred_valid)
        print(f"AUROC year {yr+1}: {auroc_value:.4f}")
        auroc_per_year[f'year{yr+1}'] = auroc_value

        avg_auroc.append(auroc_value)

        # compute fpr and tpr
        fpr, tpr, thresholds = roc_curve(y_valid, y_pred_valid)

        # Plot ROC curve for yr
        plt.figure()
        plt.plot(fpr, tpr, label=f'AUROC = {auroc_value:.2f}')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve Year {yr+1}')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.savefig(f"path/to/test_roc.png")
        plt.close()

    avg_auroc = np.mean(avg_auroc)

    return avg_auroc, auroc_per_year

def plot_kaplan_meier(data):
    """
    Plot Kaplan-Meier survival curves.
    """
    # Get cancer
    pass

def plot_hist(year, y_label, y_pred, y_mask):
    """
    Plot histogram of predictions.
    """
    # Select valid predictions and labels
    mask = y_mask[:, year] == 1
    preds = y_pred[:, year][mask]
    labels = y_label[:, year][mask]

    # Split predictions based on labels
    preds_pos = preds[labels == 1]
    preds_neg = preds[labels == 0]

    # Plot histograms
    plt.figure(figsize=(8, 5))
    plt.hist(preds_neg, bins=30, alpha=0.7, label='Negative (label=0)', color='skyblue', density=True)
    plt.hist(preds_pos, bins=30, alpha=0.7, label='Positive (label=1)', color='salmon', density=True)
    plt.axvline(preds_pos.mean(), color='red', linestyle='--', label=f'Pos mean: {preds_pos.mean():.2f}')
    plt.axvline(preds_neg.mean(), color='blue', linestyle='--', label=f'Neg mean: {preds_neg.mean():.2f}')
    plt.xlabel('Predicted Score')
    plt.ylabel('Density')
    plt.title(f'Prediction Histogram (Year {year})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"path/to/prediction_hist_{year+1}_balanced.png")


def plot_km_curves(preds, labels, masks, output_dir):
    labels_list = labels.cpu().numpy().tolist()
    masks_list = masks.cpu().numpy().tolist()

    event_time = []
    event_observed = []

    for label, mask in zip(labels_list, masks_list):
        if 1 in label:
            event_time.append(np.argmax(label))
            event_observed.append(1)
        else:
            event_time.append(sum(mask))  # time until censoring
            event_observed.append(0)

    # Get prediction score (e.g. at year 5 or mean over 5 years)
    pred_scores = preds[:, -1].cpu().numpy()  # or preds.mean(dim=1).cpu().numpy()

    # Choose threshold to split predictions (e.g. median)
    threshold = np.median(pred_scores)
    is_high_risk = pred_scores >= threshold

    kmf = KaplanMeierFitter()

    # Convert to np arrays
    event_time = np.array(event_time)
    event_observed = np.array(event_observed)

    kmf.fit(event_time[is_high_risk], event_observed[is_high_risk], label='High Risk (Predicted Positive)')
    kmf.plot_survival_function(label='High Risk Group', ci_show=False)

    kmf.fit(event_time[~is_high_risk], event_observed[~is_high_risk], label='Low Risk (Predicted Negative)')
    kmf.plot_survival_function(label='Low Risk Group', ci_show=False)

    plt.title('Survival analysis based on model predictions')
    plt.xlabel('Years to Cancer')
    plt.ylabel('Survival Probability')

    plt.savefig(os.path.join(output_dir, 'best_model_pred_km_curve.png'))


def get_data(csv_path):
    preds_df = pd.read_csv(csv_path)

    labels = preds_df['labels'].values
    masks = preds_df['masks'].values
    preds = preds_df['preds'].values

    # Convert the labels and masks to numpy arrays
    labels = np.array([np.array(eval(label)) for label in labels])
    masks = np.array([np.array(eval(mask)) for mask in masks])
    preds = np.array([np.array(eval(pred)) for pred in preds])

    return preds, labels, masks