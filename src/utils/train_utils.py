import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import WeightedRandomSampler


def overall_risk_loss(pred, batch, breast_specific=False, logits=False, cumulative_prob='none', batch_loss_avg=False):
    """
    Compute binary cross entropy loss for overall risk prediction task. The loss is computed for the patient specific risk.
    """
    # Convert from string to PyTorch tensor
    y_label = torch.tensor([[int(label)] for label in batch['label']], dtype=torch.float32, device=pred.device)

    # Compute bce loss with logits
    if logits:
        loss = F.binary_cross_entropy_with_logits(pred, y_label, reduction='mean')
    else:
        loss = F.binary_cross_entropy(pred, y_label, reduction='mean')

    return loss

def risk_pred_loss(pred, batch, breast_specific=False, clamp=False, batch_loss_avg=False, use_bce_with_logits=False):
    """
    Compute binary cross entropy loss with masking for missing data. The loss is computed for the patient specific risk
    prediction model or breast specific risk prediction model.
    The function binary_cross_entropy_with_logits expects raw logits as input and internally applied sigmoid function before
    computing binary cross-entropy loss.
    :param logit: Raw logits from the model
    :param batch: Batch of data
    :param args: Input arguments
    """

    loss = 0

    if breast_specific:
        sides = ['l', 'r']
        for side in sides:
            label, mask = batch['label_{}'.format(side)], batch['mask_{}'.format(side)]

            # Convert string list to a NumPy array
            y_label_np = np.array([np.fromstring(x.strip("[]"), sep=" ") for x in label], dtype=np.float64)
            y_mask_np = np.array([np.fromstring(x.strip("[]"), sep=" ") for x in mask], dtype=np.float64)

            # Convert to PyTorch tensors
            y_label = torch.tensor(y_label_np, dtype=torch.float32, device=pred[side].device)
            y_mask = torch.tensor(y_mask_np, dtype=torch.float32, device=pred[side].device)

            if use_bce_with_logits:
                # Compute binary cross-entropy loss with logits
                loss += F.binary_cross_entropy_with_logits(pred[side], y_label, weight=y_mask,
                                                           size_average=False) / torch.sum(y_mask)
            else:
                # Compute masked binary cross-entropy loss
                loss += F.binary_cross_entropy(pred[side], y_label, weight=y_mask,
                                                           size_average=False) / torch.sum(y_mask)

    else:
        # Compute loss for patient specific risk prediction model
        label, mask = batch['label'], batch['mask']

        # Convert string list to a NumPy array
        y_label_np = np.array([np.fromstring(x.strip("[]"), sep=" ") for x in label], dtype=np.float32)
        y_mask_np = np.array([np.fromstring(x.strip("[]"), sep=" ") for x in mask], dtype=np.float32)

        # Convert to PyTorch tensors
        y_label = torch.tensor(y_label_np, dtype=torch.float32, device=pred.device)
        y_mask = torch.tensor(y_mask_np, dtype=torch.float32, device=pred.device)

        if clamp:
            # Clamp the predictions to avoid log(0) in binary cross-entropy loss
            eps = 1e-7
            pred = torch.clamp(pred, min=eps, max=1-eps)

        if use_bce_with_logits:
            if batch_loss_avg:
                # Compute binary cross-entropy loss with logits
                loss = F.binary_cross_entropy_with_logits(pred, y_label, weight=y_mask, size_average=False) / torch.sum(y_mask) # average loss over all elements in the batch
            else:
                # Average loss over each batch element across all classes
                loss = F.binary_cross_entropy_with_logits(pred, y_label, weight=y_mask, reduction='none')
                loss = torch.mean(torch.sum(loss, dim=1) / torch.sum(y_mask, dim=1))
        else:
            #with torch.amp.autocast('cuda', enabled=False): # disable automatic mixed precision when computing loss
            if batch_loss_avg:
                loss = F.binary_cross_entropy(pred, y_label, weight=y_mask, size_average=False) / torch.sum(y_mask)
            else:
                # Average loss over each batch element across all classes
                loss = F.binary_cross_entropy(pred, y_label, weight=y_mask, reduction='none')
                loss = torch.mean(torch.sum(loss, dim=1) / torch.sum(y_mask, dim=1))

    return loss


def custom_bce_loss(pred: torch.Tensor, label: torch.Tensor, mask: torch.Tensor):
    """
    Compute binary cross entropy loss with masking for missing data. The predictions should be the cumulative risk scores
    with sigmoid activation applied.
    """

    def elementwise_bce_loss(pred: torch.Tensor, label: torch.Tensor):
        """
        Computes element-wise binary cross-entropy loss.

        Args:
            pred (torch.Tensor): Tensor of shape (batch_size, 5) with predicted probabilities.
            label (torch.Tensor): Tensor of shape (batch_size, 5) with binary labels (0 or 1).

        Returns:
            torch.Tensor: Element-wise binary cross-entropy loss of shape (batch_size, 5).
        """

        # Compute BCE loss element-wise
        loss = - (label * torch.log(pred) + (1 - label) * torch.log(1 - pred))

        return loss

    # Compute element-wise BCE loss for each year
    element_wise_loss = elementwise_bce_loss(pred, label)

    # Mask out missing data
    masked_loss = element_wise_loss * mask

    # Compute total loss
    total_loss = torch.sum(masked_loss) / torch.sum(mask)

    return total_loss


def extract_features_laterality(batch, embed_types, stats, breast_specific=False):

    x = {}

    assert batch is not None, "Batch is None"

    if breast_specific:

        # Loop over batch and extract the embeddings for the desired laterality
        for e in embed_types:
            x[e] = {}
            for s in stats:
                x[e][s] = batch['embedding'][e][s]

        x_rcc = torch.cat([x[e][s][:, :768] for e in embed_types for s in stats], dim=1)
        x_lcc = torch.cat([x[e][s][:, 768:(768*2)] for e in embed_types for s in stats], dim=1)
        x_rmlo = torch.cat([x[e][s][:, (768*2):(768*3)] for e in embed_types for s in stats], dim=1)
        x_lmlo = torch.cat([x[e][s][:, (768*3):(768*4)] for e in embed_types for s in stats], dim=1)

        x_r = torch.cat([x_rcc, x_rmlo], dim=1)
        x_l = torch.cat([x_lcc, x_lmlo], dim=1)

        print(f"Shape of Right input tensor: {x_r.shape}")
        print(f"Shape of Left input tensor: {x_l.shape}")

        y_r = batch["label_r"]
        y_l = batch["label_l"]

        return x_r, x_l, y_r, y_l

    else:
        # Extract patient-specific labels
        y = batch["label"]
        # Concatenate the embeddings for all statistics
        x = {e: torch.cat([batch["embedding"][e][s] for s in stats], dim=1) for e in embed_types}
        # Concatenate the embeddings for all embedding types
        x = torch.cat([x[e] for e in embed_types], dim=1)

        return x, y


def cumulative_max_risk(tensor):
    # Apply cumulative maximum
    #monotonic_tensor = torch.maximum(tensor, tensor.new_zeros(tensor.size(0)))
    monotonic_tensor = torch.cummax(tensor, dim=0)[0]

    return monotonic_tensor


def warmup_scheduler(optimizer, warmup_epochs, total_epochs):
       def lr_lambda(epoch):
           if epoch < warmup_epochs:
               return epoch / warmup_epochs  # Linear warmup
           return 0.5 * (1 + torch.cos(
               (epoch - warmup_epochs) / (total_epochs - warmup_epochs) * 3.1415926535))  # Cosine decay
       return LambdaLR(optimizer, lr_lambda)


def define_sampler(cancer_dataset, healthy_dataset, num_samples):
    # Assign labels for balancing: 0 for healthy, 1 for cancer (cohorts but not years to cancer < 5)
    labels = (
            [1] * len(cancer_dataset) +
            [0] * len(healthy_dataset)
    )

    num_cancer = len(cancer_dataset)
    num_healthy = len(healthy_dataset)

    weight_cancer = 1.0 / num_cancer
    weight_healthy = 1.0 / num_healthy

    weights = [weight_cancer if label == 1 else weight_healthy for label in labels]
    return WeightedRandomSampler(weights, num_samples=num_samples, replacement=True)



def compute_auroc_per_year(y_label, y_pred, y_mask, metrics, set):
    """
    Compute AUROC for each year while defining negative cases as samples that are negative across all five years.
    """

    # Convert tensors to np arrays
    y_label = y_label.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    y_mask = y_mask.cpu().numpy()

    # Identify negatives across all years
    neg_mask = np.all(y_label == 0, axis=1)
    neg_labels = y_label[neg_mask]

    avg_auroc = []

    auroc_per_year = {
        "year1": None,
        "year2": None,
        "year3": None,
        "year4": None,
        "year5": None
    }


    for yr in range(5):
        pos_mask = y_label[:, yr] == 1
        pos_labels = y_label[pos_mask]
        valid_idx = pos_mask | neg_mask

        missing_data_mask = y_mask[:, yr] == 0

        # Remove missing data
        valid_idx = valid_idx & ~missing_data_mask

        y_valid = y_label[valid_idx, yr]
        y_pred_valid = y_pred[valid_idx, yr]

        # Convert to tensor
        y_valid = torch.tensor(y_valid, dtype=torch.float64)
        y_pred_valid = torch.tensor(y_pred_valid, dtype=torch.float64)

        metrics[f"year{yr+1}_auroc"].update(y_pred_valid, y_valid)
        auroc_value = metrics[f"year{yr+1}_auroc"].compute()
        auroc_per_year[f"year{yr+1}"] = auroc_value

        # compute fpr and tpr
        fpr, tpr, thresholds = roc_curve(y_valid, y_pred_valid)

        # Plot ROC curve for yr
        plt.figure()
        plt.plot(fpr, tpr, label=f'AUROC = {auroc_value:.2f}')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve Year {yr + 1}')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.close()

        avg_auroc.append(auroc_value)

    avg_auroc = np.mean(avg_auroc)

    return avg_auroc, auroc_per_year

