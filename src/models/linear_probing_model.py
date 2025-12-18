import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.nn as nn
import torchmetrics


from src.utils.eval_utils import (
    process_test_outputs
)
from src.utils.train_utils import (
    compute_auroc_per_year,
    extract_features_laterality,
    overall_risk_loss,
    risk_pred_loss,
)


class LinearProbingModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        #input_dim = 3072 if args.embedding_type == "patch" else 768
        #input_dim = 8192 if args.model == "densenet_121" else 6144
        if args.model == "densenet_121":
            # Size of embeddings for densenet_121 is 1024, we concat 4 views
            base_dim = 4096
        else:
            # Size of embeddings for dbt_dino is 768, we concat 4 views
            base_dim = 3072
        input_dim = base_dim * len(args.stats)
        self.linear_classifier = nn.Linear(input_dim, args.nb_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        self.batch_size = args.batch_size
        if args.task == 'density':
            self.val_metrics = nn.ModuleDict(
                {
                    "acc": torchmetrics.Accuracy(
                        task="multiclass", num_classes=args.nb_classes
                    ),
                    "f1": torchmetrics.F1Score(
                        task="multiclass", num_classes=args.nb_classes
                    ),
                }
            )
        elif args.task == 'overall_risk':
            self.val_metrics = nn.ModuleDict(
                {
                    "acc": torchmetrics.Accuracy(
                        task="multiclass", num_classes=args.nb_classes
                    ),
                    "f1": torchmetrics.F1Score(
                        task="multiclass", num_classes=args.nb_classes
                    ),
                    "roc_auc": torchmetrics.classification.BinaryAUROC(
                    thresholds=None
                    ),
                }
            )
        # Add test metrics storage
        self.test_step_outputs = []

    def forward(self, x):
        return self.linear_classifier(x)

    def training_step(self, batch, batch_idx):
        embeddings, labels = batch["embedding"], batch["label"]
        outputs = self(embeddings)
        loss = self.loss_fn(outputs, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True, logger=True, batch_size=self.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        embeddings, labels = batch["embedding"], batch["label"]
        outputs = self(embeddings)
        loss = self.loss_fn(outputs, labels)
        self.log("val_loss", loss, on_step=True, on_epoch=True, logger=True, batch_size=self.batch_size)

        for metric_name, metric in self.val_metrics.items():
            metric(outputs, labels)

    def on_validation_epoch_end(self):
        for metric_name, metric in self.val_metrics.items():
            value = metric.compute()
            self.log(f"val_{metric_name}", value, on_epoch=True, batch_size=self.batch_size)
            metric.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.epochs
        )
        return [optimizer], [scheduler]

    def test_step(self, batch, batch_idx):
        embeddings, labels = batch["embedding"], batch["label"]
        outputs = self(embeddings)
        loss = self.loss_fn(outputs, labels)

        # Log metrics
        self.log("test_loss", loss, batch_size=self.batch_size)
        for metric_name, metric in self.val_metrics.items():
            metric(outputs, labels)

        # Store step outputs
        step_output = {
            "pred_probs": torch.softmax(outputs, dim=1),
            "preds": outputs.argmax(dim=1),
            "labels": labels,
            "patient_ids": batch["patient_id"],  # Store patient IDs for later analysis
        }
        self.test_step_outputs.append(step_output)
        return step_output

    def on_test_epoch_end(self):
        # Aggregate all predictions and labels
        all_preds = torch.cat([x["preds"] for x in self.test_step_outputs])
        all_labels = torch.cat([x["labels"] for x in self.test_step_outputs])
        all_probs = torch.cat([x["pred_probs"] for x in self.test_step_outputs])
        all_patient_ids = [
            id for x in self.test_step_outputs for id in x["patient_ids"]
        ]

        # Convert to numpy for easier handling
        preds_np = all_preds.cpu().float().numpy()
        labels_np = all_labels.cpu().float().numpy()
        probs_np = all_probs.cpu().float().numpy()

        # Process and save outputs
        metrics = process_test_outputs(
            preds_np, labels_np, probs_np, all_patient_ids, self.hparams.output_dir
        )

        # Log metrics
        for name, value in metrics.items():
            if name != "classification_report":
                self.log(name, value, batch_size=self.batch_size)

        # Clear the test step outputs
        self.test_step_outputs.clear()


class BaselineRiskModel(pl.LightningModule):
    def __init__(self, args, embed_types: list, stats: list):
        super().__init__()
        self.save_hyperparameters(args)
        #self.num_features = len(embed_types) * len(stats) * 3072
        self.num_features = 3072
        self.embed_types = embed_types
        self.stats = stats
        self.learning_rate = args.lr
        self.epochs = args.epochs
        self.weight_decay = args.weight_decay
        self.lr_scheduler = args.lr_scheduler
        self.batch_size = args.batch_size
        self.linear_classifier = nn.Linear(self.num_features, 1)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.val_metrics = nn.ModuleDict(
            {
                "acc": torchmetrics.Accuracy(
                    task="binary"
                ),
                "f1": torchmetrics.F1Score(
                    task="binary"
                ),
                "roc_auc": torchmetrics.classification.BinaryAUROC(
                    thresholds=None
                ),
            }
        )
        # Add test metrics storage
        self.test_step_outputs = []
        self.predictions_per_epoch = []  # Stores predictions per epoch
        self.all_epoch_predictions = {}  # Stores all epochs' data
        self.val_losses_per_epoch = []  # Stores validation losses per epoch
        self.boxplots = args.boxplots

    def forward(self, x):
        return self.linear_classifier(x)

    def training_step(self, batch, batch_idx):
        #embeddings, labels = extract_features_laterality(batch, self.embed_types, self.stats)
        embeddings, labels = batch["embedding"], batch["label"]

        # Convert labels from list of strings to tensor of size (B, 1)
        # TODO uncomment next line
        #labels = torch.tensor([[int(label)] for label in batch['label']], dtype=torch.float32, device=embeddings.device)

        # Get outputs
        outputs = self(embeddings)
        loss = self.loss_fn(outputs, labels)

        self.log("train_loss", loss, on_step=True, on_epoch=True, logger=True, batch_size=self.batch_size)

        print(f"Epoch {self.current_epoch} | Training Step {batch_idx + 1} / {self.trainer.num_training_batches} | Loss: {loss.item():.4f}")

        return loss

    def validation_step(self, batch, batch_idx):
        #embeddings, labels = extract_features_laterality(batch, self.embed_types, self.stats)
        embeddings, labels = batch["embedding"], batch["label"]

        # Convert labels from list of strings to tensor of size (B, 1)
        # TODO uncomment next line after debug
        #labels = torch.tensor([[int(label)] for label in batch['label']], dtype=torch.float32, device=embeddings.device)

        # Get outputs
        outputs = self(embeddings)
        probs = torch.sigmoid(outputs)
        loss = self.loss_fn(outputs, labels)

        print(f"Epoch {self.current_epoch} | Validation Step {batch_idx + 1} / {self.trainer.num_val_batches} | Loss: {loss.item():.4f}")

        self.log("val_loss", loss, on_step=True, on_epoch=True, logger=True, batch_size=self.batch_size)

        # Store validation losses for current epoch, all batches
        self.val_losses_per_epoch.append(loss)

        #if dataloader_idx == 0:
        #    self.log("val_loss_imbalanced", loss, on_step=False, on_epoch=True, logger=True, batch_size=self.batch_size)
        #else:
        #    self.log("val_loss_balanced", loss, on_step=False, on_epoch=True, logger=True, batch_size=self.batch_size)

        # Store predictions after sigmoid activation and labels for plotting
        self.predictions_per_epoch.append((probs.squeeze(1).cpu().detach(), labels.squeeze(1).cpu().detach()))

        for metric_name, metric in self.val_metrics.items():
            metric(outputs, labels) # sigmoid auto applied in BinaryAUROC

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.epochs
        )
        return [optimizer], [scheduler]

    def on_validation_epoch_end(self):
        for metric_name, metric in self.val_metrics.items():
            value = metric.compute()
            self.log(f"val_{metric_name}", value, on_epoch=True)
            metric.reset()

        # Compute average val loss over all batches in current epoch
        avg_val_loss_epoch = torch.stack(self.val_losses_per_epoch).mean()
        # Log avg val loss per epoch
        self.log("avg_val_loss_epoch", avg_val_loss_epoch)

        preds, labels = zip(*self.predictions_per_epoch)
        preds = torch.cat(preds).numpy()
        labels = torch.cat(labels).numpy()

        self.all_epoch_predictions[self.current_epoch] = (preds, labels)

        if self.boxplots:

            data = []

            for epoch, (epoch_preds, epoch_labels) in self.all_epoch_predictions.items():
                for pred, label in zip(epoch_preds, epoch_labels):
                    data.append({"Epoch": epoch,
                                 "Prediction": pred,
                                 "Label": "Cancer" if label==1 else "Non-Cancer"}
                                )

            df = pd.DataFrame(data)

            plt.figure(figsize=(10, 6))

            sns.boxplot(x="Epoch", y="Prediction", hue="Label", data=df, palette={"Cancer": "red", "Non-Cancer": "blue"}, width=0.6)

            plt.xlabel("Epoch")
            plt.ylabel("Model Prediction")

            plt.savefig("/autofs/space/crater_001/projects/BC_DBT_risk_assessment/linear_probing_overall/imagenet_dino/boxplots.png")

        # Reset for next epoch
        self.predictions_per_epoch = []


    def test_step(self, batch, batch_idx):
        embeddings, labels = extract_features_laterality(batch, self.embed_types, self.stats)
        #embeddings, labels = batch["embedding"], batch["label"]
        # Convert labels from list of strings to tensor of size (B, 1)
        labels = torch.tensor([int(x) for x in labels], dtype=torch.float32, device=embeddings.device).unsqueeze(1)

        # Get outputs
        outputs = self(embeddings)
        loss = self.loss_fn(outputs, labels)

        # Log metrics
        self.log("test_loss", loss)
        for metric_name, metric in self.val_metrics.items():
            metric(outputs, labels)

        # Store step outputs
        step_output = {
            "pred_probs": torch.softmax(outputs, dim=1),
            "preds": outputs.argmax(dim=1),
            "labels": labels,
            "patient_ids": batch["patient_id"],  # Store patient IDs for later analysis
        }
        self.test_step_outputs.append(step_output)
        return step_output

    def on_test_epoch_end(self):
        # Aggregate all predictions and labels
        all_preds = torch.cat([x["preds"] for x in self.test_step_outputs])
        all_labels = torch.cat([x["labels"] for x in self.test_step_outputs])
        all_probs = torch.cat([x["pred_probs"] for x in self.test_step_outputs])
        all_patient_ids = [
            id for x in self.test_step_outputs for id in x["patient_ids"]
        ]

        # Convert to numpy for easier handling
        preds_np = all_preds.cpu().float().numpy()
        labels_np = all_labels.cpu().float().numpy()
        probs_np = all_probs.cpu().float().numpy()

        # Process and save outputs
        metrics = process_test_outputs(
            preds_np, labels_np, probs_np, all_patient_ids, self.hparams.output_dir
        )

        # Log metrics
        for name, value in metrics.items():
            if name != "classification_report":
                self.log(name, value)

        # Clear the test step outputs
        self.test_step_outputs.clear()


class Cumulative_Risk_Layer(nn.Module):
    def __init__(self, num_features, args, max_followup, overall_risk=False):
        super(Cumulative_Risk_Layer, self).__init__()
        self.args = args
        self.logits = args.logits
        self.cumulative = args.cumulative_prob
        self.baseline_risk = args.baseline_risk

        if overall_risk:
            self.hazard_fc = nn.Linear(num_features, 1)
        else:
            self.hazard_fc = nn.Linear(num_features, max_followup)

        self.base_hazard_fc = nn.Linear(num_features, 1)
        self.softplus = nn.Softplus()

        mask = torch.ones([max_followup, max_followup])
        # Define lower triangular matrix
        mask = torch.tril(mask, diagonal=0)
        # Transpose mask to upper triangular matrix
        mask = torch.nn.Parameter(torch.t(mask), requires_grad=False)
        self.register_parameter('upper_triangular_mask', mask)

    def hazards(self, x):
        raw_hazard = self.hazard_fc(x)
        if self.logits:
            output_hazards = raw_hazard
        else:
            output_hazards = self.softplus(raw_hazard) # apply softplus to ensure positive hazards - range [0, +inf]
        return output_hazards

    def forward(self, x):
        hazards = self.hazards(x)
        if not self.logits:
            if self.cumulative == 'sum':
                B, T = hazards.size() #hazards is (B, T)
                expanded_hazards = hazards.unsqueeze(-1).expand(B, T, T) #expanded_hazards is (B,T, T)
                masked_hazards = expanded_hazards * self.upper_triangular_mask # masked_hazards now (B,T, T)
                if self.baseline_risk:
                    cum_prob = torch.sum(masked_hazards, dim=1) + self.base_hazard_fc(x)
                    #cum_prob = torch.matmul(hazards, self.upper_triangular_mask) + self.base_hazard_fc(x)
                else:
                    cum_prob = torch.sum(masked_hazards, dim=1) # range [0, +inf]
                    #cum_prob = torch.matmul(hazards, self.upper_triangular_mask)
            elif self.cumulative == 'max':
                cum_prob = torch.cummax(hazards, dim=1)[0]
            elif self.cumulative == 'none':
                cum_prob = hazards
            else:
                raise ValueError(f"Invalid cumulative risk type: {self.cumulative}")

            # Apply sigmoid like transformation 1-exp(-x) to ensure cumulative prediction values are mapped to [0, 1]
            output_preds = 1 - torch.exp(-cum_prob)

        else:
            output_preds = hazards

        return output_preds


class Cumulative_Risk_Model(pl.LightningModule):
    def __init__(self, args, max_followup: int, embed_type: list, stats: list):
        super().__init__()
        self.save_hyperparameters(args)

        self.experiment_name = args.experiment_name
        self.num_features = len(embed_type) * len(stats) * 3072
        self.max_followup = max_followup
        self.overall_risk = args.task == 'overall_risk'
        self.breast_specific = args.breast_specific
        self.embed_types = args.embedding_type
        self.stats = args.stats
        self.args = args

        # Define parameters for manual optimization
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.lr
        self.lr_scheduler = args.lr_scheduler
        self.weight_decay = args.weight_decay
        self.warmup_epochs = args.warmup_epochs

        # Define loss attributes
        self.logits = args.logits
        self.use_bce_w_logits = args.use_bce_w_logits
        self.cumulative = args.cumulative_prob
        self.batch_loss_avg = args.batch_loss_avg

        # Instantiate the classifier layer
        self.cumulative_prob_layer = Cumulative_Risk_Layer(self.num_features, args, max_followup, overall_risk=self.overall_risk)

        # Store loss for each epoch
        self.train_losses = []
        self.val_losses_imbalanced = []
        self.val_losses_balanced = []

        # Add test metrics storage
        self.val_step_outputs = []
        self.test_step_outputs = []

        self.val_metrics = nn.ModuleDict(
            {
                "roc_auc": torchmetrics.classification.MultilabelAUROC(num_labels=max_followup, average='weighted', thresholds=None),
                "year1_auroc": torchmetrics.classification.BinaryAUROC(thresholds=None),
                "year2_auroc": torchmetrics.classification.BinaryAUROC(thresholds=None),
                "year3_auroc": torchmetrics.classification.BinaryAUROC(thresholds=None),
                "year4_auroc": torchmetrics.classification.BinaryAUROC(thresholds=None),
                "year5_auroc": torchmetrics.classification.BinaryAUROC(thresholds=None),
            }
        )

    def forward(self, x):
        return self.cumulative_prob_layer(x)

    def training_step(self, batch, batch_idx):

        x, y = extract_features_laterality(batch, self.embed_types, self.stats, self.breast_specific)

        y_pred = self(x)

        if self.overall_risk:
            loss = overall_risk_loss(y_pred, batch, logits=self.logits)
        else:
            loss = risk_pred_loss(y_pred, batch, self.breast_specific, batch_loss_avg=self.batch_loss_avg, use_bce_with_logits=self.use_bce_w_logits)

        self.log("train_loss", loss, on_step=True, on_epoch=True, logger=True, batch_size=self.batch_size)

        # Store train losses
        self.train_losses.append(loss)

        # Get total training steps per epoch
        total_steps = self.trainer.num_training_batches  # Number of batches in the current epoch

        # Print step info (Step x / Total) + loss
        #print(f"Epoch {self.current_epoch} | Training Step {batch_idx + 1} / {total_steps} | Loss: {loss.item():.4f}")

        return loss

    def validation_step(self, batch, batch_idx):

        embeddings, labels = extract_features_laterality(batch, self.embed_types, self.stats, self.breast_specific)
        outputs = self(embeddings)  # Perform forward pass

        # Convert y to tensor:
        y_np = np.array([np.fromstring(x.strip("[]"), sep=" ") for x in labels], dtype=np.int64)
        y = torch.tensor(y_np, dtype=torch.int64, device=outputs.device)

        if self.overall_risk:
            loss = overall_risk_loss(outputs, batch, logits=self.logits)
        else:
            loss = risk_pred_loss(outputs, batch, self.breast_specific, batch_loss_avg=self.batch_loss_avg, use_bce_with_logits=self.use_bce_w_logits)

        #if dataloader_idx == 0:
        #    self.log("val_loss_imbalanced", loss, on_step=False, on_epoch=True, logger=True, batch_size=self.batch_size)
        #else:
        #    self.log("val_loss_balanced", loss, on_step=False, on_epoch=True, logger=True, batch_size=self.batch_size)

        #if dataloader_idx == 0:
        #    self.val_losses_imbalanced.append(loss)
        #else:
        #    self.val_losses_balanced.append(loss)

        self.log("val_loss", loss, on_step=False, on_epoch=True, logger=True, batch_size=self.batch_size)
        self.val_losses_balanced.append(loss)

        # Post-process outputs for storage
        if self.use_bce_w_logits:
            outputs = torch.sigmoid(outputs)

        # Get y_mask
        y_mask = batch['mask']
        y_mask_np = np.array([np.fromstring(x.strip("[]"), sep=" ") for x in y_mask], dtype=np.int64)
        y_mask = torch.tensor(y_mask_np, dtype=torch.int64, device=outputs.device)

        # Store step outputs
        step_output = {
            "preds": outputs,
            "labels": y,
            "mask": y_mask
        }

        self.val_step_outputs.append(step_output)

        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        scheduler = None
        # Set up learning rate scheduler
        if self.lr_scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.epochs
            )
        elif self.lr_scheduler == 'linear':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=1, gamma=0.9
            )
        elif self.lr_scheduler == 'constant':
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lambda epoch: 1
            )

        if scheduler:
            return [optimizer], [scheduler]
        else:
            return optimizer

    def on_train_epoch_end(self):
        avg_train_loss = torch.stack(self.train_losses).mean()
        #print(f"Epoch {self.current_epoch} - Average Train Loss: {avg_train_loss:.4f}")

    def on_validation_epoch_end(self):

        # Aggregate all predictions and labels
        all_preds = torch.cat([x["preds"] for x in self.val_step_outputs])
        all_labels = torch.cat([x["labels"] for x in self.val_step_outputs])
        all_masks = torch.cat([x["mask"] for x in self.val_step_outputs])

        avg_auroc, auroc_per_year = compute_auroc_per_year(all_labels, all_preds, all_masks, self.val_metrics, set="val")
        self.log("val_avg_auroc", avg_auroc, on_epoch=True, on_step=False, logger=True)

        self.val_metrics["roc_auc"].update(all_preds, all_labels)

        # Compute auroc per year over full validation set
        for metric_name, metric in self.val_metrics.items():
            value = metric.compute()
            self.log(f"val_{metric_name}", value, on_epoch=True)
            metric.reset()

        #avg_val_loss_imbalanced = torch.stack(self.val_losses_imbalanced).mean()
        avg_val_loss_balanced = torch.stack(self.val_losses_balanced).mean()
        #print(f"Epoch {self.current_epoch} - Average Validation Loss: {avg_val_loss_imbalanced:.4f}")
        #print(f"Epoch {self.current_epoch} - Average Validation Loss (Balanced): {avg_val_loss_balanced:.4f}")

        # Clear lists after logging
        #self.val_losses_imbalanced.clear()
        self.val_losses_balanced.clear()

        self.val_step_outputs.clear()

        return avg_auroc, auroc_per_year


    def test_step(self, batch, batch_idx):
        embeddings, labels = extract_features_laterality(batch, self.embed_types, self.stats, self.breast_specific)
        if self.breast_specific:
            x_r, x_l, y_r, y_l = extract_features_laterality(batch, self.embed_types, self.stats, self.breast_specific)
            outputs_l = self(x_r)
            outputs_r = self(y_l)
        else:
            outputs = self(embeddings) # Perform forward pass

        # Convert labels to tensor:
        y_np = np.array([np.fromstring(x.strip("[]"), sep=" ") for x in labels], dtype=np.int64)
        y = torch.tensor(y_np, dtype=torch.int64, device=outputs.device)

        # Compute loss
        if self.overall_risk:
            loss = overall_risk_loss(outputs, batch, self.breast_specific, logits=self.logits, batch_loss_avg=self.batch_loss_avg)
        else:
            loss = risk_pred_loss(outputs, batch, self.breast_specific, batch_loss_avg=self.batch_loss_avg, use_bce_with_logits=self.use_bce_w_logits)

        # Get y_masks
        y_mask = batch['mask']
        y_mask_np = np.array([np.fromstring(x.strip("[]"), sep=" ") for x in y_mask], dtype=np.int64)
        y_mask = torch.tensor(y_mask_np, dtype=torch.int64, device=outputs.device)

        # Store step outputs
        step_output = {
            "preds": outputs,
            "labels": y,
            "masks": y_mask,
            "patient_ids": batch["patient_id"],  # Store patient IDs for later analysis
            "study_ids": batch["study_id"],
            "breast_density_labels": batch["breast_density"],
            "loss": loss
        }

        self.test_step_outputs.append(step_output)

        return step_output

    def on_test_epoch_end(self):
        # Aggregate all predictions and labels
        all_preds = torch.cat([x["preds"] for x in self.test_step_outputs])
        all_labels = torch.cat([x["labels"] for x in self.test_step_outputs])
        all_masks = torch.cat([x["masks"] for x in self.test_step_outputs])
        # Aggregate metadata (assuming these are lists or 1D tensors)
        all_patient_ids = [pid for d in self.test_step_outputs for pid in d["patient_ids"]]
        all_study_ids = [sid for d in self.test_step_outputs for sid in d["study_ids"]]
        all_breast_densities = [bd for d in self.test_step_outputs for bd in d["breast_density_labels"]]

        save_df = pd.DataFrame(columns=["preds", "labels", "masks", "patient_ids", "study_ids", "breast_density_labels"])

        # Convert to numpy for easier handling
        preds_np = all_preds.cpu().float().numpy()
        labels_np = all_labels.cpu().float().numpy()

        # Save predictions, labels, and masks to DataFrame
        save_df['preds'] = all_preds.cpu().numpy().tolist()
        save_df['labels'] = all_labels.cpu().numpy().tolist()
        save_df['masks'] = all_masks.cpu().numpy().tolist()
        save_df['patient_ids'] = [str(pid) for pid in all_patient_ids]
        save_df['study_ids'] = [str(sid) for sid in all_study_ids]
        save_df['breast_density_labels'] = [str(bd) for bd in all_breast_densities]

        save_dir = "/path/to/save/dir"
        os.makedirs(save_dir, exist_ok=True)
        # save to csv
        save_df.to_csv(os.path.join(save_dir, f'preds_df_{self.experiment_name}.csv'), index=False)

        print(f"Saved predictions DataFrame to {os.path.join(save_dir, f'preds_df_{self.experiment_name}.csv')}")

        # save all preds to npy file
        np.save(os.path.join(os.path.join(save_dir, f'test_results_{self.experiment_name}.npy')), all_preds.cpu().numpy())

        avg_auroc, auroc_per_year = compute_auroc_per_year(all_labels, all_preds, all_masks, self.val_metrics, set='test')

        auroc_1 = auroc_per_year['year1']
        auroc_2 = auroc_per_year['year2']
        auroc_3 = auroc_per_year['year3']
        auroc_4 = auroc_per_year['year4']
        auroc_5 = auroc_per_year['year5']

        self.log("test_avg_auroc", avg_auroc, on_epoch=True, on_step=False, logger=True)
        self.log("test_auroc1", auroc_1, on_epoch=True, on_step=False, logger=True)
        self.log("test_auroc2", auroc_2, on_epoch=True, on_step=False, logger=True)
        self.log("test_auroc3", auroc_3, on_epoch=True, on_step=False, logger=True)
        self.log("test_auroc4", auroc_4, on_epoch=True, on_step=False, logger=True)
        self.log("test_auroc5", auroc_5, on_epoch=True, on_step=False, logger=True)

        self.test_step_outputs.clear()

        return avg_auroc, auroc_per_year

