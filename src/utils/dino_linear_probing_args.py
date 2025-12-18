import argparse


def get_args_parser():
    """
    Create and return the argument parser for the DINO linear probing script.
    """
    parser = argparse.ArgumentParser(
        "DINO feature extraction and linear probing", add_help=False
    )
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument(
        "--accum_iter",
        default=1,
        type=int,
        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)",
    )

    # Model parameters
    parser.add_argument(
        "--model",
        default="dino_dbt",
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )
    parser.add_argument(
        "--backbone_checkpoint",
        default=None,
        type=str,
        help="Path to the backbone checkpoint file. This expects the state dict of a teacher checkpoint file.",
    )

    parser.add_argument(
        "--nb_classes", default=4, type=int, help="Number of the classification types for breast density task;"
                                                  "max time frame for risk prediction task (should be 5)."
    )

    parser.add_argument(
        "--max_followup",
        default=5,
        type=int,
        help="Maximum follow-up time for risk prediction task",
    )

    parser.add_argument(
        "--baseline_risk",
        default=False,
        action="store_true",
        help="Whether to add baseline cancer risk to yearly risk predictions"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        metavar="LR",
        help="learning rate (absolute lr)",
    )

    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        choices=["cosine", "none"],
        help="learning rate schedule",
    )

    parser.add_argument("--weight_decay", type=float, default=0.0001, help="weight decay")

    parser.add_argument("--warmup_epochs", action="store_true", help="Use warmup scheduler for first epochs")

    parser.add_argument("--csv_path", help="path to csv definition file of the dataset")
    parser.add_argument("--data_dir", help="path to directory containing the images")
    parser.add_argument(
        "--embedding_dir", default="./embeddings", help="path to save/load embeddings"
    )

    parser.add_argument("--embedding_dir_cancer", default="./embeddings_cancer", help="path to save/load embeddings")
    parser.add_argument("--embedding_dir_healthy", default="./embeddings_healthy", help="path to save/load embeddings")

    parser.add_argument(
        "--output_dir",
        default="./output_dir",
        help="path where to save, empty for no saving",
    )

    parser.add_argument(
        "--log_dir", default="./output_dir", help="path where to tensorboard log"
    )
    
    parser.add_argument(
        "--db_folder", default=None, help="path where to save/load optuna database to allow synchronization across runs"
    )

    parser.add_argument("--seed", default=None, type=int)

    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=False)

    # Add new arguments for Optuna
    parser.add_argument(
        "--use_optuna",
        action="store_true",
        help="Whether to use Optuna for hyperparameter tuning",
    )
    parser.add_argument(
        "--n_trials", type=int, default=16, help="Number of Optuna trials to run"
    )
    parser.add_argument(
        "--study_name",
        type=str,
        default="dino_linear_probing",
        help="Name of the Optuna study",
    )

    parser.add_argument(
        "--embedding_type",
        type=str,
        nargs='+',
        default=["patch"],
        choices=["patch", "cls"],
        help="Type of embedding to use (patch, cls or both)",
    )

    parser.add_argument(
        "--stats",
        type=str,
        nargs='+',
        default=["mean", "std"],
        choices=["mean", "std", "max", "min"],
        help="Type of statistics to use for the embeddings (default: mean and std)",
    )

    parser.add_argument(
        "--use_ms",
        default=False,
        help="Whether to use multi-scale for the model",
    )

    parser.add_argument(
        "--task",
        type=str,
        help="Classification task to perform ('density' or 'risk' or 'overall_risk').",
    )

    parser.add_argument(
        "--binary_risk",
        action="store_true",
        help="Whether to output binary risk predictions (0 or 1)",
    )

    parser.add_argument(
        "--breast_specific",
        action="store_true",
        help="Whether to output breast specific risk predictions",
    )

    parser.add_argument(
        "--cancer_only",
        action="store_true",
        help="Whether to use cancer cohort data only",
    )

    parser.add_argument(
        "--logits",
        action="store_true",
        help="Whether to use loss function with logits for risk prediction task",
    )

    parser.add_argument(
        "--use_bce_w_logits",
        action="store_true",
        help="Whether to use loss function with logits for risk prediction task",
    )

    parser.add_argument(
        "--cumulative_prob",
        default='none',
        type=str,
        help="Cumulative probability function to use for risk prediction task. Should be 'max' or 'sum' or 'none'.",
    )

    parser.add_argument(
        "--batch_loss_avg",
        action="store_true",
        help="Whether to average the loss over the batch",
    )

    parser.add_argument(
        "--boxplots",
        action="store_true",
        help="Whether to generate boxplot figure for the risk predictions",
    )

    parser.add_argument(
        "--validation",
        type=str,
        default="balanced",
        choices=["imbalanced", "balanced", "combined"],
        help="Validation strategy to use",
    )

    parser.add_argument(
        "--dataset_fraction",
        type=float,
        default=1.0,
        help="Fraction of the dataset to use for training and validation",
    )
    parser.add_argument(
        "--attention_type",
        type=str,
        default="gated",
        choices=["gated", "standard", "self"],
        help="Attention type to use",
    )

    parser.add_argument(
        "--trial_number",
        type=int,
        default=None,
        help="Trial number to use for evaluation",
    )
    parser.add_argument(
        "--checkpoint_folder",
        type=str,
        default=None,
        help="Path to the checkpoint file to load",
    )

    parser.add_argument(
        "--all_slices",
        action="store_true",
        help="Use the *all-slices* detection dataset which creates one sample per slice and saves predictions to CSV. "
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to the Linear Probing Model checkpoint to load",
    )

    parser.add_argument(
        "--include_6_months",
        action="store_true",
        help="Whether to include less than 6 months screening data"
    )

    parser.add_argument(
        "--experiment_name",
        type=str,
        default="dino_linear_probing_experiment",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="train",
    )

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
    )

    return parser
