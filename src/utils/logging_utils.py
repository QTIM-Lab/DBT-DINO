import logging
import os


def setup_logging(output_dir, log_name):
    """Set up logging configuration."""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"{log_name}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    logging.info(f"Logging to {log_file}")
