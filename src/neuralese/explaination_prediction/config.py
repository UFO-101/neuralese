from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from neuralese.explaination_prediction.explanations_data import ActivationSite
from neuralese.file_utils import repo_path_to_abs_path


@dataclass(frozen=True)
class ExplanationTrainingConfig:
    """Configuration for training an explanation model."""

    # Model to be fine-tuned
    model_name: str = "google/gemma-2-2b-it"

    # Explanation data
    activation_site: ActivationSite = ActivationSite.RESIDUAL
    n_features: str = "1m"
    num_batches: int = 754
    cache_dir: Path = repo_path_to_abs_path(".cache/explanations")

    # Model / SAE explanations dataset to train on
    sae_repo_id: str = "google/gemma-scope-2b-pt-res"
    layer: int = 12
    width: str = "1m"
    l0_threshold: str = "107"
    max_length: int = 128  # Maximum length of an input to the model

    # Training config
    wandb_project: str = "neuralese"
    wandb_entity: str = "josephmiller101"
    save_strategy: str = "steps"
    save_steps: int = 200
    save_total_limit: int = 1
    eval_strategy: str = "steps"
    eval_steps: int = 1000
    logging_strategy: str = "no"
    report_to: str = "wanbd"
    seed: int = 42

    # Training hyperparameters
    batch_size: int = 8
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_steps: int = 100
    weight_decay: float = 0.01

    # Paths
    datetime_str: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir: Path = repo_path_to_abs_path(".models/explanation_predictors")
    save_path: Path = (
        save_dir / f"{datetime_str}_{model_name}_{layer}-{width}-{l0_threshold}.pt"
    )
