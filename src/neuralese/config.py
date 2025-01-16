from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch as t

from neuralese.file_utils import repo_path_to_abs_path


@dataclass(frozen=True)
class Config:
    save_path: Path
    target_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    translator_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    mid_layer: int = 12
    dtype: t.dtype = t.float32

    # Training
    wandb_project: str = "neuralese"
    wandb_entity: str = "josephmiller101"
    save_interval: int = 100
    eval_interval: int = 1000
    learning_rate: float = 1e-5
    kl_weight: float = 1e-5
    random_init_translator: bool = True
    random_init_mode: str = "gpt2"  # Corresponds to init_mode in TransformerLens

    # Dataset and data loading
    dataset_name: str = "OpenAssistant/oasst2"
    english_only: bool = False
    n_samples: int | None = None
    batch_size: int = 2
    shuffle: bool = True
    max_length: int = 400

    # Evaluation
    mean_resid_min_toks: int = 10_000
    mean_resid_cache_path: Path = repo_path_to_abs_path(
        f".mean_resid_cache/{target_model_name.replace('/', '_')}"
        + f"_{dataset_name.replace('/', '_')}_layer_{mid_layer}.pt"
    )
    measure_reconstruction_min_toks: int = 10_000

    @classmethod
    def from_repo_path_str(cls, repo_path_str: str, **kwargs: Any) -> "Config":
        repo_path = repo_path_to_abs_path(repo_path_str)
        return cls(repo_path, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SmallModelConfig(Config):
    translator_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    batch_size: int = 1
