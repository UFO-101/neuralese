# %%
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from neuralese.file_utils import (
    ensure_dir_exists,
)


def download_sae_decoder_weights(
    repo_id: str = "google/gemma-scope-2b-pt-res",
    layer: int = 12,
    width: str = "1m",
    l0_threshold: str = "107",
    cache_dir: Path = Path(".cache/sae_weights"),
    force_download: bool = False,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Download SAE weights from Hugging Face and extract only the decoder matrix.

    This function downloads the full SAE weights but only keeps the decoder matrix in memory,
    saving disk space after the initial download.

    Args:
        repo_id: The Hugging Face repository ID
        layer: The layer number
        width: The width of the SAE (e.g., "1m" for 1 million features)
        l0_threshold: The L0 threshold used in the SAE
        cache_dir: Directory to cache the downloaded weights
        force_download: Whether to force download even if the file exists locally

    Returns:
        Tuple containing:
        - decoder_weights: The decoder weight matrix as a torch tensor
        - metadata: Dictionary with metadata about the SAE
    """
    ensure_dir_exists(cache_dir)

    # The weights are stored in a safetensors file
    weights_file = (
        f"layer_{layer}_width_{width}_average_l0_{l0_threshold}_sae_weights.safetensors"
    )
    weights_path: Path = cache_dir / weights_file

    # The config is stored in a JSON file
    config_file = f"layer_{layer}_width_{width}_average_l0_{l0_threshold}_cfg.json"
    config_path: Path = cache_dir / config_file

    # Download the weights file
    local_weights_path = hf_hub_download(
        repo_id=repo_id,
        filename=weights_path.as_posix(),
        cache_dir=cache_dir,
        force_download=force_download,
    )

    # Download the config file
    local_config_path = hf_hub_download(
        repo_id=repo_id,
        filename=config_path.as_posix(),
        cache_dir=cache_dir,
        force_download=force_download,
    )

    state_dict = load_file(local_weights_path)
    decoder_weights = state_dict["W_dec"]

    with open(local_config_path, "r") as f:
        config = json.load(f)

    # Create metadata dictionary
    metadata = {
        "config": config,
        "layer": layer,
        "width": width,
        "l0_threshold": l0_threshold,
        "repo_id": repo_id,
        "decoder_shape": decoder_weights.shape,
    }

    # # Add some useful info from the state dict
    # if "b_dec" in state_dict:
    #     metadata["b_dec"] = state_dict["b_dec"]

    return decoder_weights, metadata


def get_feature_vectors(
    decoder_weights: torch.Tensor,
    indices: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Get specific feature vectors from the decoder weights.

    Args:
        decoder_weights: The decoder weight matrix
        indices: Optional indices of features to extract. If None, returns all features.

    Returns:
        Tensor containing the requested feature vectors
    """
    if indices is None:
        return decoder_weights

    return decoder_weights[indices]


if __name__ == "__main__":
    # Example usage
    decoder, metadata = download_sae_decoder_weights(
        repo_id="google/gemma-scope-2b-pt-res",
        layer=12,
        width="1m",
        l0_threshold="107",
    )

    print(f"Decoder shape: {decoder.shape}")
    print(f"Metadata keys: {metadata.keys()}")

    # Get the first 5 feature vectors
    features = get_feature_vectors(decoder, torch.tensor(range(5)))
    print(f"First 5 feature vectors shape: {features.shape}")
