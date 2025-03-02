# %%
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file

from neuralese.file_utils import (
    ensure_dir_exists,
    repo_path_to_abs_path,
)


def sae_decoder_weights(
    repo_id: str = "google/gemma-scope-2b-pt-res",
    layer: int = 12,
    width: str = "1m",
    l0_threshold: str = "107",
    cache_dir: Path = repo_path_to_abs_path(".cache/sae_weights"),
) -> torch.Tensor:
    """
    Download SAE weights from Hugging Face and extract only the decoder matrix.

    Download the weight directly into RAM and then save only the decoder matrix to disk.

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
    cache_path = (
        cache_dir / f"layer_{layer}_width_{width}_average_l0_{l0_threshold}.safetensors"
    )
    if cache_path.exists():
        print(f"Loading decoder weights from {cache_path}")
        return load_file(cache_path)["decoder_weights"]

    temp_dir = repo_path_to_abs_path(".cache/sae_weights")
    ensure_dir_exists(temp_dir)

    # Construct the file path within the huggingface repository
    hf_path = f"layer_{layer}/width_{width}/average_l0_{l0_threshold}/params.npz"

    # Download the weights file to the temp directory
    print(f"Downloading weights from {repo_id} to {temp_dir}")
    weights_file = hf_hub_download(
        repo_id=repo_id, filename=hf_path, cache_dir=temp_dir
    )
    print(f"Weights file downloaded to {weights_file}")

    # Load the weights
    print(f"Loading weights from {weights_file}")
    state_dict = np.load(weights_file, mmap_mode="r")
    print(f"Weights loaded from {weights_file}")
    print(f"State dict keys: {state_dict.keys()}")
    decoder_weights = state_dict["W_dec"]
    print(f"Decoder weights shape: {decoder_weights.shape}")

    # Save the decoder weights to the cache directory
    print(f"Saving decoder weights to {cache_path}")
    decoder_weights_torch = torch.from_numpy(decoder_weights)
    save_file(
        {
            "decoder_weights": decoder_weights_torch,
        },
        cache_path,
    )
    print(f"Decoder weights saved to {cache_path}")

    # Delete the temp file
    print(f"Deleting temp file {weights_file}")
    os.remove(weights_file)
    print("Temp file deleted")

    return decoder_weights


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
    decoder = sae_decoder_weights(
        repo_id="google/gemma-scope-2b-pt-res",
        layer=12,
        width="1m",
        l0_threshold="107",
    )

    print(f"Decoder shape: {decoder.shape}")

    # Get the first 5 feature vectors
    features = get_feature_vectors(decoder, torch.tensor(range(5)))
    print(f"First 5 feature vectors shape: {features.shape}")
