# %%
import gzip
import json
import random
import urllib.request
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Dict

from neuralese.file_utils import (
    ensure_dir_exists,
    repo_path_to_abs_path,
)


class ActivationSite(Enum):
    RESIDUAL = "res"


def download_and_cache_explanations(
    model_name: str,
    layer: int,
    activation_site: ActivationSite = ActivationSite.RESIDUAL,
    n_features: str = "65k",
    num_batches: int = 150,  # Number of batch files to try, starting from 0
    cache_dir: Path = repo_path_to_abs_path(".cache/explanations"),
) -> Dict[int, str]:
    """
    Download and cache explanations from Neuronpedia.

    Args:
        model_name: Name of the model (e.g., "gemma-2-2b")
        layer: Layer number (e.g., 16)
        activation_site: Activation site (e.g., RESIDUAL)
        n_features: Number of features as a string (e.g., "65k")
        num_batches: Number of batch files to download (0 to num_batches-1)
        cache_dir: Directory to cache the downloaded JSON. Defaults to .cache/explanations

    Returns:
        Dictionary mapping from index (int) to description
    """
    # Ensure cache directory exists
    ensure_dir_exists(cache_dir)

    # Create a filename for caching the final parsed dictionary
    cache_filename = f"{model_name}_{layer}-gemmascope-{activation_site.value}-{n_features}_parsed.json"
    cache_path = cache_dir / cache_filename

    # Check if the parsed dictionary is already cached
    if cache_path.exists():
        print(f"Loading cached parsed explanations from {cache_path}")
        with open(cache_path, "r") as f:
            index_to_description = json.load(f)
            # Convert string keys back to integers
            index_to_description = {int(k): v for k, v in index_to_description.items()}
            return index_to_description

    # Format the base URL
    base_url = f"https://neuronpedia-datasets.s3.amazonaws.com/v1/{model_name}/{layer}-gemmascope-{activation_site.value}-{n_features}/explanations/"

    # Generate batch file names
    batch_files = [f"batch-{i}.jsonl.gz" for i in range(num_batches)]
    print(f"Will attempt to download {len(batch_files)} batch files")

    # Initialize the dictionary
    index_to_description = {}
    successful_downloads = 0

    # Process each batch file
    for batch_file in batch_files:
        file_url = f"{base_url}{batch_file}"

        try:
            with urllib.request.urlopen(file_url) as response:
                # Read the response content
                gzip_content = response.read()
                successful_downloads += 1

                # Decompress the gzip content
                with gzip.GzipFile(fileobj=BytesIO(gzip_content)) as gzipped_data:
                    # Process each line in the JSONL file
                    for line in gzipped_data:
                        try:
                            item = json.loads(line)
                            # Extract index and description
                            if "index" in item and "description" in item:
                                index_to_description[int(item["index"])] = item[
                                    "description"
                                ]
                        except json.JSONDecodeError:
                            print(f"Error parsing JSON line in {batch_file}")
                            continue

            print(f"Successfully downloaded and processed {batch_file}")

        except Exception as e:
            print(f"Error downloading {file_url}: {e}")
            continue

    print(
        f"Successfully downloaded {successful_downloads} out of {len(batch_files)} batch files"
    )
    print(f"Processed {len(index_to_description)} explanations")

    if len(index_to_description) == 0:
        raise ValueError("Could not extract any explanations from the batch files")

    # Cache the parsed dictionary
    with open(cache_path, "w") as f:
        # Convert int keys to strings for JSON serialization
        json.dump({str(k): v for k, v in index_to_description.items()}, f)

    return index_to_description


if __name__ == "__main__":
    # Example usage
    explanations = download_and_cache_explanations(
        model_name="gemma-2-2b",
        layer=12,
        activation_site=ActivationSite.RESIDUAL,
        n_features="1m",
        num_batches=754,  # Try batches 0 through 753
    )

    # Print a few examples
    print(f"Total explanations: {len(explanations)}")

    # Print a few random examples
    for i in range(5):
        index = random.choice(list(explanations.keys()))
        print(f"Index {index}: {explanations[index]}")
