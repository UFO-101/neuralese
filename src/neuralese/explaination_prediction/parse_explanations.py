# %%
import json
import os
import gzip
import re
from enum import Enum
from pathlib import Path
from typing import Dict
import urllib.request
import xml.etree.ElementTree as ET
from io import BytesIO

from neuralese.file_utils import (
    repo_path_to_abs_path,
    ensure_dir_exists,
)


class ActivationSite(Enum):
    RESIDUAL = "res"


def download_and_cache_explanations(
    model_name: str,
    layer: int,
    activation_site: ActivationSite = ActivationSite.RESIDUAL,
    n_features: str = "65k",
    cache_dir: Path = repo_path_to_abs_path(".cache/explanations"),
) -> Dict[int, str]:
    """
    Download and cache explanations from Neuronpedia.

    Args:
        model_name: Name of the model (e.g., "gemma-2-2b")
        layer: Layer number (e.g., 16)
        activation_site: Activation site (e.g., RESIDUAL)
        n_features: Number of features as a string (e.g., "65k")
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

    # Format the base URL for the directory listing
    base_url = f"https://neuronpedia-datasets.s3.amazonaws.com/v1/{model_name}/{layer}-gemmascope-{activation_site.value}-{n_features}/explanations/"

    print(f"Fetching file list from {base_url}")

    # Get the directory listing
    try:
        with urllib.request.urlopen(base_url) as response:
            html_content = response.read().decode()

        # Try to parse as XML first (S3 returns XML listing)
        try:
            root = ET.fromstring(html_content)
            # Look for Contents elements which contain Keys (filenames)
            namespace = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
            batch_files = []

            for key in root.findall(".//s3:Contents/s3:Key", namespace):
                filename = key.text
                if filename and "batch-" in filename and filename.endswith(".jsonl.gz"):
                    batch_files.append(filename)
        except ET.ParseError:
            # If not XML, try regex on HTML
            batch_files = re.findall(
                r'href="([^"]*batch-\d+\.jsonl\.gz)"', html_content
            )

            if not batch_files:
                # Try alternative regex patterns
                batch_files = re.findall(r">(batch-\d+\.jsonl\.gz)<", html_content)

            if not batch_files:
                # Try another pattern that might match S3 listing format
                batch_files = re.findall(
                    r"<Key>([^<]*batch-\d+\.jsonl\.gz)</Key>", html_content
                )

    except Exception as e:
        print(f"Error fetching directory listing: {e}")
        # Try a direct approach - list some common batch numbers
        batch_files = [f"batch-{i}.jsonl.gz" for i in range(0, 150)]
        print(f"Falling back to predefined batch files: {len(batch_files)} files")

    if not batch_files:
        raise ValueError(f"Could not find any batch files at {base_url}")

    print(f"Found {len(batch_files)} batch files")

    # Initialize the dictionary
    index_to_description = {}

    # Process each batch file
    for batch_file in batch_files:
        # Handle both relative and absolute URLs
        if batch_file.startswith("http"):
            file_url = batch_file
        else:
            # Remove any leading path components if present
            batch_filename = os.path.basename(batch_file)
            file_url = f"{base_url}{batch_filename}"

        print(f"Downloading {file_url}")

        try:
            with urllib.request.urlopen(file_url) as response:
                # Read the response content
                gzip_content = response.read()

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
        except Exception as e:
            print(f"Error processing {file_url}: {e}")
            continue

    print(f"Processed {len(index_to_description)} explanations")

    # Cache the parsed dictionary
    with open(cache_path, "w") as f:
        # Convert int keys to strings for JSON serialization
        json.dump({str(k): v for k, v in index_to_description.items()}, f)

    return index_to_description


def download_and_cache_explanations_single_file(
    model_name: str,
    layer: int,
    activation_site: ActivationSite = ActivationSite.RESIDUAL,
    n_features: str = "65k",
    cache_dir: Path = repo_path_to_abs_path(".cache/explanations"),
) -> Dict[int, str]:
    """
    Download and cache explanations from Neuronpedia (single file version).

    Args:
        model_name: Name of the model (e.g., "gemma-2-2b")
        layer: Layer number (e.g., 16)
        activation_site: Activation site (e.g., RESIDUAL)
        n_features: Number of features as a string (e.g., "65k")
        cache_dir: Directory to cache the downloaded JSON. Defaults to .cache/explanations

    Returns:
        Dictionary mapping from index (int) to description
    """
    # Ensure cache directory exists
    ensure_dir_exists(cache_dir)

    # Create a filename for caching
    cache_filename = (
        f"{model_name}_{layer}-gemmascope-{activation_site.value}-{n_features}.json"
    )
    cache_path = cache_dir / cache_filename

    # Format the URL
    url = f"https://neuronpedia-exports.s3.amazonaws.com/explanations-only/{cache_filename}"

    # Check if the file is already cached
    if cache_path.exists():
        print(f"Loading cached explanations from {cache_path}")
        with open(cache_path, "r") as f:
            data = json.load(f)
    else:
        print(f"Downloading explanations from {url}")
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode())

        # Cache the data
        with open(cache_path, "w") as f:
            json.dump(data, f)

    # Extract index to description mapping, converting index to int
    index_to_description = {int(item["index"]): item["description"] for item in data}

    return index_to_description


if __name__ == "__main__":
    # Example usage
    explanations = download_and_cache_explanations(
        model_name="gemma-2-2b",
        layer=12,
        activation_site=ActivationSite.RESIDUAL,
        n_features="1m",
    )

    # Print a few examples
    print(f"Total explanations: {len(explanations)}")
    for i, (index, description) in enumerate(list(explanations.items())[:5]):
        print(f"Index {index}: {description}")
