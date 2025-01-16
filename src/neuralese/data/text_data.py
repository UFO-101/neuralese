# %%
from typing import Any, Dict, Iterator

import torch as t
from datasets import IterableDataset, load_dataset
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset as TorchIterableDataset
from transformer_lens import HookedTransformer

from neuralese.config import Config
from neuralese.data.data_utils import print_batch_details, tokenize_conversations
from neuralese.translator import load_model


def load_streaming_dataset(config: Config) -> IterableDataset:
    """Load the FinWeb dataset in streaming mode."""
    ds = load_dataset(config.dataset_name, streaming=True)
    data = ds[config.dataset_split]  # type: ignore
    return data  # type: ignore


class StreamingTextDataset(TorchIterableDataset):
    """PyTorch IterableDataset for streaming text data."""

    def __init__(self, texts: IterableDataset, tokenizer: Any, config: Config):
        """Initialize dataset from streaming texts.

        Args:
            texts: IterableDataset containing text samples
            tokenizer: Tokenizer (used only to format text)
            config: Configuration object for sample limit
        """
        super().__init__()
        self.texts = texts
        self.tokenizer = tokenizer
        self.n_samples = config.n_samples

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate through the dataset, yielding formatted texts.

        Yields:
            Dictionary containing text and ID for each sample
        """
        count = 0
        for text in self.texts:
            if self.n_samples is not None and count >= self.n_samples:
                break

            yield {
                "text": text["text"],
                "id": str(text["id"]) if "id" in text else str(count),
            }
            count += 1


def collate_texts(batch: list[Dict[str, Any]]) -> Dict[str, list[Any]]:
    """Collate texts into a batch.

    Args:
        batch: List of dictionaries containing text data

    Returns:
        Dictionary with batched data
    """
    return {
        "text": [item["text"] for item in batch],
        "id": [item["id"] for item in batch],
    }


def create_streaming_dataset(
    texts: IterableDataset,
    model: HookedTransformer,
    config: Config,
) -> DataLoader[Dict[str, Any]]:
    """Create a PyTorch DataLoader from streaming texts."""
    dataset = StreamingTextDataset(texts, model.tokenizer, config)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        # Note: shuffle is not supported for IterableDataset
        collate_fn=collate_texts,
        num_workers=0,  # Streaming datasets work better with single worker
    )


def process_streaming_texts(
    dataset: IterableDataset,
    target_model: HookedTransformer,
    config: Config,
    print_examples: bool = False,
) -> DataLoader[Dict[str, Any]]:
    """Process streaming texts and create a DataLoader.

    Args:
        dataset: IterableDataset containing text samples
        target_model: Model containing the tokenizer
        config: Configuration object
        print_examples: Whether to print example texts and batch details

    Returns:
        PyTorch DataLoader containing formatted texts
    """
    dataloader = create_streaming_dataset(
        dataset,
        target_model,
        config,
    )

    if print_examples:
        print("\nCreated streaming dataloader")

        # Print example texts
        print("\nShowing 3 example texts:")
        example_iter = iter(dataloader)
        for i in range(3):
            try:
                batch = next(example_iter)
                print(f"\n=== Text {i + 1} ===")
                print(f"ID: {batch['id'][0]}")
                print(f"Text: {batch['text'][0][:200]}...")

                if i == 0:
                    # Show tokenization details for first batch
                    tokenized_batch = tokenize_conversations(
                        batch, target_model.tokenizer, config
                    )
                    print_batch_details(tokenized_batch, target_model)
            except StopIteration:
                break

    return dataloader


def get_text_data(
    config: Config, target_model: HookedTransformer
) -> DataLoader[Dict[str, Any]]:
    """Load and process streaming text data.

    Args:
        config: Configuration object
        target_model: Model containing the tokenizer

    Returns:
        DataLoader containing processed texts
    """
    dataset = load_streaming_dataset(config)
    dataloader = process_streaming_texts(dataset, target_model, config)
    return dataloader


if __name__ == "__main__":
    device = "cuda:5" if t.cuda.is_available() else "cpu"
    config = Config.from_repo_path_str(
        "", n_samples=10, dataset_name="HuggingFaceFW/fineweb"
    )
    target_model = load_model(config.target_model_name, config.dtype, device)
    dataloader = get_text_data(config, target_model)

    # Test iteration
    for i, batch in enumerate(dataloader):
        if i == 0:
            print("\nFirst batch shape:", len(batch["text"]))
            print("First text preview:", batch["text"][0][:100])
        if i >= 2:
            break
