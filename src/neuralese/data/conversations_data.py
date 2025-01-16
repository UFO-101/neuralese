# %%
from typing import Any, Dict, List

import torch as t
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from transformer_lens import HookedTransformer
from transformers import PreTrainedTokenizerBase

from neuralese.config import Config
from neuralese.data.data_utils import print_batch_details, tokenize_batch
from neuralese.translator import load_model


def load_and_filter_dataset(config: Config, dataset_split: str) -> Dataset:
    """Load the OpenAssistant dataset and optionally filter for English."""
    ds = load_dataset(config.dataset_name, streaming=False)

    if config.dataset_name == "OpenAssistant/oasst2":
        # The "OpenAssistant/oasst2"dataset only has a "train" and "validation" split
        # So if dataset split is "train", use only the first 100,000 samples
        # And if dataset split is "validation", use only the remaining samples
        # And if dataset split is "test", use the validation split
        if dataset_split == "train":
            data = ds[dataset_split].select(range(100_000))  # type: ignore
        elif dataset_split == "validation":
            data = ds["train"].select(  # type: ignore
                range(100_000, len(ds["train"]))  # type: ignore
            )
        else:
            assert dataset_split == "test"
            data = ds["validation"]  # type: ignore
    else:
        data = ds[dataset_split]  # type: ignore

    if config.english_only:
        data = data.filter(lambda x: x["lang"] == "en")  # type: ignore
    return data  # type: ignore


def sample_conversations(dataset: Dataset, config: Config) -> Dataset:
    """Sample n unique conversation trees from the dataset."""
    n_samples = len(dataset) if config.n_samples is None else config.n_samples
    unique_tree_ids = list(set(dataset["message_tree_id"]))[:n_samples]
    return dataset.filter(lambda x: x["message_tree_id"] in unique_tree_ids)


def group_messages_by_tree(dataset: Dataset) -> Dict[str, List[Dict[str, Any]]]:
    """Group messages by their tree_id."""
    tree_groups = {}
    for msg in dataset:
        assert isinstance(msg, dict)
        if msg["message_tree_id"] not in tree_groups:
            tree_groups[msg["message_tree_id"]] = []
        tree_groups[msg["message_tree_id"]].append(msg)
    return tree_groups


def load_and_group_data(
    config: Config, dataset_split: str
) -> Dict[str, List[Dict[str, Any]]]:
    """Load dataset and group messages by conversation tree.

    Args:
        config: Configuration object
    Returns:
        Dictionary mapping tree_id to list of messages in that tree
    """
    train = load_and_filter_dataset(config, dataset_split)
    filtered_train = sample_conversations(train, config)
    tree_groups = group_messages_by_tree(filtered_train)
    return tree_groups


def build_conversation(messages: List[Dict[str, Any]]) -> List[List[Dict[str, str]]]:
    """Convert a tree of messages into all possible linear conversation paths.

    Args:
        messages: List of message dictionaries from the dataset

    Returns:
        List of conversations, where each conversation is a list of
        message dictionaries in the format {"role": str, "content": str}
    """

    def get_children(msg_id: str) -> List[Dict[str, Any]]:
        """Get all child messages for a given message ID."""
        return [msg for msg in messages if msg["parent_id"] == msg_id]

    def build_paths(
        current_msg: Dict[str, Any], current_path: List[Dict[str, str]]
    ) -> List[List[Dict[str, str]]]:
        """Recursively build all possible conversation paths from current message."""
        # Convert role and add current message to path
        role = "user" if current_msg["role"] == "prompter" else current_msg["role"]
        current_path = current_path + [{"role": role, "content": current_msg["text"]}]

        # Get all children of current message
        children = get_children(current_msg["message_id"])

        if not children:
            # If no children, this is a leaf node - return the current path
            return [current_path]

        # Recursively build paths for all children
        all_paths = []
        for child in children:
            child_paths = build_paths(child, current_path)
            all_paths.extend(child_paths)

        return all_paths

    # Find root message
    msg_dict = {msg["message_id"]: msg for msg in messages}
    root = next(msg for msg in messages if msg["parent_id"] not in msg_dict)

    # Build all paths starting from root
    return build_paths(root, [])


def print_conversations(
    conversations: List[List[Dict[str, str]]], model: HookedTransformer, n_samples: int
) -> None:
    """Print formatted conversations and their chat templates."""
    print(f"\nShowing {n_samples} example conversations:")
    for i, conversation in enumerate(conversations):
        print(f"\n=== Conversation {i + 1} ===")
        print("\nRaw conversation:")
        for msg in conversation:
            print(f"{msg['role']}: {msg['content'][:100]}...")

        print("\nFormatted for model:")
        tokenizer = model.tokenizer
        assert isinstance(tokenizer, PreTrainedTokenizerBase)
        text = tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        assert isinstance(text, str)
        print(text[:1000], "..." if len(text) > 1000 else "")
        print("\n" + "=" * 50)


class ConversationDataset(TorchDataset):
    """PyTorch dataset for conversations."""

    def __init__(
        self, tree_conversations: Dict[str, List[List[Dict[str, str]]]], tokenizer: Any
    ):
        """Initialize dataset from conversation trees.

        Args:
            tree_conversations: Dictionary mapping tree_id to list of conversations
            tokenizer: Tokenizer (used only to apply chat template)
        """
        self.conversations = []
        self.tokenizer = tokenizer

        # Flatten all conversations and format them
        for tree_id, conv_list in tree_conversations.items():
            for conversation in conv_list:
                # Apply chat template but don't tokenize
                text = tokenizer.apply_chat_template(
                    conversation, tokenize=False, add_generation_prompt=True
                )

                self.conversations.append(
                    {"tree_id": tree_id, "text": text, "raw_conversation": conversation}
                )

    def __len__(self) -> int:
        return len(self.conversations)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.conversations[idx]


def collate_conversations(batch: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    """Collate conversations into a batch.

    Args:
        batch: List of dictionaries containing conversation data

    Returns:
        Dictionary with batched data
    """
    return {
        "tree_id": [item["tree_id"] for item in batch],
        "text": [item["text"] for item in batch],
        "raw_conversation": [item["raw_conversation"] for item in batch],
    }


def create_conversation_dataset(
    tree_conversations: Dict[str, List[List[Dict[str, str]]]],
    model: HookedTransformer,
    config: Config,
) -> DataLoader[Dict[str, Any]]:
    """Create a PyTorch DataLoader from conversation trees."""
    dataset = ConversationDataset(tree_conversations, model.tokenizer)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        collate_fn=collate_conversations,
    )


def process_conversations(
    tree_groups: Dict[str, List[Dict[str, Any]]],
    target_model: HookedTransformer,
    config: Config,
    print_examples: bool = False,
) -> DataLoader[Dict[str, Any]]:
    """Process conversations and create a DataLoader.

    Args:
        tree_groups: Dictionary mapping tree_id to list of messages
        target_model: Model containing the tokenizer
        config: Configuration object
        print_examples: Whether to print example conversations and batch details

    Returns:
        PyTorch DataLoader containing formatted conversations
    """
    # Build conversation trees
    tree_conversations = {}
    for tree_id, messages in tree_groups.items():
        tree_conversations[tree_id] = build_conversation(messages)

    # Create DataLoader with untokenized conversations
    dataloader = create_conversation_dataset(
        tree_conversations,
        target_model,
        config,
    )

    if print_examples:
        dataset = dataloader.dataset
        assert isinstance(dataset, ConversationDataset)
        print(f"\nCreated dataloader with {len(dataset)} conversations")

        # Print example conversations (taking first conversation from each tree)
        example_conversations = [convs[0] for convs in tree_conversations.values()]
        print_conversations(example_conversations, target_model, n_samples=3)

        # Get and tokenize first batch
        raw_batch = next(iter(dataloader))
        batch = tokenize_batch(raw_batch, target_model.tokenizer, config)
        print_batch_details(batch, target_model)

    return dataloader


if __name__ == "__main__":
    device = "cuda:7" if t.cuda.is_available() else "cpu"
    config = Config.from_repo_path_str("", n_samples=4)
    target_model = load_model(config.target_model_name, config.dtype, device)
    tree_groups = load_and_group_data(config, "train")
    dataloader = process_conversations(
        tree_groups, target_model, config, print_examples=True
    )

# %%
