# %%
from typing import Any, Dict, List

import torch as t
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from transformer_lens import HookedTransformer
from transformers import PreTrainedTokenizerBase

from neuralese.generate_activations import load_model


def load_and_filter_dataset(english_only: bool = True) -> Dataset:
    """Load the OpenAssistant dataset and optionally filter for English."""
    ds = load_dataset("OpenAssistant/oasst2", streaming=False)
    train = ds["train"]  # type: ignore
    if english_only:
        train = train.filter(lambda x: x["lang"] == "en")  # type: ignore
    return train  # type: ignore


def sample_conversations(dataset: Dataset, n_samples: int) -> Dataset:
    """Sample n unique conversation trees from the dataset."""
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


def load_and_group_data(n_samples: int = 3) -> Dict[str, List[Dict[str, Any]]]:
    """Load dataset and group messages by conversation tree.

    Args:
        n_samples: Number of conversation trees to process

    Returns:
        Dictionary mapping tree_id to list of messages in that tree
    """
    train = load_and_filter_dataset(english_only=True)
    filtered_train = sample_conversations(train, n_samples)
    tree_groups = group_messages_by_tree(filtered_train)
    return tree_groups


device = "cuda:5" if t.cuda.is_available() else "cpu"
target_model = load_model("Qwen/Qwen2.5-0.5B-Instruct", device)

tree_groups = load_and_group_data(n_samples=3)
# %%


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
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader[Dict[str, Any]]:
    """Create a PyTorch DataLoader from conversation trees."""
    dataset = ConversationDataset(tree_conversations, model.tokenizer)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_conversations,
    )


def tokenize_conversations(
    batch: Dict[str, Any], tokenizer: Any, max_length: int = 2048
) -> Dict[str, t.Tensor]:
    """Tokenize a batch of conversations.

    Args:
        batch: Dictionary containing untokenized conversations
        tokenizer: Tokenizer to use for encoding
        max_length: Maximum sequence length

    Returns:
        Dictionary containing tokenized inputs with attention masks
    """
    encoded = tokenizer(
        batch["text"],
        padding="longest",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
        return_attention_mask=True,
    )

    return {
        "input_ids": encoded.input_ids,
        "attention_mask": encoded.attention_mask,
        "tree_id": batch["tree_id"],
        "raw_conversation": batch["raw_conversation"],
    }


def print_batch_details(batch: Dict[str, Any], target_model: HookedTransformer) -> None:
    """Print details about a batch of tokenized conversations.

    Args:
        batch: Dictionary containing batch tensors
        target_model: Model containing the tokenizer
    """
    print("\n=== First Batch Details ===")
    print(f"Batch size: {len(batch['input_ids'])}")
    print(f"Input shape: {batch['input_ids'].shape}")
    print(f"Attention mask shape: {batch['attention_mask'].shape}")

    # Print first sequence details
    print("\nFirst sequence in batch:")
    print(f"Non-padding tokens: {batch['attention_mask'][0].sum()}")
    print("tokens:", batch["input_ids"][0])
    print("mask values:", batch["attention_mask"][0])

    # Decode first few tokens
    tokenizer = target_model.tokenizer
    assert isinstance(tokenizer, PreTrainedTokenizerBase)
    first_tokens = tokenizer.decode(batch["input_ids"][0][:50])
    print("\nDecoded first 50 tokens:", first_tokens)


def process_conversations(
    tree_groups: Dict[str, List[Dict[str, Any]]],
    target_model: HookedTransformer,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
    print_examples: bool = False,
) -> DataLoader[Dict[str, Any]]:
    """Process conversations and create a DataLoader.

    Args:
        tree_groups: Dictionary mapping tree_id to list of messages
        target_model: Model containing the tokenizer
        batch_size: Number of conversations per batch
        shuffle: Whether to shuffle the dataset
        num_workers: Number of worker processes for data loading
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
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
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
        batch = tokenize_conversations(
            raw_batch, target_model.tokenizer, max_length=2048
        )
        print_batch_details(batch, target_model)

    return dataloader


if __name__ == "__main__":
    dataloader = process_conversations(tree_groups, target_model, print_examples=True)

# %%
