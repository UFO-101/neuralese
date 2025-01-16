from typing import Any, Dict

import torch as t
from transformer_lens import HookedTransformer
from transformers import PreTrainedTokenizerBase

from neuralese.config import Config


def tokenize_conversations(
    batch: Dict[str, Any], tokenizer: Any, config: Config
) -> Dict[str, t.Tensor]:
    """Tokenize a batch of conversations.

    Args:
        batch: Dictionary containing untokenized conversations
        tokenizer: Tokenizer to use for encoding
        config: Configuration object

    Returns:
        Dictionary containing tokenized inputs with attention masks
    """
    encoded = tokenizer(
        batch["text"],
        padding="longest",
        max_length=config.max_length,
        truncation=True,
        return_tensors="pt",
        return_attention_mask=True,
    )

    return {
        "input_ids": encoded.input_ids,
        "attn_mask": encoded.attention_mask,
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
    print(f"Attention mask shape: {batch['attn_mask'].shape}")

    # Print first sequence details
    print("\nFirst sequence in batch:")
    print(f"Non-padding tokens: {batch['attn_mask'][0].sum()}")
    print("tokens:", batch["input_ids"][0])
    print("mask values:", batch["attn_mask"][0])

    # Decode first few tokens
    tokenizer = target_model.tokenizer
    assert isinstance(tokenizer, PreTrainedTokenizerBase)
    first_tokens = tokenizer.decode(batch["input_ids"][0][:50])
    print("\nDecoded first 50 tokens:", first_tokens)
