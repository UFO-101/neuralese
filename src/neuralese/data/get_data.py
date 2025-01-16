from typing import Any, Dict

from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer

from neuralese.config import Config
from neuralese.data.conversations_data import load_and_group_data, process_conversations
from neuralese.data.text_data import get_text_data


def get_data(
    config: Config, target_model: HookedTransformer
) -> DataLoader[Dict[str, Any]]:
    """Get data based on dataset type.

    Args:
        config: Configuration object
        target_model: Model containing the tokenizer

    Returns:
        DataLoader containing processed data
    """
    if config.dataset_name == "OpenAssistant/oasst2":
        tree_groups = load_and_group_data(config)
        return process_conversations(tree_groups, target_model, config)
    else:
        return get_text_data(config, target_model)
