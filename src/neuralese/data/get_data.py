# %%
from typing import Any, Dict

import torch as t
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer

from neuralese.config import Config
from neuralese.data.conversations_data import load_and_group_data, process_conversations
from neuralese.data.data_utils import tokenize_batch
from neuralese.data.text_data import get_text_data
from neuralese.translator import load_model


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


if __name__ == "__main__":
    device = "cuda:7" if t.cuda.is_available() else "cpu"
    config = Config.from_repo_path_str(
        "", n_samples=10, dataset_name="HuggingFaceFW/fineweb"
    )
    target_model = load_model(config.target_model_name, config.dtype, device)
    dataloader = get_data(config, target_model)

    # Test iteration
    n_print = 6
    for i, batch in enumerate(dataloader):
        tokens = tokenize_batch(batch, target_model.tokenizer, config)
        print("batch size:", len(batch["text"]))
        print("First text preview:", batch["text"][0][:100])
        print("tokens shape", tokens["input_ids"].shape)
        non_padding_tokens = tokens["attn_mask"] != 0
        print("Number of non-padding tokens in each row", non_padding_tokens.sum(dim=1))
        print(
            "Mean number of non-padding tokens in each row",
            non_padding_tokens.sum(dim=1).float().mean(),
        )
        if i >= n_print - 1:
            break
