from typing import Any, Dict

import torch as t
import torch.nn.functional as F
from transformer_lens import HookedTransformer

from neuralese.next_act_prediction.config import Config
from neuralese.next_act_prediction.data.conversations_data import (
    tokenize_batch,
)
from neuralese.next_act_prediction.translator import Translator


def get_neuralese_loss(
    batch: Dict[str, Any],
    orig_translator_logprobs_BSV: t.Tensor,  # type: ignore
    target: HookedTransformer,
    translator: Translator,
    config: Config,
) -> t.Tensor:
    """MSE loss of the translator model predicting the next neuralese activation."""
    target_tokenized = tokenize_batch(batch, target.tokenizer, config)
    target_tokens_BS = target_tokenized["input_ids"].to(translator.device)
    target_attn_mask_BS = target_tokenized["attn_mask"].to(translator.device)
    d_model = target.cfg.d_model

    with t.no_grad():
        input_neuralese_BSd = target(
            target_tokens_BS,
            stop_at_layer=config.mid_layer,
            attention_mask=target_attn_mask_BS,
        )
        if config.layernorm_neuralese:
            input_neuralese_BSd = F.layer_norm(input_neuralese_BSd, (d_model,))

    # Run the neuralese through the translator model
    output_neuralese_BSd = translator.neuralese_to_neuralese(
        input_neuralese_BSd, target_attn_mask_BS
    )
    if config.layernorm_neuralese:
        output_neuralese_BSd = F.layer_norm(output_neuralese_BSd, (d_model,))

    # Next token loss on the neuralese (MSELoss)
    next_token_loss_BS = t.nn.functional.mse_loss(
        output_neuralese_BSd[:, :-1, :], input_neuralese_BSd[:, 1:, :], reduction="none"
    ).mean(dim=-1)
    # Ignore token positions which are masked out or where the next token is masked out
    next_token_mask_BS = target_attn_mask_BS[:, :-1] & target_attn_mask_BS[:, 1:]
    masked_next_token_loss_BS = next_token_loss_BS * next_token_mask_BS
    return masked_next_token_loss_BS.sum() / next_token_mask_BS.sum()
