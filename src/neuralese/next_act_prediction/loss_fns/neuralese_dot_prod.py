from typing import Any, Dict

import torch as t
import torch.nn.functional as F
from einops import einsum
from transformer_lens import HookedTransformer

from neuralese.next_act_prediction.config import Config
from neuralese.next_act_prediction.data.conversations_data import (
    tokenize_batch,
)
from neuralese.next_act_prediction.translator import Translator


def get_dot_prod_loss(
    batch: Dict[str, Any],
    orig_translator_logprobs_BSV: t.Tensor,  # type: ignore
    target: HookedTransformer,
    translator: Translator,
    config: Config,
) -> t.Tensor:
    """Dot product loss between the layernormed neuralese and the layernormed target."""
    target_tokenized = tokenize_batch(batch, target.tokenizer, config)
    target_tokens_BS = target_tokenized["input_ids"].to(translator.device)
    target_attn_mask_BS = target_tokenized["attn_mask"].to(translator.device)
    d_model = target.cfg.d_model

    with t.no_grad():
        neuralese_BSd = target(
            target_tokens_BS,
            stop_at_layer=config.mid_layer,
            attention_mask=target_attn_mask_BS,
        )
        neuralese_BSd = F.layer_norm(neuralese_BSd, (d_model,))
    # Run the neuralese through the translator model
    output_neuralese_BSd = translator.neuralese_to_neuralese(
        neuralese_BSd, target_attn_mask_BS
    )
    neuralese_pred_BSd = F.layer_norm(output_neuralese_BSd, (d_model,))

    # Dot product loss
    dot_prod_BS = -1 * einsum(
        neuralese_pred_BSd[:, :-1, :], neuralese_BSd[:, 1:, :], "b s d, b s d -> b s"
    )
    # Ignore token positions which are masked out or where the next token is masked out
    next_token_mask_BS = target_attn_mask_BS[:, :-1] & target_attn_mask_BS[:, 1:]
    masked_dot_prod_BS = dot_prod_BS * next_token_mask_BS
    return masked_dot_prod_BS.sum() / next_token_mask_BS.sum()
