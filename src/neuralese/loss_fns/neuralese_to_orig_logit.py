from typing import Any, Dict

import torch as t
import torch.nn.functional as F
from transformer_lens import HookedTransformer

from neuralese.config import Config
from neuralese.data.conversations_data import (
    tokenize_batch,
)
from neuralese.translator import Translator


def get_neuralese_to_orig_logit_loss(
    batch: Dict[str, Any],
    orig_translator_logprobs_BSV: t.Tensor,
    target: HookedTransformer,
    translator: Translator,
    config: Config,
) -> t.Tensor:
    """
    Train the translator model to predict the same logits it would predict for the
    tokens, when given the target model neuralese for those tokens.
    """
    target_tokenized = tokenize_batch(batch, target.tokenizer, config)
    target_tokens_BS = target_tokenized["input_ids"].to(translator.device)
    target_attn_mask_BS = target_tokenized["attn_mask"].to(translator.device)
    translator_tokenized = tokenize_batch(batch, translator.tokenizer, config)
    translator_tokens_BS = translator_tokenized["input_ids"].to(translator.device)
    translator_attn_mask_BS = translator_tokenized["attn_mask"].to(translator.device)

    # Assert the tokenized inputs are the same
    assert target_tokens_BS.shape == translator_tokens_BS.shape
    assert t.all(target_tokens_BS == translator_tokens_BS)
    assert t.all(target_attn_mask_BS == translator_attn_mask_BS)

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
    translator_logits_BSV = translator.neuralese_to_logits(
        input_neuralese_BSd, target_attn_mask_BS
    )
    translator_logprobs_BSV = t.nn.functional.log_softmax(translator_logits_BSV, dim=-1)

    kl_divergence_BS = t.nn.functional.kl_div(
        translator_logprobs_BSV,
        orig_translator_logprobs_BSV,
        reduction="none",
        log_target=True,
    ).sum(dim=-1)
    masked_kl_divergence_BS = kl_divergence_BS * translator_attn_mask_BS
    return masked_kl_divergence_BS.sum() / translator_attn_mask_BS.sum()
