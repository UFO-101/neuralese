from typing import Any, Dict

import torch as t
from transformer_lens import HookedTransformer

from neuralese.config import Config
from neuralese.data.conversations_data import (
    tokenize_batch,
)
from neuralese.translator import Translator


def get_distill_orig_logits_loss(
    batch: Dict[str, Any],
    orig_translator_logprobs_BSV: t.Tensor,
    target: HookedTransformer,  # type: ignore
    translator: Translator,
    config: Config,
) -> t.Tensor:
    """
    KL divergence loss between the original translator model and the translator model.
    This helps prevent catastrophic forgetting.
    """
    translator_tokenized = tokenize_batch(batch, translator.tokenizer, config)
    translator_tokens_BS = translator_tokenized["input_ids"].to(translator.device)
    translatr_attn_mask_BS = translator_tokenized["attn_mask"].to(translator.device)

    translator_logits = translator.forward_tokens(
        translator_tokens_BS, attention_mask=translatr_attn_mask_BS
    )
    translator_log_probs_BSV = t.nn.functional.log_softmax(translator_logits, dim=-1)

    kl_divergence_BS = t.nn.functional.kl_div(
        translator_log_probs_BSV,
        orig_translator_logprobs_BSV,
        reduction="none",
        log_target=True,
    ).sum(dim=-1)
    masked_kl_divergence_BS = kl_divergence_BS * translatr_attn_mask_BS
    return masked_kl_divergence_BS.sum() / translatr_attn_mask_BS.sum()
