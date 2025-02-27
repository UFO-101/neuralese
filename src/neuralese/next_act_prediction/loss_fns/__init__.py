# Write an all import statement for all the loss functions
from typing import Any, Callable, Dict

import torch as t
from transformer_lens import HookedTransformer

from neuralese.next_act_prediction.config import Config, LossType
from neuralese.next_act_prediction.data.conversations_data import tokenize_batch
from neuralese.next_act_prediction.translator import Translator

from .distill_orig_logits import get_distill_orig_logits_loss
from .neuralese_dot_prod import get_dot_prod_loss
from .neuralese_mse import get_neuralese_loss
from .neuralese_to_orig_logit import get_neuralese_to_orig_logit_loss

__all__ = [
    "get_neuralese_to_orig_logit_loss",
    "get_distill_orig_logits_loss",
    "get_dot_prod_loss",
    "get_neuralese_loss",
]


LOSS_FN_MAP: dict[LossType, Callable] = {
    "mse": get_neuralese_loss,
    "dot_prod": get_dot_prod_loss,
    "neuralese_to_orig_logit": get_neuralese_to_orig_logit_loss,
    "distill_orig_logit": get_distill_orig_logits_loss,
}


def get_orig_translator_logprobs(
    batch: Dict[str, Any],
    orig_translator: HookedTransformer,
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

    # Minimize the KL divergence between the original_translator_model and the translator_model
    with t.no_grad():
        orig_logits = orig_translator(
            translator_tokens_BS, attention_mask=translatr_attn_mask_BS
        )
        orig_logprobs_BSV = t.nn.functional.log_softmax(orig_logits, dim=-1)
    return orig_logprobs_BSV
