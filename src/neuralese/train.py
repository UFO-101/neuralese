# %%
from datetime import datetime
from typing import Any, Dict

import torch as t
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformer_lens import HookedTransformer

import wandb
from neuralese.config import Config
from neuralese.conversations_data import (
    load_and_group_data,
    process_conversations,
    tokenize_conversations,
)
from neuralese.translator import Translator, load_model


def get_neuralese_loss(
    batch: Dict[str, Any],
    target: HookedTransformer,
    translator: Translator,
    config: Config,
) -> t.Tensor:
    """MSE loss of the translator model predicting the next neuralese activation."""
    target_tokenized = tokenize_conversations(batch, target.tokenizer, config)
    target_tokens_BS = target_tokenized["input_ids"].to(translator.device)
    target_attn_mask_BS = target_tokenized["attn_mask"].to(translator.device)

    with t.no_grad():
        input_neuralese_BSd = target(target_tokens_BS, stop_at_layer=config.mid_layer)
    # Run the neuralese through the translator model
    output_neuralese_BSd = translator.forward_neuralese(
        input_neuralese_BSd, target_attn_mask_BS
    )

    # Next token loss on the neuralese (MSELoss)
    next_token_loss_BS = t.nn.functional.mse_loss(
        output_neuralese_BSd[:, :-1, :], input_neuralese_BSd[:, 1:, :], reduction="none"
    ).mean(dim=-1)
    # Ignore token positions which are masked out or where the next token is masked out
    next_token_mask_BS = target_attn_mask_BS[:, :-1] & target_attn_mask_BS[:, 1:]
    masked_next_token_loss_BS = next_token_loss_BS * next_token_mask_BS
    return masked_next_token_loss_BS.mean()


def get_kl_div_loss(
    batch: Dict[str, Any],
    orig_translator: HookedTransformer,
    translator: Translator,
    config: Config,
) -> t.Tensor:
    """
    KL divergence loss between the original translator model and the translator model.
    This helps prevent catastrophic forgetting.
    """
    translator_tokenized = tokenize_conversations(batch, translator.tokenizer, config)
    translator_tokens_BS = translator_tokenized["input_ids"].to(translator.device)
    translatr_attn_mask_BS = translator_tokenized["attn_mask"].to(translator.device)

    # Minimize the KL divergence between the original_translator_model and the translator_model
    with t.no_grad():
        orig_logits = orig_translator(translator_tokens_BS)
        orig_log_probs_BSV = t.nn.functional.log_softmax(orig_logits, dim=-1)
    translator_logits = translator.forward_tokens(translator_tokens_BS)
    translator_log_probs_BSV = t.nn.functional.log_softmax(translator_logits, dim=-1)

    kl_divergence_BS = t.nn.functional.kl_div(
        translator_log_probs_BSV, orig_log_probs_BSV, reduction="none", log_target=True
    ).sum(dim=-1)
    masked_kl_divergence_BS = kl_divergence_BS * translatr_attn_mask_BS
    return masked_kl_divergence_BS.mean()


def train_translator(
    dataloader: DataLoader[Dict[str, Any]],
    target: HookedTransformer,
    orig_translator: HookedTransformer,
    translator: Translator,
    config: Config,
) -> Translator:
    optim = t.optim.Adam(translator.parameters(), lr=config.learning_rate)

    last_save = 0
    neuralese_loss, best_neura_loss = float("inf"), float("inf")
    for i, batch in (pbar := tqdm(enumerate(dataloader))):
        neuralese_loss = get_neuralese_loss(batch, target, translator, config)
        optim.zero_grad()
        neuralese_loss.backward()
        optim.step()

        optim.zero_grad()
        kl_div_loss = get_kl_div_loss(batch, orig_translator, translator, config)
        if kl_div_loss.item() > 1e-8:
            kl_div_loss.backward()
            optim.step()

        wandb.log(
            {
                "neuralese_loss": neuralese_loss,
                "kl_div_loss": kl_div_loss,
                "total_loss": neuralese_loss + kl_div_loss,
            }
        )

        if i - last_save >= config.save_interval and neuralese_loss < best_neura_loss:
            translator.save_trained()
            last_save = i
            best_neura_loss = neuralese_loss

        pbar.set_description(f"MSE loss: {neuralese_loss:.4f} | KL: {kl_div_loss:.4f}")

    if neuralese_loss < best_neura_loss:
        translator.save_trained()

    return translator


def run_training(config: Config, device: str) -> Translator:
    target_model = load_model(config.target_model_name, config.dtype, device)
    original_translator_model = load_model(
        config.translator_model_name, config.dtype, device
    )
    target_model_dim = target_model.cfg.d_model
    translator_model_dim = original_translator_model.cfg.d_model
    translator = Translator(target_model_dim, translator_model_dim, config, device)

    tree_groups = load_and_group_data(config)
    dataloader = process_conversations(tree_groups, target_model, config)
    wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        config=config.to_dict(),
    )

    trained_translator = train_translator(
        dataloader=dataloader,
        target=target_model,
        orig_translator=original_translator_model,
        translator=translator,
        config=config,
    )

    return trained_translator


if __name__ == "__main__":
    device = "cuda:5" if t.cuda.is_available() else "cpu"
    datatime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    config = Config.from_repo_path_str(f".translators/{datatime_str}")
    # config = SmallModelConfig.from_repo_path_str(f".translators/{datatime_str}.pt")
    train_translator = run_training(config, device)
