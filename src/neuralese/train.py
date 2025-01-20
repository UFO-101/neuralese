# %%
from datetime import datetime
from typing import Any, Dict

import torch as t
import torch.nn.functional as F
from einops import einsum
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformer_lens import HookedTransformer

import wandb
from neuralese.config import Config
from neuralese.data.conversations_data import (
    tokenize_batch,
)
from neuralese.data.get_data import get_data
from neuralese.evaluate import measure_neuralese_recon
from neuralese.translator import Translator, load_model


def get_neuralese_loss(
    batch: Dict[str, Any],
    target: HookedTransformer,
    translator: Translator,
    config: Config,
) -> t.Tensor:
    """MSE loss of the translator model predicting the next neuralese activation."""
    target_tokenized = tokenize_batch(batch, target.tokenizer, config)
    target_tokens_BS = target_tokenized["input_ids"].to(translator.device)
    target_attn_mask_BS = target_tokenized["attn_mask"].to(translator.device)

    with t.no_grad():
        input_neuralese_BSd = target(
            target_tokens_BS,
            stop_at_layer=config.mid_layer,
            attention_mask=target_attn_mask_BS,
        )
        if config.loss_type == "ln_mse":
            d_model = target.cfg.d_model
            input_neuralese_BSd = F.layer_norm(input_neuralese_BSd, (d_model,))
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
    return masked_next_token_loss_BS.sum() / next_token_mask_BS.sum()


def get_ln_dot_prod_loss(
    batch: Dict[str, Any],
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
        lyrnrmd_target_BSd = target(
            target_tokens_BS,
            stop_at_layer=config.mid_layer,
            attention_mask=target_attn_mask_BS,
        )
        lyrnrmd_target_BSd = F.layer_norm(lyrnrmd_target_BSd, (d_model,))
    # Run the neuralese through the translator model
    output_neuralese_BSd = translator.forward_neuralese(
        lyrnrmd_target_BSd, target_attn_mask_BS
    )
    lyrnrmd_pred_BSd = F.layer_norm(output_neuralese_BSd, (d_model,))

    # Dot product loss
    dot_prod_BS = -1 * einsum(
        lyrnrmd_pred_BSd[:, :-1, :], lyrnrmd_target_BSd[:, 1:, :], "b s d, b s d -> b s"
    )
    # Ignore token positions which are masked out or where the next token is masked out
    next_token_mask_BS = target_attn_mask_BS[:, :-1] & target_attn_mask_BS[:, 1:]
    masked_dot_prod_BS = dot_prod_BS * next_token_mask_BS
    return masked_dot_prod_BS.sum() / next_token_mask_BS.sum()


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
    translator_tokenized = tokenize_batch(batch, translator.tokenizer, config)
    translator_tokens_BS = translator_tokenized["input_ids"].to(translator.device)
    translatr_attn_mask_BS = translator_tokenized["attn_mask"].to(translator.device)

    # Minimize the KL divergence between the original_translator_model and the translator_model
    with t.no_grad():
        orig_logits = orig_translator(
            translator_tokens_BS, attention_mask=translatr_attn_mask_BS
        )
        orig_log_probs_BSV = t.nn.functional.log_softmax(orig_logits, dim=-1)
    translator_logits = translator.forward_tokens(
        translator_tokens_BS, attention_mask=translatr_attn_mask_BS
    )
    translator_log_probs_BSV = t.nn.functional.log_softmax(translator_logits, dim=-1)

    kl_divergence_BS = t.nn.functional.kl_div(
        translator_log_probs_BSV, orig_log_probs_BSV, reduction="none", log_target=True
    ).sum(dim=-1)
    masked_kl_divergence_BS = kl_divergence_BS * translatr_attn_mask_BS
    return masked_kl_divergence_BS.sum() / translatr_attn_mask_BS.sum()


def train_translator(
    train_dataloader: DataLoader[Dict[str, Any]],
    val_dataloader: DataLoader[Dict[str, Any]],
    target: HookedTransformer,
    orig_translator: HookedTransformer,
    translator: Translator,
    config: Config,
) -> Translator:
    optim = t.optim.Adam(translator.parameters(), lr=config.learning_rate)
    loss_fn = {
        "mse": get_neuralese_loss,
        "ln_mse": get_neuralese_loss,
        "ln_dot_prod": get_ln_dot_prod_loss,
    }[config.loss_type]

    last_save = 0
    neuralese_loss, best_neura_loss = float("inf"), float("inf")
    for i, batch in (pbar := tqdm(enumerate(train_dataloader))):
        neuralese_loss = loss_fn(batch, target, translator, config)
        optim.zero_grad()
        neuralese_loss.backward()
        optim.step()

        optim.zero_grad()
        kl_div_loss = get_kl_div_loss(batch, orig_translator, translator, config)
        kl_div_loss *= config.kl_weight
        kl_div_loss.backward()
        optim.step()

        logs = {"neuralese_loss": neuralese_loss, "kl_div_loss": kl_div_loss}
        if i % config.eval_interval == 0 and i > 0:
            evals = measure_neuralese_recon(val_dataloader, target, translator, config)
            logs.update(evals)
        wandb.log(logs)

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
    if config.save_path.exists():
        translator = Translator.from_pretrained(config, device)
        train_dataloader = get_data(config, target_model, "train_2")
        print("Loaded translator from", config.save_path)
        print("Using train_2 data")
    else:
        translator = Translator(target_model_dim, config, device)
        train_dataloader = get_data(config, target_model, "train")

    val_dataloader = get_data(config, target_model, "validation")
    wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        config=config.to_dict(),
    )

    trained_translator = train_translator(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        target=target_model,
        orig_translator=original_translator_model,
        translator=translator,
        config=config,
    )

    return trained_translator


if __name__ == "__main__":
    device = "cuda:6" if t.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    datatime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    config = Config.from_repo_path_str(f".translators/{datatime_str}.pt")
    # config = SmallModelConfig.from_repo_path_str(f".translators/{datatime_str}.pt")
    train_translator = run_training(config, device)
