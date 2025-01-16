# %%
import datetime
from typing import Any, Dict

import torch as t
import torch.nn.functional as F
from einops import repeat
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformer_lens import HookedTransformer

from neuralese.config import Config
from neuralese.data.data_utils import tokenize_batch
from neuralese.data.get_data import get_data
from neuralese.file_utils import ensure_dir_exists
from neuralese.translator import Translator, load_model
from neuralese.visualize import visualize_vector_reconstruction


def mean_resid(
    dataloader: DataLoader[Dict[str, Any]],
    model: HookedTransformer,
    cfg: Config,
) -> t.Tensor:
    if cfg.mean_resid_cache_path.exists():
        return t.load(
            cfg.mean_resid_cache_path, weights_only=True, map_location=model.cfg.device
        )

    input_sum_d = t.zeros(model.cfg.d_model, device=model.cfg.device)
    non_masked_count: t.Tensor = t.tensor(0, device=model.cfg.device)
    for batch in (pbar := tqdm(dataloader)):
        translator_tokenized = tokenize_batch(batch, model.tokenizer, cfg)
        tokens_BS = translator_tokenized["input_ids"].to(model.cfg.device)
        attn_mask_BS = translator_tokenized["attn_mask"].to(model.cfg.device)

        with t.no_grad():
            input_neuralese_BSd = model(
                tokens_BS,
                stop_at_layer=cfg.mid_layer,
                attention_mask=attn_mask_BS,
            )
        attn_mask_BSd = repeat(attn_mask_BS, "B S -> B S d", d=model.cfg.d_model)
        masked_neuralese_BSd = input_neuralese_BSd * attn_mask_BSd
        input_sum_d += masked_neuralese_BSd.sum(dim=(0, 1))

        non_masked_count += attn_mask_BS.sum()
        desc = f"Mean resid: {non_masked_count.sum()} / {cfg.mean_resid_min_toks} toks"
        pbar.set_description(desc)
        if non_masked_count.sum() > cfg.mean_resid_min_toks:
            break

    mean_vector = input_sum_d / non_masked_count
    ensure_dir_exists(cfg.mean_resid_cache_path.parent)
    t.save(mean_vector, cfg.mean_resid_cache_path)
    return mean_vector


def measure_neuralese_reconstruction(
    dataloader: DataLoader[Dict[str, Any]],
    target: HookedTransformer,
    translator: Translator,
    config: Config,
    visualize: bool = False,
) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
    """MSE loss of the translator model predicting the next neuralese activation."""
    mean_resid_d = mean_resid(dataloader, target, config)

    mse_loss_sum = t.tensor(0.0, dtype=config.dtype, device=target.cfg.device)
    mse_loss_normalizd_sum = t.tensor(0.0, dtype=config.dtype, device=target.cfg.device)
    fvu_sum = t.tensor(0.0, dtype=config.dtype, device=target.cfg.device)
    non_masked_count = t.tensor(0, device=target.cfg.device)
    for batch in dataloader:
        target_tokenized = tokenize_batch(batch, target.tokenizer, config)
        target_tokens_BS = target_tokenized["input_ids"].to(translator.device)
        target_attn_mask_BS = target_tokenized["attn_mask"].to(translator.device)

        with t.no_grad():
            input_neuralese_BSd = target(
                target_tokens_BS,
                stop_at_layer=config.mid_layer,
                attention_mask=target_attn_mask_BS,
            )
            # Run the neuralese through the translator model
            output_neuralese_BSd = translator.forward_neuralese(
                input_neuralese_BSd, target_attn_mask_BS
            )

        if visualize:
            datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            visualize_vector_reconstruction(
                input_neuralese_BSd,
                output_neuralese_BSd,
                n_vectors_cutoff=50,
                vector_cutoff=100,
                target_tokens_BS=target_tokens_BS,
                target_model=target,
                save_repo_path=f".visualizations/{datetime_str}.png",
            )

        # Next token loss on the neuralese (MSELoss)
        next_tok_mse_BS = F.mse_loss(
            output_neuralese_BSd[:, :-1, :],
            input_neuralese_BSd[:, 1:, :],
            reduction="none",
        ).mean(dim=-1)
        input_neuralese_norms_BS = input_neuralese_BSd[:, 1:, :].norm(dim=-1)
        next_tok_mse_normalized_BS = next_tok_mse_BS / input_neuralese_norms_BS.pow(2)
        mean_resid_BSd = mean_resid_d.expand(*target_tokens_BS[:, 1:].shape, -1)
        variance_BS = F.mse_loss(
            mean_resid_BSd, input_neuralese_BSd[:, 1:, :], reduction="none"
        ).mean(dim=-1)
        next_tok_fvu_BS = next_tok_mse_BS / variance_BS

        # Ignore token positions which are masked out or where the next token is masked out
        next_token_mask_BS = target_attn_mask_BS[:, :-1] & target_attn_mask_BS[:, 1:]
        next_tok_mse_BS = next_tok_mse_BS * next_token_mask_BS
        next_tok_mse_normalized_BS = next_tok_mse_normalized_BS * next_token_mask_BS
        next_tok_fvu_BS = next_tok_fvu_BS * next_token_mask_BS

        mse_loss_sum += next_tok_mse_BS.sum()
        mse_loss_normalizd_sum += next_tok_mse_normalized_BS.sum()
        fvu_sum += next_tok_fvu_BS.sum()
        non_masked_count += next_token_mask_BS.sum()

        if non_masked_count.sum() > config.measure_reconstruction_min_toks:
            break

    avg_mse_loss = mse_loss_sum / non_masked_count
    avg_mse_loss_normalized = mse_loss_normalizd_sum / non_masked_count
    avg_fvu = fvu_sum / non_masked_count

    return avg_mse_loss, avg_mse_loss_normalized, avg_fvu


def run_evaluation(config: Config, device: str) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
    target_model = load_model(config.target_model_name, config.dtype, device)
    translator = Translator.from_pretrained(config, device)

    dataloader = get_data(config, target_model)

    mse_loss, mse_loss_normalized, fvu = measure_neuralese_reconstruction(
        dataloader=dataloader,
        target=target_model,
        translator=translator,
        config=config,
        visualize=True,
    )

    return mse_loss, mse_loss_normalized, fvu


if __name__ == "__main__":
    device = "cuda:7" if t.cuda.is_available() else "cpu"
    # ".translators/2025-01-13_16-32-40.pt"
    config = Config.from_repo_path_str(
        ".translators/2025-01-14_03-58-04.pt", dataset_split="validation"
    )
    mse_loss, mse_loss_normalized, fvu = run_evaluation(config, device)
    print(f"MSE loss: {mse_loss:.4f}, MSE loss normalized: {mse_loss_normalized:.4f}")
    fvu_perc = fvu.item() * 100
    fve_perc = 100 - fvu_perc
    print(f"FVU: {fvu:.4f}, FVU Perc: {fvu_perc:.2f}%, FVE Perc: {fve_perc:.2f}%")
