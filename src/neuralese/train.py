# %%
from datetime import datetime
from typing import Any, Dict

import torch as t
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformer_lens import HookedTransformer

import wandb
from neuralese.config import Config
from neuralese.data.get_data import get_data
from neuralese.evaluate import measure_neuralese_recon
from neuralese.loss_fns import LOSS_FN_MAP, get_orig_translator_logprobs
from neuralese.translator import Translator, load_model


def train_translator(
    train_dataloader: DataLoader[Dict[str, Any]],
    val_dataloader: DataLoader[Dict[str, Any]],
    target: HookedTransformer,
    orig_translator: HookedTransformer,
    translator: Translator,
    config: Config,
) -> Translator:
    optim = t.optim.Adam(translator.parameters(), lr=config.learning_rate)

    last_save = 0
    curr_loss, best_loss = float("inf"), float("inf")
    for i, batch in (pbar := tqdm(enumerate(train_dataloader))):
        orig_translator_logprobs = get_orig_translator_logprobs(
            batch, orig_translator, translator, config
        )

        logs, desc, curr_loss = {}, [], 0

        for loss_type, loss_weight in config.loss_types:
            loss_fn = LOSS_FN_MAP[loss_type]
            loss = loss_fn(batch, orig_translator_logprobs, target, translator, config)
            loss *= loss_weight
            loss.backward()
            optim.step()
            optim.zero_grad()
            logs[f"{loss_type}_loss"] = loss.item()
            desc.append(f"{loss_type}: {loss.item():.4f}")
            curr_loss += loss.item()

        neuralese_recon = "mse" in config.loss_types or "dot_prod" in config.loss_types
        if i % config.eval_interval == 0 and i > 0 and neuralese_recon:
            evals = measure_neuralese_recon(val_dataloader, target, translator, config)
            logs.update(evals)
        wandb.log(logs)

        if i - last_save >= config.save_interval and curr_loss < best_loss:
            translator.save_trained()
            last_save = i
            best_loss = curr_loss

        pbar.set_description(" | ".join(desc))

    if curr_loss < best_loss:
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
    device = "cuda:3" if t.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    datatime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    config = Config.from_repo_path_str(f".translators/{datatime_str}.pt")
    # config = SmallModelConfig.from_repo_path_str(f".translators/{datatime_str}.pt")
    train_translator = run_training(config, device)
