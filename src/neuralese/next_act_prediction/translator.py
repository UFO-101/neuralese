from typing import Any

import torch as t
import torch.nn as nn
from transformer_lens import HookedTransformer

from neuralese.next_act_prediction.config import Config
from neuralese.file_utils import ensure_dir_exists


def load_model(name: str, dtype: t.dtype, device: str) -> HookedTransformer:
    return HookedTransformer.from_pretrained_no_processing(
        name, dtype=dtype, device=device
    )


class Translator(nn.Module):
    def __init__(
        self,
        target_model_dim: int,
        config: Config,
        device: str,
    ):
        super().__init__()
        self.transformer: HookedTransformer = load_model(
            config.translator_model_name, config.dtype, device
        )
        if config.random_init_translator:
            self.transformer.cfg.init_mode = config.random_init_mode
            self.transformer.init_weights()

        self.translator_model_dim = self.transformer.cfg.d_model
        self.project_in: ProjectIn = ProjectIn(
            target_model_dim, self.translator_model_dim, config, device
        )
        self.project_out: ProjectOut = ProjectOut(
            target_model_dim, self.translator_model_dim, config, device
        )
        self.target_model_dim = target_model_dim
        self.n_layers = self.transformer.cfg.n_layers
        self.config = config

    def neuralese_to_neuralese(
        self, input_neuralese_BSd: t.Tensor, attn_mask_BS: t.Tensor
    ) -> t.Tensor:
        """
        Run the neuralese through the transformer model, projecting it into the
        translator model space and then back to the target model space.
        """
        # Rotary embeddings are applied part of attention, so we don't need to add them
        assert self.transformer.cfg.positional_embedding_type == "rotary"

        input_resid_BSD = self.project_in(input_neuralese_BSd)
        final_resid_BSD = self.transformer(
            input_resid_BSD,
            start_at_layer=0,
            stop_at_layer=self.n_layers,
            attention_mask=attn_mask_BS,
        )
        output_neuralese_BSd = self.project_out(final_resid_BSD)
        return output_neuralese_BSd

    def neuralese_to_logits(
        self, input_neuralese_BSd: t.Tensor, attn_mask_BS: t.Tensor
    ) -> t.Tensor:
        """
        Run the neuralese through the transformer model, projecting it into the
        translator model space and then run the model as normal.
        """
        # Rotary embeddings are applied part of attention, so we don't need to add them
        assert self.transformer.cfg.positional_embedding_type == "rotary"

        input_resid_BSD = self.project_in(input_neuralese_BSd)
        return self.transformer(
            input_resid_BSD,
            start_at_layer=0,
            attention_mask=attn_mask_BS,
        )

    def forward_tokens(self, *args: Any, **kwargs: Any) -> t.Tensor:
        """Run normal tokens through the transformer model."""
        return self.transformer(*args, **kwargs)

    @classmethod
    def from_pretrained(cls, config: Config, device: str) -> "Translator":
        state_dict = t.load(config.save_path, weights_only=True, map_location=device)
        target_model_dim = state_dict["target_model_dim"]
        translator = cls(target_model_dim, config, device)
        translator.load_state_dict(state_dict, strict=False)
        return translator

    def save_trained(self) -> None:
        # Add the target model dim and translator model dim to the state dict
        ensure_dir_exists(self.config.save_path.parent)
        state_dict = self.state_dict()
        state_dict["target_model_dim"] = self.target_model_dim
        t.save(state_dict, self.config.save_path)

    @property
    def device(self) -> str:
        return self.transformer.cfg.device  # type: ignore

    @property
    def tokenizer(self) -> Any:
        return self.transformer.tokenizer


class ProjectIn(nn.Module):
    def __init__(
        self,
        target_model_dim: int,
        translator_model_dim: int,
        config: Config,
        device: str,
    ):
        super().__init__()
        self.target_model_dim = target_model_dim
        self.translator_model_dim = translator_model_dim
        self.project_in = nn.Linear(
            target_model_dim,
            translator_model_dim,
            bias=True,
            device=device,
            dtype=config.dtype,
        )

    def forward(self, input_neuralese_BSd: t.Tensor) -> t.Tensor:
        return self.project_in(input_neuralese_BSd)


class ProjectOut(nn.Module):
    def __init__(
        self,
        target_model_dim: int,
        translator_model_dim: int,
        config: Config,
        device: str,
    ):
        super().__init__()
        self.target_model_dim = target_model_dim
        self.translator_model_dim = translator_model_dim
        self.project_out = nn.Linear(
            translator_model_dim,
            target_model_dim,
            bias=True,
            device=device,
            dtype=config.dtype,
        )

    def forward(self, final_resid_BSD: t.Tensor) -> t.Tensor:
        return self.project_out(final_resid_BSD)
