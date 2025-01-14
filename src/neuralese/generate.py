# %%

from typing import Any

import torch as t
from jaxtyping import Float
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from transformers import PreTrainedTokenizerBase

from neuralese.config import Config
from neuralese.translator import Translator, load_model


def combined_prompt(
    translator: Translator,
    target_model: HookedTransformer,
    cfg: Config,
    debug: bool = False,
):
    """Doesn't support batches currently (because ragged tensors are hard)."""
    translator_model = translator.transformer
    translator_tokenizer = translator.tokenizer
    assert isinstance(translator_tokenizer, PreTrainedTokenizerBase)
    target_tokenizer = target_model.tokenizer
    assert isinstance(target_tokenizer, PreTrainedTokenizerBase)

    device = translator_model.cfg.device
    assert target_model.cfg.device == device
    translator_chat_postfix_len = 5
    target_chat_postfix_len = 2
    random_tok = 13

    prefix = "Read the following:\n\n"
    messages = [{"role": "user", "content": prefix}]
    prefix_text = translator_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    assert isinstance(prefix_text, str)

    prefix_tokenized = translator_tokenizer(prefix_text, return_tensors="pt")
    prefix_toks: t.Tensor = prefix_tokenized.input_ids.to(device)

    neuralese_text = (
        "Harry Potter is a series of seven fantasy novels written by J.K. Rowling."
    )
    neuralese_tokenized = target_tokenizer(neuralese_text, return_tensors="pt")
    neuralese_toks: t.Tensor = neuralese_tokenized.input_ids.to(device)
    neuralese_len = neuralese_toks.shape[-1]
    neuralese_chat = target_tokenizer.apply_chat_template(
        [{"role": "user", "content": neuralese_text}],
        tokenize=False,
        add_generation_prompt=False,
    )
    assert isinstance(neuralese_chat, str)
    neuralese_chat_tokenized = target_tokenizer(neuralese_chat, return_tensors="pt")
    neuralese_chat_toks: t.Tensor = neuralese_chat_tokenized.input_ids.to(device)

    tok_padding_1S = (
        t.ones((1, neuralese_len), dtype=t.long, device=device) * random_tok
    )

    postfix = "\n\nExplain it in your own words."
    postfix_tokenized = translator_tokenizer(postfix, return_tensors="pt")
    postfix_toks: t.Tensor = postfix_tokenized.input_ids.to(device)

    combined_toks = t.cat(
        [
            prefix_toks[:, :-translator_chat_postfix_len],
            tok_padding_1S,
            postfix_toks,
            prefix_toks[:, -translator_chat_postfix_len:],
        ],
        dim=1,
    )
    neuralese_translator_slice = slice(
        prefix_toks.shape[1] - translator_chat_postfix_len,
        prefix_toks.shape[1] - translator_chat_postfix_len + neuralese_len,
    )
    if debug:
        neuralese_toks = combined_toks[:, neuralese_translator_slice]
        print("neuralese shape:", neuralese_toks.shape)
        print("neuralese text:", translator_tokenizer.decode(neuralese_toks[0]))

    if debug:
        combined = translator_tokenizer.decode(combined_toks[0])  # type: ignore
        print("combined text")
        print(combined)

    with t.no_grad():
        neuralese_1Sd = target_model(neuralese_chat_toks, stop_at_layer=cfg.mid_layer)
    neuralese_target_slice = slice(
        -(neuralese_len + target_chat_postfix_len), -target_chat_postfix_len
    )
    neuralese_1Sd = neuralese_1Sd[:, neuralese_target_slice, :]
    if debug:
        print(
            "neuralese chat toks decoded:",
            translator_tokenizer.decode(neuralese_chat_toks[0]),
        )
        print("neuralese 1Sd shape:", neuralese_1Sd.shape)
        neuralese_target_tokens_slice = neuralese_chat_toks[0, neuralese_target_slice]
        neuralese_slice_text = translator_tokenizer.decode(
            neuralese_target_tokens_slice
        )
        print("neuralese slice text:", neuralese_slice_text)

    combined_len = combined_toks.shape[1]

    # Add a forward hooks to the first layer
    def hook_function(
        tensor: Float[t.Tensor, "batch pos d_model"], *, hook: HookPoint
    ) -> Any | None:
        if tensor.shape[1] == combined_len:
            neuralese_input = translator.project_in(neuralese_1Sd)
            tensor[:, neuralese_translator_slice, :] = neuralese_input
        else:
            assert tensor.shape[1] == 1
        return tensor

    translator_model.reset_hooks()
    translator_model.add_hook("blocks.0.hook_resid_pre", hook_function)
    with t.no_grad():
        answer = translator_model.generate(
            combined_toks, max_new_tokens=100, return_type="str"
        )
    print("answer:", answer)


# %%
if __name__ == "__main__":
    device = "cuda:7" if t.cuda.is_available() else "cpu"
    config = Config.from_repo_path_str(".translators/2025-01-13_16-32-40.pt")
    target_model = load_model(config.target_model_name, config.dtype, device)
    translator = Translator.from_pretrained(config, device)
    # translator = Translator(target_model.cfg.d_model, config, device)

    # %%
    combined_prompt(translator, target_model, config, debug=False)
