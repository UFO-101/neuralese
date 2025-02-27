# %%

from typing import Any, Tuple

import torch as t
import torch.nn.functional as F
from jaxtyping import Float
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from transformers import PreTrainedTokenizerBase

from neuralese.next_act_prediction.config import Config
from neuralese.next_act_prediction.translator import Translator, load_model

DEBUG = False


def find_token_position(
    full_text: str,
    text: str,
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[int, int]:
    """Find the position of text tokens within full_text tokens.

    Args:
        full_text: The full text containing the template
        text: The original text to find within the template
        tokenizer: Tokenizer to use for both texts

    Returns:
        Tuple of (start_idx, end_idx) indicating the token positions

    Raises:
        ValueError: If text cannot be found in full_text
    """
    # Tokenize both texts
    full_tokens = tokenizer(full_text, return_tensors="pt").input_ids
    text_tokens = tokenizer(text, return_tensors="pt").input_ids
    text_len = text_tokens.shape[1]

    # Find the position of text_tokens within full_tokens
    for i in range(full_tokens.shape[1] - text_len + 1):
        if t.equal(full_tokens[0, i : i + text_len], text_tokens[0]):
            return i, i + text_len
    raise ValueError(f"Could not find text '{text}' in full text '{full_text}'")


def process_neuralese_text(
    target_model: HookedTransformer,
    target_tokenizer: PreTrainedTokenizerBase,
    text: str,
    cfg: Config,
    neuralese_to_translate: str | None = None,
    use_chat_template: bool = False,
) -> t.Tensor:
    """Process input text through target model to get neuralese activations.

    Args:
        target_model: Model to get activations from
        target_tokenizer: Tokenizer for the target model
        text: Input text to process
        cfg: Config containing model settings
        neuralese_to_pass_to_translator: If provided, only return activations for this substring
        use_chat_template: Whether to wrap the text in a chat template

    Returns:
        Tensor of shape (1, seq_len, d_model) containing the intermediate activations
    """
    device = target_model.cfg.device
    weight_vector = target_model.blocks[cfg.mid_layer].mlp.W_in.T[0]
    print("weight_vector.shape:", weight_vector.shape)
    return weight_vector.unsqueeze(0).unsqueeze(0)

    # Apply chat template if requested
    if use_chat_template:
        full_text = target_tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=False,
        )
        assert isinstance(full_text, str)
    else:
        full_text = text

    # Find positions of the text we want activations for
    text_to_find = text if neuralese_to_translate is None else neuralese_to_translate
    start_idx, end_idx = find_token_position(full_text, text_to_find, target_tokenizer)

    # Tokenize and get activations
    neuralese_tokenized = target_tokenizer(full_text, return_tensors="pt")
    neuralese_toks: t.Tensor = neuralese_tokenized.input_ids.to(device)

    # Get model activations
    with t.no_grad():
        neuralese_acts = target_model(neuralese_toks, stop_at_layer=cfg.mid_layer)

    # Extract just the activations for the target text
    neuralese_acts = neuralese_acts[:, start_idx:end_idx, :]

    if cfg.layernorm_neuralese:
        d_model = target_model.cfg.d_model
        neuralese_acts = F.layer_norm(neuralese_acts, (d_model,))

    if DEBUG:
        print("process neuralese text")
        print("use_chat_template:", use_chat_template)
        print("full_text:", full_text)
        print("text_to_find:", text_to_find)
        print("full_text tokens:", neuralese_toks)
        print("neuralese_activations:", neuralese_acts.shape)
        print("slice tokens:", neuralese_toks[0, start_idx:end_idx])
        print("slice", target_tokenizer.decode(neuralese_toks[0, start_idx:end_idx]))

    return neuralese_acts


def get_chat_template_postfix_len(
    tokenizer: PreTrainedTokenizerBase,
    templated_text: str,
    sample_text: str = "test",
) -> int:
    """Calculate the number of tokens added after the input in a chat template.

    Args:
        tokenizer: The tokenizer with the chat template
        templated_text: The full text with chat template applied
        sample_text: Text to find within the template

    Returns:
        Number of tokens that appear after the input text
    """
    # Find where the text appears in the template
    start_idx, end_idx = find_token_position(templated_text, sample_text, tokenizer)

    # Get full template tokens
    template_tokens = tokenizer(templated_text, return_tensors="pt").input_ids

    # Calculate postfix length
    postfix_len = template_tokens.shape[1] - end_idx

    return postfix_len


def create_prompt_structure(
    translator_tokenizer: PreTrainedTokenizerBase,
    prefix: str,
    postfix: str,
    device: str,
) -> Tuple[t.Tensor, t.Tensor, int]:
    """Create and tokenize the prompt structure."""
    # Handle prefix
    messages = [{"role": "user", "content": prefix}]
    prefix_chat_text = translator_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    assert isinstance(prefix_chat_text, str)
    prefix_chat_tokenized = translator_tokenizer(prefix_chat_text, return_tensors="pt")
    prefix_chat_toks: t.Tensor = prefix_chat_tokenized.input_ids.to(device)

    # Handle postfix
    postfix_tokenized = translator_tokenizer(postfix, return_tensors="pt")
    postfix_toks: t.Tensor = postfix_tokenized.input_ids.to(device)

    # Calculate chat template postfix length
    chat_postfix_len = get_chat_template_postfix_len(
        translator_tokenizer,
        prefix_chat_text,
        prefix,
    )

    if DEBUG:
        print("\n\ncreate prompt structure")
        print("prefix_chat_text:", prefix_chat_text)
        print("prefix_chat_toks:", prefix_chat_toks)
        print("chat_postfix_len:", chat_postfix_len)
        print(
            "prefix_chat_toks decoded up to length:",
            translator_tokenizer.decode(prefix_chat_toks[0, :-chat_postfix_len]),
        )
        print(
            "rest of prefix_chat_toks decoded:",
            translator_tokenizer.decode(prefix_chat_toks[0, -chat_postfix_len:]),
        )

    return prefix_chat_toks, postfix_toks, chat_postfix_len


def create_combined_sequence(
    prefix_toks: t.Tensor,
    neuralese_len: int,
    postfix_toks: t.Tensor,
    chat_postfix_len: int,
    translator_tokenizer: PreTrainedTokenizerBase,
    random_tok: int = 13,
) -> Tuple[t.Tensor, slice]:
    """Combine tokens into final sequence and return neuralese slice."""
    device = prefix_toks.device

    # Create padding for neuralese section
    tok_padding_1S = (
        t.ones((1, neuralese_len), dtype=t.long, device=device) * random_tok
    )

    # Combine all parts
    combined_toks = t.cat(
        [
            prefix_toks[:, :-chat_postfix_len],
            tok_padding_1S,
            postfix_toks,
            prefix_toks[:, -chat_postfix_len:],
        ],
        dim=1,
    )

    # Calculate neuralese slice
    neuralese_slice = slice(
        prefix_toks.shape[1] - chat_postfix_len,
        prefix_toks.shape[1] - chat_postfix_len + neuralese_len,
    )

    if DEBUG:
        print("\n\ncreate combined sequence")
        print("combined_toks:", combined_toks)
        print("neuralese_slice:", neuralese_slice)
        print("combined_toks decoded:", translator_tokenizer.decode(combined_toks[0]))
        print(
            "neuralese_slice decoded:",
            translator_tokenizer.decode(combined_toks[0, neuralese_slice]),
        )

    return combined_toks, neuralese_slice


def generate_translation(
    translator_model: HookedTransformer,
    combined_toks: t.Tensor,
    neuralese_activations: t.Tensor,
    neuralese_slice: slice,
    max_new_tokens: int = 100,
) -> str:
    """Generate translation from combined sequence."""

    def hook(tensor: Float[t.Tensor, "batch pos d_model"], *, hook: HookPoint) -> Any:
        if tensor.shape[1] == combined_toks.shape[1]:
            if DEBUG:
                print("\n\nhook")
                print("tensor shape:", tensor.shape)
                print("neuralese_activations shape:", neuralese_activations.shape)
                print("neuralese_slice:", neuralese_slice)
            neuralese_input = translator.project_in(neuralese_activations)
            tensor[:, neuralese_slice, :] = neuralese_input
        else:
            assert tensor.shape[1] == 1
        return tensor

    translator_model.reset_hooks()
    translator_model.add_hook("blocks.0.hook_resid_pre", hook)

    # Generate translation
    with t.no_grad():
        out = translator_model.generate(
            combined_toks, max_new_tokens=max_new_tokens, return_type="str"
        )
    assert isinstance(out, str)
    return out


def combined_prompt(
    translator: Translator,
    target_model: HookedTransformer,
    cfg: Config,
    prefix: str = "Read the following text:\n\n",
    postfix: str = "\n\nExplain it in your own words.",
    neuralese_text: str = "Harry Potter is a series of seven fantasy novels written by J.K. Rowling",
    neuralese_to_translate: str | None = None,
):
    """Process text through target model and generate translation."""
    translator_model = translator.transformer
    translator_tokenizer = translator.tokenizer
    target_tokenizer = target_model.tokenizer
    assert isinstance(translator_tokenizer, PreTrainedTokenizerBase)
    assert isinstance(target_tokenizer, PreTrainedTokenizerBase)

    # Process through target model
    neuralese_activations = process_neuralese_text(
        target_model,
        target_tokenizer,
        neuralese_text,
        cfg,
        neuralese_to_translate=neuralese_to_translate,
        use_chat_template=("Instruct" in cfg.target_model_name),
    )

    # Create prompt structure
    prefix_toks, postfix_toks, chat_postfix_len = create_prompt_structure(
        translator_tokenizer,
        prefix,
        postfix,
        device,
    )

    # Create combined sequence
    combined_toks, neuralese_slice = create_combined_sequence(
        prefix_toks,
        neuralese_activations.shape[1],
        postfix_toks,
        chat_postfix_len,
        translator_tokenizer,
    )

    out = generate_translation(
        translator_model,
        combined_toks,
        neuralese_activations,
        neuralese_slice,
        max_new_tokens=100,
    )
    print("\n\ntranslator output:", out)


# %%
if __name__ == "__main__":
    device = "cuda:7" if t.cuda.is_available() else "cpu"
    config = Config.from_repo_path_str(".translators/2025-01-21_03-07-10.pt")
    target_model = load_model(config.target_model_name, config.dtype, device)
    translator = Translator.from_pretrained(config, device)
    # translator = Translator(target_model.cfg.d_model, config, device)

    # %%
    combined_prompt(
        translator,
        target_model,
        config,
        prefix="The following are the following:\n\n",
        postfix="\n\nExplain it in your own words.",
        neuralese_text="Harry Potter is a series of seven fantasy novels written by J.K. Rowling",
        neuralese_to_translate=" of seven",
    )

# %%
