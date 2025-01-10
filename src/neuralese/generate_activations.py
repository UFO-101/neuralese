# %%
from transformer_lens import HookedTransformer


def load_model(model_name: str, device: str) -> HookedTransformer:
    return HookedTransformer.from_pretrained(model_name, device=device)


# %%
