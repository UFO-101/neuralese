# %%
# from einops import einsum
from einops import einsum
from jaxtyping import Float
from torch import Tensor


def multiply_some_tensors(
    a: Float[Tensor, "batch dim"], b: Float[Tensor, "batch dim"]
) -> Float[Tensor, " batch"]:
    return einsum(a, b, "batch dim, batch dim -> batch")
