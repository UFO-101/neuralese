# %%
#!%load_ext autoreload
#!%autoreload 2
import torch as t

from neuralese.generate_activations import multiply_some_tensors

device = t.device("cuda" if t.cuda.is_available() else "cpu")
a = t.randn(10, 10, device=device)
b = t.randn(10, 10, device=device)
c = multiply_some_tensors(a, b)
print(c)

# %%
