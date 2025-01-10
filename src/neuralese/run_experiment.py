# %%
#!%load_ext autoreload
#!%autoreload 2
import torch as t

from neuralese.generate_activations import load_model

device = "cuda" if t.cuda.is_available() else "cpu"
model = load_model("gpt2", device)

output = model.generate("Hello, world!", max_new_tokens=10)
print(output)


# %%
