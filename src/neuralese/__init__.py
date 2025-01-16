import torch as t

# Change repr for t.Tensor
current_repr = t.Tensor.__repr__
t.Tensor.__repr__ = lambda self: f"{list(self.shape)} {current_repr(self)}"  # type: ignore
