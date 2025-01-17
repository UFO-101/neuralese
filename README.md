## Getting started

1. Install [UV](https://docs.astral.sh/uv/)

```bash
wget -qO- https://astral.sh/uv/install.sh | sh
```

2. Install the dependencies:

```bash
uv sync
```

3. Install the pre-commit hooks:

```bash
pre-commit install
```

## Successful training runs

 - First 2 run overnight on Jan 11th 2025.
https://wandb.ai/josephmiller101/neuralese/runs/nnohoevt (likely-plant-22)
https://wandb.ai/josephmiller101/neuralese/runs/0mfg877c (chocolate-frost-23)

 - Fixed some minor padding issues + playing with the KL weight.

[FVU ~70%]  https://wandb.ai/josephmiller101/neuralese/runs/9cejwht0 (KL weight 0.01)
[FVU ~73%] https://wandb.ai/josephmiller101/neuralese/runs/0pmufs9v (KL weight 1)

 - Increased the learning rate to 1e-5

[FVU ~42%] https://wandb.ai/josephmiller101/neuralese/runs/hyq370vw (KL weight 0.01)

 - Increased the learning rate to 1e-5 and decreased the KL weight to 1e-5
[FVU ~40%] (*current best*) https://wandb.ai/josephmiller101/neuralese/runs/yec6npti


## Ideas

 - [Gurkenglas] Use a discriminator to train the next activation prediction.
 - [Gurkenglas] Add a noise vector somewhere in the model to improve training.
 - Condition on the next token embedding
 - [Gurkenglas] Randomly initialize the weights of the translator model.
