# %%
import random

import torch
import wandb
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from neuralese.explaination_prediction.config import ExplanationTrainingConfig
from neuralese.explaination_prediction.explanation_dataset import (
    FeatureExplanationDataset,
)
from neuralese.explaination_prediction.explanations_data import (
    ActivationSite,
    load_explanations,
)
from neuralese.explaination_prediction.load_sae_weight_vectors import (
    sae_decoder_weights,
)
from neuralese.file_utils import ensure_dir_exists


def train_explanation_model(config: ExplanationTrainingConfig) -> None:
    """
    Train a model to predict explanations from feature vectors.

    Args:
        config: Configuration for training. If None, uses default config.

    Returns:
        Tuple containing the trained model and tokenizer
    """
    # Set random seed for reproducibility
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Set output directory
    ensure_dir_exists(config.save_dir)

    # Load feature vectors
    print("Loading SAE decoder weights...")
    decoder_weights = sae_decoder_weights(
        repo_id=config.sae_repo_id,
        layer=config.layer,
        width=config.width,
        l0_threshold=config.l0_threshold,
    )
    print("sae decoder weights loaded", decoder_weights.shape)

    # Load tokenizer and model
    print(f"Loading model and tokenizer: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForCausalLM.from_pretrained(config.model_name)

    # Create datasets
    train_dataset = FeatureExplanationDataset.from_config(
        config=config,
        split="train",
        tokenizer=tokenizer,
    )

    val_dataset = FeatureExplanationDataset.from_config(
        config=config,
        split="val",
        tokenizer=tokenizer,
    )

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=config.save_path.as_posix(),
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        logging_strategy=config.logging_strategy,
        evaluation_strategy=config.eval_strategy,
        save_strategy=config.save_strategy,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        learning_rate=config.learning_rate,
        report_to=config.report_to,  # Enable wandb logging
    )

    wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        config=vars(config),  # Pass the entire config object as a dictionary
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Train the model
    print("Starting training...")
    trainer.train()

    print(f"Training complete. Model saved to {config.save_path}")


if __name__ == "__main__":
    # Create a configuration with desired parameters
    config = ExplanationTrainingConfig()

    # Train the model
    train_explanation_model(config)

    # Test the model on a few examples
    print("\nTesting the model on a few examples:")

    # Load feature vectors and explanations for testing
    decoder_weights = sae_decoder_weights(
        layer=config.layer,
        width=config.width,
    )

    explanations = load_explanations(
        model_name="gemma-2-2b",
        layer=config.layer,
        activation_site=ActivationSite.RESIDUAL,
        n_features=config.width,
    )
