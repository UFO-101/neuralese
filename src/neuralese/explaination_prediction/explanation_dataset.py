# %%
from typing import Dict, Literal, Union

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from neuralese.explaination_prediction.config import ExplanationTrainingConfig
from neuralese.explaination_prediction.explanations_data import load_explanations


class FeatureExplanationDataset(Dataset):
    def __init__(
        self,
        explanations: Dict[int, str],
        split: Literal["train", "val", "test"],
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        max_length: int = 128,
        debug: bool = True,
    ):
        """
        Dataset for feature vectors and their explanations.

        Args:
            feature_vectors: Tensor of feature vectors
            explanations: Dictionary mapping feature indices to explanations
            tokenizer: Tokenizer for encoding explanations
            max_length: Maximum length of tokenized explanations
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.debug = debug

        # Create prompt template
        self.prompt_template = "The vector <vector> represents {}"
        self.vector_token_substring = " <vector>"

        if split == "train":
            indices = range(0, 700_000)
        elif split == "val":
            indices = range(700_000, 720_000)
        else:
            assert split == "test"
            total_indices = len(explanations)
            indices = range(720_000, total_indices)
        split_explanations = {}
        for idx, (feature_idx, explanation) in enumerate(explanations.items()):
            if idx in indices:
                split_explanations[feature_idx] = explanation

        self.explanations = split_explanations
        self.feature_indices = list(split_explanations.keys())

    @classmethod
    def from_config(
        cls,
        config: ExplanationTrainingConfig,
        split: Literal["train", "val", "test"],
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    ) -> "FeatureExplanationDataset":
        explanations = load_explanations(
            model_name=config.model_name,
            layer=config.layer,
            activation_site=config.activation_site,
            n_features=config.n_features,
            num_batches=config.num_batches,
            cache_dir=config.cache_dir,
        )
        return cls(explanations, split=split, tokenizer=tokenizer)

    def __len__(self) -> int:
        return len(self.feature_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | int]:
        feature_idx = self.feature_indices[idx]
        explanation = self.explanations[feature_idx]

        # Tokenize input and target
        prompt = self.prompt_template.format(explanation)
        # Split prompt into before and after the vector
        before, after = prompt.split(self.vector_token_substring)
        # Tokenize before and after, making sure not to prepend the BOS token to the after part
        before_tokenized = self.tokenizer.encode(before, return_tensors="pt")
        after_tokenized = self.tokenizer.encode(
            after, add_special_tokens=False, return_tensors="pt"
        )
        assert type(before_tokenized) is dict
        assert type(after_tokenized) is dict

        # Combine prompt and target with a dummy token in the middle where the vector will go
        input_ids = torch.cat(
            [
                before_tokenized["input_ids"],
                torch.tensor([self.tokenizer.eos_token_id]),
                after_tokenized["input_ids"],
            ]
        )
        dtype = input_ids.dtype
        device = input_ids.device
        attention_mask = torch.cat(
            [
                before_tokenized["attention_mask"],
                torch.ones((input_ids.shape[0], 1), dtype=dtype, device=device),
                after_tokenized["attention_mask"],
            ]
        )

        if self.debug:
            print(f"Input IDs: {input_ids}")
            print(f"Input IDs shape: {input_ids.shape}")
            print("Decoded input: ", self.tokenizer.decode(input_ids))

            print("Attention mask shape: ", attention_mask.shape)
            print("Attention mask: ", attention_mask)

        labels = self.tokenizer.encode(
            explanation, add_special_tokens=False, return_tensors="pt"
        )
        assert type(labels) is dict
        labels = labels["input_ids"]

        # If the input is too long, throw an error
        if input_ids.shape[1] > self.max_length:
            raise ValueError(
                f"Input is too long. Max length {self.max_length} tokens, but the input is {input_ids.shape[1]} tokens long."
            )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "feature_idx": feature_idx,
        }


if __name__ == "__main__":
    # Load explanations
    config = ExplanationTrainingConfig()
    dataset = FeatureExplanationDataset.from_config(
        config=config,
        split="train",
        tokenizer=AutoTokenizer.from_pretrained(config.model_name),
    )
    print(len(dataset))
