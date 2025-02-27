# %%
#!%load_ext autoreload
#!%autoreload 2
from typing import List

import torch
from fairseq2.generation import TopKSampler
from sonar.inference_pipelines.text import (
    EmbeddingToTextModelPipeline,
    TextToEmbeddingModelPipeline,
)

current_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = lambda self: f"{list(self.shape)} {current_repr(self)}"  # type: ignore

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sampler = TopKSampler(k=1)


def init_sonar_models(
    encoder: str = "text_sonar_basic_encoder",
    decoder: str = "text_sonar_basic_decoder",
    tokenizer: str = "text_sonar_basic_encoder",
) -> tuple[TextToEmbeddingModelPipeline, EmbeddingToTextModelPipeline]:
    """Initialize SONAR text-to-embedding and embedding-to-text models."""
    t2vec = TextToEmbeddingModelPipeline(
        encoder=encoder, tokenizer=tokenizer, device=DEVICE
    )
    vec2text = EmbeddingToTextModelPipeline(
        decoder=decoder, tokenizer=tokenizer, device=DEVICE
    )
    return t2vec, vec2text


def explore_embedding_space(
    vec2text_model: EmbeddingToTextModelPipeline,
    start_point: torch.Tensor,
    direction: torch.Tensor | None = None,
    num_steps: int = 10,
    step_size: float = 0.1,
) -> List[str]:
    """
    Explore the embedding space by moving along a vector and converting back to text.
    If no direction is provided, generates a random direction vector.

    Args:
        vec2text_model: Model to convert embeddings back to text
        start_point: Starting embedding vector (shape: [1, dim])
        direction: Optional direction vector. If None, a random one is generated
        num_steps: Number of steps to take along the direction
        step_size: Size of each step
    """
    if direction is None:
        direction = torch.randn_like(start_point)
        # Normalize the direction vector
        direction = direction / torch.norm(direction)
    assert isinstance(direction, torch.Tensor)

    # Create steps tensor [0, 1, 2, ..., num_steps-1]
    steps = torch.arange(num_steps, device=start_point.device, dtype=torch.float)
    # Expand start_point and direction to match steps [num_steps, 1, dim]
    start_expanded = start_point.expand(num_steps, -1, -1)
    direction_expanded = direction.expand(num_steps, -1, -1)

    # Calculate all points at once [num_steps, 1, dim]
    points = start_expanded + steps[:, None, None] * step_size * direction_expanded
    # Flatten [num_steps, 1, dim] -> [num_steps, dim]
    points = points.reshape(num_steps, -1)
    print(points.shape)
    print(points)

    # Get all texts at once
    texts = vec2text_model.predict(
        points,
        target_lang="eng_Latn",
        max_seq_len=512,
        progress_bar=True,
        sampler=sampler,
    )
    return [f"Step {i}: {text}" for i, text in enumerate(texts)]


def word_vector_arithmetic(
    t2vec_model: TextToEmbeddingModelPipeline,
    vec2text_model: EmbeddingToTextModelPipeline,
    positive_words: List[str],
    negative_words: List[str],
) -> List[str]:
    """
    Perform word vector arithmetic (like king - man + woman = queen).

    Args:
        t2vec_model: Model to convert text to embeddings
        vec2text_model: Model to convert embeddings back to text
        positive_words: Words to add to the equation
        negative_words: Words to subtract from the equation
        num_results: Number of results to return when converting back to text

    Returns:
        List of resulting texts from the vector arithmetic
    """
    # Get embeddings for all words
    pos_embeddings = t2vec_model.predict(positive_words, source_lang="eng_Latn")
    if len(negative_words) > 0:
        neg_embeddings = t2vec_model.predict(negative_words, source_lang="eng_Latn")
    else:
        neg_embeddings = torch.zeros_like(pos_embeddings)

    # Sum positive and negative embeddings
    result_embedding = torch.zeros_like(pos_embeddings)
    for embedding in pos_embeddings:
        result_embedding += embedding
    for embedding in neg_embeddings:
        result_embedding -= embedding

    # Normalize the resulting vector
    # result_embedding = result_embedding / torch.norm(result_embedding)

    # Convert back to text
    results = vec2text_model.predict(
        result_embedding,
        target_lang="eng_Latn",
        max_seq_len=512,
        progress_bar=True,
        sampler=sampler,
    )

    return results


# %%
if __name__ == "__main__":
    t2vec_model, vec2text_model = init_sonar_models()
    # %%

    sentences = ["My name is SONAR.", "I can embed the sentences into vectorial space."]
    embeddings = t2vec_model.predict(sentences, source_lang="eng_Latn")

    # %%
    # Explore embedding space starting from our example sentence
    results = explore_embedding_space(
        vec2text_model=vec2text_model,
        start_point=embeddings[0:1],  # Take first embedding and keep batch dimension
        num_steps=100,
        step_size=0.01,
    )
    for result in results:
        print(result)

    # %%
    # Test word vector arithmetic
    results = word_vector_arithmetic(
        t2vec_model=t2vec_model,
        vec2text_model=vec2text_model,
        # positive_words=["king", "woman"],
        positive_words=["I'm going to take my dog for a walk", "txt u ltr k"],
        negative_words=[""],
        # negative_words=[],
    )
    print("\nWord vector arithmetic results (king - man + woman):")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result}")

# %%
