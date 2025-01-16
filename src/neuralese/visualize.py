from pathlib import Path

import torch as t
from PIL import Image, ImageDraw
from transformer_lens import HookedTransformer

from neuralese.file_utils import ensure_dir_exists, repo_path_to_abs_path


def get_rdbu_color(value: float) -> tuple[int, int, int]:
    """
    Convert a value in [0, 1] to a red-white-blue color.
    0.0 -> blue
    0.5 -> white
    1.0 -> red
    """
    value = 2 * value - 1  # Convert to [-1, 1]
    if value < 0:  # Blue to white
        return (
            int(255 * (1 + value)),  # R: 0 -> 255
            int(255 * (1 + value)),  # G: 0 -> 255
            255,  # B: 255 constant
        )
    else:  # White to red
        return (
            255,  # R: 255 constant
            int(255 * (1 - value)),  # G: 255 -> 0
            int(255 * (1 - value)),  # B: 255 -> 0
        )


def visualize_vector_reconstruction(
    target_resid_BSD: t.Tensor,
    translator_prediction_BSD: t.Tensor,
    n_vectors_cutoff: int = 10,
    vector_cutoff: int = 100,
    save_repo_path: Path | str | None = None,
    target_tokens_BS: t.Tensor | None = None,
    target_model: HookedTransformer | None = None,
) -> Image.Image:
    """
    Visualize the correct target vectors and the predicted vectors. Each pair is placed
    side by side, with space between each pair.
    Each vector is cut off at the vector_cutoff value and there are at most n_vectors_cutoff
    pairs of vectors.
    If target_tokens_BS and target_model are provided, the decoded token is shown above
    each pair of vectors.
    """
    assert (target_tokens_BS is None) == (target_model is None)
    if isinstance(save_repo_path, str):
        save_repo_path = repo_path_to_abs_path(save_repo_path)

    # Constants for visualization
    PIXELS_PER_ELEMENT = 3  # Each vector element gets this many pixels in height
    VECTOR_WIDTH = 20
    VECTOR_GAP = 4  # Gap between target and prediction vectors
    PAIR_SPACING = 2 * VECTOR_WIDTH
    TEXT_HEIGHT = 10

    # Get dimensions
    batch_size, seq_len, d_model = target_resid_BSD.shape
    n_vectors = min(
        n_vectors_cutoff, batch_size * (seq_len - 1)
    )  # -1 since predictions are offset
    d_viz = min(vector_cutoff, d_model)  # Number of vector elements to show

    # Calculate total height based on number of elements
    VECTOR_HEIGHT = d_viz * PIXELS_PER_ELEMENT

    # Flatten and normalize vectors, handling the offset
    target_flat = target_resid_BSD[:, 1:, :d_viz].reshape(-1, d_viz)[
        :n_vectors
    ]  # Skip first token
    pred_flat = translator_prediction_BSD[:, :-1, :d_viz].reshape(-1, d_viz)[
        :n_vectors
    ]  # Skip last prediction

    # Normalize to [-1, 1] for visualization
    def normalize_tensor(x: t.Tensor) -> t.Tensor:
        abs_max = t.max(t.abs(x))
        return x / abs_max

    target_flat = normalize_tensor(target_flat)
    pred_flat = normalize_tensor(pred_flat)

    # Convert to [0, 1] for color mapping
    target_flat = (target_flat + 1) / 2
    pred_flat = (pred_flat + 1) / 2

    # Create image
    total_width = n_vectors * (2 * VECTOR_WIDTH + VECTOR_GAP + PAIR_SPACING)
    total_height = VECTOR_HEIGHT + TEXT_HEIGHT
    img = Image.new("RGB", (total_width, total_height), "white")
    draw = ImageDraw.Draw(img)

    # Draw vectors
    for i in range(n_vectors):
        x_offset = i * (2 * VECTOR_WIDTH + VECTOR_GAP + PAIR_SPACING)

        # Draw target vector
        target_vec = target_flat[i].cpu().numpy()
        for j in range(d_viz):
            y = j * PIXELS_PER_ELEMENT
            color = get_rdbu_color(target_vec[j])
            draw.rectangle(
                [(x_offset, y), (x_offset + VECTOR_WIDTH, y + PIXELS_PER_ELEMENT - 1)],
                fill=color,
            )

        # Draw predicted vector
        pred_vec = pred_flat[i].cpu().numpy()
        for j in range(d_viz):
            y = j * PIXELS_PER_ELEMENT
            color = get_rdbu_color(pred_vec[j])
            draw.rectangle(
                [
                    (x_offset + VECTOR_WIDTH + VECTOR_GAP, y),
                    (
                        x_offset + 2 * VECTOR_WIDTH + VECTOR_GAP,
                        y + PIXELS_PER_ELEMENT - 1,
                    ),
                ],
                fill=color,
            )

        # Add token text if provided
        if target_tokens_BS is not None and target_model is not None:
            batch_idx = i // (seq_len - 1)
            seq_idx = i % (seq_len - 1) + 1  # +1 because we skipped first token
            token = target_tokens_BS[batch_idx, seq_idx].item()
            token_str = target_model.to_string(token)  # type: ignore
            draw.text((x_offset, VECTOR_HEIGHT), token_str, fill="black")  # type: ignore

    # Save if path provided
    if save_repo_path is not None:
        assert isinstance(save_repo_path, Path)
        ensure_dir_exists(save_repo_path.parent)
        img.save(save_repo_path)

    return img
