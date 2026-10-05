"""Feature extraction logic using streaming memory-mapped representations."""

from __future__ import annotations

import tempfile
import weakref
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from audio_benchmark.dataset import collate_audio_batch, validate_dataset
from audio_benchmark.exceptions import InvalidModelError, InvalidModelOutputError
from audio_benchmark.utils import resolve_device


def _cleanup_temp_file(path_to_delete: Path) -> None:
    """Safely unlink a temporary memory-mapped file."""
    if path_to_delete.exists():
        try:
            path_to_delete.unlink()
        except OSError:
            pass


def extract_representations(
    model: torch.nn.Module,
    dataset: Dataset,
    batch_size: int = 16,
    device: str | torch.device = "auto",
    num_workers: int = 0,
    mmap_path: str | Path | None = None,
    show_progress: bool = True,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Extract representations from a dataset using a PyTorch representation model.

    To support arbitrarily large datasets without host RAM exhaustion, representations
    are streamed directly into a memory-mapped array on disk as each batch finishes.

    Args:
        model: PyTorch module mapping audio (B, 1, T) to representations (B, T', C').
        dataset: Dataset supplying audio samples and metadata.
        batch_size: Batch size for inference.
        device: Device to run inference on ('auto', 'cpu', 'cuda', etc.).
        num_workers: Number of workers for DataLoader.
        mmap_path: Optional explicit filesystem path for the memory-mapped `.npy` file.
            If None, a temporary file is automatically created and cleaned up.
        show_progress: Whether to display a tqdm progress bar during feature extraction.

    Returns:
        tuple[torch.Tensor, list[dict[str, Any]]]:
            - representations: Memory-mapped Tensor of shape (N, T', C').
            - metadata: List of metadata dicts of length N.

    Raises:
        InvalidModelError: If model is not a torch.nn.Module.
        InvalidModelOutputError: If model output does not have shape (B, T', C').
    """
    if not isinstance(model, torch.nn.Module):
        raise InvalidModelError(
            f"Expected model to be an instance of torch.nn.Module, got {type(model).__name__}."
        )

    # Validate dataset upfront
    validate_dataset(dataset)

    total_samples = len(dataset)
    if total_samples == 0:
        return torch.empty(0, 0, 0, dtype=torch.float32), []

    target_device = resolve_device(device)
    model = model.to(target_device)
    prev_training_mode = model.training
    model.eval()

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_audio_batch,
    )

    # Determine storage path
    is_temp_mmap = False
    if mmap_path is None:
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
            target_mmap_path = Path(tmp.name)
        is_temp_mmap = True
    else:
        target_mmap_path = Path(mmap_path)
        target_mmap_path.parent.mkdir(parents=True, exist_ok=True)

    metadata_list: list[dict[str, Any]] = []
    mm = None
    cursor = 0
    t_prime = 0
    c_prime = 0

    progress_bar = tqdm(
        dataloader,
        desc="Extracting features",
        disable=not show_progress,
    )

    try:
        with torch.inference_mode():
            for audio_batch, meta_batch in progress_bar:
                audio_batch = audio_batch.to(target_device)
                output = model(audio_batch)

                if not isinstance(output, torch.Tensor):
                    raise InvalidModelOutputError(
                        f"Model must return a torch.Tensor, but returned {type(output).__name__}."
                    )

                if output.ndim != 3:
                    raise InvalidModelOutputError(
                        f"Model output must be 3-dimensional with shape (B, T', C'), "
                        f"got shape {tuple(output.shape)} (batch_size={audio_batch.shape[0]})."
                    )

                if output.shape[0] != audio_batch.shape[0]:
                    raise InvalidModelOutputError(
                        f"Model output batch dimension ({output.shape[0]}) does not match "
                        f"input batch size ({audio_batch.shape[0]})."
                    )

                batch_size_actual = output.shape[0]
                current_t = output.shape[1]
                current_c = output.shape[2]

                # Initialize memory-mapped file on first batch
                if mm is None:
                    t_prime = current_t
                    c_prime = current_c
                    mm = np.lib.format.open_memmap(
                        target_mmap_path,
                        mode="w+",
                        dtype=np.float32,
                        shape=(total_samples, t_prime, c_prime),
                    )

                out_np = output.detach().cpu().numpy().astype(np.float32)

                # Write batch directly to disk
                if current_t == t_prime and current_c == c_prime:
                    mm[cursor : cursor + batch_size_actual] = out_np
                else:
                    # Handle possible variable-length temporal padding
                    if current_t < t_prime:
                        padded_np = np.zeros(
                            (batch_size_actual, t_prime, c_prime), dtype=np.float32
                        )
                        padded_np[:, :current_t, :] = out_np[:, :, :c_prime]
                        mm[cursor : cursor + batch_size_actual] = padded_np
                    else:
                        mm[cursor : cursor + batch_size_actual] = out_np[
                            :, :t_prime, :c_prime
                        ]

                cursor += batch_size_actual
                metadata_list.extend(meta_batch)
    finally:
        model.train(prev_training_mode)

    if mm is not None:
        mm.flush()
        del mm

    # Open with copy-on-write memory mapping
    mm_read = np.lib.format.open_memmap(target_mmap_path, mode="c")
    representations = torch.from_numpy(mm_read)

    # If this was a temporary file, register cleanup when the tensor is collected
    if is_temp_mmap:
        weakref.finalize(representations, _cleanup_temp_file, target_mmap_path)

    return representations, metadata_list
