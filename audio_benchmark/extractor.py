"""Feature extraction logic using streaming memory-mapped representations."""

from __future__ import annotations

import math
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
) -> tuple[list[torch.Tensor], list[dict[str, Any]]]:
    """Extract representations from a dataset using a PyTorch representation model.

    To support arbitrarily large datasets without host RAM exhaustion, representations
    are streamed directly into memory-mapped arrays on disk as each batch finishes.
    The model returns a list of representation tensors of shape (B, C_i, T_i) with
    sequence length last, which are transposed and returned as a list of memory-mapped
    tensors of shape (N, T_i, C_i).

    Args:
        model: PyTorch module mapping audio (B, 1, T) to a list of representations (B, C_i, T_i).
        dataset: Dataset supplying audio samples and metadata.
        batch_size: Batch size for inference.
        device: Device to run inference on ('auto', 'cpu', 'cuda', etc.).
        num_workers: Number of workers for DataLoader.
        mmap_path: Optional explicit filesystem path or directory for the memory-mapped `.npy` files.
            If None, temporary files are automatically created and cleaned up.
        show_progress: Whether to display a tqdm progress bar during feature extraction.

    Returns:
        tuple[list[torch.Tensor], list[dict[str, Any]]]:
            - representations: List of memory-mapped Tensors of shape (N, T_i, C_i).
            - metadata: List of metadata dicts of length N.

    Raises:
        InvalidModelError: If model is not a torch.nn.Module.
        InvalidModelOutputError: If model output is not a non-empty list of 3D torch.Tensor objects
            with shape (B, C', T') or if batch dimensions mismatch.
    """
    if not isinstance(model, torch.nn.Module):
        raise InvalidModelError(
            f"Expected model to be an instance of torch.nn.Module, got {type(model).__name__}."
        )

    # Validate dataset upfront
    validate_dataset(dataset)

    total_samples = len(dataset)
    if total_samples == 0:
        return [], []

    target_device = resolve_device(device)
    if show_progress:
        print(f"Extracting representations on device: {target_device}")
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

    metadata_list: list[dict[str, Any]] = []
    is_temp_mmap = False
    target_mmap_paths: list[Path] = []
    mms: list[np.memmap] = []
    t_primes: list[int] = []
    c_primes: list[int] = []
    num_reps = 0
    cursor = 0

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

                if not isinstance(output, list):
                    raise InvalidModelOutputError(
                        f"Model must return a list of torch.Tensor, but returned {type(output).__name__}."
                    )
                if len(output) == 0:
                    raise InvalidModelOutputError(
                        "Model must return a non-empty list of torch.Tensor."
                    )

                batch_size_actual = audio_batch.shape[0]

                for rep_idx, rep_tensor in enumerate(output):
                    if not isinstance(rep_tensor, torch.Tensor):
                        raise InvalidModelOutputError(
                            f"Representation at index {rep_idx} must be a torch.Tensor, "
                            f"but got {type(rep_tensor).__name__}."
                        )
                    if rep_tensor.ndim != 3:
                        raise InvalidModelOutputError(
                            f"Representation at index {rep_idx} must be 3-dimensional with shape (B, C', T'), "
                            f"got shape {tuple(rep_tensor.shape)} (batch_size={batch_size_actual})."
                        )
                    if rep_tensor.shape[0] != batch_size_actual:
                        raise InvalidModelOutputError(
                            f"Representation at index {rep_idx} batch dimension ({rep_tensor.shape[0]}) does not match "
                            f"input batch size ({batch_size_actual})."
                        )

                # Transpose model output representations from (B, C_i, T_i) to (B, T_i, C_i)
                # to decouple model sequence-last convention from downstream task processing.
                output = [rep.transpose(1, 2) for rep in output]

                if len(mms) == 0:
                    num_reps = len(output)
                    if mmap_path is None:
                        is_temp_mmap = True
                        for _ in range(num_reps):
                            with tempfile.NamedTemporaryFile(
                                suffix=".npy", delete=False
                            ) as tmp:
                                target_mmap_paths.append(Path(tmp.name))
                    else:
                        is_temp_mmap = False
                        mmap_base = Path(mmap_path)
                        mmap_base.parent.mkdir(parents=True, exist_ok=True)
                        if mmap_base.suffix == ".npy":
                            target_mmap_paths = [
                                mmap_base.parent / f"{mmap_base.stem}_{i}.npy"
                                for i in range(num_reps)
                            ]
                        else:
                            mmap_base.mkdir(parents=True, exist_ok=True)
                            target_mmap_paths = [
                                mmap_base / f"representations_{i}.npy"
                                for i in range(num_reps)
                            ]

                    for rep_idx, rep_tensor in enumerate(output):
                        t_init = rep_tensor.shape[1]
                        c_init = rep_tensor.shape[2]
                        t_primes.append(t_init)
                        c_primes.append(c_init)
                        mm = np.lib.format.open_memmap(
                            target_mmap_paths[rep_idx],
                            mode="w+",
                            dtype=np.float32,
                            shape=(total_samples, t_init, c_init),
                        )
                        mms.append(mm)
                elif len(output) != num_reps:
                    raise InvalidModelOutputError(
                        f"Inconsistent number of representations returned across batches: "
                        f"expected {num_reps}, got {len(output)}."
                    )

                max_audio_len = audio_batch.shape[-1]
                curr_ts = [rep.shape[1] for rep in output]
                for meta in meta_batch:
                    audio_len = meta.get("audio_length", max_audio_len)
                    if max_audio_len > 0 and audio_len < max_audio_len:
                        scale = audio_len / max_audio_len
                        rep_lengths = [
                            max(1, min(t, math.ceil(scale * t))) for t in curr_ts
                        ]
                    else:
                        rep_lengths = list(curr_ts)
                    meta["rep_lengths"] = rep_lengths
                    meta["rep_length"] = rep_lengths[0]

                for rep_idx in range(num_reps):
                    rep_tensor = output[rep_idx]
                    current_t = rep_tensor.shape[1]
                    current_c = rep_tensor.shape[2]
                    t_prime = t_primes[rep_idx]
                    c_prime = c_primes[rep_idx]
                    target_mmap_path = target_mmap_paths[rep_idx]

                    if current_t > t_prime:
                        mms[rep_idx].flush()
                        resized_path = target_mmap_path.with_name(
                            f"{target_mmap_path.name}.resize"
                        )
                        new_mm = np.lib.format.open_memmap(
                            resized_path,
                            mode="w+",
                            dtype=np.float32,
                            shape=(total_samples, current_t, c_prime),
                        )
                        new_mm[:] = 0.0
                        if cursor > 0:
                            new_mm[:cursor, :t_prime, :] = mms[rep_idx][:cursor]
                        new_mm.flush()
                        old_mm = mms[rep_idx]
                        mms[rep_idx] = None
                        del old_mm
                        del new_mm
                        resized_path.replace(target_mmap_path)
                        t_primes[rep_idx] = current_t
                        t_prime = current_t
                        mms[rep_idx] = np.lib.format.open_memmap(
                            target_mmap_path,
                            mode="r+",
                            dtype=np.float32,
                            shape=(total_samples, t_prime, c_prime),
                        )

                    out_np = rep_tensor.detach().cpu().numpy().astype(np.float32)

                    if current_t == t_prime and current_c == c_prime:
                        mms[rep_idx][cursor : cursor + batch_size_actual] = out_np
                    else:
                        padded_np = np.zeros(
                            (batch_size_actual, t_prime, c_prime), dtype=np.float32
                        )
                        padded_np[:, :current_t, :] = out_np[:, :, :c_prime]
                        mms[rep_idx][cursor : cursor + batch_size_actual] = padded_np

                cursor += batch_size_actual
                metadata_list.extend(meta_batch)
    finally:
        model.train(prev_training_mode)

    for mm in mms:
        if mm is not None:
            mm.flush()
    del mms

    representations_list: list[torch.Tensor] = []
    for target_path in target_mmap_paths:
        mm_read = np.lib.format.open_memmap(target_path, mode="c")
        rep_tensor = torch.from_numpy(mm_read)
        if is_temp_mmap:
            weakref.finalize(rep_tensor, _cleanup_temp_file, target_path)
        representations_list.append(rep_tensor)

    return representations_list, metadata_list
