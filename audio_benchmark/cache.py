"""Representation caching system using memory-mapped storage for scalable audio benchmarks."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch


def sanitize_filename(name: str) -> str:
    """Sanitize strings for safe filesystem usage."""
    return re.sub(r"[^\w\-.]", "_", name.strip())


class RepresentationCache:
    """Caches and retrieves representations for a (model_name, dataset_name) pair using memory-mapping.

    Storage structure:
    `cache_dir/<safe_model>/<safe_dataset>/`
        - `representations.npy`: Memory-mapped NumPy array of shape (N, T', C').
        - `metadata.pt`: PyTorch serialized list of metadata dicts.
        - `manifest.json`: JSON metadata with shapes, dtypes, and configuration info.
    """

    def __init__(self, cache_dir: str | Path = ".benchmark_cache"):
        """Initialize the cache.

        Args:
            cache_dir: Directory where cached representations are stored.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_path(self, model_name: str, dataset_name: str) -> Path:
        """Get the directory path for a cached representation entry."""
        safe_model = sanitize_filename(model_name)
        safe_dataset = sanitize_filename(dataset_name)
        model_dir = self.cache_dir / safe_model
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir / safe_dataset

    def get_memmap_path(self, model_name: str, dataset_name: str) -> Path:
        """Get the filesystem path for the memory-mapped representations file."""
        return self.get_cache_path(model_name, dataset_name) / "representations.npy"

    def get_metadata_path(self, model_name: str, dataset_name: str) -> Path:
        """Get the filesystem path for the serialized metadata file."""
        return self.get_cache_path(model_name, dataset_name) / "metadata.pt"

    def _get_legacy_path(self, model_name: str, dataset_name: str) -> Path:
        """Get legacy single .pt cache path for backwards compatibility."""
        safe_model = sanitize_filename(model_name)
        safe_dataset = sanitize_filename(dataset_name)
        return self.cache_dir / safe_model / f"{safe_dataset}.pt"

    def has(self, model_name: str, dataset_name: str) -> bool:
        """Check if cached representations exist (supports both memmap and legacy .pt)."""
        memmap_file = self.get_memmap_path(model_name, dataset_name)
        if memmap_file.is_file():
            return True
        legacy_file = self._get_legacy_path(model_name, dataset_name)
        return legacy_file.is_file()

    def load(
        self, model_name: str, dataset_name: str
    ) -> tuple[torch.Tensor, list[dict[str, Any]]]:
        """Load cached representations and metadata using memory-mapping.

        Args:
            model_name: Name of the representation learning model.
            dataset_name: Name of the dataset.

        Returns:
            tuple[torch.Tensor, list[dict]]: (representations, metadata)
                where representations is a memory-mapped tensor of shape (N, T', C').

        Raises:
            FileNotFoundError: If no cache exists for the given model and dataset.
        """
        if not self.has(model_name, dataset_name):
            entry_dir = self.get_cache_path(model_name, dataset_name)
            raise FileNotFoundError(
                f"No cached representations found for model='{model_name}' and dataset='{dataset_name}' at {entry_dir}"
            )

        memmap_file = self.get_memmap_path(model_name, dataset_name)
        metadata_file = self.get_metadata_path(model_name, dataset_name)

        if memmap_file.is_file():
            # Memory-map the representations with copy-on-write mode
            mm = np.lib.format.open_memmap(memmap_file, mode="c")
            representations = torch.from_numpy(mm)
            metadata = torch.load(metadata_file, weights_only=False)
            return representations, metadata

        # Fallback to legacy single .pt file
        legacy_file = self._get_legacy_path(model_name, dataset_name)
        data = torch.load(legacy_file, weights_only=False)
        return data["representations"], data["metadata"]

    def save(
        self,
        model_name: str,
        dataset_name: str,
        representations: torch.Tensor,
        metadata: list[dict[str, Any]],
    ) -> Path:
        """Save representations and metadata to disk using memory-mapped format.

        Args:
            model_name: Name of the representation learning model.
            dataset_name: Name of the dataset.
            representations: Tensor of shape (N, T', C').
            metadata: List of metadata dicts corresponding to each sample.

        Returns:
            Path: The directory path of the saved cache entry.
        """
        entry_dir = self.get_cache_path(model_name, dataset_name)
        temp_dir = entry_dir.with_name(f"{entry_dir.name}.tmp")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

        mmap_path = temp_dir / "representations.npy"
        reps_shape = tuple(representations.shape)
        mm = np.lib.format.open_memmap(
            mmap_path,
            mode="w+",
            dtype=np.float32,
            shape=reps_shape,
        )
        mm[:] = representations.detach().cpu().numpy()
        mm.flush()
        del mm

        self.commit_temp_dir(
            temp_dir=temp_dir,
            model_name=model_name,
            dataset_name=dataset_name,
            metadata=metadata,
            representations_shape=reps_shape,
        )
        return entry_dir

    def commit_temp_dir(
        self,
        temp_dir: Path,
        model_name: str,
        dataset_name: str,
        metadata: list[dict[str, Any]],
        representations_shape: tuple[int, ...],
    ) -> Path:
        """Atomically commit a prepared temp directory to the permanent cache entry directory."""
        entry_dir = self.get_cache_path(model_name, dataset_name)

        # Save metadata and manifest
        torch.save(metadata, temp_dir / "metadata.pt")

        manifest = {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "num_samples": representations_shape[0],
            "shape": list(representations_shape),
            "dtype": "float32",
            "storage_format": "memmap_npy",
        }
        with open(temp_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        # Atomically swap temp_dir into entry_dir
        if entry_dir.exists():
            shutil.rmtree(entry_dir)
        temp_dir.replace(entry_dir)
        return entry_dir

    def clear(
        self, model_name: str | None = None, dataset_name: str | None = None
    ) -> None:
        """Clear cache files.

        Args:
            model_name: If specified, clear only for this model.
            dataset_name: If specified (along with model_name), clear only this entry.
        """
        if model_name is not None and dataset_name is not None:
            entry_dir = self.get_cache_path(model_name, dataset_name)
            if entry_dir.exists():
                shutil.rmtree(entry_dir)
            legacy_file = self._get_legacy_path(model_name, dataset_name)
            if legacy_file.exists():
                legacy_file.unlink()
        elif model_name is not None:
            safe_model = sanitize_filename(model_name)
            model_dir = self.cache_dir / safe_model
            if model_dir.exists():
                shutil.rmtree(model_dir)
        else:
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
