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
        - `representations_0.npy`, `representations_1.npy`, ...: Memory-mapped NumPy arrays of shape (N, T_i, D_i).
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

    def get_memmap_path(
        self, model_name: str, dataset_name: str, representation_index: int = 0
    ) -> Path:
        """Get the filesystem path for a memory-mapped representation file at a given index.

        Args:
            model_name: Name of the representation learning model.
            dataset_name: Name of the dataset.
            representation_index: Index of the representation tensor in the list.

        Returns:
            Path: Path to the representations_{index}.npy file.
        """
        return (
            self.get_cache_path(model_name, dataset_name)
            / f"representations_{representation_index}.npy"
        )

    def get_metadata_path(self, model_name: str, dataset_name: str) -> Path:
        """Get the filesystem path for the serialized metadata file."""
        return self.get_cache_path(model_name, dataset_name) / "metadata.pt"

    def get_manifest_path(self, model_name: str, dataset_name: str) -> Path:
        """Get the filesystem path for the manifest JSON file."""
        return self.get_cache_path(model_name, dataset_name) / "manifest.json"

    def has(self, model_name: str, dataset_name: str) -> bool:
        """Check if cached representations exist.

        Args:
            model_name: Name of the representation learning model.
            dataset_name: Name of the dataset.

        Returns:
            bool: True if complete cached representations exist.
        """
        metadata_file = self.get_metadata_path(model_name, dataset_name)
        manifest_file = self.get_manifest_path(model_name, dataset_name)
        if not (metadata_file.is_file() and manifest_file.is_file()):
            return False

        try:
            with open(manifest_file, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            num_reps = int(manifest.get("num_representations", 1))
            for i in range(num_reps):
                if not self.get_memmap_path(model_name, dataset_name, i).is_file():
                    return False
            return True
        except (OSError, json.JSONDecodeError, ValueError):
            return False

    def load(
        self, model_name: str, dataset_name: str
    ) -> tuple[list[torch.Tensor], list[dict[str, Any]]]:
        """Load cached representations and metadata using memory-mapping.

        Args:
            model_name: Name of the representation learning model.
            dataset_name: Name of the dataset.

        Returns:
            tuple[list[torch.Tensor], list[dict[str, Any]]]:
                - representations: List of memory-mapped tensors of shape (N, T_i, D_i).
                - metadata: List of sample metadata dicts.

        Raises:
            FileNotFoundError: If no cache exists for the given model and dataset.
        """
        if not self.has(model_name, dataset_name):
            entry_dir = self.get_cache_path(model_name, dataset_name)
            raise FileNotFoundError(
                f"No cached representations found for model='{model_name}' and dataset='{dataset_name}' at {entry_dir}"
            )

        manifest_file = self.get_manifest_path(model_name, dataset_name)
        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        num_reps = int(manifest.get("num_representations", 1))

        representations: list[torch.Tensor] = []
        for i in range(num_reps):
            memmap_file = self.get_memmap_path(model_name, dataset_name, i)
            mm = np.lib.format.open_memmap(memmap_file, mode="c")
            representations.append(torch.from_numpy(mm))

        metadata_file = self.get_metadata_path(model_name, dataset_name)
        metadata = torch.load(metadata_file, weights_only=False)
        return representations, metadata

    def save(
        self,
        model_name: str,
        dataset_name: str,
        representations: list[torch.Tensor],
        metadata: list[dict[str, Any]],
    ) -> Path:
        """Save representations and metadata to disk using memory-mapped format.

        Args:
            model_name: Name of the representation learning model.
            dataset_name: Name of the dataset.
            representations: List of tensors of shape (N, T_i, D_i).
            metadata: List of metadata dicts corresponding to each sample.

        Returns:
            Path: The directory path of the saved cache entry.

        Raises:
            TypeError: If representations is not a list.
        """
        if not isinstance(representations, list):
            raise TypeError(
                f"Expected representations to be a list of torch.Tensor, got {type(representations).__name__}."
            )

        entry_dir = self.get_cache_path(model_name, dataset_name)
        temp_dir = entry_dir.with_name(f"{entry_dir.name}.tmp")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

        shapes: list[tuple[int, ...]] = []
        for i, rep in enumerate(representations):
            mmap_path = temp_dir / f"representations_{i}.npy"
            reps_shape = tuple(rep.shape)
            shapes.append(reps_shape)
            mm = np.lib.format.open_memmap(
                mmap_path,
                mode="w+",
                dtype=np.float32,
                shape=reps_shape,
            )
            mm[:] = rep.detach().cpu().numpy()
            mm.flush()
            del mm

        self.commit_temp_dir(
            temp_dir=temp_dir,
            model_name=model_name,
            dataset_name=dataset_name,
            metadata=metadata,
            representations_shapes=shapes,
        )
        return entry_dir

    def commit_temp_dir(
        self,
        temp_dir: Path,
        model_name: str,
        dataset_name: str,
        metadata: list[dict[str, Any]],
        representations_shapes: list[tuple[int, ...]],
    ) -> Path:
        """Atomically commit a prepared temp directory to the permanent cache entry directory.

        Args:
            temp_dir: Temporary directory containing the prepared memory-mapped files.
            model_name: Name of the representation model.
            dataset_name: Name of the dataset.
            metadata: List of metadata dictionaries.
            representations_shapes: List of shape tuples for each representation.

        Returns:
            Path: Path to the committed entry directory.
        """
        entry_dir = self.get_cache_path(model_name, dataset_name)

        # Save metadata and manifest
        torch.save(metadata, temp_dir / "metadata.pt")

        manifest = {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "num_samples": (
                representations_shapes[0][0] if representations_shapes else 0
            ),
            "num_representations": len(representations_shapes),
            "shapes": [list(s) for s in representations_shapes],
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
        elif model_name is not None:
            safe_model = sanitize_filename(model_name)
            model_dir = self.cache_dir / safe_model
            if model_dir.exists():
                shutil.rmtree(model_dir)
        else:
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
