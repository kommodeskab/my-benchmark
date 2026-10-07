"""Main Audio Representation Benchmark suite runner."""

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from audio_benchmark.cache import RepresentationCache
from audio_benchmark.dataset import get_dataset_name
from audio_benchmark.extractor import extract_representations
from audio_benchmark.tasks.base import BaseTask


@dataclass
class BenchmarkResult:
    """Stores the outcome of a benchmark execution.

    Attributes:
        model_name: Identifier of the representation model.
        dataset_name: Identifier of the dataset.
        task_name: Identifier of the executed benchmark task.
        metrics: Dictionary of metric names and numeric results.
        cached: Whether representations were retrieved from cache.
        representations_shape: Shape tuple of the extracted representations (N, T', C').
    """

    model_name: str
    dataset_name: str
    task_name: str
    metrics: dict[str, Any]
    cached: bool
    representations_shape: tuple[int, ...]

    def __post_init__(self) -> None:
        """Ensure that metrics always has a 'value' key representing the main result."""
        if not isinstance(self.metrics, dict):
            return

        if "value" not in self.metrics:
            # Fallback detection for standard primary metric keys
            for candidate in (
                "accuracy",
                "r2",
                "token_error_rate",
                "mse",
                "mae",
                "score",
            ):
                if candidate in self.metrics:
                    self.metrics["value"] = self.metrics[candidate]
                    break
            else:
                for v in self.metrics.values():
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        self.metrics["value"] = v
                        break

    @property
    def value(self) -> Any:
        """Primary result metric value for the benchmark task."""
        return self.metrics.get("value")

    def print_summary(self) -> None:
        """Log a formatted summary table to stdout."""
        print("-------------------- Benchmark Results --------------------")
        print(f"  Model:           {self.model_name}")
        print(f"  Dataset:         {self.dataset_name}")
        print(f"  Task:            {self.task_name}")
        print(f"  Cached Reps:     {self.cached}")
        print(f"  Rep Shape:       {self.representations_shape}")
        if "value" in self.metrics:
            val = self.metrics["value"]
            val_str = f"{val:.4f}" if isinstance(val, float) else str(val)
            print(f"  Main Metric (value): {val_str}")
        print("  Metrics:")
        for k, v in self.metrics.items():
            if k == "value":
                continue
            if isinstance(v, float):
                print(f"    {k:20s}: {v:.4f}")
            else:
                print(f"    {k:20s}: {v}")
        print("-----------------------------------------------------------")


class AudioBenchmark:
    """Benchmark suite for evaluating audio representation learning models."""

    def __init__(
        self,
        model: torch.nn.Module,
        model_name: str,
        dataset: Dataset,
        testset: Dataset,
        task: BaseTask,
        dataset_name: str | None = None,
        testset_name: str | None = None,
        cache_dir: str | Path = ".benchmark_cache",
        enable_cache: bool = True,
    ) -> None:
        """Initialize the benchmark suite with model, datasets, task, and cache settings.

        Args:
            model: PyTorch representation learning model mapping (B, 1, T) to (B, T', C').
            model_name: User-specified identifier for the representation model.
            dataset: Audio dataset returning samples and metadata for training.
            testset: Audio dataset returning samples and metadata for testing.
            task: Task to evaluate on (must be an instance of BaseTask).
            dataset_name: Optional explicit name for the train dataset. Inferred if not given.
            testset_name: Optional explicit name for the test dataset. Inferred if not given.
            cache_dir: Filesystem directory for caching representations.
            enable_cache: Whether to use caching for computed representations.

        Raises:
            TypeError: If task is not an instance of BaseTask, or if model/dataset/testset is None.
        """
        if model is None:
            raise TypeError(
                "model must be an instantiated torch.nn.Module instance, got None."
            )
        if dataset is None:
            raise TypeError(
                "dataset must be an instantiated Dataset instance, got None. "
                "AudioBenchmark requires both dataset and testset to be explicitly passed."
            )
        if testset is None:
            raise TypeError(
                "testset must be an instantiated Dataset instance, got None. "
                "AudioBenchmark requires both dataset and testset to be explicitly passed."
            )
        if not isinstance(task, BaseTask):
            raise TypeError(
                f"task must be an instance of BaseTask, got {type(task).__name__}."
            )

        if dataset_name is not None and testset_name is not None:
            resolved_dataset_name = dataset_name
            resolved_testset_name = testset_name
        elif dataset_name is not None:
            resolved_dataset_name = f"{dataset_name}_train"
            resolved_testset_name = f"{dataset_name}_test"
        elif testset_name is not None:
            resolved_dataset_name = f"{get_dataset_name(dataset)}_train"
            resolved_testset_name = testset_name
        else:
            ds_name = get_dataset_name(dataset)
            ts_name = get_dataset_name(testset)
            if ds_name == ts_name:
                resolved_dataset_name = f"{ds_name}_train"
                resolved_testset_name = f"{ts_name}_test"
            else:
                resolved_dataset_name = ds_name
                resolved_testset_name = ts_name

        task.validate_dataset(dataset)
        task.validate_dataset(testset)

        self.model = model
        self.model_name = model_name
        self.dataset = dataset
        self.testset = testset
        self.task = task
        self.dataset_name = dataset_name
        self.testset_name = testset_name
        self.resolved_dataset_name = resolved_dataset_name
        self.resolved_testset_name = resolved_testset_name
        self.cache_dir = Path(cache_dir)
        self.enable_cache = enable_cache
        self.cache = RepresentationCache(cache_dir=self.cache_dir)

    def _get_representations(
        self,
        dataset: Dataset,
        resolved_name: str,
        batch_size: int,
        device: str | torch.device,
        force_recompute: bool,
        show_progress: bool,
    ) -> tuple[torch.Tensor, list[dict[str, Any]], bool]:
        """Load or extract representations for a dataset."""
        cached = False
        if (
            self.enable_cache
            and not force_recompute
            and self.cache.has(self.model_name, resolved_name)
        ):
            representations, metadata = self.cache.load(self.model_name, resolved_name)
            cached = True
        else:
            if self.enable_cache:
                entry_dir = self.cache.get_cache_path(self.model_name, resolved_name)
                temp_dir = entry_dir.with_name(f"{entry_dir.name}.tmp")
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
                temp_dir.mkdir(parents=True, exist_ok=True)
                mmap_path = temp_dir / "representations.npy"

                representations, metadata = extract_representations(
                    model=self.model,
                    dataset=dataset,
                    batch_size=batch_size,
                    device=device,
                    mmap_path=mmap_path,
                    show_progress=show_progress,
                )

                self.cache.commit_temp_dir(
                    temp_dir=temp_dir,
                    model_name=self.model_name,
                    dataset_name=resolved_name,
                    metadata=metadata,
                    representations_shape=tuple(representations.shape),
                )
            else:
                representations, metadata = extract_representations(
                    model=self.model,
                    dataset=dataset,
                    batch_size=batch_size,
                    device=device,
                    mmap_path=None,
                    show_progress=show_progress,
                )
        return representations, metadata, cached

    def run(
        self,
        batch_size: int = 16,
        device: str | torch.device = "auto",
        force_recompute: bool = False,
        show_progress: bool = True,
    ) -> BenchmarkResult:
        """Run representation extraction and benchmark evaluation for a model on train and test datasets.

        Args:
            batch_size: Batch size for representation extraction.
            device: Device to run representation model on ('auto', 'cpu', 'cuda', etc.).
            force_recompute: If True, bypass cache and recompute representations.
            show_progress: Whether to display a tqdm progress bar during feature extraction.

        Returns:
            BenchmarkResult: The result containing metrics and metadata.
        """
        train_reps, train_meta, train_cached = self._get_representations(
            dataset=self.dataset,
            resolved_name=self.resolved_dataset_name,
            batch_size=batch_size,
            device=device,
            force_recompute=force_recompute,
            show_progress=show_progress,
        )

        test_reps, test_meta, test_cached = self._get_representations(
            dataset=self.testset,
            resolved_name=self.resolved_testset_name,
            batch_size=batch_size,
            device=device,
            force_recompute=force_recompute,
            show_progress=show_progress,
        )

        cached = train_cached and test_cached

        metrics = self.task.evaluate(
            train_representations=train_reps,
            train_metadata=train_meta,
            test_representations=test_reps,
            test_metadata=test_meta,
        )

        return BenchmarkResult(
            model_name=self.model_name,
            dataset_name=self.resolved_dataset_name,
            task_name=self.task.name,
            metrics=metrics,
            cached=cached,
            representations_shape=tuple(train_reps.shape),
        )

    def clear_cache(
        self, model_name: str | None = None, dataset_name: str | None = None
    ) -> None:
        """Clear cached representations.

        Args:
            model_name: Optional model name to filter which cache entries to delete.
            dataset_name: Optional dataset name to filter which cache entries to delete.
        """
        self.cache.clear(model_name=model_name, dataset_name=dataset_name)
