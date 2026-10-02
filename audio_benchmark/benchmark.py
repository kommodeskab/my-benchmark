"""Main Audio Representation Benchmark suite runner."""

import shutil
from dataclasses import dataclass, field
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
        extra_info: Auxiliary dataset statistics and baseline metric results.
    """

    model_name: str
    dataset_name: str
    task_name: str
    metrics: dict[str, Any]
    cached: bool
    representations_shape: tuple[int, ...]
    extra_info: dict[str, Any] = field(default_factory=dict)

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

        dataset_info = {**self.extra_info, **self.metrics}

        if "classifier" in self.metrics:
            print(f"  Classifier:      {self.metrics['classifier'].upper()}")
        if "regressor" in self.metrics:
            print(f"  Regressor:       {self.metrics['regressor'].upper()}")
        if "method" in self.metrics:
            print(f"  Method:          {self.metrics['method'].upper()}")
        if "token_error_rate" in self.metrics:
            print(f"  Token Error Rate: {self.metrics['token_error_rate'] * 100:.2f}%")
        if "sequence_accuracy" in self.metrics:
            print(f"  Sequence Acc:    {self.metrics['sequence_accuracy'] * 100:.2f}%")
        if "accuracy" in self.metrics:
            print(f"  Test Accuracy:   {self.metrics['accuracy'] * 100:.2f}%")
        if "train_accuracy" in self.metrics:
            print(f"  Train Accuracy:  {self.metrics['train_accuracy'] * 100:.2f}%")
        if "macro_f1" in self.metrics:
            print(f"  Macro F1:        {self.metrics['macro_f1']:.4f}")
        if "micro_f1" in self.metrics:
            print(f"  Micro F1:        {self.metrics['micro_f1']:.4f}")
        if "mse" in self.metrics:
            print(f"  Test MSE:        {self.metrics['mse']:.4f}")
        if "rmse" in self.metrics:
            print(f"  Test RMSE:       {self.metrics['rmse']:.4f}")
        if "mae" in self.metrics:
            print(f"  Test MAE:        {self.metrics['mae']:.4f}")
        if "r2" in self.metrics:
            print(f"  Test R²:         {self.metrics['r2']:.4f}")
        if "pearson_r" in self.metrics:
            print(f"  Pearson r:       {self.metrics['pearson_r']:.4f}")
        if "value" in self.metrics and not any(
            k in self.metrics
            for k in (
                "token_error_rate",
                "sequence_accuracy",
                "accuracy",
                "r2",
                "mse",
                "rmse",
                "mae",
            )
        ):
            print(f"  Main Metric (value): {self.metrics['value']}")
        if "training_time" in self.metrics:
            print(f"  Training Time:   {self.metrics['training_time']:.4f}s")

        # Dataset statistics & baseline from dataset examination
        if "baseline_value" in dataset_info:
            metric_name = dataset_info.get("baseline_metric", "metric")
            strat = dataset_info.get("baseline_strategy", "")
            strat_str = f" ({strat})" if strat else ""
            b_val = dataset_info["baseline_value"]
            if isinstance(b_val, float):
                print(f"  Baseline ({metric_name}){strat_str}: {b_val:.4f}")
            else:
                print(f"  Baseline ({metric_name}){strat_str}: {b_val}")
        if "vocab_size" in dataset_info:
            print(f"  Vocab Size:      {dataset_info['vocab_size']}")
        if "num_classes" in dataset_info:
            print(f"  Num Classes:     {dataset_info['num_classes']}")
        if "train_samples" in dataset_info and "test_samples" in dataset_info:
            print(
                f"  Split (Tr/Te):   {dataset_info['train_samples']} / {dataset_info['test_samples']} samples"
            )
        if "train_frames" in dataset_info and "test_frames" in dataset_info:
            print(
                f"  Frame Split:     {dataset_info['train_frames']} / {dataset_info['test_frames']} frames"
            )
        if "train_clips" in dataset_info and "test_clips" in dataset_info:
            print(
                f"  Clip Split:      {dataset_info['train_clips']} / {dataset_info['test_clips']} clips"
            )
        if "T_orig" in dataset_info and "T_downsampled" in dataset_info:
            print(
                f"  Resolution (T):  {dataset_info['T_orig']} -> {dataset_info['T_downsampled']} frames"
            )

        ignored_keys = {
            "classifier",
            "regressor",
            "method",
            "token_error_rate",
            "sequence_accuracy",
            "accuracy",
            "train_accuracy",
            "macro_f1",
            "micro_f1",
            "mse",
            "rmse",
            "mae",
            "r2",
            "pearson_r",
            "training_time",
            "train_loss",
            "train_mse",
            "train_r2",
            "vocab_size",
            "num_classes",
            "train_samples",
            "test_samples",
            "train_frames",
            "test_frames",
            "train_clips",
            "test_clips",
            "T_orig",
            "T_downsampled",
            "classes",
            "task",
            "pooling",
            "target_key",
            "value",
            "main_metric",
            "baseline_value",
            "baseline_metric",
            "baseline_strategy",
            "baseline_accuracy",
            "baseline_macro_f1",
            "baseline_micro_f1",
            "baseline_r2",
            "baseline_mse",
            "baseline_rmse",
            "baseline_mae",
            "baseline_token_error_rate",
            "baseline_sequence_accuracy",
            "chance_level",
            "majority_class",
            "class_counts",
            "total_samples",
            "total_clips",
            "total_frames",
            "frame_length",
            "vocabulary",
            "min_sequence_length",
            "max_sequence_length",
            "mean_sequence_length",
            "target_min",
            "target_max",
            "target_mean",
            "target_std",
        }

        for k, v in self.metrics.items():
            if k not in ignored_keys:
                print(f"  {k}: {v}")

        print("-----------------------------------------------------------")


class AudioBenchmark:
    """Benchmark suite for evaluating audio representation learning models."""

    def __init__(
        self,
        cache_dir: str | Path = ".benchmark_cache",
        enable_cache: bool = True,
    ):
        """Initialize the benchmark suite.

        Args:
            cache_dir: Filesystem directory for caching representations.
            enable_cache: Whether to use caching for computed representations.
        """
        self.cache_dir = Path(cache_dir)
        self.enable_cache = enable_cache
        self.cache = RepresentationCache(cache_dir=self.cache_dir)

    def run(
        self,
        model: torch.nn.Module,
        model_name: str,
        dataset: Dataset,
        task: BaseTask,
        dataset_name: str | None = None,
        batch_size: int = 16,
        device: str | torch.device = "auto",
        force_recompute: bool = False,
    ) -> BenchmarkResult:
        """Run representation extraction and benchmark evaluation for a model on a dataset.

        Args:
            model: PyTorch representation learning model mapping (B, 1, T) to (B, T', C').
            model_name: User-specified identifier for the representation model.
            dataset: Audio dataset returning samples and metadata.
            task: Task to evaluate on (must be an instance of BaseTask).
            dataset_name: Optional explicit name for the dataset. Inferred if not given.
            batch_size: Batch size for representation extraction.
            device: Device to run representation model on ('auto', 'cpu', 'cuda', etc.).
            force_recompute: If True, bypass cache and recompute representations.

        Returns:
            BenchmarkResult: The result containing metrics and metadata.
        """
        if not isinstance(task, BaseTask):
            raise TypeError(
                f"task must be an instance of BaseTask, got {type(task).__name__}."
            )

        resolved_dataset_name = dataset_name or get_dataset_name(dataset)

        # 1. Validate dataset against task requirements
        task.validate_dataset(dataset)

        # 2. Examine dataset in task context
        dataset_info = task.examine_data(dataset)

        # 3. Check cache or compute representations
        cached = False
        representations: torch.Tensor
        metadata: list[dict[str, Any]]

        if (
            self.enable_cache
            and not force_recompute
            and self.cache.has(model_name, resolved_dataset_name)
        ):
            representations, metadata = self.cache.load(
                model_name, resolved_dataset_name
            )
            cached = True
        else:
            if self.enable_cache:
                entry_dir = self.cache.get_cache_path(model_name, resolved_dataset_name)
                temp_dir = entry_dir.with_name(f"{entry_dir.name}.tmp")
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
                temp_dir.mkdir(parents=True, exist_ok=True)
                mmap_path = temp_dir / "representations.npy"

                representations, metadata = extract_representations(
                    model=model,
                    dataset=dataset,
                    batch_size=batch_size,
                    device=device,
                    mmap_path=mmap_path,
                )

                self.cache.commit_temp_dir(
                    temp_dir=temp_dir,
                    model_name=model_name,
                    dataset_name=resolved_dataset_name,
                    metadata=metadata,
                    representations_shape=tuple(representations.shape),
                )
            else:
                representations, metadata = extract_representations(
                    model=model,
                    dataset=dataset,
                    batch_size=batch_size,
                    device=device,
                    mmap_path=None,
                )

        # 4. Evaluate representations on the task
        metrics = task.evaluate(representations, metadata)

        return BenchmarkResult(
            model_name=model_name,
            dataset_name=resolved_dataset_name,
            task_name=task.name,
            metrics=metrics,
            cached=cached,
            representations_shape=tuple(representations.shape),
            extra_info=dataset_info,
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
