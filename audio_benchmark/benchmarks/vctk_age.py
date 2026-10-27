"""Pre-defined VCTK age regression benchmark."""

from pathlib import Path

from torch.utils.data import Subset

from audio_benchmark.benchmark import AudioBenchmark
from audio_benchmark.dataset import BaseDataset
from audio_benchmark.datasets.vctk import VCTKDataset
from audio_benchmark.regressors.models import LinearBenchmarkRegressor
from audio_benchmark.tasks.regression import RegressionTask


class VCTKAgeRegressionBenchmark(AudioBenchmark):
    """Pre-defined VCTK speaker age regression benchmark.

    Evaluates audio representation models on predicting speaker age using
    a global regression task with a linear regressor head on the VCTK
    training and validation splits. Audio samples are resampled to 16 kHz
    and padded or cropped to 48,000 samples (3.0 seconds) via BaseDataset.

    Args:
        max_size: Optional maximum number of samples to evaluate on for each split.
        data_dir: Directory where VCTK data is cached.
        cache_dir: Directory path for caching representations. Defaults to ".benchmark_cache".
        enable_cache: Whether to cache extracted representations. Defaults to True.

    Raises:
        ValueError: If max_size is non-positive.
    """

    def __init__(
        self,
        max_size: int | None = None,
        data_dir: str | Path | None = None,
        cache_dir: str | Path = ".benchmark_cache",
        enable_cache: bool = True,
    ) -> None:
        """Initialize VCTKAgeRegressionBenchmark."""
        if max_size is not None and (not isinstance(max_size, int) or max_size <= 0):
            raise ValueError(f"max_size must be a positive integer, got {max_size!r}")

        task = RegressionTask(
            regressor=LinearBenchmarkRegressor(),
            target_key="age",
        )
        train_ds = VCTKDataset(
            split="train",
            data_dir=data_dir,
        )
        val_ds = VCTKDataset(
            split="val",
            data_dir=data_dir,
        )

        if max_size is not None:
            train_ds = Subset(train_ds, range(min(len(train_ds), max_size)))
            val_ds = Subset(val_ds, range(min(len(val_ds), max_size)))

        wrapped_train_ds = BaseDataset(
            dataset=train_ds,
            dataset_name="VCTK train (age)",
            sample_rate=16000,
            length=48000,
            normalize=True,
        )
        wrapped_val_ds = BaseDataset(
            dataset=val_ds,
            dataset_name="VCTK val (age)",
            sample_rate=16000,
            length=48000,
            normalize=True,
        )

        super().__init__(
            dataset=wrapped_train_ds,
            testset=wrapped_val_ds,
            task=task,
            dataset_name="VCTK train (age)",
            testset_name="VCTK val (age)",
            cache_dir=cache_dir,
            enable_cache=enable_cache,
        )
