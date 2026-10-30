"""Pre-defined PTDB-TUG pitch and voice activity classification benchmarks."""

from pathlib import Path

from audio_benchmark.benchmark import AudioBenchmark
from audio_benchmark.classifiers.models import LinearBenchmarkClassifier
from audio_benchmark.dataset import BaseDataset, deterministic_random_subset
from audio_benchmark.datasets.ptdb import PTDBTUGDataset
from audio_benchmark.tasks.frame_classification import FrameClassificationTask


class PTDBPitchClassificationBenchmark(AudioBenchmark):
    """Pre-defined PTDB-TUG frame-level pitch classification benchmark.

    Evaluates audio representation models on frame-level pitch bin classification
    across 10 voiced cents bins (excluding unvoiced class 0 via ignore_label=0)
    using a Ridge linear classifier head on a deterministic 80/20 train/validation
    split of PTDB-TUG. Audio samples are resampled to 16 kHz and normalized without
    fixed-length trimming via BaseDataset.

    Args:
        max_size: Optional maximum number of samples to evaluate on for each split.
            If provided, a deterministic random permutation (seed=seed) is used.
        seed: Random seed used for deterministic subset sampling when max_size is set. Defaults to 42.
        data_dir: Directory where PTDB-TUG data is cached.
        cache_dir: Directory path for caching representations. Defaults to ".benchmark_cache".
        enable_cache: Whether to cache extracted representations. Defaults to True.

    Raises:
        ValueError: If max_size is non-positive.
        TypeError: If seed is not an integer.
    """

    def __init__(
        self,
        max_size: int | None = None,
        seed: int = 42,
        data_dir: str | Path | None = None,
        cache_dir: str | Path = ".benchmark_cache",
        enable_cache: bool = True,
    ) -> None:
        """Initialize PTDBPitchClassificationBenchmark."""
        if max_size is not None and (not isinstance(max_size, int) or max_size <= 0):
            raise ValueError(f"max_size must be a positive integer, got {max_size!r}")
        if not isinstance(seed, int):
            raise TypeError(f"seed must be an integer, got {seed!r}")

        task = FrameClassificationTask(
            classifier=LinearBenchmarkClassifier(model_type="ridge"),
            label_key="cents_class",
            ignore_label=0,
        )
        train_ds = PTDBTUGDataset(
            split="train",
            data_dir=data_dir,
        )
        val_ds = PTDBTUGDataset(
            split="val",
            data_dir=data_dir,
        )

        train_ds = deterministic_random_subset(train_ds, max_size=max_size, seed=seed)
        val_ds = deterministic_random_subset(val_ds, max_size=max_size, seed=seed)

        wrapped_train_ds = BaseDataset(
            dataset=train_ds,
            dataset_name="PTDB-TUG train",
            sample_rate=16000,
            normalize=True,
        )
        wrapped_val_ds = BaseDataset(
            dataset=val_ds,
            dataset_name="PTDB-TUG val",
            sample_rate=16000,
            normalize=True,
        )

        super().__init__(
            dataset=wrapped_train_ds,
            testset=wrapped_val_ds,
            task=task,
            dataset_name="PTDB-TUG train",
            testset_name="PTDB-TUG val",
            cache_dir=cache_dir,
            enable_cache=enable_cache,
        )


class PTDBVoiceActivityClassificationBenchmark(AudioBenchmark):
    """Pre-defined PTDB-TUG frame-level voice activity classification benchmark.

    Evaluates audio representation models on binary frame-level voice presence/silence
    detection using the 'voice_activity' target and a Ridge linear classifier head on
    a deterministic 80/20 train/validation split of PTDB-TUG. Audio samples are
    resampled to 16 kHz and normalized without fixed-length trimming via BaseDataset.

    Args:
        max_size: Optional maximum number of samples to evaluate on for each split.
            If provided, a deterministic random permutation (seed=seed) is used.
        seed: Random seed used for deterministic subset sampling when max_size is set. Defaults to 42.
        data_dir: Directory where PTDB-TUG data is cached.
        cache_dir: Directory path for caching representations. Defaults to ".benchmark_cache".
        enable_cache: Whether to cache extracted representations. Defaults to True.

    Raises:
        ValueError: If max_size is non-positive.
        TypeError: If seed is not an integer.
    """

    def __init__(
        self,
        max_size: int | None = None,
        seed: int = 42,
        data_dir: str | Path | None = None,
        cache_dir: str | Path = ".benchmark_cache",
        enable_cache: bool = True,
    ) -> None:
        """Initialize PTDBVoiceActivityClassificationBenchmark."""
        if max_size is not None and (not isinstance(max_size, int) or max_size <= 0):
            raise ValueError(f"max_size must be a positive integer, got {max_size!r}")
        if not isinstance(seed, int):
            raise TypeError(f"seed must be an integer, got {seed!r}")

        task = FrameClassificationTask(
            classifier=LinearBenchmarkClassifier(model_type="ridge"),
            label_key="voice_activity",
        )
        train_ds = PTDBTUGDataset(
            split="train",
            data_dir=data_dir,
        )
        val_ds = PTDBTUGDataset(
            split="val",
            data_dir=data_dir,
        )

        train_ds = deterministic_random_subset(train_ds, max_size=max_size, seed=seed)
        val_ds = deterministic_random_subset(val_ds, max_size=max_size, seed=seed)

        wrapped_train_ds = BaseDataset(
            dataset=train_ds,
            dataset_name="PTDB-TUG train",
            sample_rate=16000,
            normalize=True,
        )
        wrapped_val_ds = BaseDataset(
            dataset=val_ds,
            dataset_name="PTDB-TUG val",
            sample_rate=16000,
            normalize=True,
        )

        super().__init__(
            dataset=wrapped_train_ds,
            testset=wrapped_val_ds,
            task=task,
            dataset_name="PTDB-TUG train",
            testset_name="PTDB-TUG val",
            cache_dir=cache_dir,
            enable_cache=enable_cache,
        )
