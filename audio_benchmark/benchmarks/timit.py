"""Pre-defined TIMIT speech duration regression benchmark."""

from pathlib import Path

from torch.utils.data import Subset

from audio_benchmark.benchmark import AudioBenchmark
from audio_benchmark.dataset import BaseDataset
from audio_benchmark.datasets.timit import TIMITDataset
from audio_benchmark.regressors.models import TransformerBenchmarkRegressor
from audio_benchmark.tasks.regression import RegressionTask


class TIMITSpeechDurationRegressionBenchmark(AudioBenchmark):
    """Pre-defined TIMIT continuous speech duration regression benchmark.

    Evaluates audio representation models on predicting speech duration using
    a global regression task with a Transformer regressor head on the TIMIT
    training and test splits. Audio samples are resampled to 16 kHz and padded
    or cropped to 48,000 samples (3.0 seconds) via BaseDataset.

    Args:
        max_size: Optional maximum number of samples to evaluate on for each split.
        data_dir: Directory where TIMIT data is cached.
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
        """Initialize TIMITSpeechDurationRegressionBenchmark."""
        if max_size is not None and (
            not isinstance(max_size, int) or isinstance(max_size, bool) or max_size <= 0
        ):
            raise ValueError(f"max_size must be a positive integer, got {max_size!r}")

        task = RegressionTask(
            regressor=TransformerBenchmarkRegressor(),
            target_key="speech_duration",
        )
        train_ds = TIMITDataset(
            split="train",
            data_dir=data_dir,
        )
        test_ds = TIMITDataset(
            split="test",
            data_dir=data_dir,
        )

        if max_size is not None:
            train_ds = Subset(train_ds, range(min(len(train_ds), max_size)))
            test_ds = Subset(test_ds, range(min(len(test_ds), max_size)))

        wrapped_train_ds = BaseDataset(
            dataset=train_ds,
            dataset_name="TIMIT train (speech_duration)",
            sample_rate=16000,
            length=48000,
        )
        wrapped_test_ds = BaseDataset(
            dataset=test_ds,
            dataset_name="TIMIT test (speech_duration)",
            sample_rate=16000,
            length=48000,
        )

        super().__init__(
            dataset=wrapped_train_ds,
            testset=wrapped_test_ds,
            task=task,
            dataset_name="TIMIT train (speech_duration)",
            testset_name="TIMIT test (speech_duration)",
            cache_dir=cache_dir,
            enable_cache=enable_cache,
        )
