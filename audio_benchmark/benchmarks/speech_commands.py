"""Pre-defined Speech Commands classification benchmark."""

from pathlib import Path

from torch.utils.data import Subset

from audio_benchmark.benchmark import AudioBenchmark
from audio_benchmark.classifiers.models import LinearBenchmarkClassifier
from audio_benchmark.dataset import BaseDataset
from audio_benchmark.datasets.speechcommands import SpeechCommandsDataset
from audio_benchmark.tasks.global_classification import GlobalClassificationTask


class SpeechCommandsClassificationBenchmark(AudioBenchmark):
    """Pre-defined Speech Commands 35-class spoken command classification benchmark.

    Evaluates audio representation models on 35-class spoken command classification
    using a global classification task with a Ridge linear classifier head on
    the training and validation splits. Audio samples are resampled to 16 kHz and
    padded or cropped to 16,000 samples (1.0 second) via BaseDataset.

    Args:
        max_size: Optional maximum number of samples to evaluate on for each split.
        data_dir: Directory where Speech Commands data is stored or will be downloaded.
        cache_dir: Directory path for caching representations. Defaults to ".benchmark_cache".
        enable_cache: Whether to cache extracted representations. Defaults to True.
        download: Whether to download data files if not present. Defaults to True.

    Raises:
        ValueError: If max_size is non-positive.
    """

    def __init__(
        self,
        max_size: int | None = None,
        data_dir: str | Path | None = None,
        cache_dir: str | Path = ".benchmark_cache",
        enable_cache: bool = True,
        download: bool = True,
    ) -> None:
        """Initialize SpeechCommandsClassificationBenchmark."""
        if max_size is not None and (
            not isinstance(max_size, int) or isinstance(max_size, bool) or max_size <= 0
        ):
            raise ValueError(f"max_size must be a positive integer, got {max_size!r}")

        task = GlobalClassificationTask(
            classifier=LinearBenchmarkClassifier(model_type="ridge")
        )
        train_ds = SpeechCommandsDataset(
            split="train",
            data_dir=data_dir,
            download=download,
        )
        val_ds = SpeechCommandsDataset(
            split="val",
            data_dir=data_dir,
            download=download,
        )

        if max_size is not None:
            train_ds = Subset(train_ds, range(min(len(train_ds), max_size)))
            val_ds = Subset(val_ds, range(min(len(val_ds), max_size)))

        wrapped_train_ds = BaseDataset(
            dataset=train_ds,
            dataset_name="Speech Commands train",
            sample_rate=16000,
            length=16000,
        )
        wrapped_val_ds = BaseDataset(
            dataset=val_ds,
            dataset_name="Speech Commands val",
            sample_rate=16000,
            length=16000,
        )

        super().__init__(
            dataset=wrapped_train_ds,
            testset=wrapped_val_ds,
            task=task,
            dataset_name="Speech Commands train",
            testset_name="Speech Commands val",
            cache_dir=cache_dir,
            enable_cache=enable_cache,
        )
