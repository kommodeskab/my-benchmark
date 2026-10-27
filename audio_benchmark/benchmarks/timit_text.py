"""Pre-defined TIMIT text sequence modeling benchmark."""

from pathlib import Path

from torch.utils.data import Subset

from audio_benchmark.benchmark import AudioBenchmark
from audio_benchmark.dataset import BaseDataset
from audio_benchmark.datasets.timit import TIMITDataset
from audio_benchmark.tasks.sequence_modeling import SequenceModelingTask
from audio_benchmark.tasks.sequence_models import TransformerDecoderProbe


class TIMITTextSequenceModelingBenchmark(AudioBenchmark):
    """Pre-defined TIMIT text sequence modeling benchmark.

    Evaluates audio representation models on predicting variable-length word
    token sequences using an autoregressive Transformer decoder probe on the TIMIT
    training and test splits. Audio samples are resampled to 16 kHz without
    fixed-length trimming via BaseDataset.

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
        """Initialize TIMITTextSequenceModelingBenchmark."""
        if max_size is not None and (not isinstance(max_size, int) or max_size <= 0):
            raise ValueError(f"max_size must be a positive integer, got {max_size!r}")

        task = SequenceModelingTask(
            probe=TransformerDecoderProbe(),
            target_key="text",
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
            dataset_name="TIMIT train (text)",
            sample_rate=16000,
            normalize=True,
        )
        wrapped_test_ds = BaseDataset(
            dataset=test_ds,
            dataset_name="TIMIT test (text)",
            sample_rate=16000,
            normalize=True,
        )

        super().__init__(
            dataset=wrapped_train_ds,
            testset=wrapped_test_ds,
            task=task,
            dataset_name="TIMIT train (text)",
            testset_name="TIMIT test (text)",
            cache_dir=cache_dir,
            enable_cache=enable_cache,
        )
