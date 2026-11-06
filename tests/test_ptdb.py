"""Tests for PTDB-TUG dataset implementation."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from audio_benchmark import (
    DummyAudioModel,
    PTDBPitchClassificationBenchmark,
    PTDBVoiceActivityClassificationBenchmark,
)
from audio_benchmark.classifiers import KNNBenchmarkClassifier
from audio_benchmark.datasets import PTDBTUGDataset
from audio_benchmark.regressors import LinearBenchmarkRegressor
from audio_benchmark.tasks import FrameClassificationTask, FrameRegressionTask


def _create_mock_hf_ptdb_dataset(size: int = 10):
    samples = []
    for i in range(size):
        # 100 frames at 10ms stride = 1.0s audio at 16000Hz (16000 samples)
        sample_rate = 16000
        stride_samples = 160
        num_frames = 100
        audio_len = num_frames * stride_samples

        # Create alternating voiced/unvoiced frames
        f0_data = []
        for f in range(num_frames):
            if 20 <= f <= 80:
                pitch = 100.0 + (f % 50) * 5.0
                voiced = 1.0
            else:
                pitch = 0.0
                voiced = 0.0
            f0_data.append([pitch, voiced])

        samples.append(
            {
                "mic_audio": {
                    "array": [0.01 * (f % 10) for f in range(audio_len)],
                    "sampling_rate": sample_rate,
                },
                "f0_data": f0_data,
            }
        )
    mock_ds = MagicMock()
    mock_ds.__len__.return_value = len(samples)
    mock_ds.__getitem__.side_effect = lambda idx: samples[idx]
    return mock_ds


class TestPTDBTUGDataset:
    """Tests for PTDBTUGDataset."""

    def test_invalid_split_raises(self) -> None:
        with pytest.raises(ValueError, match="split must be 'train' or 'val'"):
            PTDBTUGDataset(split="test")  # type: ignore[arg-type]

    def test_invalid_num_bins_raises(self) -> None:
        with pytest.raises(ValueError, match="num_bins must be a positive integer"):
            PTDBTUGDataset(split="train", num_bins=0)

        with pytest.raises(ValueError, match="num_bins must be a positive integer"):
            PTDBTUGDataset(split="train", num_bins=-5)

    @patch("audio_benchmark.datasets.ptdb.load_dataset")
    def test_splits_and_output_schema(self, mock_load_dataset: MagicMock) -> None:
        mock_ds = _create_mock_hf_ptdb_dataset(size=10)
        mock_load_dataset.return_value = mock_ds

        train_ds = PTDBTUGDataset(split="train", data_dir="/custom/cache", seed=42)
        val_ds = PTDBTUGDataset(split="val", data_dir="/custom/cache", seed=42)

        mock_load_dataset.assert_called_with(
            "gilkeyio/ptdb-tug",
            cache_dir="/custom/cache",
            split="train",
        )

        assert len(train_ds) == 8
        assert len(val_ds) == 2

        item = train_ds[0]
        assert "waveform" in item
        assert "sample_rate" in item
        assert "metadata" in item

        waveform = item["waveform"]
        assert isinstance(waveform, torch.Tensor)
        assert waveform.ndim == 2
        assert waveform.shape[0] == 1

        sample_rate = item["sample_rate"]
        assert isinstance(sample_rate, int)
        assert sample_rate == 16000

        meta = item["metadata"]
        assert "f0_values" in meta
        assert "voice_activity" in meta
        assert "cents" in meta
        assert "cents_class" in meta
        assert "audio_length" in meta

        # Verify all targets are flat (1D) tensors
        for key in ("f0_values", "voice_activity", "cents", "cents_class"):
            target = meta[key]
            assert isinstance(target, torch.Tensor)
            assert target.ndim == 1
            assert len(target) > 0

        # cents_class values must be integers in 0..num_bins
        classes = meta["cents_class"]
        assert classes.dtype == torch.long
        assert (classes >= 0).all()
        assert (classes <= 10).all()

        # Unvoiced frames must map to class 0 and -inf cents
        unvoiced_mask = meta["voice_activity"] == 0.0
        assert (classes[unvoiced_mask] == 0).all()
        assert torch.isneginf(meta["cents"][unvoiced_mask]).all()

    @patch("audio_benchmark.datasets.ptdb.load_dataset")
    def test_all_unvoiced_silent_clip(self, mock_load_dataset: MagicMock) -> None:
        # Test edge case: clip with zero voice activity throughout
        sample_rate = 16000
        num_frames = 50
        audio_len = num_frames * 160
        f0_data = [[0.0, 0.0] for _ in range(num_frames)]
        samples = [
            {
                "mic_audio": {
                    "array": [0.0] * audio_len,
                    "sampling_rate": sample_rate,
                },
                "f0_data": f0_data,
            }
        ] * 5

        mock_ds = MagicMock()
        mock_ds.__len__.return_value = len(samples)
        mock_ds.__getitem__.side_effect = lambda idx: samples[idx]
        mock_load_dataset.return_value = mock_ds

        ds = PTDBTUGDataset(split="train")
        item = ds[0]
        meta = item["metadata"]

        assert (meta["voice_activity"] == 0.0).all()
        assert (meta["cents_class"] == 0).all()
        assert torch.isneginf(meta["cents"]).all()

    @patch("audio_benchmark.datasets.ptdb.load_dataset")
    def test_single_voiced_frame(self, mock_load_dataset: MagicMock) -> None:
        # Test edge case: clip with exactly one voiced frame
        sample_rate = 16000
        num_frames = 50
        audio_len = num_frames * 160
        f0_data = [[0.0, 0.0] for _ in range(num_frames)]
        f0_data[25] = [220.0, 1.0]
        samples = [
            {
                "mic_audio": {
                    "array": [0.0] * audio_len,
                    "sampling_rate": sample_rate,
                },
                "f0_data": f0_data,
            }
        ] * 5

        mock_ds = MagicMock()
        mock_ds.__len__.return_value = len(samples)
        mock_ds.__getitem__.side_effect = lambda idx: samples[idx]
        mock_load_dataset.return_value = mock_ds

        ds = PTDBTUGDataset(split="train")
        item = ds[0]
        meta = item["metadata"]

        assert (meta["cents_class"] > 0).sum() == 1

    @patch("audio_benchmark.datasets.ptdb.load_dataset")
    def test_compatibility_with_frame_tasks(self, mock_load_dataset: MagicMock) -> None:
        mock_ds = _create_mock_hf_ptdb_dataset(size=10)
        mock_load_dataset.return_value = mock_ds

        train_ds = PTDBTUGDataset(split="train")
        val_ds = PTDBTUGDataset(split="val")

        # 1. Frame classification on cents_class
        task_clf = FrameClassificationTask(
            classifier=KNNBenchmarkClassifier(),
            label_key="cents_class",
        )
        task_clf.validate_dataset(train_ds)
        task_clf.validate_dataset(val_ds)

        # 2. Frame regression on f0_values
        task_reg = FrameRegressionTask(
            regressor=LinearBenchmarkRegressor(),
            target_key="f0_values",
        )
        task_reg.validate_dataset(train_ds)
        task_reg.validate_dataset(val_ds)


class TestPTDBPitchClassificationBenchmark:
    """Tests for PTDBPitchClassificationBenchmark."""

    def test_invalid_arguments_raise(self) -> None:
        with pytest.raises(ValueError, match="max_size must be a positive integer"):
            PTDBPitchClassificationBenchmark(max_size=0)

        with pytest.raises(ValueError, match="max_size must be a positive integer"):
            PTDBPitchClassificationBenchmark(max_size=-3)

        with pytest.raises(TypeError, match="seed must be an integer"):
            PTDBPitchClassificationBenchmark(seed="42")  # type: ignore[arg-type]

    @patch("audio_benchmark.datasets.ptdb.load_dataset")
    def test_benchmark_runs_with_variable_length_audio(
        self, mock_load_dataset: MagicMock, tmp_path
    ) -> None:
        # Create samples with varying lengths (60, 80, 100 frames) at 48kHz
        samples = []
        for i in range(10):
            sample_rate = 48000
            stride_samples = 480
            num_frames = 60 + (i % 3) * 20
            audio_len = num_frames * stride_samples
            f0_data = [
                [110.0 + (f % 40) * 5.0, 1.0]
                if 15 <= f <= num_frames - 15
                else [0.0, 0.0]
                for f in range(num_frames)
            ]
            samples.append(
                {
                    "mic_audio": {
                        "array": [0.05 * ((f % 7) - 3) for f in range(audio_len)],
                        "sampling_rate": sample_rate,
                    },
                    "f0_data": f0_data,
                }
            )
        mock_ds = MagicMock()
        mock_ds.__len__.return_value = len(samples)
        mock_ds.__getitem__.side_effect = lambda idx: samples[idx]
        mock_load_dataset.return_value = mock_ds

        benchmark = PTDBPitchClassificationBenchmark(
            max_size=4,
            seed=42,
            cache_dir=tmp_path,
            enable_cache=False,
        )

        # Verify variable-length wrapping (length is None, resampled to 16kHz) and ignore_label=0
        assert benchmark.dataset.length is None
        assert benchmark.dataset.sample_rate == 16000
        assert benchmark.task.ignore_label == 0

        data_info = benchmark.task.examine_data(benchmark.dataset, benchmark.testset)
        assert "0" not in data_info["classes"]

        model = DummyAudioModel(representation_dim=16)
        result = benchmark.run(
            model=model,
            model_name="dummy_ptdb_model",
            batch_size=2,
            device="cpu",
            show_progress=False,
        )[0]

        assert result.task_name == "frame_classification"
        assert 0.0 <= result.metrics["accuracy"] <= 1.0
        assert result.dataset_name == "PTDB-TUG train"
        assert benchmark.testset_name == "PTDB-TUG val"


class TestPTDBVoiceActivityClassificationBenchmark:
    """Tests for PTDBVoiceActivityClassificationBenchmark."""

    def test_invalid_arguments_raise(self) -> None:
        with pytest.raises(ValueError, match="max_size must be a positive integer"):
            PTDBVoiceActivityClassificationBenchmark(max_size=0)

        with pytest.raises(ValueError, match="max_size must be a positive integer"):
            PTDBVoiceActivityClassificationBenchmark(max_size=-3)

        with pytest.raises(TypeError, match="seed must be an integer"):
            PTDBVoiceActivityClassificationBenchmark(seed="42")  # type: ignore[arg-type]

    @patch("audio_benchmark.datasets.ptdb.load_dataset")
    def test_benchmark_runs_with_variable_length_audio(
        self, mock_load_dataset: MagicMock, tmp_path
    ) -> None:
        mock_ds = _create_mock_hf_ptdb_dataset(size=10)
        mock_load_dataset.return_value = mock_ds

        benchmark = PTDBVoiceActivityClassificationBenchmark(
            max_size=4,
            seed=42,
            cache_dir=tmp_path,
            enable_cache=False,
        )

        assert benchmark.dataset.length is None
        assert benchmark.dataset.sample_rate == 16000
        assert benchmark.task.label_key == "voice_activity"
        assert benchmark.task.ignore_label is None

        data_info = benchmark.task.examine_data(benchmark.dataset, benchmark.testset)
        assert data_info["num_classes"] == 2

        model = DummyAudioModel(representation_dim=16)
        result = benchmark.run(
            model=model,
            model_name="dummy_ptdb_vad_model",
            batch_size=2,
            device="cpu",
            show_progress=False,
        )[0]

        assert result.task_name == "frame_classification"
        assert 0.0 <= result.metrics["accuracy"] <= 1.0
        assert result.dataset_name == "PTDB-TUG train"
        assert benchmark.testset_name == "PTDB-TUG val"
