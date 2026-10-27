"""Targeted tests for pre-defined benchmarks."""

from unittest.mock import patch

import pytest
import torch
from torch.utils.data import Dataset

from audio_benchmark.benchmarks import (
    CREMADClassificationBenchmark,
    SpeechCommandsClassificationBenchmark,
    TIMITPhonemeSequenceModelingBenchmark,
    TIMITSpeechDurationRegressionBenchmark,
    TIMITTextSequenceModelingBenchmark,
    VCTKAgeRegressionBenchmark,
    VCTKGenderClassificationBenchmark,
)
from audio_benchmark.classifiers import LinearBenchmarkClassifier
from audio_benchmark.dataset import BaseDataset
from audio_benchmark.dummy import DummyAudioModel
from audio_benchmark.regressors import (
    LinearBenchmarkRegressor,
    TransformerBenchmarkRegressor,
)
from audio_benchmark.tasks import (
    GlobalClassificationTask,
    RegressionTask,
    SequenceModelingTask,
    TransformerDecoderProbe,
)


class _MockClassificationDataset(Dataset):
    def __init__(self, size: int = 10, label_key: str = "label"):
        self.size = size
        self.label_key = label_key

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict[str, object]:
        return {
            "waveform": torch.randn(1, 8000),
            "sample_rate": 8000,
            "metadata": {self.label_key: idx % 3},
        }


class _MockTIMITDataset(Dataset):
    def __init__(self, size: int = 10):
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict[str, object]:
        return {
            "waveform": torch.randn(1, 8000 + idx * 800),
            "sample_rate": 8000,
            "metadata": {
                "speech_duration": 0.5 + 0.1 * idx,
                "speaker_id": f"SPK_{idx}",
                "phonemes": ["sh", "iy", "hh"],
                "text": ["she", "had"],
            },
        }


class TestSpeechCommandsClassificationBenchmark:
    """Tests for SpeechCommandsClassificationBenchmark."""

    @patch("audio_benchmark.benchmarks.speech_commands.SpeechCommandsDataset")
    def test_default_configuration_and_run(self, mock_sc_dataset, tmp_path) -> None:
        mock_sc_dataset.side_effect = [
            _MockClassificationDataset(size=8),
            _MockClassificationDataset(size=4),
        ]

        benchmark = SpeechCommandsClassificationBenchmark(
            cache_dir=tmp_path / "cache",
            enable_cache=False,
        )

        assert isinstance(benchmark.dataset, BaseDataset)
        assert isinstance(benchmark.testset, BaseDataset)
        assert benchmark.dataset.normalize is True
        assert benchmark.testset.normalize is True
        assert benchmark.dataset.sample_rate == 16000
        assert benchmark.dataset.length == 16000
        assert benchmark.dataset[0]["waveform"].shape == (1, 16000)
        assert benchmark.dataset[0]["sample_rate"] == 16000

        assert isinstance(benchmark.task, GlobalClassificationTask)
        assert isinstance(benchmark.task.classifier, LinearBenchmarkClassifier)
        assert benchmark.task.classifier.model_type == "ridge"
        assert benchmark.dataset_name == "Speech Commands train"
        assert benchmark.testset_name == "Speech Commands val"

        model = DummyAudioModel(representation_dim=32)
        res = benchmark.run(
            model=model, model_name="dummy", batch_size=4, show_progress=False
        )
        assert "accuracy" in res.metrics

    @patch("audio_benchmark.benchmarks.speech_commands.SpeechCommandsDataset")
    def test_max_size(self, mock_sc_dataset, tmp_path) -> None:
        mock_sc_dataset.side_effect = [
            _MockClassificationDataset(size=10),
            _MockClassificationDataset(size=6),
        ]

        benchmark = SpeechCommandsClassificationBenchmark(
            max_size=3,
            cache_dir=tmp_path / "cache",
            enable_cache=False,
        )

        assert len(benchmark.dataset) == 3
        assert len(benchmark.testset) == 3

    def test_invalid_max_size_raises(self) -> None:
        with pytest.raises(ValueError, match="max_size must be a positive integer"):
            SpeechCommandsClassificationBenchmark(max_size=0)


class TestCREMADClassificationBenchmark:
    """Tests for CREMADClassificationBenchmark."""

    @patch("audio_benchmark.benchmarks.cremad.CREMADDataset")
    def test_default_configuration_and_run(self, mock_crema_dataset, tmp_path) -> None:
        mock_crema_dataset.side_effect = [
            _MockClassificationDataset(size=8),
            _MockClassificationDataset(size=4),
        ]

        benchmark = CREMADClassificationBenchmark(
            cache_dir=tmp_path / "cache",
            enable_cache=False,
        )

        assert isinstance(benchmark.dataset, BaseDataset)
        assert isinstance(benchmark.testset, BaseDataset)
        assert benchmark.dataset.normalize is True
        assert benchmark.testset.normalize is True
        assert benchmark.dataset.sample_rate == 16000
        assert benchmark.dataset.length == 48000
        assert benchmark.dataset[0]["waveform"].shape == (1, 48000)
        assert benchmark.dataset[0]["sample_rate"] == 16000

        assert isinstance(benchmark.task, GlobalClassificationTask)
        assert isinstance(benchmark.task.classifier, LinearBenchmarkClassifier)
        assert benchmark.task.classifier.model_type == "ridge"
        assert benchmark.dataset_name == "CREMA-D train"
        assert benchmark.testset_name == "CREMA-D val"

        model = DummyAudioModel(representation_dim=32)
        res = benchmark.run(
            model=model, model_name="dummy", batch_size=4, show_progress=False
        )
        assert "accuracy" in res.metrics

    @patch("audio_benchmark.benchmarks.cremad.CREMADDataset")
    def test_max_size(self, mock_crema_dataset, tmp_path) -> None:
        mock_crema_dataset.side_effect = [
            _MockClassificationDataset(size=10),
            _MockClassificationDataset(size=6),
        ]

        benchmark = CREMADClassificationBenchmark(
            max_size=4,
            cache_dir=tmp_path / "cache",
            enable_cache=False,
        )

        assert len(benchmark.dataset) == 4
        assert len(benchmark.testset) == 4

    def test_invalid_max_size_raises(self) -> None:
        with pytest.raises(ValueError, match="max_size must be a positive integer"):
            CREMADClassificationBenchmark(max_size=-1)


class TestTIMITSpeechDurationRegressionBenchmark:
    """Tests for TIMITSpeechDurationRegressionBenchmark."""

    @patch("audio_benchmark.benchmarks.timit.TIMITDataset")
    def test_default_configuration_and_run(self, mock_timit_dataset, tmp_path) -> None:
        mock_timit_dataset.side_effect = [
            _MockTIMITDataset(size=8),
            _MockTIMITDataset(size=4),
        ]

        benchmark = TIMITSpeechDurationRegressionBenchmark(
            cache_dir=tmp_path / "cache",
            enable_cache=False,
        )

        assert isinstance(benchmark.dataset, BaseDataset)
        assert isinstance(benchmark.testset, BaseDataset)
        assert benchmark.dataset.normalize is True
        assert benchmark.testset.normalize is True
        assert benchmark.dataset.sample_rate == 16000
        assert benchmark.dataset.length == 48000
        assert benchmark.dataset[0]["waveform"].shape == (1, 48000)
        assert benchmark.dataset[0]["sample_rate"] == 16000
        assert "speech_duration" in benchmark.dataset[0]["metadata"]

        assert isinstance(benchmark.task, RegressionTask)
        assert isinstance(benchmark.task.regressor, TransformerBenchmarkRegressor)
        assert benchmark.task.target_key == "speech_duration"
        assert benchmark.dataset_name == "TIMIT train (speech_duration)"
        assert benchmark.testset_name == "TIMIT test (speech_duration)"

        model = DummyAudioModel(representation_dim=32)
        res = benchmark.run(
            model=model, model_name="dummy", batch_size=4, show_progress=False
        )
        assert "r2" in res.metrics or "value" in res.metrics

    @patch("audio_benchmark.benchmarks.timit.TIMITDataset")
    def test_max_size(self, mock_timit_dataset, tmp_path) -> None:
        mock_timit_dataset.side_effect = [
            _MockTIMITDataset(size=10),
            _MockTIMITDataset(size=6),
        ]

        benchmark = TIMITSpeechDurationRegressionBenchmark(
            max_size=5,
            cache_dir=tmp_path / "cache",
            enable_cache=False,
        )

        assert len(benchmark.dataset) == 5
        assert len(benchmark.testset) == 5

    def test_invalid_max_size_raises(self) -> None:
        with pytest.raises(ValueError, match="max_size must be a positive integer"):
            TIMITSpeechDurationRegressionBenchmark(max_size=0)


class TestVCTKGenderClassificationBenchmark:
    """Tests for VCTKGenderClassificationBenchmark."""

    @patch("audio_benchmark.benchmarks.vctk.VCTKDataset")
    def test_default_configuration_and_run(self, mock_vctk_dataset, tmp_path) -> None:
        mock_vctk_dataset.side_effect = [
            _MockClassificationDataset(size=8, label_key="gender"),
            _MockClassificationDataset(size=4, label_key="gender"),
        ]

        benchmark = VCTKGenderClassificationBenchmark(
            cache_dir=tmp_path / "cache",
            enable_cache=False,
        )

        assert isinstance(benchmark.dataset, BaseDataset)
        assert isinstance(benchmark.testset, BaseDataset)
        assert benchmark.dataset.normalize is True
        assert benchmark.testset.normalize is True
        assert benchmark.dataset.sample_rate == 16000
        assert benchmark.dataset.length == 48000
        assert benchmark.dataset[0]["waveform"].shape == (1, 48000)
        assert benchmark.dataset[0]["sample_rate"] == 16000
        assert "gender" in benchmark.dataset[0]["metadata"]

        assert isinstance(benchmark.task, GlobalClassificationTask)
        assert isinstance(benchmark.task.classifier, LinearBenchmarkClassifier)
        assert benchmark.task.classifier.model_type == "ridge"
        assert benchmark.task.label_key == "gender"
        assert benchmark.dataset_name == "VCTK train"
        assert benchmark.testset_name == "VCTK val"

        model = DummyAudioModel(representation_dim=32)
        res = benchmark.run(
            model=model, model_name="dummy", batch_size=4, show_progress=False
        )
        assert "accuracy" in res.metrics

    @patch("audio_benchmark.benchmarks.vctk.VCTKDataset")
    def test_max_size(self, mock_vctk_dataset, tmp_path) -> None:
        mock_vctk_dataset.side_effect = [
            _MockClassificationDataset(size=10, label_key="gender"),
            _MockClassificationDataset(size=6, label_key="gender"),
        ]

        benchmark = VCTKGenderClassificationBenchmark(
            max_size=4,
            cache_dir=tmp_path / "cache",
            enable_cache=False,
        )

        assert len(benchmark.dataset) == 4
        assert len(benchmark.testset) == 4

    def test_invalid_max_size_raises(self) -> None:
        with pytest.raises(ValueError, match="max_size must be a positive integer"):
            VCTKGenderClassificationBenchmark(max_size=-2)


class TestVCTKAgeRegressionBenchmark:
    """Tests for VCTKAgeRegressionBenchmark."""

    @patch("audio_benchmark.benchmarks.vctk_age.VCTKDataset")
    def test_default_configuration_and_run(self, mock_vctk_dataset, tmp_path) -> None:
        mock_vctk_dataset.side_effect = [
            _MockClassificationDataset(size=8, label_key="age"),
            _MockClassificationDataset(size=4, label_key="age"),
        ]

        benchmark = VCTKAgeRegressionBenchmark(
            cache_dir=tmp_path / "cache",
            enable_cache=False,
        )

        assert isinstance(benchmark.dataset, BaseDataset)
        assert isinstance(benchmark.testset, BaseDataset)
        assert benchmark.dataset.normalize is True
        assert benchmark.testset.normalize is True
        assert benchmark.dataset.sample_rate == 16000
        assert benchmark.dataset.length == 48000
        assert benchmark.dataset[0]["waveform"].shape == (1, 48000)
        assert benchmark.dataset[0]["sample_rate"] == 16000
        assert "age" in benchmark.dataset[0]["metadata"]

        assert isinstance(benchmark.task, RegressionTask)
        assert isinstance(benchmark.task.regressor, LinearBenchmarkRegressor)
        assert benchmark.task.target_key == "age"
        assert benchmark.dataset_name == "VCTK train (age)"
        assert benchmark.testset_name == "VCTK val (age)"

        model = DummyAudioModel(representation_dim=32)
        res = benchmark.run(
            model=model, model_name="dummy", batch_size=4, show_progress=False
        )
        assert "r2" in res.metrics or "value" in res.metrics

    @patch("audio_benchmark.benchmarks.vctk_age.VCTKDataset")
    def test_max_size(self, mock_vctk_dataset, tmp_path) -> None:
        mock_vctk_dataset.side_effect = [
            _MockClassificationDataset(size=10, label_key="age"),
            _MockClassificationDataset(size=6, label_key="age"),
        ]

        benchmark = VCTKAgeRegressionBenchmark(
            max_size=3,
            cache_dir=tmp_path / "cache",
            enable_cache=False,
        )

        assert len(benchmark.dataset) == 3
        assert len(benchmark.testset) == 3

    def test_invalid_max_size_raises(self) -> None:
        with pytest.raises(ValueError, match="max_size must be a positive integer"):
            VCTKAgeRegressionBenchmark(max_size=0)


class TestTIMITPhonemeSequenceModelingBenchmark:
    """Tests for TIMITPhonemeSequenceModelingBenchmark."""

    @patch("audio_benchmark.benchmarks.timit_phonemes.TIMITDataset")
    def test_default_configuration_and_run(self, mock_timit_dataset, tmp_path) -> None:
        mock_timit_dataset.side_effect = [
            _MockTIMITDataset(size=4),
            _MockTIMITDataset(size=2),
        ]

        benchmark = TIMITPhonemeSequenceModelingBenchmark(
            cache_dir=tmp_path / "cache",
            enable_cache=False,
        )

        assert isinstance(benchmark.dataset, BaseDataset)
        assert isinstance(benchmark.testset, BaseDataset)
        assert benchmark.dataset.normalize is True
        assert benchmark.testset.normalize is True
        assert benchmark.dataset.sample_rate == 16000
        assert benchmark.dataset.length is None
        assert benchmark.dataset[0]["waveform"].shape == (1, 16000)
        assert benchmark.dataset[1]["waveform"].shape == (1, 17600)
        assert benchmark.dataset[0]["sample_rate"] == 16000
        assert "phonemes" in benchmark.dataset[0]["metadata"]

        assert isinstance(benchmark.task, SequenceModelingTask)
        assert isinstance(benchmark.task.probe, TransformerDecoderProbe)
        assert benchmark.task.target_key == "phonemes"
        assert benchmark.dataset_name == "TIMIT train (phonemes)"
        assert benchmark.testset_name == "TIMIT test (phonemes)"

        benchmark.task.probe.epochs = 1
        model = DummyAudioModel(representation_dim=16)
        res = benchmark.run(
            model=model, model_name="dummy", batch_size=2, show_progress=False
        )
        assert "token_error_rate" in res.metrics or "value" in res.metrics

    @patch("audio_benchmark.benchmarks.timit_phonemes.TIMITDataset")
    def test_max_size(self, mock_timit_dataset, tmp_path) -> None:
        mock_timit_dataset.side_effect = [
            _MockTIMITDataset(size=10),
            _MockTIMITDataset(size=6),
        ]

        benchmark = TIMITPhonemeSequenceModelingBenchmark(
            max_size=3,
            cache_dir=tmp_path / "cache",
            enable_cache=False,
        )

        assert len(benchmark.dataset) == 3
        assert len(benchmark.testset) == 3

    def test_invalid_max_size_raises(self) -> None:
        with pytest.raises(ValueError, match="max_size must be a positive integer"):
            TIMITPhonemeSequenceModelingBenchmark(max_size=-1)


class TestTIMITTextSequenceModelingBenchmark:
    """Tests for TIMITTextSequenceModelingBenchmark."""

    @patch("audio_benchmark.benchmarks.timit_text.TIMITDataset")
    def test_default_configuration_and_run(self, mock_timit_dataset, tmp_path) -> None:
        mock_timit_dataset.side_effect = [
            _MockTIMITDataset(size=4),
            _MockTIMITDataset(size=2),
        ]

        benchmark = TIMITTextSequenceModelingBenchmark(
            cache_dir=tmp_path / "cache",
            enable_cache=False,
        )

        assert isinstance(benchmark.dataset, BaseDataset)
        assert isinstance(benchmark.testset, BaseDataset)
        assert benchmark.dataset.normalize is True
        assert benchmark.testset.normalize is True
        assert benchmark.dataset.sample_rate == 16000
        assert benchmark.dataset.length is None
        assert benchmark.dataset[0]["waveform"].shape == (1, 16000)
        assert benchmark.dataset[1]["waveform"].shape == (1, 17600)
        assert benchmark.dataset[0]["sample_rate"] == 16000
        assert "text" in benchmark.dataset[0]["metadata"]

        assert isinstance(benchmark.task, SequenceModelingTask)
        assert isinstance(benchmark.task.probe, TransformerDecoderProbe)
        assert benchmark.task.target_key == "text"
        assert benchmark.dataset_name == "TIMIT train (text)"
        assert benchmark.testset_name == "TIMIT test (text)"

        benchmark.task.probe.epochs = 1
        model = DummyAudioModel(representation_dim=16)
        res = benchmark.run(
            model=model, model_name="dummy", batch_size=2, show_progress=False
        )
        assert "token_error_rate" in res.metrics or "value" in res.metrics

    @patch("audio_benchmark.benchmarks.timit_text.TIMITDataset")
    def test_max_size(self, mock_timit_dataset, tmp_path) -> None:
        mock_timit_dataset.side_effect = [
            _MockTIMITDataset(size=10),
            _MockTIMITDataset(size=6),
        ]

        benchmark = TIMITTextSequenceModelingBenchmark(
            max_size=3,
            cache_dir=tmp_path / "cache",
            enable_cache=False,
        )

        assert len(benchmark.dataset) == 3
        assert len(benchmark.testset) == 3

    def test_invalid_max_size_raises(self) -> None:
        with pytest.raises(ValueError, match="max_size must be a positive integer"):
            TIMITTextSequenceModelingBenchmark(max_size=0)
