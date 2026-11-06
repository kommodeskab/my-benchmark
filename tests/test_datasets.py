"""Tests for CREMA-D, TIMIT, and Speech Commands dataset implementations."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from audio_benchmark.dataset import BaseDataset, parse_dataset_item
from audio_benchmark.datasets import (
    CREMADDataset,
    SpeechCommandsDataset,
    TIMITDataset,
    VCTKDataset,
)
from audio_benchmark.datasets.speechcommands import SPEECHCOMMANDS_LABELS


def _create_mock_hf_cremad_dataset(size: int = 10):
    samples = []
    for i in range(size):
        samples.append(
            {
                "audio": {
                    "array": [0.1 * (i + 1), 0.2 * (i + 1), 0.3 * (i + 1)],
                    "sampling_rate": 16000,
                },
                "gender": "Male" if i % 2 == 0 else "Female",
                "transcription": f"Sentence {i}",
                "major_emotion": "HAP" if i % 2 == 0 else "SAD",
            }
        )
    mock_ds = MagicMock()
    mock_ds.__len__.return_value = len(samples)
    mock_ds.__getitem__.side_effect = lambda idx: samples[idx]
    return mock_ds


def _create_mock_hf_timit_dataset(size: int = 4):
    samples = []
    for i in range(size):
        samples.append(
            {
                "audio": {
                    "array": [0.05, 0.1, 0.15, 0.2],
                    "sampling_rate": 16000,
                },
                "dialect_region": "DR1",
                "speaker_id": f"SPK{i:02d}",
                "words": {
                    "text": ["she", "had", "your"],
                    "start": [1600, 4800, 8000],
                    "end": [4800, 8000, 16000],
                },
                "phonemes": {
                    "text": ["sh", "iy", "hh", "ae", "d"],
                    "start": [1600, 3200, 4800, 6400, 8000],
                    "end": [3200, 4800, 6400, 8000, 9600],
                },
            }
        )
    mock_ds = MagicMock()
    mock_ds.__len__.return_value = len(samples)
    mock_ds.__getitem__.side_effect = lambda idx: samples[idx]
    return mock_ds


def _create_mock_torchaudio_speechcommands(size: int = 3):
    labels = ["bed", "cat", "zero"]
    samples = []
    for i in range(size):
        waveform = torch.zeros((1, 16000), dtype=torch.float32)
        sr = 16000
        label_str = labels[i % len(labels)]
        speaker = f"speaker_{i}"
        utterance = i
        samples.append((waveform, sr, label_str, speaker, utterance))

    mock_ds = MagicMock()
    mock_ds.__len__.return_value = len(samples)
    mock_ds.__getitem__.side_effect = lambda idx: samples[idx]
    return mock_ds


class TestCREMADDataset:
    """Tests for CREMADDataset."""

    def test_invalid_split_raises(self) -> None:
        with pytest.raises(ValueError, match="split must be 'train' or 'val'"):
            CREMADDataset(split="test")  # type: ignore[arg-type]

    @patch("audio_benchmark.datasets.cremad.load_dataset")
    def test_splits_and_output_schema(self, mock_load_dataset: MagicMock) -> None:
        mock_ds = _create_mock_hf_cremad_dataset(size=10)
        mock_load_dataset.return_value = mock_ds

        train_ds = CREMADDataset(split="train", data_dir="/custom/cache", seed=42)
        val_ds = CREMADDataset(split="val", data_dir="/custom/cache", seed=42)

        mock_load_dataset.assert_called_with(
            "AbstractTTS/CREMA-D",
            cache_dir="/custom/cache",
            split="train",
        )

        assert len(train_ds) == 8
        assert len(val_ds) == 2

        sample = train_ds[0]
        parsed = parse_dataset_item(sample)
        assert parsed["waveform"].shape == (1, 3)
        assert parsed["sample_rate"] == 16000
        assert parsed["metadata"]["gender"] in ("Male", "Female")
        assert "transcription" in parsed["metadata"]
        assert parsed["metadata"]["label"] in ("HAP", "SAD")

    @patch("audio_benchmark.datasets.cremad.load_dataset")
    def test_base_dataset_wrapping(self, mock_load_dataset: MagicMock) -> None:
        mock_ds = _create_mock_hf_cremad_dataset(size=5)
        mock_load_dataset.return_value = mock_ds

        train_ds = CREMADDataset(split="train", seed=42)
        wrapped = BaseDataset(
            dataset=train_ds,
            dataset_name="crema_d_train",
            sample_rate=16000,
            length=16000,
        )

        assert len(wrapped) == len(train_ds)
        item = wrapped[0]
        assert item["waveform"].shape == (1, 16000)
        assert item["sample_rate"] == 16000
        assert "label" in item["metadata"]


class TestTIMITDataset:
    """Tests for TIMITDataset."""

    def test_invalid_split_raises(self) -> None:
        with pytest.raises(ValueError, match="split must be 'train' or 'test'"):
            TIMITDataset(split="val")  # type: ignore[arg-type]

    @patch("audio_benchmark.datasets.timit.load_dataset")
    def test_metadata_keys_and_schema(self, mock_load_dataset: MagicMock) -> None:
        mock_ds = _create_mock_hf_timit_dataset(size=4)
        mock_load_dataset.return_value = mock_ds

        timit = TIMITDataset(split="train", data_dir="/data/timit")
        mock_load_dataset.assert_called_once_with(
            "Andemand11/TIMIT",
            split="train",
            cache_dir="/data/timit",
        )

        assert len(timit) == 4
        sample = timit[0]
        parsed = parse_dataset_item(sample)
        assert parsed["waveform"].shape == (1, 4)
        assert parsed["sample_rate"] == 16000

        meta = parsed["metadata"]
        assert set(meta.keys()) == {
            "dialect_region",
            "speaker_id",
            "text",
            "phonemes",
            "speech_start",
            "speech_end",
            "speech_duration",
        }
        assert meta["dialect_region"] == "DR1"
        assert meta["speaker_id"] == "SPK00"
        assert meta["speech_start"] == 1600 / 16000
        assert meta["speech_end"] == 16000 / 16000
        assert meta["speech_duration"] == (16000 - 1600) / 16000


class TestSpeechCommandsDataset:
    """Tests for SpeechCommandsDataset."""

    def test_invalid_split_raises(self) -> None:
        with pytest.raises(ValueError, match="split must be one of"):
            SpeechCommandsDataset(split="unknown")  # type: ignore[arg-type]

    @patch("torchaudio.datasets.SPEECHCOMMANDS")
    def test_output_and_label_encoding(self, mock_torchaudio_sc: MagicMock) -> None:
        mock_ds = _create_mock_torchaudio_speechcommands(size=3)
        mock_torchaudio_sc.return_value = mock_ds

        sc = SpeechCommandsDataset(split="train", data_dir="/data/sc", download=False)
        mock_torchaudio_sc.assert_called_once_with(
            root="/data/sc",
            subset="training",
            download=False,
        )

        assert len(sc) == 3
        sample = sc[0]
        parsed = parse_dataset_item(sample)
        assert parsed["waveform"].shape == (1, 16000)
        assert parsed["sample_rate"] == 16000

        meta = parsed["metadata"]
        assert meta["text"] == "bed"
        assert meta["label"] == SPEECHCOMMANDS_LABELS.index("bed")
        assert meta["speaker_id"] == "speaker_0"
        assert meta["utterance_number"] == 0

    @patch("torchaudio.datasets.SPEECHCOMMANDS")
    def test_base_dataset_wrapping(self, mock_torchaudio_sc: MagicMock) -> None:
        mock_ds = _create_mock_torchaudio_speechcommands(size=3)
        mock_torchaudio_sc.return_value = mock_ds

        sc = SpeechCommandsDataset(split="val", download=True)
        wrapped = BaseDataset(
            dataset=sc,
            dataset_name="sc_val",
            sample_rate=16000,
        )
        assert len(wrapped) == 3
        item = wrapped[0]
        assert "label" in item["metadata"]

    @patch("torchaudio.datasets.SPEECHCOMMANDS")
    @patch("audio_benchmark.datasets.speechcommands.os.makedirs")
    def test_default_data_dir_creates_data_directory(
        self,
        mock_makedirs: MagicMock,
        mock_torchaudio_sc: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("DATA_PATH", raising=False)
        mock_torchaudio_sc.return_value = _create_mock_torchaudio_speechcommands(size=1)
        SpeechCommandsDataset(split="train", data_dir=None)
        mock_makedirs.assert_called_once_with("./data", exist_ok=True)


def _create_mock_hf_vctk_dataset(size: int = 4):
    samples = []
    for i in range(size):
        samples.append(
            {
                "flac": {
                    "array": [
                        0.01 * (i + 1),
                        0.02 * (i + 1),
                        0.03 * (i + 1),
                        0.04 * (i + 1),
                        0.05 * (i + 1),
                        0.06 * (i + 1),
                    ],
                    "sampling_rate": 48000,
                },
                "accent": "English",
                "age": 25 + i,
                "gender": "M" if i % 2 == 0 else "F",
                "txt": f"Sample transcription {i}",
            }
        )
    mock_ds = MagicMock()
    mock_ds.__len__.return_value = len(samples)
    mock_ds.__getitem__.side_effect = lambda idx: samples[idx]
    return mock_ds


class TestVCTKDataset:
    """Tests for VCTKDataset."""

    def test_invalid_split_raises(self) -> None:
        with pytest.raises(ValueError, match="split must be 'train' or 'val'"):
            VCTKDataset(split="test")  # type: ignore[arg-type]

    @patch("audio_benchmark.datasets.vctk.load_dataset")
    def test_splits_and_output_schema(self, mock_load_dataset: MagicMock) -> None:
        mock_ds = _create_mock_hf_vctk_dataset(size=4)
        mock_load_dataset.return_value = mock_ds

        train_ds = VCTKDataset(split="train", data_dir="/data/vctk")
        mock_load_dataset.assert_called_with(
            "badayvedat/VCTK",
            split="train",
            cache_dir="/data/vctk",
        )

        val_ds = VCTKDataset(split="val", data_dir="/data/vctk")
        mock_load_dataset.assert_called_with(
            "badayvedat/VCTK",
            split="validation",
            cache_dir="/data/vctk",
        )

        assert len(train_ds) == 4
        assert len(val_ds) == 4

        sample = train_ds[0]
        parsed = parse_dataset_item(sample)
        assert parsed["waveform"].shape == (1, 6)
        assert parsed["sample_rate"] == 48000

        meta = parsed["metadata"]
        assert meta["accent"] == "English"
        assert meta["age"] == 25
        assert meta["gender"] == "M"
        assert meta["text"] == "Sample transcription 0"

    @patch("audio_benchmark.datasets.vctk.load_dataset")
    def test_base_dataset_wrapping(self, mock_load_dataset: MagicMock) -> None:
        mock_ds = _create_mock_hf_vctk_dataset(size=2)
        mock_load_dataset.return_value = mock_ds

        train_ds = VCTKDataset(split="train")
        wrapped = BaseDataset(
            dataset=train_ds,
            dataset_name="vctk_train",
            sample_rate=16000,
            length=48000,
        )

        assert len(wrapped) == 2
        item = wrapped[0]
        assert item["waveform"].shape == (1, 48000)
        assert item["sample_rate"] == 16000
        assert item["metadata"]["gender"] == "M"
