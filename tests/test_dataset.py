"""Tests for dataset validation and utilities."""

import pytest
import torch
from torch.utils.data import Dataset

from audio_benchmark.dataset import (
    collate_audio_batch,
    get_dataset_name,
    parse_dataset_item,
    validate_dataset,
)
from audio_benchmark.exceptions import InvalidDatasetError


class ValidDictDataset(Dataset):
    def __init__(self, length=10):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {
            "waveform": torch.randn(1, 1600),
            "sample_rate": 16000,
            "metadata": {"label": idx % 2},
        }


class RawTupleDataset(Dataset):
    """Dataset returning a tuple instead of the required 3-key dictionary."""

    def __init__(self, length=5):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return torch.randn(1, 1600), {"label": idx % 2}


class AudioDatasetWrapper(Dataset):
    """Example of user wrapper converting non-compliant tuple dataset to 3-key dict."""

    def __init__(self, raw_ds):
        self.raw_ds = raw_ds

    def __len__(self):
        return len(self.raw_ds)

    def __getitem__(self, idx):
        audio, meta = self.raw_ds[idx]
        return {
            "waveform": audio,
            "sample_rate": 16000,
            "metadata": meta,
        }


def test_parse_dataset_item_valid_dict():
    audio = torch.randn(1, 800)
    meta = {"label": 0}
    item = {
        "waveform": audio,
        "sample_rate": 16000,
        "metadata": meta,
    }
    parsed = parse_dataset_item(item)
    assert parsed["waveform"].shape == (1, 800)
    assert parsed["sample_rate"] == 16000
    assert parsed["metadata"] == meta


def test_parse_dataset_item_1d_tensor():
    item = {
        "waveform": torch.randn(800),
        "sample_rate": 16000,
        "metadata": {"label": 1},
    }
    parsed = parse_dataset_item(item)
    assert parsed["waveform"].shape == (1, 800)


def test_parse_dataset_item_int_waveform_converted_to_float():
    item = {
        "waveform": torch.zeros((1, 800), dtype=torch.int32),
        "sample_rate": 16000,
        "metadata": {"label": 0},
    }
    parsed = parse_dataset_item(item)
    assert parsed["waveform"].dtype == torch.float32


def test_parse_dataset_item_rejects_tuple_and_encourages_wrapper():
    item = (torch.randn(1, 800), {"label": 0})
    with pytest.raises(InvalidDatasetError, match="must return a dict with keys"):
        parse_dataset_item(item)


def test_parse_dataset_item_rejects_missing_keys():
    # Missing sample_rate
    item = {"waveform": torch.randn(1, 800), "metadata": {"label": 0}}
    with pytest.raises(InvalidDatasetError, match="missing required key"):
        parse_dataset_item(item)

    # Missing waveform
    item2 = {"sample_rate": 16000, "metadata": {"label": 0}}
    with pytest.raises(InvalidDatasetError, match="missing required key"):
        parse_dataset_item(item2)

    # Missing metadata
    item3 = {"waveform": torch.randn(1, 800), "sample_rate": 16000}
    with pytest.raises(InvalidDatasetError, match="missing required key"):
        parse_dataset_item(item3)


def test_parse_dataset_item_rejects_non_tensor_waveform():
    item = {
        "waveform": [0.1, 0.2, 0.3],
        "sample_rate": 16000,
        "metadata": {"label": 0},
    }
    with pytest.raises(InvalidDatasetError, match="must be a torch.Tensor"):
        parse_dataset_item(item)


def test_parse_dataset_item_rejects_non_numeric_sample_rate():
    item = {
        "waveform": torch.randn(1, 800),
        "sample_rate": "sixteen_thousand",
        "metadata": {"label": 0},
    }
    with pytest.raises(InvalidDatasetError, match="positive integer or float"):
        parse_dataset_item(item)


def test_parse_dataset_item_rejects_non_dict_metadata():
    item = {
        "waveform": torch.randn(1, 800),
        "sample_rate": 16000,
        "metadata": "not_a_dict",
    }
    with pytest.raises(InvalidDatasetError, match="must be a dict"):
        parse_dataset_item(item)


def test_parse_dataset_item_invalid_audio_channels():
    item = {
        "waveform": torch.randn(3, 800),
        "sample_rate": 16000,
        "metadata": {"label": 0},
    }
    with pytest.raises(InvalidDatasetError, match="must have 1 channel"):
        parse_dataset_item(item)


def test_validate_dataset_success():
    ds = ValidDictDataset()
    validate_dataset(ds, required_metadata_keys=["label"])


def test_user_wrapper_resolves_tuple_dataset():
    raw_ds = RawTupleDataset()
    # Raw dataset fails validation because it returns a tuple
    with pytest.raises(InvalidDatasetError, match="must return a dict"):
        validate_dataset(raw_ds)

    # Wrapped dataset succeeds
    wrapped_ds = AudioDatasetWrapper(raw_ds)
    validate_dataset(wrapped_ds, required_metadata_keys=["label"])
    assert len(wrapped_ds) == 5
    item = wrapped_ds[0]
    assert isinstance(item, dict)
    assert isinstance(item["waveform"], torch.Tensor)
    assert isinstance(item["sample_rate"], int)
    assert isinstance(item["metadata"], dict)


def test_validate_dataset_missing_key():
    ds = ValidDictDataset()
    with pytest.raises(InvalidDatasetError, match="missing required key"):
        validate_dataset(ds, required_metadata_keys=["non_existent_key"])


def test_validate_empty_dataset():
    class EmptyDataset(Dataset):
        def __len__(self):
            return 0

        def __getitem__(self, idx):
            raise IndexError

    with pytest.raises(InvalidDatasetError, match="len == 0"):
        validate_dataset(EmptyDataset())


def test_collate_audio_batch_uniform_length():
    batch = [
        {
            "waveform": torch.randn(1, 1000),
            "sample_rate": 16000,
            "metadata": {"label": 0},
        },
        {
            "waveform": torch.randn(1, 1000),
            "sample_rate": 16000,
            "metadata": {"label": 1},
        },
    ]
    audio_batch, meta_batch = collate_audio_batch(batch)
    assert audio_batch.shape == (2, 1, 1000)
    assert len(meta_batch) == 2


def test_collate_audio_batch_variable_length_padding():
    batch = [
        {
            "waveform": torch.randn(1, 500),
            "sample_rate": 16000,
            "metadata": {"label": 0},
        },
        {
            "waveform": torch.randn(1, 1200),
            "sample_rate": 16000,
            "metadata": {"label": 1},
        },
    ]
    audio_batch, _meta_batch = collate_audio_batch(batch)
    assert audio_batch.shape == (2, 1, 1200)
    # Check that the first audio was padded with zeros
    assert torch.all(audio_batch[0, 0, 500:] == 0.0)


def test_get_dataset_name():
    ds = ValidDictDataset()
    assert get_dataset_name(ds) == "ValidDictDataset"

    ds.name = "custom_dataset_name"
    assert get_dataset_name(ds) == "custom_dataset_name"
