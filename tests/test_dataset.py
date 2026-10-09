"""Tests for dataset validation and utilities."""

import pytest
import torch
from torch.utils.data import Dataset

from audio_benchmark.dataset import (
    BaseDataset,
    collate_audio_batch,
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


def test_parse_dataset_item_rejects_1d_tensor():
    item = {
        "waveform": torch.randn(800),
        "sample_rate": 16000,
        "metadata": {"label": 1},
    }
    with pytest.raises(InvalidDatasetError, match="shape \\(1, T\\)"):
        parse_dataset_item(item)


def test_parse_dataset_item_valid_float_waveform():
    item = {
        "waveform": torch.zeros((1, 800), dtype=torch.float32),
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
    with pytest.raises(InvalidDatasetError, match="positive integer"):
        parse_dataset_item(item)


def test_parse_dataset_item_rejects_float_sample_rate():
    item = {
        "waveform": torch.randn(1, 800),
        "sample_rate": 16000.0,
        "metadata": {"label": 0},
    }
    with pytest.raises(InvalidDatasetError, match="positive integer"):
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
    with pytest.raises(InvalidDatasetError, match="shape \\(1, T\\)"):
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


class RichMetadataDataset(Dataset):
    """Synthetic dataset with allowed metadata types for testing BaseDataset."""

    def __init__(self, audio_length: int = 16000, sample_rate: int = 16000):
        self.audio_length = audio_length
        self.sample_rate = sample_rate

    def __len__(self) -> int:
        return 3

    def __getitem__(self, idx: int):
        return {
            "waveform": torch.ones(1, self.audio_length) * (idx + 1),
            "sample_rate": self.sample_rate,
            "metadata": {
                "label": idx,
                "tokens": ["word_a", "word_b", "word_c"],
                "target": 42.0,
                "frame_targets": [1.5 * (idx + 1)] * self.audio_length,
                "frame_labels": [idx] * self.audio_length,
            },
        }


def test_base_dataset_resample_only():
    raw_ds = RichMetadataDataset(audio_length=16000, sample_rate=16000)
    wrapped = BaseDataset(raw_ds, dataset_name="rich_ds", sample_rate=8000, length=None)
    assert len(wrapped) == 3
    assert wrapped.dataset_name == "rich_ds"
    assert wrapped.name == "rich_ds"

    item = wrapped[0]
    assert item["waveform"].shape == (1, 8000)
    assert item["sample_rate"] == 8000
    assert len(item["metadata"]["frame_targets"]) == 8000
    assert len(item["metadata"]["frame_labels"]) == 8000
    # tokens, label, and target should not be cropped or modified
    assert item["metadata"]["tokens"] == ["word_a", "word_b", "word_c"]
    assert item["metadata"]["label"] == 0
    assert item["metadata"]["target"] == 42.0


def test_base_dataset_crop_waveform_and_metadata():
    raw_ds = RichMetadataDataset(audio_length=20000, sample_rate=16000)
    wrapped = BaseDataset(
        raw_ds, dataset_name="rich_ds", sample_rate=16000, length=16000
    )

    item = wrapped[0]
    assert item["waveform"].shape == (1, 16000)
    assert item["sample_rate"] == 16000
    assert len(item["metadata"]["frame_targets"]) == 16000
    assert len(item["metadata"]["frame_labels"]) == 16000
    # tokens, label, target must not be cropped
    assert item["metadata"]["tokens"] == ["word_a", "word_b", "word_c"]
    assert item["metadata"]["label"] == 0


def test_base_dataset_pad_waveform_and_metadata():
    raw_ds = RichMetadataDataset(audio_length=10000, sample_rate=16000)
    wrapped = BaseDataset(
        raw_ds, dataset_name="rich_ds", sample_rate=16000, length=16000
    )

    item = wrapped[0]
    assert item["waveform"].shape == (1, 16000)
    assert torch.all(item["waveform"][0, 10000:] == 0.0)
    assert len(item["metadata"]["frame_targets"]) == 16000
    assert len(item["metadata"]["frame_labels"]) == 16000
    # verify zero padding in metadata
    assert item["metadata"]["frame_targets"][10000:] == [0.0] * 6000
    assert item["metadata"]["frame_labels"][10000:] == [0] * 6000
    # tokens and label must remain unpadded / uncropped
    assert item["metadata"]["tokens"] == ["word_a", "word_b", "word_c"]
    assert item["metadata"]["label"] == 0


def test_base_dataset_resample_and_crop():
    raw_ds = RichMetadataDataset(audio_length=24000, sample_rate=16000)
    wrapped = BaseDataset(raw_ds, dataset_name="rich_ds", sample_rate=8000, length=4000)

    item = wrapped[0]
    assert item["waveform"].shape == (1, 4000)
    assert item["sample_rate"] == 8000
    assert len(item["metadata"]["frame_targets"]) == 4000
    assert len(item["metadata"]["frame_labels"]) == 4000
    assert item["metadata"]["tokens"] == ["word_a", "word_b", "word_c"]


def test_base_dataset_tensor_metadata():
    class TensorFrameDataset(Dataset):
        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return {
                "waveform": torch.ones(1, 800),
                "sample_rate": 16000,
                "metadata": {
                    "frame_targets": torch.ones(800, dtype=torch.float32),
                    "tokens": ["tok"],
                },
            }

    ds = TensorFrameDataset()
    wrapped = BaseDataset(ds, dataset_name="tensor_ds", sample_rate=16000, length=400)
    item = wrapped[0]

    assert item["waveform"].shape == (1, 400)
    assert isinstance(item["metadata"]["frame_targets"], torch.Tensor)
    assert item["metadata"]["frame_targets"].shape == (400,)
    assert item["metadata"]["tokens"] == ["tok"]


def test_base_dataset_validation_errors():
    ds = ValidDictDataset()

    with pytest.raises(ValueError, match="dataset_name must be a non-empty string"):
        BaseDataset(ds, dataset_name="", sample_rate=16000)

    with pytest.raises(ValueError, match="sample_rate must be a positive integer"):
        BaseDataset(ds, dataset_name="ds", sample_rate=-16000)

    with pytest.raises(ValueError, match="length must be a positive integer"):
        BaseDataset(ds, dataset_name="ds", sample_rate=16000, length=0)

    with pytest.raises(InvalidDatasetError, match="must implement __len__"):
        BaseDataset(object(), dataset_name="ds", sample_rate=16000)

    with pytest.raises(InvalidDatasetError, match="cannot be a string or bytes"):
        BaseDataset("not_a_dataset", dataset_name="ds", sample_rate=16000)


def test_base_dataset_rejects_1d_waveform():
    class OneDWaveformDataset(Dataset):
        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return {
                "waveform": torch.randn(1600),  # 1D waveform not allowed
                "sample_rate": 16000,
                "metadata": {"label": 0},
            }

    wrapped = BaseDataset(
        OneDWaveformDataset(), dataset_name="bad_audio", sample_rate=16000
    )
    with pytest.raises(InvalidDatasetError, match="shape \\(1, T\\)"):
        _ = wrapped[0]


def test_base_dataset_rejects_empty_or_unknown_metadata():
    class BadMetadataDataset(Dataset):
        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return {
                "waveform": torch.randn(1, 1600),
                "sample_rate": 16000,
                "metadata": {"unknown_key": "val"},
            }

    wrapped = BaseDataset(
        BadMetadataDataset(), dataset_name="bad_meta", sample_rate=16000
    )
    with pytest.raises(
        InvalidDatasetError, match="Metadata must contain at least one of"
    ):
        _ = wrapped[0]


def test_base_dataset_task_compatibility():
    from audio_benchmark.dummy import DummyAudioDataset
    from audio_benchmark.regressors.models import LinearBenchmarkRegressor
    from audio_benchmark.tasks.frame_regression import FrameRegressionTask

    dummy_ds = DummyAudioDataset(num_samples=4, audio_length=16000, sample_rate=16000)
    wrapped = BaseDataset(
        dummy_ds, dataset_name="dummy_wrapped", sample_rate=8000, length=8000
    )

    task = FrameRegressionTask(regressor=LinearBenchmarkRegressor())
    task.validate_dataset(wrapped)

    batch = [wrapped[0], wrapped[1]]
    batched_audio, metadatas = collate_audio_batch(batch)
    assert batched_audio.shape == (2, 1, 8000)
    assert len(metadatas) == 2
