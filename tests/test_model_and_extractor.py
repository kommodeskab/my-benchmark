"""Tests for representation model validation and feature extraction."""

import pytest
import torch
from torch import nn

from audio_benchmark.dummy import DummyAudioDataset, DummyAudioModel
from audio_benchmark.exceptions import InvalidModelError, InvalidModelOutputError
from audio_benchmark.extractor import extract_representations


class Invalid2DModel(nn.Module):
    """Outputs (B, C) instead of required (B, T', C')."""

    def forward(self, x):
        return torch.randn(x.shape[0], 64)


class InvalidBatchModel(nn.Module):
    """Returns wrong batch dimension."""

    def forward(self, x):
        return torch.randn(1, 10, 64)


def test_dummy_model_shapes():
    model = DummyAudioModel(in_channels=1, representation_dim=32)
    dummy_input = torch.randn(2, 1, 320)
    output = model(dummy_input)

    assert output.ndim == 3
    assert output.shape[0] == 2
    assert output.shape[2] == 32
    assert output.shape[1] > 0


def test_extract_representations_success():
    dataset = DummyAudioDataset(num_samples=6, audio_length=320, num_classes=2)
    model = DummyAudioModel(representation_dim=16)

    representations, metadata = extract_representations(
        model=model,
        dataset=dataset,
        batch_size=3,
        device="cpu",
    )

    assert isinstance(representations, torch.Tensor)
    assert representations.shape[0] == 6
    assert representations.shape[2] == 16
    assert len(metadata) == 6
    assert metadata[0]["label"] in [0, 1]


def test_extract_representations_rejects_non_module():
    dataset = DummyAudioDataset(num_samples=2, audio_length=320)
    with pytest.raises(InvalidModelError, match="instance of torch.nn.Module"):
        extract_representations(model="not_a_model", dataset=dataset)


def test_extract_representations_rejects_invalid_output_dimensions():
    dataset = DummyAudioDataset(num_samples=4, audio_length=320)
    bad_model = Invalid2DModel()
    with pytest.raises(
        InvalidModelOutputError,
        match="must be 3-dimensional with shape \\(B, T', C'\\)",
    ):
        extract_representations(model=bad_model, dataset=dataset, batch_size=2)


def test_extract_representations_rejects_batch_dimension_mismatch():
    dataset = DummyAudioDataset(num_samples=4, audio_length=320)
    bad_model = InvalidBatchModel()
    with pytest.raises(InvalidModelOutputError, match="batch dimension"):
        extract_representations(model=bad_model, dataset=dataset, batch_size=2)


def test_extract_representations_streaming_mmap(tmp_path):
    dataset = DummyAudioDataset(num_samples=8, audio_length=320, num_classes=2)
    model = DummyAudioModel(representation_dim=16)
    mmap_file = tmp_path / "streaming_reps.npy"

    reps, metadata = extract_representations(
        model=model,
        dataset=dataset,
        batch_size=2,
        device="cpu",
        mmap_path=mmap_file,
    )

    assert mmap_file.is_file()
    assert isinstance(reps, torch.Tensor)
    assert reps.shape[0] == 8
    assert reps.shape[2] == 16
    assert len(metadata) == 8


def test_extract_representations_empty_dataset():
    empty_dataset = DummyAudioDataset(num_samples=0, audio_length=320)
    model = DummyAudioModel(representation_dim=16)

    from audio_benchmark.exceptions import InvalidDatasetError

    with pytest.raises(InvalidDatasetError, match="Dataset is empty"):
        extract_representations(
            model=model,
            dataset=empty_dataset,
            batch_size=2,
            device="cpu",
        )


def test_extract_representations_progress_bar(monkeypatch):
    calls = []

    def mock_tqdm(iterable, *args, **kwargs):
        calls.append((args, kwargs))
        return iterable

    monkeypatch.setattr("audio_benchmark.extractor.tqdm", mock_tqdm)

    dataset = DummyAudioDataset(num_samples=4, audio_length=320, num_classes=2)
    model = DummyAudioModel(representation_dim=16)

    # 1. Default show_progress=True
    extract_representations(model=model, dataset=dataset, batch_size=2, device="cpu")
    assert len(calls) == 1
    assert calls[0][1].get("desc") == "Extracting features"
    assert calls[0][1].get("disable") is False

    # 2. Explicit show_progress=False
    extract_representations(
        model=model, dataset=dataset, batch_size=2, device="cpu", show_progress=False
    )
    assert len(calls) == 2
    assert calls[1][1].get("disable") is True


def test_extract_representations_variable_length_batches_expands_memmap(tmp_path):
    """Verify memmap grows when a later batch has longer audio than the first batch."""

    class _VarLenDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.lengths = [160, 240, 480, 640]

        def __len__(self):
            return len(self.lengths)

        def __getitem__(self, idx):
            return {
                "waveform": torch.randn(1, self.lengths[idx]),
                "sample_rate": 16000,
                "metadata": {"tokens": ["a", "b"]},
            }

    dataset = _VarLenDataset()
    model = DummyAudioModel(representation_dim=16)
    mmap_file = tmp_path / "var_len_reps.npy"

    reps, metadata = extract_representations(
        model=model,
        dataset=dataset,
        batch_size=2,
        device="cpu",
        mmap_path=mmap_file,
        show_progress=False,
    )

    expected_max_t = model(torch.randn(1, 1, 640)).shape[1]
    assert reps.shape == (4, expected_max_t, 16)
    assert all("rep_length" in m for m in metadata)
    assert metadata[0]["rep_length"] < metadata[1]["rep_length"]
    assert metadata[2]["rep_length"] < metadata[3]["rep_length"]
    assert metadata[3]["rep_length"] == expected_max_t
