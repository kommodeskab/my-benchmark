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
