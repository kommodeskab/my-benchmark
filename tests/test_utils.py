"""Tests for utility functions and model training state preservation."""

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from audio_benchmark.extractor import extract_representations
from audio_benchmark.utils import resolve_device, to_numpy


class SimpleAudioDataset(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, idx):
        return {
            "waveform": torch.randn(1, 160),
            "sample_rate": 16000,
            "metadata": {"label": idx},
        }


class SimpleAudioModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(1, 8, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x).transpose(1, 2)


def test_to_numpy():
    # PyTorch Tensor
    t = torch.tensor([1.0, 2.0, 3.0])
    arr = to_numpy(t)
    assert isinstance(arr, np.ndarray)
    np.testing.assert_array_equal(arr, np.array([1.0, 2.0, 3.0]))

    # Existing NumPy array
    arr_in = np.array([4, 5, 6])
    arr_out = to_numpy(arr_in)
    assert arr_out is arr_in

    # Python list
    arr_list = to_numpy([7, 8])
    assert isinstance(arr_list, np.ndarray)
    np.testing.assert_array_equal(arr_list, np.array([7, 8]))


def test_resolve_device():
    # 'auto' or None
    dev_auto = resolve_device("auto")
    expected = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert dev_auto == expected

    dev_none = resolve_device(None)
    assert dev_none == expected

    # Explicit string
    dev_cpu = resolve_device("cpu")
    assert dev_cpu == torch.device("cpu")

    # torch.device instance
    d = torch.device("cpu")
    assert resolve_device(d) == d


def test_extractor_preserves_model_training_mode():
    model = SimpleAudioModel()
    dataset = SimpleAudioDataset()

    # 1. Model was originally in training mode
    model.train()
    assert model.training is True
    _reps, _meta = extract_representations(model, dataset, batch_size=2, device="cpu")
    # Must restore to training mode
    assert model.training is True

    # 2. Model was originally in eval mode
    model.eval()
    assert model.training is False
    _reps, _meta = extract_representations(model, dataset, batch_size=2, device="cpu")
    # Must remain in eval mode
    assert model.training is False
