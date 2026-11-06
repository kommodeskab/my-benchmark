"""Tests for multi-representation list support across models, extractor, cache, and benchmark."""

import pytest
import torch
from torch import nn

from audio_benchmark.benchmark import AudioBenchmark, BenchmarkResult
from audio_benchmark.classifiers import KNNBenchmarkClassifier
from audio_benchmark.dummy import DummyAudioDataset, DummyAudioModel
from audio_benchmark.exceptions import InvalidModelOutputError
from audio_benchmark.extractor import extract_representations
from audio_benchmark.tasks.global_classification import GlobalClassificationTask


class SingleRepModel(nn.Module):
    """Model returning a list containing a single representation tensor of shape (B, C, T)."""

    def __init__(self, representation_dim: int = 24):
        super().__init__()
        self.conv = nn.Conv1d(1, representation_dim, kernel_size=8, stride=4, padding=2)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return [self.conv(x)]


class MultiRepModel(nn.Module):
    """Model returning 3 representations with distinct temporal and feature dimensions (B, C_i, T_i)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=8, stride=2, padding=3)  # Higher T
        self.conv2 = nn.Conv1d(16, 32, kernel_size=4, stride=2, padding=1)  # Mid T
        self.conv3 = nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1)  # Lower T

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        f1 = torch.relu(self.conv1(x))
        f2 = torch.relu(self.conv2(f1))
        f3 = torch.relu(self.conv3(f2))
        return [f1, f2, f3]


def test_single_representation_list_extraction(tmp_path):
    dataset = DummyAudioDataset(num_samples=6, audio_length=320, num_classes=2)
    model = SingleRepModel(representation_dim=24)

    reps, meta = extract_representations(
        model=model,
        dataset=dataset,
        batch_size=3,
        device="cpu",
    )

    assert isinstance(reps, list)
    assert len(reps) == 1
    assert reps[0].shape[0] == 6
    assert reps[0].shape[2] == 24
    assert len(meta) == 6
    assert len(meta[0]["rep_lengths"]) == 1


def test_multi_representation_list_variable_dims_extraction(tmp_path):
    dataset = DummyAudioDataset(num_samples=8, audio_length=320, num_classes=2)
    model = MultiRepModel()
    mmap_path = tmp_path / "multi_reps.npy"

    reps, meta = extract_representations(
        model=model,
        dataset=dataset,
        batch_size=4,
        device="cpu",
        mmap_path=mmap_path,
    )

    assert len(reps) == 3
    # Check that T and D vary across representations in the list
    t_dims = [r.shape[1] for r in reps]
    d_dims = [r.shape[2] for r in reps]
    assert t_dims[0] > t_dims[1] > t_dims[2]
    assert d_dims == [16, 32, 64]

    # Verify each memmap file was saved
    assert (tmp_path / "multi_reps_0.npy").is_file()
    assert (tmp_path / "multi_reps_1.npy").is_file()
    assert (tmp_path / "multi_reps_2.npy").is_file()

    # Verify metadata rep_lengths contains all 3 temporal lengths
    assert len(meta[0]["rep_lengths"]) == 3
    assert meta[0]["rep_lengths"] == t_dims


def test_benchmark_run_returns_list_of_results_for_all_representations(tmp_path):
    train_dataset = DummyAudioDataset(num_samples=12, audio_length=320, num_classes=3)
    test_dataset = DummyAudioDataset(num_samples=6, audio_length=320, num_classes=3)
    model = MultiRepModel()
    task = GlobalClassificationTask(classifier=KNNBenchmarkClassifier())

    benchmark = AudioBenchmark(
        dataset=train_dataset,
        testset=test_dataset,
        task=task,
        dataset_name="multi_rep_bench",
        cache_dir=tmp_path,
        enable_cache=True,
    )

    results = benchmark.run(
        model=model,
        model_name="multi_rep_test",
        batch_size=4,
        device="cpu",
    )

    assert isinstance(results, list)
    assert len(results) == 3
    for i, res in enumerate(results):
        assert isinstance(res, BenchmarkResult)
        assert res.representation_index == i
        assert res.model_name == "multi_rep_test"
        assert not res.cached
        assert "accuracy" in res.metrics

    # Check that shapes reflect varying (T_i, D_i)
    assert results[0].representations_shape[2] == 16
    assert results[1].representations_shape[2] == 32
    assert results[2].representations_shape[2] == 64

    # Second run hits cache for all representations
    cached_results = benchmark.run(
        model=model,
        model_name="multi_rep_test",
        batch_size=4,
        device="cpu",
    )
    assert len(cached_results) == 3
    assert all(r.cached for r in cached_results)
    for i in range(3):
        assert cached_results[i].metrics["accuracy"] == results[i].metrics["accuracy"]


def test_benchmark_inconsistent_train_test_reps_raises(monkeypatch, tmp_path):
    train_dataset = DummyAudioDataset(num_samples=6, audio_length=320, num_classes=2)
    test_dataset = DummyAudioDataset(num_samples=4, audio_length=320, num_classes=2)
    task = GlobalClassificationTask(classifier=KNNBenchmarkClassifier())

    benchmark = AudioBenchmark(
        dataset=train_dataset,
        testset=test_dataset,
        task=task,
        dataset_name="inconsistent_bench",
        cache_dir=tmp_path,
        enable_cache=False,
    )

    call_count = 0

    def mock_extract(model, dataset, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return [torch.randn(len(dataset), 10, 16)], [
                {"label": 0} for _ in range(len(dataset))
            ]
        else:
            return [
                torch.randn(len(dataset), 10, 16),
                torch.randn(len(dataset), 5, 32),
            ], [{"label": 0} for _ in range(len(dataset))]

    monkeypatch.setattr(
        "audio_benchmark.benchmark.extract_representations", mock_extract
    )

    model = DummyAudioModel()
    with pytest.raises(
        InvalidModelOutputError,
        match="Number of train representations \\(1\\) does not match test representations \\(2\\)",
    ):
        benchmark.run(model=model, model_name="dummy_inconsistent")
