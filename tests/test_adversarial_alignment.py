"""Adversarial audit tests for arbitrary-length and dictionary-mapped temporal alignment."""

from typing import Any

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from audio_benchmark.benchmark import AudioBenchmark
from audio_benchmark.classifiers import KNNBenchmarkClassifier
from audio_benchmark.dummy import DummyAudioModel
from audio_benchmark.exceptions import InvalidDatasetError
from audio_benchmark.regressors import LinearBenchmarkRegressor
from audio_benchmark.tasks.downsampling import (
    align_features_to_targets,
    parse_frame_targets,
)
from audio_benchmark.tasks.frame_classification import FrameClassificationTask
from audio_benchmark.tasks.frame_regression import FrameRegressionTask


def test_align_features_mathematical_downsampling_exact_formulas():
    # T' = 6 representation frames downsampled to target_len = 3 via linear interpolation
    reps = torch.tensor(
        [
            [1.0, 10.0, 100.0],
            [3.0, 20.0, 200.0],
            [5.0, 30.0, 300.0],
            [7.0, 40.0, 400.0],
            [11.0, 50.0, 500.0],
            [13.0, 60.0, 600.0],
        ],
        dtype=torch.float32,
    )
    out = align_features_to_targets(reps, target_len=3)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 3)
    assert torch.allclose(out[0], torch.tensor([2.0, 15.0, 150.0]))
    assert torch.allclose(out[1], torch.tensor([6.0, 35.0, 350.0]))
    assert torch.allclose(out[2], torch.tensor([12.0, 55.0, 550.0]))


def test_align_features_mathematical_upsampling_exact_formulas():
    # T' = 3 representation frames upsampled to target_len = 6 via linear interpolation
    reps = torch.tensor(
        [
            [1.0, 2.0],
            [10.0, 20.0],
            [100.0, 200.0],
        ],
        dtype=torch.float32,
    )
    aligned = align_features_to_targets(reps, target_len=6)
    assert isinstance(aligned, torch.Tensor)
    assert aligned.shape == (6, 2)
    assert torch.allclose(aligned[0], torch.tensor([1.0, 2.0]))
    assert torch.allclose(aligned[1], torch.tensor([3.25, 6.5]))
    assert torch.allclose(aligned[2], torch.tensor([7.75, 15.5]))
    assert torch.allclose(aligned[3], torch.tensor([32.5, 65.0]))
    assert torch.allclose(aligned[4], torch.tensor([77.5, 155.0]))
    assert torch.allclose(aligned[5], torch.tensor([100.0, 200.0]))


def test_align_features_boundaries_tprime_1_and_k_1():
    # Case 1: T' = 1, target_len = 1 (identity single frame)
    reps_single = torch.tensor([[3.14, 2.71]], dtype=torch.float32)
    out1 = align_features_to_targets(reps_single, target_len=1)
    assert isinstance(out1, torch.Tensor)
    assert out1.shape == (1, 2)
    assert torch.allclose(out1[0], torch.tensor([3.14, 2.71]))

    # Case 2: T' = 1, target_len = 4 (upsampling single rep frame)
    out2 = align_features_to_targets(reps_single, target_len=4)
    assert isinstance(out2, torch.Tensor)
    assert out2.shape == (4, 2)
    for row in out2:
        assert torch.allclose(row, torch.tensor([3.14, 2.71]))

    # Case 3: T' = 4, target_len = 1 (downsampling 4 frames to 1)
    reps_four = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)
    out_down = align_features_to_targets(reps_four, target_len=1)
    assert isinstance(out_down, torch.Tensor)
    assert out_down.shape == (1, 1)
    assert torch.allclose(out_down[0], torch.tensor([2.5]))


def test_align_features_arbitrary_scale_downsampling():
    reps = torch.tensor(
        [
            [10.0],
            [20.0],
            [30.0],
            [40.0],
            [100.0],
            [200.0],
            [1000.0],
            [2000.0],
        ],
        dtype=torch.float32,
    )
    aligned = align_features_to_targets(reps, target_len=5)
    assert isinstance(aligned, torch.Tensor)
    assert aligned.shape == (5, 1)


def test_align_features_dense_length():
    reps = torch.ones((5, 8), dtype=torch.float32)
    out = align_features_to_targets(reps, target_len=50)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (50, 8)


def test_parse_frame_targets_stringified_and_dense_keys():
    # Valid list
    vals_list = parse_frame_targets([10, 20, 30], key_name="test_labels")
    assert vals_list == [10, 20, 30]

    # Valid flat torch.Tensor
    vals_tensor = parse_frame_targets(
        torch.tensor([10, 20, 30]), key_name="test_labels"
    )
    assert vals_tensor == [10, 20, 30]

    # Dictionary targets are rejected
    with pytest.raises(
        InvalidDatasetError, match="must be a flat torch.Tensor or list"
    ):
        parse_frame_targets({0: 10, 50: 20, 99: 30})

    with pytest.raises(
        InvalidDatasetError, match="must be a flat torch.Tensor or list"
    ):
        parse_frame_targets({"10.5": 1})

    # Empty dictionary raises InvalidDatasetError
    with pytest.raises(
        InvalidDatasetError, match="must be a flat torch.Tensor or list"
    ):
        parse_frame_targets({})

    # Empty sequence raises InvalidDatasetError
    with pytest.raises(InvalidDatasetError, match="must not be empty"):
        parse_frame_targets([])

    # Empty tensor raises InvalidDatasetError
    with pytest.raises(InvalidDatasetError, match="must not be empty"):
        parse_frame_targets(torch.empty(0))

    # Multi-dimensional tensor raises InvalidDatasetError
    with pytest.raises(InvalidDatasetError, match="must be 1D"):
        parse_frame_targets(torch.ones((2, 2)))

    # Non-supported sequence containers (ndarray, tuple) raise InvalidDatasetError
    with pytest.raises(
        InvalidDatasetError, match="must be a flat torch.Tensor or list"
    ):
        parse_frame_targets(np.array([]))

    with pytest.raises(
        InvalidDatasetError, match="must be a flat torch.Tensor or list"
    ):
        parse_frame_targets((1, 2))


def test_strict_validation_negative_out_of_bounds_and_invalid_pooling():
    reps = torch.ones((4, 2), dtype=torch.float32)

    # Must be a torch.Tensor
    with pytest.raises(TypeError, match="must be a torch.Tensor"):
        align_features_to_targets(np.ones((4, 2)), target_len=2)

    # Invalid target_len
    with pytest.raises(ValueError, match="target_len must be a positive integer"):
        align_features_to_targets(reps, target_len=0)

    with pytest.raises(ValueError, match="target_len must be a positive integer"):
        align_features_to_targets(reps, target_len=-5)

    # Invalid representation shapes
    with pytest.raises(ValueError, match="must have shape"):
        align_features_to_targets(torch.ones(4), target_len=2)

    with pytest.raises(ValueError, match="temporal length 0"):
        align_features_to_targets(torch.empty((0, 2)), target_len=2)


def test_mismatched_clip_durations_evaluation():
    # 3 clips with different audio durations and target counts:
    # Clip 0: audio_length 60, T' = 6, K = 3 (downsampling 6 -> 3)
    # Clip 1: audio_length 120, T' = 4, K = 8 (upsampling 4 -> 8)
    # Clip 2: audio_length 80, T' = 4, flat tensor K = 2 (downsampling 4 -> 2)
    task_clf = FrameClassificationTask(classifier=KNNBenchmarkClassifier())
    task_reg = FrameRegressionTask(regressor=LinearBenchmarkRegressor())

    train_reps = [
        torch.randn(6, 4),
        torch.randn(4, 4),
        torch.randn(4, 4),
    ]
    train_meta_clf = [
        {"frame_labels": [0, 1, 0], "audio_length": 60},
        {"frame_labels": [1, 0, 1, 0, 1, 0, 1, 0], "audio_length": 120},
        {"frame_labels": torch.tensor([0, 1]), "audio_length": 80},
    ]
    test_reps = [torch.randn(4, 4)]
    test_meta_clf = [{"frame_labels": [0, 1, 0, 1], "audio_length": 80}]

    res_clf = task_clf.evaluate(train_reps, train_meta_clf, test_reps, test_meta_clf)
    assert 0.0 <= res_clf["accuracy"] <= 1.0

    train_meta_reg = [
        {"frame_targets": [1.0, 2.0, 3.0], "audio_length": 60},
        {"frame_targets": [float(i) for i in range(8)], "audio_length": 120},
        {"frame_targets": torch.tensor([1.5, 3.5]), "audio_length": 80},
    ]
    test_meta_reg = [{"frame_targets": [0.5, 1.5, 2.5, 3.5], "audio_length": 80}]

    res_reg = task_reg.evaluate(train_reps, train_meta_reg, test_reps, test_meta_reg)
    assert "mse" in res_reg
    assert "r2" in res_reg


def test_dictionary_target_rejected_raises():
    task_clf = FrameClassificationTask(classifier=KNNBenchmarkClassifier())
    task_reg = FrameRegressionTask(regressor=LinearBenchmarkRegressor())

    reps = torch.randn(1, 4, 4)
    meta_clf = [{"frame_labels": {10: 0, 20: 1}}]
    meta_reg = [{"frame_targets": {10: 1.0, 20: 2.0}}]

    with pytest.raises(
        InvalidDatasetError, match="must be a flat torch.Tensor or list"
    ):
        task_clf.evaluate(reps, meta_clf, reps, meta_clf)

    with pytest.raises(
        InvalidDatasetError, match="must be a flat torch.Tensor or list"
    ):
        task_reg.evaluate(reps, meta_reg, reps, meta_reg)


def test_end_to_end_benchmark_frame_classification_sequence_and_tensor(
    tmp_path,
):
    class CustomArbitraryFrameClfDataset(Dataset):
        def __init__(self, num_samples: int, is_tensor: bool):
            self.num_samples = num_samples
            self.is_tensor = is_tensor

        def __len__(self) -> int:
            return self.num_samples

        def __getitem__(self, idx: int) -> dict[str, Any]:
            if self.is_tensor:
                labels = torch.tensor([0, 1, 0])
            else:
                labels = [0, 1, 0, 1, 0, 1, 0]
            return {
                "waveform": torch.randn(1, 320),
                "sample_rate": 16000,
                "metadata": {"frame_labels": labels},
            }

    model = DummyAudioModel(representation_dim=16)
    task = FrameClassificationTask(classifier=KNNBenchmarkClassifier())

    # 1. Sequence targets
    bench_seq = AudioBenchmark(
        dataset=CustomArbitraryFrameClfDataset(num_samples=6, is_tensor=False),
        testset=CustomArbitraryFrameClfDataset(num_samples=4, is_tensor=False),
        task=task,
        dataset_name="custom_seq_clf",
        cache_dir=tmp_path / "clf_seq",
        enable_cache=True,
    )
    res_seq = bench_seq.run(
        model=model,
        model_name="dummy_clf_seq",
        batch_size=2,
        device="cpu",
    )
    assert res_seq.task_name == "frame_classification"
    assert "accuracy" in res_seq.metrics
    assert 0.0 <= res_seq.metrics["accuracy"] <= 1.0

    # 2. Tensor targets
    bench_tensor = AudioBenchmark(
        dataset=CustomArbitraryFrameClfDataset(num_samples=6, is_tensor=True),
        testset=CustomArbitraryFrameClfDataset(num_samples=4, is_tensor=True),
        task=task,
        dataset_name="custom_tensor_clf",
        cache_dir=tmp_path / "clf_tensor",
        enable_cache=True,
    )
    res_tensor = bench_tensor.run(
        model=model,
        model_name="dummy_clf_tensor",
        batch_size=2,
        device="cpu",
    )
    assert res_tensor.task_name == "frame_classification"
    assert "accuracy" in res_tensor.metrics
    assert 0.0 <= res_tensor.metrics["accuracy"] <= 1.0


def test_end_to_end_benchmark_frame_regression_sequence_and_tensor(tmp_path):
    class CustomArbitraryFrameRegDataset(Dataset):
        def __init__(self, num_samples: int, is_tensor: bool):
            self.num_samples = num_samples
            self.is_tensor = is_tensor

        def __len__(self) -> int:
            return self.num_samples

        def __getitem__(self, idx: int) -> dict[str, Any]:
            if self.is_tensor:
                targets = torch.tensor([1.2, 2.8, 4.1])
            else:
                targets = [1.0, 1.5, 2.0, 2.5, 3.0]
            return {
                "waveform": torch.randn(1, 320),
                "sample_rate": 16000,
                "metadata": {"frame_targets": targets},
            }

    model = DummyAudioModel(representation_dim=16)
    task = FrameRegressionTask(regressor=LinearBenchmarkRegressor())

    # 1. Sequence targets
    bench_seq = AudioBenchmark(
        dataset=CustomArbitraryFrameRegDataset(num_samples=6, is_tensor=False),
        testset=CustomArbitraryFrameRegDataset(num_samples=4, is_tensor=False),
        task=task,
        dataset_name="custom_seq_reg",
        cache_dir=tmp_path / "reg_seq",
        enable_cache=True,
    )
    res_seq = bench_seq.run(
        model=model,
        model_name="dummy_reg_seq",
        batch_size=2,
        device="cpu",
    )
    assert res_seq.task_name == "frame_regression"
    assert "mse" in res_seq.metrics
    assert "r2" in res_seq.metrics

    # 2. Tensor targets
    bench_tensor = AudioBenchmark(
        dataset=CustomArbitraryFrameRegDataset(num_samples=6, is_tensor=True),
        testset=CustomArbitraryFrameRegDataset(num_samples=4, is_tensor=True),
        task=task,
        dataset_name="custom_tensor_reg",
        cache_dir=tmp_path / "reg_tensor",
        enable_cache=True,
    )
    res_tensor = bench_tensor.run(
        model=model,
        model_name="dummy_reg_tensor",
        batch_size=2,
        device="cpu",
    )
    assert res_tensor.task_name == "frame_regression"
    assert "mse" in res_tensor.metrics
    assert "r2" in res_tensor.metrics
