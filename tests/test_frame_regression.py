"""Tests for FrameRegressionTask."""

import pytest
import torch
from torch.utils.data import Dataset

from audio_benchmark import (
    AudioBenchmark,
    DummyAudioDataset,
    DummyAudioModel,
    KNNBenchmarkRegressor,
    LinearBenchmarkRegressor,
    MLPBenchmarkRegressor,
)
from audio_benchmark.exceptions import InvalidDatasetError
from audio_benchmark.tasks.frame_regression import FrameRegressionTask


class MismatchedFrameRegDataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, idx):
        return {
            "waveform": torch.randn(1, 100),
            "sample_rate": 16000,
            "metadata": {"frame_targets": [1.0] * 40},
        }


def test_frame_regression_with_regressors():
    task_linear = FrameRegressionTask(regressor=LinearBenchmarkRegressor())
    task_knn = FrameRegressionTask(regressor=KNNBenchmarkRegressor())
    task_mlp = FrameRegressionTask(
        regressor=MLPBenchmarkRegressor(epochs=1, device="cpu"),
    )

    train_reps = torch.randn(6, 4, 8)
    train_meta = [
        {"frame_targets": [float(i * 2 + j * 0.1) for j in range(16)]} for i in range(6)
    ]
    test_reps = torch.randn(2, 4, 8)
    test_meta = [
        {"frame_targets": [float(i * 2 + j * 0.1) for j in range(16)]} for i in range(2)
    ]

    info = task_linear.examine_data(train_meta, test_meta)
    assert info["train_clips"] == 6
    assert info["test_clips"] == 2
    assert info["frame_length"] == 16
    assert info["total_frames"] == 8 * 16
    assert info["train_frames"] == 6 * 16
    assert info["test_frames"] == 2 * 16
    assert info["baseline_strategy"] == "mean_predictor"
    assert info["baseline_metric"] == "r2"
    assert "baseline_value" in info

    for task in (task_linear, task_knn, task_mlp):
        res = task.evaluate(train_reps, train_meta, test_reps, test_meta)
        assert "train_clips" not in res
        assert "test_clips" not in res
        assert "train_frames" not in res
        assert "test_frames" not in res
        assert "T_orig" not in res
        assert "T_downsampled" not in res
        assert "mse" in res
        assert "r2" in res
        assert "training_time" in res
        assert res["training_time"] >= 0.0


def test_frame_regression_dataset_validation():
    task = FrameRegressionTask(
        regressor=LinearBenchmarkRegressor(), target_key="frame_targets"
    )
    # Arbitrary length 1D sequence (40 elements for 100 audio samples) is now accepted
    valid_seq_ds = MismatchedFrameRegDataset()
    task.validate_dataset(valid_seq_ds)

    # Valid dictionary targets dataset
    class DictFrameRegDataset(Dataset):
        def __len__(self):
            return 2

        def __getitem__(self, idx):
            return {
                "waveform": torch.randn(1, 100),
                "sample_rate": 16000,
                "metadata": {"frame_targets": {0: 1.5, 50: 3.2}},
            }

    task.validate_dataset(DictFrameRegDataset())

    # Invalid case: out-of-bounds timestamp
    class OutOfBoundsFrameRegDataset(Dataset):
        def __len__(self):
            return 2

        def __getitem__(self, idx):
            return {
                "waveform": torch.randn(1, 100),
                "sample_rate": 16000,
                "metadata": {"frame_targets": {100: 1.0}},
            }

    with pytest.raises(InvalidDatasetError, match="exceeds audio length"):
        task.validate_dataset(OutOfBoundsFrameRegDataset())

    # Invalid case: empty sequence
    class EmptyFrameRegDataset(Dataset):
        def __len__(self):
            return 2

        def __getitem__(self, idx):
            return {
                "waveform": torch.randn(1, 100),
                "sample_rate": 16000,
                "metadata": {"frame_targets": []},
            }

    with pytest.raises(InvalidDatasetError, match="must not be empty"):
        task.validate_dataset(EmptyFrameRegDataset())


def test_frame_regression_variable_length_and_dictionary():
    task_mean = FrameRegressionTask(
        regressor=LinearBenchmarkRegressor(), pooling="mean"
    )
    task_max = FrameRegressionTask(regressor=LinearBenchmarkRegressor(), pooling="max")

    train_reps = torch.randn(4, 6, 8)
    train_meta = [
        {"frame_targets": [1.0, 2.0, 3.0], "audio_length": 60},  # K=3
        {"frame_targets": [0.5, 1.5, 2.5, 3.5, 4.5], "audio_length": 60},  # K=5
        {"frame_targets": {0: 1.2, 30: 2.4}, "audio_length": 60},  # dict K=2
        {
            "frame_targets": [float(i * 0.1) for i in range(12)],
            "audio_length": 60,
        },  # K=12
    ]
    test_reps = torch.randn(2, 6, 8)
    test_meta = [
        {"frame_targets": {10: 0.8, 40: 1.6}, "audio_length": 60},  # dict K=2
        {"frame_targets": [0.1, 0.2, 0.3, 0.4], "audio_length": 60},  # K=4
    ]

    for task in (task_mean, task_max):
        res = task.evaluate(train_reps, train_meta, test_reps, test_meta)
        assert "mse" in res
        assert "r2" in res
        assert "training_time" in res

    info = task_mean.examine_data(train_meta, test_meta)
    assert info["train_clips"] == 4
    assert info["test_clips"] == 2
    assert info["train_frames"] == 22
    assert info["test_frames"] == 6
    assert info["total_frames"] == 28


def test_frame_regression_invalid_pooling():
    with pytest.raises(ValueError, match="Unknown pooling"):
        FrameRegressionTask(regressor=LinearBenchmarkRegressor(), pooling="unsupported")


def test_benchmark_runner_with_frame_regression(tmp_path):
    train_dataset = DummyAudioDataset(num_samples=8, audio_length=320, num_classes=2)
    test_dataset = DummyAudioDataset(num_samples=4, audio_length=320, num_classes=2)
    model = DummyAudioModel(representation_dim=16)
    task = FrameRegressionTask(regressor=LinearBenchmarkRegressor())

    benchmark = AudioBenchmark(
        dataset=train_dataset,
        testset=test_dataset,
        task=task,
        dataset_name="dummy_audio",
        cache_dir=tmp_path,
        enable_cache=True,
    )

    result = benchmark.run(
        model=model,
        model_name="dummy_frame_reg_model",
        batch_size=4,
        device="cpu",
    )

    assert result.task_name == "frame_regression"
    assert "mse" in result.metrics
    assert "T_orig" not in result.metrics
    info = task.examine_data(train_dataset, test_dataset)
    assert info["frame_length"] == 320
    assert info["train_clips"] == 8
    assert info["test_clips"] == 4


def test_frame_regression_string_regressor_raises():
    with pytest.raises(TypeError, match="Passing strings as models is not supported"):
        FrameRegressionTask(regressor="linear")


def test_frame_regression_none_or_missing_regressor_raises():
    with pytest.raises(TypeError, match="Passing None as a model is not allowed"):
        FrameRegressionTask(regressor=None)
    with pytest.raises(TypeError):
        FrameRegressionTask()
