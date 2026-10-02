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
    task_linear = FrameRegressionTask(
        regressor=LinearBenchmarkRegressor(), test_size=0.25, random_state=42
    )
    task_knn = FrameRegressionTask(
        regressor=KNNBenchmarkRegressor(), test_size=0.25, random_state=42
    )
    task_mlp = FrameRegressionTask(
        regressor=MLPBenchmarkRegressor(epochs=1, device="cpu"),
        test_size=0.25,
        random_state=42,
    )

    reps = torch.randn(8, 4, 8)
    meta = [
        {"frame_targets": [float(i * 2 + j * 0.1) for j in range(16)]} for i in range(8)
    ]

    info = task_linear.examine_data(meta)
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
        res = task.evaluate(reps, meta)
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
    bad_ds = MismatchedFrameRegDataset()

    with pytest.raises(InvalidDatasetError, match="Length mismatch in 'frame_targets'"):
        task.validate_dataset(bad_ds)


def test_frame_regression_with_split_key():
    task = FrameRegressionTask(regressor=LinearBenchmarkRegressor(), split_key="split")
    reps = torch.randn(6, 3, 8)
    meta = []
    for i in range(6):
        split = "train" if i < 4 else "test"
        meta.append({"frame_targets": [float(j) for j in range(12)], "split": split})

    info = task.examine_data(meta)
    assert info["train_clips"] == 4
    assert info["test_clips"] == 2

    res = task.evaluate(reps, meta)
    assert "train_clips" not in res
    assert "train_frames" not in res


def test_benchmark_runner_with_frame_regression(tmp_path):
    benchmark = AudioBenchmark(cache_dir=tmp_path, enable_cache=True)
    dataset = DummyAudioDataset(num_samples=8, audio_length=320, num_classes=2)
    model = DummyAudioModel(representation_dim=16)
    task = FrameRegressionTask(regressor=LinearBenchmarkRegressor())

    result = benchmark.run(
        model=model,
        model_name="dummy_frame_reg_model",
        dataset=dataset,
        task=task,
        batch_size=4,
        device="cpu",
    )

    assert result.task_name == "frame_regression"
    assert "mse" in result.metrics
    assert "T_orig" not in result.metrics
    assert result.extra_info["frame_length"] == 320
    assert result.extra_info["train_clips"] == 6


def test_frame_regression_string_regressor_raises():
    with pytest.raises(TypeError, match="Passing strings as models is not supported"):
        FrameRegressionTask(regressor="linear")


def test_frame_regression_none_or_missing_regressor_raises():
    with pytest.raises(TypeError, match="Passing None as a model is not allowed"):
        FrameRegressionTask(regressor=None)
    with pytest.raises(TypeError):
        FrameRegressionTask()
