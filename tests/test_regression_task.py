"""Tests for RegressionTask."""

import pytest
import torch

from audio_benchmark import AudioBenchmark, DummyAudioDataset, DummyAudioModel
from audio_benchmark.exceptions import InvalidDatasetError, TaskError
from audio_benchmark.tasks.regression import RegressionTask


def test_regression_task_with_different_regressors():
    task_linear = RegressionTask(regressor="linear", test_size=0.25, random_state=42)
    task_knn = RegressionTask(regressor="knn", test_size=0.25, random_state=42)
    task_mlp = RegressionTask(
        regressor="mlp",
        test_size=0.25,
        random_state=42,
        regressor_kwargs={"epochs": 1, "device": "cpu"},
    )

    reps = torch.randn(20, 4, 16)
    meta = [{"target": float(i * 1.5 + 2.0)} for i in range(20)]

    info = task_linear.examine_data(meta)
    assert info["train_samples"] == 15
    assert info["test_samples"] == 5
    assert info["baseline_strategy"] == "mean_predictor"
    assert info["baseline_metric"] == "r2"
    assert "baseline_value" in info
    assert "target_min" in info
    assert "target_max" in info
    assert "target_mean" in info

    for task in (task_linear, task_knn, task_mlp):
        res = task.evaluate(reps, meta)
        assert "train_samples" not in res
        assert "test_samples" not in res
        assert "target_key" not in res
        assert "mse" in res
        assert "rmse" in res
        assert "mae" in res
        assert "r2" in res
        assert "pearson_r" in res
        assert "training_time" in res
        assert res["training_time"] >= 0.0


def test_regression_task_with_transformer():
    task_tf = RegressionTask(
        regressor="transformer",
        test_size=0.25,
        random_state=42,
        regressor_kwargs={
            "d_model": 16,
            "nhead": 2,
            "num_layers": 1,
            "dim_feedforward": 32,
            "epochs": 1,
            "device": "cpu",
        },
    )

    reps = torch.randn(20, 4, 16)
    meta = [{"target": float(i * 1.5 + 2.0)} for i in range(20)]

    res = task_tf.evaluate(reps, meta)
    assert res["regressor"] == "transformer"
    assert res["pooling"] == "none (transformer)"
    assert "train_samples" not in res
    assert "mse" in res
    assert "training_time" in res
    assert res["training_time"] > 0.0


def test_regression_task_pooling_modes():
    reps = torch.randn(12, 6, 8)
    meta = [{"target": float(i)} for i in range(12)]

    for pool in ["mean", "max", "first", "last"]:
        task = RegressionTask(regressor="linear", pooling=pool)
        res = task.evaluate(reps, meta)
        assert res["pooling"] == pool
        assert "mse" in res


def test_regression_task_with_split_key():
    reps = torch.randn(15, 4, 8)
    meta = []
    for i in range(15):
        split = "train" if i < 10 else "test"
        meta.append({"target": float(i * 0.5), "split": split})

    task = RegressionTask(regressor="linear", split_key="split")
    info = task.examine_data(meta)
    assert info["train_samples"] == 10
    assert info["test_samples"] == 5

    res = task.evaluate(reps, meta)
    assert "train_samples" not in res


def test_regression_task_missing_target_raises():
    task = RegressionTask(target_key="pitch")
    reps = torch.randn(6, 4, 8)
    meta = [{"wrong_target": 1.0} for _ in range(6)]

    with pytest.raises(InvalidDatasetError, match="does not contain target key"):
        task.evaluate(reps, meta)


def test_regression_task_invalid_target_value_raises():
    task = RegressionTask(target_key="target")
    reps = torch.randn(6, 4, 8)
    meta = [{"target": "not_a_number"} for _ in range(6)]

    with pytest.raises(TaskError, match="cannot be converted to float"):
        task.evaluate(reps, meta)


def test_benchmark_runner_with_regression_task(tmp_path):
    benchmark = AudioBenchmark(cache_dir=tmp_path, enable_cache=True)
    dataset = DummyAudioDataset(num_samples=10, audio_length=320, num_classes=2)
    model = DummyAudioModel(representation_dim=16)
    task = RegressionTask(regressor="linear")

    result = benchmark.run(
        model=model,
        model_name="dummy_reg_model",
        dataset=dataset,
        task=task,
        batch_size=4,
        device="cpu",
    )

    assert result.task_name == "regression"
    assert "mse" in result.metrics
    assert "r2" in result.metrics
    assert result.metrics["regressor"] == "linear"
