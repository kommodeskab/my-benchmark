"""Tests for RegressionTask."""

import pytest
import torch

from audio_benchmark import (
    AudioBenchmark,
    DummyAudioDataset,
    DummyAudioModel,
    KNNBenchmarkRegressor,
    LinearBenchmarkRegressor,
    MLPBenchmarkRegressor,
    TransformerBenchmarkRegressor,
)
from audio_benchmark.exceptions import InvalidDatasetError, TaskError
from audio_benchmark.tasks.regression import RegressionTask


def test_regression_task_with_different_regressors():
    task_linear = RegressionTask(regressor=LinearBenchmarkRegressor())
    task_knn = RegressionTask(regressor=KNNBenchmarkRegressor())
    task_mlp = RegressionTask(
        regressor=MLPBenchmarkRegressor(epochs=1, device="cpu"),
    )

    train_reps = torch.randn(15, 4, 16)
    train_meta = [{"target": float(i * 1.5 + 2.0)} for i in range(15)]
    test_reps = torch.randn(5, 4, 16)
    test_meta = [{"target": float(i * 1.5 + 2.0)} for i in range(5)]

    info = task_linear.examine_data(train_meta, test_meta)
    assert info["train_samples"] == 15
    assert info["test_samples"] == 5
    assert info["baseline_strategy"] == "mean_predictor"
    assert info["baseline_metric"] == "r2"
    assert "baseline_value" in info
    assert "target_min" in info
    assert "target_max" in info
    assert "target_mean" in info

    for task in (task_linear, task_knn, task_mlp):
        res = task.evaluate(train_reps, train_meta, test_reps, test_meta)
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
        regressor=TransformerBenchmarkRegressor(
            d_model=16,
            nhead=2,
            num_layers=1,
            dim_feedforward=32,
            epochs=1,
            device="cpu",
        ),
    )

    train_reps = torch.randn(15, 4, 16)
    train_meta = [{"target": float(i * 1.5 + 2.0)} for i in range(15)]
    test_reps = torch.randn(5, 4, 16)
    test_meta = [{"target": float(i * 1.5 + 2.0)} for i in range(5)]

    res = task_tf.evaluate(train_reps, train_meta, test_reps, test_meta)
    assert res["regressor"] == "TransformerBenchmarkRegressor"
    assert res["pooling"] == "none (transformer)"
    assert "train_samples" not in res
    assert "mse" in res
    assert "training_time" in res
    assert res["training_time"] > 0.0


def test_regression_task_pooling_modes():
    train_reps = torch.randn(8, 6, 8)
    train_meta = [{"target": float(i)} for i in range(8)]
    test_reps = torch.randn(4, 6, 8)
    test_meta = [{"target": float(i)} for i in range(4)]

    for pool in ["mean", "max", "first", "last"]:
        task = RegressionTask(regressor=LinearBenchmarkRegressor(), pooling=pool)
        res = task.evaluate(train_reps, train_meta, test_reps, test_meta)
        assert res["pooling"] == pool
        assert "mse" in res


def test_regression_task_missing_target_raises():
    task = RegressionTask(regressor=LinearBenchmarkRegressor(), target_key="pitch")
    train_reps = torch.randn(4, 4, 8)
    train_meta = [{"wrong_target": 1.0} for _ in range(4)]
    test_reps = torch.randn(2, 4, 8)
    test_meta = [{"wrong_target": 1.0} for _ in range(2)]

    with pytest.raises(InvalidDatasetError, match="does not contain target key"):
        task.evaluate(train_reps, train_meta, test_reps, test_meta)


def test_regression_task_invalid_target_value_raises():
    task = RegressionTask(regressor=LinearBenchmarkRegressor(), target_key="target")
    train_reps = torch.randn(4, 4, 8)
    train_meta = [{"target": "not_a_number"} for _ in range(4)]
    test_reps = torch.randn(2, 4, 8)
    test_meta = [{"target": 1.0} for _ in range(2)]

    with pytest.raises(TaskError, match="cannot be converted to float"):
        task.evaluate(train_reps, train_meta, test_reps, test_meta)


def test_benchmark_runner_with_regression_task(tmp_path):
    train_dataset = DummyAudioDataset(num_samples=10, audio_length=320, num_classes=2)
    test_dataset = DummyAudioDataset(num_samples=5, audio_length=320, num_classes=2)
    model = DummyAudioModel(representation_dim=16)
    task = RegressionTask(regressor=LinearBenchmarkRegressor())

    benchmark = AudioBenchmark(
        model=model,
        model_name="dummy_reg_model",
        dataset=train_dataset,
        testset=test_dataset,
        task=task,
        cache_dir=tmp_path,
        enable_cache=True,
    )

    result = benchmark.run(
        batch_size=4,
        device="cpu",
    )

    assert result.task_name == "regression"
    assert "mse" in result.metrics
    assert "r2" in result.metrics
    assert result.metrics["regressor"] == "LinearBenchmarkRegressor"


def test_regression_task_string_regressor_raises():
    with pytest.raises(TypeError, match="Passing strings as models is not supported"):
        RegressionTask(regressor="linear")


def test_regression_task_none_or_missing_regressor_raises():
    with pytest.raises(TypeError, match="Passing None as a model is not allowed"):
        RegressionTask(regressor=None)
    with pytest.raises(TypeError):
        RegressionTask()
