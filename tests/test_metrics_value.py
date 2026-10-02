"""Tests ensuring BenchmarkResult and all tasks always provide a 'value' metric."""

import pytest
import torch

from audio_benchmark import (
    AudioBenchmark,
    BenchmarkResult,
    DummyAudioDataset,
    DummyAudioModel,
    FrameClassificationTask,
    FrameRegressionTask,
    GlobalClassificationTask,
    KNNBenchmarkClassifier,
    LinearBenchmarkRegressor,
    RegressionTask,
    SequenceModelingTask,
)


def test_benchmark_result_value_property_and_post_init():
    # 1. When 'value' is explicitly present
    res = BenchmarkResult(
        model_name="model_a",
        dataset_name="dataset_a",
        task_name="classification",
        metrics={"accuracy": 0.85, "value": 0.85},
        cached=False,
        representations_shape=(10, 20, 64),
    )
    assert res.metrics["value"] == 0.85
    assert res.value == 0.85

    # 2. When 'value' is missing, it should infer from candidate keys
    res_inferred_acc = BenchmarkResult(
        model_name="model_a",
        dataset_name="dataset_a",
        task_name="classification",
        metrics={"accuracy": 0.92, "macro_f1": 0.90},
        cached=False,
        representations_shape=(10, 20, 64),
    )
    assert res_inferred_acc.metrics["value"] == 0.92
    assert res_inferred_acc.value == 0.92

    res_inferred_r2 = BenchmarkResult(
        model_name="model_b",
        dataset_name="dataset_b",
        task_name="regression",
        metrics={"r2": 0.78, "mse": 0.05},
        cached=False,
        representations_shape=(10, 20, 64),
    )
    assert res_inferred_r2.metrics["value"] == 0.78
    assert res_inferred_r2.value == 0.78

    res_inferred_seq = BenchmarkResult(
        model_name="model_c",
        dataset_name="dataset_c",
        task_name="sequence_modeling",
        metrics={"token_error_rate": 0.15, "sequence_accuracy": 0.80},
        cached=False,
        representations_shape=(10, 20, 64),
    )
    assert res_inferred_seq.metrics["value"] == 0.15
    assert res_inferred_seq.value == 0.15


def test_benchmark_result_print_summary_handles_value(capsys):
    res = BenchmarkResult(
        model_name="model_a",
        dataset_name="dataset_a",
        task_name="custom_task",
        metrics={"custom_score": 0.95, "value": 0.95},
        cached=False,
        representations_shape=(10, 20, 64),
    )
    res.print_summary()
    captured = capsys.readouterr().out
    assert "Main Metric (value): 0.95" in captured


def test_global_classification_task_value():
    task_default = GlobalClassificationTask(classifier=KNNBenchmarkClassifier())
    reps = torch.randn(12, 4, 16)
    meta = [{"label": i % 3} for i in range(12)]

    res_default = task_default.evaluate(reps, meta)
    assert "value" in res_default
    assert "accuracy" in res_default
    assert res_default["value"] == res_default["accuracy"]

    # Custom metric: macro_f1
    task_custom = GlobalClassificationTask(
        classifier=KNNBenchmarkClassifier(), metric="macro_f1"
    )
    res_custom = task_custom.evaluate(reps, meta)
    assert "value" in res_custom
    assert res_custom["value"] == res_custom["macro_f1"]

    # Invalid metric
    with pytest.raises(ValueError, match="Unknown metric 'invalid'"):
        GlobalClassificationTask(classifier=KNNBenchmarkClassifier(), metric="invalid")


def test_frame_classification_task_value():
    task_default = FrameClassificationTask(classifier=KNNBenchmarkClassifier())
    reps = torch.randn(8, 4, 16)
    meta = [{"frame_labels": [i % 2] * 16} for i in range(8)]

    res_default = task_default.evaluate(reps, meta)
    assert "value" in res_default
    assert "accuracy" in res_default
    assert res_default["value"] == res_default["accuracy"]

    # Custom metric: macro_f1
    task_custom = FrameClassificationTask(
        classifier=KNNBenchmarkClassifier(), metric="macro_f1"
    )
    res_custom = task_custom.evaluate(reps, meta)
    assert "value" in res_custom
    assert res_custom["value"] == res_custom["macro_f1"]

    with pytest.raises(ValueError, match="Unknown metric 'invalid'"):
        FrameClassificationTask(classifier=KNNBenchmarkClassifier(), metric="invalid")


def test_regression_task_value():
    task_default = RegressionTask(regressor=LinearBenchmarkRegressor())
    reps = torch.randn(12, 4, 16)
    meta = [{"target": float(i * 0.5)} for i in range(12)]

    res_default = task_default.evaluate(reps, meta)
    assert "value" in res_default
    assert "r2" in res_default
    assert res_default["value"] == res_default["r2"]

    # Custom metric: mse
    task_custom = RegressionTask(regressor=LinearBenchmarkRegressor(), metric="mse")
    res_custom = task_custom.evaluate(reps, meta)
    assert "value" in res_custom
    assert "mse" in res_custom
    assert res_custom["value"] == res_custom["mse"]

    with pytest.raises(ValueError, match="Unknown metric 'invalid'"):
        RegressionTask(regressor=LinearBenchmarkRegressor(), metric="invalid")


def test_frame_regression_task_value():
    task_default = FrameRegressionTask(regressor=LinearBenchmarkRegressor())
    reps = torch.randn(8, 4, 16)
    meta = [
        {"frame_targets": [float(i + j * 0.1) for j in range(16)]} for i in range(8)
    ]

    res_default = task_default.evaluate(reps, meta)
    assert "value" in res_default
    assert "r2" in res_default
    assert res_default["value"] == res_default["r2"]

    # Custom metric: mae
    task_custom = FrameRegressionTask(
        regressor=LinearBenchmarkRegressor(), metric="mae"
    )
    res_custom = task_custom.evaluate(reps, meta)
    assert "value" in res_custom
    assert "mae" in res_custom
    assert res_custom["value"] == res_custom["mae"]

    with pytest.raises(ValueError, match="Unknown metric 'invalid'"):
        FrameRegressionTask(regressor=LinearBenchmarkRegressor(), metric="invalid")


def test_sequence_modeling_task_value():
    task_default = SequenceModelingTask(method="ctc", epochs=1)
    reps = torch.randn(8, 10, 16)
    meta = [{"tokens": [f"tok_{j}" for j in range(3)]} for i in range(8)]

    res_default = task_default.evaluate(reps, meta)
    assert "value" in res_default
    assert "token_error_rate" in res_default
    assert res_default["value"] == res_default["token_error_rate"]

    # Custom metric: sequence_accuracy
    task_custom = SequenceModelingTask(
        method="ctc", epochs=1, metric="sequence_accuracy"
    )
    res_custom = task_custom.evaluate(reps, meta)
    assert "value" in res_custom
    assert "sequence_accuracy" in res_custom
    assert res_custom["value"] == res_custom["sequence_accuracy"]

    with pytest.raises(ValueError, match="Unknown metric 'invalid'"):
        SequenceModelingTask(metric="invalid")


def test_audio_benchmark_end_to_end_has_value_key(tmp_path):
    benchmark = AudioBenchmark(cache_dir=tmp_path, enable_cache=False)
    dataset = DummyAudioDataset(num_samples=10, audio_length=320, num_classes=2)
    model = DummyAudioModel(representation_dim=16)

    # Global classification
    res_clf = benchmark.run(
        model=model,
        model_name="test_model",
        dataset=dataset,
        task=GlobalClassificationTask(classifier=KNNBenchmarkClassifier()),
    )
    assert "value" in res_clf.metrics
    assert res_clf.metrics["value"] == res_clf.metrics["accuracy"]
    assert res_clf.value == res_clf.metrics["accuracy"]

    # Regression
    res_reg = benchmark.run(
        model=model,
        model_name="test_model",
        dataset=dataset,
        task=RegressionTask(regressor=LinearBenchmarkRegressor()),
    )
    assert "value" in res_reg.metrics
    assert res_reg.metrics["value"] == res_reg.metrics["r2"]
    assert res_reg.value == res_reg.metrics["r2"]
