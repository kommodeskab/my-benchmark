"""Tests ensuring BenchmarkResult and all tasks always provide 'value' and 'normalized_value' metrics."""

import pytest
import torch

from audio_benchmark import (
    AudioBenchmark,
    BaseTask,
    BenchmarkResult,
    CTCSequenceProbe,
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


def test_base_task_normalize_metric():
    # Accuracy / R2 bounds (m_min=0.0, m_max=1.0)
    assert BaseTask.normalize_metric(0.85, 0.0, 1.0) == pytest.approx(0.85)
    assert BaseTask.normalize_metric(-0.5, 0.0, 1.0) == 0.0
    assert BaseTask.normalize_metric(1.2, 0.0, 1.0) == 1.0

    # Token Error Rate bounds (m_min=1.0, m_max=0.0)
    assert BaseTask.normalize_metric(0.2, 1.0, 0.0) == pytest.approx(0.8)
    assert BaseTask.normalize_metric(0.0, 1.0, 0.0) == 1.0
    assert BaseTask.normalize_metric(1.5, 1.0, 0.0) == 0.0


def test_benchmark_result_value_and_normalized_value_properties():
    # 1. When 'value' and 'normalized_value' are explicitly present
    res = BenchmarkResult(
        model_name="model_a",
        dataset_name="dataset_a",
        task_name="classification",
        metrics={"accuracy": 0.85, "value": 0.85, "normalized_value": 0.85},
        cached=False,
        representations_shape=(10, 20, 64),
    )
    assert res.metrics["value"] == 0.85
    assert res.value == 0.85
    assert res.normalized_value == 0.85

    # 2. When 'value' / 'normalized_value' are missing, properties return None without guessing
    res_inferred_acc = BenchmarkResult(
        model_name="model_a",
        dataset_name="dataset_a",
        task_name="classification",
        metrics={"accuracy": 0.92, "macro_f1": 0.90},
        cached=False,
        representations_shape=(10, 20, 64),
    )
    assert "value" not in res_inferred_acc.metrics
    assert res_inferred_acc.value is None
    assert res_inferred_acc.normalized_value is None


def test_benchmark_result_print_summary_handles_value(capsys):
    res = BenchmarkResult(
        model_name="model_a",
        dataset_name="dataset_a",
        task_name="custom_task",
        metrics={"custom_score": 0.95, "value": 0.95, "normalized_value": 0.95},
        cached=False,
        representations_shape=(10, 20, 64),
    )
    res.print_summary()
    captured = capsys.readouterr().out
    assert "Main Metric (value): 0.95" in captured
    assert "Normalized Metric (normalized_value): 0.95" in captured


def test_global_classification_task_value():
    task = GlobalClassificationTask(classifier=KNNBenchmarkClassifier())
    assert task.metric == "accuracy"
    train_reps = torch.randn(12, 4, 16)
    train_meta = [{"label": i % 3} for i in range(12)]
    test_reps = torch.randn(6, 4, 16)
    test_meta = [{"label": i % 3} for i in range(6)]

    res = task.evaluate(train_reps, train_meta, test_reps, test_meta)
    assert "value" in res
    assert "normalized_value" in res
    assert res["value"] == res["accuracy"]
    assert res["normalized_value"] == pytest.approx(res["accuracy"])
    assert 0.0 <= res["normalized_value"] <= 1.0

    with pytest.raises(TypeError):
        GlobalClassificationTask(classifier=KNNBenchmarkClassifier(), metric="macro_f1")  # type: ignore[call-arg]


def test_frame_classification_task_value():
    task = FrameClassificationTask(classifier=KNNBenchmarkClassifier())
    assert task.metric == "accuracy"
    train_reps = torch.randn(8, 4, 16)
    train_meta = [{"frame_labels": [i % 2] * 16} for i in range(8)]
    test_reps = torch.randn(4, 4, 16)
    test_meta = [{"frame_labels": [i % 2] * 16} for i in range(4)]

    res = task.evaluate(train_reps, train_meta, test_reps, test_meta)
    assert "value" in res
    assert "normalized_value" in res
    assert res["value"] == res["accuracy"]
    assert res["normalized_value"] == pytest.approx(res["accuracy"])
    assert 0.0 <= res["normalized_value"] <= 1.0

    with pytest.raises(TypeError):
        FrameClassificationTask(classifier=KNNBenchmarkClassifier(), metric="macro_f1")  # type: ignore[call-arg]


def test_regression_task_value():
    task = RegressionTask(regressor=LinearBenchmarkRegressor())
    assert task.metric == "r2"
    train_reps = torch.randn(12, 4, 16)
    train_meta = [{"target": float(i * 0.5)} for i in range(12)]
    test_reps = torch.randn(6, 4, 16)
    test_meta = [{"target": float(i * 0.5)} for i in range(6)]

    res = task.evaluate(train_reps, train_meta, test_reps, test_meta)
    assert "value" in res
    assert "normalized_value" in res
    assert res["value"] == res["r2"]
    assert res["normalized_value"] == pytest.approx(max(0.0, min(1.0, res["r2"])))
    assert 0.0 <= res["normalized_value"] <= 1.0

    with pytest.raises(TypeError):
        RegressionTask(regressor=LinearBenchmarkRegressor(), metric="mse")  # type: ignore[call-arg]


def test_frame_regression_task_value():
    task = FrameRegressionTask(regressor=LinearBenchmarkRegressor())
    assert task.metric == "r2"
    train_reps = torch.randn(8, 4, 16)
    train_meta = [
        {"frame_targets": [float(i + j * 0.1) for j in range(16)]} for i in range(8)
    ]
    test_reps = torch.randn(4, 4, 16)
    test_meta = [
        {"frame_targets": [float(i + j * 0.1) for j in range(16)]} for i in range(4)
    ]

    res = task.evaluate(train_reps, train_meta, test_reps, test_meta)
    assert "value" in res
    assert "normalized_value" in res
    assert res["value"] == res["r2"]
    assert res["normalized_value"] == pytest.approx(max(0.0, min(1.0, res["r2"])))
    assert 0.0 <= res["normalized_value"] <= 1.0

    with pytest.raises(TypeError):
        FrameRegressionTask(regressor=LinearBenchmarkRegressor(), metric="mae")  # type: ignore[call-arg]


def test_sequence_modeling_task_value():
    task = SequenceModelingTask(probe=CTCSequenceProbe(epochs=1))
    assert task.metric == "token_error_rate"
    train_reps = torch.randn(8, 10, 16)
    train_meta = [{"tokens": [f"tok_{j}" for j in range(3)]} for i in range(8)]
    test_reps = torch.randn(4, 10, 16)
    test_meta = [{"tokens": [f"tok_{j}" for j in range(3)]} for i in range(4)]

    res = task.evaluate(train_reps, train_meta, test_reps, test_meta)
    assert "value" in res
    assert "normalized_value" in res
    assert res["value"] == res["token_error_rate"]
    assert res["normalized_value"] == pytest.approx(
        max(0.0, min(1.0, 1.0 - res["token_error_rate"]))
    )
    assert 0.0 <= res["normalized_value"] <= 1.0

    with pytest.raises(TypeError):
        SequenceModelingTask(
            probe=CTCSequenceProbe(epochs=1),
            metric="sequence_accuracy",  # type: ignore[call-arg]
        )


def test_audio_benchmark_end_to_end_has_value_key(tmp_path):
    dataset = DummyAudioDataset(num_samples=10, audio_length=320, num_classes=2)
    testset = DummyAudioDataset(num_samples=6, audio_length=320, num_classes=2)
    model = DummyAudioModel(representation_dim=16)

    # Global classification
    benchmark_clf = AudioBenchmark(
        dataset=dataset,
        testset=testset,
        task=GlobalClassificationTask(classifier=KNNBenchmarkClassifier()),
        dataset_name="dummy_audio",
        cache_dir=tmp_path,
        enable_cache=False,
    )
    res_clf = benchmark_clf.run(
        model=model,
        model_name="test_model",
    )[0]
    assert "value" in res_clf.metrics
    assert "normalized_value" in res_clf.metrics
    assert res_clf.metrics["value"] == res_clf.metrics["accuracy"]
    assert res_clf.value == res_clf.metrics["accuracy"]
    assert res_clf.normalized_value == res_clf.metrics["normalized_value"]

    # Regression
    benchmark_reg = AudioBenchmark(
        dataset=dataset,
        testset=testset,
        task=RegressionTask(regressor=LinearBenchmarkRegressor()),
        dataset_name="dummy_audio",
        cache_dir=tmp_path,
        enable_cache=False,
    )
    res_reg = benchmark_reg.run(
        model=model,
        model_name="test_model",
    )[0]
    assert "value" in res_reg.metrics
    assert "normalized_value" in res_reg.metrics
    assert res_reg.metrics["value"] == res_reg.metrics["r2"]
    assert res_reg.value == res_reg.metrics["r2"]
    assert res_reg.normalized_value == res_reg.metrics["normalized_value"]
