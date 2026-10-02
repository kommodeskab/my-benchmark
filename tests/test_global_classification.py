"""Tests for GlobalClassificationTask."""

import pytest
import torch

from audio_benchmark.classifiers import (
    KNNBenchmarkClassifier,
    LinearBenchmarkClassifier,
    MLPBenchmarkClassifier,
    TransformerBenchmarkClassifier,
)
from audio_benchmark.exceptions import InvalidDatasetError, TaskError
from audio_benchmark.tasks.global_classification import GlobalClassificationTask


def test_global_classification_with_different_classifiers():
    task_linear = GlobalClassificationTask(
        classifier=LinearBenchmarkClassifier(), test_size=0.25, random_state=42
    )
    task_knn = GlobalClassificationTask(
        classifier=KNNBenchmarkClassifier(), test_size=0.25, random_state=42
    )
    task_mlp = GlobalClassificationTask(
        classifier=MLPBenchmarkClassifier(epochs=1, device="cpu"),
        test_size=0.25,
        random_state=42,
    )

    reps = torch.randn(16, 4, 16)
    meta = [{"label": f"cls_{i % 2}"} for i in range(16)]

    # Test examine_data provides dataset statistics and baseline metric
    info = task_knn.examine_data(meta)
    assert info["num_classes"] == 2
    assert info["train_samples"] == 12
    assert info["test_samples"] == 4
    assert info["baseline_strategy"] == "majority_class"
    assert info["baseline_metric"] == "accuracy"
    assert 0.0 <= info["baseline_value"] <= 1.0
    assert info["chance_level"] == 0.5
    assert set(info["classes"]) == {"cls_0", "cls_1"}

    for task in (task_linear, task_knn, task_mlp):
        res = task.evaluate(reps, meta)
        # evaluate should only return probe/task name, metrics, and training time
        assert "num_classes" not in res
        assert "train_samples" not in res
        assert "test_samples" not in res
        assert "classes" not in res
        assert 0.0 <= res["accuracy"] <= 1.0
        assert "macro_f1" in res
        assert "training_time" in res
        assert res["training_time"] >= 0.0


def test_global_classification_with_transformer():
    task_tf = GlobalClassificationTask(
        classifier=TransformerBenchmarkClassifier(
            d_model=16,
            nhead=2,
            num_layers=1,
            dim_feedforward=32,
            epochs=1,
            device="cpu",
        ),
        test_size=0.25,
        random_state=42,
    )

    reps = torch.randn(16, 4, 16)
    meta = [{"label": f"cls_{i % 2}"} for i in range(16)]

    res = task_tf.evaluate(reps, meta)
    assert res["classifier"] == "TransformerBenchmarkClassifier"
    assert res["pooling"] == "none (transformer)"
    assert "train_samples" not in res
    assert 0.0 <= res["accuracy"] <= 1.0
    assert "training_time" in res
    assert res["training_time"] > 0.0


def test_global_classification_pooling_modes():
    reps = torch.randn(12, 6, 8)
    meta = [{"label": i % 2} for i in range(12)]

    for pool in ["mean", "max", "first", "last"]:
        task = GlobalClassificationTask(
            classifier=KNNBenchmarkClassifier(), pooling=pool
        )
        res = task.evaluate(reps, meta)
        assert res["pooling"] == pool
        assert 0.0 <= res["accuracy"] <= 1.0


def test_global_classification_with_split_key():
    reps = torch.randn(15, 4, 8)
    meta = []
    for i in range(15):
        split = "train" if i < 10 else "test"
        meta.append({"label": i % 2, "split": split})

    task = GlobalClassificationTask(
        classifier=KNNBenchmarkClassifier(), split_key="split"
    )
    info = task.examine_data(meta)
    assert info["train_samples"] == 10
    assert info["test_samples"] == 5

    res = task.evaluate(reps, meta)
    assert "train_samples" not in res
    assert "test_samples" not in res


def test_global_classification_missing_label_raises():
    task = GlobalClassificationTask(
        classifier=KNNBenchmarkClassifier(), label_key="target_label"
    )
    reps = torch.randn(6, 4, 8)
    meta = [{"wrong_key": 1} for _ in range(6)]

    with pytest.raises(InvalidDatasetError, match="does not contain label key"):
        task.evaluate(reps, meta)


def test_global_classification_single_class_raises():
    task = GlobalClassificationTask(classifier=KNNBenchmarkClassifier())
    reps = torch.randn(6, 4, 8)
    meta = [{"label": 0} for _ in range(6)]

    with pytest.raises(TaskError, match="requires at least 2 distinct classes"):
        task.evaluate(reps, meta)


def test_global_classification_string_classifier_raises():
    with pytest.raises(TypeError, match="Passing strings as models is not supported"):
        GlobalClassificationTask(classifier="knn")


def test_global_classification_none_or_missing_classifier_raises():
    with pytest.raises(TypeError, match="Passing None as a model is not allowed"):
        GlobalClassificationTask(classifier=None)
    with pytest.raises(TypeError):
        GlobalClassificationTask()
