"""Tests for downstream classifiers (KNN, MLP)."""

import numpy as np
import pytest

from audio_benchmark.classifiers import (
    KNNBenchmarkClassifier,
    MLPBenchmarkClassifier,
    TransformerBenchmarkClassifier,
    create_classifier,
)


@pytest.fixture
def synthetic_data():
    np.random.seed(42)
    # Small synthetic dataset for fast execution: 20 samples, 8 features, 2 classes
    X = np.random.randn(20, 8).astype(np.float32)
    y = np.array([0] * 10 + [1] * 10)
    X[y == 1] += 2.0
    return X[:14], y[:14], X[14:], y[14:]


def test_knn_classifier(synthetic_data):
    X_train, y_train, X_test, y_test = synthetic_data
    clf = KNNBenchmarkClassifier(n_neighbors=3)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    assert len(preds) == len(X_test)

    metrics = clf.evaluate(X_test, y_test)
    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert "micro_f1" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_mlp_classifier(synthetic_data):
    X_train, y_train, X_test, y_test = synthetic_data
    clf = MLPBenchmarkClassifier(hidden_dim=16, epochs=1, batch_size=8, device="cpu")
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    assert len(preds) == len(X_test)

    metrics = clf.evaluate(X_test, y_test)
    assert "accuracy" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_transformer_classifier():
    np.random.seed(42)
    # 3D sequences: (B=10, T=6, C=8)
    X = np.random.randn(10, 6, 8).astype(np.float32)
    y = np.array([0, 1] * 5)

    clf_cls = TransformerBenchmarkClassifier(
        d_model=16,
        nhead=2,
        num_layers=1,
        dim_feedforward=32,
        aggregation="cls",
        epochs=1,
        batch_size=4,
        device="cpu",
    )
    clf_cls.fit(X[:8], y[:8])
    preds = clf_cls.predict(X[8:])
    assert len(preds) == 2
    metrics = clf_cls.evaluate(X[8:], y[8:])
    assert "accuracy" in metrics
    assert clf_cls.accepts_sequence is True

    # Test mean aggregation
    clf_mean = TransformerBenchmarkClassifier(
        d_model=16,
        nhead=2,
        num_layers=1,
        dim_feedforward=32,
        aggregation="mean",
        epochs=1,
        batch_size=4,
        device="cpu",
    )
    clf_mean.fit(X[:8], y[:8])
    assert len(clf_mean.predict(X[8:])) == 2


def test_create_classifier_factory():
    knn = create_classifier("knn")
    assert isinstance(knn, KNNBenchmarkClassifier)

    mlp = create_classifier("mlp")
    assert isinstance(mlp, MLPBenchmarkClassifier)

    tf = create_classifier("transformer")
    assert isinstance(tf, TransformerBenchmarkClassifier)

    same_knn = create_classifier(knn)
    assert same_knn is knn

    with pytest.raises(ValueError, match="Unsupported classifier type"):
        create_classifier("unsupported_classifier")
