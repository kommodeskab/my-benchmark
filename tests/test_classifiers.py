import pytest
import torch

from audio_benchmark.classifiers import (
    KNNBenchmarkClassifier,
    LinearBenchmarkClassifier,
    MLPBenchmarkClassifier,
    TransformerBenchmarkClassifier,
)


@pytest.fixture
def synthetic_data():
    torch.manual_seed(42)
    # Small synthetic dataset for fast execution: 20 samples, 8 features, 2 classes
    X = torch.randn(20, 8, dtype=torch.float32)
    y = torch.tensor([0] * 10 + [1] * 10, dtype=torch.long)
    X[y == 1] += 2.0
    return X[:14], y[:14], X[14:], y[14:]


def test_linear_classifier(synthetic_data):
    X_train, y_train, X_test, y_test = synthetic_data

    # Default LogisticRegression linear classifier
    clf = LinearBenchmarkClassifier(C=1.0)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    assert isinstance(preds, torch.Tensor)
    assert len(preds) == len(X_test)

    proba = clf.predict_proba(X_test)
    assert isinstance(proba, torch.Tensor)
    assert proba.shape == (len(X_test), 2)
    assert torch.allclose(
        proba.sum(dim=1).to(torch.float32), torch.ones(len(X_test), dtype=torch.float32)
    )

    metrics = clf.evaluate(X_test, y_test)
    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert "micro_f1" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0

    # Ridge classifier mode
    clf_ridge = LinearBenchmarkClassifier(model_type="ridge")
    clf_ridge.fit(X_train, y_train)
    preds_ridge = clf_ridge.predict(X_test)
    assert isinstance(preds_ridge, torch.Tensor)
    assert len(preds_ridge) == len(X_test)
    metrics_ridge = clf_ridge.evaluate(X_test, y_test)
    assert 0.0 <= metrics_ridge["accuracy"] <= 1.0

    # Ridge does not support predict_proba
    with pytest.raises(AttributeError, match="does not support predict_proba"):
        clf_ridge.predict_proba(X_test)

    # Disallowed alias for model_type
    with pytest.raises(ValueError, match="Unknown model_type"):
        LinearBenchmarkClassifier(model_type="ridge_classifier")


def test_knn_classifier(synthetic_data):
    X_train, y_train, X_test, y_test = synthetic_data
    clf = KNNBenchmarkClassifier(n_neighbors=3)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    assert isinstance(preds, torch.Tensor)
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
    assert isinstance(preds, torch.Tensor)
    assert len(preds) == len(X_test)

    metrics = clf.evaluate(X_test, y_test)
    assert "accuracy" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_transformer_classifier():
    torch.manual_seed(42)
    # 3D sequences: (B=10, T=6, C=8)
    X = torch.randn(10, 6, 8, dtype=torch.float32)
    y = torch.tensor([0, 1] * 5, dtype=torch.long)

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
    assert isinstance(preds, torch.Tensor)
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
    preds_mean = clf_mean.predict(X[8:])
    assert isinstance(preds_mean, torch.Tensor)
    assert len(preds_mean) == 2


def test_classifier_feature_normalization(synthetic_data):
    X_train, y_train, X_test, _ = synthetic_data

    # Check defaults: normalize_features is True
    for clf in (
        LinearBenchmarkClassifier(),
        KNNBenchmarkClassifier(),
        MLPBenchmarkClassifier(epochs=1, batch_size=8, device="cpu"),
        TransformerBenchmarkClassifier(
            d_model=16, nhead=2, epochs=1, batch_size=4, device="cpu"
        ),
    ):
        assert clf.normalize_features is True
        assert clf.scaler is None

    # Check scaler is fitted and applied for 2D models
    for clf_class in (
        LinearBenchmarkClassifier,
        KNNBenchmarkClassifier,
        lambda: MLPBenchmarkClassifier(epochs=1, batch_size=8, device="cpu"),
    ):
        clf_norm = clf_class()
        clf_norm.fit(X_train, y_train)
        assert clf_norm.scaler is not None
        preds_norm = clf_norm.predict(X_test)
        assert isinstance(preds_norm, torch.Tensor)
        assert len(preds_norm) == len(X_test)

        # Check normalize_features=False
        clf_no_norm = clf_class()
        clf_no_norm.normalize_features = False
        clf_no_norm.fit(X_train, y_train)
        assert clf_no_norm.scaler is None
        preds_no_norm = clf_no_norm.predict(X_test)
        assert isinstance(preds_no_norm, torch.Tensor)
        assert len(preds_no_norm) == len(X_test)


def test_mlp_classifier_architecture_and_layers(synthetic_data):
    X_train, y_train, X_test, _ = synthetic_data

    # Verify default hidden_dim is 1024 and num_layers is 1
    default_mlp = MLPBenchmarkClassifier(device="cpu")
    assert default_mlp.hidden_dim == 1024
    assert default_mlp.num_layers == 1

    # Verify multi-layer MLP (num_layers=2)
    mlp_2layer = MLPBenchmarkClassifier(
        hidden_dim=32, num_layers=2, epochs=1, batch_size=8, device="cpu"
    )
    mlp_2layer.fit(X_train, y_train)
    preds_2layer = mlp_2layer.predict(X_test)
    assert isinstance(preds_2layer, torch.Tensor)
    assert len(preds_2layer) == len(X_test)

    # Verify linear projection (num_layers=0)
    mlp_0layer = MLPBenchmarkClassifier(
        num_layers=0, epochs=1, batch_size=8, device="cpu"
    )
    mlp_0layer.fit(X_train, y_train)
    preds_0layer = mlp_0layer.predict(X_test)
    assert isinstance(preds_0layer, torch.Tensor)
    assert len(preds_0layer) == len(X_test)

    # Verify negative num_layers raises ValueError
    with pytest.raises(ValueError, match="num_layers must be non-negative"):
        MLPBenchmarkClassifier(num_layers=-1, device="cpu")


def test_transformer_classifier_feature_normalization():
    torch.manual_seed(42)
    X = torch.randn(10, 6, 8, dtype=torch.float32) * 50.0 + 100.0
    y = torch.tensor([0, 1] * 5, dtype=torch.long)
    clf = TransformerBenchmarkClassifier(
        d_model=16,
        nhead=2,
        epochs=1,
        batch_size=4,
        device="cpu",
        normalize_features=True,
    )
    clf.fit(X[:8], y[:8])
    assert clf.scaler is not None
    preds = clf.predict(X[8:])
    assert isinstance(preds, torch.Tensor)
    assert len(preds) == 2
