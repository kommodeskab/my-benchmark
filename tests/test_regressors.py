"""Tests for downstream regression models (Linear/Ridge, KNN, MLP)."""

import numpy as np
import pytest

from audio_benchmark.regressors import (
    KNNBenchmarkRegressor,
    LinearBenchmarkRegressor,
    MLPBenchmarkRegressor,
    TransformerBenchmarkRegressor,
    create_regressor,
)


@pytest.fixture
def synthetic_regression_data():
    np.random.seed(42)
    # 16 samples, 8 features, continuous target
    X = np.random.randn(16, 8).astype(np.float32)
    weights = np.random.randn(8).astype(np.float32)
    y = X @ weights
    return X[:12], y[:12], X[12:], y[12:]


def test_linear_regressor(synthetic_regression_data):
    X_train, y_train, X_test, y_test = synthetic_regression_data
    reg = LinearBenchmarkRegressor(alpha=1.0)
    reg.fit(X_train, y_train)

    preds = reg.predict(X_test)
    assert len(preds) == len(X_test)

    metrics = reg.evaluate(X_test, y_test)
    assert "mse" in metrics
    assert "rmse" in metrics
    assert "mae" in metrics
    assert "r2" in metrics
    assert "pearson_r" in metrics
    assert metrics["mse"] >= 0.0


def test_knn_regressor(synthetic_regression_data):
    X_train, y_train, X_test, y_test = synthetic_regression_data
    reg = KNNBenchmarkRegressor(n_neighbors=2)
    reg.fit(X_train, y_train)

    preds = reg.predict(X_test)
    assert len(preds) == len(X_test)

    metrics = reg.evaluate(X_test, y_test)
    assert metrics["mse"] >= 0.0


def test_mlp_regressor(synthetic_regression_data):
    X_train, y_train, X_test, y_test = synthetic_regression_data
    clf = MLPBenchmarkRegressor(hidden_dim=16, epochs=1, batch_size=8, device="cpu")
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    assert len(preds) == len(X_test)

    metrics = clf.evaluate(X_test, y_test)
    assert metrics["mse"] >= 0.0


def test_transformer_regressor():
    np.random.seed(42)
    # 3D sequences: (B=10, T=6, C=8)
    X = np.random.randn(10, 6, 8).astype(np.float32)
    y = np.random.randn(10).astype(np.float32)

    reg_cls = TransformerBenchmarkRegressor(
        d_model=16,
        nhead=2,
        num_layers=1,
        dim_feedforward=32,
        aggregation="cls",
        epochs=1,
        batch_size=4,
        device="cpu",
    )
    reg_cls.fit(X[:8], y[:8])
    preds = reg_cls.predict(X[8:])
    assert len(preds) == 2
    metrics = reg_cls.evaluate(X[8:], y[8:])
    assert "mse" in metrics
    assert reg_cls.accepts_sequence is True

    # Test mean aggregation
    reg_mean = TransformerBenchmarkRegressor(
        d_model=16,
        nhead=2,
        num_layers=1,
        dim_feedforward=32,
        aggregation="mean",
        epochs=1,
        batch_size=4,
        device="cpu",
    )
    reg_mean.fit(X[:8], y[:8])
    assert len(reg_mean.predict(X[8:])) == 2


def test_create_regressor_factory():
    linear = create_regressor("linear")
    assert isinstance(linear, LinearBenchmarkRegressor)

    knn = create_regressor("knn")
    assert isinstance(knn, KNNBenchmarkRegressor)

    mlp = create_regressor("mlp")
    assert isinstance(mlp, MLPBenchmarkRegressor)

    tf = create_regressor("transformer")
    assert isinstance(tf, TransformerBenchmarkRegressor)

    same_knn = create_regressor(knn)
    assert same_knn is knn

    with pytest.raises(ValueError, match="Unsupported regressor type"):
        create_regressor("unsupported_regressor")

    for alias in (
        "ridge",
        "linear_regression",
        "k_nearest_neighbors",
        "multilayer_perceptron",
        "transformer_encoder",
    ):
        with pytest.raises(ValueError, match="Unsupported regressor type"):
            create_regressor(alias)
