import pytest
import torch

from audio_benchmark.regressors import (
    KNNBenchmarkRegressor,
    LinearBenchmarkRegressor,
    MLPBenchmarkRegressor,
    TransformerBenchmarkRegressor,
)


@pytest.fixture
def synthetic_regression_data():
    torch.manual_seed(42)
    # 16 samples, 8 features, continuous target
    X = torch.randn(16, 8, dtype=torch.float32)
    weights = torch.randn(8, 1, dtype=torch.float32)
    y = (X @ weights).squeeze(-1)
    return X[:12], y[:12], X[12:], y[12:]


def test_linear_regressor(synthetic_regression_data):
    X_train, y_train, X_test, y_test = synthetic_regression_data
    reg = LinearBenchmarkRegressor(alpha=1.0)
    reg.fit(X_train, y_train)

    preds = reg.predict(X_test)
    assert isinstance(preds, torch.Tensor)
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
    assert isinstance(preds, torch.Tensor)
    assert len(preds) == len(X_test)

    metrics = reg.evaluate(X_test, y_test)
    assert metrics["mse"] >= 0.0


def test_mlp_regressor(synthetic_regression_data):
    X_train, y_train, X_test, y_test = synthetic_regression_data
    clf = MLPBenchmarkRegressor(hidden_dim=16, epochs=1, batch_size=8, device="cpu")
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    assert isinstance(preds, torch.Tensor)
    assert len(preds) == len(X_test)

    metrics = clf.evaluate(X_test, y_test)
    assert metrics["mse"] >= 0.0


def test_transformer_regressor():
    torch.manual_seed(42)
    # 3D sequences: (B=10, T=6, C=8)
    X = torch.randn(10, 6, 8, dtype=torch.float32)
    y = torch.randn(10, dtype=torch.float32)

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
    assert isinstance(preds, torch.Tensor)
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
    preds_mean = reg_mean.predict(X[8:])
    assert isinstance(preds_mean, torch.Tensor)
    assert len(preds_mean) == 2


def test_regressor_feature_normalization(synthetic_regression_data):
    X_train, y_train, X_test, _ = synthetic_regression_data

    # Check defaults: normalize_features is True
    for reg in (
        LinearBenchmarkRegressor(),
        KNNBenchmarkRegressor(),
        MLPBenchmarkRegressor(epochs=1, batch_size=8, device="cpu"),
        TransformerBenchmarkRegressor(
            d_model=16, nhead=2, epochs=1, batch_size=4, device="cpu"
        ),
    ):
        assert reg.normalize_features is True
        assert reg.scaler is None

    # Check scaler is fitted and applied for 2D models
    for reg_class in (
        LinearBenchmarkRegressor,
        KNNBenchmarkRegressor,
        lambda: MLPBenchmarkRegressor(epochs=1, batch_size=8, device="cpu"),
    ):
        reg_norm = reg_class()
        reg_norm.fit(X_train, y_train)
        assert reg_norm.scaler is not None
        preds_norm = reg_norm.predict(X_test)
        assert isinstance(preds_norm, torch.Tensor)
        assert len(preds_norm) == len(X_test)

        # Check normalize_features=False
        reg_no_norm = reg_class()
        reg_no_norm.normalize_features = False
        reg_no_norm.fit(X_train, y_train)
        assert reg_no_norm.scaler is None
        preds_no_norm = reg_no_norm.predict(X_test)
        assert isinstance(preds_no_norm, torch.Tensor)
        assert len(preds_no_norm) == len(X_test)


def test_mlp_regressor_architecture_and_layers(synthetic_regression_data):
    X_train, y_train, X_test, _ = synthetic_regression_data

    # Verify default hidden_dim is 1024 and num_layers is 1
    default_mlp = MLPBenchmarkRegressor(device="cpu")
    assert default_mlp.hidden_dim == 1024
    assert default_mlp.num_layers == 1

    # Verify multi-layer MLP (num_layers=2)
    mlp_2layer = MLPBenchmarkRegressor(
        hidden_dim=32, num_layers=2, epochs=1, batch_size=8, device="cpu"
    )
    mlp_2layer.fit(X_train, y_train)
    preds_2layer = mlp_2layer.predict(X_test)
    assert isinstance(preds_2layer, torch.Tensor)
    assert len(preds_2layer) == len(X_test)

    # Verify linear projection (num_layers=0)
    mlp_0layer = MLPBenchmarkRegressor(
        num_layers=0, epochs=1, batch_size=8, device="cpu"
    )
    mlp_0layer.fit(X_train, y_train)
    preds_0layer = mlp_0layer.predict(X_test)
    assert isinstance(preds_0layer, torch.Tensor)
    assert len(preds_0layer) == len(X_test)

    # Verify negative num_layers raises ValueError
    with pytest.raises(ValueError, match="num_layers must be non-negative"):
        MLPBenchmarkRegressor(num_layers=-1, device="cpu")


def test_transformer_regressor_feature_normalization():
    torch.manual_seed(42)
    X = torch.randn(10, 6, 8, dtype=torch.float32) * 50.0 + 100.0
    y = torch.randn(10, dtype=torch.float32)
    reg = TransformerBenchmarkRegressor(
        d_model=16,
        nhead=2,
        epochs=1,
        batch_size=4,
        device="cpu",
        normalize_features=True,
    )
    reg.fit(X[:8], y[:8])
    assert reg.scaler is not None
    preds = reg.predict(X[8:])
    assert isinstance(preds, torch.Tensor)
    assert len(preds) == 2
