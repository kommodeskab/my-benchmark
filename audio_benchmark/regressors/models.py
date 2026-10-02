"""Downstream regression models for audio representation benchmarking."""

from abc import ABC, abstractmethod

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from torch import nn
from tqdm import tqdm

from audio_benchmark.utils import PositionalEncoding, resolve_device, to_numpy


class BaseRegressor(ABC):
    """Abstract base class for downstream benchmark regressors."""

    @abstractmethod
    def fit(
        self, X: np.ndarray | torch.Tensor, y: np.ndarray | torch.Tensor
    ) -> "BaseRegressor":
        """Train the regressor on representation features X and continuous targets y.

        Args:
            X: Input feature representations of shape (N, C) or (N, T', C').
            y: Target continuous values of shape (N,).

        Returns:
            BaseRegressor: Fitted regressor instance.
        """

    @abstractmethod
    def predict(self, X: np.ndarray | torch.Tensor) -> np.ndarray:
        """Predict continuous targets for features X.

        Args:
            X: Input feature representations of shape (N, C) or (N, T', C').

        Returns:
            np.ndarray: Predicted continuous targets of shape (N,).
        """

    def evaluate(
        self, X_test: np.ndarray | torch.Tensor, y_test: np.ndarray | torch.Tensor
    ) -> dict[str, float]:
        """Evaluate predictions against ground truth targets.

        Args:
            X_test: Test feature representations.
            y_test: Ground truth continuous targets.

        Returns:
            dict[str, float]: Dictionary containing 'mse', 'rmse', 'mae', 'r2', and 'pearson_r'.
        """
        y_test_np = to_numpy(y_test).ravel()
        y_pred = self.predict(X_test).ravel()

        mse = float(mean_squared_error(y_test_np, y_pred))
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_test_np, y_pred))
        r2 = float(r2_score(y_test_np, y_pred))

        # Compute Pearson correlation r if variance is non-zero
        std_true = np.std(y_test_np)
        std_pred = np.std(y_pred)
        if std_true > 1e-8 and std_pred > 1e-8:
            corr = float(np.corrcoef(y_test_np, y_pred)[0, 1])
        else:
            corr = 0.0

        return {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "pearson_r": corr,
        }


class LinearBenchmarkRegressor(BaseRegressor):
    """Linear / Ridge downstream regressor.

    Args:
        alpha: Regularization strength for Ridge regression (larger values specify stronger regularization).
        **kwargs: Additional keyword arguments passed to scikit-learn Ridge.
    """

    def __init__(self, alpha: float = 1.0, **kwargs):
        self.alpha = alpha
        self.kwargs = kwargs
        self.model = Ridge(alpha=alpha, **kwargs)

    def fit(
        self, X: np.ndarray | torch.Tensor, y: np.ndarray | torch.Tensor
    ) -> "LinearBenchmarkRegressor":
        """Fit the linear regressor on features X and targets y.

        Args:
            X: Input feature matrix of shape (N, C).
            y: Target continuous values of shape (N,).

        Returns:
            LinearBenchmarkRegressor: Fitted regressor instance.
        """
        X_np = to_numpy(X)
        y_np = to_numpy(y).ravel()
        self.model.fit(X_np, y_np)
        return self

    def predict(self, X: np.ndarray | torch.Tensor) -> np.ndarray:
        """Predict continuous targets for features X.

        Args:
            X: Input feature matrix of shape (N, C).

        Returns:
            np.ndarray: Predicted continuous targets of shape (N,).
        """
        X_np = to_numpy(X)
        return self.model.predict(X_np)


class KNNBenchmarkRegressor(BaseRegressor):
    """K-Nearest Neighbors downstream regressor.

    Args:
        n_neighbors: Number of nearest neighbors to query.
        metric: Distance metric to use for neighbor search (default: 'cosine').
        **kwargs: Additional keyword arguments passed to KNeighborsRegressor.
    """

    def __init__(self, n_neighbors: int = 5, metric: str = "cosine", **kwargs):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.kwargs = kwargs
        self.model = KNeighborsRegressor(
            n_neighbors=n_neighbors, metric=metric, **kwargs
        )

    def fit(
        self, X: np.ndarray | torch.Tensor, y: np.ndarray | torch.Tensor
    ) -> "KNNBenchmarkRegressor":
        """Fit the KNN regressor on features X and targets y.

        Args:
            X: Input feature matrix of shape (N, C).
            y: Target continuous values of shape (N,).

        Returns:
            KNNBenchmarkRegressor: Fitted regressor instance.
        """
        X_np = to_numpy(X)
        y_np = to_numpy(y).ravel()
        effective_k = min(self.n_neighbors, len(X_np))
        if effective_k != self.n_neighbors:
            self.model = KNeighborsRegressor(
                n_neighbors=effective_k, metric=self.metric, **self.kwargs
            )
        self.model.fit(X_np, y_np)
        return self

    def predict(self, X: np.ndarray | torch.Tensor) -> np.ndarray:
        """Predict continuous targets for features X.

        Args:
            X: Input feature matrix of shape (N, C).

        Returns:
            np.ndarray: Predicted continuous targets of shape (N,).
        """
        X_np = to_numpy(X)
        return self.model.predict(X_np)


class _TorchMLPRegressor(nn.Module):
    """Simple 2-layer MLP for regression.

    Args:
        in_dim: Input feature dimension.
        hidden_dim: Number of hidden units.
        dropout: Dropout rate applied after the hidden layer.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, in_dim).

        Returns:
            torch.Tensor: Predicted continuous values of shape (B,).
        """
        return self.net(x).squeeze(-1)


class MLPBenchmarkRegressor(BaseRegressor):
    """Multi-Layer Perceptron downstream regressor.

    Args:
        hidden_dim: Number of units in the hidden layer.
        dropout: Dropout probability applied after the hidden layer.
        lr: Learning rate for Adam optimizer.
        epochs: Number of training epochs.
        batch_size: Batch size for training DataLoader.
        weight_decay: L2 penalty for Adam optimizer.
        device: Device to use for training and inference ('auto', 'cpu', 'cuda').
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        lr: float = 1e-3,
        epochs: int = 35,
        batch_size: int = 32,
        weight_decay: float = 1e-4,
        device: str = "auto",
        show_progress: bool = True,
    ):
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.device = resolve_device(device)
        self.show_progress = show_progress
        self.model: _TorchMLPRegressor | None = None
        self.y_mean: float = 0.0
        self.y_std: float = 1.0

    def fit(
        self, X: np.ndarray | torch.Tensor, y: np.ndarray | torch.Tensor
    ) -> "MLPBenchmarkRegressor":
        """Train the MLP regressor on features X and targets y.

        Args:
            X: Input feature representations of shape (N, C).
            y: Target continuous values of shape (N,).

        Returns:
            MLPBenchmarkRegressor: Fitted regressor instance.
        """
        X_t = torch.as_tensor(to_numpy(X), dtype=torch.float32)
        y_t = torch.as_tensor(to_numpy(y), dtype=torch.float32).ravel()

        self.y_mean = float(y_t.mean().item())
        self.y_std = float(y_t.std().item()) if float(y_t.std().item()) > 1e-8 else 1.0
        y_norm = (y_t - self.y_mean) / self.y_std

        in_dim = X_t.shape[1]
        self.model = _TorchMLPRegressor(
            in_dim=in_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
        ).to(self.device)

        dataset = torch.utils.data.TensorDataset(X_t, y_norm)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        criterion = nn.MSELoss()

        self.model.train()
        epoch_iter = tqdm(
            range(self.epochs),
            desc="Training MLPRegressor",
            disable=not self.show_progress,
            leave=False,
        )
        for _ in epoch_iter:
            running_loss = 0.0
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                preds = self.model(batch_x)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += float(loss.item())
            epoch_iter.set_postfix(
                {"loss": f"{running_loss / max(1, len(loader)):.4f}"}
            )

        return self

    def predict(self, X: np.ndarray | torch.Tensor) -> np.ndarray:
        """Predict continuous targets for features X.

        Args:
            X: Input feature representations of shape (N, C).

        Returns:
            np.ndarray: Predicted continuous targets of shape (N,).
        """
        if self.model is None:
            raise RuntimeError("Regressor has not been fitted yet.")

        self.model.eval()
        X_t = torch.as_tensor(to_numpy(X), dtype=torch.float32).to(self.device)
        with torch.inference_mode():
            preds = self.model(X_t).cpu().numpy()
        return preds * self.y_std + self.y_mean


class _TorchTransformerRegressor(nn.Module):
    """Non-causal Transformer Encoder regressor.

    Args:
        in_dim: Input representation feature dimension per frame.
        d_model: Embedding dimension of the transformer encoder.
        nhead: Number of attention heads in multi-head attention.
        num_layers: Number of transformer encoder layers.
        dim_feedforward: Dimension of the feedforward network model.
        dropout: Dropout probability.
        aggregation: Aggregation strategy over time ('cls' token or 'mean' pooling).
    """

    def __init__(
        self,
        in_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        aggregation: str = "cls",
    ):
        super().__init__()
        self.aggregation = aggregation
        self.d_model = d_model
        if in_dim != d_model:
            self.proj = nn.Linear(in_dim, d_model)
        else:
            self.proj = nn.Identity()

        if aggregation == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        else:
            self.cls_token = None

        self.pos_encoder = PositionalEncoding(d_model=d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input sequence tensor of shape (B, T, in_dim).

        Returns:
            torch.Tensor: Predicted continuous values of shape (B,).
        """
        # x is (B, T, in_dim)
        h = self.proj(x)
        if self.aggregation == "cls":
            cls = self.cls_token.expand(x.size(0), -1, -1)
            h = torch.cat([cls, h], dim=1)
        h = self.pos_encoder(h)
        out = self.encoder(h)
        rep = out[:, 0] if self.aggregation == "cls" else out.mean(dim=1)
        return self.head(rep).squeeze(-1)


class TransformerBenchmarkRegressor(BaseRegressor):
    """Non-causal Transformer downstream regressor.

    Attends bidirectional across all features in the representation sequence
    without pooling before the transformer.

    Args:
        d_model: Transformer hidden dimension (must be divisible by nhead).
        nhead: Number of parallel attention heads.
        num_layers: Number of transformer encoder layers.
        dim_feedforward: Dimension of the feedforward network layer.
        dropout: Dropout probability.
        aggregation: Sequence pooling method ('cls' or 'mean').
        lr: Learning rate for Adam optimizer.
        epochs: Number of training epochs.
        batch_size: Batch size for training DataLoader.
        weight_decay: L2 penalty for Adam optimizer.
        device: Device to use for training and inference ('auto', 'cpu', 'cuda').
    """

    accepts_sequence: bool = True

    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        aggregation: str = "cls",
        lr: float = 1e-3,
        epochs: int = 30,
        batch_size: int = 16,
        weight_decay: float = 1e-4,
        device: str = "auto",
        show_progress: bool = True,
    ):
        if d_model % nhead != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by nhead ({nhead})."
            )
        if aggregation not in ("cls", "mean"):
            raise ValueError(
                f"aggregation must be 'cls' or 'mean', got '{aggregation}'."
            )

        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.aggregation = aggregation
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.device_str = device
        self.device = resolve_device(device)
        self.show_progress = show_progress
        self.model: _TorchTransformerRegressor | None = None
        self.y_mean: float = 0.0
        self.y_std: float = 1.0

    def fit(
        self, X: np.ndarray | torch.Tensor, y: np.ndarray | torch.Tensor
    ) -> "TransformerBenchmarkRegressor":
        """Train the transformer regressor on representation sequences X and targets y.

        Args:
            X: Input sequence representations of shape (N, T', C') or (N, C').
            y: Target continuous values of shape (N,).

        Returns:
            TransformerBenchmarkRegressor: Fitted regressor instance.
        """
        X_t = torch.as_tensor(to_numpy(X), dtype=torch.float32)
        if X_t.ndim == 2:
            X_t = X_t.unsqueeze(1)
        y_t = torch.as_tensor(to_numpy(y), dtype=torch.float32).ravel()

        self.y_mean = float(y_t.mean().item())
        self.y_std = float(y_t.std().item()) if float(y_t.std().item()) > 1e-8 else 1.0
        y_norm = (y_t - self.y_mean) / self.y_std

        in_dim = X_t.shape[-1]
        self.model = _TorchTransformerRegressor(
            in_dim=in_dim,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            aggregation=self.aggregation,
        ).to(self.device)

        dataset = torch.utils.data.TensorDataset(X_t, y_norm)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        criterion = nn.MSELoss()

        self.model.train()
        epoch_iter = tqdm(
            range(self.epochs),
            desc="Training TransformerRegressor",
            disable=not self.show_progress,
            leave=False,
        )
        for _ in epoch_iter:
            running_loss = 0.0
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                preds = self.model(batch_x)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += float(loss.item())
            epoch_iter.set_postfix(
                {"loss": f"{running_loss / max(1, len(loader)):.4f}"}
            )

        return self

    def predict(self, X: np.ndarray | torch.Tensor) -> np.ndarray:
        """Predict continuous targets for representations X.

        Args:
            X: Input sequence representations of shape (N, T', C') or (N, C').

        Returns:
            np.ndarray: Predicted continuous targets of shape (N,).
        """
        if self.model is None:
            raise RuntimeError("Regressor has not been fitted yet.")

        self.model.eval()
        X_t = torch.as_tensor(to_numpy(X), dtype=torch.float32)
        if X_t.ndim == 2:
            X_t = X_t.unsqueeze(1)
        X_t = X_t.to(self.device)

        with torch.inference_mode():
            preds = self.model(X_t).cpu().numpy()
        return preds * self.y_std + self.y_mean


def create_regressor(regressor_type: str | BaseRegressor, **kwargs) -> BaseRegressor:
    """Factory to create a downstream regressor by name or return an existing instance.

    Supported types: 'linear', 'knn', 'mlp', 'transformer', 'bayes_opt'.

    Args:
        regressor_type: Regressor identifier name or an existing BaseRegressor instance.
        **kwargs: Hyperparameters forwarded to the regressor constructor.

    Returns:
        BaseRegressor: Instantiated downstream regressor.
    """
    if isinstance(regressor_type, BaseRegressor):
        return regressor_type

    normalized = str(regressor_type).lower().strip()
    if normalized == "linear":
        return LinearBenchmarkRegressor(**kwargs)
    elif normalized == "knn":
        return KNNBenchmarkRegressor(**kwargs)
    elif normalized == "mlp":
        return MLPBenchmarkRegressor(**kwargs)
    elif normalized == "transformer":
        return TransformerBenchmarkRegressor(**kwargs)
    elif normalized in ("bayes_opt", "bayesopt", "bayesian_optimization"):
        from audio_benchmark.bayes_opt import BayesOptRegressor

        return BayesOptRegressor(**kwargs)
    else:
        raise ValueError(
            f"Unsupported regressor type: '{regressor_type}'. Available options: 'linear', 'knn', 'mlp', 'transformer', 'bayes_opt'."
        )
