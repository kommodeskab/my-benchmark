"""Downstream classification models for audio representation benchmarking."""

from abc import ABC, abstractmethod

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from torch import nn

from audio_benchmark.tasks.sequence_models import PositionalEncoding
from audio_benchmark.utils import resolve_device, to_numpy


class BaseClassifier(ABC):
    """Abstract base class for downstream benchmark classifiers."""

    @abstractmethod
    def fit(
        self, X: np.ndarray | torch.Tensor, y: np.ndarray | torch.Tensor
    ) -> "BaseClassifier":
        """Train the classifier on representation features X and labels y."""

    @abstractmethod
    def predict(self, X: np.ndarray | torch.Tensor) -> np.ndarray:
        """Predict class labels for features X."""

    def evaluate(
        self, X_test: np.ndarray | torch.Tensor, y_test: np.ndarray | torch.Tensor
    ) -> dict[str, float]:
        """Evaluate predictions against ground truth labels."""
        y_test_np = to_numpy(y_test)
        y_pred = self.predict(X_test)

        acc = float(accuracy_score(y_test_np, y_pred))
        macro_f1 = float(f1_score(y_test_np, y_pred, average="macro", zero_division=0))
        micro_f1 = float(f1_score(y_test_np, y_pred, average="micro", zero_division=0))

        return {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "micro_f1": micro_f1,
        }


class LinearBenchmarkClassifier(BaseClassifier):
    """Linear downstream classifier (Logistic Regression or Ridge Classifier)."""

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        random_state: int | None = 42,
        model_type: str = "logistic",
        alpha: float | None = None,
        **kwargs,
    ):
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self.alpha = alpha
        self.kwargs = kwargs

        if alpha is not None or str(model_type).lower() in (
            "ridge",
            "ridge_classifier",
        ):
            self.model_type = "ridge"
            ridge_alpha = alpha if alpha is not None else 1.0
            self.model = RidgeClassifier(
                alpha=ridge_alpha, random_state=random_state, **kwargs
            )
        else:
            self.model_type = "logistic"
            self.model = LogisticRegression(
                C=C, max_iter=max_iter, random_state=random_state, **kwargs
            )

    def fit(
        self, X: np.ndarray | torch.Tensor, y: np.ndarray | torch.Tensor
    ) -> "LinearBenchmarkClassifier":
        X_np = to_numpy(X)
        y_np = to_numpy(y).ravel()
        self.model.fit(X_np, y_np)
        return self

    def predict(self, X: np.ndarray | torch.Tensor) -> np.ndarray:
        X_np = to_numpy(X)
        return self.model.predict(X_np)

    def predict_proba(self, X: np.ndarray | torch.Tensor) -> np.ndarray:
        """Predict class probabilities for features X (when supported, e.g. LogisticRegression)."""
        if hasattr(self.model, "predict_proba"):
            X_np = to_numpy(X)
            return self.model.predict_proba(X_np)
        raise AttributeError(
            f"{self.model.__class__.__name__} does not support predict_proba."
        )


class KNNBenchmarkClassifier(BaseClassifier):
    """K-Nearest Neighbors downstream classifier."""

    def __init__(self, n_neighbors: int = 5, metric: str = "minkowski", **kwargs):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.kwargs = kwargs
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors, metric=metric, **kwargs
        )

    def fit(
        self, X: np.ndarray | torch.Tensor, y: np.ndarray | torch.Tensor
    ) -> "KNNBenchmarkClassifier":
        X_np = to_numpy(X)
        y_np = to_numpy(y)
        # Adjust n_neighbors if dataset has fewer samples than n_neighbors
        effective_k = min(self.n_neighbors, len(X_np))
        if effective_k != self.n_neighbors:
            self.model = KNeighborsClassifier(
                n_neighbors=effective_k, metric=self.metric, **self.kwargs
            )
        self.model.fit(X_np, y_np)
        return self

    def predict(self, X: np.ndarray | torch.Tensor) -> np.ndarray:
        X_np = to_numpy(X)
        return self.model.predict(X_np)


class _TorchMLP(nn.Module):
    """Simple 2-layer MLP."""

    def __init__(
        self, in_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.1
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPBenchmarkClassifier(BaseClassifier):
    """Multi-Layer Perceptron downstream classifier."""

    def __init__(
        self,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        lr: float = 1e-3,
        epochs: int = 30,
        batch_size: int = 32,
        weight_decay: float = 1e-4,
        device: str = "auto",
    ):
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.device_str = device
        self.model: _TorchMLP | None = None
        self.device = resolve_device(device)

    def fit(
        self, X: np.ndarray | torch.Tensor, y: np.ndarray | torch.Tensor
    ) -> "MLPBenchmarkClassifier":
        X_t = torch.as_tensor(to_numpy(X), dtype=torch.float32)
        y_t = torch.as_tensor(to_numpy(y), dtype=torch.long)

        in_dim = X_t.shape[1]
        num_classes = int(y_t.max().item()) + 1

        self.model = _TorchMLP(
            in_dim=in_dim,
            hidden_dim=self.hidden_dim,
            num_classes=num_classes,
            dropout=self.dropout,
        ).to(self.device)

        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for _ in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

        return self

    def predict(self, X: np.ndarray | torch.Tensor) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Classifier has not been fitted yet.")

        self.model.eval()
        X_t = torch.as_tensor(to_numpy(X), dtype=torch.float32).to(self.device)
        with torch.inference_mode():
            logits = self.model(X_t)
            preds = logits.argmax(dim=-1).cpu().numpy()
        return preds


class _TorchTransformerClassifier(nn.Module):
    """Non-causal Transformer Encoder classifier."""

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
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
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (B, T, in_dim)
        h = self.proj(x)
        if self.aggregation == "cls":
            cls = self.cls_token.expand(x.size(0), -1, -1)
            h = torch.cat([cls, h], dim=1)
        h = self.pos_encoder(h)
        out = self.encoder(h)
        rep = out[:, 0] if self.aggregation == "cls" else out.mean(dim=1)
        return self.head(rep)


class TransformerBenchmarkClassifier(BaseClassifier):
    """Non-causal Transformer downstream classifier.

    Attends bidirectional across all features in the representation sequence
    without pooling before the transformer.
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
        epochs: int = 25,
        batch_size: int = 16,
        weight_decay: float = 1e-4,
        device: str = "auto",
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
        self.model: _TorchTransformerClassifier | None = None
        self.device = resolve_device(device)

    def fit(
        self, X: np.ndarray | torch.Tensor, y: np.ndarray | torch.Tensor
    ) -> "TransformerBenchmarkClassifier":
        X_t = torch.as_tensor(to_numpy(X), dtype=torch.float32)
        if X_t.ndim == 2:
            X_t = X_t.unsqueeze(1)
        y_t = torch.as_tensor(to_numpy(y), dtype=torch.long)

        in_dim = X_t.shape[-1]
        num_classes = int(y_t.max().item()) + 1

        self.model = _TorchTransformerClassifier(
            in_dim=in_dim,
            num_classes=num_classes,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            aggregation=self.aggregation,
        ).to(self.device)

        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for _ in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

        return self

    def predict(self, X: np.ndarray | torch.Tensor) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Classifier has not been fitted yet.")

        self.model.eval()
        X_t = torch.as_tensor(to_numpy(X), dtype=torch.float32)
        if X_t.ndim == 2:
            X_t = X_t.unsqueeze(1)
        X_t = X_t.to(self.device)

        with torch.inference_mode():
            logits = self.model(X_t)
            preds = logits.argmax(dim=-1).cpu().numpy()
        return preds


def create_classifier(
    classifier_type: str | BaseClassifier, **kwargs
) -> BaseClassifier:
    """Factory to create a downstream classifier by name or return an existing instance.

    Supported types: 'linear' (or 'logistic', 'ridge'), 'knn', 'mlp', 'transformer'.
    """
    if isinstance(classifier_type, BaseClassifier):
        return classifier_type

    normalized = str(classifier_type).lower().strip()
    if normalized in ("linear", "logistic", "logistic_regression", "linear_classifier"):
        return LinearBenchmarkClassifier(**kwargs)
    elif normalized in ("ridge", "ridge_classifier"):
        return LinearBenchmarkClassifier(model_type="ridge", **kwargs)
    elif normalized in ("knn", "k_nearest_neighbors"):
        return KNNBenchmarkClassifier(**kwargs)
    elif normalized in ("mlp", "multilayer_perceptron"):
        return MLPBenchmarkClassifier(**kwargs)
    elif normalized in ("transformer", "transformer_encoder"):
        return TransformerBenchmarkClassifier(**kwargs)
    else:
        raise ValueError(
            f"Unsupported classifier type: '{classifier_type}'. Available options: 'linear', 'knn', 'mlp', 'transformer'."
        )
