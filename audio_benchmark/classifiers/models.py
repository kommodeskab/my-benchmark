"""Downstream classification models for audio representation benchmarking."""

from abc import ABC, abstractmethod

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from torch import nn

from audio_benchmark.utils import (
    PositionalEncoding,
    resolve_device,
    train_torch_model,
)


class BaseClassifier(ABC):
    """Abstract base class for downstream benchmark classifiers."""

    @abstractmethod
    def fit(self, X: torch.Tensor, y: torch.Tensor) -> "BaseClassifier":
        """Train the classifier on representation features X and labels y.

        Args:
            X: Input feature representations of shape (N, C) or (N, T', C').
            y: Target class labels of shape (N,).

        Returns:
            BaseClassifier: Fitted classifier instance.
        """

    @abstractmethod
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Predict class labels for features X.

        Args:
            X: Input feature representations of shape (N, C) or (N, T', C').

        Returns:
            torch.Tensor: Predicted class labels of shape (N,).
        """

    def evaluate(self, X_test: torch.Tensor, y_test: torch.Tensor) -> dict[str, float]:
        """Evaluate predictions against ground truth labels.

        Args:
            X_test: Test feature representations.
            y_test: Ground truth class labels.

        Returns:
            dict[str, float]: Dictionary containing 'accuracy', 'macro_f1', and 'micro_f1'.
        """
        y_test_np = y_test.detach().cpu().numpy()
        y_pred = self.predict(X_test).detach().cpu().numpy()

        acc = float(accuracy_score(y_test_np, y_pred))
        macro_f1 = float(f1_score(y_test_np, y_pred, average="macro", zero_division=0))
        micro_f1 = float(f1_score(y_test_np, y_pred, average="micro", zero_division=0))

        return {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "micro_f1": micro_f1,
        }


class LinearBenchmarkClassifier(BaseClassifier):
    """Linear downstream classifier (Logistic Regression or Ridge Classifier).

    Args:
        C: Inverse regularization strength for logistic regression (smaller values specify stronger regularization).
        max_iter: Maximum number of solver iterations.
        random_state: Seed used by the random number generator.
        model_type: Linear model variant: 'logistic' (default) or 'ridge'.
        normalize_features: Whether to standardize input features to zero mean and unit variance.
    """

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        random_state: int | None = 42,
        model_type: str = "logistic",
        normalize_features: bool = True,
    ):
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self.normalize_features = normalize_features
        self.scaler: StandardScaler | None = None

        norm_type = str(model_type).lower().strip()
        if norm_type == "ridge":
            self.model_type = "ridge"
            self.model = RidgeClassifier(random_state=random_state)
        elif norm_type == "logistic":
            self.model_type = "logistic"
            self.model = LogisticRegression(
                C=C, max_iter=max_iter, random_state=random_state
            )
        else:
            raise ValueError(
                f"Unknown model_type: '{model_type}'. Supported: 'logistic', 'ridge'."
            )

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> "LinearBenchmarkClassifier":
        """Fit the linear classifier on features X and labels y.

        Args:
            X: Input feature matrix of shape (N, C).
            y: Target class labels of shape (N,).

        Returns:
            LinearBenchmarkClassifier: Fitted classifier instance.
        """
        X_np = X.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy().ravel()
        if self.normalize_features:
            self.scaler = StandardScaler()
            X_np = self.scaler.fit_transform(X_np).astype(np.float32)
        else:
            self.scaler = None
        self.model.fit(X_np, y_np)
        return self

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Predict class labels for features X.

        Args:
            X: Input feature matrix of shape (N, C).

        Returns:
            torch.Tensor: Predicted class labels of shape (N,).
        """
        X_np = X.detach().cpu().numpy()
        if self.scaler is not None:
            X_np = self.scaler.transform(X_np).astype(np.float32)
        return torch.as_tensor(self.model.predict(X_np))

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        """Predict class probabilities for features X (when supported, e.g. LogisticRegression).

        Args:
            X: Input feature matrix of shape (N, C).

        Returns:
            torch.Tensor: Predicted class probabilities of shape (N, num_classes).
        """
        if hasattr(self.model, "predict_proba"):
            X_np = X.detach().cpu().numpy()
            if self.scaler is not None:
                X_np = self.scaler.transform(X_np).astype(np.float32)
            return torch.as_tensor(self.model.predict_proba(X_np))
        raise AttributeError(
            f"{self.model.__class__.__name__} does not support predict_proba."
        )


class KNNBenchmarkClassifier(BaseClassifier):
    """K-Nearest Neighbors downstream classifier.

    Args:
        n_neighbors: Number of nearest neighbors to query.
        metric: Distance metric to use for neighbor search (default: 'minkowski').
        weights: Weight function used in prediction ('uniform' or 'distance').
        normalize_features: Whether to standardize input features to zero mean and unit variance.
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        metric: str = "minkowski",
        weights: str = "uniform",
        normalize_features: bool = True,
    ):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.weights = weights
        self.normalize_features = normalize_features
        self.scaler: StandardScaler | None = None
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors, metric=metric, weights=weights
        )

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> "KNNBenchmarkClassifier":
        """Fit the KNN classifier on features X and labels y.

        Args:
            X: Input feature matrix of shape (N, C).
            y: Target class labels of shape (N,).

        Returns:
            KNNBenchmarkClassifier: Fitted classifier instance.
        """
        X_np = X.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        if self.normalize_features:
            self.scaler = StandardScaler()
            X_np = self.scaler.fit_transform(X_np).astype(np.float32)
        else:
            self.scaler = None
        # Adjust n_neighbors if dataset has fewer samples than n_neighbors
        effective_k = min(self.n_neighbors, len(X_np))
        if effective_k != self.n_neighbors:
            self.model = KNeighborsClassifier(
                n_neighbors=effective_k, metric=self.metric, weights=self.weights
            )
        self.model.fit(X_np, y_np)
        return self

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Predict class labels for features X.

        Args:
            X: Input feature matrix of shape (N, C).

        Returns:
            torch.Tensor: Predicted class labels of shape (N,).
        """
        X_np = X.detach().cpu().numpy()
        if self.scaler is not None:
            X_np = self.scaler.transform(X_np).astype(np.float32)
        return torch.as_tensor(self.model.predict(X_np))


class _TorchMLP(nn.Module):
    """Multi-Layer Perceptron for classification.

    Args:
        in_dim: Input feature dimension.
        hidden_dim: Number of hidden units per layer (default: 1024).
        num_classes: Number of target output classes.
        dropout: Dropout rate applied after each hidden layer.
        num_layers: Number of hidden layers (default: 1).
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 1024,
        num_classes: int = 2,
        dropout: float = 0.1,
        num_layers: int = 1,
    ):
        super().__init__()
        if num_layers < 0:
            raise ValueError(f"num_layers must be non-negative, got {num_layers}.")
        self.num_layers = num_layers
        layers: list[nn.Module] = []
        cur_dim = in_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(cur_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            cur_dim = hidden_dim
        layers.append(nn.Linear(cur_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, in_dim).

        Returns:
            torch.Tensor: Class logits of shape (B, num_classes).
        """
        return self.net(x)


class MLPBenchmarkClassifier(BaseClassifier):
    """Multi-Layer Perceptron downstream classifier.

    Args:
        hidden_dim: Number of units in each hidden layer (default: 1024).
        num_layers: Number of hidden layers (default: 1).
        dropout: Dropout probability applied after each hidden layer.
        lr: Learning rate for Adam optimizer.
        epochs: Number of training epochs.
        batch_size: Batch size for training DataLoader.
        weight_decay: L2 penalty for Adam optimizer.
        device: Device to use for training and inference ('auto', 'cpu', 'cuda').
        show_progress: Whether to display a tqdm progress bar during training.
        normalize_features: Whether to standardize input features to zero mean and unit variance.
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        num_layers: int = 1,
        dropout: float = 0.1,
        lr: float = 1e-3,
        epochs: int = 30,
        batch_size: int = 32,
        weight_decay: float = 1e-4,
        device: str = "auto",
        show_progress: bool = True,
        normalize_features: bool = True,
    ):
        if num_layers < 0:
            raise ValueError(f"num_layers must be non-negative, got {num_layers}.")
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.device_str = device
        self.show_progress = show_progress
        self.normalize_features = normalize_features
        self.scaler: StandardScaler | None = None
        self.model: _TorchMLP | None = None
        self.device = resolve_device(device)

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> "MLPBenchmarkClassifier":
        """Train the MLP classifier on features X and labels y.

        Args:
            X: Input feature representations of shape (N, C).
            y: Target class labels of shape (N,).

        Returns:
            MLPBenchmarkClassifier: Fitted classifier instance.
        """
        if self.normalize_features:
            self.scaler = StandardScaler()
            X_np = self.scaler.fit_transform(X.detach().cpu().numpy()).astype(
                np.float32
            )
            X_t = torch.as_tensor(X_np, dtype=torch.float32)
        else:
            self.scaler = None
            X_t = X.detach().to(dtype=torch.float32)

        y_t = y.detach().to(dtype=torch.long)

        in_dim = X_t.shape[1]
        num_classes = int(y_t.max().item()) + 1

        self.model = _TorchMLP(
            in_dim=in_dim,
            hidden_dim=self.hidden_dim,
            num_classes=num_classes,
            dropout=self.dropout,
            num_layers=self.num_layers,
        ).to(self.device)

        train_torch_model(
            model=self.model,
            X=X_t,
            y=y_t,
            criterion=nn.CrossEntropyLoss(),
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            weight_decay=self.weight_decay,
            device=self.device,
            show_progress=self.show_progress,
            desc="Training MLPClassifier",
        )
        return self

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Predict class labels for features X.

        Args:
            X: Input feature representations of shape (N, C).

        Returns:
            torch.Tensor: Predicted class labels of shape (N,).
        """
        if self.model is None:
            raise RuntimeError("Classifier has not been fitted yet.")

        if self.scaler is not None:
            X_np = self.scaler.transform(X.detach().cpu().numpy()).astype(np.float32)
            X_t = torch.as_tensor(X_np, dtype=torch.float32).to(self.device)
        else:
            X_t = X.detach().to(device=self.device, dtype=torch.float32)

        self.model.eval()
        with torch.inference_mode():
            logits = self.model(X_t)
            preds = logits.argmax(dim=-1).cpu()
        return preds


class _TorchTransformerClassifier(nn.Module):
    """Non-causal Transformer Encoder classifier.

    Args:
        in_dim: Input representation feature dimension per frame.
        num_classes: Number of target output classes.
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
        """Forward pass.

        Args:
            x: Input sequence tensor of shape (B, T, in_dim).

        Returns:
            torch.Tensor: Class logits of shape (B, num_classes).
        """
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
        epochs: int = 25,
        batch_size: int = 16,
        weight_decay: float = 1e-4,
        device: str = "auto",
        show_progress: bool = True,
        normalize_features: bool = True,
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
        self.show_progress = show_progress
        self.normalize_features = normalize_features
        self.scaler: StandardScaler | None = None
        self.model: _TorchTransformerClassifier | None = None
        self.device = resolve_device(device)

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> "TransformerBenchmarkClassifier":
        """Train the transformer classifier on representation sequences X and labels y.

        Args:
            X: Input sequence representations of shape (N, T', C') or (N, C').
            y: Target class labels of shape (N,).

        Returns:
            TransformerBenchmarkClassifier: Fitted classifier instance.
        """
        if self.normalize_features:
            self.scaler = StandardScaler()
            X_np = X.detach().cpu().numpy()
            orig_shape = X_np.shape
            X_flat = X_np.reshape(-1, orig_shape[-1])
            X_np = (
                self.scaler.fit_transform(X_flat).astype(np.float32).reshape(orig_shape)
            )
            X_t = torch.as_tensor(X_np, dtype=torch.float32)
        else:
            self.scaler = None
            X_t = X.detach().to(dtype=torch.float32)

        if X_t.ndim == 2:
            X_t = X_t.unsqueeze(1)
        y_t = y.detach().to(dtype=torch.long)

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

        train_torch_model(
            model=self.model,
            X=X_t,
            y=y_t,
            criterion=nn.CrossEntropyLoss(),
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            weight_decay=self.weight_decay,
            device=self.device,
            show_progress=self.show_progress,
            desc="Training TransformerClassifier",
        )
        return self

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Predict class labels for representations X.

        Args:
            X: Input sequence representations of shape (N, T', C') or (N, C').

        Returns:
            torch.Tensor: Predicted class labels of shape (N,).
        """
        if self.model is None:
            raise RuntimeError("Classifier has not been fitted yet.")

        if self.scaler is not None:
            X_np = X.detach().cpu().numpy()
            orig_shape = X_np.shape
            X_flat = X_np.reshape(-1, orig_shape[-1])
            X_np = self.scaler.transform(X_flat).astype(np.float32).reshape(orig_shape)
            X_t = torch.as_tensor(X_np, dtype=torch.float32)
        else:
            X_t = X.detach().to(dtype=torch.float32)

        if X_t.ndim == 2:
            X_t = X_t.unsqueeze(1)
        X_t = X_t.to(self.device)

        self.model.eval()
        with torch.inference_mode():
            logits = self.model(X_t)
            preds = logits.argmax(dim=-1).cpu()
        return preds
