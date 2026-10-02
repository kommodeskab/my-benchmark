"""Global audio classification task."""

import time
from collections import Counter
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

from audio_benchmark.classifiers.models import BaseClassifier
from audio_benchmark.dataset import validate_dataset
from audio_benchmark.exceptions import InvalidDatasetError, TaskError
from audio_benchmark.tasks.base import BaseTask


class GlobalClassificationTask(BaseTask):
    """Global classification task: classifying a single label for the entire audio clip.

    The label is extracted from the metadata dictionary of each sample.
    The sequence representations (N, T', C') are pooled over time to produce
    clip-level features (N, C'), which are then classified using an instantiated BaseClassifier.
    """

    def __init__(
        self,
        classifier: BaseClassifier,
        name: str = "global_classification",
        label_key: str = "label",
        pooling: str | Callable[[torch.Tensor], torch.Tensor] = "mean",
        test_size: float = 0.2,
        random_state: int = 42,
        split_key: str | None = None,
        metric: str = "accuracy",
    ):
        """Initialize the GlobalClassificationTask.

        Args:
            classifier: Downstream classifier instance conforming to BaseClassifier.
            name: Identifier for the task.
            label_key: Key in the sample metadata dictionary containing the target label.
            pooling: Temporal pooling strategy: 'mean', 'max', 'first', 'last',
                or a callable f(representations) -> pooled_tensor.
            test_size: Proportion of samples to use for test split if split_key is not used.
            random_state: Random seed for train/test split.
            split_key: Optional metadata key (e.g. 'split') specifying 'train' or 'test'.
            metric: Primary metric for the task ('accuracy', 'macro_f1', 'micro_f1').
        """
        super().__init__(name=name)
        valid_metrics = ("accuracy", "macro_f1", "micro_f1")
        if metric not in valid_metrics:
            raise ValueError(
                f"Unknown metric '{metric}'. Supported metrics: {valid_metrics}"
            )
        self.metric = metric
        self.label_key = label_key
        if classifier is None:
            raise TypeError(
                "classifier must be an instantiated BaseClassifier instance, got None. "
                "Passing None as a model is not allowed."
            )
        elif isinstance(classifier, str):
            raise TypeError(
                f"Passing strings as models is not supported (got '{classifier}'). "
                "Please pass an instantiated BaseClassifier instance."
            )
        elif not isinstance(classifier, BaseClassifier):
            raise TypeError(
                f"classifier must be an instantiated BaseClassifier instance, got {type(classifier).__name__}."
            )
        self.classifier = classifier
        self.pooling = pooling
        self.test_size = test_size
        self.random_state = random_state
        self.split_key = split_key

    def validate_dataset(self, dataset: Dataset) -> None:
        """Verify that the dataset metadata contains the expected label_key."""
        validate_dataset(dataset, required_metadata_keys=[self.label_key])

    def pool_representations(self, representations: torch.Tensor) -> np.ndarray:
        """Pool sequence representations (N, T', C') into clip-level features (N, C')."""
        return self.apply_pooling(representations, self.pooling)

    def evaluate(
        self,
        representations: torch.Tensor,
        metadata: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Train downstream classifier on pooled representations and return evaluation metrics.

        Args:
            representations: Computed representations of shape (N, T', C').
            metadata: List of metadata dicts corresponding to each sample.

        Returns:
            dict[str, Any]: Dictionary of evaluation results.
        """
        if len(representations) != len(metadata):
            raise TaskError(
                f"Length mismatch: {len(representations)} representations vs {len(metadata)} metadata entries."
            )

        if len(representations) == 0:
            raise TaskError("No representations provided for evaluation.")

        # Extract labels
        raw_labels = []
        for i, meta in enumerate(metadata):
            if self.label_key not in meta:
                raise InvalidDatasetError(
                    f"Sample at index {i} metadata does not contain label key '{self.label_key}'."
                )
            raw_labels.append(meta[self.label_key])

        # Encode labels to 0..C-1
        encoder = LabelEncoder()
        encoded_labels = encoder.fit_transform(raw_labels)
        num_classes = len(encoder.classes_)

        if num_classes < 2:
            raise TaskError(
                f"Global classification requires at least 2 distinct classes, but found {num_classes}."
            )

        # Classifier instance
        classifier_inst = self.classifier
        classifier_name = classifier_inst.__class__.__name__

        is_sequence_probe = getattr(classifier_inst, "accepts_sequence", False)
        if is_sequence_probe:
            features = representations.detach().cpu().numpy()
            pooling_name = "none (transformer)"
        else:
            features = self.pool_representations(representations)
            pooling_name = self.pooling if isinstance(self.pooling, str) else "custom"

        # Handle train/test split
        if self.split_key is not None and all(self.split_key in m for m in metadata):
            train_mask = [m[self.split_key] == "train" for m in metadata]
            test_mask = [m[self.split_key] == "test" for m in metadata]

            X_train, y_train = features[train_mask], encoded_labels[train_mask]
            X_test, y_test = features[test_mask], encoded_labels[test_mask]

            if len(X_train) == 0 or len(X_test) == 0:
                raise TaskError(
                    f"Invalid train/test split via split_key '{self.split_key}': "
                    f"{len(X_train)} train samples, {len(X_test)} test samples."
                )
        else:
            # Automatic train/test split
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    features,
                    encoded_labels,
                    test_size=self.test_size,
                    random_state=self.random_state,
                    stratify=encoded_labels,
                )
            except ValueError:
                # If some class has only 1 sample, stratified split fails; fallback to unstratified
                X_train, X_test, y_train, y_test = train_test_split(
                    features,
                    encoded_labels,
                    test_size=self.test_size,
                    random_state=self.random_state,
                )

        # Fit classifier with timing
        start_time = time.perf_counter()
        classifier_inst.fit(X_train, y_train)
        training_time = float(time.perf_counter() - start_time)

        # Evaluate on test set
        metrics = classifier_inst.evaluate(X_test, y_test)

        # Also evaluate on train set for reference
        train_metrics = classifier_inst.evaluate(X_train, y_train)

        results = {
            "task": self.name,
            "classifier": classifier_name,
            "pooling": pooling_name,
            "value": float(metrics[self.metric]),
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "micro_f1": metrics["micro_f1"],
            "train_accuracy": train_metrics["accuracy"],
            "training_time": training_time,
        }
        if hasattr(classifier_inst, "best_params") and classifier_inst.best_params:
            results["best_params"] = classifier_inst.best_params
        if (
            hasattr(classifier_inst, "best_score")
            and classifier_inst.best_score is not None
        ):
            results["best_cv_score"] = classifier_inst.best_score

        return results

    def examine_data(self, dataset: Dataset) -> dict[str, Any]:
        """Examine the dataset in the context of global classification.

        Validates the dataset, checks class distributions and train/test splits,
        and computes a trivial baseline (majority class guessing) expected metric.

        Args:
            dataset: Audio dataset to examine.

        Returns:
            dict[str, Any]: Dictionary containing dataset metadata and baseline metrics.
        """
        self._validate_for_examination(dataset, [self.label_key])
        metadata = self.extract_metadata(dataset)

        if len(metadata) == 0:
            raise TaskError("Cannot examine an empty dataset.")

        raw_labels = []
        for i, meta in enumerate(metadata):
            if self.label_key not in meta:
                raise InvalidDatasetError(
                    f"Sample at index {i} metadata does not contain label key '{self.label_key}'."
                )
            raw_labels.append(meta[self.label_key])

        encoder = LabelEncoder()
        encoded_labels = encoder.fit_transform(raw_labels)
        classes = [str(c) for c in encoder.classes_]
        num_classes = len(classes)

        if num_classes < 2:
            raise TaskError(
                f"Global classification requires at least 2 distinct classes, but found {num_classes}."
            )

        indices = np.arange(len(raw_labels))
        if self.split_key is not None and all(self.split_key in m for m in metadata):
            train_mask = np.array([m[self.split_key] == "train" for m in metadata])
            test_mask = np.array([m[self.split_key] == "test" for m in metadata])
            train_indices = indices[train_mask]
            test_indices = indices[test_mask]
            if len(train_indices) == 0 or len(test_indices) == 0:
                raise TaskError(
                    f"Invalid train/test split via split_key '{self.split_key}': "
                    f"{len(train_indices)} train samples, {len(test_indices)} test samples."
                )
        else:
            try:
                train_indices, test_indices = train_test_split(
                    indices,
                    test_size=self.test_size,
                    random_state=self.random_state,
                    stratify=encoded_labels,
                )
            except ValueError:
                train_indices, test_indices = train_test_split(
                    indices,
                    test_size=self.test_size,
                    random_state=self.random_state,
                )

        # Class counts
        all_counts = Counter(raw_labels)
        class_counts = {str(c): int(all_counts[c]) for c in encoder.classes_}

        # Baseline predictor: predict majority class from train split on test split
        y_train = encoded_labels[train_indices]
        y_test = encoded_labels[test_indices]

        train_counts = Counter(y_train)
        majority_encoded = train_counts.most_common(1)[0][0]
        majority_class = str(encoder.classes_[majority_encoded])

        y_pred_baseline = np.full_like(y_test, majority_encoded)
        baseline_acc = float(accuracy_score(y_test, y_pred_baseline))
        baseline_macro_f1 = float(
            f1_score(y_test, y_pred_baseline, average="macro", zero_division=0)
        )
        baseline_micro_f1 = float(
            f1_score(y_test, y_pred_baseline, average="micro", zero_division=0)
        )

        if self.metric == "accuracy":
            baseline_value = baseline_acc
        elif self.metric == "macro_f1":
            baseline_value = baseline_macro_f1
        elif self.metric == "micro_f1":
            baseline_value = baseline_micro_f1
        else:
            baseline_value = baseline_acc

        return {
            "task": self.name,
            "total_samples": len(dataset),
            "train_samples": len(train_indices),
            "test_samples": len(test_indices),
            "num_classes": int(num_classes),
            "classes": classes,
            "class_counts": class_counts,
            "majority_class": majority_class,
            "baseline_strategy": "majority_class",
            "baseline_metric": self.metric,
            "baseline_value": float(baseline_value),
            "baseline_accuracy": baseline_acc,
            "baseline_macro_f1": baseline_macro_f1,
            "baseline_micro_f1": baseline_micro_f1,
            "chance_level": float(1.0 / num_classes) if num_classes > 0 else 0.0,
        }
