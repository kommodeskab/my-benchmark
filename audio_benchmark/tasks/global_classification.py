"""Global audio classification task."""

import time
from collections import Counter
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
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
        pooling: str = "mean",
    ):
        """Initialize the GlobalClassificationTask.

        Args:
            classifier: Downstream classifier instance conforming to BaseClassifier.
            name: Identifier for the task.
            label_key: Key in the sample metadata dictionary containing the target label.
            pooling: Temporal pooling strategy: 'mean', 'max', 'first', 'last'.

        Raises:
            TypeError: If classifier is not an instance of BaseClassifier.
        """
        super().__init__(name=name)
        if not isinstance(classifier, BaseClassifier):
            raise TypeError(
                f"classifier must be an instantiated BaseClassifier instance, got {type(classifier).__name__}."
            )

        self.classifier = classifier
        self.label_key = label_key
        self.pooling = pooling
        self.metric = "accuracy"
        self.m_min = 0.0
        self.m_max = 1.0

    def validate_dataset(self, dataset: Dataset) -> None:
        """Verify that the dataset metadata contains the expected label_key."""
        validate_dataset(dataset, required_metadata_keys=[self.label_key])

    def pool_representations(self, representations: torch.Tensor) -> torch.Tensor:
        """Pool sequence representations (N, T', C') into clip-level features (N, C')."""
        return self.apply_pooling(representations, self.pooling)

    def _extract_split_labels(
        self,
        representations: Any,
        metadata: list[dict[str, Any]],
        split_name: str,
    ) -> list[Any]:
        """Validate split lengths and extract labels from metadata."""
        if len(representations) != len(metadata):
            raise TaskError(
                f"Length mismatch: {len(representations)} {split_name} representations vs "
                f"{len(metadata)} {split_name} metadata entries."
            )
        if len(representations) == 0:
            raise TaskError(f"No {split_name} representations provided for evaluation.")

        labels: list[Any] = []
        for idx, meta in enumerate(metadata):
            if self.label_key not in meta:
                raise InvalidDatasetError(
                    f"{split_name.capitalize()} sample at index {idx} metadata does not contain label key '{self.label_key}'."
                )
            labels.append(meta[self.label_key])
        return labels

    def evaluate(
        self,
        train_representations: torch.Tensor,
        train_metadata: list[dict[str, Any]],
        test_representations: torch.Tensor,
        test_metadata: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Train downstream classifier on pooled representations and return evaluation metrics.

        Args:
            train_representations: Computed train representations of shape (N_train, T', C').
            train_metadata: List of metadata dicts corresponding to each train sample.
            test_representations: Computed test representations of shape (N_test, T', C').
            test_metadata: List of metadata dicts corresponding to each test sample.

        Returns:
            dict[str, Any]: Dictionary of evaluation results.
        """
        raw_train_labels = self._extract_split_labels(
            train_representations, train_metadata, "train"
        )
        raw_test_labels = self._extract_split_labels(
            test_representations, test_metadata, "test"
        )

        encoder = LabelEncoder()
        encoder.fit(raw_train_labels + raw_test_labels)
        num_classes = len(encoder.classes_)
        if num_classes < 2:
            raise TaskError(
                f"Global classification requires at least 2 distinct classes, but found {num_classes}."
            )

        y_train = torch.as_tensor(encoder.transform(raw_train_labels), dtype=torch.long)
        y_test = torch.as_tensor(encoder.transform(raw_test_labels), dtype=torch.long)

        if getattr(self.classifier, "accepts_sequence", False):
            X_train = train_representations
            X_test = test_representations
            pooling_name = "none (transformer)"
        else:
            X_train = self.pool_representations(train_representations)
            X_test = self.pool_representations(test_representations)
            pooling_name = self.pooling

        start_time = time.perf_counter()
        self.classifier.fit(X_train, y_train)
        training_time = float(time.perf_counter() - start_time)

        metrics = self.classifier.evaluate(X_test, y_test)
        train_metrics = self.classifier.evaluate(X_train, y_train)
        value = float(metrics[self.metric])

        return {
            "task": self.name,
            "classifier": self.classifier.__class__.__name__,
            "pooling": pooling_name,
            "value": value,
            "normalized_value": self.normalize_metric(value, self.m_min, self.m_max),
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "micro_f1": metrics["micro_f1"],
            "train_accuracy": train_metrics["accuracy"],
            "training_time": training_time,
        }

    def examine_data(
        self,
        dataset: Dataset,
        testset: Dataset | None = None,
    ) -> dict[str, Any]:
        """Examine the dataset in the context of global classification.

        Validates the datasets, checks class distributions across train and test sets,
        and computes a trivial baseline (majority class guessing) expected metric.

        Args:
            dataset: Audio dataset to examine for training.
            testset: Optional audio dataset to examine for testing.

        Returns:
            dict[str, Any]: Dictionary containing dataset metadata and baseline metrics.
        """
        if not isinstance(dataset, Dataset):
            raise TypeError(
                f"dataset must be an instance of Dataset, got {type(dataset).__name__}."
            )
        if testset is not None and not isinstance(testset, Dataset):
            raise TypeError(
                f"testset must be an instance of Dataset, got {type(testset).__name__}."
            )
        self.validate_dataset(dataset)
        if testset is not None:
            self.validate_dataset(testset)

        train_metadata = self.extract_metadata(dataset)
        test_metadata = (
            self.extract_metadata(testset) if testset is not None else train_metadata
        )

        raw_train_labels = [m[self.label_key] for m in train_metadata]
        raw_test_labels = [m[self.label_key] for m in test_metadata]

        encoder = LabelEncoder()
        encoder.fit(raw_train_labels + raw_test_labels)
        classes = [str(c) for c in encoder.classes_]
        num_classes = len(classes)
        if num_classes < 2:
            raise TaskError(
                f"Global classification requires at least 2 distinct classes, but found {num_classes}."
            )

        train_counts = Counter(raw_train_labels)
        majority_raw = train_counts.most_common(1)[0][0]
        class_counts = {str(c): int(train_counts[c]) for c in encoder.classes_}

        y_test = encoder.transform(raw_test_labels)
        y_pred_baseline = np.full_like(y_test, encoder.transform([majority_raw])[0])

        baseline_accuracy = float(accuracy_score(y_test, y_pred_baseline))
        baseline_macro_f1 = float(
            f1_score(y_test, y_pred_baseline, average="macro", zero_division=0)
        )
        baseline_micro_f1 = float(
            f1_score(y_test, y_pred_baseline, average="micro", zero_division=0)
        )

        return {
            "task": self.name,
            "total_samples": len(train_metadata)
            + (len(test_metadata) if testset is not None else 0),
            "train_samples": len(train_metadata),
            "test_samples": len(test_metadata) if testset is not None else 0,
            "num_classes": int(num_classes),
            "classes": classes,
            "class_counts": class_counts,
            "majority_class": str(majority_raw),
            "baseline_strategy": "majority_class",
            "baseline_metric": self.metric,
            "baseline_value": baseline_accuracy,
            "baseline_normalized_value": self.normalize_metric(
                baseline_accuracy, self.m_min, self.m_max
            ),
            "baseline_accuracy": baseline_accuracy,
            "baseline_macro_f1": baseline_macro_f1,
            "baseline_micro_f1": baseline_micro_f1,
            "chance_level": float(1.0 / num_classes),
        }
