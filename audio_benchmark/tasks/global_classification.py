"""Global audio classification task."""

import time
from collections import Counter
from collections.abc import Callable
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
        pooling: str | Callable[[torch.Tensor], torch.Tensor] = "mean",
        metric: str = "accuracy",
    ):
        """Initialize the GlobalClassificationTask.

        Args:
            classifier: Downstream classifier instance conforming to BaseClassifier.
            name: Identifier for the task.
            label_key: Key in the sample metadata dictionary containing the target label.
            pooling: Temporal pooling strategy: 'mean', 'max', 'first', 'last',
                or a callable f(representations) -> pooled_tensor.
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

    def validate_dataset(self, dataset: Dataset) -> None:
        """Verify that the dataset metadata contains the expected label_key."""
        validate_dataset(dataset, required_metadata_keys=[self.label_key])

    def pool_representations(self, representations: torch.Tensor) -> np.ndarray:
        """Pool sequence representations (N, T', C') into clip-level features (N, C')."""
        return self.apply_pooling(representations, self.pooling)

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
        if len(train_representations) != len(train_metadata):
            raise TaskError(
                f"Length mismatch: {len(train_representations)} train representations vs {len(train_metadata)} train metadata entries."
            )
        if len(test_representations) != len(test_metadata):
            raise TaskError(
                f"Length mismatch: {len(test_representations)} test representations vs {len(test_metadata)} test metadata entries."
            )
        if len(train_representations) == 0:
            raise TaskError("No train representations provided for evaluation.")
        if len(test_representations) == 0:
            raise TaskError("No test representations provided for evaluation.")

        raw_train_labels = []
        for i, meta in enumerate(train_metadata):
            if self.label_key not in meta:
                raise InvalidDatasetError(
                    f"Train sample at index {i} metadata does not contain label key '{self.label_key}'."
                )
            raw_train_labels.append(meta[self.label_key])

        raw_test_labels = []
        for i, meta in enumerate(test_metadata):
            if self.label_key not in meta:
                raise InvalidDatasetError(
                    f"Test sample at index {i} metadata does not contain label key '{self.label_key}'."
                )
            raw_test_labels.append(meta[self.label_key])

        encoder = LabelEncoder()
        encoder.fit(list(raw_train_labels) + list(raw_test_labels))
        y_train = encoder.transform(raw_train_labels)
        y_test = encoder.transform(raw_test_labels)
        num_classes = len(encoder.classes_)

        if num_classes < 2:
            raise TaskError(
                f"Global classification requires at least 2 distinct classes, but found {num_classes}."
            )

        classifier_inst = self.classifier
        classifier_name = classifier_inst.__class__.__name__

        is_sequence_probe = getattr(classifier_inst, "accepts_sequence", False)
        if is_sequence_probe:
            X_train = train_representations.detach().cpu().numpy()
            X_test = test_representations.detach().cpu().numpy()
            pooling_name = "none (transformer)"
        else:
            X_train = self.pool_representations(train_representations)
            X_test = self.pool_representations(test_representations)
            pooling_name = self.pooling if isinstance(self.pooling, str) else "custom"

        start_time = time.perf_counter()
        classifier_inst.fit(X_train, y_train)
        training_time = float(time.perf_counter() - start_time)

        metrics = classifier_inst.evaluate(X_test, y_test)
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

    def examine_data(self, dataset: Any, testset: Any | None = None) -> dict[str, Any]:
        """Examine the dataset in the context of global classification.

        Validates the datasets, checks class distributions across train and test sets,
        and computes a trivial baseline (majority class guessing) expected metric.

        Args:
            dataset: Audio dataset to examine for training.
            testset: Optional audio dataset to examine for testing.

        Returns:
            dict[str, Any]: Dictionary containing dataset metadata and baseline metrics.
        """
        self._validate_for_examination(dataset, [self.label_key])
        train_metadata = self.extract_metadata(dataset)

        if testset is not None:
            self._validate_for_examination(testset, [self.label_key])
            test_metadata = self.extract_metadata(testset)
        else:
            test_metadata = train_metadata

        if len(train_metadata) == 0:
            raise TaskError("Cannot examine an empty dataset.")

        raw_train_labels = [m[self.label_key] for m in train_metadata]
        raw_test_labels = [m[self.label_key] for m in test_metadata]

        encoder = LabelEncoder()
        encoder.fit(list(raw_train_labels) + list(raw_test_labels))
        y_train = encoder.transform(raw_train_labels)
        y_test = encoder.transform(raw_test_labels)
        classes = [str(c) for c in encoder.classes_]
        num_classes = len(classes)

        if num_classes < 2:
            raise TaskError(
                f"Global classification requires at least 2 distinct classes, but found {num_classes}."
            )

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

        all_counts = Counter(raw_train_labels)
        class_counts = {str(c): int(all_counts[c]) for c in encoder.classes_}

        return {
            "task": self.name,
            "total_samples": len(train_metadata)
            + (len(test_metadata) if testset is not None else 0),
            "train_samples": len(train_metadata),
            "test_samples": len(test_metadata) if testset is not None else 0,
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
