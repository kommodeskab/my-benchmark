"""Frame-level audio classification task."""

import time
from collections import Counter
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

from audio_benchmark.classifiers.models import BaseClassifier
from audio_benchmark.dataset import parse_dataset_item, validate_dataset
from audio_benchmark.exceptions import InvalidDatasetError, TaskError
from audio_benchmark.tasks.base import BaseTask
from audio_benchmark.tasks.downsampling import downsample_sequence


class FrameClassificationTask(BaseTask):
    """Frame-level audio classification task.

    Each frame of the input audio has a ground truth label in sample metadata.
    When representation models downsample audio along the temporal dimension
    from T to T', this task downsamples the frame labels to T' using the
    most common label in each downsampled region.
    """

    def __init__(
        self,
        classifier: BaseClassifier,
        name: str = "frame_classification",
        label_key: str = "frame_labels",
        metric: str = "accuracy",
    ):
        """Initialize the FrameClassificationTask.

        Args:
            classifier: Downstream classifier instance conforming to BaseClassifier.
            name: Identifier for the task.
            label_key: Metadata key containing the per-frame label sequence (length T).
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

    def validate_dataset(self, dataset: Dataset) -> None:
        """Verify that dataset metadata contains per-frame labels matching audio length T."""
        validate_dataset(dataset, required_metadata_keys=[self.label_key])

        parsed = parse_dataset_item(dataset[0])
        audio_length = parsed["waveform"].shape[-1]
        labels = parsed["metadata"][self.label_key]
        if not hasattr(labels, "__len__"):
            raise InvalidDatasetError(
                f"Metadata key '{self.label_key}' must be a sequence of length {audio_length}, "
                f"got {type(labels).__name__}."
            )
        if len(labels) != audio_length:
            raise InvalidDatasetError(
                f"Length mismatch in '{self.label_key}': audio length is {audio_length} frames, "
                f"but metadata '{self.label_key}' has {len(labels)} elements."
            )

    def evaluate(
        self,
        train_representations: torch.Tensor,
        train_metadata: list[dict[str, Any]],
        test_representations: torch.Tensor,
        test_metadata: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Downsample frame labels to T' and evaluate downstream classifier.

        Args:
            train_representations: Tensor of shape (N_train, T', C').
            train_metadata: List of metadata dicts corresponding to each train clip.
            test_representations: Tensor of shape (N_test, T', C').
            test_metadata: List of metadata dicts corresponding to each test clip.

        Returns:
            dict[str, Any]: Frame-level classification metrics.
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

        _, target_len_train, feature_dim = train_representations.shape
        _, target_len_test, _ = test_representations.shape

        train_labels_per_clip = []
        for idx, meta in enumerate(train_metadata):
            if self.label_key not in meta:
                raise InvalidDatasetError(
                    f"Train sample at index {idx} metadata does not contain '{self.label_key}'."
                )
            downsampled = downsample_sequence(
                meta[self.label_key], target_len=target_len_train, mode="mode"
            )
            train_labels_per_clip.append(downsampled)

        test_labels_per_clip = []
        for idx, meta in enumerate(test_metadata):
            if self.label_key not in meta:
                raise InvalidDatasetError(
                    f"Test sample at index {idx} metadata does not contain '{self.label_key}'."
                )
            downsampled = downsample_sequence(
                meta[self.label_key], target_len=target_len_test, mode="mode"
            )
            test_labels_per_clip.append(downsampled)

        train_flat = np.concatenate(train_labels_per_clip)
        test_flat = np.concatenate(test_labels_per_clip)
        all_labels = np.concatenate([train_flat, test_flat])

        encoder = LabelEncoder()
        encoder.fit(all_labels)
        num_classes = len(encoder.classes_)

        if num_classes < 2:
            raise TaskError(
                f"Frame classification requires at least 2 distinct classes, but found {num_classes}."
            )

        y_train = encoder.transform(train_flat)
        y_test = encoder.transform(test_flat)

        X_train = train_representations.detach().cpu().numpy().reshape(-1, feature_dim)
        X_test = test_representations.detach().cpu().numpy().reshape(-1, feature_dim)

        classifier_inst = self.classifier
        classifier_name = classifier_inst.__class__.__name__

        start_time = time.perf_counter()
        classifier_inst.fit(X_train, y_train)
        training_time = float(time.perf_counter() - start_time)

        metrics = classifier_inst.evaluate(X_test, y_test)
        train_metrics = classifier_inst.evaluate(X_train, y_train)

        results = {
            "task": self.name,
            "classifier": classifier_name,
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
        """Examine the dataset in the context of frame classification.

        Validates the datasets, computes clip/frame distributions, class counts,
        and evaluates a trivial baseline (majority frame class) on test frames.

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

        train_labels = [label for m in train_metadata for label in m[self.label_key]]
        test_labels = [label for m in test_metadata for label in m[self.label_key]]
        all_labels = train_labels + (test_labels if testset is not None else [])

        encoder = LabelEncoder()
        encoder.fit(all_labels)
        classes = [str(c) for c in encoder.classes_]
        num_classes = len(classes)

        if num_classes < 2:
            raise TaskError(
                f"Frame classification requires at least 2 distinct classes, but found {num_classes}."
            )

        frame_len_sample = (
            len(train_metadata[0][self.label_key]) if train_metadata else 0
        )
        class_counts = {
            str(c): int(count) for c, count in Counter(train_labels).items()
        }

        train_counts = Counter(train_labels)
        majority_raw = train_counts.most_common(1)[0][0]
        majority_class = str(majority_raw)

        y_test_encoded = encoder.transform(test_labels)
        majority_encoded = encoder.transform([majority_raw])[0]
        y_pred_baseline = np.full_like(y_test_encoded, majority_encoded)

        baseline_acc = float(accuracy_score(y_test_encoded, y_pred_baseline))
        baseline_macro_f1 = float(
            f1_score(y_test_encoded, y_pred_baseline, average="macro", zero_division=0)
        )
        baseline_micro_f1 = float(
            f1_score(y_test_encoded, y_pred_baseline, average="micro", zero_division=0)
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
            "total_clips": len(train_metadata)
            + (len(test_metadata) if testset is not None else 0),
            "train_clips": len(train_metadata),
            "test_clips": len(test_metadata) if testset is not None else 0,
            "frame_length": int(frame_len_sample),
            "total_frames": len(train_labels)
            + (len(test_labels) if testset is not None else 0),
            "train_frames": len(train_labels),
            "test_frames": len(test_labels) if testset is not None else 0,
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
