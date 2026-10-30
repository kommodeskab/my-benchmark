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
from audio_benchmark.tasks.downsampling import (
    align_features_to_targets,
    parse_frame_targets,
)


class FrameClassificationTask(BaseTask):
    """Frame-level audio classification task.

    Evaluates frame-level audio classification by linearly interpolating representation
    features to match the sequence length of raw ground-truth targets.
    Raw ground-truth targets are never modified.
    """

    def __init__(
        self,
        classifier: BaseClassifier,
        name: str = "frame_classification",
        label_key: str = "frame_labels",
        metric: str = "accuracy",
        pooling: str = "linear",
    ):
        """Initialize the FrameClassificationTask.

        Args:
            classifier: Downstream classifier instance conforming to BaseClassifier.
            name: Identifier for the task.
            label_key: Metadata key containing per-frame labels (flat torch.Tensor or list).
            metric: Primary metric for the task ('accuracy', 'macro_f1', 'micro_f1').
            pooling: Temporal alignment strategy ('linear' interpolation).
        """
        super().__init__(name=name)
        valid_metrics = ("accuracy", "macro_f1", "micro_f1")
        if metric not in valid_metrics:
            raise ValueError(
                f"Unknown metric '{metric}'. Supported metrics: {valid_metrics}"
            )
        valid_pooling = ("linear",)
        if pooling not in valid_pooling:
            raise ValueError(
                f"Unknown pooling '{pooling}'. Supported pooling: {valid_pooling}"
            )

        self.metric = metric
        self.label_key = label_key
        self.pooling = pooling

        if not isinstance(classifier, BaseClassifier):
            raise TypeError(
                f"classifier must be an instantiated BaseClassifier instance, got {type(classifier).__name__}."
            )
        self.classifier = classifier

    def validate_dataset(self, dataset: Dataset) -> None:
        """Verify that dataset metadata contains valid per-frame labels."""
        validate_dataset(dataset, required_metadata_keys=[self.label_key])

        parsed = parse_dataset_item(dataset[0])
        labels = parsed["metadata"][self.label_key]
        parse_frame_targets(
            labels,
            key_name=self.label_key,
        )

    def evaluate(
        self,
        train_representations: torch.Tensor,
        train_metadata: list[dict[str, Any]],
        test_representations: torch.Tensor,
        test_metadata: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Align representation features to raw frame targets and evaluate classifier.

        Ground-truth targets are never downsampled or modified. Features are aligned
        to targets via 1D linear interpolation.

        Args:
            train_representations: Tensor of shape (N_train, T', C') or sequence of representations.
            train_metadata: List of metadata dicts corresponding to each train clip.
            test_representations: Tensor of shape (N_test, T', C') or sequence of representations.
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

        train_aligned: list[torch.Tensor] = []
        train_labels_list: list[list[Any]] = []
        for idx, meta in enumerate(train_metadata):
            if self.label_key not in meta:
                raise InvalidDatasetError(
                    f"Train sample at index {idx} metadata does not contain '{self.label_key}'."
                )
            raw_target = meta[self.label_key]
            raw_labels = parse_frame_targets(
                raw_target,
                key_name=self.label_key,
            )
            clip_reps = train_representations[idx]
            aligned = align_features_to_targets(
                representations=clip_reps,
                target_len=len(raw_labels),
            )
            train_aligned.append(aligned)
            train_labels_list.append(raw_labels)

        test_aligned: list[torch.Tensor] = []
        test_labels_list: list[list[Any]] = []
        for idx, meta in enumerate(test_metadata):
            if self.label_key not in meta:
                raise InvalidDatasetError(
                    f"Test sample at index {idx} metadata does not contain '{self.label_key}'."
                )
            raw_target = meta[self.label_key]
            raw_labels = parse_frame_targets(
                raw_target,
                key_name=self.label_key,
            )
            clip_reps = test_representations[idx]
            aligned = align_features_to_targets(
                representations=clip_reps,
                target_len=len(raw_labels),
            )
            test_aligned.append(aligned)
            test_labels_list.append(raw_labels)

        X_train = torch.cat(train_aligned, dim=0)
        X_test = torch.cat(test_aligned, dim=0)

        train_flat = [lbl for sub in train_labels_list for lbl in sub]
        test_flat = [lbl for sub in test_labels_list for lbl in sub]
        all_labels = train_flat + test_flat

        encoder = LabelEncoder()
        encoder.fit(all_labels)
        num_classes = len(encoder.classes_)

        if num_classes < 2:
            raise TaskError(
                f"Frame classification requires at least 2 distinct classes, but found {num_classes}."
            )

        y_train = torch.as_tensor(encoder.transform(train_flat), dtype=torch.long)
        y_test = torch.as_tensor(encoder.transform(test_flat), dtype=torch.long)

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

        return results

    def examine_data(
        self,
        dataset: Dataset,
        testset: Dataset | None = None,
    ) -> dict[str, Any]:
        """Examine the dataset in the context of frame classification.

        Validates the datasets, computes clip/frame distributions, class counts,
        and evaluates a trivial baseline (majority frame class) on raw test targets.

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
        if testset is not None:
            test_metadata = self.extract_metadata(testset)
        else:
            test_metadata = train_metadata

        if len(train_metadata) == 0:
            raise TaskError("Cannot examine an empty dataset.")

        train_labels: list[Any] = []
        for meta in train_metadata:
            targets = parse_frame_targets(
                meta[self.label_key],
                key_name=self.label_key,
            )
            train_labels.extend(targets)

        test_labels: list[Any] = []
        for meta in test_metadata:
            targets = parse_frame_targets(
                meta[self.label_key],
                key_name=self.label_key,
            )
            test_labels.extend(targets)

        all_labels = train_labels + (test_labels if testset is not None else [])

        encoder = LabelEncoder()
        encoder.fit(all_labels)
        classes = [str(c) for c in encoder.classes_]
        num_classes = len(classes)

        if num_classes < 2:
            raise TaskError(
                f"Frame classification requires at least 2 distinct classes, but found {num_classes}."
            )

        sample_targets = parse_frame_targets(
            train_metadata[0][self.label_key],
            key_name=self.label_key,
        )
        frame_len_sample = len(sample_targets) if train_metadata else 0

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
