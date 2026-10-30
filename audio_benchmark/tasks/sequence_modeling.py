"""Variable-length sequence modeling task with pluggable sequence probes."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from audio_benchmark.dataset import parse_dataset_item, validate_dataset
from audio_benchmark.exceptions import InvalidDatasetError, TaskError
from audio_benchmark.tasks.base import BaseTask
from audio_benchmark.tasks.sequence_models import BaseSequenceProbe
from audio_benchmark.tasks.sequence_utils import Vocabulary, compute_token_error_rate


class SequenceModelingTask(BaseTask):
    """Task for modeling variable-length target token sequences (e.g. phonemes or words).

    Accepts any downstream probe implementing `BaseSequenceProbe` (such as `CTCSequenceProbe`
    or `TransformerDecoderProbe`).

    Args:
        probe: Explicit instantiated probe model.
        name: Identifier for the task.
        target_key: Key in sample metadata containing the target token sequence.
    """

    def __init__(
        self,
        probe: BaseSequenceProbe,
        name: str = "sequence_modeling",
        target_key: str = "tokens",
    ):
        """Initialize the SequenceModelingTask.

        Args:
            probe: Explicit instantiated BaseSequenceProbe instance.
            name: Identifier for the task.
            target_key: Key in sample metadata containing the target token sequence.

        Raises:
            TypeError: If probe is not an instance of BaseSequenceProbe.
        """
        super().__init__(name=name)
        if not isinstance(probe, BaseSequenceProbe):
            raise TypeError(
                f"probe must be an instance of BaseSequenceProbe, got {type(probe).__name__}."
            )
        self.probe = probe
        self.target_key = target_key
        self.metric = "token_error_rate"
        self.m_min = 1.0
        self.m_max = 0.0

    def validate_dataset(self, dataset: Dataset) -> None:
        """Verify that dataset metadata contains non-empty target sequences.

        Args:
            dataset: PyTorch dataset to validate.
        """
        validate_dataset(dataset, required_metadata_keys=[self.target_key])

        parsed = parse_dataset_item(dataset[0])
        seq = parsed["metadata"][self.target_key]
        if not isinstance(seq, list):
            raise InvalidDatasetError(
                f"Metadata key '{self.target_key}' must be a list of tokens, "
                f"got {type(seq).__name__}."
            )
        if len(seq) == 0:
            raise InvalidDatasetError(
                f"Sample at index 0 has empty sequence for key '{self.target_key}'."
            )

    def _extract_sequences(
        self,
        metadata: list[dict[str, Any]],
        prefix: str = "Sample",
    ) -> list[list[Any]]:
        """Extract and validate token sequence lists from metadata."""
        sequences: list[list[Any]] = []
        for idx, meta in enumerate(metadata):
            if self.target_key not in meta:
                raise InvalidDatasetError(
                    f"{prefix} at index {idx} metadata does not contain '{self.target_key}'."
                )
            seq = meta[self.target_key]
            if not isinstance(seq, list):
                raise InvalidDatasetError(
                    f"Target for {prefix.lower()} {idx} must be a list of tokens, got {type(seq).__name__}."
                )
            sequences.append(seq)
        return sequences

    def _prepare_split(
        self,
        representations: torch.Tensor,
        metadata: list[dict[str, Any]],
        split_name: str,
    ) -> tuple[list[torch.Tensor], list[list[Any]]]:
        """Extract unpadded per-clip representations and token sequences for a split."""
        prefix = f"{split_name.capitalize()} sample"
        targets = self._extract_sequences(metadata, prefix=prefix)
        reps = [
            representations[
                i, : int(meta.get("rep_length", representations.shape[1]))
            ].to(torch.float32)
            for i, meta in enumerate(metadata)
        ]
        return reps, targets

    def evaluate(
        self,
        train_representations: torch.Tensor,
        train_metadata: list[dict[str, Any]],
        test_representations: torch.Tensor,
        test_metadata: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Train probe on train representations and evaluate on test representations.

        Args:
            train_representations: Tensor of shape (N_train, T', C').
            train_metadata: List of metadata dicts for train samples.
            test_representations: Tensor of shape (N_test, T', C').
            test_metadata: List of metadata dicts for test samples.

        Returns:
            dict[str, Any]: Evaluation metrics dictionary.
        """
        train_reps, raw_train_targets = self._prepare_split(
            train_representations, train_metadata, "train"
        )
        test_reps, raw_test_targets = self._prepare_split(
            test_representations, test_metadata, "test"
        )

        vocab = Vocabulary(raw_train_targets + raw_test_targets)
        vocab_size = len(vocab)
        if vocab_size < 1:
            raise TaskError("Target vocabulary is empty.")

        encoded_train_targets = [vocab.encode(seq) for seq in raw_train_targets]

        start_time = time.perf_counter()
        self.probe.fit(train_reps, encoded_train_targets, vocab_size=vocab_size)
        training_time = float(time.perf_counter() - start_time)

        pred_id_sequences = self.probe.predict(test_reps)
        pred_tokens = [vocab.decode(ids) for ids in pred_id_sequences]

        ter, seq_acc = compute_token_error_rate(raw_test_targets, pred_tokens)
        value = float(ter)

        return {
            "task": self.name,
            "probe": type(self.probe).__name__,
            "token_error_rate": ter,
            "sequence_accuracy": seq_acc,
            "value": value,
            "normalized_value": self.normalize_metric(value, self.m_min, self.m_max),
            "main_metric": self.metric,
            "training_time": training_time,
        }

    def examine_data(
        self,
        dataset: Dataset,
        testset: Dataset | None = None,
    ) -> dict[str, Any]:
        """Examine the dataset in the context of sequence modeling.

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

        raw_train_targets = self._extract_sequences(train_metadata)
        raw_test_targets = self._extract_sequences(test_metadata, prefix="Test sample")
        all_targets = raw_train_targets + (
            raw_test_targets if testset is not None else []
        )

        vocab = Vocabulary(raw_train_targets + raw_test_targets)
        vocab_size = len(vocab)
        if vocab_size < 1:
            raise TaskError("Target vocabulary is empty.")

        seq_lengths = [len(seq) for seq in all_targets]
        pred_empty = [[] for _ in raw_test_targets]
        baseline_ter, baseline_seq_acc = compute_token_error_rate(
            raw_test_targets, pred_empty
        )

        return {
            "task": self.name,
            "total_clips": len(all_targets),
            "train_clips": len(train_metadata),
            "test_clips": len(test_metadata) if testset is not None else 0,
            "vocab_size": int(vocab_size),
            "vocabulary": list(vocab.tokens),
            "target_key": self.target_key,
            "min_sequence_length": min(seq_lengths),
            "max_sequence_length": max(seq_lengths),
            "mean_sequence_length": float(np.mean(seq_lengths)),
            "baseline_strategy": "empty_sequence",
            "baseline_metric": self.metric,
            "baseline_value": baseline_ter,
            "baseline_normalized_value": self.normalize_metric(
                baseline_ter, self.m_min, self.m_max
            ),
            "baseline_token_error_rate": baseline_ter,
            "baseline_sequence_accuracy": baseline_seq_acc,
        }
