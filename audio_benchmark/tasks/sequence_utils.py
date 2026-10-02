"""Utilities for sequence modeling: edit distance metrics and vocabulary mapping."""

from collections.abc import Sequence
from typing import Any


def levenshtein_distance(ref: Sequence[Any], hyp: Sequence[Any]) -> int:
    """Compute the Levenshtein (edit) distance between reference and hypothesis sequences.

    Returns the minimum number of insertions, deletions, and substitutions
    required to transform hyp into ref.

    Args:
        ref: Ground truth reference token sequence.
        hyp: Predicted hypothesis token sequence.

    Returns:
        int: Total edit distance.
    """
    n, m = len(ref), len(hyp)
    if n == 0:
        return m
    if m == 0:
        return n

    # Dynamic programming with O(m) space
    dp = list(range(m + 1))

    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            temp = dp[j]
            if ref[i - 1] == hyp[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp

    return dp[m]


def compute_token_error_rate(
    references: Sequence[Sequence[Any]],
    hypotheses: Sequence[Sequence[Any]],
) -> tuple[float, float]:
    """Compute Token Error Rate (PER/CER) and Exact Sequence Accuracy.

    Args:
        references: Collection of ground truth reference token sequences.
        hypotheses: Collection of predicted hypothesis token sequences.

    Returns:
        tuple[float, float]: (token_error_rate, sequence_accuracy)
    """
    total_errors = 0
    total_ref_len = 0
    exact_matches = 0

    for ref, hyp in zip(references, hypotheses):
        dist = levenshtein_distance(ref, hyp)
        total_errors += dist
        total_ref_len += len(ref)
        if list(ref) == list(hyp):
            exact_matches += 1

    ter = float(total_errors / max(1, total_ref_len))
    acc = float(exact_matches / max(1, len(references)))
    return ter, acc


class Vocabulary:
    """Vocabulary mapping between raw tokens and contiguous integer indices.

    Args:
        sequences: Collection of token sequences used to build the vocabulary.
    """

    def __init__(self, sequences: Sequence[Sequence[Any]]):
        unique_tokens = sorted({tok for seq in sequences for tok in seq})
        self.token_to_id = {tok: idx for idx, tok in enumerate(unique_tokens)}
        self.id_to_token = {idx: tok for tok, idx in self.token_to_id.items()}

    def __len__(self) -> int:
        return len(self.token_to_id)

    @property
    def tokens(self) -> list[Any]:
        """List of tokens ordered by token ID."""
        return [self.id_to_token[i] for i in range(len(self.id_to_token))]

    def encode(self, sequence: Sequence[Any]) -> list[int]:
        """Convert a sequence of raw tokens to integer IDs.

        Args:
            sequence: Sequence of raw tokens.

        Returns:
            list[int]: List of corresponding token integer IDs.
        """
        return [self.token_to_id[tok] for tok in sequence]

    def decode(self, indices: Sequence[int]) -> list[Any]:
        """Convert a sequence of integer IDs back to raw tokens.

        Args:
            indices: Sequence of integer token IDs.

        Returns:
            list[Any]: List of corresponding original tokens.
        """
        return [self.id_to_token[idx] for idx in indices if idx in self.id_to_token]
