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
    dp = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        prev, dp[0] = dp[0], i
        for j, h in enumerate(hyp, start=1):
            prev, dp[j] = (
                dp[j],
                prev if r == h else 1 + min(prev, dp[j], dp[j - 1]),
            )
    return dp[-1]


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
    total_errors = sum(
        levenshtein_distance(ref, hyp)
        for ref, hyp in zip(references, hypotheses, strict=True)
    )
    total_ref_len = sum(len(ref) for ref in references)
    exact_matches = sum(
        list(ref) == list(hyp) for ref, hyp in zip(references, hypotheses, strict=True)
    )
    ter = float(total_errors / max(1, total_ref_len))
    acc = float(exact_matches / max(1, len(references)))
    return ter, acc


class Vocabulary:
    """Vocabulary mapping between raw tokens and contiguous integer indices.

    Args:
        sequences: Collection of token sequences used to build the vocabulary.
    """

    def __init__(self, sequences: Sequence[Sequence[Any]]):
        self.tokens: list[Any] = sorted({tok for seq in sequences for tok in seq})
        self.token_to_id: dict[Any, int] = {
            tok: idx for idx, tok in enumerate(self.tokens)
        }

    def __len__(self) -> int:
        return len(self.tokens)

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
        return [self.tokens[idx] for idx in indices if 0 <= idx < len(self.tokens)]
