"""Tests for temporal downsampling utility."""

import numpy as np
import pytest

from audio_benchmark.tasks.downsampling import downsample_sequence


def test_downsample_mean_regression():
    # Sequence of 8 numbers downsampled to 4 (regions of size 2)
    # [1, 3] -> 2.0, [2, 4] -> 3.0, [5, 5] -> 5.0, [10, 20] -> 15.0
    seq = [1, 3, 2, 4, 5, 5, 10, 20]
    out = downsample_sequence(seq, target_len=4, mode="mean")
    assert len(out) == 4
    assert np.allclose(out, [2.0, 3.0, 5.0, 15.0])


def test_downsample_mode_classification():
    # Sequence of 6 labels downsampled to 2 (regions of size 3)
    # ['dog', 'dog', 'cat'] -> 'dog'
    # ['cat', 'cat', 'bird'] -> 'cat'
    seq = ["dog", "dog", "cat", "cat", "cat", "bird"]
    out = downsample_sequence(seq, target_len=2, mode="mode")
    assert len(out) == 2
    assert list(out) == ["dog", "cat"]


def test_downsample_identical_length():
    seq = [1.0, 2.0, 3.0]
    out = downsample_sequence(seq, target_len=3, mode="mean")
    assert np.allclose(out, seq)


def test_downsample_invalid_inputs():
    with pytest.raises(ValueError, match="empty sequence"):
        downsample_sequence([], target_len=4)

    with pytest.raises(ValueError, match="target_len must be positive"):
        downsample_sequence([1, 2, 3], target_len=0)

    with pytest.raises(ValueError, match="Unknown downsample mode"):
        downsample_sequence([1, 2, 3], target_len=2, mode="median")
