"""Tests for temporal downsampling utility."""

import numpy as np
import pytest
import torch

from audio_benchmark.tasks.downsampling import align_features_to_targets


def test_align_features_linear_interpolation_downsampling():
    # Representations T'=4 frames downsampled to target_len=2 via linear interpolation
    reps = torch.tensor(
        [
            [1.0, 10.0],
            [3.0, 20.0],
            [5.0, 30.0],
            [9.0, 50.0],
        ],
        dtype=torch.float32,
    )
    aligned = align_features_to_targets(reps, target_len=2)
    assert isinstance(aligned, torch.Tensor)
    assert aligned.shape == (2, 2)
    assert torch.allclose(aligned[0], torch.tensor([2.0, 15.0]))
    assert torch.allclose(aligned[1], torch.tensor([7.0, 40.0]))


def test_align_features_linear_interpolation_upsampling():
    # Representations T'=2 frames upsampled to target_len=4 via linear interpolation
    reps = torch.tensor([[10.0, 20.0], [30.0, 40.0]], dtype=torch.float32)
    aligned = align_features_to_targets(reps, target_len=4)
    assert isinstance(aligned, torch.Tensor)
    assert aligned.shape == (4, 2)
    assert torch.allclose(aligned[0], torch.tensor([10.0, 20.0]))
    assert torch.allclose(aligned[1], torch.tensor([15.0, 25.0]))
    assert torch.allclose(aligned[2], torch.tensor([25.0, 35.0]))
    assert torch.allclose(aligned[3], torch.tensor([30.0, 40.0]))


def test_align_features_identity_and_single_frame():
    # Identity: T' == target_len
    reps_3 = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float32)
    aligned_identity = align_features_to_targets(reps_3, target_len=3)
    assert aligned_identity is reps_3

    # Single frame T'=1 upsampled to target_len=3
    reps_1 = torch.tensor([[42.0, 84.0]], dtype=torch.float32)
    aligned_single = align_features_to_targets(reps_1, target_len=3)
    assert aligned_single.shape == (3, 2)
    expected = torch.tensor([[42.0, 84.0], [42.0, 84.0], [42.0, 84.0]])
    assert torch.allclose(aligned_single, expected)


def test_align_features_validation():
    reps = torch.ones((4, 8), dtype=torch.float32)

    with pytest.raises(TypeError, match="must be a torch.Tensor"):
        align_features_to_targets(np.ones((4, 8)), target_len=2)

    with pytest.raises(ValueError, match="target_len must be a positive integer"):
        align_features_to_targets(reps, target_len=0)

    with pytest.raises(ValueError, match="target_len must be a positive integer"):
        align_features_to_targets(reps, target_len=-3)

    with pytest.raises(ValueError, match="must have shape"):
        align_features_to_targets(torch.ones(4), target_len=2)

    with pytest.raises(ValueError, match="cannot have temporal length 0"):
        align_features_to_targets(torch.zeros((0, 8)), target_len=2)


def test_parse_frame_targets_tensor_and_sequence():
    from audio_benchmark.exceptions import InvalidDatasetError
    from audio_benchmark.tasks.downsampling import parse_frame_targets

    # Uniform sequence
    vals = parse_frame_targets([10, 20, 30])
    assert vals == [10, 20, 30]

    # Flat torch.Tensor
    vals_tensor = parse_frame_targets(torch.tensor([10.0, 20.0, 30.0]))
    assert vals_tensor == [10.0, 20.0, 30.0]

    # Error cases: empty sequence or empty tensor
    with pytest.raises(InvalidDatasetError, match="must not be empty"):
        parse_frame_targets([])

    with pytest.raises(InvalidDatasetError, match="must not be empty"):
        parse_frame_targets(torch.empty(0))

    # Error cases: multi-dimensional tensor
    with pytest.raises(InvalidDatasetError, match="must be 1D"):
        parse_frame_targets(torch.ones((3, 2)))

    # Error cases: dictionary is no longer allowed
    with pytest.raises(
        InvalidDatasetError, match="must be a flat torch.Tensor or list"
    ):
        parse_frame_targets({0: 10, 50: 20})

    # Error cases: invalid container types
    with pytest.raises(
        InvalidDatasetError, match="must be a flat torch.Tensor or list"
    ):
        parse_frame_targets((10, 20))

    with pytest.raises(
        InvalidDatasetError, match="must be a flat torch.Tensor or list"
    ):
        parse_frame_targets("invalid")
