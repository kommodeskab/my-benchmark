"""Custom exceptions for the audio representation learning benchmark suite."""


class BenchmarkError(Exception):
    """Base exception for benchmark suite errors."""


class InvalidDatasetError(BenchmarkError):
    """Raised when a supplied dataset does not meet the benchmark requirements."""


class InvalidModelError(BenchmarkError):
    """Raised when a supplied model is not a valid PyTorch module."""


class InvalidModelOutputError(BenchmarkError):
    """Raised when a representation model output shape does not match (B, C', T')."""


class TaskError(BenchmarkError):
    """Raised when a task configuration or execution fails."""
