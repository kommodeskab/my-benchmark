---
name: audio-benchmark-dev
description: Guidelines and design principles for developing and generating code for the downstream audio task benchmark suite. Activate this skill whenever writing, modifying, refactoring, or testing code in this repository, including downstream classification/regression models, task evaluators, feature extractors, and benchmarks.
---

# Audio Benchmark Development Guide

This skill guides code generation and refactoring in this repository. The project is a modular benchmark suite for downstream audio tasks using both classification and regression models (such as Linear, k-NN, MLP, and Transformer downstream heads).

Adhere strictly to the following principles:

---

## 1. Simplicity Over Complexity

- **Keep code direct and readable**: Choose the simplest implementation that satisfies the requirements.
- **Avoid premature abstractions**: Do not introduce helper classes, wrappers, or meta-programming unless there is a clear, immediate necessity.
- **Flat is better than nested**: Avoid deep inheritance or convoluted control flow.

---

## 2. Explicit Arguments (Avoid `**kwargs`)

- **Never use `**kwargs`** unless strictly mandated by an external library signature.
- **Explicit parameter lists**: Every function and method parameter must be explicitly named, typed, and passed directly.
- **Explicit object instantiation**:
  - Do NOT accept model names as strings or implicitly instantiate models with hidden default configurations.
  - Require users/callers to pass explicit, fully instantiated objects (e.g., an instantiated `BaseClassifier` or `BaseRegressor`).
  - Do NOT use `None` as a magical placeholder for a default model if a model is required; require the explicit object.

```python
# GOOD: Explicit, typed parameters and model instances
def evaluate_task(
    task: BaseTask,
    model: BaseClassifier,
    dataset: Dataset,
    testset: Dataset,
) -> TaskResult: ...


# BAD: Hiding configuration behind **kwargs or string instantiation
def evaluate_task(task, model="mlp", **kwargs): ...
```

---

## 3. Composability & Swappable Downstream Heads

- **Uniform protocol**: All downstream models must adhere strictly to common base classes:
  - Classifiers inherit from `BaseClassifier` (`fit(X, y)`, `predict(X)`, `predict_proba(X)`).
  - Regressors inherit from `BaseRegressor` (`fit(X, y)`, `predict(X)`).
- **Interchangeability**: Benchmark tasks (global classification, frame classification, regression) must treat downstream models as interchangeable components. A user should be able to swap `LinearBenchmarkClassifier` for `MLPBenchmarkClassifier` or `KNNBenchmarkClassifier` with zero changes to task logic.
- **Decoupled features**: Models accept feature matrices/tensors (2D `(N, C)` or 3D `(N, T, C)`) and produce standard numpy outputs. Feature extraction and downstream prediction must remain decoupled.

---

## 4. Documentation Standards

- **All user-facing functions, methods, and classes MUST be documented**:
  - Provide a concise summary of purpose.
  - Include Google-style `Args:` for every argument with its type and meaning.
  - Include `Returns:` specifying the return type and structure.
  - Include `Raises:` if specific exceptions are explicitly raised.
- **Keep docstrings accurate and synchronized** whenever modifying parameters or behavior.

```python
def fit(self, X: np.ndarray | torch.Tensor, y: np.ndarray | torch.Tensor) -> Self:
    """Fit the classifier on extracted audio features and targets.

    Args:
        X: Audio feature array of shape (N, C) or (N, T, C).
        y: Ground truth integer class labels of shape (N,).

    Returns:
        The fitted classifier instance.

    Raises:
        ValueError: If input features or labels have invalid dimensions.
    """
```

---

## 5. Minimal In-Code Comments

- **Do NOT explain obvious code**: Avoid line-by-line narration or stating what the code is visibly doing.
- **Only comment when strictly necessary**:
  - When a workaround or non-obvious algorithm is required.
  - When behavior is counter-intuitive, tricky, or addresses an edge case (e.g. handling dimension flattening for 3D tensors before standard scaling).
- Prefer self-documenting code with clear variable and function names.

```python
# BAD: Obvious comment
# Convert features to numpy array
X_np = to_numpy(X)

# GOOD: Explaining weird or non-obvious requirement
# StandardScaler requires 2D input (N*T, C) to standardize across time and batches
orig_shape = X_np.shape
X_flat = X_np.reshape(-1, orig_shape[-1])
X_norm = self.scaler.fit_transform(X_flat).reshape(orig_shape)
```

---

## 6. Targeted Testing (Test What Matters)

- **Do NOT write excessive or redundant tests**:
  - Focus tests on critical functionality, core mathematical/downstream logic, and edge cases.
  - Do not write repetitive tests for trivial attributes, trivial getters, or simple pass-throughs.
- **Key invariants to test**:
  - Correct output shapes and types (`predict` returns correct array shape).
  - Feature normalization behaviors (`normalize_features=True` vs `False`).
  - Parameter validation (e.g., negative layer counts raising `ValueError`).
  - Downstream task metric computation and model interchangeability.
- Keep the test suite fast and lightweight.

---

## 7. Strict Input Validation (Check, Never Fix or Infer)

- **The user must explicitly pass arguments and correctly formatted information**:
  - The library must not try to guess what the user meant.
  - Require explicit parameter values rather than inferring them from environment, class names, or attributes (e.g., explicit `dataset_name`, explicit model instances).
- **Check if inputs are valid, but never try to fix them**:
  - Check that dataset items have the exact expected types and shapes (e.g. waveform strictly `(1, T)` `torch.Tensor`, `sample_rate` strictly a positive `int`).
  - Do NOT reshape 1D audio to 2D automatically or convert channels; raise `InvalidDatasetError` instead.
  - Do NOT cast/coerce float sample rates to `int`; check `isinstance(sr, int)` directly.
  - Do NOT infer metadata keys or fall back to guessing; assume metadata strictly contains the supported keys (`label`, `target`, `frame_labels`, `frame_targets`, `tokens`).
- **Fail fast with actionable errors**:
  - When validation fails, raise an informative exception describing what was expected, what was received, and how the user can correctly wrap or supply the input.

---

## 8. Verification & Tooling

After generating or modifying code, always verify:
1. **Formatting**: `uv run ruff format .`
2. **Linting**: `uv run ruff check .`
3. **Tests**: `uv run pytest`
