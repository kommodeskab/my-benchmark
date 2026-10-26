# Project Development Rules: Audio Benchmark Suite

This repository implements a modular benchmark suite for downstream audio tasks using classification and regression models.

## Core Rules for Code Generation

1. **Simplicity First**:
   - Keep the codebase as simple, clean, and direct as possible.
   - Avoid unnecessary abstractions, wrapper classes, or premature optimization.

2. **Explicit Arguments (No `**kwargs`)**:
   - Never use `**kwargs` unless strictly required by external library interfaces.
   - Explicitly define and type all parameters.
   - Downstream models must be explicitly instantiated objects passed by the user (no string names, no hidden instantiation, no `None` defaults for required models).

3. **Composability**:
   - Downstream models must be modular and interchangeable.
   - All classifiers inherit from `BaseClassifier` and implement `fit`, `predict`, and `predict_proba`.
   - All regressors inherit from `BaseRegressor` and implement `fit` and `predict`.
   - Benchmark tasks must accept any compatible model without altering task evaluation logic.

4. **Documentation**:
   - Every function, method, and class exposed to the user must have complete docstrings (Google style: Summary, `Args:`, `Returns:`, `Raises:`).

5. **Minimal Comments**:
   - Do NOT use comments to describe obvious logic or narrate code.
   - Only add comments when strictly necessary to explain non-obvious workarounds, edge-case handling, or weird/subtle behavior.

6. **Targeted Testing**:
   - Only write tests for important functionality, mathematical/protocol invariants, edge cases, and public APIs.
   - Avoid bloated or redundant tests for trivial getters or pass-through properties.

7. **Strict Validation Without Auto-Fixing or Inference**:
   - The user must explicitly pass arguments and correctly formatted information to the library.
   - Check if inputs, datasets, and arguments meet requirements, raising clear and explicit errors when invalid.
   - Never try to fix malformed data silently (e.g., auto-reshaping audio, coercing types like float sample rates to int).
   - Never try to infer information (e.g., inferring dataset names or guessing metadata keys).

8. **No Aliases (One Canonical Name)**:
   - Every class, function, and benchmark must have exactly ONE canonical name.
   - Never create aliases (e.g. `TIMITRegressionBenchmark = TIMITSpeechDurationRegressionBenchmark`). All classes in the repo must only have one name.

9. **Skill Reference**:
   - Detailed examples and guidelines can be found in the skill at [`.agents/skills/audio-benchmark-dev/SKILL.md`](.agents/skills/audio-benchmark-dev/SKILL.md).
