# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install runtime deps
pip install -r requirements.txt

# Tests use pytest (not unittest) — bare test_* functions with plain `assert`.
# pytest is not in requirements.txt; install it into the active env if missing:
pip install pytest

# Run all tests
pytest tests/ -v

# Run one test
pytest tests/test_self_attention.py::test_output_shape_matches_input
```

If PyCharm reports "Empty suite / 0 tests ran" while invoking `_jb_unittest_runner.py`, the run config is using `unittest` discovery, which cannot find pytest-style bare functions. Set **Settings → Tools → Python Integrated Tools → Testing → Default test runner → pytest** *and* delete any pre-existing run configurations (changing the default does not migrate cached configs).

## Architecture

This is a small "zoo" of attention-mechanism implementations in PyTorch — currently only one variant exists.

- `attention/self_attention.py` — `SelfAttention(nn.Module)`: single-head scaled dot-product self-attention. Q/K/V are three `nn.Linear(input_dim, input_dim)` projections; forward is `softmax(Q·Kᵀ / √d) · V` via `torch.bmm`. No multi-head split, no masking (encoder-style only), output dim equals input dim.
- `attention/__init__.py` is empty — modules are imported by their full path (`from attention.self_attention import SelfAttention`).
- `tests/test_self_attention.py` covers shape preservation, softmax row-sum invariant, gradient flow through Q/K/V, and batch independence.

When adding new variants (multi-head, causal/masked, cross-attention, etc.), add them as sibling modules under `attention/` and mirror the test layout under `tests/`.

## Git workflow

- Only commit or push when the user explicitly asks. Do not commit or push proactively at the end of a task.
- Keep commit messages short — one or two sentences, focused on the why.
- Do not mention Claude or include any Claude/AI co-authorship trailer in commit messages.
