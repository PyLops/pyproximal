# Agent Guide for PyProximal

## What This Project Is
PyProximal implements **proximal operators** and **proximal algorithms** for non-smooth, constrained convex optimization. It deliberately does *not* implement linear operators: those come from [PyLops](https://pylops.readthedocs.io) (`pylops.LinearOperator` is a hard dependency, `>= 2.4.0`). Any change that would add a linear-operator implementation here is out of scope by design.

## Where Things Live
- `pyproximal/`: library code.
  - `proximal/`: proximal operators (one file per operator family, `UpperCaseCamelCase` filename matching the class).
  - `projection/`: orthogonal projections onto sets, usually wrapped by the indicator-function proximal operators.
  - `optimization/`: solvers.
  - `utils/`: `moreau` and `gradtest_proximal` test helpers, `BilinearOperator`, backend/typing helpers.
  - `ProxOperator.py`: base class of every proximal operator.
- `pytests/`: pytest suite.
- `docs/`, `examples/`, `tutorials/`, `testdata/`: docs, examples, tutorial assets, and test data.
- `pyproject.toml`: build, test, lint, and packaging config.
- `Makefile`: preferred entry point for local work.

## Working Here
- Prefer `make` targets. Use the `*_uv` variants when working in a `uv` environment.
- Common commands: `make dev-install_uv`, `make tests` or `make tests_uv`, `make lint` or `make lint_uv`, `make typeannot` or `make typeannot_uv`, `make docupdate` or `make docupdate_uv`.
- Packaging uses `hatchling`; keep build and version settings in `pyproject.toml`.

## Core Design
- `ProxOperator` implements `prox` and `proxdual` each in terms of the other via the Moreau decomposition, so a subclass needs only one of them; implement both when closed forms exist.
- Decorate every `prox`/`proxdual` with `@_check_tau`.
- `grad` on the base class is the gradient of the Moreau envelope, not of the function; pass `hasgrad=True` and override `grad` when a true gradient is known.
- Solvers exist twice: class-based implementations in `optimization/cls_primal.py` and `cls_primaldual.py` hold the logic; `primal.py` and `primaldual.py` are thin functional wrappers over them. Change behavior in the `cls_*` files.
- Solvers should support both `numpy` and `cupy` arrays (`get_array_module(x0)`); numba/CUDA paths are optional and selected through an `engine` argument with a runtime fallback (see `proximal/Simplex.py`).

## Style And Tests
- Follow the `ruff` rules in `pyproject.toml`; compliance is enforced in CI, as is `mypy` in `strict` mode over `pyproximal/`.
- Keep imports tidy and follow PEP 8 rules.
- Follow `numpydoc` style for docstrings, with a `Notes` section giving the maths and a reference.
- Export new operators in the subpackage `__init__.py`: docstring table, star import, and `__all__`, and list them in `docs/source/api/index.rst`.
- Add or update tests in `pytests/` and examples in `examples/` and/or `tutorials/` when changing behavior or public APIs. Validate a new operator with `moreau` when both prox and dual prox are available, otherwise with ad-hoc edge cases.

## Contribution Flow
- Use `docs/source/contributing.rst` as the source of truth for longer contribution workflows, and `docs/source/adding.rst` / `docs/source/addingsolver.rst` when implementing a new operator or solver.
- If functionality changes, update docs and run the relevant tests before handing off.
- Avoid editing generated artifacts or build output unless the task explicitly requires it.
