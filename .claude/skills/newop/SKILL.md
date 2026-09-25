---
name: newop
description: Create a new PyProximal proximal operator following docs/source/adding.rst - class file, docstring, registration, Moreau/edge-case tests, docs entry and example. Use when the user asks to add/implement/port a proximal operator, penalty, norm, or indicator/projection into PyProximal, including porting an existing prox implementation from a paper, a URL or a local file (e.g. "add a Foo proximal operator", "implement the prox of this penalty", "port the prox at <link>").
---

Goal: add a new, PyProximal-compliant `ProxOperator` to the library, complete with
docstring, registration, tests, docs entry and a gallery example, following
`docs/source/adding.rst` (the authoritative guide - read it if unsure).

The operator may be written from scratch (from a mathematical description or a
paper) or **ported** from an existing implementation supplied as a **web link** or a
**local file**. Ask for the operator name and the source only if neither is
inferable from the invocation.

## 0. Get the source material

- **Web link**: fetch it with `WebFetch` (or the browser tools if the page needs
  JS). Extract the actual prox/dual prox code or formulas, not the prose.
- **Local file**: read it in full.
- **Neither**: work from the user's mathematical description, and state the
  assumed definition of the function before writing code.

Then write down explicitly, before touching `pyproximal/`:

- the function \(f(\mathbf{x})\), its parameters and its domain;
- whether a closed form exists for `prox`, for `proxdual`, or both;
- whether the source's "prox" really is \(\prox_{\tau f}\) and not \(\prox_f\) -
  **a missing or misplaced `tau` is the most common porting bug**;
- whether \(f\) is differentiable (`hasgrad=True` and a `grad` override);
- whether \(f\) is convex or non-convex (decides the docs section);
- whether \(f\) is an indicator function (then a projection class is needed);
- whether it acts on vectors or on matrices (e.g. singular-value penalties);
- which source parameters become `__init__` arguments, which become derived
  members, and which are irrelevant (plotting, I/O, CLI args).

## 1. Place the file

- File `pyproximal/proximal/<ClassName>.py` in **UpperCamelCase**, matching the
  class (`HalfSpace.py` holds `HalfSpace`). Closely related variants share a file
  (`L1`/`L1Ball` in `L1.py`, `Log`/`Log1` in `Log.py`).
- For an **indicator function**, first add `pyproximal/projection/<Name>.py` with a
  `<Name>Proj` class exposing `__call__(x)` (pattern: `projection/HalfSpace.py`),
  register it in `projection/__init__.py` (docstring table, star import, `__all__`),
  then wrap it in the proximal operator (pattern: `proximal/HalfSpace.py`).
- Reuse existing helpers instead of re-deriving them, e.g. `_softthreshold` and
  `_current_sigma` (callable `sigma`) in `proximal/L1.py`, and the projections in
  `pyproximal.projection` (`BoxProj`, `L1BallProj`, ...).

## 2. Write the class

Use `reference/operator_template.py` as the skeleton. Key rules:

- Inherit from `pyproximal.ProxOperator.ProxOperator` and call
  `super().__init__(Op, hasgrad)`: pass a PyLops `LinearOperator` as `Op` only if
  the function uses one (else `None`), and `hasgrad=True` only if you override
  `grad` with the true gradient (else `False`, and the base `grad` returns the
  gradient of the Moreau envelope).
- `__call__(x)` returns the function value (`float`), or membership (`bool`) for an
  indicator.
- Implement `prox(self, x, tau)` and/or `proxdual(self, x, tau)`, **each decorated
  with `@_check_tau`**. At least one is required: the base class derives the other
  through the Moreau identity. Implement both when closed forms exist.
- Keep it backend-agnostic: use `ncp = get_array_module(x)` from
  `pylops.utils.backend` for array operations inside `prox`/`proxdual`, so CuPy
  inputs work. Numba/CUDA paths are optional and go through an `engine` argument
  with a runtime fallback (see `proximal/Simplex.py`); do not add them unless asked.
- Type-annotate everything (`NDArray` from `pylops.utils.typing`,
  `FloatCallableLike` from `pyproximal.utils.typing`): `mypy --strict` runs over
  `pyproximal/`.
- Validate inputs in `__init__` as `msg = "..."; raise ValueError(msg)` (ruff `EM`).
- Write the `numpydoc` docstring (`r"""`) with, at minimum: one-line summary, the
  definition of the function in maths, `.. versionadded:: X.Y.Z` (next release after
  `git describe --tags`), `Parameters`, `Raises` (when `__init__` validates), and a
  `Notes` section giving the prox (and dual prox, if implemented) in `.. math::`
  blocks with a `.. [1]` reference. Match the detail of neighbouring operators.

## 3. Register

In `pyproximal/proximal/__init__.py` add: a row in the module docstring table,
`from .<ClassName> import *`, and the class name in `__all__`. The top-level
`pyproximal/__init__.py` re-exports with `from .proximal import *`, so the operator
is then available as `pyproximal.<ClassName>`.

## 4. Add tests

Add to the matching existing file (`test_norms.py`, `test_proximal.py`,
`test_concave_penalties.py`, `test_projection.py`) or create a new one. Follow
`reference/test_template.py`:

- module-level `par*` dicts (float32 even size / float64 odd size) with
  `@pytest.mark.parametrize("par", [(par1), (par2)])` and `np.random.seed(10)`;
- a `__call__` check against an independently computed value;
- **both `prox` and `proxdual` in closed form**: `assert moreau(op, x, tau)`;
- **only one in closed form**: ad-hoc edge cases with known answers (x=0, points
  inside/outside the set, `tau`→0 returns `x`, large `tau`), and the defining
  property where cheap - for indicators the output lies in the set and projecting
  twice is idempotent; in general, compare against a brute-force
  `scipy.optimize.minimize` of \(f(\mathbf{y}) + \|\mathbf{y}-\mathbf{x}\|_2^2/(2\tau)\)
  on a small vector;
- with `hasgrad=True`, add a `gradtest_proximal` test in `test_grads.py`;
- an error-path test for every `raise` in `__init__` (see
  `test_Quadratic_nonsquare` in `test_proximal.py`).

Never loosen the `moreau` tolerance to make a wrong dual pass: fix the maths.

## 5. Run

Always use `uv`:

```bash
uv run pytest pytests/test_<file>.py -k <ClassName> -q
make lint_uv
make typeannot_uv
```

Iterate until all tests pass and lint/mypy are clean.

## 6. Document

- Add the class name (alphabetically) to the right `autosummary` block in
  `docs/source/api/index.rst`: *Vector → Convex* / *Vector → Non-Convex*,
  *Matrix-only*, or *Other*; a new projection goes under *Orthogonal projections*.
- Use the operator in at least one example: extend an existing gallery script when
  it fits (`examples/plot_norms.py`, `plot_indicators.py`,
  `plot_concave_penalties.py`), or add `examples/plot_<name>.py` in sphinx-gallery
  format (`r"""` title/underline/description header, `###...` comment blocks between
  narrative and code, a matplotlib figure of `x` vs `prox(x)` vs `proxdual(x)`).
  Prefer a script in `tutorials/` when the operator is best shown inside a solver.

## 7. Final checklist (from `docs/source/adding.rst`)

Report back confirming each item:

- [ ] class in its own UpperCamelCase file in `pyproximal/proximal/` (plus a
      projection in `pyproximal/projection/` for indicators)
- [ ] `__init__`, `__call__`, and `prox` and/or `proxdual` (both with `@_check_tau`)
      implemented; `grad` only with `hasgrad=True`
- [ ] exported in `pyproximal/proximal/__init__.py` (table, import, `__all__`)
- [ ] numpydoc docstring with `Parameters` and a mathematical `Notes` section with
      a reference
- [ ] tests added (Moreau or edge cases, `__call__`, raises) and passing
- [ ] listed in `docs/source/api/index.rst`
- [ ] used in at least one `examples/` or `tutorials/` script
- [ ] `make lint_uv` and `make typeannot_uv` clean

When porting, close with a short note on what differed between the source
implementation and the PyProximal version (`tau` scaling, dual in closed form vs
derived via Moreau, shape/dtype handling, removed I/O).
