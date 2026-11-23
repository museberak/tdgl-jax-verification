# TDGL Solver Verification with JAX

Verification of a Richardson solver for the time-dependent Ginzburg–Landau
(TDGL) equation, using manufactured solutions, convergence-order checks, an
independent SciPy reference, and mutation testing.

## What this repository is

A verification harness that answers one question: **does this discrete
solver actually solve the discrete equations it claims to solve, and would
the checks notice if it did not?** The solver is tested against manufactured
solutions with known analytic answers, on a sequence of grids, with an
independent sparse reference implementation and with deliberately injected
faults (mutations).

## What this repository is not

This is **solver verification, not a physics result**. Nothing in this
repository is a result about superconductivity, and it must not be cited as
one:

- The order parameter `psi` is real, not complex; there is no magnetic field
  and no vector potential.
- `alpha = +1` is above the critical temperature (the normal state). A
  non-zero `psi` exists here only because of an artificial forcing term `g`,
  which has no physical meaning.
- No physical observable is computed (no free energy, no supercurrent, no
  critical fields).
- The equation is solved as a steady manufactured-solution problem, not as a
  physical time evolution.

The honest status of this work is **benchmark-validated solver
verification**. It is not research-grade physics and not a novel
contribution. See [Limitations](#limitations).

## The problem that was fixed

The starting code (`examples/original_gigo.py`) built its right-hand side by
applying the same discrete operator it was testing. That is an inverse
crime: the test cannot fail, because the "exact" solution is defined by the
operator under test. The diagnosis, measured before any fix:

- **Float32 collapse.** In float32, the Poisson error stopped converging
  (observed order 2.03, then 0.13). In float64 the orders were 2.07, 2.03.
- **Inverse crime, demonstrated.** Multiplying the production operator by 2
  still gave a residual of exactly 0.0 — the same as the correct operator.
  The test was tautological.
- **Hidden discretisation error.** The difference between the
  operator-built right-hand side and the analytic right-hand side was
  4.36e-05 (N=32) and 1.07e-05 (N=64): real error that the original test
  could never see.

The fix, in one sentence: the right-hand side is now built from the analytic
solution, not from the operator under test, and the solver is checked against
an independent implementation.

## Method

The equation solved on the unit square, with homogeneous Dirichlet
boundaries, is

```
-ξ² Δψ + αψ + βψ³ = g
```

with the manufactured solution `ψ = A·sin(πx)·sin(πy)`. Because
`Δψ = -2π²ψ`, the analytic forcing is
`g = (2π²ξ² + α)ψ + βψ³`. The discrete operator is the 5-point Laplacian.

Three validators are run (`tdgl_validation.py`, `run_all()`):

| Validator | What it checks | Grids |
| --- | --- | --- |
| F1 | Poisson operator against the analytic solution | 33, 65, 129, 257 |
| F2 | Linear TDGL, Richardson iteration, manufactured solution | 33, 65, 129 |
| F3 | Nonlinear TDGL (βψ³), Richardson iteration | 33, 65, 129 |

Design points that the checks depend on:

- **Float64 is mandatory.** `jax_enable_x64` must be true; the validator
  refuses to run otherwise (mutation M7 proves this).
- **Exact discrete eigenvalues.** The Richardson step is `dt = 1/λ_max`,
  with `λ_min` and `λ_max` from the exact discrete Dirichlet spectrum,
  `λ = (8/h²)·sin²(π/(2(N-1)))` and `(8/h²)·cos²(π/(2(N-1)))`. The
  continuous limit `2π²` is wrong by 0.016 at N=33 and is not used. The
  formulas are verified against `eigsh` in `test_eigsh_evidence.py`.
- **Grids with an exact refinement ratio of 2.** `h = 1/(N-1)` with
  N = 33, 65, 129, 257 gives `h = 1/32, 1/64, 1/128, 1/256`. (The earlier
  grids 32/64/128 had ratios 2.032 and 2.016, while the order formula
  assumed exactly 2.)
- **Residuals are recomputed from the returned solution**, not taken from
  the last iteration, and the relative residual is gated separately from
  the solution error: a small residual is not the same as an accurate
  solution.
- **Independent reference.** Each F2/F3 solution is compared with a SciPy
  sparse solve (Newton iteration for the nonlinear case). The reference's
  own final residual is recomputed and gated before the comparison.
- **Fail-fast nonlinear bound.** The running maximum of `|ψ|` is tracked
  inside `lax.scan` at every iteration. If a candidate step would exceed
  the bound `2|A|`, the step is rejected, the state is frozen at the last
  valid state, and the error is raised with the violating step and value.
  The step index is zero-based; the human-readable step number is the
  reported index plus one.
- **Invalid inputs are rejected, not computed.** Non-finite parameters,
  `xi ≤ 0`, `tol ≤ 0`, `max_steps ≤ 0`, `beta < 0`, `A = 0` (a degenerate,
  trivial manufactured solution), and `N < 5` all raise `INVALID INPUT`.

Acceptance gates (unchanged through all versions; never loosened to obtain
a PASS): observed order in [1.8, 2.2]; relative residual < 1e-8;
cross-check error < 1e-6; boundary `max|ψ| ≤ 1e-12`; steps ≤ `max_steps`;
stability `dt·λ_max < 2`.

## Mutation testing

`mutation_test.py` injects eight faults and checks that the validator
rejects each one, plus one full-pipeline spot check. Statuses are
CAUGHT (the validator rejects the fault with a gate failure), MISSED
(the validator passes), and INVALID_TEST (the test itself failed in an
unexpected way). **Scope, stated precisely:** this is mutation testing of
the production validator components (grids 33/65). It is not full-pipeline
mutation testing of `run_all()`; only M1-full is a full-pipeline check.

| Mutation | Expected gate | Result |
| --- | --- | --- |
| M1 Laplacian ×2 | cross-check | CAUGHT, cross error 1.414e-01 |
| M2 one neighbour removed | cross-check | CAUGHT, cross error 8.956e-01 |
| M3 Laplacian sign flipped | divergence | CAUGHT, non-finite final residual |
| M4 α → −α in the operator | relative residual | CAUGHT, relative residual 7.66e+83 |
| M5 β removed | cross-check (Newton) | CAUGHT, cross error 1.223e-01 |
| M6 boundary set to 0.5 | boundary condition | CAUGHT, boundary 5.000e-01 |
| M7 float32 configuration | precision policy | CAUGHT, `jax_enable_x64 must be True` |
| M8 dt = 4/λ_max | stability criterion | CAUGHT, `dt*lambda_max = 4.000 >= 2` |
| M1-full, `run_all()` with Laplacian ×2 | observed order | CAUGHT, F1 order 0.02 |

Result: **9/9 CAUGHT, 0 MISSED, 0 INVALID_TEST.** Caught mutations prove
sensitivity to these eight faults only; they do not prove that every
possible bug would be caught.

## Results obtained

Executed 23–24 November 2026 (JAX 0.4.35, CPU backend; SciPy 1.18.1;
NumPy 2.5.5; Python 3.12.3; Linux x86_64). Parameters: ξ = 0.1, α = 1.0,
β = 1.0, A = 0.5, tol = 1e-10, max_steps = 100000. Benchmark runtime:
about 19 s. Full output: `logs/run.log`; machine-readable results:
`results.json`.

**F1 — Poisson, L2 error and observed order**

| N | L2 error | Order |
| --- | --- | --- |
| 33 | 8.180259e-03 | — |
| 65 | 2.013088e-03 | 2.02 |
| 129 | 4.993394e-04 | 2.01 |
| 257 | 1.243472e-04 | 2.01 |

**F2 — linear TDGL**

| N | L2 error | Relative residual | Cross-check error | Steps |
| --- | --- | --- | --- | --- |
| 33 | 3.209267e-05 | 2.18e-13 | 2.173e-13 | 2000 |
| 65 | 8.147681e-06 | 3.05e-10 | 3.050e-10 | 6000 |
| 129 | 2.052727e-06 | 3.02e-10 | 3.018e-10 | 24000 |

Observed orders: 1.98, 1.99.

**F3 — nonlinear TDGL**

| N | L2 error | Relative residual | Cross-check error | Steps |
| --- | --- | --- | --- | --- |
| 33 | 2.408077e-05 | 5.46e-11 | 4.676e-11 | 1300 |
| 65 | 6.113660e-06 | 1.87e-10 | 1.599e-10 | 4800 |
| 129 | 1.540268e-06 | 2.89e-10 | 2.477e-10 | 18700 |

Observed orders: 1.98, 1.99.

**Eigenvalue evidence** (`eigsh_evidence.json`): discrete `λ_min` agrees
with `eigsh` to 6.2e-13 (N=33) and 1.3e-12 (N=65); the continuous value
`2π²` is wrong by 0.016 at N=33.

## Repository layout

```
tdgl_validation.py        The validators F1/F2/F3, the Richardson solver,
                          the SciPy references, and run_all() (v4, frozen).
mutation_test.py          Mutation suite M1–M8 plus the M1-full spot check.
test_eigsh_evidence.py    Eigsh verification of the discrete eigenvalues.
results.json              Machine-readable results of the executed run.
eigsh_evidence.json       Machine-readable eigenvalue evidence.
provenance.json           Solver configuration, gates, environment,
                          runtime, and SHA-256 hashes of the artifacts.
logs/run.log              Full output of the executed benchmark.
logs/mutation.log         Full output of the executed mutation suite.
logs/eigsh.log            Full output of the executed eigenvalue check.
docs/HISTORY.md           How the code reached v4: the original flaw and
                          four rounds of external audit, with numbers.
examples/original_gigo.py The original flawed code, kept unchanged for
                          reference. Do not use it as a validator.
```

## How to run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python tdgl_validation.py     # benchmark F1–F3; writes results.json
python mutation_test.py       # mutation suite M1–M8 + M1-full
python test_eigsh_evidence.py # eigenvalue evidence; writes eigsh_evidence.json
```

JAX 0.4.35 on CPU is sufficient; no GPU is required. The validator
enables 64-bit mode itself and refuses to run in float32.

## Limitations

Read these before reusing any claim from this repository:

1. **Not physics.** Real `psi`, no magnetic field, `alpha > 0`, an
   artificial forcing term, and no observable. A physicist would not
   accept this as a physics result, and neither should the reader.
2. **No proof of the nonlinear bound.** The bound `2|A|` is enforced by a
   runtime guard, not proved. No maximum principle or monotonicity proof
   for the discrete operator is claimed.
3. **State freeze, not computational fail-fast.** After a bound violation
   the state is frozen, but the residual is still computed inside the
   scan. The freeze saves the state; it does not save the computation.
4. **Mutation classification is keyword-based.** A CAUGHT verdict is
   assigned when an exception of an expected type carries an expected
   gate keyword. For the executed run, every mutation was additionally
   verified against its specific expected gate (see `docs/HISTORY.md`),
   and all nine matched — but the classifier itself could in principle
   produce a false CAUGHT.
5. **Mutation scope is component-level.** Only M1-full exercises the full
   `run_all()` pipeline. The suite measures sensitivity to eight chosen
   faults, not coverage of all possible bugs.
6. **One environment, one parameter set.** The results above were obtained
   once, on CPU, at ξ = 0.1, α = 1.0, β = 1.0, A = 0.5. No parameter sweep
   and no second platform were run.
7. **Richardson iteration is slow by design.** Convergence to a relative
   residual below 1e-8 takes up to 24,000 steps at N=129. This harness
   prioritises auditability over solver speed.

## Principle

Write what was done; do what was written. If a numerical result fails, the
failure is audited — the thresholds are never loosened to make the word
PASS appear.

## License

MIT. See `LICENSE`.
