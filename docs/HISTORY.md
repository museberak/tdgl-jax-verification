# History: How the Validator Reached v4

This document records what was actually done, in order, including the
mistakes that were found. The code in this repository is version 4 (v4),
frozen after the fourth round of external audit. Earlier versions are not
shipped; their defects and the numbers they produced are recorded here so
that the claims in the README can be checked against the work.

## 0. The original code (22 November 2026)

The starting code is preserved in `examples/original_gigo.py`. It had three
defects, measured before any fix:

- **Inverse crime.** The right-hand side was built by applying the
  production operator to the manufactured solution, so the test could not
  fail. Demonstrated directly: multiplying the operator by 2 still gave a
  residual of exactly 0.0.
- **Float32 collapse.** The Poisson error stopped converging in float32
  (observed orders 2.03, then 0.13); in float64 the orders were 2.07, 2.03.
- **Hidden discretisation error.** The operator-built right-hand side
  differed from the analytic right-hand side by 4.36e-05 (N=32) and
  1.07e-05 (N=64).

The code also used a hardcoded time step, a plain Python loop, no 64-bit
mode, and no entry point.

## 1. Version 1 — the inverse crime removed

Fixes: analytic right-hand side `g = (2π²ξ² + α)ψ + βψ³`, float64,
`dt = 1/λ_max`, `jax.jit` with `lax.scan`, residual tolerance 1e-10, and an
order gate of [1.8, 2.2]. Measured results: F1 orders 2.07/2.03; F2 orders
2.02/2.01; F3 orders 2.02/2.01. A falsification check (operator ×2 with the
analytic right-hand side) gave L2 3.42e-02 against 3.42e-05 for the correct
operator — a factor of 1,000, which is what a working test must show.

## 2. Version 2 — response to the first audit (7 findings, all correct)

The audit found: the grid refinement was not exactly a factor of 2
(N = 32/64/128 gives ratios 63/31 = 2.032 and 127/63 = 2.016, while the
order formula assumed 2 — so the reported order of 2.02 was in fact 1.97);
`max_steps` was not a hard cap (a request for 1,500 steps ran 2,000);
the printed residual belonged to the previous iterate (off by one); the
nonlinear stability bound used the manufactured amplitude without proof
and was invalid for β < 0; a small residual was treated as if it implied
an accurate solution; there was no independent reference; and there was
no mutation testing.

Version 2 fixed all seven: grids 33/65/129/257 with the observed order
computed from the actual spacing ratio, a true hard cap (verified:
500 → 500, 1,500 → 1,500), the final residual recomputed from the
returned solution plus a relative-residual gate, β < 0 rejected with a
runtime bound check, a SciPy sparse reference (cross errors 1e-13 to
1e-10), and a mutation suite with 8/8 caught. Results: F1 2.02/2.01/2.01,
F2 1.98/1.99, F3 1.98/1.99.

## 3. Version 3 — response to the second audit (verification integrity)

The second audit found that the cross-check computed the relative residual
but never gated it; the nonlinear bound was checked only at chunk
endpoints, not per iteration; the Newton reference stopped on a
pre-update residual; the results were not stored in a machine-readable
form; and the mutation suite tested a custom gate, not the production
validator, while treating any exception as CAUGHT.

Version 3 fixed these: explicit cross-check gates (steps, relative
residual, finite solution and reference, reference residual), a running
maximum tracked inside `lax.scan` at every iteration, the Newton final
residual recomputed and gated, provenance written to JSON, exact discrete
eigenvalues (verified against `eigsh`: N=33 numeric 19.723360, matching
the formula; the continuous value 19.739209 is wrong by 0.016),
parameter guards raising INVALID INPUT, and a mutation suite that calls
the full production validator with three statuses (CAUGHT / MISSED /
INVALID_TEST). Results: F1 2.02/2.01/2.01, F2 1.98/1.99, F3 1.98/1.99,
cross errors 1e-13 to 1e-10, mutations 8/8 CAUGHT with 0 MISSED and
0 INVALID_TEST.

## 4. Version 4 — response to the third audit (P0 and P1)

The third audit closed B7 and B9, found a new problem in B8, and asked for
the mutation scope to be stated honestly:

- **B7.** `psi_bound = 2A` became negative for a negative amplitude.
  Fixed as `2|A|`; `A = 0` is rejected as a degenerate test. Verified:
  A = −0.5 gives L2 2.408077e-05, identical to A = +0.5.
- **B8.** The bound was still checked only after each chunk. Fixed with a
  fail-fast inside the scan: the carry holds a violation flag, the
  violating step is rejected, and the state freezes at the last valid
  state. Verified with a forced bound of 0.1: the violation is detected
  at global step 12, candidate 1.005e-01 > bound 1.000e-01, frozen at
  9.347e-02. During implementation, the first error message wrongly
  reported the frozen maximum as if it exceeded the bound; this was
  corrected by recording the violating candidate value separately.
- **B9.** The cross-check solved the system a second time because
  `tdgl_solve` did not return the solution. Fixed: it now returns `psi`
  and the cross-check uses the same solution. Benchmark runtime fell
  from 30 s to about 18 s, with identical numbers.
- **B10.** The mutation scope is now stated explicitly in the file
  header: component-level testing, not full-pipeline testing, plus one
  full-pipeline spot check (M1-full).

Results (v4, executed): F1 2.02/2.01/2.01, F2 1.98/1.99, F3 1.98/1.99,
cross errors 1e-13 to 1e-10, mutations 9/9 CAUGHT (8 components +
M1-full), 0 MISSED, 0 INVALID_TEST.

## 5. Version 4, execution audit — no v5

The fourth audit concluded that v4 was ready for execution and that no
v5 should be created: the frozen source was to be run as it was. The
following were closed without changing the source:

- **Violation-step convention.** The audit corrected itself: the formula
  `total − chunk_now + vstep` is consistent with a zero-based
  convention, not off by one; the convention was undocumented. Verified
  empirically: chunk sizes 100 and 10 both report global step 12 for
  the forced bound. The human-readable step is the reported index
  plus one.
- **State freeze vs. computational fail-fast.** The freeze protects the
  state, but the residual is still computed inside the scan after a
  violation. This is a known limitation, recorded in the README, not a
  defect that was silently fixed.
- **Eigsh evidence.** The claim that the discrete eigenvalues were
  verified against `eigsh` was a comment without stored evidence. It is
  now a script (`test_eigsh_evidence.py`) with stored output
  (`eigsh_evidence.json`).
- **Provenance.** The run provenance stored only the main parameters.
  It is now completed in `provenance.json` (solver configuration, all
  gates, environment, runtime, SHA-256 hashes).
- **Mutation classifier.** Classification by exception type and message
  keyword can in principle produce a false CAUGHT. For the executed
  run, every mutation was verified after the fact against its specific
  expected gate (M1 cross-check, M2 cross-check, M3 divergence, M4
  relative residual, M5 cross-check, M6 boundary, M7 precision policy,
  M8 stability, M1-full observed order): 9/9 matched, so there is no
  false CAUGHT in this run. The classifier weakness remains a design
  limitation; fixing it belongs to a future rewrite of the suite, not
  to a change of the frozen v4.

## 6. Translation to this repository

The Indonesian sources (v4) were translated into formal English for
this repository: comments, docstrings, and user-facing messages only.
No logic was changed. The translated code was re-executed and reproduced
the v4 numbers exactly (`results.json` is numerically identical to the
v4 results), the mutation suite again gave 9/9 CAUGHT with the same
reasons, and the eigenvalue evidence passed again.
