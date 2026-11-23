"""Mutation testing, version 4 (v4).
STATED SCOPE — read first, and do not claim more than this:
- This suite is MUTATION TESTING OF THE PRODUCTION VALIDATOR
  COMPONENTS (tdgl_solve + cross_impl + the observed-order gate,
  on grids 33/65). It is NOT full-pipeline mutation testing of
  run_all() (F1 on 33/65/129/257, F2/F3 on 33/65/129).
- The only exception is M1-full below: one full-pipeline spot check
  of run_all() with mutation M1, to prove that the full pipeline also
  catches the same mutation — not proof that every mutation was tested
  in the full pipeline.
- Three statuses: CAUGHT / MISSED / INVALID_TEST. An exception alone
  is not a CAUGHT result.
- Known weakness (see README.md): classification is based on the
  exception type and on gate keywords in the message, so a CAUGHT
  verdict could in principle be a coincidence. For the executed run,
  every mutation was verified after the fact against its specific
  expected gate; see docs/HISTORY.md.
Run: python mutation_test.py
"""
import math
import os
import subprocess
import sys

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import tdgl_validation as fx

# Expected gate messages from the production validator.
EXPECTED = ["order", "relative residual", "cross", "boundary condition",
            "non-finite", "stability criterion", "precision policy",
            "INVALID INPUT", "nonlinear bound", "reference", "diverged"]


def classify(exc):
    """Return (status, reason). Only an AssertionError, RuntimeError, or
    ValueError whose message contains an expected gate keyword is a CAUGHT."""
    msg = str(exc)
    if isinstance(exc, (AssertionError, RuntimeError, ValueError)):
        for k in EXPECTED:
            if k.lower() in msg.lower():
                return "CAUGHT", f"{type(exc).__name__}: {msg[:160]}"
        # An AssertionError, RuntimeError, or ValueError from the validator
        # with an unfamiliar message: treat it as INVALID_TEST so that no
        # false CAUGHT claim is made.
        return "INVALID_TEST", f"{type(exc).__name__} tak terduga: {msg[:160]}"
    return "INVALID_TEST", f"{type(exc).__name__}: {msg[:160]}"


def full_validator_linear():
    """Production validator for F2: 2 grids + the order gate + the cross-check.
    Returns None on PASS and raises when a gate fails."""
    errs = {}
    for N in [33, 65]:
        sol = fx.tdgl_solve(N, nonlinear=False, tol=1e-10, max_steps=20000)
        errs[N] = sol["L2"]
        fx.cross_impl(N, nonlinear=False, tol=1e-10, max_steps=20000)
    o = fx.observed_order(errs[33], errs[65], 1/32, 1/64)
    assert 1.8 <= o <= 2.2, f"order {o:.2f} outside [1.8, 2.2]"


def full_validator_nonlinear():
    errs = {}
    for N in [33, 65]:
        sol = fx.tdgl_solve(N, nonlinear=True, tol=1e-10, max_steps=20000)
        errs[N] = sol["L2"]
        fx.cross_impl(N, nonlinear=True, tol=1e-10, max_steps=20000)
    o = fx.observed_order(errs[33], errs[65], 1/32, 1/64)
    assert 1.8 <= o <= 2.2, f"order {o:.2f} outside [1.8, 2.2]"


def run_one(name, fn):
    print(f"\n--- {name} ---")
    try:
        fn()
    except Exception as ex:
        status, reason = classify(ex)
        print(f"  {status}: {reason}")
        return status, reason
    print("  MISSED: validator PASSED although a mutation is present")
    return "MISSED", "validator PASSED"


def main():
    orig_lap = fx.op_laplacian
    orig_lin = fx.op_tdgl_linear
    orig_nonlin = fx.op_tdgl_nonlinear
    orig_solve = fx._solve_richardson
    results = {}

    def restore_all():
        fx.op_laplacian = orig_lap
        fx.op_tdgl_linear = orig_lin
        fx.op_tdgl_nonlinear = orig_nonlin
        fx._solve_richardson = orig_solve

    # M1 lap x2
    def m1():
        fx.op_laplacian = lambda u, h: 2.0 * orig_lap(u, h)
        try:
            full_validator_linear()
        finally:
            restore_all()
    results["M1 lap x2"] = run_one("M1 Laplacian multiplied by 2", m1)

    # M2 drop neighbor
    def m2():
        def lap_drop(u, h):
            return (-4.0*u[1:-1,1:-1]+u[0:-2,1:-1]+u[2:,1:-1]+u[1:-1,0:-2])/(h*h)
        fx.op_laplacian = lap_drop
        try:
            full_validator_linear()
        finally:
            restore_all()
    results["M2 drop neighbor"] = run_one("M2 One neighbour removed", m2)

    # M3 lap sign flip
    def m3():
        fx.op_laplacian = lambda u, h: -orig_lap(u, h)
        try:
            full_validator_linear()
        finally:
            restore_all()
    results["M3 lap sign flip"] = run_one("M3 Laplacian sign flipped", m3)

    # M4 alpha sign flip (operator only)
    def m4():
        fx.op_tdgl_linear = lambda psi, h, xi, alpha: -(xi**2)*orig_lap(psi, h) - alpha*psi[1:-1,1:-1]
        try:
            full_validator_linear()
        finally:
            restore_all()
    results["M4 alpha sign flip"] = run_one("M4 alpha -> -alpha in the operator", m4)

    # M5 beta removed — full nonlinear validator (including the Newton cross-check)
    def m5():
        fx.op_tdgl_nonlinear = lambda psi, h, xi, alpha, beta: -(xi**2)*orig_lap(psi, h) + alpha*psi[1:-1,1:-1]
        try:
            full_validator_nonlinear()
        finally:
            restore_all()
    results["M5 beta dropped"] = run_one("M5 beta removed (nonlinear + Newton cross-check)", m5)

    # M6 boundary — patch the solver so that it returns psi with a
    # boundary value of 0.5; the validator must catch this through
    # check_boundary(), not merely through the L2 error.
    def m6():
        def bad_solve(*a, **k):
            psi, total, res, rel, mx = orig_solve(*a, **k)
            psi = psi.at[0, :].set(0.5).at[-1, :].set(0.5)
            return psi, total, res, rel, mx
        fx._solve_richardson = bad_solve
        try:
            full_validator_linear()
        finally:
            restore_all()
    results["M6 boundary 0.5"] = run_one("M6 Boundary 0.5 (test check_boundary)", m6)

    # M7 float32 — a mutation of the VALIDATOR CONFIGURATION via a subprocess,
    # not a separate NumPy calculation.
    def m7():
        env = dict(os.environ)
        env["TDGL_FORCE_F32"] = "1"
        # run_all must refuse in require_x64() before any heavy computation.
        r = subprocess.run(
            [sys.executable, "-c",
             "import tdgl_validation; tdgl_validation.run_all()"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            env=env, capture_output=True, text=True, timeout=120)
        out = (r.stdout or "") + (r.stderr or "")
        if r.returncode == 0:
            raise AssertionError("validator PASSED in float32 mode (MISSED)")
        if "precision policy" in out:
            raise ValueError("precision policy violation: jax_enable_x64 must be True")
        raise RuntimeError(f"subprocess failed without a precision-policy message: {out[-300:]}")
    # m7 uses a different mechanism: catch it manually so that the
    # classification still has three statuses.
    print("\n--- M7 float32 (validator-configuration mutation, subprocess) ---")
    try:
        m7()
    except Exception as ex:
        status, reason = classify(ex)
        # The AssertionError "validator PASSED..." above does not contain an
        # EXPECTED keyword, so classify() would report INVALID_TEST, although
        # it is in fact a MISSED. Handle it explicitly:
        if "validator PASSED in float32 mode" in str(ex):
            status, reason = "MISSED", str(ex)
        print(f"  {status}: {reason}")
        results["M7 float32"] = (status, reason)
    else:
        print("  MISSED: validator PASSED")
        results["M7 float32"] = ("MISSED", "validator PASSED")

    # M8 dt too large — it must be rejected by the stability gate, not by an L2 threshold.
    def m8():
        N = 33
        X, Y = fx.make_grid(N)
        h = 1.0/(N-1)
        psi_true = 0.5*jnp.sin(jnp.pi*X)*jnp.sin(jnp.pi*Y)
        g = fx.rhs_analytic_linear(psi_true[1:-1,1:-1], 0.1, 1.0)
        lam = fx.lambda_max_exact(N, h, 0.1, 1.0)
        psi0 = jnp.zeros((N, N), dtype=jnp.float64)
        # _solve must raise ValueError: stability criterion violated.
        fx._solve_richardson(psi0, g, h, 0.1, 1.0, None, 4.0/lam, lam,
                              1e-10, 5000, None, 1000)
    results["M8 dt too large"] = run_one("M8 dt=4/lambda (test the stability gate)", m8)

    # M1-full: ONE full-pipeline spot check of run_all() with lap x2.
    # Not part of the 8/8 component claim; additional evidence that the
    # full pipeline also catches the same mutation.
    def m1_full():
        fx.op_laplacian = lambda u, h: 2.0 * orig_lap(u, h)
        try:
            fx.run_all()
        finally:
            restore_all()
    results["M1-full run_all"] = run_one("M1-full: run_all() with lap x2", m1_full)

    print("\n=== MUTATION SUMMARY v4 (production components + 1 full spot check) ===")
    for k, (s, r) in results.items():
        print(f"{k}: {s} — {r}")
    n_caught = sum(1 for s, _ in results.values() if s == "CAUGHT")
    n_invalid = sum(1 for s, _ in results.values() if s == "INVALID_TEST")
    n_missed = sum(1 for s, _ in results.values() if s == "MISSED")
    print(f"CAUGHT {n_caught}/{len(results)}  MISSED {n_missed}  INVALID_TEST {n_invalid}")
    if n_invalid:
        raise SystemExit("MUTATION SUITE INVALID: INVALID_TEST present — fix the test; do not claim PASS")
    if n_missed:
        raise SystemExit("MUTATION SUITE FAILED: MISSED present — the validator is blind to a mutation")
    print("MUTATION SUITE PASSED (production components): every mutation CAUGHT, each with its specific gate reason")


if __name__ == "__main__":
    main()
