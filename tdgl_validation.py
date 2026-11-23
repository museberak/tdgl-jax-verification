"""
TDGL validators, version 4 (v4) — response to the third external audit.
Scope is FROZEN: no new TDGL features, only verification-integrity fixes.

P0 (mandatory): B7 psi_bound = 2|A| and A = 0 is rejected; B8 fail-fast
    inside the scan.
P1 (verification integrity): B9 tdgl_solve returns psi and the cross-check
    uses the same solution (no second solve); B10 the mutation-test scope
    is stated explicitly (production validator components, not the full
    run_all() pipeline, except for one full-pipeline spot check, M1-full,
    in mutation_test.py).

What this file is: a verification harness for a Richardson solver of the
time-dependent Ginzburg-Landau (TDGL) equation on manufactured solutions.
What it is not: a physics result. See README.md and docs/HISTORY.md.
"""
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import jax

# Precision policy: 64-bit floating point (x64) is MANDATORY, except when
# the environment variable TDGL_FORCE_F32=1 is set. That variable is used
# ONLY by mutation test M7, to prove that the validator rejects a float32
# configuration. Without it, x64 is always enabled.
_FORCE_F32 = os.environ.get("TDGL_FORCE_F32") == "1"
if not _FORCE_F32:
    jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

VERSION = "v4"


def require_x64():
    if not jax.config.read("jax_enable_x64"):
        raise ValueError(
            "precision policy violation: jax_enable_x64 must be True "
            "(float64 is required; TDGL_FORCE_F32 is intended only for mutation test M7)"
        )


def validate_params(N=None, xi=None, alpha=None, beta=None, A=None,
                    tol=None, max_steps=None, dt=None):
    """INVALID INPUT for invalid parameters — never emit numerical output."""
    def _bad(msg):
        raise ValueError(f"INVALID INPUT: {msg}")
    if N is not None:
        if not isinstance(N, int) or N < 5:
            _bad(f"N must be an int >= 5, got {N!r}")
    for name, v in [("xi", xi), ("alpha", alpha), ("beta", beta), ("A", A)]:
        if v is not None:
            if not math.isfinite(float(v)):
                _bad(f"{name} must be finite, got {v!r}")
    if xi is not None and xi <= 0:
        _bad(f"xi (coherence length) must be > 0, got {xi!r}")
    if A is not None and float(A) == 0.0:
        # B7: A = 0 makes the manufactured solution trivial (g = 0, psi = 0) —
        # a degenerate test that always passes. Reject it explicitly.
        _bad("A must be != 0 (A = 0 makes the manufactured solution trivial)")
    if tol is not None:
        if not math.isfinite(float(tol)) or tol <= 0:
            _bad(f"tol must be finite and > 0, got {tol!r}")
    if max_steps is not None:
        if not isinstance(max_steps, int) or max_steps <= 0:
            _bad(f"max_steps must be an int > 0, got {max_steps!r}")
    if dt is not None:
        if not math.isfinite(float(dt)) or dt <= 0:
            _bad(f"dt must be finite and > 0, got {dt!r}")


# ----------------------------------------------------------------
# Grid & operator
# ----------------------------------------------------------------

def make_grid(N):
    validate_params(N=N)
    x = jnp.linspace(0.0, 1.0, N, dtype=jnp.float64)
    y = jnp.linspace(0.0, 1.0, N, dtype=jnp.float64)
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    return X, Y


def op_laplacian(u, h):
    return (
        -4.0 * u[1:-1, 1:-1]
        + u[0:-2, 1:-1]
        + u[2:, 1:-1]
        + u[1:-1, 0:-2]
        + u[1:-1, 2:]
    ) / (h * h)


def op_tdgl_linear(psi, h, xi, alpha):
    return -(xi ** 2) * op_laplacian(psi, h) + alpha * psi[1:-1, 1:-1]


def op_tdgl_nonlinear(psi, h, xi, alpha, beta):
    p = psi[1:-1, 1:-1]
    return -(xi ** 2) * op_laplacian(psi, h) + alpha * p + beta * p ** 3


def rhs_analytic_linear(psi_int, xi, alpha):
    return (2.0 * (jnp.pi ** 2) * (xi ** 2) + alpha) * psi_int


def rhs_analytic_nonlinear(psi_int, xi, alpha, beta):
    return (2.0 * (jnp.pi ** 2) * (xi ** 2) + alpha) * psi_int + beta * psi_int ** 3


def observed_order(e_coarse, e_fine, h_coarse, h_fine):
    return math.log(e_coarse / e_fine) / math.log(h_coarse / h_fine)


# Exact discrete eigenvalues (B6A) — verified against eigsh;
# see test_eigsh_evidence.py for the stored evidence.
def lambda_min_exact(N, h, xi, alpha):
    return (xi ** 2) * (8.0 / (h * h)) * (math.sin(math.pi / (2 * (N - 1))) ** 2) + alpha


def lambda_max_exact(N, h, xi, alpha):
    return (xi ** 2) * (8.0 / (h * h)) * (math.cos(math.pi / (2 * (N - 1))) ** 2) + alpha


def lambda_max_nonlinear(N, h, xi, alpha, beta, psi_bound):
    if beta < 0:
        raise ValueError(
            "INVALID INPUT: beta < 0 is not supported by this Richardson test — "
            "sup(alpha + 3*beta*psi^2) is attained at psi = 0, not at psi_bound."
        )
    if beta == 0:
        return lambda_max_exact(N, h, xi, alpha)
    return lambda_max_exact(N, h, xi, alpha) + 3.0 * beta * (psi_bound ** 2)


# ----------------------------------------------------------------
# F1
# ----------------------------------------------------------------

def poisson_validate(N):
    require_x64()
    validate_params(N=N)
    X, Y = make_grid(N)
    h = 1.0 / (N - 1)
    u_true = jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * Y)
    f = 2.0 * (jnp.pi ** 2) * u_true
    err = -op_laplacian(u_true, h) - f[1:-1, 1:-1]
    assert jnp.all(jnp.isfinite(err)), "F1: non-finite"
    L2 = float(jnp.sqrt(jnp.mean(err ** 2)))
    Linf = float(jnp.max(jnp.abs(err)))
    print(f"F1 N={N:3d} h={h:.6f}  L2={L2:.6e}  Linf={Linf:.6e}")
    return L2


# ----------------------------------------------------------------
# Richardson solver, v4 — hard iteration cap + FAIL-FAST inside the scan (B8)
# ----------------------------------------------------------------

def _solve_richardson(psi0, g_int, h, xi, alpha, beta, dt, lam_max, tol,
                      max_steps, psi_bound=None, chunk=1000):
    validate_params(tol=tol, max_steps=max_steps, dt=dt)
    if not math.isfinite(float(lam_max)) or lam_max <= 0:
        raise ValueError(f"INVALID INPUT: lam_max must be finite and > 0, got {lam_max!r}")
    if dt * lam_max >= 2.0:
        raise ValueError(
            f"stability criterion violated: dt*lambda_max={dt*lam_max:.3f} >= 2 "
            "(Richardson requires dt*lambda_max < 2)"
        )

    def apply_op(psi):
        if beta is None:
            return op_tdgl_linear(psi, h, xi, alpha)
        return op_tdgl_nonlinear(psi, h, xi, alpha, beta)

    # B8: carry = (psi, running_max, violated, violation_step, violation_max).
    # If violated is already True, the state is FROZEN (psi no longer changes):
    # subsequent iterations become logical no-ops instead of wasted steps
    # that would change the state. If a new candidate violates the bound,
    # that step is NOT accepted: the state is frozen at the last valid
    # state, and both the step and the violating candidate value
    # (violation_max) are recorded, so that the error message does not
    # claim that the frozen maximum exceeds the bound when it does not.
    # xs = the step index inside the chunk, so that violation_step is tracked.
    #
    # Known limitation (see README.md): this is a STATE freeze, not a
    # computational fail-fast — the residual is still computed inside the
    # scan after a violation. The error is raised after the chunk finishes.
    def step(carry, i):
        psi, m, violated, vstep, vmax = carry
        resid = apply_op(psi) - g_int
        psi_cand = psi.at[1:-1, 1:-1].set(psi[1:-1, 1:-1] - dt * resid)
        if psi_bound is None:
            return (psi_cand, jnp.maximum(m, jnp.max(jnp.abs(psi_cand))),
                    violated, vstep, vmax), jnp.sqrt(jnp.mean(resid ** 2))
        m_cand = jnp.maximum(m, jnp.max(jnp.abs(psi_cand)))
        newly = (~violated) & (m_cand > psi_bound + 1e-12)
        # Freeze: if already violated, keep the old psi; if newly violated,
        # also keep the old psi (reject the violating step) and record this step.
        psi_new = jnp.where(violated | newly, psi, psi_cand)
        m_new = jnp.where(violated | newly, m, m_cand)
        violated_new = violated | newly
        vstep_new = jnp.where(newly, i, vstep)
        vmax_new = jnp.where(newly, m_cand, vmax)
        return (psi_new, m_new, violated_new, vstep_new, vmax_new), jnp.sqrt(jnp.mean(resid ** 2))

    cache = {}

    def run_chunk(carry, length):
        if length not in cache:
            cache[length] = jax.jit(
                lambda c: jax.lax.scan(step, c, jnp.arange(length)))
        return cache[length](carry)

    carry = (psi0, jnp.max(jnp.abs(psi0)), jnp.array(False), jnp.array(-1),
             jnp.array(0.0))
    total = 0
    g_norm = float(jnp.sqrt(jnp.mean(g_int ** 2)))
    max_over_iterates = 0.0
    while total < max_steps:
        chunk_now = min(chunk, max_steps - total)
        carry, _ = run_chunk(carry, chunk_now)
        total += chunk_now
        psi, m, violated, vstep, vmax = carry
        max_over_iterates = max(max_over_iterates, float(m))
        if bool(violated):
            raise RuntimeError(
                f"nonlinear bound violated (fail-fast, frozen): "
                f"candidate max|psi|={float(vmax):.3e} > bound "
                f"{psi_bound:.3e} at global step {total - chunk_now + int(vstep)}; "
                f"state frozen at max|psi|={max_over_iterates:.3e}"
            )
        r = apply_op(psi) - g_int
        if float(jnp.sqrt(jnp.mean(r ** 2))) < tol:
            break
    assert total <= max_steps, f"hard cap violated: {total} > {max_steps}"
    psi, _, _, _, _ = carry
    final_r = apply_op(psi) - g_int
    final_res = float(jnp.sqrt(jnp.mean(final_r ** 2)))
    final_rel = final_res / max(g_norm, 1e-300)
    if not math.isfinite(final_res):
        raise RuntimeError("diverged: the final residual is non-finite")
    return psi, total, final_res, final_rel, max_over_iterates


def check_boundary(psi, tol=1e-12):
    """Boundary-specific assertion (M6): the homogeneous Dirichlet boundary must be 0."""
    b = float(jnp.max(jnp.abs(jnp.concatenate([
        psi[0, :], psi[-1, :], psi[:, 0], psi[:, -1]]))))
    if b > tol:
        raise AssertionError(f"boundary condition violation: max|psi_boundary|={b:.3e} > {tol:.0e}")
    return b


def tdgl_solve(N, nonlinear=False, xi=0.1, alpha=1.0, beta=1.0,
               A=0.5, tol=1e-10, max_steps=100000):
    require_x64()
    validate_params(N=N, xi=xi, alpha=alpha, beta=beta, A=A, tol=tol, max_steps=max_steps)
    tag = "F3-nonlinear" if nonlinear else "F2-linear"
    X, Y = make_grid(N)
    h = 1.0 / (N - 1)
    lam_min = lambda_min_exact(N, h, xi, alpha)
    if lam_min <= 0:
        raise ValueError(f"INVALID INPUT: discrete lambda_min={lam_min:.3e} <= 0; the operator is not SPD.")
    psi_true = A * jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * Y)
    psi_int = psi_true[1:-1, 1:-1]
    if nonlinear:
        g_int = rhs_analytic_nonlinear(psi_int, xi, alpha, beta)
        # B7: the bound uses |A|, not A. A negative amplitude is valid
        # mathematically (the solution simply changes sign); a negative
        # bound is impossible because max|psi| >= 0. A = 0 is already
        # rejected in validate_params.
        psi_bound = 2.0 * abs(A)
        lam = lambda_max_nonlinear(N, h, xi, alpha, beta, psi_bound)
        b = beta
        chunk = 100  # B3: small chunk for the nonlinear audit
    else:
        g_int = rhs_analytic_linear(psi_int, xi, alpha)
        psi_bound = None
        lam = lambda_max_exact(N, h, xi, alpha)
        b = None
        chunk = 1000
    dt = 1.0 / lam
    psi0 = jnp.zeros((N, N), dtype=jnp.float64)
    psi, steps, res, rel, max_it = _solve_richardson(
        psi0, g_int, h, xi, alpha, b, dt, lam, tol, max_steps, psi_bound, chunk)
    assert jnp.all(jnp.isfinite(psi)), f"{tag}: non-finite solution"
    assert steps <= max_steps
    assert rel < 1e-8, f"{tag}: relative residual {rel:.2e} >= 1e-8"
    check_boundary(psi)
    err = psi - psi_true
    L2 = float(jnp.sqrt(jnp.mean(err ** 2)))
    Linf = float(jnp.max(jnp.abs(err)))
    print(f"{tag} N={N:3d} h={h:.6f} dt={dt:.3e} lam={lam:.1f} lam_min={lam_min:.3f} "
          f"steps={steps} res_final={res:.2e} rel={rel:.2e} max|psi|_it={max_it:.3f} "
          f"L2={L2:.6e} Linf={Linf:.6e}")
    # B9: return psi so that the cross-check uses the SAME solution,
    # not a second solve. psi is excluded from the provenance JSON in run_all().
    return {"L2": L2, "Linf": Linf, "steps": steps, "res_final": res,
            "rel": rel, "max_iterate": max_it, "dt": dt, "lam_max": lam,
            "psi": psi}


# ----------------------------------------------------------------
# Independent SciPy reference (B4) — with final-residual verification
# ----------------------------------------------------------------

def scipy_reference_linear(N, xi=0.1, alpha=1.0, A=0.5):
    import numpy as np
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    n = N - 2
    h = 1.0 / (N - 1)
    T = sparse.diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(n, n), format="csc")
    I_n = sparse.eye(n, format="csc")
    Lap = (sparse.kron(T, I_n) + sparse.kron(I_n, T)) / (h * h)
    Ah = -(xi ** 2) * Lap + alpha * sparse.eye(n * n, format="csc")
    x = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, x, indexing="ij")
    psi_true = A * np.sin(np.pi * X) * np.sin(np.pi * Y)
    g = ((2 * np.pi ** 2 * xi ** 2 + alpha) * psi_true[1:-1, 1:-1]).reshape(-1)
    sol = spsolve(Ah.tocsc(), g)
    assert np.all(np.isfinite(sol)), "linear reference is non-finite"
    r = Ah @ sol - g
    rel = float(np.sqrt(np.mean(r ** 2))) / max(float(np.sqrt(np.mean(g ** 2))), 1e-300)
    assert rel < 1e-8, f"linear reference: relative residual {rel:.2e} >= 1e-8"
    return sol.reshape(n, n), psi_true, rel


def scipy_reference_nonlinear_newton(N, xi=0.1, alpha=1.0, beta=1.0, A=0.5,
                                     tol=1e-12, max_iter=50):
    import numpy as np
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    n = N - 2
    h = 1.0 / (N - 1)
    T = sparse.diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(n, n), format="csc")
    I_n = sparse.eye(n, format="csc")
    Lap = (sparse.kron(T, I_n) + sparse.kron(I_n, T)) / (h * h)
    A_lin = -(xi ** 2) * Lap + alpha * sparse.eye(n * n, format="csc")
    x = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, x, indexing="ij")
    psi_true = A * np.sin(np.pi * X) * np.sin(np.pi * Y)
    g = (((2 * np.pi ** 2 * xi ** 2 + alpha) * psi_true[1:-1, 1:-1]
          + beta * psi_true[1:-1, 1:-1] ** 3)).reshape(-1)
    psi = np.zeros(n * n)
    for _ in range(max_iter):
        r = A_lin @ psi + beta * psi ** 3 - g
        if float(np.sqrt(np.mean(r ** 2))) < tol:
            break
        J = A_lin + sparse.diags(3 * beta * psi ** 2, format="csc")
        psi = psi + spsolve(J.tocsc(), -r)
    else:
        raise RuntimeError("Newton reference did not converge")
    # B4B: the final residual is recomputed from the final solution and gated
    r_final = A_lin @ psi + beta * psi ** 3 - g
    rel = float(np.sqrt(np.mean(r_final ** 2))) / max(float(np.sqrt(np.mean(g ** 2))), 1e-300)
    assert np.all(np.isfinite(psi)), "Newton reference is non-finite"
    assert rel < 1e-8, f"Newton reference: final relative residual {rel:.2e} >= 1e-8"
    return psi.reshape(n, n), psi_true, rel


def cross_impl(N, nonlinear=False, xi=0.1, alpha=1.0, beta=1.0, A=0.5,
               tol=1e-10, max_steps=100000, sol=None):
    """Cross-check with its own gates (B2/B4) — it does not assume the solver
    is correct. B9: uses psi from an already validated solution (one solve
    only). If sol is None, run tdgl_solve once here."""
    if sol is None:
        sol = tdgl_solve(N, nonlinear=nonlinear, xi=xi, alpha=alpha, beta=beta,
                         A=A, tol=tol, max_steps=max_steps)
    # Solver gates (repeated explicitly here so that the cross-check stands on its own)
    assert sol["steps"] <= max_steps, "cross: steps > max_steps"
    assert sol["rel"] < 1e-8, f"cross: relative residual {sol['rel']:.2e} >= 1e-8"
    assert math.isfinite(sol["L2"]), "cross: L2 non-finite"
    psi = sol["psi"]  # B9: the same solution, without a second solve
    assert jnp.all(jnp.isfinite(psi)), "cross: JAX solution is non-finite"
    check_boundary(psi)
    # Independent reference, with its final residual gated
    if nonlinear:
        ref_int, _, ref_rel = scipy_reference_nonlinear_newton(N, xi, alpha, beta, A)
    else:
        ref_int, _, ref_rel = scipy_reference_linear(N, xi, alpha, A)
    assert math.isfinite(float(ref_rel)) and ref_rel < 1e-8, f"cross: reference relative residual {ref_rel:.2e}"
    import numpy as np
    jax_int = np.array(psi[1:-1, 1:-1])
    assert np.all(np.isfinite(ref_int)), "cross: reference is non-finite"
    denom = float(np.sqrt(np.mean(ref_int ** 2)))
    e_cross = float(np.sqrt(np.mean((jax_int - ref_int) ** 2))) / max(denom, 1e-300)
    tag = "F3" if nonlinear else "F2"
    print(f"  cross {tag} N={N}  E_cross={e_cross:.3e} rel_solver={sol['rel']:.2e} "
          f"rel_ref={ref_rel:.2e} steps={sol['steps']} (gate < 1e-6)")
    assert e_cross < 1e-6, f"cross-check failed: {e_cross:.3e}"
    return {"e_cross": e_cross, "rel_solver": sol["rel"], "rel_ref": ref_rel,
            "steps": sol["steps"], "L2": sol["L2"]}


# ----------------------------------------------------------------
# run_all + provenance (B4C)
# ----------------------------------------------------------------

def run_all():
    require_x64()
    import scipy
    print(f"JAX {jax.__version__}  x64={jax.config.read('jax_enable_x64')}  "
          f"dtype={jnp.zeros(1).dtype}  scipy {scipy.__version__}")
    out = {"version": VERSION, "jax": jax.__version__, "scipy": scipy.__version__,
           "x64": bool(jax.config.read("jax_enable_x64")),
           "params": {"xi": 0.1, "alpha": 1.0, "beta": 1.0, "A": 0.5, "tol": 1e-10},
           "timestamp_utc": datetime.now(timezone.utc).isoformat()}
    print("\n=== F1 Poisson (33,65,129,257) ===")
    Ns = [33, 65, 129, 257]
    f1 = {N: poisson_validate(N) for N in Ns}
    hs = {N: 1.0 / (N - 1) for N in Ns}
    out["F1"] = {"L2": {str(N): f1[N] for N in Ns}, "orders": {}}
    for a, b in [(33, 65), (65, 129), (129, 257)]:
        o = observed_order(f1[a], f1[b], hs[a], hs[b])
        out["F1"]["orders"][f"{a}->{b}"] = o
        print(f"F1 order {a}->{b}: {o:.2f}")
        assert 1.8 <= o <= 2.2, f"F1 failed order {a}->{b}: {o:.2f}"

    print("\n=== F2 linear (33,65,129) ===")
    f2, cross2 = {}, {}
    for N in [33, 65, 129]:
        f2[N] = tdgl_solve(N, nonlinear=False)
        # B9: pass the same solution to the cross-check — no second solve.
        cross2[N] = cross_impl(N, nonlinear=False, sol=f2[N])
    out["F2"] = {"L2": {str(N): f2[N]["L2"] for N in f2},
                 "rel": {str(N): f2[N]["rel"] for N in f2},
                 "cross": {str(N): cross2[N] for N in cross2}, "orders": {}}
    for a, b in [(33, 65), (65, 129)]:
        o = observed_order(f2[a]["L2"], f2[b]["L2"], hs[a], hs[b])
        out["F2"]["orders"][f"{a}->{b}"] = o
        print(f"F2 order {a}->{b}: {o:.2f}")
        assert 1.8 <= o <= 2.2

    print("\n=== F3 nonlinear (33,65,129) ===")
    f3, cross3 = {}, {}
    for N in [33, 65, 129]:
        f3[N] = tdgl_solve(N, nonlinear=True)
        cross3[N] = cross_impl(N, nonlinear=True, sol=f3[N])
    out["F3"] = {"L2": {str(N): f3[N]["L2"] for N in f3},
                 "rel": {str(N): f3[N]["rel"] for N in f3},
                 "cross": {str(N): cross3[N] for N in cross3}, "orders": {}}
    for a, b in [(33, 65), (65, 129)]:
        o = observed_order(f3[a]["L2"], f3[b]["L2"], hs[a], hs[b])
        out["F3"]["orders"][f"{a}->{b}"] = o
        print(f"F3 order {a}->{b}: {o:.2f}")
        assert 1.8 <= o <= 2.2

    out["verdict"] = "PASS"
    path = Path(__file__).parent / "results.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\n=== ALL VALIDATORS PASSED (v4) — provenance: {path} ===")
    return out


if __name__ == "__main__":
    run_all()
