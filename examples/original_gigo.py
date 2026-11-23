# NOTE (English header added for this repository; the code below is
# unchanged from the original submission).
# This is the ORIGINAL flawed code that the verification work started
# from. It is preserved for reference only. Do NOT use it as a validator:
# it builds the right-hand side by applying the operator under test
# (an inverse crime), runs in float32, and has no entry point.
# See README.md ("The problem that was fixed") and docs/HISTORY.md.

import jax
import jax.numpy as jnp

# ============================================================
# GEOMETRY & GRID
# ============================================================

def make_grid(N):
    x = jnp.linspace(0.0, 1.0, N)
    y = jnp.linspace(0.0, 1.0, N)
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    return X, Y

# ============================================================
# OPERATOR PRODUKSI (STAND-IN)
# Nanti di v9: ganti isi fungsi ini dengan operator asli
# ============================================================

def op_laplacian_prod(u, h):
    """Laplacian produksi (di sini: 5-point stencil)."""
    return (
        -4.0 * u[1:-1, 1:-1]
        + u[0:-2, 1:-1]
        + u[2:  , 1:-1]
        + u[1:-1, 0:-2]
        + u[1:-1, 2:  ]
    ) / (h * h)

def op_tdgl_linear_prod(psi, h, xi, alpha):
    """Operator TDGL linear produksi: -xi^2 Δψ + α ψ."""
    lap = op_laplacian_prod(psi, h)
    psi_int = psi[1:-1, 1:-1]
    return - (xi**2) * lap + alpha * psi_int

def op_tdgl_nonlinear_prod(psi, h, xi, alpha, beta):
    """Operator TDGL nonlinear produksi: -xi^2 Δψ + α ψ + β ψ^3."""
    lap = op_laplacian_prod(psi, h)
    psi_int = psi[1:-1, 1:-1]
    return - (xi**2) * lap + alpha * psi_int + beta * psi_int**3

# ============================================================
# F1 – POISSON (OPERATOR CHECK)
# ============================================================

def poisson_validate(N):
    print(f"\n=== F1: POISSON VALIDATOR (N={N}) ===")

    X, Y = make_grid(N)
    h = 1.0 / (N - 1)

    u_true = jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * Y)
    f = 2.0 * (jnp.pi**2) * u_true

    Au_int = -op_laplacian_prod(u_true, h)
    f_int = f[1:-1, 1:-1]

    L2 = float(jnp.sqrt(jnp.mean((Au_int - f_int) ** 2)))
    Linf = float(jnp.max(jnp.abs(Au_int - f_int)))

    print(f"L2 error(A(u_true), f)   : {L2:.6e}")
    print(f"Linf error(A(u_true), f) : {Linf:.6e}")

# ============================================================
# F2 – TDGL LINEAR STEADY (PSEUDO-TIME, OP PRODUKSI)
# ============================================================

def tdgl_linear_steady_solve(
    N,
    xi=0.1,
    alpha=1.0,
    A=0.5,
    dt=5e-4,
    steps=20000,
    print_every=5000,
):
    print(f"\n=== F2: TDGL LINEAR STEADY (N={N}) ===")

    X, Y = make_grid(N)
    h = 1.0 / (N - 1)

    psi_true = A * jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * Y)

    # RHS dibangun dengan operator produksi
    g_int = op_tdgl_linear_prod(psi_true, h, xi, alpha)

    psi = jnp.zeros((N, N))

    for k in range(steps):
        Nh = op_tdgl_linear_prod(psi, h, xi, alpha)

        resid_int = Nh - g_int
        resid_full = jnp.zeros_like(psi).at[1:-1, 1:-1].set(resid_int)

        psi = psi - dt * resid_full

        if (k + 1) % print_every == 0 or k == steps - 1:
            L2_res = float(jnp.sqrt(jnp.mean(resid_int**2)))
            print(f"step {k+1:6d} | resid L2 = {L2_res:.6e}")

    L2_err = float(jnp.sqrt(jnp.mean((psi - psi_true) ** 2)))
    Linf_err = float(jnp.max(jnp.abs(psi - psi_true)))

    print(f"Solution L2 error   : {L2_err:.6e}")
    print(f"Solution Linf error : {Linf_err:.6e}")

    return psi, psi_true

# ============================================================
# F3 – TDGL NONLINEAR STEADY (PSEUDO-TIME, OP PRODUKSI)
# ============================================================

def tdgl_nonlinear_steady_solve(
    N,
    xi=0.1,
    alpha=1.0,
    beta=1.0,
    A=0.5,
    dt=5e-4,
    steps=20000,
    print_every=5000,
):
    print(f"\n=== F3: TDGL NONLINEAR STEADY (N={N}) ===")

    X, Y = make_grid(N)
    h = 1.0 / (N - 1)

    psi_true = A * jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * Y)

    # RHS dibangun dengan operator produksi nonlinear
    g_int = op_tdgl_nonlinear_prod(psi_true, h, xi, alpha, beta)

    psi = jnp.zeros((N, N))

    for k in range(steps):
        Nh = op_tdgl_nonlinear_prod(psi, h, xi, alpha, beta)

        resid_int = Nh - g_int
        resid_full = jnp.zeros_like(psi).at[1:-1, 1:-1].set(resid_int)

        psi = psi - dt * resid_full

        if (k + 1) % print_every == 0 or k == steps - 1:
            L2_res = float(jnp.sqrt(jnp.mean(resid_int**2)))
            print(f"step {k+1:6d} | resid L2 = {L2_res:.6e}")

    L2_err = float(jnp.sqrt(jnp.mean((psi - psi_true) ** 2)))
    Linf_err = float(jnp.max(jnp.abs(psi - psi_true)))

    print(f"Solution L2 error   : {L2_err:.6e}")
    print(f"Solution Linf error : {Linf_err:.6e}")

    return psi, psi_true

# ============================================================
# RUN ALL – SIAP JADI pde_test_mode DI v9
# ============================================================

def run_all():
    for N in [32, 64, 128]:
        poisson_validate(N)

    for N in [32, 64]:
        tdgl_linear_steady_solve(N)

    for N in [32, 64]:
        tdgl_nonlinear_steady_solve(N)

    print("\n=== ALL VALIDATORS COMPLETED ===")

if __name__ == "__main__":
    run_all()
