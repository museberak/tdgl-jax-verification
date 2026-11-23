"""Evidence for the exact discrete eigenvalue claim (eigsh check).
This script is NOT part of tdgl_validation.py — v4 is FROZEN and unchanged.
It exists only to prove, with reproducible numbers, the comment in
tdgl_validation.py that the discrete eigenvalue formulas were verified
against eigsh. It stores its results in eigsh_evidence.json.
Run: python test_eigsh_evidence.py
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh


def lambda_min_discrete(N, h):
    return (8.0 / h**2) * math.sin(math.pi / (2 * (N - 1))) ** 2


def lambda_max_discrete(N, h):
    return (8.0 / h**2) * math.cos(math.pi / (2 * (N - 1))) ** 2


def lambda_min_continuous():
    return 2.0 * math.pi**2


def main():
    out = {"note": "eigsh numerics vs the discrete formula vs the continuous 2*pi^2",
           "results": {}}
    for N in [33, 65]:
        n = N - 2
        h = 1.0 / (N - 1)
        T = sparse.diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(n, n), format="csc")
        I_n = sparse.eye(n, format="csc")
        Lap = (sparse.kron(T, I_n) + sparse.kron(I_n, T)) / (h * h)
        neg_lap = -Lap
        lam_min_num = float(eigsh(neg_lap, k=1, which="SM", return_eigenvectors=False)[0])
        lam_max_num = float(eigsh(neg_lap, k=1, which="LM", return_eigenvectors=False)[0])
        lam_min_f = lambda_min_discrete(N, h)
        lam_max_f = lambda_max_discrete(N, h)
        lam_min_c = lambda_min_continuous()
        row = {
            "N": N, "h": h,
            "lambda_min_eigsh": lam_min_num,
            "lambda_min_formula_discrete": lam_min_f,
            "diff_min_formula": abs(lam_min_num - lam_min_f),
            "lambda_min_continuous_2pi2": lam_min_c,
            "error_continuous": abs(lam_min_num - lam_min_c),
            "lambda_max_eigsh": lam_max_num,
            "lambda_max_formula_discrete": lam_max_f,
            "diff_max_formula": abs(lam_max_num - lam_max_f),
            "bound_8_over_h2": 8.0 / h**2,
        }
        out["results"][str(N)] = row
        print(f"N={N} h={h:.6f}")
        print(f"  lam_min eigsh={lam_min_num:.6f} formula={lam_min_f:.6f} "
              f"diff={abs(lam_min_num-lam_min_f):.2e} | continuous err={abs(lam_min_num-lam_min_c):.4f}")
        print(f"  lam_max eigsh={lam_max_num:.1f} formula={lam_max_f:.1f} "
              f"diff={abs(lam_max_num-lam_max_f):.2e} | bound 8/h^2={8.0/h**2:.1f}")
        assert abs(lam_min_num - lam_min_f) < 1e-6, "lambda_min formula is wrong"
        assert abs(lam_max_num - lam_max_f) < 1e-3, "lambda_max formula is wrong"
    out["verdict"] = "PASS: the exact discrete formulas agree with eigsh; the continuous 2*pi^2 is wrong by ~0.016"
    path = Path(__file__).parent / "eigsh_evidence.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nEIGSH EVIDENCE PASS — {path}")


if __name__ == "__main__":
    main()
