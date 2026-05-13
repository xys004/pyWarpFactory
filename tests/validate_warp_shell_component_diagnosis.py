import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(REPO_ROOT)
sys.path.append(os.path.join(REPO_ROOT, "Examples"))

from diagnose_warp_shell_components import build_parser, run


def validate():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--grid",
            "8",
            "--nt",
            "5",
            "--spacing",
            "0.5",
            "--mass",
            "1e26",
            "--velocity",
            "0.1",
            "--r-inner",
            "1.0",
            "--r-outer",
            "2.0",
            "--num-vecs",
            "4",
            "--radial-bins",
            "6",
            "--energy-condition-method",
            "warpfactory",
        ]
    )
    report = run(args)

    assert len(report["cases"]) == 3
    by_case = {case["case"]: case for case in report["cases"]}
    assert set(by_case) == {"static_mass", "pure_shift", "full_shell"}

    for case in report["cases"]:
        assert np.isfinite(case["metric_diagnostics"]["max_abs_deviation_from_minkowski"])
        for condition in ("Null", "Weak", "Strong", "Dominant"):
            data = case["energy_conditions"][condition]
            assert np.isfinite(data["min"])
            assert data["r_at_min"] >= 0.0
            assert data["radial_profile"]

    assert by_case["pure_shift"]["energy_conditions"]["Null"]["min"] < 0.0
    assert by_case["full_shell"]["energy_conditions"]["Null"]["min"] < 0.0

    print("SUCCESS: Warp Shell component diagnosis runs and identifies shift/full violations.")
    for name, case in by_case.items():
        null = case["energy_conditions"]["Null"]
        print(
            f"{name:12s} NEC_min={null['min']:.4e} "
            f"frac={null['violating_fraction']:.4e} "
            f"r_min={null['r_at_min']:.4e}"
        )


if __name__ == "__main__":
    validate()
