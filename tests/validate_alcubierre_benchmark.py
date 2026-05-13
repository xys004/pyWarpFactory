import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(REPO_ROOT)
sys.path.append(os.path.join(REPO_ROOT, "Examples"))

from verify_alcubierre_warpfactory import build_parser, run


def validate_alcubierre_negative_control():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--nt",
            "5",
            "--grid",
            "12",
            "--spacing",
            "0.2",
            "--v",
            "0.5",
            "--radius",
            "0.8",
            "--sigma",
            "4.0",
            "--num-vecs",
            "6",
            "--energy-condition-method",
            "warpfactory",
        ]
    )
    report = run(args)

    for condition in ("Null", "Weak", "Strong", "Dominant"):
        condition_report = report["energy_conditions"][condition]
        assert np.isfinite(condition_report["min"])
        assert condition_report["min"] < 0.0
        assert condition_report["violating_points"] > 0
        assert condition_report["r_from_bubble_at_min"] >= 0.0

    assert report["wall_alignment"]["Null"]["fraction_of_violations_in_wall_band"] > 0.0
    assert report["parameters"]["energy_condition_method"] == "warpfactory"

    print("SUCCESS: Alcubierre benchmark finds the expected energy-condition violations.")
    for condition, condition_report in report["energy_conditions"].items():
        print(
            f"{condition:8s} min={condition_report['min']:.4e} "
            f"fraction={condition_report['violating_fraction']:.4e} "
            f"r_min={condition_report['r_from_bubble_at_min']:.4e}"
        )


def validate():
    validate_alcubierre_negative_control()


if __name__ == "__main__":
    validate()
