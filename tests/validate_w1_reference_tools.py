import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tools")))

from compare_w1_reference import compare_pair
from compare_w1_component_audit import KEY_MAP as COMPONENT_KEY_MAP


def validate():
    reference = np.arange(16, dtype=float).reshape(4, 4)
    candidate = reference.copy()
    same = compare_pair(reference, candidate)
    assert same["status"] == "ok"
    assert same["max_abs"] == 0.0
    assert same["max_rel"] == 0.0

    candidate[0, 0] = 2.0
    changed = compare_pair(reference, candidate)
    assert changed["status"] == "ok"
    assert changed["max_abs"] == 2.0

    mismatch = compare_pair(reference, candidate.reshape(2, 8))
    assert mismatch["status"] == "shape_mismatch"

    for key in (
        "static_ricci_tensor",
        "static_ricci_scalar",
        "static_einstein_tensor",
        "shifted_ricci_tensor",
        "shifted_ricci_scalar",
        "shifted_einstein_tensor",
        "static_energy_tensor",
        "shifted_energy_tensor",
    ):
        assert key in COMPONENT_KEY_MAP

    print("SUCCESS: W1 reference comparison helper works on synthetic arrays.")


if __name__ == "__main__":
    validate()
