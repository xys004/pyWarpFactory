import argparse
import json

import numpy as np


KEY_MAP = {
    "metric_tensor": "MetricTensor",
    "eulerian_energy_tensor": "EnergyEulerianTensor",
    "null": "NullMap",
    "weak": "WeakMap",
    "strong": "StrongMap",
    "dominant": "DominantMap",
    "r_sample": "r_sample",
    "rho": "rho",
    "P": "P",
    "M": "M",
    "A": "A",
    "B": "B",
}


def squeeze_trailing_singletons(array):
    return np.asarray(array).squeeze()


def compare_pair(reference, candidate):
    ref = squeeze_trailing_singletons(reference)
    cand = squeeze_trailing_singletons(candidate)
    if ref.shape != cand.shape:
        return {
            "status": "shape_mismatch",
            "reference_shape": ref.shape,
            "candidate_shape": cand.shape,
        }

    diff = cand - ref
    scale = np.maximum(np.abs(ref), 1.0)
    return {
        "status": "ok",
        "shape": ref.shape,
        "max_abs": float(np.nanmax(np.abs(diff))),
        "max_rel": float(np.nanmax(np.abs(diff) / scale)),
        "mean_abs": float(np.nanmean(np.abs(diff))),
        "reference_min": float(np.nanmin(ref)),
        "reference_max": float(np.nanmax(ref)),
        "candidate_min": float(np.nanmin(cand)),
        "candidate_max": float(np.nanmax(cand)),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare Python W1 arrays against MATLAB W1 reference arrays.")
    parser.add_argument("python_npz", help="w1_python_arrays.npz from examples/w1_warp_shell_python.py")
    parser.add_argument("reference_npz", help="w1_reference.npz converted from MATLAB export")
    args = parser.parse_args()

    py = np.load(args.python_npz)
    ref = np.load(args.reference_npz)

    report = {}
    for py_key, ref_key in KEY_MAP.items():
        if py_key not in py:
            report[py_key] = {"status": "missing_python_key"}
        elif ref_key not in ref:
            report[py_key] = {"status": "missing_reference_key", "reference_key": ref_key}
        else:
            report[py_key] = compare_pair(ref[ref_key], py[py_key])

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
