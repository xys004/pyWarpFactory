import argparse
import json

import numpy as np


KEY_MAP = {
    "static_metric_tensor": "static_MetricTensor",
    "static_ricci_tensor": "static_RicciTensor",
    "static_ricci_scalar": "static_RicciScalar",
    "static_einstein_tensor": "static_EinsteinTensor",
    "static_energy_tensor": "static_EnergyTensor",
    "static_eulerian_energy_tensor": "static_EnergyEulerianTensor",
    "static_null": "static_NullMap",
    "static_weak": "static_WeakMap",
    "static_strong": "static_StrongMap",
    "static_dominant": "static_DominantMap",
    "shifted_metric_tensor": "shifted_MetricTensor",
    "shifted_ricci_tensor": "shifted_RicciTensor",
    "shifted_ricci_scalar": "shifted_RicciScalar",
    "shifted_einstein_tensor": "shifted_EinsteinTensor",
    "shifted_energy_tensor": "shifted_EnergyTensor",
    "shifted_eulerian_energy_tensor": "shifted_EnergyEulerianTensor",
    "shifted_null": "shifted_NullMap",
    "shifted_weak": "shifted_WeakMap",
    "shifted_strong": "shifted_StrongMap",
    "shifted_dominant": "shifted_DominantMap",
}


def squeeze(array):
    return np.asarray(array).squeeze()


def compare_pair(reference, candidate):
    ref = squeeze(reference)
    cand = squeeze(candidate)
    if ref.shape != cand.shape:
        return {
            "status": "shape_mismatch",
            "reference_shape": list(ref.shape),
            "candidate_shape": list(cand.shape),
        }

    diff = cand - ref
    scale = np.maximum(np.abs(ref), 1.0)
    abs_diff = np.abs(diff)
    flat_index = int(np.nanargmax(abs_diff))
    max_index = np.unravel_index(flat_index, abs_diff.shape)
    return {
        "status": "ok",
        "shape": list(ref.shape),
        "max_abs": float(np.nanmax(abs_diff)),
        "max_rel": float(np.nanmax(abs_diff / scale)),
        "mean_abs": float(np.nanmean(abs_diff)),
        "median_abs": float(np.nanmedian(abs_diff)),
        "reference_min": float(np.nanmin(ref)),
        "reference_max": float(np.nanmax(ref)),
        "candidate_min": float(np.nanmin(cand)),
        "candidate_max": float(np.nanmax(cand)),
        "max_abs_index": [int(item) for item in max_index],
        "reference_at_max_abs": float(ref[max_index]),
        "candidate_at_max_abs": float(cand[max_index]),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare Python W1 component audit arrays against MATLAB WarpFactory reference arrays."
    )
    parser.add_argument(
        "python_npz",
        help="fuchs_w1_component_audit_arrays.npz from Examples/audit_fuchs_w1_components.py",
    )
    parser.add_argument(
        "reference_npz",
        help="w1_component_audit_reference.npz converted from MATLAB export",
    )
    parser.add_argument("--output", default=None)
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

    text = json.dumps(report, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.write("\n")
    print(text)


if __name__ == "__main__":
    main()
