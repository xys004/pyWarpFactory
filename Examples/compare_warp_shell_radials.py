import argparse
import json

import numpy as np


PROFILE_KEYS = ("r_sample", "rho", "P", "M", "A", "B")


def load_npz(path):
    data = np.load(path)
    return {key: data[key] for key in PROFILE_KEYS if key in data}


def compare_arrays(reference, candidate):
    rows = {}
    for key in PROFILE_KEYS:
        if key not in reference or key not in candidate:
            rows[key] = {"status": "missing"}
            continue

        ref = np.asarray(reference[key])
        cand = np.asarray(candidate[key])
        if ref.shape != cand.shape:
            rows[key] = {
                "status": "shape_mismatch",
                "reference_shape": ref.shape,
                "candidate_shape": cand.shape,
            }
            continue

        diff = cand - ref
        scale = np.maximum(np.abs(ref), 1.0)
        rows[key] = {
            "status": "ok",
            "max_abs": float(np.nanmax(np.abs(diff))),
            "max_rel": float(np.nanmax(np.abs(diff) / scale)),
            "mean_abs": float(np.nanmean(np.abs(diff))),
        }

    return rows


def main():
    parser = argparse.ArgumentParser(description="Compare exported WarpShell radial profiles.")
    parser.add_argument("reference", help="Reference .npz radial export, e.g. MATLAB-converted profiles.")
    parser.add_argument("candidate", help="Candidate .npz radial export, e.g. pyWarpFactory profiles.")
    args = parser.parse_args()

    report = compare_arrays(load_npz(args.reference), load_npz(args.candidate))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
