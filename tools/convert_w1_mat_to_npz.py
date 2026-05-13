import argparse
import sys

import numpy as np


W1_KEYS = (
    "MetricTensor",
    "EnergyEulerianTensor",
    "r_sample",
    "rho",
    "rho_smooth",
    "P",
    "P_smooth",
    "M",
    "A",
    "B",
    "NullMap",
    "WeakMap",
    "StrongMap",
    "DominantMap",
    "gridSize",
    "gridScaling",
    "worldCenter",
    "static_MetricTensor",
    "static_RicciTensor",
    "static_RicciScalar",
    "static_EinsteinTensor",
    "static_EnergyTensor",
    "static_EnergyEulerianTensor",
    "static_NullMap",
    "static_WeakMap",
    "static_StrongMap",
    "static_DominantMap",
    "shifted_MetricTensor",
    "shifted_RicciTensor",
    "shifted_RicciScalar",
    "shifted_EinsteinTensor",
    "shifted_EnergyTensor",
    "shifted_EnergyEulerianTensor",
    "shifted_NullMap",
    "shifted_WeakMap",
    "shifted_StrongMap",
    "shifted_DominantMap",
    "numAngularVec",
    "numTimeVec",
)


def load_with_h5py(path):
    import h5py

    out = {}
    with h5py.File(path, "r") as f:
        for key in W1_KEYS:
            if key in f:
                value = np.array(f[key])
                # MATLAB v7.3/HDF5 stores dimensions reversed relative to NumPy.
                out[key] = np.transpose(value)
    return out


def load_with_scipy(path):
    from scipy.io import loadmat

    mat = loadmat(path)
    return {key: np.asarray(mat[key]) for key in W1_KEYS if key in mat}


def load_mat(path):
    try:
        return load_with_h5py(path)
    except ImportError:
        pass
    except OSError:
        # Not HDF5/v7.3; try scipy below.
        pass

    try:
        return load_with_scipy(path)
    except ImportError as exc:
        raise SystemExit(
            "Cannot read .mat files because neither h5py nor scipy is installed. "
            "Install one of them or run this script in the project environment."
        ) from exc


def main():
    parser = argparse.ArgumentParser(description="Convert flat W1 MATLAB reference .mat to .npz.")
    parser.add_argument("mat_file", help="Path to tools/w1_reference.mat")
    parser.add_argument("--output", default="tools/w1_reference.npz")
    args = parser.parse_args()

    arrays = load_mat(args.mat_file)
    if not arrays:
        raise SystemExit("No W1 arrays found in the MAT file. Re-run matlab_export_w1_reference.m.")

    np.savez_compressed(args.output, **arrays)
    print(f"Saved {len(arrays)} arrays to {args.output}")
    print("Keys:", ", ".join(sorted(arrays)))


if __name__ == "__main__":
    main()
