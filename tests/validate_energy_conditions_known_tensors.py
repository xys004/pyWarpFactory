import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from warpfactory.analyzer.energy_conditions import (
    _evaluate_warpfactory_compatible_maps_from_eulerian,
)


def isotropic_tensor(rho, pressure, shape=(1, 2, 2, 2)):
    tensor = np.zeros((4, 4) + shape)
    tensor[0, 0] = rho
    tensor[1, 1] = pressure
    tensor[2, 2] = pressure
    tensor[3, 3] = pressure
    return tensor


def assert_constant_map(values, expected, atol=1e-12):
    np.testing.assert_allclose(values, np.full(values.shape, expected), rtol=1e-12, atol=atol)


def test_positive_dust():
    maps = _evaluate_warpfactory_compatible_maps_from_eulerian(
        isotropic_tensor(rho=2.0, pressure=0.0),
        num_vecs=8,
        num_time_vecs=5,
    )
    assert_constant_map(maps["Null"], 1.0)
    assert np.min(maps["Weak"]) > 0.0
    assert np.min(maps["Strong"]) > 0.0
    assert np.min(maps["Dominant"]) > 0.0


def test_radiation_like_fluid():
    rho = 3.0
    pressure = 1.0
    maps = _evaluate_warpfactory_compatible_maps_from_eulerian(
        isotropic_tensor(rho=rho, pressure=pressure),
        num_vecs=8,
        num_time_vecs=5,
    )
    # MATLAB null vectors are Euclidean-normalized (1, n) / sqrt(2).
    assert_constant_map(maps["Null"], 0.5 * (rho + pressure))
    assert np.min(maps["Weak"]) > 0.0
    assert np.min(maps["Strong"]) > 0.0
    assert np.min(maps["Dominant"]) > 0.0


def test_negative_energy_density_violates_nec_wec_but_not_warpfactory_dec():
    maps = _evaluate_warpfactory_compatible_maps_from_eulerian(
        isotropic_tensor(rho=-1.0, pressure=0.0),
        num_vecs=8,
        num_time_vecs=5,
    )
    assert np.max(maps["Null"]) < 0.0
    assert np.max(maps["Weak"]) < 0.0
    # MATLAB WarpFactory's DEC branch tests causal character of the flux vector
    # and flips the sign; it does not independently include the WEC/future-energy
    # requirement. Negative pure dust is therefore caught by NEC/WEC/SEC here,
    # not by this DEC map.
    assert np.min(maps["Dominant"]) > 0.0


def test_large_pressure_violates_dec_only():
    maps = _evaluate_warpfactory_compatible_maps_from_eulerian(
        isotropic_tensor(rho=1.0, pressure=2.0),
        num_vecs=8,
        num_time_vecs=5,
    )
    assert np.min(maps["Null"]) > 0.0
    assert np.min(maps["Weak"]) > 0.0
    assert np.min(maps["Strong"]) > 0.0
    assert np.max(maps["Dominant"]) < 0.0


def test_cosmological_constant_violates_strong_only():
    maps = _evaluate_warpfactory_compatible_maps_from_eulerian(
        isotropic_tensor(rho=1.0, pressure=-1.0),
        num_vecs=8,
        num_time_vecs=5,
    )
    assert_constant_map(maps["Null"], 0.0)
    assert np.min(maps["Weak"]) >= -1e-12
    assert np.max(maps["Strong"]) < 0.0
    assert np.min(maps["Dominant"]) >= -1e-7


def main():
    test_positive_dust()
    test_radiation_like_fluid()
    test_negative_energy_density_violates_nec_wec_but_not_warpfactory_dec()
    test_large_pressure_violates_dec_only()
    test_cosmological_constant_violates_strong_only()
    print("WarpFactory-compatible energy-condition known-tensor checks passed.")


if __name__ == "__main__":
    main()
