import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from warpfactory import analyze_metric, summarize_energy_conditions
from warpfactory.analyzer.energy_conditions import _lower_indices_minkowski, generate_warpfactory_field
from warpfactory.recipes.fuchs_warp_shell import create_fuchs_constant_warp_shell


def _radius(metric):
    return np.sqrt(metric.coords["x"] ** 2 + metric.coords["y"] ** 2 + metric.coords["z"] ** 2)


def _nearest_index(metric, target):
    distance2 = (
        (metric.coords["x"] - target[0]) ** 2
        + (metric.coords["y"] - target[1]) ** 2
        + (metric.coords["z"] - target[2]) ** 2
    )
    return tuple(int(i) for i in np.unravel_index(int(np.argmin(distance2)), distance2.shape))


def _best_null_tkk(T_hat_upper, num_vecs):
    T_hat_lower = _lower_indices_minkowski(T_hat_upper[..., np.newaxis])[..., 0]
    vectors = generate_warpfactory_field("nulllike", num_vecs)
    best = np.inf
    for i in range(vectors.shape[1]):
        k = vectors[:, i]
        best = min(best, float(k @ T_hat_lower @ k))
    return best


def validate():
    num_vecs = 40
    metric = create_fuchs_constant_warp_shell(profile="original")
    params = metric.params["fuchs_recipe"]
    assert params["v_warp"] == 0.02

    # Regression for the Swarp orientation bug: this point used to have
    # g01 ~ -9e-4 with the inverted transition. It should sit near the
    # smoothed interior plateau after the fix.
    critical_index = _nearest_index(metric, (0.1, -11.7, 0.0))
    g01 = float(metric.tensor[0, 1][critical_index])
    print(f"Regression point g01: {g01:.12e}")
    assert np.isclose(g01, -0.01935533259556508, rtol=2e-6, atol=1e-12)

    result = analyze_metric(
        metric,
        num_vecs=num_vecs,
        num_time_vecs=10,
        energy_condition_method="warpfactory",
        solver_method="christoffel",
    )
    shell = (_radius(metric) >= params["R1"]) & (_radius(metric) <= params["R2"])
    summary = summarize_energy_conditions(result.energy_conditions, mask=shell)

    print("Fuchs W1 shifted physical regression, material shell:")
    for condition in ("Null", "Weak", "Strong", "Dominant"):
        condition_min = summary[f"{condition}_min"]
        condition_frac = summary[f"{condition}_violating_fraction"]
        print(f"  {condition}: min={condition_min:.4e}, frac={condition_frac:.4e}")
        assert condition_min > 0.0
        assert condition_frac == 0.0

    local_tkk = _best_null_tkk(
        result.eulerian_energy_tensor[(slice(None), slice(None)) + critical_index],
        num_vecs,
    )
    print(f"Regression point best sampled null Tkk: {local_tkk:.4e}")
    assert local_tkk > 0.0

    print("SUCCESS: W1 shifted vWarp=0.02 remains physical after Swarp fix.")


if __name__ == "__main__":
    validate()
