import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from warpfactory import analyze_metric
from warpfactory.generator.alcubierre import create_alcubierre_metric
from warpfactory.generator.minkowski import create_minkowski_metric


def validate_minkowski_optimized():
    metric = create_minkowski_metric((5, 6, 6, 6), (1.0, 1.0, 1.0, 1.0))
    result = analyze_metric(
        metric,
        num_vecs=4,
        observer_mode="optimized",
        audit_points=1,
        optimized_max_speed=0.95,
    )

    assert result.methodology["observer_mode"] == "optimized"
    assert result.observer_audit["mode"] == "optimized"

    for entries in result.observer_audit["conditions"].values():
        for entry in entries:
            assert abs(entry["optimized_value"]) < 1e-8
            assert not entry["violates"]

    print("SUCCESS: optimized pipeline audit leaves Minkowski unviolated.")


def validate_alcubierre_optimized():
    metric = create_alcubierre_metric(
        grid_size=(5, 10, 10, 10),
        grid_scale=(1.0, 0.2, 0.2, 0.2),
        world_center=(1.0, 1.0, 1.0, 1.0),
        v=0.5,
        R=0.8,
        sigma=4.0,
    )
    result = analyze_metric(
        metric,
        num_vecs=6,
        observer_mode="optimized",
        audit_points=1,
        optimized_max_speed=0.95,
    )

    dec_entries = result.observer_audit["conditions"]["Dominant"]
    assert dec_entries
    assert any(entry["optimized_value"] < 0.0 for entry in dec_entries)

    print("SUCCESS: optimized pipeline audit confirms an Alcubierre DEC violation.")
    print(f"Sampled DEC min     : {result.summary['Dominant_min']:.4e}")
    print(f"Optimized DEC value : {dec_entries[0]['optimized_value']:.4e}")
    print(f"Audit point         : {dec_entries[0]['point']}")


def validate():
    validate_minkowski_optimized()
    validate_alcubierre_optimized()


if __name__ == "__main__":
    validate()
