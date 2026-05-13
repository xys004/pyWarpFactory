import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from warpfactory import analyze_metric
from warpfactory.generator.alcubierre import create_alcubierre_metric


def validate():
    grid_size = (5, 12, 12, 12)
    grid_scale = (1.0, 0.2, 0.2, 0.2)
    center = (1.0, 1.2, 1.2, 1.2)

    metric = create_alcubierre_metric(
        grid_size,
        grid_scale,
        center,
        v=0.5,
        R=0.8,
        sigma=4.0,
    )

    result = analyze_metric(metric, num_vecs=8)

    assert "Null" in result.energy_conditions
    assert "Weak" in result.energy_conditions
    assert "Strong" in result.energy_conditions
    assert "Dominant" in result.energy_conditions
    assert "Null_min" in result.summary
    assert "Dominant_violating_fraction" in result.summary
    assert result.eulerian_energy_tensor.shape == metric.tensor.shape

    print("SUCCESS: high-level analysis pipeline passed.")
    print(f"Minimum NEC value: {result.min_condition('Null'):.4e}")
    print(f"Minimum DEC value: {result.min_condition('Dominant'):.4e}")
    print(f"NEC violation detected: {result.has_violation('Null')}")


if __name__ == "__main__":
    validate()
