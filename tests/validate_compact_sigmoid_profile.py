import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from warpfactory.utils.helpers import compact_sigmoid


def validate():
    R1 = 10.0
    R2 = 20.0
    Rbuff = 0.0
    sigma = 0.0
    r = np.array([0.0, 5.0, 9.9, 10.0, 10.277159, 11.700427, 15.0, 19.663672, 20.0, 25.0])
    values = compact_sigmoid(r, R1, R2, sigma, Rbuff)

    print("compact_sigmoid profile")
    for radius, value in zip(r, values):
        print(f"r={radius:9.6f} S={value:.12e}")

    assert np.all(values[r <= R1 + Rbuff] == 1.0)
    assert np.all(values[r >= R2 - Rbuff] == 0.0)
    assert values[4] > 0.99, "Swarp should still be near the inner plateau at r~10.277"
    assert values[5] > 0.95, "Swarp should still be high at r~11.7"
    assert np.isclose(values[6], 0.5, atol=1e-14), "Swarp midpoint should be 0.5"
    assert values[7] < 0.05, "Swarp should be near exterior at r~19.66"

    print("SUCCESS: compact_sigmoid follows the MATLAB-style 1-to-0 radial profile.")


if __name__ == "__main__":
    validate()
