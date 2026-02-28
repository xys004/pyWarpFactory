import numpy as np
from warpfactory.solver.tensor_utils import (
    get_c4_inv,
    get_ricci_tensor,
    get_ricci_scalar,
    get_einstein_tensor,
    get_energy_tensor,
)
from warpfactory.generator.base import Metric
from warpfactory.analyzer.transform import do_frame_transfer


def generate_uniform_field(field_type, num_angular, num_time=1):
    """
    Generate a dense set of null test vectors uniformly distributed over the sphere
    using the Fibonacci lattice algorithm.

    Args:
        field_type (str): Currently only 'nulllike' is used.
        num_angular (int): Number of vectors to sample.
        num_time (int): Unused legacy parameter.

    Returns:
        np.ndarray: shape (4, num_angular).  v^0 = 1, |v^i| = 1 (null-like in flat space).
    """
    indices = np.arange(0, num_angular, dtype=float) + 0.5
    phi   = np.arccos(1 - 2 * indices / num_angular)
    theta = np.pi * (1 + 5**0.5) * indices

    vecs = np.zeros((4, num_angular))
    vecs[0, :] = 1.0
    vecs[1, :] = np.cos(theta) * np.sin(phi)
    vecs[2, :] = np.sin(theta) * np.sin(phi)
    vecs[3, :] = np.cos(phi)
    return vecs


def _lower_indices_minkowski(T_upper):
    """
    Lower both indices of a rank-2 contravariant tensor T^{uv} using the Minkowski
    metric  eta = diag(-1, +1, +1, +1).

    In the locally flat Eulerian frame:
        T_{00} =  T^{00}      (eta_00 applied twice: (-1)(-1) = +1)
        T_{0i} = -T^{0i}      (eta_00 = -1, eta_ii = +1)
        T_{i0} = -T^{i0}
        T_{ij} =  T^{ij}

    Args:
        T_upper (np.ndarray): Contravariant tensor, shape (4, 4, ...).

    Returns:
        np.ndarray: Covariant tensor, same shape.
    """
    T_lower = T_upper.copy()
    T_lower[0, :] = -T_upper[0, :]   # flip row 0
    T_lower[:, 0] = -T_upper[:, 0]   # flip col 0
    T_lower[0, 0] =  T_upper[0, 0]   # double flip → restore sign
    return T_lower


def calculate_energy_conditions(metric_tensor, grid_scale, num_vecs=50):
    """
    Compute Null and Weak energy conditions from a covariant spacetime metric.

    This is a full-pipeline function that internally:
      1. Solves the Einstein field equations to get T^{uv}.
      2. Transforms T to the locally flat Eulerian reference frame.
      3. Evaluates T_{uv} k^u k^v for a dense sample of null vectors k  (NEC).
      4. Checks energy density rho = T^{00} >= 0 combined with NEC           (WEC).

    Args:
        metric_tensor (np.ndarray): Covariant 4-metric g_{uv},
            shape (4, 4, Nt, Nx, Ny, Nz).
        grid_scale (tuple): Grid spacing (dt, dx, dy, dz).
        num_vecs (int): Number of null test vectors sampled on the unit sphere
            (default 50).  More vectors give a tighter lower bound.

    Returns:
        results (dict): Violation maps with keys 'Null' and 'Weak'.
            Shape of each value: (Nt, Nx, Ny, Nz).
            A negative value at a grid point means the condition is violated there.
              - 'Null' : min_{null k}  T_{uv} k^u k^v
              - 'Weak' : min(rho, NEC map)
        T_euler (np.ndarray): Contravariant stress-energy tensor in the Eulerian
            (locally flat) frame, shape (4, 4, Nt, Nx, Ny, Nz).
            T_euler[0, 0] is the energy density rho seen by a local observer.
    """
    # ------------------------------------------------------------------
    # Step 1: Compute stress-energy tensor T^{uv} from Einstein equations
    # ------------------------------------------------------------------
    g_inv = get_c4_inv(metric_tensor)
    R_mn  = get_ricci_tensor(metric_tensor, grid_scale)
    R_sc  = get_ricci_scalar(R_mn, g_inv)
    G_mn  = get_einstein_tensor(R_mn, R_sc, metric_tensor)
    T_uv  = get_energy_tensor(G_mn, g_inv)   # contravariant T^{uv}

    # ------------------------------------------------------------------
    # Step 2: Wrap in Metric objects required by do_frame_transfer
    # ------------------------------------------------------------------
    spatial_shape = metric_tensor.shape[2:]
    _z = np.zeros(spatial_shape)
    dummy_coords = {'t': _z, 'x': _z, 'y': _z, 'z': _z}

    metric_obj = Metric(
        tensor=metric_tensor,
        coords=dummy_coords,
        scaling=np.array(grid_scale),
        name="metric",
        index="covariant",
    )

    energy_obj = Metric(
        tensor=T_uv,
        coords=dummy_coords,
        scaling=np.array(grid_scale),
        name="energy_tensor",
        index="contravariant",
    )
    energy_obj.type = "Stress-Energy"

    # ------------------------------------------------------------------
    # Step 3: Transform to Eulerian (locally flat) frame
    # ------------------------------------------------------------------
    energy_eu = do_frame_transfer(metric_obj, energy_obj, "Eulerian")
    T_euler   = energy_eu.tensor     # T^{hat{u}hat{v}} in Eulerian frame (contravariant)

    # ------------------------------------------------------------------
    # Step 4: Lower indices using the flat Minkowski metric
    # ------------------------------------------------------------------
    T_lower = _lower_indices_minkowski(T_euler)

    # ------------------------------------------------------------------
    # Step 5: Null Energy Condition (NEC)
    #   NEC: T_{uv} k^u k^v >= 0  for every null vector k.
    #   We sample num_vecs null directions and record the minimum value.
    # ------------------------------------------------------------------
    vecs    = generate_uniform_field("nulllike", num_vecs)   # (4, num_vecs)
    # Move tensor indices to last two axes for batched matmul: (..., 4, 4)
    T_end   = np.moveaxis(T_lower, [0, 1], [-2, -1])

    nec_map = np.full(T_lower.shape[2:], np.inf)
    for i in range(num_vecs):
        k   = vecs[:, i]                                 # (4,)
        # T_{uv} k^v  →  (..., 4) ; then contract with k^u  →  (...)
        Tk  = T_end @ k                                  # (..., 4)
        val = Tk  @ k                                    # (...)
        nec_map = np.minimum(nec_map, val)

    # ------------------------------------------------------------------
    # Step 6: Weak Energy Condition (WEC)
    #   WEC = (rho >= 0) AND NEC (null is the limiting timelike direction)
    #   rho = T^{00} in Eulerian frame
    # ------------------------------------------------------------------
    rho     = T_euler[0, 0]                              # energy density (...)
    wec_map = np.minimum(rho, nec_map)

    results = {'Null': nec_map, 'Weak': wec_map}
    return results, T_euler
