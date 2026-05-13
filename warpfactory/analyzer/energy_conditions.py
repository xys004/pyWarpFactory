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


def generate_warpfactory_field(field_type, num_angular, num_time=10):
    """
    Generate vectors using MATLAB WarpFactory's generateUniformField convention.

    Vectors are normalized by Euclidean norm, as in the MATLAB implementation.
    For contractions this overall positive scale does not change the sign, but
    matching it keeps exported maps closer numerically.
    """
    if field_type.lower() not in {"nulllike", "timelike"}:
        raise ValueError("field_type must be 'nulllike' or 'timelike'.")

    spatial = _matlab_even_points_on_sphere(1.0, num_angular)
    if field_type.lower() == "nulllike":
        vecs = np.zeros((4, num_angular))
        vecs[0] = 1.0
        vecs[1:] = spatial
        norm = np.sqrt(np.sum(vecs**2, axis=0))
        return vecs / norm

    bb = np.linspace(0.0, 1.0, num_time)
    vecs = np.ones((4, num_angular, num_time))
    for j, b in enumerate(bb):
        vecs[0, :, j] = 1.0
        vecs[1:, :, j] = (1.0 - b) * spatial
        norm = np.sqrt(np.sum(vecs[:, :, j] ** 2, axis=0))
        vecs[:, :, j] = vecs[:, :, j] / norm
    return vecs


def _matlab_even_points_on_sphere(radius, number_of_points):
    """
    Port of MATLAB WarpFactory's getEvenPointsOnSphere.m.

    This is intentionally separate from generate_uniform_field because the
    MATLAB implementation uses i = 0..N-1 in the azimuth and i+0.5 only in the
    polar angle. The sign of an energy-condition map is independent of vector
    normalization, but not independent of the sampled directions.
    """
    golden_ratio = (1.0 + 5.0**0.5) / 2.0
    points = np.zeros((3, number_of_points), dtype=float)
    for i in range(number_of_points):
        theta = 2.0 * np.pi * i / golden_ratio
        phi = np.arccos(1.0 - 2.0 * (i + 0.5) / number_of_points)
        points[0, i] = radius * np.cos(theta) * np.sin(phi)
        points[1, i] = radius * np.sin(theta) * np.sin(phi)
        points[2, i] = radius * np.cos(phi)
    return np.real(points)


def generate_timelike_observers(num_angular, speeds=None):
    """
    Generate future-directed unit timelike observers in a local Minkowski frame.

    Returns vectors u^a with eta_ab u^a u^b = -1 and u^0 > 0.
    """
    if speeds is None:
        speeds = (0.0, 0.5, 0.9, 0.99)

    null_dirs = generate_uniform_field("nulllike", num_angular)
    spatial_dirs = null_dirs[1:]
    observers = []

    for speed in speeds:
        if speed < 0 or speed >= 1:
            raise ValueError("Timelike observer speeds must satisfy 0 <= speed < 1.")

        gamma = 1.0 / np.sqrt(1.0 - speed**2)
        if speed == 0:
            observers.append(np.array([[1.0], [0.0], [0.0], [0.0]]))
            continue

        vecs = np.zeros((4, num_angular))
        vecs[0, :] = gamma
        vecs[1:, :] = gamma * speed * spatial_dirs
        observers.append(vecs)

    return np.concatenate(observers, axis=1)


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


def _trace_minkowski(T_lower):
    return -T_lower[0, 0] + T_lower[1, 1] + T_lower[2, 2] + T_lower[3, 3]


def _contract_rank2(T_end, vector):
    return (T_end @ vector) @ vector


def _mixed_up_down_minkowski(T_lower):
    mixed = T_lower.copy()
    mixed[0, :] = -mixed[0, :]
    return mixed


def _evaluate_energy_condition_maps_from_eulerian(T_euler, num_vecs=50, requested=None):
    """
    Evaluate NEC/WEC/SEC/DEC in the local Eulerian orthonormal frame.

    WEC, SEC, and DEC are evaluated on a finite family of sampled timelike
    observers. This is a practical diagnostic, not yet a proof over all
    observers.
    """
    if requested is None:
        requested = {"Null", "Weak", "Strong", "Dominant"}
    else:
        requested = set(requested)

    T_lower = _lower_indices_minkowski(T_euler)
    T_end = np.moveaxis(T_lower, [0, 1], [-2, -1])
    results = {}

    needs_nec = bool(requested & {"Null", "Weak"})
    needs_timelike = bool(requested & {"Weak", "Strong", "Dominant"})

    if needs_nec:
        null_vecs = generate_uniform_field("nulllike", num_vecs)
        nec_map = np.full(T_lower.shape[2:], np.inf)
        for i in range(num_vecs):
            k = null_vecs[:, i]
            nec_map = np.minimum(nec_map, _contract_rank2(T_end, k))
        if "Null" in requested:
            results["Null"] = nec_map

    if needs_timelike:
        timelike_vecs = generate_timelike_observers(num_vecs)
        if "Weak" in requested:
            wec_timelike = np.full(T_lower.shape[2:], np.inf)
        if "Strong" in requested:
            sec_map = np.full(T_lower.shape[2:], np.inf)
            trace = _trace_minkowski(T_lower)
            eta = np.diag([-1.0, 1.0, 1.0, 1.0])
            eta_shape = (4, 4) + (1,) * len(trace.shape)
            trace_reversed = T_lower - 0.5 * trace[np.newaxis, np.newaxis, ...] * eta.reshape(eta_shape)
            trace_reversed_end = np.moveaxis(trace_reversed, [0, 1], [-2, -1])
        if "Dominant" in requested:
            dec_map = np.full(T_lower.shape[2:], np.inf)
            T_mixed = _mixed_up_down_minkowski(T_lower)
            T_mixed_end = np.moveaxis(T_mixed, [0, 1], [-2, -1])

        for i in range(timelike_vecs.shape[1]):
            u = timelike_vecs[:, i]

            rho_u = _contract_rank2(T_end, u)
            if "Weak" in requested:
                wec_timelike = np.minimum(wec_timelike, rho_u)
            if "Strong" in requested:
                sec_map = np.minimum(sec_map, _contract_rank2(trace_reversed_end, u))
            if "Dominant" in requested:
                flux = -(T_mixed_end @ u)
                flux_norm = -flux[..., 0] ** 2 + np.sum(flux[..., 1:] ** 2, axis=-1)
                dec_observer = np.minimum.reduce((rho_u, flux[..., 0], -flux_norm))
                dec_map = np.minimum(dec_map, dec_observer)

        if "Weak" in requested:
            if needs_nec:
                results["Weak"] = np.minimum(wec_timelike, nec_map)
            else:
                results["Weak"] = wec_timelike
        if "Strong" in requested:
            results["Strong"] = sec_map
        if "Dominant" in requested:
            results["Dominant"] = dec_map

    return results


def _evaluate_warpfactory_compatible_maps_from_eulerian(T_euler, num_vecs=50, num_time_vecs=10, requested=None):
    """
    Evaluate energy conditions with conventions close to MATLAB WarpFactory.

    The main distinction from the standard pyWarpFactory diagnostic is DEC:
    MATLAB `getEnergyConditions.m` evaluates the dominant condition with
    nulllike test vectors and uses the Minkowski norm of the flux vector,
    then flips the sign so negative values denote violation.
    """
    if requested is None:
        requested = {"Null", "Weak", "Strong", "Dominant"}
    else:
        requested = set(requested)

    results = {}
    T_lower = _lower_indices_minkowski(T_euler)
    T_end = np.moveaxis(T_lower, [0, 1], [-2, -1])

    if "Null" in requested:
        null_vecs = generate_warpfactory_field("nulllike", num_vecs)
        nec_map = np.full(T_lower.shape[2:], np.inf)
        for i in range(num_vecs):
            k = null_vecs[:, i]
            nec_map = np.minimum(nec_map, _contract_rank2(T_end, k))
        results["Null"] = nec_map

    if requested & {"Weak", "Strong"}:
        timelike_vecs = generate_warpfactory_field("timelike", num_vecs, num_time=num_time_vecs)
        if "Weak" in requested:
            weak_map = np.full(T_lower.shape[2:], np.inf)
        if "Strong" in requested:
            strong_map = np.full(T_lower.shape[2:], np.inf)
            trace = _trace_minkowski(T_lower)
            eta = np.diag([-1.0, 1.0, 1.0, 1.0])
            eta_shape = (4, 4) + (1,) * len(trace.shape)
            trace_reversed = T_lower - 0.5 * trace[np.newaxis, np.newaxis, ...] * eta.reshape(eta_shape)
            trace_reversed_end = np.moveaxis(trace_reversed, [0, 1], [-2, -1])

        for j in range(timelike_vecs.shape[2]):
            for i in range(num_vecs):
                u = timelike_vecs[:, i, j]
                if "Weak" in requested:
                    weak_map = np.minimum(weak_map, _contract_rank2(T_end, u))
                if "Strong" in requested:
                    strong_map = np.minimum(strong_map, _contract_rank2(trace_reversed_end, u))

        if "Weak" in requested:
            results["Weak"] = weak_map
        if "Strong" in requested:
            results["Strong"] = strong_map

    if "Dominant" in requested:
        T_mixed = _mixed_up_down_minkowski(T_lower)
        T_mixed_end = np.moveaxis(T_mixed, [0, 1], [-2, -1])
        null_vecs = generate_warpfactory_field("nulllike", num_vecs)

        dec_map = np.full(T_lower.shape[2:], np.inf)
        for i in range(num_vecs):
            k = null_vecs[:, i]
            flux = -(T_mixed_end @ k)
            flux_norm = -flux[..., 0] ** 2 + np.sum(flux[..., 1:] ** 2, axis=-1)
            signed_norm = np.sign(flux_norm) * np.sqrt(np.abs(flux_norm))
            dec_map = np.minimum(dec_map, -signed_norm)

        results["Dominant"] = dec_map

    return results


def _calculate_energy_condition_maps(metric_tensor, grid_scale, num_vecs=50):
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

    results = _evaluate_energy_condition_maps_from_eulerian(T_euler, num_vecs=num_vecs)
    return results, T_euler


def _condition_alias(condition):
    aliases = {
        "null": "Null",
        "nec": "Null",
        "weak": "Weak",
        "wec": "Weak",
        "strong": "Strong",
        "sec": "Strong",
        "dominant": "Dominant",
        "dec": "Dominant",
    }
    key = condition.lower()
    if key not in aliases:
        supported = ", ".join(sorted(set(aliases.values())))
        raise NotImplementedError(
            f"Energy condition '{condition}' is not implemented yet. "
            f"Supported conditions: {supported}."
        )
    return aliases[key]


def calculate_energy_conditions(
    metric_or_energy,
    metric_or_grid_scale,
    condition=None,
    num_vecs=50,
    num_time_vecs=10,
    method="standard",
):
    """
    Evaluate energy-condition violation maps.

    Two call styles are supported:

      1. Full pipeline from a covariant metric tensor:
         ``calculate_energy_conditions(metric.tensor, metric.scaling)``
         returns ``(results, T_euler)``.

      2. User-facing API from the solved stress-energy tensor:
         ``calculate_energy_conditions(energy_tensor, metric, condition='Null')``
         returns the requested violation map.

    Negative values indicate a violation at that grid point.
    """
    if method not in {"standard", "warpfactory"}:
        raise ValueError("method must be 'standard' or 'warpfactory'.")

    if hasattr(metric_or_energy, "tensor") and hasattr(metric_or_grid_scale, "tensor"):
        energy_tensor = metric_or_energy
        metric = metric_or_grid_scale

        energy_eu = do_frame_transfer(metric, energy_tensor, "Eulerian")
        T_euler = energy_eu.tensor
        requested = None if condition is None else [_condition_alias(condition)]
        evaluator = (
            _evaluate_energy_condition_maps_from_eulerian
            if method == "standard"
            else _evaluate_warpfactory_compatible_maps_from_eulerian
        )
        results = evaluator(
            T_euler,
            num_vecs=num_vecs,
            num_time_vecs=num_time_vecs,
            requested=requested,
        ) if method == "warpfactory" else evaluator(
            T_euler,
            num_vecs=num_vecs,
            requested=requested,
        )

        if condition is None:
            return results
        return results[requested[0]]

    if condition is not None:
        raise TypeError(
            "The 'condition' argument is only valid when calling with "
            "(energy_tensor, metric)."
        )

    return _calculate_energy_condition_maps(
        metric_or_energy,
        metric_or_grid_scale,
        num_vecs=num_vecs,
    )
