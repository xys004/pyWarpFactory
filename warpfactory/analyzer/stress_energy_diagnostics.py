import numpy as np

from warpfactory.analyzer.energy_conditions import _lower_indices_minkowski, _mixed_up_down_minkowski


def eulerian_stress_decomposition(T_euler, coords=None):
    """
    Decompose an Eulerian-frame stress-energy tensor into local diagnostics.

    T_euler is expected to be contravariant in an orthonormal Eulerian frame.
    The returned quantities are computed after lowering with eta_ab:
      rho = T_00
      flux_i = -T_0i
      stress_ij = T_ij
    """
    T_lower = _lower_indices_minkowski(T_euler)
    rho = T_lower[0, 0]
    flux = -T_lower[0, 1:4]
    stress = T_lower[1:4, 1:4]

    flux_norm = np.sqrt(np.sum(flux**2, axis=0))
    stress_matrix = np.moveaxis(stress, [0, 1], [-2, -1])
    principal_pressures = np.linalg.eigvalsh(stress_matrix)
    pressure_min = principal_pressures[..., 0]
    pressure_mid = principal_pressures[..., 1]
    pressure_max = principal_pressures[..., 2]
    pressure_trace = stress[0, 0] + stress[1, 1] + stress[2, 2]
    pressure_iso = pressure_trace / 3.0
    anisotropy = pressure_max - pressure_min

    result = {
        "rho": rho,
        "flux": flux,
        "flux_norm": flux_norm,
        "stress": stress,
        "principal_pressures": principal_pressures,
        "pressure_min": pressure_min,
        "pressure_mid": pressure_mid,
        "pressure_max": pressure_max,
        "pressure_trace": pressure_trace,
        "pressure_iso": pressure_iso,
        "anisotropy": anisotropy,
        "rho_plus_p_min": rho + pressure_min,
        "rho_plus_p_max": rho + pressure_max,
        "flux_over_abs_rho": flux_norm / np.maximum(np.abs(rho), 1e-300),
    }

    if coords is not None:
        x = coords["x"]
        y = coords["y"]
        z = coords["z"]
        radius = np.sqrt(x**2 + y**2 + z**2)
        safe_radius = np.where(radius == 0.0, 1.0, radius)
        radial_unit = np.stack((x / safe_radius, y / safe_radius, z / safe_radius), axis=0)
        radial_unit[:, radius == 0.0] = 0.0
        radial_pressure = np.einsum("i...,ij...,j...->...", radial_unit, stress, radial_unit)
        radial_flux = np.einsum("i...,i...->...", flux, radial_unit)
        tangential_pressure = 0.5 * (pressure_trace - radial_pressure)
        result.update(
            {
                "radius": radius,
                "radial_pressure": radial_pressure,
                "tangential_pressure": tangential_pressure,
                "radial_flux": radial_flux,
                "rho_plus_radial_pressure": rho + radial_pressure,
                "rho_plus_tangential_pressure": rho + tangential_pressure,
            }
        )

    mixed = _mixed_up_down_minkowski(T_lower)
    mixed_matrix = np.moveaxis(mixed, [0, 1], [-2, -1])
    result["mixed_eigenvalues"] = np.linalg.eigvals(mixed_matrix)

    return result


def stress_energy_summary(T_euler, coords=None):
    """Return scalar summaries for the main Eulerian stress diagnostics."""
    fields = eulerian_stress_decomposition(T_euler, coords=coords)
    names = [
        "rho",
        "flux_norm",
        "pressure_min",
        "pressure_mid",
        "pressure_max",
        "pressure_iso",
        "anisotropy",
        "rho_plus_p_min",
        "rho_plus_p_max",
        "flux_over_abs_rho",
    ]
    if coords is not None:
        names.extend(
            [
                "radial_pressure",
                "tangential_pressure",
                "radial_flux",
                "rho_plus_radial_pressure",
                "rho_plus_tangential_pressure",
            ]
        )

    summary = {}
    for name in names:
        values = np.real_if_close(fields[name])
        summary[f"{name}_min"] = float(np.nanmin(values))
        summary[f"{name}_max"] = float(np.nanmax(values))
        summary[f"{name}_median"] = float(np.nanmedian(values))
        summary[f"{name}_abs_max"] = float(np.nanmax(np.abs(values)))
    return summary
