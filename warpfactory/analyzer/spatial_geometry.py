import numpy as np

from warpfactory.solver.finite_difference import derive_1st_4th_order
from warpfactory.solver.tensor_utils import get_c3_inv


def extract_spatial_metric(metric):
    """Return the covariant spatial metric gamma_ij from a 4D metric."""
    tensor = metric.tensor if hasattr(metric, "tensor") else metric
    return tensor[1:4, 1:4]


def spatial_christoffel(gamma, spatial_scale):
    """
    Compute 3D Christoffel symbols Gamma^i_jk for gamma_ij.

    gamma shape is (3, 3, Nt, Nx, Ny, Nz). Time is treated as a batch axis;
    derivatives are taken only along x, y, z.
    """
    gamma_up = get_c3_inv(gamma)
    deltas = spatial_scale

    d_gamma = []
    for coord in range(3):
        d_gamma.append(derive_1st_4th_order(gamma, coord + 3, deltas[coord]))
    d_gamma = np.stack(d_gamma, axis=0)

    shape = gamma.shape[2:]
    christoffel = np.zeros((3, 3, 3) + shape)
    for upper in range(3):
        for lower_a in range(3):
            for lower_b in range(3):
                value = 0.0
                for summed in range(3):
                    term = (
                        d_gamma[lower_a, lower_b, summed]
                        + d_gamma[lower_b, lower_a, summed]
                        - d_gamma[summed, lower_a, lower_b]
                    )
                    value += 0.5 * gamma_up[upper, summed] * term
                christoffel[upper, lower_a, lower_b] = value
    return christoffel


def spatial_ricci_tensor(gamma, spatial_scale):
    """Compute the 3D Ricci tensor of a spatial metric gamma_ij."""
    christoffel = spatial_christoffel(gamma, spatial_scale)
    deltas = spatial_scale
    shape = gamma.shape[2:]

    # R_ij = d_k Gamma^k_ij - d_j Gamma^k_ik
    #        + Gamma^k_ij Gamma^l_kl - Gamma^l_ik Gamma^k_jl
    ricci = np.zeros((3, 3) + shape)

    trace_conn = np.zeros((3,) + shape)
    for lower in range(3):
        for upper in range(3):
            trace_conn[lower] += christoffel[upper, lower, upper]

    for i in range(3):
        for j in range(3):
            term1 = np.zeros(shape)
            for k in range(3):
                term1 += derive_1st_4th_order(christoffel[k, i, j], k + 1, deltas[k])

            term2 = derive_1st_4th_order(trace_conn[i], j + 1, deltas[j])

            term3 = np.zeros(shape)
            for k in range(3):
                term3 += christoffel[k, i, j] * trace_conn[k]

            term4 = np.zeros(shape)
            for lower in range(3):
                for k in range(3):
                    term4 += christoffel[lower, i, k] * christoffel[k, j, lower]

            ricci[i, j] = term1 - term2 + term3 - term4

    return ricci


def spatial_ricci_scalar(gamma, spatial_scale):
    """Compute the 3D Ricci scalar of a spatial metric gamma_ij."""
    ricci = spatial_ricci_tensor(gamma, spatial_scale)
    gamma_up = get_c3_inv(gamma)
    scalar = np.zeros(gamma.shape[2:])
    for i in range(3):
        for j in range(3):
            scalar += gamma_up[i, j] * ricci[i, j]
    return scalar


def spatial_geometry_summary(metric, compute_curvature=True):
    """
    Summarize the geometry of the spatial hypersurface gamma_ij.

    This is a diagnostic: it is meant to show whether the slice is flat-like
    or carries non-trivial intrinsic spatial curvature.
    """
    gamma = extract_spatial_metric(metric)
    gamma_matrix = np.moveaxis(gamma, [0, 1], [-2, -1])
    identity = np.eye(3).reshape((1,) * (gamma_matrix.ndim - 2) + (3, 3))
    gamma_minus_identity = gamma_matrix - identity

    eigenvalues = np.linalg.eigvalsh(gamma_matrix)
    determinant = np.linalg.det(gamma_matrix)

    summary = {
        "max_abs_gamma_minus_identity": float(np.nanmax(np.abs(gamma_minus_identity))),
        "max_abs_spatial_offdiag": float(
            np.nanmax(np.abs(gamma - np.eye(3).reshape(3, 3, 1, 1, 1, 1) * gamma))
        ),
        "gamma_det_min": float(np.nanmin(determinant)),
        "gamma_det_max": float(np.nanmax(determinant)),
        "gamma_eigen_min": float(np.nanmin(eigenvalues)),
        "gamma_eigen_max": float(np.nanmax(eigenvalues)),
    }

    if compute_curvature:
        spatial_scale = tuple(float(x) for x in metric.scaling[1:4])
        ricci_scalar = spatial_ricci_scalar(gamma, spatial_scale)
        summary.update(
            {
                "spatial_ricci_scalar_min": float(np.nanmin(ricci_scalar)),
                "spatial_ricci_scalar_max": float(np.nanmax(ricci_scalar)),
                "spatial_ricci_scalar_abs_max": float(np.nanmax(np.abs(ricci_scalar))),
                "spatial_ricci_scalar_median": float(np.nanmedian(ricci_scalar)),
            }
        )

    return summary
