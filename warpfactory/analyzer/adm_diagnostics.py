import numpy as np

from warpfactory.analyzer.spatial_geometry import spatial_christoffel, spatial_ricci_scalar
from warpfactory.solver.finite_difference import derive_1st_4th_order
from warpfactory.solver.tensor_utils import get_c3_inv


def decompose_adm(metric):
    """
    Decompose a metric into ADM alpha, beta_i, beta^i, gamma_ij, gamma^ij.

    The shift returned as beta_down is covariant beta_i = g_0i.
    """
    g = metric.tensor if hasattr(metric, "tensor") else metric
    gamma = g[1:4, 1:4]
    beta_down = g[0, 1:4]
    gamma_up = get_c3_inv(gamma)
    beta_up = np.einsum("ij...,j...->i...", gamma_up, beta_down)
    beta_sq = np.einsum("i...,i...->...", beta_down, beta_up)
    alpha_sq = beta_sq - g[0, 0]
    alpha = np.sqrt(np.maximum(alpha_sq, 0.0))
    return {
        "alpha": alpha,
        "beta_down": beta_down,
        "beta_up": beta_up,
        "beta_sq": beta_sq,
        "gamma": gamma,
        "gamma_up": gamma_up,
    }


def covariant_spatial_derivative_oneform(oneform, gamma, spatial_scale):
    """Return D_i oneform_j for a spatial one-form."""
    christoffel = spatial_christoffel(gamma, spatial_scale)
    derivative = np.zeros((3, 3) + oneform.shape[1:])

    for i in range(3):
        partial = derive_1st_4th_order(oneform, i + 2, spatial_scale[i])
        for j in range(3):
            correction = np.zeros(oneform.shape[1:])
            for k in range(3):
                correction += christoffel[k, i, j] * oneform[k]
            derivative[i, j] = partial[j] - correction

    return derivative


def extrinsic_curvature(metric, sign_convention="adm"):
    """
    Compute K_ij from ADM variables.

    For the common ADM convention:
        K_ij = (D_i beta_j + D_j beta_i - dt gamma_ij) / (2 alpha)

    The opposite sign is available for literature comparisons; scalar
    contractions K_ij K^ij and K^2 are unchanged by a global sign flip.
    """
    if sign_convention not in {"adm", "opposite"}:
        raise ValueError("sign_convention must be 'adm' or 'opposite'.")

    adm = decompose_adm(metric)
    alpha = adm["alpha"]
    beta_down = adm["beta_down"]
    gamma = adm["gamma"]
    spatial_scale = tuple(float(x) for x in metric.scaling[1:4])
    dt = float(metric.scaling[0])

    d_beta = covariant_spatial_derivative_oneform(beta_down, gamma, spatial_scale)
    dt_gamma = derive_1st_4th_order(gamma, 2, dt)

    numerator = d_beta + np.swapaxes(d_beta, 0, 1) - dt_gamma
    K = numerator / (2.0 * np.where(alpha == 0.0, np.nan, alpha))
    if sign_convention == "opposite":
        K = -K
    return K


def contract_spatial_rank2(tensor, gamma_up):
    """Return tensor_ij tensor^ij for a covariant spatial rank-2 tensor."""
    raised = np.einsum("ik...,jl...,kl...->ij...", gamma_up, gamma_up, tensor)
    return np.einsum("ij...,ij...->...", tensor, raised)


def trace_spatial_rank2(tensor, gamma_up):
    """Return gamma^ij tensor_ij."""
    return np.einsum("ij...,ij...->...", gamma_up, tensor)


def adm_constraint_terms(metric, sign_convention="adm"):
    """
    Compute ADM scalar terms useful for diagnosing energy balance.

    Returns R3, K, K_ij K^ij, K^2, and Hamiltonian combination
    R3 + K^2 - K_ij K^ij. With GR units, the Eulerian energy density is
    proportional to this Hamiltonian combination.
    """
    adm = decompose_adm(metric)
    gamma = adm["gamma"]
    gamma_up = adm["gamma_up"]
    spatial_scale = tuple(float(x) for x in metric.scaling[1:4])

    K_ij = extrinsic_curvature(metric, sign_convention=sign_convention)
    K_trace = trace_spatial_rank2(K_ij, gamma_up)
    KijKij = contract_spatial_rank2(K_ij, gamma_up)
    K_sq = K_trace**2
    R3 = spatial_ricci_scalar(gamma, spatial_scale)
    hamiltonian = R3 + K_sq - KijKij
    lapse = lapse_spatial_derivatives(metric)

    return {
        "alpha": adm["alpha"],
        "beta_sq": adm["beta_sq"],
        "gamma": gamma,
        "gamma_up": gamma_up,
        "K_ij": K_ij,
        "K_trace": K_trace,
        "KijKij": KijKij,
        "K_sq": K_sq,
        "R3": R3,
        "hamiltonian": hamiltonian,
        "lapse_grad_norm": lapse["grad_norm"],
        "lapse_log_grad_norm": lapse["log_grad_norm"],
        "lapse_laplacian": lapse["laplacian"],
        "lapse_log_laplacian": lapse["log_laplacian"],
    }


def covariant_spatial_laplacian_scalar(scalar, gamma, spatial_scale):
    """Return D^i D_i scalar for a spatial scalar field."""
    gamma_up = get_c3_inv(gamma)
    christoffel = spatial_christoffel(gamma, spatial_scale)

    partial = []
    for i in range(3):
        partial.append(derive_1st_4th_order(scalar, i + 1, spatial_scale[i]))
    partial = np.stack(partial, axis=0)

    hessian = np.zeros((3, 3) + scalar.shape)
    for i in range(3):
        partial_i = derive_1st_4th_order(partial, i + 2, spatial_scale[i])
        for j in range(3):
            correction = np.zeros(scalar.shape)
            for k in range(3):
                correction += christoffel[k, i, j] * partial[k]
            hessian[i, j] = partial_i[j] - correction

    return np.einsum("ij...,ij...->...", gamma_up, hessian)


def lapse_spatial_derivatives(metric):
    """
    Compute lapse-gradient diagnostics.

    Spatial gradients of alpha do not appear in the Hamiltonian constraint the
    way R3 does, but they contribute to spatial/evolution equations and are
    therefore important for pressure and NEC diagnostics.
    """
    adm = decompose_adm(metric)
    alpha = adm["alpha"]
    gamma = adm["gamma"]
    gamma_up = adm["gamma_up"]
    spatial_scale = tuple(float(x) for x in metric.scaling[1:4])

    safe_alpha = np.maximum(alpha, 1e-300)
    log_alpha = np.log(safe_alpha)

    grad = []
    log_grad = []
    for i in range(3):
        grad.append(derive_1st_4th_order(alpha, i + 1, spatial_scale[i]))
        log_grad.append(derive_1st_4th_order(log_alpha, i + 1, spatial_scale[i]))
    grad = np.stack(grad, axis=0)
    log_grad = np.stack(log_grad, axis=0)

    grad_norm_sq = np.einsum("ij...,i...,j...->...", gamma_up, grad, grad)
    log_grad_norm_sq = np.einsum("ij...,i...,j...->...", gamma_up, log_grad, log_grad)

    return {
        "grad": grad,
        "log_grad": log_grad,
        "grad_norm": np.sqrt(np.maximum(grad_norm_sq, 0.0)),
        "log_grad_norm": np.sqrt(np.maximum(log_grad_norm_sq, 0.0)),
        "laplacian": covariant_spatial_laplacian_scalar(alpha, gamma, spatial_scale),
        "log_laplacian": covariant_spatial_laplacian_scalar(log_alpha, gamma, spatial_scale),
    }


def adm_diagnostics_summary(metric):
    """Compact scalar summary for ADM balance diagnostics."""
    terms = adm_constraint_terms(metric)
    summary = {}
    for name in (
        "alpha",
        "beta_sq",
        "K_trace",
        "KijKij",
        "K_sq",
        "R3",
        "hamiltonian",
        "lapse_grad_norm",
        "lapse_log_grad_norm",
        "lapse_laplacian",
        "lapse_log_laplacian",
    ):
        values = terms[name]
        summary[f"{name}_min"] = float(np.nanmin(values))
        summary[f"{name}_max"] = float(np.nanmax(values))
        summary[f"{name}_abs_max"] = float(np.nanmax(np.abs(values)))
        summary[f"{name}_median"] = float(np.nanmedian(values))
    return summary
