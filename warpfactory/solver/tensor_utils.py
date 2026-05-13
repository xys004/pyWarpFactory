import numpy as np
from warpfactory.solver.finite_difference import derive_1st_4th_order, derive_2nd_4th_order
from warpfactory.constants import C, G

def verify_tensor(tensor, raise_error=True):
    """
    Verifies if the input object is a valid tensor/metric structure.
    
    Args:
        tensor: Metric object or dict or custom structure.
        raise_error (bool): If True, raises ValueError on failure.
        
    Returns:
        bool: True if valid.
    """
    # Check if it has 'tensor', 'coords', 'type', 'index' attributes
    # The Metric class in base.py has these.
    # Note: 'type' might not be in base Metric class yet, but is used in MATLAB.
    # We should allow it to be optional or check if it's a Metric instance.
    
    required_attrs = ['tensor', 'coords', 'index']
    
    for attr in required_attrs:
        if not hasattr(tensor, attr) and not (isinstance(tensor, dict) and attr in tensor):
            if raise_error:
                raise ValueError(f"Tensor object missing required attribute: {attr}")
            return False
            
    # Get values
    t_val = tensor.tensor if hasattr(tensor, 'tensor') else tensor['tensor']
    
    # Check tensor shape
    # Must be (4, 4, ...)
    if not isinstance(t_val, np.ndarray):
        if raise_error:
            raise ValueError("Tensor 'tensor' field must be a numpy array.")
        return False
        
    if t_val.ndim < 2 or t_val.shape[0] != 4 or t_val.shape[1] != 4:
        if raise_error:
            raise ValueError(f"Tensor must have shape (4, 4, ...). Got {t_val.shape}")
        return False
        
    return True

def get_c4_inv(metric_tensor):
    """
    Computes the inverse of the metric tensor (g^{mu,nu}).
    Input shape: (4, 4, ...)
    Output shape: (4, 4, ...)
    """
    return _get_inv(metric_tensor)

def get_c3_inv(spatial_metric):
    """
    Computes the inverse of a 3x3 spatial metric tensor (gamma^{ij}).
    Input shape: (3, 3, ...)
    Output shape: (3, 3, ...)
    """
    return _get_inv(spatial_metric)

def _get_inv(tensor):
    """Helper to invert tensor of shape (N, N, ...)."""
    # Move axes to end
    tensor_T = np.moveaxis(tensor, [0, 1], [-2, -1])
    inv_T = np.linalg.inv(tensor_T)
    inv = np.moveaxis(inv_T, [-2, -1], [0, 1])
    return inv

def get_christoffel(metric_tensor, grid_scale):
    """
    Computes Christoffel symbols Gamma^lambda_{mu,nu}.
    Shape: (4, 4, 4, ...) [lambda, mu, nu, ...]
    """
    g_inv = get_c4_inv(metric_tensor)
    
    # Compute derivatives of metric: partial_rho g_{mu,nu}
    # Dimensions: (4 [rho], 4 [mu], 4 [nu], t, x, y, z)
    # Note: spatial dimensions are 0 (t), 1 (x), 2 (y), 3 (z) effectively in the grid
    # But in the metric tensor (4, 4, ...), the "..." are dimensions 2, 3, 4, 5.
    
    dt, dx, dy, dz = grid_scale
    deltas = [dt, dx, dy, dz]
    
    d_g = []
    for rho in range(4):
        # We need to differentiate along axis `rho + 2` (since 0,1 are mu,nu)
        # derive_1st_4th_order takes axis index relative to the array
        deriv = derive_1st_4th_order(metric_tensor, rho + 2, deltas[rho])
        d_g.append(deriv)
        
    d_g = np.stack(d_g, axis=0) # (4 [rho], 4 [mu], 4 [nu], ...)
    
    # Standard formula:
    # Gamma^lam_{mu,nu} = 0.5 * g^{lam,rho} * (d_mu g_{nu,rho} + d_nu g_{mu,rho} - d_rho g_{mu,nu})
    
    # We use Einstein summation for broadcasting
    # g_inv: (lam, rho, ...)
    # d_g: (tau, mu, nu, ...) where tau is derivative index
    
    # d_mu g_{nu,rho}: d_g[mu, nu, rho]
    # d_nu g_{mu,rho}: d_g[nu, mu, rho]
    # d_rho g_{mu,nu}: d_g[rho, mu, nu]
    
    # Let's perform the term inside parenthesis first
    # We need intermediate shape: (lam, mu, nu, ...)
    
    # This is heavy on memory if we just broadcast everything.
    # Loop over lambda, mu, nu is safest for memory if grid is large.
    # But for optimization, we can use einsum on the first few axes.
    # But the spatial dimensions ... are large.
    
    # Efficient approach with loops for 4x4x4:
    shape = metric_tensor.shape
    gamma = np.zeros((4, 4, 4) + shape[2:]) # (lam, mu, nu, ...)
    
    for lam in range(4):
        for mu in range(4):
            for nu in range(4):
                val = 0
                for rho in range(4):
                    term = d_g[mu, nu, rho] + d_g[nu, mu, rho] - d_g[rho, mu, nu]
                    val += 0.5 * g_inv[lam, rho] * term
                gamma[lam, mu, nu] = val
                
    return gamma

def get_ricci_tensor(metric_tensor, grid_scale):
    """
    Computes Ricci tensor R_{mu,nu}.
    """
    gamma = get_christoffel(metric_tensor, grid_scale)
    
    # Derivatives of Gamma: partial_rho Gamma^lambda_{mu,nu}
    # We need partial_rho Gamma^rho_{mu,nu} and partial_nu Gamma^rho_{mu,rho}
    
    dt, dx, dy, dz = grid_scale
    deltas = [dt, dx, dy, dz]
    
    # R_mn = d_rho Gamma^rho_mn - d_nu Gamma^rho_mrho + Gamma^rho_mn Gamma^sigma_rho_sigma - Gamma^sigma_mr Gamma^rho_ns
    
    # Term 1: d_rho Gamma^rho_{mu,nu}
    # Sum over rho of partial_rho (Gamma[rho, mu, nu])
    term1 = np.zeros(metric_tensor.shape)
    for rho in range(4):
        # Differentiate Gamma[rho, mu, nu] along axis rho+3 (rho index is axis 0, mu=1, nu=2, then t,x,y,z start at 3)
        # Wait, Gamma shape is (4, 4, 4, t, x, y, z).
        # Axis corresponding to coordinate 'rho' is 'rho + 3'.
        # derive_1st_4th_order(A, axis, delta)
        g_rho_mu_nu = gamma[rho, :, :] # shape (4, 4, t, x, y, z)
        term1 += derive_1st_4th_order(g_rho_mu_nu, rho + 2, deltas[rho]) # +2 because shape is (mu, nu, t, x, y, z)
        
    # Term 2: d_nu Gamma^rho_{mu,rho}
    # We first contract Gamma^rho_{mu,rho} -> Vector V_mu
    # V_mu = sum_rho Gamma[rho, mu, rho]
    V = np.zeros((4,) + metric_tensor.shape[2:]) # (mu, ...)
    for rho in range(4):
        V += gamma[rho, :, rho]
        
    term2 = np.zeros(metric_tensor.shape)
    for nu in range(4):
        # differentiate V_mu along nu
        # V is (mu, ...). axis for nu is nu+1
        d_nu_V = derive_1st_4th_order(V, nu + 1, deltas[nu])
        # d_nu_V shape is (mu, ...)
        # We want result (mu, nu, ...)
        # So we place it into term2
        # d_nu_V[mu] goes to term2[mu, nu]
        for mu in range(4):
            term2[mu, nu] = d_nu_V[mu]
            
    # Term 3: Gamma^rho_{mu,nu} Gamma^sigma_{rho,sigma}
    # Contract Gamma^sigma_{rho,sigma} -> Vector V_rho (same as V above)
    # Result += Gamma[rho, mu, nu] * V[rho]
    term3 = np.zeros(metric_tensor.shape)
    for rho in range(4):
        term3 += gamma[rho, :, :] * V[rho] # Broadcasting V[rho] (shape ...) against Gamma (4, 4, ...)
        
    # Term 4: Gamma^sigma_{mu,rho} Gamma^rho_{nu,sigma}
    # For fixed sigma, rho:
    #   G1[mu] = Gamma^sigma_{mu, rho}  -> shape (4, ...)
    #   G2[nu] = Gamma^rho_{nu, sigma}  -> shape (4, ...)
    #   Outer product: G1[mu] * G2[nu]  -> shape (4, 4, ...)
    # Sum over sigma and rho gives the full (4, 4, ...) term.
    term4 = np.zeros(metric_tensor.shape)
    for sigma in range(4):
        for rho in range(4):
            G1 = gamma[sigma, :, rho]                            # (4, ...)
            G2 = gamma[rho, :, sigma]                            # (4, ...)
            term4 += G1[:, np.newaxis, ...] * G2[np.newaxis, :, ...]  # (4, 4, ...)
            
    R_mn = term1 - term2 + term3 - term4
    return R_mn


def get_warpfactory_direct_metric_derivatives(metric_tensor, grid_scale):
    """
    Build the metric derivative arrays used by WarpFactory's direct Ricci path.

    Returns:
        diff_1[mu][nu][coord] = partial_coord g_{mu,nu}
        diff_2[mu][nu][coord1][coord2] = partial_coord1 partial_coord2 g_{mu,nu}

    The time-derivative scaling mirrors get_ricci_tensor_warpfactory_direct.
    """
    deltas = [float(x) for x in grid_scale]
    diff_1 = [[[None for _ in range(4)] for _ in range(4)] for _ in range(4)]
    diff_2 = [[[[None for _ in range(4)] for _ in range(4)] for _ in range(4)] for _ in range(4)]

    for i in range(4):
        for j in range(4):
            gij = metric_tensor[i, j]
            for k in range(4):
                value = derive_1st_4th_order(gij, k, deltas[k])
                if k == 0:
                    value = value / C
                diff_1[i][j][k] = value

                for n in range(4):
                    value2 = derive_2nd_4th_order(gij, k, n, deltas[k], deltas[n])
                    if (n == 0 and k != 0) or (n != 0 and k == 0):
                        value2 = value2 / C
                    elif n == 0 and k == 0:
                        value2 = value2 / (C**2)
                    diff_2[i][j][k][n] = value2

    return diff_1, diff_2


def get_christoffel_from_metric_derivatives(metric_tensor, diff_1):
    """
    Compute Christoffel symbols from precomputed metric first derivatives.

    The diff_1 layout is diff_1[mu][nu][coord], matching
    get_warpfactory_direct_metric_derivatives.
    """
    g_inv = get_c4_inv(metric_tensor)
    shape = metric_tensor.shape[2:]
    gamma = np.zeros((4, 4, 4) + shape)

    for lam in range(4):
        for mu in range(4):
            for nu in range(4):
                value = np.zeros(shape)
                for rho in range(4):
                    term = (
                        diff_1[nu][rho][mu]
                        + diff_1[mu][rho][nu]
                        - diff_1[mu][nu][rho]
                    )
                    value = value + 0.5 * g_inv[lam, rho] * term
                gamma[lam, mu, nu] = value

    return gamma


def get_christoffel_derivative_from_metric_derivatives(metric_tensor, diff_1, diff_2):
    """
    Compute partial_sigma Gamma^lam_{mu,nu} from metric first/second derivatives.

    The returned layout is d_gamma[sigma, lam, mu, nu, ...].
    """
    g_inv = get_c4_inv(metric_tensor)
    shape = metric_tensor.shape[2:]
    d_g_inv = np.zeros((4, 4, 4) + shape)

    for sigma in range(4):
        for lam in range(4):
            for rho in range(4):
                value = np.zeros(shape)
                for alpha in range(4):
                    for beta in range(4):
                        value = value - (
                            g_inv[lam, alpha]
                            * g_inv[rho, beta]
                            * diff_1[alpha][beta][sigma]
                        )
                d_g_inv[sigma, lam, rho] = value

    d_gamma = np.zeros((4, 4, 4, 4) + shape)
    for sigma in range(4):
        for lam in range(4):
            for mu in range(4):
                for nu in range(4):
                    value = np.zeros(shape)
                    for rho in range(4):
                        first_term = (
                            diff_1[nu][rho][mu]
                            + diff_1[mu][rho][nu]
                            - diff_1[mu][nu][rho]
                        )
                        second_term = (
                            diff_2[nu][rho][sigma][mu]
                            + diff_2[mu][rho][sigma][nu]
                            - diff_2[mu][nu][sigma][rho]
                        )
                        value = value + 0.5 * (
                            d_g_inv[sigma, lam, rho] * first_term
                            + g_inv[lam, rho] * second_term
                        )
                    d_gamma[sigma, lam, mu, nu] = value

    return d_gamma


def get_ricci_tensor_christoffel_from_warpfactory_derivatives(metric_tensor, grid_scale):
    """
    Compute Ricci via the Christoffel formula using the direct solver's
    precomputed metric derivative arrays.

    This is a diagnostic bridge: if it agrees with get_ricci_tensor while
    get_ricci_tensor_warpfactory_direct disagrees, the finite-difference arrays
    are not the primary problem and the direct Ricci expansion is suspect.
    """
    diff_1, diff_2 = get_warpfactory_direct_metric_derivatives(metric_tensor, grid_scale)
    gamma = get_christoffel_from_metric_derivatives(metric_tensor, diff_1)
    d_gamma = get_christoffel_derivative_from_metric_derivatives(metric_tensor, diff_1, diff_2)
    shape = metric_tensor.shape[2:]

    trace_gamma = np.zeros((4,) + shape)
    for rho in range(4):
        for sigma in range(4):
            trace_gamma[rho] = trace_gamma[rho] + gamma[sigma, rho, sigma]

    ricci = np.zeros(metric_tensor.shape)
    for mu in range(4):
        for nu in range(4):
            value = np.zeros(shape)
            for rho in range(4):
                value = value + d_gamma[rho, rho, mu, nu]
                value = value - d_gamma[nu, rho, mu, rho]
                value = value + gamma[rho, mu, nu] * trace_gamma[rho]
                for sigma in range(4):
                    value = value - gamma[sigma, mu, rho] * gamma[rho, nu, sigma]
            ricci[mu, nu] = value

    return ricci


def get_ricci_tensor_warpfactory_direct(metric_tensor, grid_scale):
    """
    Port of MATLAB WarpFactory's ricciT.m direct Ricci implementation.

    This uses first and second finite differences of the covariant metric
    directly, instead of differentiating Christoffel symbols. It is intended
    for numerical parity with MATLAB WarpFactory's met2den.m.
    """
    g_inv = get_c4_inv(metric_tensor)
    shape = metric_tensor.shape[2:]
    diff_1, diff_2 = get_warpfactory_direct_metric_derivatives(metric_tensor, grid_scale)

    ricci = np.zeros((4, 4) + shape)
    for i in range(4):
        for j in range(i, 4):
            rij = np.zeros(shape)
            for a in range(4):
                for b in range(4):
                    temp2 = -(
                        diff_2[i][j][a][b]
                        + diff_2[a][b][i][j]
                        - diff_2[i][b][j][a]
                        - diff_2[j][b][i][a]
                    )

                    for r in range(4):
                        temp3 = np.zeros(shape)
                        temp4 = np.zeros(shape)
                        temp5 = np.zeros(shape)
                        for d in range(4):
                            temp3 = temp3 + diff_1[b][d][j] * g_inv[r, d]
                            temp4 = temp4 + (diff_1[j][d][b] - diff_1[j][b][d]) * g_inv[r, d]
                            temp5 = temp5 - (
                                diff_1[b][d][a]
                                + diff_1[b][d][a]
                                - diff_1[a][b][d]
                            ) * g_inv[r, d]

                        temp2 = temp2 + (
                            temp4 * diff_1[i][r][a]
                            + 0.5
                            * temp3
                            * diff_1[a][r][i]
                            + 0.5
                            * temp5
                            * (diff_1[j][r][i] + diff_1[i][r][j] - diff_1[j][i][r])
                        )

                    rij = rij + g_inv[a, b] * temp2
            ricci[i, j] = 0.5 * rij
            ricci[j, i] = ricci[i, j]

    return ricci

def get_ricci_scalar(R_mn, g_inv):
    """
    Computes Ricci scalar R = R_{mu,nu} g^{mu,nu}
    """
    # Sum over mu, nu provided by einsum or double loop
    R = np.zeros(R_mn.shape[2:]) # Scalar field (...)
    
    for mu in range(4):
        for nu in range(4):
            R += R_mn[mu, nu] * g_inv[mu, nu]
            
    return R

def get_einstein_tensor(R_mn, R, metric_tensor):
    """
    Computes Einstein tensor G_{mu,nu} = R_{mu,nu} - 0.5 * R * g_{mu,nu}
    """
    G_mn = np.zeros(metric_tensor.shape)
    
    for mu in range(4):
        for nu in range(4):
            G_mn[mu, nu] = R_mn[mu, nu] - 0.5 * R * metric_tensor[mu, nu]
            
    return G_mn

def get_energy_tensor(G_mn, g_inv):
    """
    Computes Stress-Energy Tensor T^{mu,nu} (Contravariant).
    G_{mu,nu} = (8 pi G / c^4) T_{mu,nu}
    => T_{mu,nu} = (c^4 / 8 pi G) G_{mu,nu}
    
    Then raise indices: T^{mu,nu} = T_{ab} g^{am} g^{bn}
    """
    factor = (C**4) / (8 * np.pi * G)
    T_covariant = factor * G_mn
    
    # Raise indices
    T_contravariant = np.zeros(G_mn.shape)
    
    # T^{mu,nu} = sum_a sum_b T_{ab} g^{am} g^{bn}
    # T^{mu,nu} = sum_a g^{am} (sum_b T_{ab} g^{bn})
    
    # Let's do matrix multiplication for the 4x4 logic axis
    # Treat T_cov as matrix T, g_inv as matrix Inv
    # T_con = Inv * T * Inv^T (since T symmetric? No, generally indices order matters)
    # T^{mu,nu} = g^{mu, alpha} T_{alpha, beta} g^{nu, beta} (g symmetric means g^{beta, nu} = g^{nu, beta})
    # So T_con = g_inv @ T_cov @ g_inv
    
    # We need to perform this matrix mult over the last axes if we used the transposed shape
    # My arrays are (4, 4, ...)
    # Let's move axes again
    T_cov_T = np.moveaxis(T_covariant, [0, 1], [-2, -1])
    g_inv_T = np.moveaxis(g_inv, [0, 1], [-2, -1])
    
    T_con_T = np.matmul(np.matmul(g_inv_T, T_cov_T), g_inv_T)
    
    T_contravariant = np.moveaxis(T_con_T, [-2, -1], [0, 1])
    
    return T_contravariant
