import numpy as np
from warpfactory.solver.finite_difference import derive_1st_4th_order
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
    term4 = np.zeros(metric_tensor.shape)
    for sigma in range(4):
        for rho in range(4):
            term4 += gamma[sigma, :, rho] * gamma[rho, :, sigma] # Note indices (mu from first, nu from second)
            # gamma[sigma, :, rho] is (mu, ...)
            # gamma[rho, :, sigma] is (nu, ...) --- NO!
            # gamma[rho, :, sigma] is (mu=nu_effective, ...)
            # Wait. Python array slicing gamma[rho, :, sigma] returns (mu_dim, ...)
            # We need (mu, nu) matrix.
            # Let's interact properly.
            # Gamma^sigma_{mu,rho} is matrix M1[mu]
            # Gamma^rho_{nu,sigma} is matrix M2[nu]
            # We want M1[mu] * M2[nu] -> (mu, nu) product
            # Outer product of vectors? No.
            # It's product of scalars at each point.
            # For fixed sigma, rho:
            # G1 = gamma[sigma, :, rho] -> shape (4, ...) (index mu)
            # G2 = gamma[rho, :, sigma] -> shape (4, ...) (index nu)
            # G1[mu] * G2[nu] -> (mu, nu)
            # We can use np.outer-like broadcasting?
            # G1[:, None, ...] * G2[None, :, ...]
            
            G1 = gamma[sigma, :, rho]
            G2 = gamma[rho, :, sigma]
            term4 += G1[:, np.newaxis, ...] * G2[np.newaxis, :, ...]
            
    R_mn = term1 - term2 + term3 - term4
    return R_mn

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
