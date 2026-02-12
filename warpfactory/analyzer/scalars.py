import numpy as np
from warpfactory.solver.tensor_utils import get_c4_inv
from warpfactory.solver.finite_difference import derive_1st_4th_order
from warpfactory.generator.commons import get_minkowski_metric

def three_plus_one_decomposer(metric):
    """
    Decomposes the metric into 3+1 ADM variables.
    
    Args:
        metric (Metric): Metric object or tensor
        
    Returns:
        tuple: (alpha, beta_down, gamma, beta_up, gamma_up)
               alpha: Lapse function (scalar)
               beta_down: Shift vector (covariant)
               gamma: Spatial metric (covariant)
               beta_up: Shift vector (contravariant)
               gamma_up: Spatial metric (contravariant)
    """
    g = metric.tensor if hasattr(metric, 'tensor') else metric
    
    # Grid scale needed for derivatives? No, just algebraic decomposition.
    
    # ADM Formalism:
    # ds^2 = -alpha^2 dt^2 + gamma_ij (dx^i + beta^i dt)(dx^j + beta^j dt)
    # g_00 = -alpha^2 + beta_k beta^k
    # g_0i = beta_i (lowered with gamma) -> Wait. g_0i = gamma_ij beta^j = beta_i.
    # g_ij = gamma_ij
    
    # 1. Extract gamma_ij (spatial metric)
    # Indices 1, 2, 3 correspond to x, y, z
    gamma = g[1:, 1:] # Shape (3, 3, ...)
    
    # 2. Extract beta_i (covariant shift) = g_0i
    beta_down = g[0, 1:] # Shape (3, ...)
    
    # 3. Calculate inverse spatial metric gamma^{ij}
    # Move axes for inversion
    gamma_T = np.moveaxis(gamma, [0, 1], [-2, -1])
    gamma_up_T = np.linalg.inv(gamma_T)
    gamma_up = np.moveaxis(gamma_up_T, [-2, -1], [0, 1])
    
    # 4. Calculate beta^i (contravariant shift)
    # beta^i = gamma^{ij} beta_j
    # einsum: ij...,j... -> i...
    # gamma_up (3, 3, ...)
    # beta_down (3, ...)
    beta_up = np.einsum('ij...,j...->i...', gamma_up, beta_down)
    
    # 5. Calculate alpha (lapse)
    # g_00 = -alpha^2 + beta_k beta^k
    # alpha^2 = beta_k beta^k - g_00
    # beta_k beta^k = beta_i beta^i = beta_down_i beta_up^i
    beta_sq = np.sum(beta_down * beta_up, axis=0) # Sum over i
    
    g_00 = g[0, 0]
    alpha_sq = beta_sq - g_00
    
    # Avoid negative sqrt issues (numerical noise)
    alpha = np.sqrt(np.abs(alpha_sq)) 
    
    return alpha, beta_down, gamma, beta_up, gamma_up

def get_covariant_derivative(tensor, metric, grid_scale):
    """
    Computes covariant derivative of a tensor.
    Currently only supports vector (rank 1) or specific tensors needed for scalars?
    The MATLAB code implements specific `covDiv` function.
    
    For scalars calculation:
    We need del_mu U_nu. U is 4-velocity.
    del_mu U_nu = partial_mu U_nu - Gamma^lambda_{mu,nu} U_lambda
    
    Args:
        tensor: The tensor to differentiate.
        metric: The metric tensor.
        grid_scale: grid spacing.
        
    Returns:
        covariant derivative.
    """
    from warpfactory.solver.tensor_utils import get_christoffel
    
    gamma = get_christoffel(metric.tensor, grid_scale)
    
    # Assume tensor is rank 1 covariant (U_nu) for now as used in getScalars
    # If generic needed, need rank detection.
    # In getScalars: uUp and uDown are calculated.
    # delU{i,j} = covariant derivative of u_down?
    # MATLAB: covDiv(..., uUpCell, uDownCell, i, j, ...)
    # It calculates nabla_i U_j.
    # i is derivative index? j is component index?
    # "delU{a,b} = \nabla_a U_b"
    
    # So we want nabla_mu U_nu.
    # Input tensor should be U_nu (covariant vector)
    pass 

def calculate_scalars(metric, grid_scale):
    """
    Calculates Expansion, Shear, and Vorticity scalars.
    
    Args:
        metric (Metric): Metric object
        grid_scale (tuple): (dt, dx, dy, dz)
        
    Returns:
        dict: {'expansion': ..., 'shear': ..., 'vorticity': ...}
    """
    # 1. 3+1 Decomposition
    alpha, _, _, beta_up, _ = three_plus_one_decomposer(metric)
    
    # 2. Construct Eulerian Observer 4-velocity n^mu (or u^mu)
    # n^mu = (1/alpha, -beta^i/alpha)
    # indices 0, 1, 2, 3
    
    u_up = np.zeros_like(metric.tensor[:, 0]) # Shape (4, ...)
    u_up[0] = 1.0 / alpha
    u_up[1] = -beta_up[0] / alpha
    u_up[2] = -beta_up[1] / alpha
    u_up[3] = -beta_up[2] / alpha
    
    # u_down = g_munu u^nu
    u_down = np.einsum('mn...,n...->m...', metric.tensor, u_up)
    
    # 3. Covariant Derivative nabla_mu u_nu
    # partial_mu u_nu
    dt, dx, dy, dz = grid_scale
    deltas = [dt, dx, dy, dz]
    
    d_u_down = np.zeros((4, 4) + u_down.shape[1:]) # (mu, nu, ...)
    
    for mu in range(4):
        # Differentiate u_down along axis mu (corresponding to t,x,y,z)
        # u_down has shape (4, t, x, y, z).
        # derive axis: mu + 1
        deriv = derive_1st_4th_order(u_down, mu + 1, deltas[mu]) # result (4, ...)
        
        # deriv is partial_mu u_nu (where nu is the first dimension of u_down)
        # We want to store as d_u_down[mu, nu]
        # reshape/moveaxis?
        # deriv[nu] is partial_mu u_nu
        d_u_down[mu] = deriv # (nu, ...)
        
    # Christoffel
    from warpfactory.solver.tensor_utils import get_christoffel
    Gamma = get_christoffel(metric.tensor, grid_scale) # (lam, mu, nu, ...)
    
    # nabla_mu u_nu = partial_mu u_nu - Gamma^lam_{mu,nu} u_lam
    # Gamma contraction with u_lam (u_down is covariant, we need u_lam as component of u_down? No. u_lam is COVARIANT in formula?
    # Standard: nabla_mu V_nu = partial_mu V_nu - Gamma^lam_{mu,nu} V_lam
    # Wait, indices.
    # V_lam term: Gamma^lam_{mu,nu} matches upper index lam. V must be V_lam (covariant)?
    # NO. Christoffel matches upper index with vector component?
    # Covariant derivative of One-form (covariant vector):
    # nabla_mu w_nu = partial_mu w_nu - Gamma^lambda_{mu,nu} w_lambda
    # So we contract with w_lambda (covariant vector)?
    # NO. 
    # Standard formula: nabla_mu V^nu = partial_mu V^nu + Gamma^nu_{mu,lam} V^lam
    # nabla_mu V_nu = partial_mu V_nu - Gamma^lam_{mu,nu} V_lam
    # Yes, contract with V_lam (covariant component)?
    # Let's check Einstein summation.
    # The term is subtracting connection coefficient * vector component.
    # Gamma^lam_{mu,nu} has one upper index lam.
    # We need to sum over lam.
    # So we need V_lam (covariant)?
    # Logic: V_nu is component. We want correction.
    # Actually, usually written as Gamma^rho_{mu,nu} V_rho ?
    # Let's check dimensions.
    # Gamma (lam, mu, nu).
    # We sum over lam.
    # So we need a quantity with index lam.
    # Is it V_lam (covariant) or V^lam (contravariant)?
    # For covariant derivative of COVARIANT vector V_nu:
    # Correction term is - Gamma^lam_{mu,nu} V_lam.
    # Yes, V_lam is the covariant vector.
    
    conn_term = np.einsum('lmn...,l...->mn...', Gamma, u_down)
    
    nabla_u = d_u_down - conn_term # (mu, nu, ...)
    
    # 4. Decomposition of nabla_u
    # Projector h_ab = g_ab + u_a u_b
    # P{i,j} in MATLAB
    u_down_u_down = u_down[:, np.newaxis, ...] * u_down[np.newaxis, :, ...]
    h = metric.tensor + u_down_u_down
    
    # Mixed projector P^a_b = delta^a_b + u^a u_b
    delta = np.eye(4)
    # Broadcast delta
    delta_shape = (4, 4) + (1,)*(len(u_down.shape)-1)
    delta_broad = delta.reshape(delta_shape)
    
    P_mix = delta_broad + u_up[:, np.newaxis, ...] * u_down[np.newaxis, :, ...]
    
    # Kinematical decomposition
    # \nabla_{alpha} U_{beta}
    delU = nabla_u
    
    # Symmetric part (Expansion tensor + Shear) vs Antisymmetric (Vorticity)
    # theta_ab = P^m_a P^n_b (1/2) (nabla_m u_n + nabla_n u_m)
    # omega_ab = P^m_a P^n_b (1/2) (nabla_m u_n - nabla_n u_m)
    
    sym_delU = 0.5 * (delU + np.moveaxis(delU, [0, 1], [1, 0]))
    asym_delU = 0.5 * (delU - np.moveaxis(delU, [0, 1], [1, 0]))
    
    # Project indices
    # theta_ab = P_mix[m, a] * P_mix[n, b] * sym[m, n]
    # P_mix is (upper, lower). P^m_a is P_mix[m, a].
    # We want contraction on m, n.
    # theta[a, b] = sum_m sum_n P_mix[m, a] * P_mix[n, b] * sym[m, n]
    
    theta_tensor = np.einsum('ma...,nb...,mn...->ab...', P_mix, P_mix, sym_delU)
    omega_tensor = np.einsum('ma...,nb...,mn...->ab...', P_mix, P_mix, asym_delU)
    
    # 5. Scalars
    # Expansion scalar theta = trace(theta) = g^ab theta_ab
    # Or just trace of mixed? theta^a_a.
    # theta^a_b = g^ac theta_cb
    # g_inv needed.
    g_inv = get_c4_inv(metric.tensor)
    
    theta_up_down = np.einsum('ac...,cb...->ab...', g_inv, theta_tensor)
    expansion_scalar = np.trace(theta_up_down) # trace over first two axes?
    # np.trace returns sum of diag.
    # But shape is (4, 4, ...). trace(A) sums A[i,i].
    # We want trace for each grid point.
    expansion_scalar = np.einsum('aa...->...', theta_up_down)
    
    # Shear scalar sigma^2 = 1/2 sigma_ab sigma^ab
    # sigma_ab = theta_ab - 1/3 theta h_ab
    # h_ab is projector (metric on hypersurface)
    h_tensor = h
    
    # Broadcast expansion scalar
    theta_shape = (1, 1) + expansion_scalar.shape
    theta_broad = expansion_scalar.reshape(theta_shape)
    
    sigma_tensor = theta_tensor - (1.0/3.0) * theta_broad * h_tensor
    
    # Raise indices for contraction
    sigma_up = np.einsum('ac...,bd...,cd...->ab...', g_inv, g_inv, sigma_tensor)
    # sigma_ab sigma^ab
    shear_contract = np.einsum('ab...,ab...->...', sigma_tensor, sigma_up)
    shear_scalar = 0.5 * shear_contract
    
    # Vorticity scalar omega^2 = 1/2 omega_ab omega^ab
    omega_up = np.einsum('ac...,bd...,cd...->ab...', g_inv, g_inv, omega_tensor)
    vorticity_contract = np.einsum('ab...,ab...->...', omega_tensor, omega_up)
    vorticity_scalar = 0.5 * vorticity_contract
    
    return {
        'expansion': expansion_scalar,
        'shear': shear_scalar,
        'vorticity': vorticity_scalar
    }
