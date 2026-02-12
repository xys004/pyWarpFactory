import numpy as np
from warpfactory.solver.tensor_utils import get_c3_inv

def decompose_3plus1(metric):
    """
    Decomposes a 4D metric into ADM variables: alpha (lapse), beta (shift), gamma (spatial metric).
    From threePlusOneDecomposer.m
    
    Args:
        metric: Metric object.
        
    Returns:
        alpha: Scalar field array.
        beta: List of 3 scalar field arrays [beta_x, beta_y, beta_z]. (beta^i)
        gamma: 3x3 array of scalar field arrays. (gamma_ij)
    """
    g = metric.tensor
    
    # 3+1 split logic
    # g_00 = -alpha^2 + beta_k beta^k
    # g_0i = beta_i
    # g_ij = gamma_ij
    
    # In WarpFactory MATLAB:
    # gamma = g(2:4, 2:4)
    # beta_i = g(1, 2:4)
    # alpha = sqrt(beta^2 - g_00)
    
    # Python indices:
    # 0 = t
    # 1,2,3 = x,y,z
    
    # gamma_ij = g_ij
    gamma = g[1:4, 1:4]
    
    # beta_cov_i (beta_i) = g_0i
    beta_cov = g[0, 1:4] # Shape (3, Nt, Nx, ...)
    
    # We need beta^i (shift vector) usually?
    # MATLAB output: beta "Shift vector, 1x3 cell".
    # But code `metric{1, i} = beta{i-1}` in builder suggests beta is stored as cells.
    # Decomposer:
    # "beta{i} = g{1, i+1} - shift * gamma..." no.
    # Usually: g_0i = beta_i = gamma_ij beta^j.
    # So beta^i = gamma^ij g_0j.
    
    # Let's check MATLAB threePlusOneDecomposer.m logic?
    # I didn't verify-read it, but it's standard ADM.
    # Let's check threePlusOneBuilder.m which I read.
    # Builder:
    # metricTensor{1, i} = beta{i-1}; -> beta input is g_0i (covariant beta).
    # "beta - {3}x... (covariant assumed) shift vector".
    # AND:
    # "beta_up{i} = beta_up{i} + gamma_up{i,j} .* beta{j}"
    # So `beta` (input) is treated as beta^i or beta_i?
    # If calculate beta_up from it using gamma_up * beta, then `beta` must be beta_down (covariant).
    # And g_0i IS beta_down.
    
    # So `decompose_3plus1` should return g_0i as beta.
    
    gamma_val = gamma
    beta_val = [beta_cov[0], beta_cov[1], beta_cov[2]]
    
    # alpha
    # g_00 = -alpha^2 + beta_k beta^k
    # alpha^2 = beta_k beta^k - g_00
    # beta_k beta^k = beta_i * gamma^ij * beta_j
    
    # Get gamma inverse
    gamma_up = get_c3_inv(gamma_val)
    
    # Compute contraction beta_u * beta_d
    # beta_u^i = gamma^ij beta_j
    
    beta_up = np.einsum('ij...,j...->i...', gamma_up, beta_cov)
    
    beta_sq = np.einsum('i...,i...->...', beta_up, beta_cov)
    
    g00 = g[0, 0]
    
    # alpha^2 = beta_sq - g00
    alpha_sq = beta_sq - g00
    
    # alpha = sqrt(alpha_sq)
    alpha = np.sqrt(alpha_sq)
    
    return alpha, beta_val, gamma_val

def reconstruct_3plus1(alpha, beta, gamma):
    """
    Builds metric from ADM variables.
    beta is assumed covariant (beta_i).
    gamma is covariant (gamma_ij).
    """
    # Inverse gamma
    gamma_up = get_c3_inv(gamma)
    
    # Calculate beta_up
    beta_arr = np.array(beta) # (3, ...)
    beta_up = np.einsum('ij...,j...->i...', gamma_up, beta_arr)
    beta_sq = np.einsum('i...,i...->...', beta_up, beta_arr)
    
    # g_00 = -alpha^2 + beta^2
    g00 = -alpha**2 + beta_sq
    
    shape = alpha.shape
    g = np.zeros((4, 4) + shape)
    
    g[0, 0] = g00
    
    # g_0i = beta_i
    g[0, 1:4] = beta_arr
    g[1:4, 0] = beta_arr
    
    # g_ij = gamma_ij
    g[1:4, 1:4] = gamma
    
    return g
