import numpy as np
from warpfactory.solver.tensor_utils import get_c4_inv

def get_eulerian_transformation_matrix(g):
    """
    Returns the Cholesky transformation matrix for a given metric.
    Computes M such that M^T * g * M = eta (Minkowski).
    
    Args:
        g: Metric tensor array of shape (4, 4, ...). 
           INDICES MUST BE (mu, nu, ...).
           
    Returns:
        M: Transformation matrix of shape (4, 4, ...).
    """
    # Based on explicit Cholesky decomposition from WarpFactory MATLAB code.
    # The MATLAB indices are 1-based (1..4). Python are 0-based (0..3).
    # MATLAB g(4,4) -> Python g[3,3]
    # The formulas are recursive.
    
    # Check shape
    shape = g.shape
    M = np.zeros(shape)
    
    # Extract components for readability and vectorization
    # Assuming g is symmetric
    g00 = g[0,0]
    g01 = g[0,1]; g02 = g[0,2]; g03 = g[0,3]
    g11 = g[1,1]
    g12 = g[1,2]; g13 = g[1,3]
    g22 = g[2,2]
    g23 = g[2,3]
    g33 = g[3,3]
    
    # In MATLAB code:
    # 4 is index 4 (z?)
    # 3 is index 3 (y?)
    # ...
    # Wait, the MATLAB code uses 1..4.
    # Usually 1=t, 2=x, 3=y, 4=z or 1=x, 2=y, 3=z, 4=t?
    # In base WarpFactory:
    # "The tensor indices are defined as 0: t, 1: x, 2: y, 3: z" for Python port compatibility?
    # Let's check commons.py or list_dir strings.
    # In MATLAB: usually 1=t, 2=x, 3=y, 4=z because of 3+1 split 0-index.
    # But let's look at the formula structure.
    # g(4,4) is used as the first factor divisor.
    # If the signature is (-1, 1, 1, 1), g(4,4) (z) is positive. g(1,1) (t) is negative.
    # The algorithm seems to start from the end (4,4) which suggests a reverse Cholesky or LDL decomposition.
    # Explicit mapping:
    # MATLAB 1 -> Python 0
    # MATLAB 2 -> Python 1
    # MATLAB 3 -> Python 2
    # MATLAB 4 -> Python 3
    
    g_11 = g00 # g(1,1)
    g_12 = g01; g_13 = g02; g_14 = g03
    g_22 = g11; g_23 = g12; g_24 = g13
    g_33 = g22; g_34 = g23
    g_44 = g33 # g(4,4)
    
    # Note: MATLAB code uses `.` element-wise ops. Arrays are vectorized.
    
    # factor0 = g(4,4);
    factor0 = g_44
    
    # factor1 = (- g(3,4).^2 + g(3,3).*factor0);
    factor1 = -g_34**2 + g_33 * factor0
    
    # factor2 = (2.*g(2,3).*g(2,4).*g(3,4) - g(4,4).*g(2,3).^2 - g(3,3).*g(2,4).^2 + g(2,2).*factor1);
    factor2 = (2 * g_23 * g_24 * g_34
               - g_44 * g_23**2
               - g_33 * g_24**2
               + g_22 * factor1)
               
    # factor3 huge expression
    # factor3 = (- 2.*g(4,4).*g(1,2).*g(1,3).*g(2,3) + 2.*g(1,3).*g(1,4).*g(2,3).*g(2,4) + ...
    # ... - g(1,1).*factor2);
    
    term_huge = (- 2 * g_44 * g_12 * g_13 * g_23
                 + 2 * g_13 * g_14 * g_23 * g_24
                 + 2 * g_12 * g_13 * g_24 * g_34
                 + 2 * g_12 * g_14 * g_23 * g_34
                 - g_12**2 * g_34**2
                 - g_13**2 * g_24**2
                 - g_14**2 * g_23**2
                 + g_33 * (-2 * g_12 * g_14 * g_24 + g_44 * g_12**2)
                 + g_22 * (-2 * g_13 * g_14 * g_34 + g_44 * g_13**2 + g_33 * g_14**2))
                 
    factor3 = term_huge - g_11 * factor2
    
    # We use np.sqrt, but need to be careful with negative values if metric is violating things?
    # Complex support?
    # MATLAB code: `if ~isreal(M) warning...`
    # We assume valid for now.
    
    # M(4,4) = (1./factor0).^(1./2);
    M[3,3] = np.sqrt(1.0 / factor0)
    
    # M(3,3) = (factor0./factor1).^(1./2);
    M[2,2] = np.sqrt(factor0 / factor1)
    
    # M(4,3) = -g(3,4)./(factor0.*factor1).^(1./2);
    M[3,2] = -g_34 / np.sqrt(factor0 * factor1)
    
    # M(2,2) = (factor1./factor2).^(1./2);
    M[1,1] = np.sqrt(factor1 / factor2)
    
    # M(3,2) = (g(2,4).*g(3,4) - g(2,3).*g(4,4))./(factor1.*factor2).^(1./2);
    M[2,1] = (g_24 * g_34 - g_23 * g_44) / np.sqrt(factor1 * factor2)
    
    # M(4,2) = (g(2,3).*g(3,4) - g(2,4).*g(3,3))./(factor1.*factor2).^(1./2);
    M[3,1] = (g_23 * g_34 - g_24 * g_33) / np.sqrt(factor1 * factor2)
    
    # M(1,1) = (factor2./factor3).^(1./2);
    M[0,0] = np.sqrt(factor2 / factor3)
    
    # M(2,1) = (g(1,2).*g(3,4).^2 + g(1,3).*g(2,3).*g(4,4) - g(1,3).*g(2,4).*g(3,4) - ...
    # ... - g(1,2).*g(3,3).*g(4,4))./(factor2.*factor3).^(1./2);
    
    num_21 = (g_12 * g_34**2 + g_13 * g_23 * g_44 
              - g_13 * g_24 * g_34 - g_14 * g_23 * g_34 
              + g_14 * g_24 * g_33 - g_12 * g_33 * g_44)
    M[1,0] = num_21 / np.sqrt(factor2 * factor3)
    
    # M(3,1) = (g(1,3).*g(2,4).^2 - g(1,4).*g(2,3).*g(2,4) + g(1,2).*g(2,3).*g(4,4) - ...
    # ... + g(1,4).*g(2,2).*g(3,4))./(factor2.*factor3).^(1./2);
    num_31 = (g_13 * g_24**2 - g_14 * g_23 * g_24 
              + g_12 * g_23 * g_44 - g_12 * g_24 * g_34 
              - g_13 * g_22 * g_44 + g_14 * g_22 * g_34)
    M[2,0] = num_31 / np.sqrt(factor2 * factor3)
    
    # M(4,1) = (g(1,4).*g(2,3).^2 - g(1,3).*g(2,3).*g(2,4) - g(1,2).*g(2,3).*g(3,4) + ...
    # ... - g(1,4).*g(2,2).*g(3,3))./(factor2.*factor3).^(1./2);
    num_41 = (g_14 * g_23**2 - g_13 * g_23 * g_24 
              - g_12 * g_23 * g_34 + g_12 * g_24 * g_33 
              + g_13 * g_22 * g_34 - g_14 * g_22 * g_33)
    M[3,0] = num_41 / np.sqrt(factor2 * factor3)
    
    return M

def change_tensor_index(tensor, new_index, metric_tensor):
    """
    Raises or lowers indices of a tensor.
    Currently only supports converting between 'covariant' and 'contravariant' for rank-2 tensors.
    """
    
    # helper to get attr
    def get_attr(obj, attr):
        if isinstance(obj, dict):
            return obj.get(attr)
        return getattr(obj, attr, None)
        
    current_index = get_attr(tensor, 'index')
    
    if current_index == new_index:
        return tensor
        
    metric_val = get_attr(metric_tensor, 'tensor')
    tensor_val = get_attr(tensor, 'tensor')
    
    # Get inverse metric
    metric_index = get_attr(metric_tensor, 'index')
    
    if metric_index == 'covariant':
        g_lower = metric_val
        g_raise = get_c4_inv(metric_val)
    elif metric_index == 'contravariant':
        g_raise = metric_val
        g_lower = get_c4_inv(metric_val)
    else:
        raise ValueError("Metric must be covariant or contravariant.")
        
    result_val = None
    
    if current_index == 'covariant' and new_index == 'contravariant':
        # Raise: T^uv = g^ua T_ab g^bv (using matrix mul convention: T_con = G_inv * T_cov * G_inv)
        # Assuming symmetric logic or standard contraction
        # T^uv = sum_a,b g^ua T_ab g^vb
        # Let's use einsum for clarity
        # tensor is (4, 4, ...)
        # g is (4, 4, ...)
        # We broadcast over ...
        
        # result[u, v] = g_raise[u, a] * tensor[a, b] * g_raise[v, b]
        #              = g_raise[u, a] * tensor[a, b] * g_raise[b, v] (since g symmetric)
        
        # Using matmul with moving axes:
        # Move spatial dimensions to batch? No, move 4x4 to end.
        
        # (4, 4, t, x, y, z) -> (t, x, y, z, 4, 4)
        g_T = np.moveaxis(g_raise, [0, 1], [-2, -1])
        t_T = np.moveaxis(tensor_val, [0, 1], [-2, -1])
        
        # res = g * t * g
        res_T = np.matmul(np.matmul(g_T, t_T), g_T)
        
        result_val = np.moveaxis(res_T, [-2, -1], [0, 1])
        
    elif current_index == 'contravariant' and new_index == 'covariant':
        # Lower: T_uv = g_ua T^ab g_vb
        
        g_T = np.moveaxis(g_lower, [0, 1], [-2, -1])
        t_T = np.moveaxis(tensor_val, [0, 1], [-2, -1])
        
        res_T = np.matmul(np.matmul(g_T, t_T), g_T)
        result_val = np.moveaxis(res_T, [-2, -1], [0, 1])
        
    else:
        raise ValueError(f"Unsupported index transformation: {current_index} to {new_index}")
        
    # Return new object (copy)
    new_tensor = tensor.__class__(
        tensor=result_val,
        coords=tensor.coords,
        scaling=tensor.scaling,
        name=tensor.name,
        index=new_index,
        params=tensor.params
    )
    # If type exists, copy it
    if hasattr(tensor, 'type'):
        new_tensor.type = tensor.type
        
    return new_tensor

def do_frame_transfer(metric, energy_tensor, frame='Eulerian'):
    """
    Transforms the energy tensor into the selected frame.
    Only 'Eulerian' is supported.
    """
    if frame != 'Eulerian':
        raise ValueError("Only 'Eulerian' frame is supported.")
        
    if not (hasattr(energy_tensor, 'frame') and energy_tensor.frame == 'Eulerian'):
        # 1. Convert to covariant
        if energy_tensor.index != 'covariant':
            energy_tensor_cov = change_tensor_index(energy_tensor, 'covariant', metric)
        else:
            energy_tensor_cov = energy_tensor
            
        # 2. Compute Transformation Matrix M
        M = get_eulerian_transformation_matrix(metric.tensor)
        
        # 3. Transform: T_eulerian = M^T * T_cov * M
        # The MATLAB code: transformedTempTensor = M' * T * M
        # M is (4, 4, ...)
        
        M_T = np.moveaxis(M, [0, 1], [-2, -1])
        # Transpose of M in the first two dimensions?
        # M_transpose = permute(M, [something]).
        # In MATLAB M' is conjugate transpose.
        # Here M is real, so transpose.
        # M_T[... , i, j] = M[..., j, i]
        # Actually I constructed M_T above as (..., 4, 4) from (4, 4, ...).
        # Wait, if I use matmul with (..., 4, 4), then A @ B is correct standard matrix mult.
        # We want M^T @ T @ M.
        # So we need M_T_matrix (transpose of the matrix M).
        
        M_matrix = np.moveaxis(M, [0, 1], [-2, -1])
        T_matrix = np.moveaxis(energy_tensor_cov.tensor, [0, 1], [-2, -1])
        
        # Matrix transpose of M_matrix (swap last two axes)
        M_matrix_T = np.swapaxes(M_matrix, -1, -2)
        
        # Res = M_T @ T @ M
        Res_matrix = np.matmul(np.matmul(M_matrix_T, T_matrix), M_matrix)
        
        result_tensor_val = np.moveaxis(Res_matrix, [-2, -1], [0, 1])
        
        # 4. Transform to contravariant T^{0,i} = -T_{0,i} logic from MATLAB?
        # MATLAB:
        # "Transform to contravariant T^{0, i} = -T_{0, i}"
        # It creates a new struct index="contravariant".
        # It explicitly flips signs of T(0, i) and T(i, 0). (indices 0, 1..3)
        
        # Why?
        # Eulerian frame usually means projecting onto the tetrad.
        # T_hat_ab = e_a^mu e_b^nu T_mu_nu.
        # If the tetrad is orthonormal (Minkowski metric in this bases), then to raise index, we use eta.
        # eta = diag(-1, 1, 1, 1).
        # T_hat^ab = eta^ac eta^bd T_hat_cd.
        # T_hat^00 = (-1)(-1) T_hat_00 = T_hat_00.
        # T_hat^0i = (-1)(1) T_hat_0i = -T_hat_0i.
        # T_hat^ij = (1)(1) T_hat_ij = T_hat_ij.
        # So indeed, mixed time-space components flip sign.
        
        # Let's apply this sign flip.
        res_copy = result_tensor_val.copy()
        for i in range(1, 4):
            res_copy[0, i] = -res_copy[0, i]
            res_copy[i, 0] = -res_copy[i, 0]
            
        # Wrap result
        new_tensor = energy_tensor.__class__(
            tensor=res_copy,
            coords=energy_tensor.coords,
            scaling=energy_tensor.scaling,
            name=f"{energy_tensor.name} (Eulerian)",
            index="contravariant",
            params=energy_tensor.params
        )
        new_tensor.frame = "Eulerian"
        new_tensor.type = "Stress-Energy"
        
        return new_tensor
        
    return energy_tensor
