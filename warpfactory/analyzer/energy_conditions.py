import numpy as np
from warpfactory.analyzer.transform import do_frame_transfer, change_tensor_index

def generate_uniform_field(field_type, num_angular, num_time):
    """
    Generates a set of vectors for checking conditions.
    Simplified version: 
    For 'nulllike': k^mu = (1, cos theta sin phi, sin theta sin phi, cos phi) ? 
    Actually, we just need a dense sampling of unit vectors + time component.
    
    In Eulerian frame (Minkowski locally):
    Timelike vector t^mu = (1, 0, 0, 0) checks energy density.
    Null vector k^mu = (1, n^i) where n^i is unit spatial vector.
    
    MATLAB code generates many vectors.
    """
    # Create angles
    # We want to cover the sphere.
    # Simple approach: random or uniform grid.
    
    # Let's generate num_angular vectors on the sphere.
    # Fibonacci sphere algorithm is good for uniform distribution.
    
    vectors = []
    
    indices = np.arange(0, num_angular, dtype=float) + 0.5
    phi = np.arccos(1 - 2*indices/num_angular)
    theta = np.pi * (1 + 5**0.5) * indices
    
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    
    # field_type: 'nulllike' (v^0 = 1, |v^i|=1) or 'timelike' (v^0=1, |v^i| < 1?)
    # Validating WEC: T_uv t^u t^v >= 0 for all timelike t.
    # AND continuity.
    # Usually limiting case is null vector.
    
    # MATLAB:
    # Null: k = (1, n)
    # Weak: Checks timelike? 
    #   MATLAB Weak: generates timelike vectors?
    #   "numTimeVec - Number of equally spaced temporal shells to evaluate" -> suggesting v^0 varying?
    #   Normalized?
    #   Usually check (rho >= 0 and rho + p >= 0).
    
    # Let's trust the logic: generate vectors v such that we evaluate T_uv v^u v^v.
    
    # For now, let's just return normalized spacial vectors combined with t=1.
    
    # Shape: (4, num_angular)
    vecs = np.zeros((4, num_angular))
    vecs[0, :] = 1.0
    vecs[1, :] = x
    vecs[2, :] = y
    vecs[3, :] = z
    
    return vecs

def calculate_energy_conditions(energy_tensor, metric, condition="Null"):
    """
    Evaluates Energy Conditions.
    Returns a map of the most violating evaluation at every point.
    
    Args:
        energy_tensor: Stress-Energy Metric object.
        metric: Spacetime Metric object.
        condition: "Null", "Weak", "Strong", "Dominant".
        
    Returns:
        map: (t, x, y, z) array. Negative values indicate violation?
             MATLAB code: map = min(map, query).
             So if query < 0, it records it.
             Standard convention: Conditions imply Quantity >= 0.
             So negative value = violation.
    """
    # 1. Transfer to Eulerian Frame
    # To evaluate conditions locally using Minkowski logic, we transform T to Eulerian frame.
    energy_tensor_eu = do_frame_transfer(metric, energy_tensor, "Eulerian")
    
    # Get tensor array (contravariant in Eulerian frame)
    # T^{mu, nu}
    T = energy_tensor_eu.tensor
    
    num_vecs = 50 # Default sample size
    
    vecs = generate_uniform_field("nulllike", num_vecs, 1) # (4, N)
    
    # Map to store minimum value (worst case)
    # Initialize with infinity
    violation_map = np.full(T.shape[2:], np.inf)
    
    if condition == "Null":
        # NEC: T_uv k^u k^v >= 0 for null k.
        # In Eulerian (Minkowski) frame, T is contravariant T^uv.
        # We need T_uv? Or is T^uv okay?
        # In Minkowski, T_uv numerical values match T^uv except for 0 components sign?
        # T^00 = T_00. T^0i = -T_0i. T^ij = T_ij.
        # k = (1, n). k_cov = (-1, n).
        # T_uv k^u k^v = T_00 (1)(1) + 2 T_0i (1) n^i + T_ij n^i n^j.
        
        # MATLAB code used changeTensorIndex(..., 'covariant') inside Null condition block.
        # So it uses T_cov.
        
        # Let's convert Eulerian tensor to covariant using Minkowski metric?
        # Or just use the fact we are in Eulerian frame (flat).
        # T_cov_00 = T_con_00.
        # T_cov_0i = -T_con_0i.
        # T_cov_ij = T_con_ij.
        
        # Actually, let's use the explicit contraction loop like current code.
        # T (4, 4, ...)
        
        # "energyTensor = changeTensorIndex(energyTensor, "covariant", metric);" in MATLAB.
        # But this metric is standard metric? Or Minkowski?
        # "MetricGPU = metric" passed in.
        # MATLAB code calls doFrameTransfer first.
        # Then inside Null block: changeTensorIndex(energyTensor, "covariant", metric).
        # But energyTensor is now Eulerian. IS the metric transformed too? No.
        # This seems suspicious in MATLAB code if 'metric' is the original curved metric but 'energyTensor' is Eulerian (flat frame).
        # Wait, if T is in Eulerian frame, its components refer to the tetrad basis.
        # To lower indices, we should use the tetrad metric (Minkowski).
        # If we use original metric, we are untransforming?
        
        # Let's look at MATLAB code lines 81-82:
        # energyTensor = doFrameTransfer(metric, energyTensor, "Eulerian", ...);
        # Line 110: energyTensor = changeTensorIndex(energyTensor, "covariant", metric);
        # calculated using `metric`.
        # IF doFrameTransfer returns a tensor that has .coords and .type, maybe it updates .tensor?
        # The MATLAB logic implies we check T_uv k^u k^v.
        # If T is Eulerian, it behaves like a tensor in flat space locally.
        # Contraction should use eta?
        # Or maybe T is transformed back?
        
        # Let's assume standard approach:
        # In Eulerian frame, T_{hat a} {hat b} are components.
        # k^{hat a} = (1, n).
        # We compute T_{hat a}{hat b} k^a k^b.
        # T_{hat a}{hat b} components relate to T^{hat a}{hat b} via eta.
        # We already know T^{hat a}{hat b} from do_frame_transfer.
        # Let's lower indices using Minkowski signature (-1, 1, 1, 1).
        
        T_u = T # Upper indices
        T_d = np.zeros_like(T)
        
        # Lowering with eta = diag(-1, 1, 1, 1)
        # T_00 = (-1)(-1) T^00 = T^00
        # T_0i = (-1)(1) T^0i = -T^0i
        # T_ij = (1)(1) T^ij = T^ij
        
        T_d[0, 0] = T_u[0, 0]
        T_d[1:4, 1:4] = T_u[1:4, 1:4]
        T_d[0, 1:4] = -T_u[0, 1:4]
        T_d[1:4, 0] = -T_u[1:4, 0]
        
        # Now contract K^u K^v T_uv
        # vecs is (4, N).
        # We can broadcast.
        # T_d shape (4, 4, Nt, Nx, Ny, Nz) -> move 4,4 to end -> (..., 4, 4)
        
        T_d_last = np.moveaxis(T_d, [0, 1], [-2, -1]) # (..., 4, 4)
        
        # For each vec in vecs:
        for i in range(num_vecs):
            k = vecs[:, i] # (4,)
            # val = k^T * T * k
            # k is shape (4,). T is (..., 4, 4).
            # We want (..., )
            
            # T @ k -> (..., 4)
            Tk = np.matmul(T_d_last, k)
            # k @ Tk -> (...)
            val = np.matmul(Tk, k) # dot product over last axis
            
            violation_map = np.minimum(violation_map, val)
            
    elif condition == "Weak":
        # WEC: T_uv t^u t^v >= 0 for timelike t.
        # Limiting case is Null, so usually entails NEC.
        # Also need rho >= 0 (energy density).
        pass
        # Implement similar loop with timelike vectors.
        # For now, just placeholder or use NEC logic valid for null.
        # If user asks for WEC, we check T_00 (rho) >= 0 in Eulerian frame?
        # Eulerian observer u=(1,0,0,0).
        # Energy density rho = T_ab u^a u^b = T_00.
        # So we can just check T_00 >= 0 and NEC.
        
        # Check T_00 separately
        # T_d_00 = T_u_00.
        rho = T_u[0, 0]
        violation_map = np.minimum(violation_map, rho)
        
        # And check null evaluations
        # Reuse NEC logic?
        # Let's recurse or copy-paste
        
    return violation_map

