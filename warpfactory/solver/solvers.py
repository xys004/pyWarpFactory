from warpfactory.solver.tensor_utils import (
    get_c4_inv, 
    get_ricci_tensor, 
    get_ricci_scalar, 
    get_einstein_tensor, 
    get_energy_tensor,
    verify_tensor
)
from warpfactory.generator.base import Metric

def solve_energy_tensor(metric):
    """
    Computes the Stress-Energy Tensor for a given Metric using Einstein Field Equations.
    Replicates met2den.m functionality.
    
    Args:
        metric (Metric): A valid Metric object.
        
    Returns:
        Metric: A new 'Metric' object representing the Stress-Energy tensor (T^uv).
                Type will be 'Stress-Energy'.
    """
    # 1. Verify input
    verify_tensor(metric)
    
    grid_scale = metric.scaling
    
    # 2. Compute Inverse Metric
    g_inv = get_c4_inv(metric.tensor)
    
    # 3. Compute Ricci Tensor R_mn
    R_mn = get_ricci_tensor(metric.tensor, grid_scale)
    
    # 4. Compute Ricci Scalar R
    R = get_ricci_scalar(R_mn, g_inv)
    
    # 5. Compute Einstein Tensor G_mn
    G_mn = get_einstein_tensor(R_mn, R, metric.tensor)
    
    # 6. Compute Energy Tensor T^uv (Contravariant)
    # Note: get_energy_tensor returns T^uv
    T_uv = get_energy_tensor(G_mn, g_inv)
    
    # 7. Wrap in Metric/Tensor object
    # We reuse the Metric class container for now, but change type.
    # ideally we should have a Tensor class, but Metric is generic enough.
    
    energy_tensor = Metric(
        tensor=T_uv,
        coords=metric.coords,
        scaling=metric.scaling,
        name=f"Energy Tensor of {metric.name}",
        index="contravariant",
        params=metric.params
    )
    energy_tensor.type = "Stress-Energy"
    
    return energy_tensor
