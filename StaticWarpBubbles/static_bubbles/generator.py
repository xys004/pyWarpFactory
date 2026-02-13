import numpy as np
from scipy.interpolate import interp1d
from warpfactory.generator.base import Metric
from warpfactory.generator.commons import create_grid
from warpfactory.constants import C, G

def create_static_bubble_metric(grid_size, grid_scale, world_center, 
                                rho_profile=None, beta_profile=None, 
                                r0=None, integration_constant=8*np.pi):
    """
    Generates the Static Spherically-Symmetric Warp Bubble metric (2025).
    Ref: Bolívar, Abellán, Vasilev, Annals of Physics 481 (2025) 170147.
    
    Metric:
    ds^2 = -dt^2 + (dr - beta(r) dt)^2 + r^2 dOmega^2
    Lapse alpha = 1.
    Shift vector is purely radial: beta^i = beta(r) * x^i / r.
    
    Args:
        grid_size (tuple): (Nt, Nx, Ny, Nz)
        grid_scale (tuple): (dt, dx, dy, dz)
        world_center (tuple): (tc, xc, yc, zc)
        rho_profile (callable or tuple): 
            - If callable: rho(r).
            - If tuple: (rho_minus_func, rho_plus_func) for piecewise at r0.
        beta_profile (callable): Optional direct beta(r) function. Overrides rho_profile.
        r0 (float): Junction radius for piecewise rho.
        integration_constant (float): Factor k in beta^2 = k/r^2 * int(rho s^2). 
                                      Default 8*pi (Geometric units).
        
    Returns:
        Metric object.
    """
    Nt, Nx, Ny, Nz = grid_size
    
    # 1. Create Grid
    # We want physical coordinates centered at world_center
    grid = create_grid(grid_size, grid_scale)
    t, x, y, z = grid['t'], grid['x'], grid['y'], grid['z']
    
    t_phys = (t + grid_scale[0]) - world_center[0]
    x_phys = (x + grid_scale[1]) - world_center[1] # Should we shift by dx/2 for parity? Using direct center sub.
    y_phys = (y + grid_scale[2]) - world_center[2]
    z_phys = (z + grid_scale[3]) - world_center[3]
    
    # Radial coordinate r
    # Since it's static, r depends only on x,y,z
    # But for broadcasting, we keep (Nt, Nx, Ny, Nz) shape or broadcast later.
    r_grid = np.sqrt(x_phys**2 + y_phys**2 + z_phys**2)
    
    # 2. Determine Beta Profile
    # We need to evaluate beta(r) on the grid.
    # To avoid re-integrating for every point, we integrate on a 1D sample array and interpolate.
    
    max_r = np.max(r_grid) * 1.05
    r_samples = np.linspace(0, max_r, 5000) # Grid for 1D integration
    
    beta_vals = np.zeros_like(r_samples)
    
    if beta_profile:
        # direct evaluation
        beta_vals = beta_profile(r_samples)
        
    elif rho_profile:
        # Integrate rho
        if isinstance(rho_profile, tuple):
            # Piecewise
            if r0 is None:
                raise ValueError("r0 must be provided for piecewise rho_profile")
            rho_inner_func, rho_outer_func = rho_profile
            
            # Mask for regions
            mask_in = r_samples < r0
            mask_out = r_samples >= r0
            
            # Evaluate rho
            rho_vals = np.zeros_like(r_samples)
            rho_vals[mask_in] = rho_inner_func(r_samples[mask_in])
            rho_vals[mask_out] = rho_outer_func(r_samples[mask_out])
            
            # Integrate I(r) = int_0^r rho(s) s^2 ds
            integrand = rho_vals * r_samples**2
            
            # We can use cumulative trapezoid
            integral = np.zeros_like(r_samples)
            integral[1:] = np.cumsum(0.5 * (integrand[:-1] + integrand[1:]) * np.diff(r_samples))
            
            # Compute beta^2
            # beta^2 = k / r^2 * integral + C / r^2
            # For purely static bubble with no horizon:
            # Usually we want beta regular at origin -> C=0 for r->0 branch.
            # And at junction? Usually continuous beta.
            # The simple integral accumulates mass.
            # continuity is handled by the integral accumulating across r0 naturally.
            
            k = integration_constant
            beta_sq = np.zeros_like(r_samples)
            
            # Avoid div by zero at r=0
            # limit r->0 of (1/r^2 * int_0^r rho s^2) = 1/r^2 * (rho(0) * r^3 / 3) = rho(0) * r / 3 -> 0.
            # So beta(0) = 0.
            
            with np.errstate(divide='ignore', invalid='ignore'):
                beta_sq[1:] = (k / r_samples[1:]**2) * integral[1:]
                
            beta_vals = np.sqrt(np.maximum(beta_sq, 0))
            
            # Sign of beta? 
            # "dr - beta dt". beta>0 usually means inflow?
            # Paper convention: beta(r) can be negative. 
            # Usually beta(r) corresponds to velocity.
            # We define beta = sqrt(...). This assumes beta > 0 (inward or outward depending on definition).
            # Paper Eq 20 solves for beta^2. So sign is chosen.
            # "We can choose beta(r) > 0".
            
        else:
            # Single profile
            rho_func = rho_profile
            rho_vals = rho_func(r_samples)
            integrand = rho_vals * r_samples**2
            integral = np.zeros_like(r_samples)
            integral[1:] = np.cumsum(0.5 * (integrand[:-1] + integrand[1:]) * np.diff(r_samples))
            
            k = integration_constant
            beta_sq = np.zeros_like(r_samples)
            with np.errstate(divide='ignore', invalid='ignore'):
                beta_sq[1:] = (k / r_samples[1:]**2) * integral[1:]
            
            beta_vals = np.sqrt(np.maximum(beta_sq, 0))
            
    else:
        # Default zero
        pass
        
    # Check |beta| < 1
    if np.any(np.abs(beta_vals) >= 1.0):
        print("WARNING: Beta >= 1 detected. Horizon or singularity present.")
        
    # Interpolate to 3D grid
    # Use linear (or cubic) interpolation
    f_interp = interp1d(r_samples, beta_vals, kind='linear', bounds_error=False, fill_value=(0, 0))
    
    beta_r_grid = f_interp(r_grid) # Scalar field beta(r)
    
    # 3. Construct Vector Shift
    # beta^i = beta(r) * x^i / r
    # Handle r=0: if r=0 beta=0.
    
    # Unit vector n^i = x^i / r
    n_x = np.zeros_like(x_phys)
    n_y = np.zeros_like(y_phys)
    n_z = np.zeros_like(z_phys)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        n_x = x_phys / r_grid
        n_y = y_phys / r_grid
        n_z = z_phys / r_grid
        
    n_x[r_grid == 0] = 0
    n_y[r_grid == 0] = 0
    n_z[r_grid == 0] = 0
    
    beta_x = beta_r_grid * n_x
    beta_y = beta_r_grid * n_y
    beta_z = beta_r_grid * n_z
    
    # 4. Build Metric Tensor
    # ADM form: ds^2 = -(alpha^2 - beta^2) dt^2 + 2 beta_i dx^i dt + gamma_ij dx^i dx^j
    # alpha = 1
    # gamma_ij = delta_ij
    # beta_i = beta^i (since gamma is flat)
    
    # g_00 = -1 + beta^2
    # g_0i = beta_i
    # g_ij = delta_ij
    
    beta_sq_grid = beta_x**2 + beta_y**2 + beta_z**2 
    # This should equal beta_r_grid**2
    
    g = np.zeros((4, 4, Nt, Nx, Ny, Nz))
    
    # g00
    g[0, 0] = -1.0 + beta_sq_grid
    
    # g0i / gi0
    g[0, 1] = beta_x; g[1, 0] = beta_x
    g[0, 2] = beta_y; g[2, 0] = beta_y
    g[0, 3] = beta_z; g[3, 0] = beta_z
    
    # gij
    g[1, 1] = 1.0
    g[2, 2] = 1.0
    g[3, 3] = 1.0
    
    return Metric(
        tensor=g,
        coords={'t': t_phys, 'x': x_phys, 'y': y_phys, 'z': z_phys},
        scaling=np.array(grid_scale),
        name="Static Bubble",
        index="covariant",
        params={
            'beta_r': beta_vals,
            'r_samples': r_samples,
            'world_center': world_center
        }
    )
