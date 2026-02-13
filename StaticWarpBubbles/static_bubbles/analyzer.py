import numpy as np
from static_bubbles.generator import create_static_bubble_metric

def analyze_static_bubble(rho_profile, r_grid, r0=None):
    """
    Computes analytic Energy Conditions for the static bubble.
    
    Relations:
    p_r = -rho
    p_perp = -rho - r/2 * rho'
    
    Conditions:
    NEC: rho + p_r >= 0 => 0 >= 0 (Trivial along radial?)
         rho + p_perp >= 0
    WEC: rho >= 0
    
    Args:
        rho_profile: callable or tuple (rho_in, rho_out)
        r_grid: 3D or 1D array of radial coordinates.
        r0: Junction radius needed if rho_profile is tuple.
        
    Returns:
        dict: {
            'rho': values,
            'p_perp': values,
            'NEC': values (rho + p_perp),
            'WEC': values (min(rho, rho+p_perp)),
            'DEC': values (rho - |p_perp|)
        }
    """
    
    # 1. Evaluate rho and rho' (derivative)
    # We can use finite difference on the r_grid if it's monotonic,
    # or better, evaluate derivative analytically if we knew it to avoid noise.
    # But usually we only have the profile function.
    # Let's use numerical derivative on a fine 1D line and interpolate, similar to generator.
    
    # Flatten r_grid to find range
    r_max = np.max(r_grid)
    r_line = np.linspace(0, r_max * 1.05, 5000)
    dr = r_line[1] - r_line[0]
    
    rho_vals = np.zeros_like(r_line)
    
    if isinstance(rho_profile, tuple):
        if r0 is None: raise ValueError("r0 needed for piecewise")
        func_in, func_out = rho_profile
        mask_in = r_line < r0
        mask_out = r_line >= r0
        rho_vals[mask_in] = func_in(r_line[mask_in])
        rho_vals[mask_out] = func_out(r_line[mask_out])
    else:
        rho_vals = rho_profile(r_line)
        
    # Derivative rho'
    rho_prime = np.gradient(rho_vals, dr)
    
    # Handle discontinuity at r0?
    # Derivative might spike. The physics analysis discusses Delta functions.
    # For numerical visualizer, the spike might be large.
    
    # Calculate p_perp on 1D line
    # p_perp = -rho - r/2 * rho'
    p_perp_vals = -rho_vals - (r_line / 2.0) * rho_prime
    
    # Interpolate to 3D grid
    from scipy.interpolate import interp1d
    
    f_rho = interp1d(r_line, rho_vals, kind='linear', fill_value=0, bounds_error=False)
    f_pperp = interp1d(r_line, p_perp_vals, kind='linear', fill_value=0, bounds_error=False)
    
    rho_3d = f_rho(r_grid)
    pperp_3d = f_pperp(r_grid)
    
    # Energy Conditions
    # Null: rho + p_i >= 0.
    # p_r = -rho => rho + p_r = 0 (Satisfied)
    # rho + p_perp >= 0.
    nec_val = rho_3d + pperp_3d
    
    # Weak: rho >= 0 AND rho + p_i >= 0
    wec_val = np.minimum(rho_3d, nec_val)
    
    # Dominant: rho >= |p_i|
    # rho >= |-rho| => rho >= rho (True if rho>=0)
    # rho >= |p_perp|
    dec_val = rho_3d - np.abs(pperp_3d)
    
    # Strong: rho + sum(p) >= 0 and rho + p_i >= 0
    # sum(p) = p_r + 2*p_perp = -rho + 2*p_perp
    # rho + sum = 2*p_perp >= 0?
    sec_val = 2 * pperp_3d
    # And we need NEC.
    sec_val = np.minimum(sec_val, nec_val)
    
    return {
        'rho': rho_3d,
        'p_perp': pperp_3d,
        'NEC': nec_val,
        'WEC': wec_val,
        'DEC': dec_val,
        'SEC': sec_val
    }
