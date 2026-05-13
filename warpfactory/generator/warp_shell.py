import numpy as np
from warpfactory.generator.base import Metric
from warpfactory.generator.commons import create_grid
from warpfactory.constants import C, G
from warpfactory.utils.helpers import (
    legendre_radial_interp,
    tov_const_density,
    alpha_numeric_solver,
    compact_sigmoid,
    sph2cart_diag
)


def _matlab_moving_smooth(values, span):
    """Approximate MATLAB smooth(y, span) with the default moving average."""
    span = int(span)
    if span < 2:
        return values.copy()

    values = np.asarray(values, dtype=float)
    n_values = values.size
    if n_values == 0:
        return values.copy()

    if span > n_values:
        span = n_values
    if span % 2 == 0 and span > 1:
        span -= 1
    if span < 2:
        return values.copy()

    half_width = span // 2
    cumulative = np.concatenate(([0.0], np.cumsum(values)))
    smoothed = np.empty_like(values, dtype=float)

    for index in range(n_values):
        start = max(0, index - half_width)
        stop = min(n_values, index + half_width + 1)
        smoothed[index] = (cumulative[stop] - cumulative[start]) / (stop - start)

    return smoothed


def _repeat_matlab_smooth(values, span, passes):
    smoothed = np.asarray(values, dtype=float)
    for _ in range(passes):
        smoothed = _matlab_moving_smooth(smoothed, span)
    return smoothed


def create_warp_shell_metric(grid_size, grid_scale, world_center, 
                             mass, r_inner, r_outer, r_buff, sigma, smooth_factor, 
                             v_warp=0.0, do_warp=False, adm_shift_g00=False,
                             density_smooth_ratio=1.79):
    """
    Generates the Warp Shell metric in a comoving frame.
    
    Args:
        grid_size (tuple): (Nt, Nx, Ny, Nz)
        grid_scale (tuple): (dt, dx, dy, dz)
        world_center (tuple): (tc, xc, yc, zc)
        mass (float): Total mass of shell.
        r_inner (float): Inner radius.
        r_outer (float): Outer radius.
        r_buff (float): Buffer distance.
        sigma (float): Sigmoid sharpness.
        smooth_factor (int): Smoothing iterations.
        v_warp (float): Warp velocity (fraction of c?). MATLAB says "factors of c".
        do_warp (bool): Enable warp effect.
        adm_shift_g00 (bool): If True, include beta_i beta^i in g_tt so the
            intended static lapse alpha remains the ADM lapse after adding
            shift. Defaults to False for WarpFactory MATLAB parity.
        density_smooth_ratio (float): Ratio between density and pressure moving
            average spans. WarpFactory's public MATLAB example uses 1.79, while
            Fuchs et al. report approximately 1.72 in the paper.
        
    Returns:
        Metric object.
    """
    Nt, Nx, Ny, Nz = grid_size
    
    # Create coordinate grids
    # We iterate manually as per MATLAB code to optimize/parallelize or just handle the complex logic per point.
    # But Python should vectorize if possible.
    # The interpolation logic is point-wise.
    
    # 1. 1D Radial Solve
    # Define r_sample. MATLAB uses the distance from worldCenter to the
    # final 1-based grid point, not the full domain diagonal from the origin.
    extent = np.sqrt(
        (grid_scale[1] * Nx - world_center[1]) ** 2
        + (grid_scale[2] * Ny - world_center[2]) ** 2
        + (grid_scale[3] * Nz - world_center[3]) ** 2
    )
    
    r_sample_res = 10000 # Reduced from 10^5 for performance in Python? Or keep high?
    # Python arrays are efficient. 10^5 is fine.
    r_sample_res = 100_000
    r_sample = np.linspace(0, extent * 1.2, r_sample_res)
    
    # Rho profile
    # rho = M / Volume * (R1 < r < R2)
    # Volume = 4/3 pi (R2^3 - R1^3)
    vol = (4.0/3.0) * np.pi * (r_outer**3 - r_inner**3)
    rho_val = mass / vol
    
    rho = np.zeros_like(r_sample)
    mask = (r_sample > r_inner) & (r_sample < r_outer)
    rho[mask] = rho_val
    
    # maxR logic (first point where rho becomes 0 after being positive?)
    # "min(diff(rho>0))". diff(rho>0) is -1 where it falls.
    # We just know it is R2.
    max_r = r_outer
    
    # Mass profile
    # cumtrapz(4 pi rho r^2)
    integrand = 4 * np.pi * rho * r_sample**2
    M_prof = np.concatenate(([0], np.cumsum(0.5 * (integrand[:-1] + integrand[1:]) * np.diff(r_sample))))
    
    # Pressure profile
    P_prof = tov_const_density(r_outer, M_prof, rho, r_sample)
    
    # MATLAB smooth(y, span) defaults to a moving average with truncated
    # endpoint windows. Zero-padded convolution changes the shell edges and can
    # dominate the energy-condition balance for large smoothFactor values.
    if smooth_factor > 0:
        span = int(density_smooth_ratio * smooth_factor)
        if span > 1:
            rho = _repeat_matlab_smooth(rho, span, 4)
            P_prof = _repeat_matlab_smooth(P_prof, smooth_factor, 4)
                
    # Re-integrate Mass
    integrand = 4 * np.pi * rho * r_sample**2
    M_prof = np.concatenate(([0], np.cumsum(0.5 * (integrand[:-1] + integrand[1:]) * np.diff(r_sample))))
    
    # Fix negative M at end?
    # M(M<0) = max(M)
    M_prof[M_prof < 0] = np.max(M_prof)
    
    # Shift Radial Vector
    # shift = compactSigmoid(...)
    shift_radial = compact_sigmoid(r_sample, r_inner, r_outer, sigma, r_buff)
    if smooth_factor > 0:
        shift_radial = _repeat_matlab_smooth(shift_radial, smooth_factor, 2)
        
    # Solve for B = (1 - 2GM/rc^2)^-1
    # Avoid div by zero.
    term = np.zeros_like(r_sample)
    np.divide(
        2 * G * M_prof,
        r_sample * C**2,
        out=term,
        where=r_sample != 0,
    )
    term[0] = 0 # r=0
    B_prof = 1.0 / (1.0 - term)
    B_prof[0] = 1.0
    
    # Solve for a (alpha = exp(a)?)
    # alphaNumericSolver returns 'a' (log alpha) or alpha?
    # MATLAB: "solve for a... alphaNumericSolver". then "A = -exp(2*a)".
    # alphaNumericSolver returns the log?
    # Helper code: "alpha = 1/2*log(...)". Yes, it computes ln(alpha).
    a_prof = alpha_numeric_solver(M_prof, P_prof, max_r, r_sample)
    
    # A = -exp(2a) -> -alpha^2
    A_prof = -np.exp(2 * a_prof)
    
    # Prepare Metric
    # Initialize arrays
    g_xx = np.zeros(grid_size)
    g_yy = np.zeros(grid_size)
    g_zz = np.zeros(grid_size)
    g_xy = np.zeros(grid_size)
    g_xz = np.zeros(grid_size)
    g_yz = np.zeros(grid_size)
    g_tt = np.zeros(grid_size) # Will be filled with A
    
    shift_grid = np.zeros(grid_size)
    
    # Iterate over grid points to interpolate
    # This is slow in Python loop. Vectorize!
    
    coords = create_grid(grid_size, grid_scale)
    T, X, Y, Z = coords['t'], coords['x'], coords['y'], coords['z']
    
    # Centers
    # MATLAB WarpFactory loops i=1..N and evaluates i*dx - center.
    # create_grid is 0-based, so add one grid spacing for parity.
    Xc = (X + grid_scale[1]) - world_center[1]
    Yc = (Y + grid_scale[2]) - world_center[2]
    Zc = (Z + grid_scale[3]) - world_center[3]
    
    R_grid = np.sqrt(Xc**2 + Yc**2 + Zc**2)
    # Avoid R=0 issues for angles
    R_grid_safe = R_grid + 1e-15
    
    Theta = np.arctan2(np.sqrt(Xc**2 + Yc**2), Zc)
    Phi = np.arctan2(Yc, Xc)
    
    # Interpolation
    # We need fractional index in r_sample corresponding to R_grid
    # r_sample is linspace(0, max, N).
    # idx = R / dr
    
    dr = r_sample[1] - r_sample[0]
    r_indices = R_grid / dr
    
    # Clamp indices
    r_indices = np.clip(r_indices, 0, len(r_sample)-1.001)
    
    # We need to map vectorized interp.
    # helpers.legendre_radial_interp takes scalar r.
    # We should vectorize implementation of interp or just use np.interp (linear) or cubic spline.
    # 3rd order Legendre interp is basically cubic interpolation.
    # np.interp is linear.
    # For high quality metric derivatives, smooth interp is crucial.
    # SciPy's CubicSpline would be best.
    # But to avoid scipy dep if not present? (WarpFactory requirements.txt was empty).
    # We can use simple map or vectorized custom interp.
    
    # Let's use map_coordinates from scipy.ndimage if available, or just implement vectorized Lagrange.
    # Or just use numpy's interp (linear) if resolution is high enough?
    # User might strict on "Translation".
    # Let's vectorize the detailed interpolation logic.
    
    def vec_interp(arr, idxs):
        # idxs is array of float indices
        x1 = np.floor(idxs).astype(int)
        x0 = x1 - 1
        x2 = x1 + 1
        x3 = x1 + 2
        
        N = len(arr)
        x0 = np.clip(x0, 0, N-1)
        x1 = np.clip(x1, 0, N-1)
        x2 = np.clip(x2, 0, N-1)
        x3 = np.clip(x3, 0, N-1)
        
        y0 = arr[x0]
        y1 = arr[x1]
        y2 = arr[x2]
        y3 = arr[x3]
        
        x = idxs
        xi0, xi1, xi2, xi3 = x1-1, x1, x1+1, x1+2 # Bases for interpolation
        # Actually bases should be x0,x1,x2,x3 integer values relative to 'x'.
        # The formula used constant denominators assuming unit spacing.
        # x is the float index. Nodes are integers.
        
        # x - x1 is diff from floor. let u = x - x1. (0 <= u < 1)
        u = x - x1
        
        # Nodes: -1, 0, 1, 2.
        # u is dist from 0.
        # L0 (node -1): (u-0)(u-1)(u-2) / (-1-0)(-1-1)(-1-2) = u(u-1)(u-2) / -6
        # L1 (node 0): (u-(-1))(u-1)(u-2) / (0--1)(0-1)(0-2) = (u+1)(u-1)(u-2) / 2
        # L2 (node 1): (u+1)(u)(u-2) / (1--1)(1-0)(1-2) = (u+1)u(u-2) / -2
        # L3 (node 2): (u+1)(u)(u-1) / (2--1)(2-0)(2-1) = (u+1)u(u-1) / 6
        
        l0 = u * (u-1) * (u-2) / -6.0
        l1 = (u+1) * (u-1) * (u-2) / 2.0
        l2 = (u+1) * u * (u-2) / -2.0
        l3 = (u+1) * u * (u-1) / 6.0
        
        return y0*l0 + y1*l1 + y2*l2 + y3*l3
        
    g11_sph = vec_interp(A_prof, r_indices) # A
    g22_sph = vec_interp(B_prof, r_indices) # B
    shift_val = vec_interp(shift_radial, r_indices)
    
    # sph2cartDiag vectorized
    # It just used simple formulas with arrays.
    res = sph2cart_diag(Theta, Phi, g11_sph, g22_sph)
    # Returns 7 arrays
    
    g11_c, g22_c, g23_c, g24_c, g33_c, g34_c, g44_c = res
    
    # Map to tensor
    # 0 based: 0=t, 1=x, 2=y, 3=z
    tensor = np.zeros((4, 4, Nt, Nx, Ny, Nz))
    
    tensor[0, 0] = g11_c # tt
    
    tensor[1, 1] = g22_c # xx
    tensor[2, 2] = g33_c # yy
    tensor[3, 3] = g44_c # zz
    
    tensor[1, 2] = g23_c; tensor[2, 1] = g23_c # xy
    tensor[1, 3] = g24_c; tensor[3, 1] = g24_c # xz
    tensor[2, 3] = g34_c; tensor[3, 2] = g34_c # yz
    
    # ShiftMatrix logic
    # ShiftMatrix = shift_val.
    # Warp effect:
    # "Metric.tensor{1,2} = Metric.tensor{1,2}-Metric.tensor{1,2}.*ShiftMatrix - ShiftMatrix*vWarp"
    # tensor{1,2} is tx.
    # Originally tensor[0, 1] is 0 (setMinkowski init).
    # here we didn't init with -1, but A is -alpha^2 (~ -1).
    # Wait, in the loop:
    # Metric.tensor{1,2} is NOT set. It is 0.
    # So `Metric.tensor{1,2} .* ShiftMatrix` is 0.
    # Result: - ShiftMatrix * vWarp.
    
    if do_warp:
        # tx component
        tx = -shift_val * v_warp
        tensor[0, 1] = tx
        tensor[1, 0] = tx
        if adm_shift_g00:
            gamma = tensor[1:, 1:]
            gamma_T = np.moveaxis(gamma, [0, 1], [-2, -1])
            gamma_up_T = np.linalg.inv(gamma_T)
            beta_down = np.stack((tx, np.zeros_like(tx), np.zeros_like(tx)), axis=-1)
            beta_up = np.einsum("...ij,...j->...i", gamma_up_T, beta_down)
            beta_sq = np.einsum("...i,...i->...", beta_down, beta_up)
            tensor[0, 0] = tensor[0, 0] + beta_sq
        
    # Create Metric object
    metric_obj = Metric(
        tensor=tensor,
        coords={
            't': (T + grid_scale[0]) - world_center[0],
            'x': Xc,
            'y': Yc,
            'z': Zc,
        },
        scaling=np.array(grid_scale),
        name="Comoving Warp Shell",
        index="covariant",
        params={
            'mass': mass,
            'rho': rho,
            'P': P_prof,
            'M': M_prof,
            'r_sample': r_sample,
            'A': A_prof,
            'B': B_prof,
            'shift_radial': shift_radial,
            'adm_shift_g00': adm_shift_g00,
            'density_smooth_ratio': density_smooth_ratio,
        }
    )
    
    return metric_obj
