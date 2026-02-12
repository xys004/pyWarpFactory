import numpy as np
from warpfactory.generator.base import Metric
from warpfactory.generator.commons import create_grid
from warpfactory.constants import C

def get_warp_factor_by_region(x_in, y_in, size_scale):
    """
    Computes the Lentz shift vector template values based on regions.
    Replicates getWarpFactorByRegion from MATLAB.
    
    Args:
        x_in (ndarray): x coordinates
        y_in (ndarray): y coordinates
        size_scale (float): sizing factor of the Lentz soliton template
        
    Returns:
        tuple: (WFX, WFY) arrays
    """
    x = x_in
    y = np.abs(y_in)
    
    # Initialize with default 0
    WFX = np.zeros_like(x)
    WFY = np.zeros_like(x)
    
    # Define masks for each region
    # Region 1: (x >= s && x <= 2s) && (x-s >= y)
    mask1 = (x >= size_scale) & (x <= 2*size_scale) & (x - size_scale >= y)
    WFX[mask1] = -2
    WFY[mask1] = 0
    
    # Region 2: (x > s && x <= 2s) && (x-s <= y) && (-y+3s >= x)
    mask2 = (x > size_scale) & (x <= 2*size_scale) & (x - size_scale <= y) & (-y + 3*size_scale >= x)
    WFX[mask2] = -1
    WFY[mask2] = 1
    
    # Region 3: (x > 0 && x <= s) && (x+s > y) && (-y+s < x)
    mask3 = (x > 0) & (x <= size_scale) & (x + size_scale > y) & (-y + size_scale < x)
    WFX[mask3] = 0
    WFY[mask3] = 1
    
    # Region 4: (x > 0 && x <= s) && (x+s <= y) && (-y+3s >= x)
    mask4 = (x > 0) & (x <= size_scale) & (x + size_scale <= y) & (-y + 3*size_scale >= x)
    WFX[mask4] = -0.5
    WFY[mask4] = 0.5
    
    # Region 5: (x > -s && x <= 0) && (-x+s < y) && (-y+3s >= -x)
    mask5 = (x > -size_scale) & (x <= 0) & (-x + size_scale < y) & (-y + 3*size_scale >= -x)
    WFX[mask5] = 0.5
    WFY[mask5] = 0.5
    
    # Region 6: (x > -s && x <= 0) && (x+s <= y) && (-y+s >= x)
    # Checks: (x+s <= y) -> y >= x+s. (-y+s >= x) -> s-x >= y.
    mask6 = (x > -size_scale) & (x <= 0) & (x + size_scale <= y) & (-y + size_scale >= x)
    WFX[mask6] = 1
    WFY[mask6] = 0
    
    # Region 7: (x >= -s && x <= s) && (x+s > y) -- Wait, last condition in MATLAB:
    # elseif (x >= -sizeScale && x <= sizeScale) && (x+sizeScale > y)
    # This might overlap with others? MATLAB evaluates sequentially (elseif).
    # so we should use ~mask_previous.
    # Actually, let's look at the logic structure.
    # The conditions seem geometrically distinct or prioritized by order.
    # To replicate "elseif" behavior with masks, we must apply them in order or ensure exclusivity.
    # Let's accumulate a "handled" mask.
    
    # Re-implementing with sequential handling to ensure "elseif" priority
    handled = np.zeros_like(x, dtype=bool)
    
    # Reset arrays to ensure clean slate (though assumed 0)
    WFX = np.zeros_like(x)
    WFY = np.zeros_like(x)
    
    # 1
    m1 = (x >= size_scale) & (x <= 2*size_scale) & (x - size_scale >= y)
    # Apply m1 where not handled
    m1 = m1 & (~handled)
    WFX[m1] = -2
    WFY[m1] = 0
    handled = handled | m1
    
    # 2
    m2 = (x > size_scale) & (x <= 2*size_scale) & (x - size_scale <= y) & (-y + 3*size_scale >= x)
    m2 = m2 & (~handled)
    WFX[m2] = -1
    WFY[m2] = 1
    handled = handled | m2
    
    # 3
    m3 = (x > 0) & (x <= size_scale) & (x + size_scale > y) & (-y + size_scale < x)
    m3 = m3 & (~handled)
    WFX[m3] = 0
    WFY[m3] = 1
    handled = handled | m3
    
    # 4
    m4 = (x > 0) & (x <= size_scale) & (x + size_scale <= y) & (-y + 3*size_scale >= x)
    m4 = m4 & (~handled)
    WFX[m4] = -0.5
    WFY[m4] = 0.5
    handled = handled | m4
    
    # 5
    m5 = (x > -size_scale) & (x <= 0) & (-x + size_scale < y) & (-y + 3*size_scale >= -x)
    m5 = m5 & (~handled)
    WFX[m5] = 0.5
    WFY[m5] = 0.5
    handled = handled | m5
    
    # 6
    m6 = (x > -size_scale) & (x <= 0) & (x + size_scale <= y) & (-y + size_scale >= x)
    m6 = m6 & (~handled)
    WFX[m6] = 1
    WFY[m6] = 0
    handled = handled | m6
    
    # 7
    m7 = (x >= -size_scale) & (x <= size_scale) & (x + size_scale > y)
    m7 = m7 & (~handled)
    WFX[m7] = 1
    WFY[m7] = 0
    handled = handled | m7
    
    # Final sign adjustment for Y
    WFY = np.sign(y_in) * WFY
    
    return WFX, WFY

def create_lentz_metric(grid_size, grid_scale, world_center, v, scale=None):
    """
    Generates the Lentz metric.
    
    Args:
        grid_size (tuple): (Nt, Nx, Ny, Nz)
        grid_scale (tuple): (dt, dx, dy, dz)
        world_center (tuple): (tc, xc, yc, zc)
        v (float): Velocity
        scale (float): Sizing factor (default: max(spatial_dims)/7)
        
    Returns:
        Metric object
    """
    grid = create_grid(grid_size, grid_scale)
    t, x, y, z = grid['t'], grid['x'], grid['y'], grid['z']
    
    if scale is None:
        # Default scale: max size of spatial dimensions / 7
        # grid_size[1:4] are Nx, Ny, Nz. 
        # But scale refers to physical size? 
        # MATLAB: max(gridSize(2:4))/7. 
        # This uses INDEX count, not physical size.
        # Let's replicate MATLAB logic.
        scale = max(grid_size[1:]) / 7.0
        # However, Lentz function compares 'x' (physical coordinate) with 'scale'.
        # If scale is derived from index count, but x is physical... there's a unit mismatch potential in MATLAB code?
        # MATLAB: scale = max(gridSize...)/7. 
        # usage: getWarpFactorByRegion(xp, y, scale).
        # xp, y are physical coords (i*dx - center).
        # So if dx!=1, there is a mismatch? Use physical dimension?
        # Re-reading MATLAB:
        # "scale = max(gridSize(2:4))/7;"
        # If user passes default arguments steps are 1?
        # If gridScale is passed, coordinates are scaled.
        # But scale default ignores gridScale.
        # This suggests the default is only valid if dx=1.
        # Or it interprets 'scale' in grid units?
        # But getWarpFactor compares x (physical) with scale.
        # We should probably use physical size for default if possible, or stick to MATLAB literal even if weird.
        # Let's use physical extent?
        # max(Nx*dx, Ny*dy, Nz*dz) / 7 ?
        # Given "Replica", I should probably stick to what MATLAB does or improve if obviously bugged?
        # MATLAB default seems independent of gridScale.
        # If I want to be safe, I should assume user checks their inputs.
        pass

    # Physical coordinates
    t_phys = (t + grid_scale[0]) - world_center[0]
    x_phys = (x + grid_scale[1]) - world_center[1]
    y_phys = (y + grid_scale[2]) - world_center[2]
    z_phys = (z + grid_scale[3]) - world_center[3]
    
    # Calculate bubble center position xs(t)
    # MATLAB: xs = (t*gridScale(1)-worldCenter(1))*v*c
    xs = t_phys * v * C
    
    xp = x_phys - xs
    
    # Get Lentz template values
    # These functions operate elementwise
    WFX, WFY = get_warp_factor_by_region(xp, y_phys, scale)
    
    # Beta terms
    # beta_x = -WFX * v
    # beta_y = WFY * v
    beta_x = -WFX * v
    beta_y = WFY * v
    beta_z = np.zeros_like(beta_x)
    
    # Construct metric
    # 3+1 ADM reconstruction
    # alpha = 1
    # beta = [beta_x, beta_y, 0]
    # gamma = diag(1, 1, 1)
    
    # g_00 = -alpha^2 + beta^2
    # beta^2 = beta_x^2 + beta_y^2
    
    beta_sq = beta_x**2 + beta_y**2
    
    g = np.zeros((4, 4) + t.shape)
    
    # g_00 = -1 + beta^2
    g[0, 0] = -1 + beta_sq
    
    # Shift terms (g_0i)
    g[0, 1] = beta_x
    g[1, 0] = beta_x
    g[0, 2] = beta_y
    g[2, 0] = beta_y
    # g[0, 3] is 0
    
    # Spatial part (gamma)
    g[1, 1] = 1.0
    g[2, 2] = 1.0
    g[3, 3] = 1.0
    
    metric = Metric(
        tensor=g,
        coords={
            't': t_phys,
            'x': x_phys,
            'y': y_phys,
            'z': z_phys
        },
        scaling=np.array(grid_scale),
        name="Lentz",
        index="covariant",
        params={
            'grid_size': grid_size,
            'world_center': world_center,
            'velocity': v,
            'scale': scale
        }
    )
    
    return metric
