import numpy as np

def trilinear_interp(data, point):
    """
    Trilinear interpolation for a 3D point in a 3D array.
    Assumes data is on a grid with integer indices starting at 0.
    
    Args:
        data: 3D numpy array (Nx, Ny, Nz)
        point: tuple or list (x, y, z) coordinates in index space (float)
    
    Returns:
        Interpolated value.
    """
    x, y, z = point
    
    # Grid dimensions
    Nx, Ny, Nz = data.shape
    
    # Indices
    x0 = int(np.floor(x))
    x1 = x0 + 1
    y0 = int(np.floor(y))
    y1 = y0 + 1
    z0 = int(np.floor(z))
    z1 = z0 + 1
    
    # Check bounds
    if x0 < 0 or x1 >= Nx or y0 < 0 or y1 >= Ny or z0 < 0 or z1 >= Nz:
        return np.nan
        
    # Weights
    xd = x - x0
    yd = y - y0
    zd = z - z0
    
    # Interpolate along x
    c00 = data[x0, y0, z0] * (1 - xd) + data[x1, y0, z0] * xd
    c01 = data[x0, y0, z1] * (1 - xd) + data[x1, y0, z1] * xd
    c10 = data[x0, y1, z0] * (1 - xd) + data[x1, y1, z0] * xd
    c11 = data[x0, y1, z1] * (1 - xd) + data[x1, y1, z1] * xd
    
    # Interpolate along y
    c0 = c00 * (1 - yd) + c10 * yd
    c1 = c01 * (1 - yd) + c11 * yd
    
    # Interpolate along z
    c = c0 * (1 - zd) + c1 * zd
    
    return c

def get_momentum_flow_lines(energy_tensor, start_points, step_size, max_steps, scale_factor):
    """
    Computes momentum flow lines.
    Replicates getMomentumFlowLines.m
    
    Args:
        energy_tensor (ndarray): Energy tensor T^mn or similar.
                                 MATLAB expects 'contravariant'.
                                 Likely passing T^0i components?
                                 Structure: (4, 4, t, x, y, z)
        start_points (list of tuples): [(x, y, z), ...] starting points.
        step_size (float): Integration step size.
        max_steps (int): Max steps.
        scale_factor (float): Scaling factor for momentum.
        
    Returns:
        list of ndarrays: List of paths (N_steps, 3)
    """
    # Extract momentum components T^0i (or similar)
    # MATLAB: 
    # Xmom = squeeze(energyTensor.tensor{1, 2}) * scaleFactor;
    # Ymom = squeeze(energyTensor.tensor{1, 3}) * scaleFactor;
    # Zmom = squeeze(energyTensor.tensor{1, 4}) * scaleFactor;
    # Indices 1,2 -> 0,1 (Tx)
    # Indices 1,3 -> 0,2 (Ty)
    # Indices 1,4 -> 0,3 (Tz)
    
    # Expecting energy_tensor shape (4, 4, t, x, y, z)
    # We generally visualize flow on a specific time slice?
    # MATLAB `getMomentumFlowLines` loop structure:
    # `floor(Pos(i,1)) ... size(Xmom,1)`
    # The Xmom in MATLAB seems to be 3D? Or 4D?
    # "squeeze(tensor{1,2})". If tensor is 4D (t,x,y,z), squeeze keeps it 4D?
    # Unless t=1.
    # Usually flow lines are calculated on a spatial slice.
    # Let's assume input energy_tensor is 3D (x,y,z) or we slice it.
    # If passed full 4D tensor, we need to know WHICH time slice.
    # MATLAB code doesn't specify time slice index selection, just squeeze.
    # If T dim > 1, squeeze(T(1,2,:,:,:,:)) -> (t,x,y,z).
    # Then access Xmom(floor(x), floor(y), floor(z)).
    # Wait, MATLAB: `Pos(i,2)` is compared to `size(Xmom,2)`.
    # And interps using `Pos`.
    # It treats dimensions as X, Y, Z.
    # If 4D, this would fail or need T index.
    # We will assume the user passes a SINGLE TIME SLICE of the tensor (4, 4, x, y, z).
    
    T_0x = energy_tensor[0, 1] * scale_factor
    T_0y = energy_tensor[0, 2] * scale_factor
    T_0z = energy_tensor[0, 3] * scale_factor
    
    # Dimensions
    dim_x, dim_y, dim_z = T_0x.shape
    
    paths = []
    
    for start_pt in start_points:
        path = []
        curr_pos = np.array(start_pt, dtype=float)
        path.append(curr_pos.copy())
        
        for _ in range(max_steps):
            x, y, z = curr_pos
            
            # Check bounds (0-based)
            # Safe margin 1.0 like MATLAB?
            # MATLAB: floor(Pos) <= 1 or ceil >= size. (1-based)
            # Python equivalent: < 0 or >= size - 1 (for interpolation safety)
            if (x < 0 or x >= dim_x - 1 or 
                y < 0 or y >= dim_y - 1 or 
                z < 0 or z >= dim_z - 1):
                break
                
            # Interpolate
            vals = [
                trilinear_interp(T_0x, curr_pos),
                trilinear_interp(T_0y, curr_pos),
                trilinear_interp(T_0z, curr_pos)
            ]
            
            if np.any(np.isnan(vals)):
                break
                
            mom_vec = np.array(vals)
            
            # Update position
            curr_pos += mom_vec * step_size
            path.append(curr_pos.copy())
            
        paths.append(np.array(path))
        
    return paths
