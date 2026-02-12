import numpy as np
import matplotlib.pyplot as plt
import os
from warpfactory.analyzer.scalars import three_plus_one_decomposer

def get_slice_data(data, slice_planes, slice_locations):
    """
    Extracts a 2D slice from N-dimensional data.
    
    Args:
        data: N-dimensional numpy array.
        slice_planes: Tuple of two indices (axis1, axis2) defining the slice plane.
                      e.g. (1, 2) for x-y plane (if t is 0).
        slice_locations: list/tuple of indices for the NON-sliced dimensions.
                         Must be ordered corresponding to dimensions NOT in slice_planes.
                         Wait, MATLAB logic: passing location for ALL dimensions?
                         MATLAB: "sliceLocations(1) = ...".
                         Let's simplify: pass dictionary {axis: index}?
                         Or a list of indices for all dimensions, ignoring those in slice_planes?
                         
                         Let's stick to: list of length ndim, values at slice_planes are ignored.
                         
    Returns:
        2D numpy array.
    """
    ndim = data.ndim
    
    # Construct slice object
    slices = [slice(None)] * ndim
    
    # slice_planes are the axes we keep (e.g. x, y)
    # The others are fixed to slice_locations[axis]
    
    for i in range(ndim):
        if i not in slice_planes:
            # Fix this dimension
            # Ensure slice_locations has this index?
            # User must provide a full location array or we assume center?
            # Let's assume slice_locations is a list of len(ndim)
            if slice_locations is None or len(slice_locations) <= i:
                 idx = data.shape[i] // 2 
            else:
                 idx = slice_locations[i]
                 
            slices[i] = idx
            
    return data[tuple(slices)]

def plot_tensor(tensor, slice_planes=(1, 2), slice_locations=None, save_dir=None, filename_prefix="tensor"):
    """
    Plots unique elements of a tensor on a 2D slice.
    
    Args:
        tensor: Metric object or numpy array (4, 4, ...)
        slice_planes: Tuple (axis1, axis2) to plot.
                      0=t, 1=x, 2=y, 3=z.
                      Be careful with indices. 
                      Metric.tensor is (4, 4, t, x, y, z).
                      So data axes are 2, 3, 4, 5.
                      Slice planes (1, 2) usually means x-y.
                      In data array terms, this matches axes 3 and 4?
                      Let's assume user passes (1, 2) meaning x, y.
                      We map this to data axes + 2.
    """
    if hasattr(tensor, 'tensor'):
        data = tensor.tensor
        coords = tensor.coords
    else:
        data = tensor
        coords = None
        
    # Map visual axes (0=t, 1=x, 2=y, 3=z) to array axes (2=t, 3=x, 4=y, 5=z)
    # The first two axes of data are 4x4 components.
    array_slice_planes = (slice_planes[0] + 2, slice_planes[1] + 2)
    
    # Prepare locations
    # We need locations for t, x, y, z.
    # slice_locations should be length 4.
    ndim_spatial = 4
    if slice_locations is None:
        slice_locations = [data.shape[i+2] // 2 for i in range(ndim_spatial)]
    
    # Adjust for array indexing
    # We need to pass a full index list to get_slice_data for the spatial part.
    # But get_slice_data handles generic ndim.
    # Our data is (4, 4, t, x, y, z).
    # We want to slice components.
    
    # We will iterate over components 0..3, 0..3
    
    unique_components = []
    # Symmetric tensor? Metric is symmetric.
    # Plot diagonal and upper triangle.
    
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    
    x_label = f"Axis {slice_planes[0]}"
    y_label = f"Axis {slice_planes[1]}"
    
    for i in range(4):
        for j in range(4):
            ax = axes[i, j]
            if i > j: 
                ax.axis('off')
                continue
                
            # Extract component data (t, x, y, z)
            comp_data = data[i, j]
            
            # Slice it
            # comp_data has shape (t, x, y, z) i.e. 4 dims.
            # user slice_planes (1, 2) refers to x, y.
            # corresponds to axes 1, 2 of comp_data (if t is 0).
            slice_2d = get_slice_data(comp_data, slice_planes, slice_locations)
            
            # Transpose for correct x-y plotting (imshow is row-col -> y-x usually)
            # We want axis1 horizontal, axis2 vertical? OR standard matrix?
            # Standard: imshow(M) -> M[row, col]. row=y, col=x.
            # So if slice_2d is (x, y), we transpose to (y, x).
            im = ax.imshow(slice_2d.T, origin='lower') 
            ax.set_title(f"g_{{{i}{j}}}")
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            fig.colorbar(im, ax=ax)
            
    plt.tight_layout()
    
    if save_dir:
        import os
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(os.path.join(save_dir, f"{filename_prefix}.png"))
        plt.close(fig)
    else:
        plt.show()

def plot_3plus1(metric, slice_planes=(1, 2), slice_locations=None, save_dir=None, filename_prefix="adm"):
    """
    Plots 3+1 ADM variables.
    """
    alpha, beta_down, gamma_down, _, _ = three_plus_one_decomposer(metric)
    
    # alpha is scalar field (t, x, y, z)
    # beta_down is vector (3, t, x, y, z). indices 0,1,2 correspond to x,y,z? No, 1,2,3 in MATLAB. 0,1,2 here.
    # gamma_down is tensor (3, 3, t, x, y, z).
    
    # Plot Alpha
    # Slice
    alpha_slice = get_slice_data(alpha, slice_planes, slice_locations)
    
    plt.figure(figsize=(6, 5))
    plt.imshow(alpha_slice.T, origin='lower')
    plt.colorbar(label='Alpha')
    plt.title('Lapse Function (Alpha)')
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"{filename_prefix}_alpha.png"))
        plt.close()
    else:
        plt.show()

    # Plot Beta (3 components)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i in range(3):
        beta_comp = beta_down[i]
        beta_slice = get_slice_data(beta_comp, slice_planes, slice_locations)
        im = axes[i].imshow(beta_slice.T, origin='lower')
        axes[i].set_title(f"Beta_{i+1}")
        fig.colorbar(im, ax=axes[i])
    
    plt.suptitle("Shift Vector (Beta)")
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"{filename_prefix}_beta.png"))
        plt.close()
    else:
        plt.show()
        
    # Plot Gamma (6 components)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    # 11, 12, 13
    # 22, 23, 33
    comps = [(0,0), (0,1), (0,2), (1,1), (1,2), (2,2)]
    
    for idx, (i, j) in enumerate(comps):
        row = idx // 3
        col = idx % 3
        
        gamma_comp = gamma_down[i, j]
        gamma_slice = get_slice_data(gamma_comp, slice_planes, slice_locations)
        
        im = axes[row, col].imshow(gamma_slice.T, origin='lower')
        axes[row, col].set_title(f"Gamma_{{{i+1}{j+1}}}")
        fig.colorbar(im, ax=axes[row, col])
        
    plt.suptitle("Spatial Metric (Gamma)")
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"{filename_prefix}_gamma.png"))
        plt.close()
    else:
        plt.show()
