import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

@dataclass
class Metric:
    """
    Represents a spacetime metric tensor and associated metadata.
    Replaces the MATLAB struct used in the original Warp Factory.
    """
    # 4x4 metric tensor. In 3+1 decomposition, this is usually constructed from alpha, beta, gamma.
    # Dimensions: (4, 4, t, x, y, z)
    tensor: np.ndarray
    
    # Coordinate grid (t, x, y, z)
    # Each is a 4D array of the same shape as the spacetime volume
    coords: Dict[str, np.ndarray]
    
    # Grid scaling parameters [dt, dx, dy, dz]
    scaling: np.ndarray
    
    # Metric type/name
    name: str = "Generic"
    
    # Metric index type: 'covariant' (lower indices) or 'contravariant' (upper indices)
    index: str = "covariant"
    
    # Additional parameters (e.g., velocity, radius for Alcubierre)
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validates the metric structure after initialization."""
        if self.tensor.shape[:2] != (4, 4):
            raise ValueError(f"Metric tensor must have shape (4, 4, ...), got {self.tensor.shape}")
        
        # Verify consistent dimensions
        spatial_shape = self.tensor.shape[2:]
        for key, coord in self.coords.items():
            if coord.shape != spatial_shape:
                raise ValueError(f"Coordinate {key} shape {coord.shape} does not match tensor spatial shape {spatial_shape}")
