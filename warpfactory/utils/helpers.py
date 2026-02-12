import numpy as np
from warpfactory.constants import C, G

def legendre_radial_interp(input_array, r):
    """
    3rd Order Legendre Polynomial Interpolation.
    Interp at index r (float). 
    Assume input_array is 1D array of values at integer indices 0, 1, 2...
    Or rather, r is the index?
    MATLAB code:
        x0 = floor(r/rScale-1);
        ...
        y0 = inputArray(max(x0,1));
    MATLAB indices are 1-based.
    
    If r is the index in the array (0-based in Python?):
    """
    # r is float index.
    
    # MATLAB: x0 = floor(r-1), x1=floor(r), x2=ceil(r), x3=ceil(r+1)
    # y0 = arr(max(x0, 1)) -> arr(max(floor(r)-1, 1))
    
    # Let's adjust for 0-based indexing.
    # If MATLAB r=1.5, it interpolates between index 1 and 2.
    # Python equivalent r' = r - 1 = 0.5.
    
    # Input r is likely the 0-based float index in this context?
    # In metricGet_WarpShellComoving.m:
    # minIdx = minIdx + (r-rsample(minIdx))/(rsample(minIdx+1)-rsample(minIdx));
    # This produces a float index relative to the array. 
    # If minIdx was 1-based index of left bin.
    # So r is the fractional index.
    
    # Let's implement Lagrange polynomial interpolation (it says Legendre but formula looks like Lagrange).
    
    # x values are integers around r_idx.
    
    r_idx = r
    
    x1 = np.floor(r_idx).astype(int)
    x0 = x1 - 1
    x2 = x1 + 1 # ceil(r) if r is not int
    x3 = x1 + 2
    
    # Handle boundaries (clamp to array size)
    # MATLAB used max(x, 1). 
    # Python: max(x, 0). And min(x, len-1).
    
    N = len(input_array)
    
    idx0 = np.clip(x0, 0, N-1)
    idx1 = np.clip(x1, 0, N-1)
    idx2 = np.clip(x2, 0, N-1)
    idx3 = np.clip(x3, 0, N-1)
    
    y0 = input_array[idx0]
    y1 = input_array[idx1]
    y2 = input_array[idx2]
    y3 = input_array[idx3]
    
    x = r_idx
    
    # Formula:
    # L0 = (x-x1)(x-x2)(x-x3) / ((x0-x1)(x0-x2)(x0-x3))
    # Denominators are constant for integer grid x0, x1, x2, x3 = k-1, k, k+1, k+2
    # x0-x1 = -1
    # x0-x2 = -2
    # x0-x3 = -3
    # Denom0 = (-1)(-2)(-3) = -6
    
    # x1-x0 = 1
    # x1-x2 = -1
    # x1-x3 = -2
    # Denom1 = (1)(-1)(-2) = 2
    
    # x2-x0 = 2
    # x2-x1 = 1
    # x2-x3 = -1
    # Denom2 = (2)(1)(-1) = -2
    
    # x3-x0 = 3
    # x3-x1 = 2
    # x3-x2 = 1
    # Denom3 = (3)(2)(1) = 6
    
    # Wait, the code computes denoms explicitly.
    # If x1 = x2 (r is integer), then denoms are 0?
    # No, x0, x1, x2, x3 are distinct integers usually.
    # x1 = floor(r). x2 = ceil(r).
    # If r is integer k: x1=k, x2=k. x0=k-1, x3=k+1.
    # Duplicate points!
    # MATLAB: x2 = ceil. If r=1.0, x1=1, x2=1.
    # Div by zero in MATLAB code?
    # ((x1-x0)*(x1-x2)*(x1-x3)) -> (1-0)*(1-1)*(...) = 0.
    # Does MATLAB handle this?
    # Maybe r is never exactly integer in the calling code?
    # Or maybe we need to protect against x1==x2.
    
    # If x1 == x2, then r is integer. Just return y1.
    
    # Let's check logic:
    if abs(x1 - x2) < 1e-9: 
        # Integer case (mostly)
        # But x3 is x1+2?
        # Let's enforce distinct points for interpolation if possible, or just linear/nearest if coincident.
        # But for 3rd order we need 4 points. 
        # Let's assume standard grid points: x_floor-1, x_floor, x_floor+1, x_floor+2.
        # Even if r is integer, we use these 4 distinct points.
        # Then x = x_floor is one of them.
        pass
        
    # Overriding x0, x1, x2, x3 to be distinct integers
    x_base = np.floor(r_idx).astype(int)
    x0 = x_base - 1
    x1 = x_base
    x2 = x_base + 1
    x3 = x_base + 2
    
    # Update y's
    idx0 = np.clip(x0, 0, N-1)
    idx1 = np.clip(x1, 0, N-1)
    idx2 = np.clip(x2, 0, N-1)
    idx3 = np.clip(x3, 0, N-1)
    
    y0 = input_array[idx0]
    y1 = input_array[idx1]
    y2 = input_array[idx2]
    y3 = input_array[idx3]
    
    # Term 0
    l0 = (x-x1)*(x-x2)*(x-x3) / (-6.0)
    # Term 1
    l1 = (x-x0)*(x-x2)*(x-x3) / (2.0)
    # Term 2
    l2 = (x-x0)*(x-x1)*(x-x3) / (-2.0)
    # Term 3
    l3 = (x-x0)*(x-x1)*(x-x2) / (6.0)
    
    val = y0*l0 + y1*l1 + y2*l2 + y3*l3
    return val

def tov_const_density(R, M, rho, r):
    """
    Calculates pressure profile P(r) for constant density star (TOV equation sol).
    """
    # R: outer radius
    # M: Total (or cumulative?) mass array or scalar?
    # In MATLAB: M is passed in.
    # M(end) is used as Total Mass.
    # rho: density array
    # r: radius array
    
    # Formula:
    # P = c^2 * rho * ( (R*sqrt(R-Rg) - sqrt(R^3 - Rg*r^2/R^? ...)) / (...) )
    # Rg = 2GM/c^2 (Schwarzschild radius of total mass)
    
    # MATLAB:
    # M_tot = M(end)
    # factor = 2*G*M_tot/c^2
    # num = R * sqrt(R - factor) - sqrt(R^3 - factor * r^2)
    # den = sqrt(R^3 - factor * r^2) - 3 * R * sqrt(R - factor)
    
    # Wait, units. R is likely Schwarz coordinates.
    # If r > R, P should be 0.
    # MATLAB: .* (r < R)
    
    M_tot = M[-1]
    Rg = 2 * G * M_tot / C**2
    
    # Avoid sqrt negative
    # R^3 - Rg * r^2.
    # At r=R, R^3 - Rg*R^2 = R^2(R-Rg).
    # Require R > Rg (Not a black hole) for static star.
    
    # Vectorized
    term1 = R * np.sqrt(R - Rg + 0j) # Force complex if needed? Or assume real.
    term2 = np.sqrt(R**3 - Rg * r**2 + 0j)
    
    num = term1 - term2
    den = term2 - 3 * term1
    
    P = C**2 * rho * (num / den)
    P = np.real(P) # Should be real inside star
    
    mask = r < R
    P = P * mask
    return P

def alpha_numeric_solver(M, P, R, r):
    """
    Solves for lapse function alpha.
    """
    # dalpha = (G*M/c^2 + 4*pi*G*r^3*P/c^4) / (r^2 - 2*G*M*r/c^2)
    # alpha = integral(dalpha) + offset
    
    term1 = G * M / C**2
    term2 = 4 * np.pi * G * r**3 * P / C**4
    denom = r**2 - 2 * G * M * r / C**2
    
    # Avoid division by zero at r=0
    # At r=0, M~r^3, P~const.
    # num ~ r^3. denom ~ r^2.
    # limit -> 0.
    
    dalpha = (term1 + term2) / denom
    dalpha[0] = 0 # manually set
    
    # Integrate
    alpha_temp = np.concatenate(([0], np.cumsum(0.5 * (dalpha[:-1] + dalpha[1:]) * np.diff(r))))
    # cumtrapz replacement
    
    # Boundary condition at infinity (or large r)?
    # Matches Schwarzschild metric at boundary?
    # C = 1/2 * log(1 - 2GM/rc^2)
    # At r=r_end (large), P=0, M=M_tot.
    # alpha should match 1/2 ln(1 - Rs/r) (standard Droste coords for alpha^2 = 1-Rs/r?)
    # Schwarzschild alpha = sqrt(1-Rs/r). log(alpha) = 1/2 log(...).
    # So 'alpha' variable here is actually ln(alpha_metric)?
    # MATLAB code: A = -exp(2*a).
    # If a = ln(alpha), exp(2a) = alpha^2.
    # A = -alpha^2 = -gqt. Correct.
    
    M_tot = M[-1]
    r_end = r[-1]
    
    C_const = 0.5 * np.log(1 - 2 * G * M_tot / (r_end * C**2))
    
    offset = C_const - alpha_temp[-1]
    alpha = alpha_temp + offset
    
    return alpha

def compact_sigmoid(r, R1, R2, sigma, Rbuff):
    """
    Compact sigmoid bump function.
    Non-zero between R1+Rbuff and R2-Rbuff?
    Or rather transitions between 0 and 1 or -1?
    
    MATLAB:
    f = 1 / (exp(...) + 1)
    Scaled and shifted.
    """
    # Argument of exp:
    # A = ((R2-R1-2*Rbuff)*(sigma+2))/2 * (1/(r-R2+Rbuff) + 1/(r-R1-Rbuff))
    # This blows up at boundaries.
    
    # Only evaluate inside interval
    lower = R1 + Rbuff
    upper = R2 - Rbuff
    
    mask = (r > lower) & (r < upper)
    
    # Initialize f
    f = np.zeros_like(r)
    # Region inside
    r_in = r[mask]
    
    if len(r_in) > 0:
        arg = ((upper - lower) * (sigma + 2) / 2.0) * (1.0 / (r_in - upper) + 1.0 / (r_in - lower))
        # exp(arg)
        # If arg is large positive -> exp huge -> f -> 0
        # If arg large negative -> exp 0 -> f -> 1/(0+1) = 1
        
        # When r -> lower+, 1/(r-lower) -> +inf. arg -> +inf. f -> 0.
        # When r -> upper-, 1/(r-upper) -> -inf. arg -> -inf. f -> 1.
        
        # Wait, looking at MATLAB code:
        # f = abs( 1/(...) * (mask) + (r >= upper) - 1 )
        # Using analytical continuation?
        # If r >= upper, f = abs(0 + 1 - 1) = 0?
        # Ideally we want a bump 0 to 1?
        
        # Actually checking logic:
        # If r > R1 (inside shell?), value should be non-zero shift?
        # Comoving WarpShell usually has Shift=v inside hole.
        # So we expect f=1 inside hole (r < R1).
        # But here logic is: (r>R1+Rbuff) and (r<R2-Rbuff).
        # This defines the "wall" region?
        # Or is it 0 outside, 1 inside?
        
        term = 1.0 / (np.exp(arg) + 1.0)
        f[mask] = term
        
    # Add (r >= upper) - 1 ?
    # If r >= upper, term is 0. f = abs(0 + 1 - 1) = 0.
    # If r <= lower, term is 0. f = abs(0 + 0 - 1) = 1.
    
    # So f=1 for r <= lower. f=0 for r >= upper.
    # Transition in between.
    
    f[r <= lower] = 1.0
    f[r >= upper] = 0.0
    
    # MATLAB code has the -1 outside abs?
    # f = abs ( sig * mask + (r>=upper) - 1 )
    # If r >= upper: sig=0, mask=0. abs(0+1-1) = 0.
    # If r <= lower: sig=0, mask=0. abs(0+0-1) = 1.
    # inside: mask=1. abs(sig - 1).
    # Since sig is between 0 and 1, abs(sig-1) = 1-sig.
    # So it flips the sigmoid?
    # 1/(exp+1). r->lower (arg->inf) -> sig->0. f -> 1.
    # r->upper (arg->-inf) -> sig->1. f -> 0.
    # Yes.
    
    return f

def sph2cart_diag(theta, phi_ang, g11_sph, g22_sph):
    """
    Transforms diagonal spherical metric components to Cartesian components.
    """
    # g33_sph = g22_sph * sin^2(theta) ? No, assumed isotropic/spherically symmetric part?
    # MATLAB: E = g22_sph. 
    # Assumes g_theta_theta = r^2 * ...
    # Wait, g22_sph here is likely just the coefficient of dOmega^2?
    # Usually ds^2 = A dr^2 + B r^2 dOmega^2.
    # g11_sph = A.
    # g22_sph = B * r^2 ? Or just B?
    # The helper takes g22_sph as 'E'. 
    # And transforms...
    # Looks like a rotation of the tensor.
    
    cosTheta = np.cos(theta)
    sinTheta = np.sin(theta)
    cosPhi = np.cos(phi_ang)
    sinPhi = np.sin(phi_ang)
    
    E = g22_sph
    # g11_sph is radial component (A).
    # E is tangential component.
    
    # Transformation matrix from Spherical(r, theta, phi) to Cartesian(x, y, z)?
    # Actually simpler:
    # T_cart = R T_sph R^T
    # If T_sph = diag(g11, E, E) (if theta part matches phi part/sin theta factor removed?)
    # In orthonormal basis?
    # Let's assume the formula extracts Cartesian components g_xx etc.
    # MATLAB code:
    # g22_cart = (E*cosPhi^2*sinTheta^2 + (cosPhi^2*cosTheta^2)) + sinPhi^2;
    # Wait, where is g11_sph? It's set to g11_cart line 3.
    # "g11_cart = g11_sph;" -> This assumes x is r?
    # No, indices 1,2,3,4.
    # MATLAB: 1=x ? No, usually 1=t or 4=t.
    # WarpFactory metric indices:
    # "The tensor indices are defined as 0: t, 1: x, 2: y, 3: z" (Py) (1: t, 2: x, 3: y, 4: z (Mat?))
    # In `sph2cartDiag`:
    # Returns g11, g22, ...
    # And calls them g11_cart, etc.
    # In `metricGet_WarpShellComoving`:
    # Metric.tensor{1,1} = g11_cart; -> t-t component?
    # Metric.tensor{2,2} = g22_cart; -> x-x component?
    # Arguments to sph2cartDiag:
    # theta, phi, g11_sph (A), g22_sph (B).
    # g11_sph (A) is -alpha^2 ?
    # In `metricGet...`:
    # g11_sph = legendre(A...).
    # A = -exp(2a). Time component.
    # g22_sph = legendre(B...).
    # B is radial spatial component? (1-2M/r)^-1.
    # So input is Time, Radial.
    
    # Wait.
    # The MATLAB output assignments:
    # Metric{1,1} = g11_cart (Time?)
    # Metric{2,2} = g22_cart (X)
    # Metric{2,3} = g23_cart (XY)
    # ...
    # Metric{3,3} = g33_cart (Y)
    # Metric{4,4} = g44_cart (Z)
    
    # But A is time component of metric (-alpha^2).
    # B is spatial radial component.
    # Spherical metric: ds^2 = A dt^2 + B dr^2 + B r^2 dOmega^2 ?
    # Isotropic coordinates? B(dr^2 + r^2 dOmega^2).
    # If B is isotropic factor, then g_xx = B, g_xy = 0.
    
    # But TOV/Schwarzschild is usually:
    # ds^2 = -e^{2a} dt^2 + (1-2M/r)^-1 dr^2 + r^2 dOmega^2.
    # This is NOT isotropic.
    # Transforming (-e^2a, (1-2M/r)^-1, r^2, r^2 sin^2 theta) to Cartesian (t, x, y, z).
    # t is unchanged. g_tt = A.
    # So g11_cart should be A.
    # Spatial part:
    # g_rr = B. g_theta_theta = r^2. g_phi_phi = r^2 sin^2 theta.
    # The helper logic:
    # `g11_cart = g11_sph` -> t component passed through.
    # `g22_cart`, `g23_cart` etc are spatial.
    # Formula uses E = g22_sph and '1' for the other pars?
    # (cosPhi^2*cosTheta^2) + sinPhi^2.
    # Where is B (radial part)?
    # Note: `g22_sph` in the call is `B`.
    # And the formulas look like:
    # g_xx = g_rr * (dx/dr)^2 + g_th_th * ...
    # x = r sin th cos phi.
    # dr/dx = ...
    # Inverse: dx = dr sin th cos phi + r cos th cos phi dth - r sin th sin phi dphi.
    # If we have ds^2_spatial = B dr^2 + r^2 dth^2 + r^2 sin^2 th dphi^2.
    # If we assume B is passed as `E`?
    # But the formula `(E*cosPhi^2*sinTheta^2 + ...)` suggests E is multiplying the radial direction part (dx/dr = sin th cos phi).
    # dx/dr squared is sin^2 th cos^2 phi.
    # So E is B (g_rr).
    # The other terms correspond to g_th_th and g_phi_phi.
    # The formula has simply `cosPhi^2...` (implied coefficient 1?).
    # This implies g_th_th = r^2 ? But r is not in the formula.
    # Ah, `metricGet` calls `sph2cartDiag` AT A POINT (x,y,z).
    # The transformation tensor Jacobian likely has 1/r factors that cancel r^2?
    # Or maybe the metric was constructed such that tangential parts are 1 relative to radial?
    # This seems specific to the WarpShell implementation where B is defined, and maybe angular part is standard?
    # "construct metric using spherical symmetric solution... B = ...".
    # It assumes standard spherical metric structure.
    # But transformation to Cartesian requires mixing B and 1?
    # If ds^2 = B dr^2 + r^2 dOmega^2.
    # dx^2 + dy^2 + dz^2 = dr^2 + r^2 dOmega^2.
    # So r^2 dOmega^2 = dx^2 + ... - dr^2.
    # ds^2 = B dr^2 + (dx^2... - dr^2) = (B-1) dr^2 + (dx^2+dy^2+dz^2).
    # So g_ij = delta_ij + (B-1) n_i n_j.
    # where n_i = x_i / r.
    # Let's see if the formula matches.
    # n_x = sin th cos phi.
    # g_xx = 1 + (B-1) n_x^2
    #      = 1 + (B-1) sin^2 th cos^2 phi
    #      = 1 - sin^2 th cos^2 phi + B sin^2 th cos^2 phi.
    #      = (sin^2 th + cos^2 th)(cos^2 phi + sin^2 phi) - ...
    #      = ...
    # The MATLAB formula:
    # g22 = E * (sin th cos phi)^2 + (cos th cos phi)^2 + sin^2 phi.
    #      = B * n_x^2 + (something else).
    # (cos th cos phi)^2 + sin^2 phi.
    # n_x^2 + n_y^2 + n_z^2 = 1.
    # n_x = sin th cos phi.
    # n_y = sin th sin phi.
    # n_z = cos th.
    # If B dr^2 + r^2 dOmega^2, this is effectively B P_r + P_perp?
    # Yes.
    # So g_cart = B P_r + 1 P_perp.
    #           = B (n n^T) + (I - n n^T)
    #           = I + (B-1) n n^T.
    # Let's check if the formula equals 1 + (E-1) n_x^2.
    # Formula: E n_x^2 + (cos th cos phi)^2 + sin^2 phi.
    # We know 1 = n_x^2 + n_y^2 + n_z^2.
    # 1 - n_x^2 = n_y^2 + n_z^2 = sin^2 th sin^2 phi + cos^2 th.
    # Formula term 2: cos^2 th cos^2 phi + sin^2 phi.
    # Is this n_y^2 + n_z^2?
    # cos^2 th (1 - sin^2 phi) + sin^2 phi
    # = cos^2 th - cos^2 th sin^2 phi + sin^2 phi
    # = cos^2 th + sin^2 phi (1 - cos^2 th)
    # = cos^2 th + sin^2 phi sin^2 th.
    # = n_z^2 + n_y^2.
    # YES.
    # So g_xx = E n_x^2 + (1 - n_x^2) = 1 + (E-1) n_x^2.
    # This confirms the logic.
    # The formula implements g = I + (B-1) n n^T.
    
    # We can implement this simply using vectors instead of trig spaghetti.
    
    n_x = sinTheta * cosPhi
    n_y = sinTheta * sinPhi
    n_z = cosTheta
    
    # g_xx
    g22_cart = 1 + (E - 1) * n_x**2
    # g_yy
    g33_cart = 1 + (E - 1) * n_y**2
    # g_zz
    g44_cart = 1 + (E - 1) * n_z**2
    
    # g_xy
    g23_cart = (E - 1) * n_x * n_y
    # g_xz
    g24_cart = (E - 1) * n_x * n_z
    # g_yz
    g34_cart = (E - 1) * n_y * n_z
    
    return g11_sph, g22_cart, g23_cart, g24_cart, g33_cart, g34_cart, g44_cart
