import numpy
from scipy import ndimage
from typing import *

# TODO: replace n_domain_radius and n_kernel_radius with floats, its more interpretable
# TODO: i think next steps, it let u* be the largest fixed point of g(u,a,r)
# then start the initial datum from (u*, 0) and track the points
# alpha(t) = sup { x | forall y leq x, |u_t(y) - g^t(u*)| < epsilon }
# and beta(t) = inf { x | forall y geq x, |u_t(y)| < epsilon }
# and track (beta(t) - alpha(t)) / t ?

# Ricker model. https://en.wikipedia.org/wiki/Ricker_model
def ricker_map(r: float) -> Callable[[float], float]:
    return lambda u: u * numpy.exp(r * (1 - u))

# Vortkamp growth model. https://doi.org/10.1007/s11538-020-00750-x
def vortkamp_map(r: float, a: float) -> Callable[[float], float]:
    return lambda u: u * numpy.exp(r * (1 - u) * (u/a - 1))

# global argmax of vortkamp map.
def vortkamp_argmax (r: float, a: float) -> Callable[[float], float]:
    return 1/4 * (1 + a + (1 + 2*a + 8*a/r + a**2)**(1/2))

# global max of vortkamp map.
def vortkamp_max (r: float, a: float) -> Callable[[float], float]:
    g = vortkamp_map(r, a)
    return g(vortkamp_argmax(r, a))

# Vortkamp et al model, alternate parameterization.
def vortkamp_map_alt(r: float, a: float) -> Callable[[float], float]:
    return lambda u: u * numpy.exp(r*(1 - u)*(u - a))

# Schreiber growth model. https://doi.org/10.1016/S0040-5809(03)00072-8
def schreiber_map(r: float, m: float, s: float) -> Callable[[float], float]:
    return lambda u: u * numpy.exp(r*(1 - u) - m/(1 + s*u))

# Laplace kernel. https://en.wikipedia.org/wiki/Laplace_distribution
def laplace_kernel(a: float = 1) -> Callable[[float], float]:
    return lambda x: a/2 * numpy.exp(-a * numpy.abs(x))

def detect_period_vortkamp(r: float, a: float, n: int = 10, eps: float = 1e-10, delta: float = 1e-10) -> dict:
    """
    A recursive period-detecting function specialized to the Vortkamp map.
    Exploits the fact that any periodic point will lie in (1, umax) where umax is the global maximum.

    Inumpy.ts:
        r: float, a: float
            Vortkamp map parameters. We need to keep a around for the recursion.
        n: int
            Max number of recursions and/or reductions in the padding parameter eps.
        eps: float
            Used for padding the search interval. Can be dynamically decreased at the cost of n.
        delta: float
            Used to approximate derivative of iterates of f.
    Output:
        Either None, or a dict with two fields:
            'root': float
                One of the periodic points.
            'period': int
                The period of the root.
    """
    # fetch the vortkamp max point
    g = vortkamp_map(r, a)
    umax = vortkamp_max(r, a)
    from scipy.optimize import root_scalar
    def _detect_period_recursor(f: Callable[[float], float], root: float, period: int, n: int, eps: float):
        if n == 0:
            return None
        abs_deriv_at_root = abs((f(root + delta) - f(root))/delta)
        if abs_deriv_at_root < 1:
            return {'root': root, 'period': period}
        elif abs_deriv_at_root == 1:
            return None
        else:
            ff = lambda u: f(f(u))
            try:
                root_next = root_scalar(lambda u: ff(u) - u, bracket=[root + eps, umax - eps], method='brentq').root
                #print(root_next)
                return _detect_period_recursor(ff, root_next, 2*period, n - 1, eps)
            except ValueError:
                return _detect_period_recursor(f, root, period, n - 1, eps/10)
    return _detect_period_recursor(g, root = 1.0, period = 1, n = n, eps = eps)

def classify_vortkamp_map(r: float, a: float) -> dict:
    """
    Partial classification of the dynamics of the Vortkamp model.
    Useful for autogenerating heatmaps.
    
    Inumpy.ts:
        r: float
            Intrinsic per-capita growth rate for the Vortkamp model.
        a: float
            Allee threshold for the Vortkamp model.

    Output:
        dict with several fields:
            'zero_stable': bool
                Whether u = 0 is (asymptotically) stable. This is true iff. as r > 0.
            'positive_attractor': Optional[bool]
                Does a positive attractor exist?
            'cycle': Optional[numpy.ndarray]
                The positive stable cycle, if it can be found.
                Chosen so that the maximal element is first.
            'period': Optional[int]
                The period of the cycle.
    """
    if r < 0:
        raise ValueError(f"r = {r} but must be nonnegative.")
    if a <= 0 or a > 1:
        raise ValueError(f"a = {a} but must be in (0, 1].")
    
    # prefill result
    data = {
        'zero_stable': None,
        'positive_attractor': None,
        'cycle': None,
        'period': None
    }
    
    data['zero_stable'] = r > 0
    if not data['zero_stable']:
        data['positive_attractor'] = False
        return data
    g = vortkamp_map(r, a)

    # the positive attractor exists iff. g(umax) > a where umax is the global max.
    umax = vortkamp_max(r, a)
    u0 = g(umax)
    if u0 < a:
        data['positive_attractor'] = False
    elif u0 > a:
        data['positive_attractor'] = True
        period = detect_period_vortkamp(r, a)
        if period:
            root = period['root']
            p = period['period']
            cycle = [root]
            for i in range(p - 1):
                cycle.append(g(cycle[i]))
            data['cycle'] = numpy.array(cycle, dtype=numpy.float64)
            data['period'] = p
    return data

def positive_attractor(r, a):
    # there is a positive attractor if g(umax) > a where umax is the global max.
    g = vortkamp_map(r, a)
    umax = vortkamp_max(r, a)
    return g(umax) > a

def find_r_threshold(a: float) -> float:
    """
    Given a, finds the corresponding threshold above which the map has no stable attractor.
    """
    r0 = 1
    if positive_attractor(r0, a):
        s = 2
        while positive_attractor(r0, a):
            r0 *= s
    else:
        s = 1/2
        while not positive_attractor(r0, a):
            r0 *= s
    f = lambda r: vortkamp_map(r, a)(vortkamp_max(r, a)) - a
    from scipy.optimize import root_scalar
    root = root_scalar(f, bracket=[r0 / s, r0], method='brentq').root
    return root

    #print(r0, a)
    #print(positive_attractor(r0, a))
# print(find_r_threshold(a=0.5))
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib as mpl

#print(classify_vortkamp_map(3.5, 0.3))
# I think just do a bifurcation plot
from tqdm import tqdm
def heatmap_periods(a_vals, r_vals, p_max=16):
    Z = numpy.full((len(a_vals), len(r_vals)), numpy.nan)

    for j, r in enumerate(r_vals):
        sweep = False
        for i, a in enumerate(a_vals):
            print(a, r)
            if sweep:
                Z[i, j] = 1
            else:

                try:
                    out = classify_vortkamp_map(r, a)
                    if out['positive_attractor'] and out['period'] is not None:
                        p = out['period']
                        Z[i, j] = min(p, p_max)  # clamp if too large
                        if p == 1:
                            sweep = True
                        #print(r, a, p)
                except Exception:
                    pass
    #return Z

    # grids
    a_vals = numpy.linspace(0.2, 0.8, 150)
    r_vals = numpy.linspace(1, 3.0, 150)

    Z = heatmap_periods(a_vals, r_vals, p_max=16)

    # colormap: discrete integers, grey for NaN
    P_MAX = 4
    cmap = plt.cm.get_cmap('tab20', 2**P_MAX).copy()
    cmap.set_bad('lightgrey')
    bounds = numpy.arange(0.5, 2**P_MAX + 1.5)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(
        Z,
        origin='lower',
        aspect='auto',
        extent=[r_vals[0], r_vals[-1], a_vals[0], a_vals[-1]],
        cmap=cmap, norm=norm, interpolation='nearest'
    )
    ax.set_xlabel("r")
    ax.set_ylabel("a")
    cbar = fig.colorbar(im, ax=ax, ticks=[2**p for p in range(P_MAX)])
    cbar.set_label("detected period")
    ax.set_title("Vortkamp map: period of positive attractor")

    plt.tight_layout()
    plt.show()

# a_vals = numpy.linspace(0, 1, 300)
# s_vals = numpy.linspace(0, 1, 300)
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# from tqdm import tqdm
# # map periods directly to indices
# color_dict = {1: 0, 2: 1, 4: 2, 8: 3, 16: 4}
# colors = ["blue", "orange", "green", "red", "purple"]  # one per category
# color_dict = {k: i for (i, k) in enumerate(colors)}
# Z = numpy.full((len(a_vals), len(s_vals)), numpy.nan)

# for j, s in tqdm(list(enumerate(s_vals))):
#     sweep = False
    
#     r = s / (1 - s)
#     for i, a in enumerate(a_vals):
#         if sweep:
#             Z[i,j] = color_dict["blue"]  # index for period 1
#         else:
#             try:
#                 out = classify_vortkamp_map(r, a)
#                 if out['positive_attractor'] and out['period'] is not None:
#                     p = out['period']
#                     if p == 1:
#                         Z[i,j] = color_dict["blue"]
#                     if p == 2:
#                         Z[i,j] = color_dict["red"]
#                     if p == 3:
#                         Z[i,j] = color_dict["blue"]
#                     if p == 4:
#                         Z[i,j] = color_dict["rede"]
#                     #if p in color_dict:
#                     #    Z[i, j] = color_dict[p]
#                     if p == 1:
#                         sweep = True
#             except Exception:
#                 pass

# # build colormap and plot
# cmap = mpl.colors.ListedColormap(colors)
# norm = mpl.colors.BoundaryNorm(numpy.arange(-0.5, len(colors)+0.5, 1), cmap.N)

# fig, ax = plt.subplots()
# im = ax.imshow(Z, cmap=cmap, norm=norm, origin="lower",
#                extent=[s_vals[0], s_vals[-1], a_vals[0], a_vals[-1]])

# cbar = fig.colorbar(im, ax=ax, ticks=list(color_dict.values()))
# cbar.ax.set_yticklabels(list(color_dict.keys()))  # show periods instead of indices
# cbar.set_label("detected period")

#plt.show()


# # colormap: discrete integers, grey for NaN
# valid_periods = numpy.array([1, 2, 4, 8, 16])
# Z_clean = Z.copy()
# Z_clean[Z_clean > 16] = 16
# is_pow2 = numpy.isin(Z_clean, valid_periods)
# Z_clean[~is_pow2] = numpy.nan   # render as grey

# # discrete colormap with exactly 5 bins
# colors = ["#1271b4", "#e6c244", "#f07d50", "#ff0e0e",]  # pick your 5
# cmap = mpl.colors.ListedColormap(colors)
# cmap.set_bad("lightgrey")  # NaNs

# # bin edges centered around the categories
# bounds = numpy.array([0.5, 1.5, 2.5, 4.5, 8.5,])
# norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

# fig, ax = plt.subplots(figsize=(7, 5))
# im = ax.imshow(
#     Z_clean,
#     origin="lower",
#     aspect="auto",
#     extent=[s_vals[0], s_vals[-1], a_vals[0], a_vals[-1]],
#     cmap=cmap,
#     norm=norm,
#     interpolation="nearest",
# )

# ax.set_xlabel("s")
# ax.set_ylabel("a")

# # cbar = fig.colorbar(im, ax=ax, ticks=valid_periods)
# # cbar.set_label("detected period")
# # ax.set_xlabel("s")
# # ax.set_ylabel("a")
# #cbar = fig.colorbar(im, ax=ax)#, ticks=[2**p for p in range(P_MAX)])
# #cbar.set_label("detected period")
# ax.set_title("Vortkamp map: period of positive attractor")
# s_curve = numpy.linspace(s_vals[0], s_vals[-1], 1000)

# ax.plot(s_curve, s_curve, color="black", lw=1.5, ls="--", label="a = s")
# ax.plot(2*s_curve/(1+s_curve), s_curve, color="red", lw=1.5, ls="--", label="2a = s")
# ax.set_xlim(0, 1)
# ax.set_ylim(0, 1)
# ax.legend(frameon=False, loc="upper left")
# plt.tight_layout()
# plt.show()

def solve_ide(
    growth_map: Callable[[float], float],
    kernel_density_fn: Callable[[float], float],
    initial_data: Callable[[float], float],
    xmin: float,
    xmax: float,
    kernel_radius: float,
    dx: float,
    n_steps: int,
    normalize_kernel: bool = True
) -> dict:
    """
    Solve the IDE with generic initial data:
        u(x, t + 1) = ∫ k(x - y) g(u(y, t)) dy
        u(x, 0)     = u0(x)
    Assumes the solution is constant outside [xmin, xmax].

    Alternatively, can be viewed as solving the IDE on a bounded domain:
        u(x, t + 1) = ∫_∞^xmin    k(x - z) g(u(xmin, t)) dz
                    + ∫_xmin^xmax k(x - z) g(u(z,    t)) dz
                    + ∫_xmax^∞    k(x - z) g(u(xmax, t)) dz
    with dynamic boundary conditions:
        u(xmin, t + 1) = g(u(xmin, t))
        u(xmax, t + 1) = g(u(xmax, t))

    Inumpy.ts:
        growth_map: Callable[[float], float]
            Scalar growth function.
        kernel_density_fn: Callable[[float], float]
            Kernel function. Assumed to be symmetric.
        initial_data: Callable[[float], float]
            Initial data function.
        xmin: float, xmax: float
            Left and right domain endpoints.
        kernel_radius: float
            Kernel radius.
        dx: float,
            Spacing between grid points.
            Must divide both (xmax - xmin) and kernel_radius.
        n_steps: int
            Number of time steps to simulate.
        normalize_kernel: bool
            Whether to divide the kernel by its sum.
            Default is true so it is a probability kernel.
    Output:
        dict with two fields:
            'domain': ndarray
                1d array of domain points,
                i.e. {-n*dx, ..., n*dx} where n = (xmax - xmin) / dx.
            'solution': ndarray
                2d array of shape (n_steps + 1, n) containing the solution surface.
    """

    assert(((xmax - xmin) / dx).is_integer())
    assert((kernel_radius / dx).is_integer())

    domain = numpy.arange(xmin, xmax + dx, dx)
    kernel_domain = numpy.arange(-kernel_radius, kernel_radius + dx, dx)

    kernel = kernel_density_fn(kernel_domain)
    if normalize_kernel:
        kernel /= numpy.sum(kernel)

    soln = numpy.zeros((n_steps + 1, len(domain)))
    soln[0] = initial_data(domain)

    for t in range(n_steps):
        soln[t + 1] = ndimage.convolve(growth_map(soln[t]), kernel, mode='nearest')
        # set boundary data
        soln[t + 1][0] = growth_map(soln[t][0])
        soln[t + 1][-1] = growth_map(soln[t][-1])

    return {
        'domain': domain,
        'solution': soln
    }

def solve_ide_step_data_adaptive(
    growth_map: Callable[[float], float],
    kernel_density_fn: Callable[[float], float],
    u_left: float,
    u_right: float,
    kernel_radius: float,
    dx: float,
    n_steps: int,
    tol: float,
    normalize_kernel: bool = True
) -> dict:
    """
    Solve the IDE with step initial data:
        u(x, t + 1) = ∫ k(x - y) g(u(y, t)) dy
        u(x, 0)     = { u_left  for x <= 0
                      { u_right for x > 0
    It uses an adaptive domain algorithm assuming everything outside the simulated region is constant.

    Inumpy.ts:
        growth_map: Callable[[float], float]
            Scalar growth function.
        kernel_density_fn: Callable[[float], float]
            Kernel function. Assumed to be symmetric.
        u_left: float, u_right: float
            Left and right values for the step initial data.
        kernel_radius: int
            Kernel radius.
        dx: float
            Distance between neighboring domain points.
            Must divide kernel_radius.
        n_steps: int
            Number of time steps to simulate.
        tol: float
            Tolerance for the adaptive domain algorithm.
        normalize_kernel: bool
            Whether to divide the kernel by its sum.
            Default is true so it is a probability kernel.
    Output:
        dict with two fields:
            'domain': ndarray
                1d spatial grid corresponding to the domain points,
                i.e. {-N*dx, ..., N*dx} where Nx is adaptively determined.
            'solution': ndarray
                2d array of shape (n_steps + 1, N) containing the solution surface.
    """
    from scipy import ndimage

    assert((kernel_radius / dx).is_integer())

    # set up the kernel
    n_kernel_radius = int(kernel_radius / dx)
    kernel_domain = numpy.arange(-n_kernel_radius, n_kernel_radius + 1) * dx
    kernel = kernel_density_fn(kernel_domain)
    if normalize_kernel:
        kernel /= numpy.sum(kernel)

    # the initial array has just two points
    # the left endpoint tracks the global position of the array
    # (to be glued together at the end)
    u0 = numpy.array([u_left, u_right])
    soln = []
    soln.append({
        'state': u0,
        'left_endpoint': 0,
    })
    
    for t in range(n_steps):
        
        u = soln[t]['state']
        left_endpoint = soln[t]['left_endpoint']

        u_left = u[0]
        u_right = u[-1]

        u = numpy.pad(u, (n_kernel_radius, n_kernel_radius), mode='edge')
        left_endpoint -= n_kernel_radius

        u = ndimage.convolve(growth_map(u), kernel, mode='nearest')

        u_left = growth_map(u_left)
        u_right = growth_map(u_right)

        index_chop_left  = numpy.where(numpy.abs(u - u_left) >= tol)[0][0]
        index_chop_right = numpy.where(numpy.abs(u - u_right) >= tol)[0][-1]
        
        u_chopped = u[(index_chop_left + 1):index_chop_right]
        left_endpoint += index_chop_left
        u = numpy.concat([[u_left], u_chopped, [u_right]])

        soln.append({
            'state': u,
            'left_endpoint': left_endpoint,
        })

    # wrap up: glue time points together
    imin = min(soln[t]['left_endpoint'] for t in range(len(soln)))
    imax = max(soln[t]['left_endpoint'] + len(soln[t]['state']) for t in range(len(soln))) - 1
    domain = numpy.arange(imin, imax + 1) * dx
    
    for t in range(len(soln)):
        u = soln[t]['state']
        left_endpoint = soln[t]['left_endpoint']
        u_left  = u[0]
        u_right = u[-1]
        n_pad_left  = left_endpoint - imin
        n_pad_right = imax - left_endpoint + 1 - len(u)
        soln[t]['state_padded'] = numpy.concat([[u_left]*n_pad_left, u, [u_right]*n_pad_right])
    
    soln = numpy.array([soln[t]['state_padded'] for t in range(len(soln))])

    return {
        'domain': domain,
        'solution': soln
    }

def estimate_leading_vortkamp_wavespeed(r: float, a: float, l: float, n_steps: int) -> Optional[dict]:
    """
    Given Vortkamp parameters (a, r), assuming they generate a stable cycle, this function:
    1. Computes the largest element of the cycle u_plus
    2. Solves the Riemann problem via the adaptive domain method:
        u(x, t + 1) = ∫ k(x - y) g(u(y, t)) dy
        u(x, 0)     = { u_plus for x <= 0
                      { 0      for x > 0
    3. For the given level l, approximates the sequence
        xi(t) = inf {x | |u(t, z)| < l for all z < x}
    The idea is that if p is the period of u_plus, then the average displacement,
        v(t) = (xi(t + p) - xi(t)) / p
    converges to the velocity of the leading wave.

    Inputs:
        r: float, a: float
            Vortkamp growth parameters.
        l: float
            Level at which to take successive differences.
            A necessary condition is l is smaller than every point in the stable cycle inhabited by u_plus.
        n_steps: int
            Number of steps to burn in.

    Output:
        Either None, or dict with two fields:
            'domain': ndarray
                1d spatial grid corresponding to the domain points,
                i.e. {-N*dx, ..., N*dx} where Nx is adaptively determined.
            'solution': ndarray
                2d array of shape (n_steps + 1, N) containing the solution surface.
            'c_estimate': float
                Estimate of the leading wave speed.
    """
    # first get the stable cycle
    map_data = classify_vortkamp_map(r, a)
    g = vortkamp_map(r, a)
    if map_data['cycle'] is None:
        return None
    u_plus = map_data['cycle'][0]
    p = map_data['period']
    dx = 0.01
    soln = solve_ide_step_data_adaptive(
        growth_map = g,
        kernel_density_fn = laplace_kernel(),
        u_left = u_plus,
        u_right = 0,
        kernel_radius = 4,
        dx = dx,
        n_steps = n_steps,
        tol = 0.01,
        normalize_kernel = True
    )
    domain = soln['domain']
    soln = soln['solution']
    xmin = min(domain)
    dx = dx
    tol = 0.01
    marks = []
    for t in range(len(soln)):
        mark_i = numpy.where(numpy.abs(soln[t]) >= tol)[0][-1]
        mark_x = xmin + mark_i*dx
        marks.append(mark_x)
    #print(marks)
    mark_diffs = numpy.diff(marks)
    #print(mark_diffs)
    c_estimate = numpy.mean(mark_diffs[10:])
    return {
        'domain': domain,
        'solution': soln,
        'c_estimate': c_estimate
    }

def estimate_leading_vortkamp_wavespeed_heatmap():
    # this should call the previous function multiple times and make a heatmap...
    # methinks I shall iterate over (s, a) parameters 
    # and put into a dict.
    data = {}
    S = numpy.arange(0, 1, 0.01)
    A = numpy.arange(0.01, 1, 0.01)

    S, A = numpy.meshgrid(S, A)

    from tqdm import tqdm
    for s, a in tqdm(list(numpy.nditer([S, A]))):
        s, a = float(s), float(a)
        r = s / (1 - s)
        #print(s, a)
        estimation = estimate_leading_vortkamp_wavespeed(r, a, l=0.05, n_steps=20)
        #print(estimation)
        if estimation:
            data[(s, a)] = float(estimation['c_estimate'])
    #print(data)
    # Extract coordinates and values
    import matplotlib.pyplot as plt
    coords = numpy.array(list(data.keys()))
    values = numpy.array(list(data.values()))

    # Pivot into grid form
    s_vals = numpy.unique(coords[:, 0])
    a_vals = numpy.unique(coords[:, 1])
    Z = numpy.full((len(a_vals), len(s_vals)), numpy.nan)

    for (s, a), val in data.items():
        i = numpy.where(a_vals == a)[0][0]
        j = numpy.where(s_vals == s)[0][0]
        Z[i, j] = val

    # Make heatmap
    plt.figure(figsize=(8, 6))
    im = plt.imshow(Z, origin='lower',
                    extent=[s_vals.min(), s_vals.max(), a_vals.min(), a_vals.max()],
                    aspect='auto', cmap='seismic')
    plt.colorbar(im, label='c_estimate')
    plt.xlabel('s')
    plt.ylabel('a')
    plt.title('Heatmap of Estimated Leading Vortkamp Wavespeed')
    plt.show()

estimate_leading_vortkamp_wavespeed_heatmap()

#a = 0.7
#r = 5.5
#map_data = classify_vortkamp_map(r, a)
#g = vortkamp_map(r, a)
#print(map_data)
#estimation = estimate_leading_vortkamp_wavespeed(r, a, l=0.05, n_steps=20)
#import matplotlib.pyplot as plt
#plt.imshow(estimation['solution'], aspect='auto')
#plt.show()

def estimate_vortkamp_period2_wavespeed_pipeline(r: float, a: float) -> dict:

    """
    Estimate the wavespeeds for the Vortkamp growth model in the case of a stable 2-cycle.
    Idea: solve the IDE with step initial data connecting u_plus to zero.
    Then locate intersection points with the lines u = u_plus - eps and u = eps and take the first difference.

    """
    n_transient_steps = 0
    assert(n_transient_steps % 2 == 0)
    
    try:
        map_data = classify_vortkamp_map(r, a)
    except:
        return None # TODO?
    if not map_data['period'] == 2:
        return None # TODO: what to do here
    
    g = vortkamp_map(r, a)
    u_plus = map_data['cycle'][0]
    dx = 0.01 # TODO
    
    soln = solve_ide_step_data_adaptive(
        growth_map = g,
        kernel_density_fn = laplace_kernel(), # todo
        u_left = u_plus,
        u_right = 0,
        dx = dx,
        kernel_radius = 4,
        n_steps = 20,
        tol = 0.01, # TODO
        normalize_kernel = True
    )
    domain = soln['domain']
    soln = soln['solution']
    xmin = min(domain)
    tol = 0.05
    soln = soln[n_transient_steps:,:]

    marks = []
    for t in range(len(soln)):
        mark_i = numpy.where(numpy.abs(soln[t]) >= tol)[0][-1]
        mark_x = xmin + mark_i*dx
        marks.append(mark_x)

    c1_estimate = (marks[-1] - marks[-2])
    marks = []
    for t in range(len(soln) // 2):
        mark_i = numpy.where(numpy.abs(soln[2*t] - u_plus) >= tol)[0][0]
        mark_x = xmin + mark_i*dx
        marks.append(mark_x)
    c2_estimate = (marks[-1] - marks[-2]) / 2

    return {
        'domain': domain,
        'solution': soln,
        'c1_estimate': c1_estimate,
        'c2_estimate': c2_estimate,
    }

def estimate_vortkamp_general_wavespeed_pipeline(r: float, a: float) -> dict:
    """
    Similar to the previous 
    """
    # compute largest fixedpoint
    g = vortkamp_map(r, a)
    u = [1 + 0.001]
    for t in range(100):
        u.append(g(u[-1]))
    u_plus = max(u)
    print(u_plus)

    n_transient_steps = 0
    assert(n_transient_steps % 2 == 0)
    
    
   # g = vortkamp_map(r, a)
    #u_plus = classification['u_plus']
    dx = 0.05 # TODO
    
    soln = solve_ide_step_data_adaptive(
        growth_map = g,
        kernel_density_fn = laplace_kernel(), # todo
        u_left = u_plus,
        u_right = 0,
        dx = dx,
        kernel_radius = 4,
        n_steps = 20,
        tol = 0.01, # TODO
        normalize_kernel = True
    )
    domain = soln['domain']
    soln = soln['solution']
    xmin = min(domain)
    tol = 0.01
    soln = soln[n_transient_steps:,:]

    marks = []
    for t in range(len(soln)):
        mark_i = numpy.where(numpy.abs(soln[t]) >= tol)[0][-1]
        mark_x = xmin + mark_i*dx
        marks.append(mark_x)

    c1_estimate = (marks[-1] - marks[-2])
    marks = []
    for t in range(len(soln)):
        mark_i = numpy.where(numpy.abs(soln[t] - soln[t][0]) >= tol)[0][0]
        mark_x = xmin + mark_i*dx
        marks.append(mark_x)
    c2_estimate = (marks[-1] - marks[-2])

    return {
        'domain': domain,
        'solution': soln,
        'c1_estimate': c1_estimate,
        'c2_estimate': c2_estimate,
    }


#if __name__ == '__main__':
    #a = 0.25
    #r = 0.82
    #g = vortkamp_map(r, a)
    #map_data = classify_vortkamp_map(r, a)
    #print(result)
    #result = estimate_vortkamp_period2_wavespeed_pipeline(r, a)
    #print(result)
    #print(detect_period_vortkamp(r, a))
    #print(result)
    #print(g(result['cycle'][0]))
    #print(g(g(result['cycle'][0])))
    #print(g(2))



if __name__ != '__main__':

    #print(estimate_vortkamp_general_wavespeed_pipeline(0.4, 0.1))

    a_vals = numpy.linspace(0.1, 0.8, 100)
    # pick a consistent r grid large enough to cover all runs
    r_vals = numpy.linspace(0, 2, 100)
    heatmap = numpy.full((len(a_vals), len(r_vals)), numpy.nan)
        
    for i, a in enumerate(a_vals):
        rmin = 2*a/(1-a)
        for j, r in enumerate(r_vals):
        #    if r <= rmin: 
        #        continue
            try:
                output = estimate_vortkamp_general_wavespeed_pipeline(r, a)
                c1 = output['c1_estimate']
                c2 = output['c2_estimate']
                heatmap[i, j] = abs(c1 - c2)
            except:
                pass
    
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color='lightgrey')  # grey for NaNs
    im = ax.imshow(
        heatmap,
        aspect='auto',
        origin='lower',
        extent=[r_vals[0], r_vals[-1], a_vals[0], a_vals[-1]],
        cmap=cmap,
        interpolation='none'
    )
    ax.set_xlabel('r')
    ax.set_ylabel('a')
    fig.colorbar(im, ax=ax, label='|c1 - c2|')
    a_curve = numpy.linspace(a_vals[0], a_vals[-1], 500)
    r_curve = 2 * a_curve / (1 - a_curve)

    ax.plot(r_curve, a_curve, color='red', lw=2, label=r'$r=\frac{2a}{1-a}$')
    ax.legend()
    plt.show()
    # r = 0.26
    # a = 0.1
    # classification = classify_vortkamp_parameters(r, a, root_pad=1e-4, deriv_delta=1e-4)
    # u_plus = classification['u_plus']
    # dx = 0.01
    # g = vortkamp_map(r, a)
    # soln = solve_ide_step_data_adaptive(
    #     growth_map = g,
    #     kernel_density_fn = laplace_kernel(), # todo
    #     u_left = u_plus,
    #     u_right = 0,
    #     dx = dx,
    #     kernel_radius = 4,
    #     n_steps = 100,
    #     tol = 0.01, # TODO
    #     normalize_kernel = True
    # )
    # import matplotlib.pyplot as plt
    # domain = soln['domain']
    # soln = soln['solution']
    # plt.imshow(soln, aspect='auto')
    # plt.show()
    # output = estimate_vortkamp_wavespeed_pipeline(r, a)
    # if output:
    #     #print(output)
    #     c1 = output['c1_estimate']
    #     c2 = output['c2_estimate']
    #     print(c1, c2)
    #     print(abs(c1 - c2))
    # for a in numpy.arange(0.2, 0.8, 0.1):
    #     rmin = 2*a/(1-a)
    #     for r in numpy.arange(rmin+0.05, rmin+1, 0.1):
    #         classification = classify_vortkamp_parameters(r, a, root_pad = 1e-4, deriv_delta = 1e-4)
    #         u_plus = classification['u_plus']
    #         output = estimate_vortkamp_wavespeed_pipeline(r, a)
    #         if output:
    #             #print(output)
    #             c1 = output['c1_estimate']
    #             c2 = output['c2_estimate']
    #             print(abs(c1 - c2))
    # a_vals = numpy.arange(0.2, 0.23, 0.001)
    # # pick a consistent r grid large enough to cover all runs
    # r_vals = numpy.arange(0.5, 0.75, 0.001)
    # heatmap = numpy.full((len(a_vals), len(r_vals)), numpy.nan)
        
    # for i, a in enumerate(a_vals):
    #     rmin = 2*a/(1-a)
    #     for j, r in enumerate(r_vals):
    #         if r <= rmin: 
    #             continue
    #         try:
    #             classification = classify_vortkamp_parameters(r, a, root_pad=1e-4, deriv_delta=1e-4)
            
    #             if classification['two_cycle_stable']:
    #                 output = estimate_vortkamp_wavespeed_pipeline(r, a)
    #                 if output:
    #                     c1 = output['c1_estimate']
    #                     c2 = output['c2_estimate']
    #                     heatmap[i, j] = abs(c1 - c2)
    #         except:
    #             pass
    
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # cmap = plt.cm.viridis.copy()
    # cmap.set_bad(color='lightgrey')  # grey for NaNs
    # im = ax.imshow(
    #     heatmap,
    #     aspect='auto',
    #     origin='lower',
    #     extent=[r_vals[0], r_vals[-1], a_vals[0], a_vals[-1]],
    #     cmap=cmap,
    #     interpolation='none'
    # )
    # ax.set_xlabel('r')
    # ax.set_ylabel('a')
    # fig.colorbar(im, ax=ax, label='|c1 - c2|')
    # plt.show()