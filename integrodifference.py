import numpy
from scipy import ndimage
from typing import *
import matplotlib.pyplot as plt

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
    if a < 0 or a > 1:
        raise ValueError(f"a = {a} but must be in (0, 1].")
    
    # prefill result
    data = {
        'zero_stable': None,
        'positive_attractor': None,
        'cycle': None,
        'period': None
    }
    if a == 0:
        return data

    
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

#estimate_leading_vortkamp_wavespeed_heatmap()

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

# use a continuation to estimate wavespeed
def capture_leading_wave_continuation(
    r: float,
    a: float,
    kernel_density_fn: Callable,
    T0: int,
    T1: int,
    T2: int,
    domain_radius: float,
    dx: float,
    kernel_radius: float,
    normalize_kernel=True
):
    
    assert((domain_radius / dx).is_integer())
    assert((kernel_radius / dx).is_integer())
    n_domain_radius = int(domain_radius / dx)
    domain = numpy.arange(-domain_radius, domain_radius + dx, dx)
    kernel_domain = numpy.arange(-kernel_radius, kernel_radius + dx, dx)

    kernel = kernel_density_fn(kernel_domain)
    if normalize_kernel:
        kernel /= numpy.sum(kernel)

    initial_data = lambda x: numpy.heaviside(-x, 1)
    soln = numpy.zeros((T2 + 1, len(domain)))
    soln[0] = initial_data(domain)

    r0 = a / (1 - a)
    # starting r (based on Vortkamp map)
    for t in range(T2):
        if t < T0:
            r_t = r0
        elif t < T1:
            r_t = r0 + (t - T0) / (T1 - T0) * (r - r0)
        else:
            r_t = r
        growth_map = vortkamp_map(r_t, a)
        u = soln[t]
        u = ndimage.convolve(growth_map(u), kernel, mode='nearest')
        # estimate position
        l = 0.5
        i = numpy.where(numpy.abs(u) >= l)[0][-1]
        # assume positive?
        di = i - n_domain_radius
        #print(di)
        #assume di > 0
        # chop di from left, add di to right
        if di >= 0:
            u = u[di:]
            u = numpy.concat([u, [0]*di])
        else:
            u = u[:(di)]
            u = numpy.concat([[1]*(-di), u])

        # set boundary data
        u[0] = growth_map(u[0])
        u[-1] = growth_map(u[-1])
        soln[t + 1] = u

    return {
        'domain': domain,
        'solution': soln,
        'c_estimate': di*dx
    }

def simulate_leading_wave_continuation(
    r: float,
    a: float,
    kernel_density_fn: Callable,
    T0: int,
    T1: int,
    T2: int,
    domain_radius: float,
    dx: float,
    kernel_radius: float,
    normalize_kernel=True
):
    
    assert((domain_radius / dx).is_integer())
    assert((kernel_radius / dx).is_integer())
    n_domain_radius = int(domain_radius / dx)
    domain = numpy.arange(-domain_radius, domain_radius + dx, dx)
    kernel_domain = numpy.arange(-kernel_radius, kernel_radius + dx, dx)

    kernel = kernel_density_fn(kernel_domain)
    if normalize_kernel:
        kernel /= numpy.sum(kernel)

    initial_data = lambda x: numpy.heaviside(-x, 1)
    soln = numpy.zeros((T2 + 1, len(domain)))
    soln[0] = initial_data(domain)

    r0 = a / (1 - a)
    marks = []
    # starting r (based on Vortkamp map)
    for t in range(T2):
        if t < T0:
            r_t = r0
        elif t < T1:
            r_t = r0 + (t - T0) / (T1 - T0) * (r - r0)
        else:
            r_t = r
        growth_map = vortkamp_map(r_t, a)
        u = soln[t]
        u = ndimage.convolve(growth_map(u), kernel, mode='nearest')
        # estimate position
        #l = 0.5
        #i = numpy.where(numpy.abs(u) >= l)[0][-1]
        # assume positive?
        #di = i - n_domain_radius
        #print(di)
        #assume di > 0
        # chop di from left, add di to right
        #if di >= 0:
        #    u = u[di:]
        #    u = numpy.concat([u, [0]*di])
        #else:
        #    u = u[:(di)]
        #    u = numpy.concat([[1]*(-di), u])

        # set boundary data
        # fetch leading wave mark?
        l = a / 2
        if t % 2 == 0:
            i = numpy.where(numpy.abs(u) >= l)[0][-1]
            marks.append(i)

        u[0] = growth_map(u[0])
        u[-1] = growth_map(u[-1])
        soln[t + 1] = u
    marks = [-domain_radius +mark*dx for mark in marks]
    return {
        'domain': domain,
        'solution': soln,
        'marks': marks
    }
    
# # try with some known parameters...
# T0 = 10
# T1= 20
# T2 = 40
# solution = capture_leading_wave_continuation(
#     r=3/4,
#     a=0.25,
#     kernel_density_fn=laplace_kernel(),
#     T0=T0,
#     T1=T1,
#     T2=T2,
#     domain_radius=50,
#     dx=0.01,
#     kernel_radius=4,
#     normalize_kernel=True
# )
# print(solution['c_estimate'])
# import matplotlib.pyplot as plt
# plt.imshow(solution['solution'], aspect='auto')
# plt.show()

# #for t in range(T2):
# #    plt.plot(solution['solution'][t], color=(t/500, 1-t/500, 0))

# #plt.show()
    



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


def generate_growth_map_figure():

    a = 0.5
    r = 1
    import matplotlib.pyplot as plt
    g = vortkamp_map(r, a)
    u = numpy.linspace(0, 1.2, 400)

    plt.figure(figsize=(6,4))
    plt.plot(u, g(u), label=f"a = {a}, r = {r}", color="blue")
    plt.plot(u, u, linestyle="--", color="black")
    r = 3
    g = vortkamp_map(r, a)
    plt.plot(u, g(u), label=f"a = {a}, r = {r}", color="red")

    fixed = [0, a, 1]
    plt.plot(fixed, fixed, "o", color="black", markersize=6)

    plt.grid(False)
    plt.xlabel("u(t)")
    plt.ylabel("u(t + 1)", rotation=0, labelpad=15)
    plt.legend()
    plt.xticks(fixed, ["0", "a", "1"])
    plt.yticks(fixed, ["0", "a", "1"])
    plt.gca().set_aspect("equal", adjustable="box")

    plt.savefig("figures/growthmap.png", dpi=300, bbox_inches="tight")

def generate_timeseries_figures():
    # time series for the two primary cases of interest
    import matplotlib.pyplot as plt
    for (i, params) in enumerate([(0.1, 0.23), (0.25, 0.75)]):
        a, r = params
        # find initial data
        map_data = classify_vortkamp_map(r, a)
        if not map_data['period'] == 2:
            raise ValueError()
        u_plus, u_minus = map_data['cycle']
        #u_minus = g(u_plus)
        solution = solve_ide(
            growth_map = vortkamp_map(r, a),
            kernel_density_fn=laplace_kernel(),
            initial_data=lambda x: u_plus * numpy.heaviside(-x, 1),
            xmin=-10,
            xmax=100,
            kernel_radius=4,
            dx=0.01,
            n_steps=100
        )
        # plot heatmap
        plt.figure()


        # # define the colormap function...
        # def rgb_func(u):
        #     r = u
        #     g = 1 - u
        #     b = 0.5*numpy.sin(4*numpy.pi*u) + 0.5
        #     return r, g, b

        # # Sample it
        # u = numpy.linspace(0,1,256)

        # Example usage:
        from matplotlib.colors import LinearSegmentedColormap

        points = [
            (0, (1, 1, 1)),   # red at 0.2
            #(a, (1, 0.5, 0)),   # red at 0.2
            (u_minus, (1, 0.5, 0)),
            (1, (1, 0, 0)),   # green at 0.5
            (u_plus, (0, 0, 1))    # blue at 0.8
        ]
        #print(points)
        u_min, u_max = points[0][0], points[-1][0]
        points_norm = [((u - u_min) / (u_max - u_min), c) for u, c in points]

        # Extract positions and colors
        us, cs = zip(*points_norm)

        # Build colormap
        cmap = LinearSegmentedColormap.from_list("linear_custom", list(zip(us, cs)), N=256)

        

        #cmap = LinearSegmentedColormap.from_list("my_map", colors, N=256)

        #plt.imshow(Z, cmap=cmap, origin="lower")
        #plt.text(0.1, 50, "≈ 1", color="red", fontsize=12, ha="left", va="center")
        #plt.text(0.8, 20, "≈ u+", color="blue", fontsize=12, ha="right", va="center")
        import matplotlib.patches as mpatches

        red_patch = mpatches.Patch(color=(1,0,0), label="u(x, t) ≈ 1")
        blue_patch = mpatches.Patch(color=(0,0,1), label="u(x, t) ≈ u+")
        white_patch = mpatches.Patch(color=(1, 1,1), label="u(x, t) ≈ 0")
        plt.legend(handles=[red_patch, blue_patch, white_patch], loc="upper right")
        

        im = plt.imshow(solution['solution'][::2,:], extent=[solution['domain'][0], solution['domain'][-1], len(solution['solution']), 0], cmap=cmap, aspect='auto', interpolation='none')
        cbar = plt.colorbar(im)
        #cbar.set_label("u", rotation=0)
        cbar.set_ticks([0, a, u_minus, 1, u_plus])
        cbar.set_ticklabels(["0", f"a = {a}", f"u- ≈ {u_minus:.2f}", "1", f"u+ ≈ {u_plus:.2f}"])
        plt.xlabel("location, x")
        plt.ylabel("time, t (even)", rotation=0, labelpad=20)
        plt.savefig(f"figures/example_spacetime{i}.png", dpi=300, bbox_inches="tight")

def generate_leading_trailing_heatmap():
    # idea: for a range of (a, r) values
    # calculate the leading wavefront, the trailing wavefront, etc
    a_vals = numpy.linspace(0.1, 0.6, 150)
    r_vals = numpy.linspace(0.1, 3, 150)
    #r_vals = []
    outputs = {}
    for a in a_vals:
        for r in r_vals:
            # do thing
            map_data = classify_vortkamp_map(r, a)
            if map_data['cycle'] is None:
                continue
            if map_data['period'] == 1:
                continue
            if map_data['period'] != 2:
                continue
            
            u_plus = map_data['cycle'][0]

            # set up IVP
            #g = vortkamp_map(r, a)
            #u = [1 + 0.001]
            #for t in range(100):
            #    u.append(g(u[-1]))
            #u_plus = max(u)
            #print(u_plus)

            n_transient_steps = 0
            assert(n_transient_steps % 2 == 0)
            
            
            # g = vortkamp_map(r, a)
            #u_plus = classification['u_plus']
            dx = 0.02 # TODO
            g = vortkamp_map(r, a)
            tol = min(u_plus - 1, a) / 10
            soln = solve_ide_step_data_adaptive(
                growth_map = g,
                kernel_density_fn = laplace_kernel(), # todo
                u_left = u_plus,
                u_right = 0,
                dx = dx,
                kernel_radius = 4,
                n_steps = 30,
                tol = tol, # TODO
                normalize_kernel = True
            )
            domain = soln['domain']
            soln = soln['solution']
            xmin = min(domain)
            
            soln = soln[n_transient_steps:,:]

            marks = []
            for t in range(len(soln)):
                mark_i = numpy.where(numpy.abs(soln[t]) >= tol)[0][-1]
                mark_x = xmin + mark_i*dx
                marks.append(mark_x)
            #print(marks[-1])
            c1_estimate = (marks[-1] - marks[-2])
            marks = []
            for t in range(len(soln)):
                mark_i = numpy.where(numpy.abs(soln[t] - soln[t][0]) >= tol)[0][0]
                mark_x = xmin + mark_i*dx
                marks.append(mark_x)
            c2_estimate = (marks[-1] - marks[-3]) / 2

            outputs[(r, a)] = {
                'domain': domain,
                'solution': soln,
                'c1_estimate': c1_estimate,
                'c2_estimate': c2_estimate,
            }

            # if
            #if c1_estimate - c2_estimate > 0.5:
            #    plt.imshow(soln, aspect='auto')
            #    plt.show()

    #for ((r, a), output) in outputs.items():
    #    print(r, a, output)
    

    import matplotlib.colors as mcolors

    divnorm = mcolors.TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2)
    heatmap = numpy.full((len(a_vals), len(r_vals)), numpy.nan)
    for i, a in enumerate(a_vals):
        for j, r in enumerate(r_vals):
            if (r, a) in outputs:
                heatmap[i, j] = outputs[(r, a)]['c1_estimate']

    # Plot
    plt.figure(figsize=(8,6))
    cmap = plt.get_cmap("seismic").copy()  # or whatever cmap you're using
    cmap.set_bad(color="0.5")  # "0.5" = grey
    im = plt.imshow(
        heatmap,
        origin="lower",
        extent=[r_vals[0], r_vals[-1], a_vals[0], a_vals[-1]],
        aspect="auto",
        cmap=cmap,
        norm=divnorm
    )
    plt.colorbar(im)#, label="c1 estimate")
    plt.xlabel("r")
    plt.ylabel("a")
    #plt.title("Heatmap of c1_estimate")
    plt.savefig(f"figures/c1est.png", dpi=300, bbox_inches="tight")

    heatmap = numpy.full((len(a_vals), len(r_vals)), numpy.nan)
    for i, a in enumerate(a_vals):
        for j, r in enumerate(r_vals):
            if (r, a) in outputs:
                heatmap[i, j] = outputs[(r, a)]['c2_estimate']

    # Plot
    plt.figure(figsize=(8,6))
    im = plt.imshow(
        heatmap,
        origin="lower",
        extent=[r_vals[0], r_vals[-1], a_vals[0], a_vals[-1]],
        aspect="auto",
        cmap=cmap,
        norm=divnorm
    )
    plt.colorbar(im)#, label="c2 estimate")
    plt.xlabel("r")
    plt.ylabel("a", rotation=0)
    #plt.title("Heatmap of c2_estimate")
    plt.savefig(f"figures/c2est.png", dpi=300, bbox_inches="tight")


    #divnorm = mcolors.TwoSlopeNorm(vmin=0, vcenter=0, vmax=2)

    heatmap = numpy.full((len(a_vals), len(r_vals)), numpy.nan)
    for i, a in enumerate(a_vals):
        for j, r in enumerate(r_vals):
            if (r, a) in outputs:
                heatmap[i, j] = outputs[(r, a)]['c1_estimate'] - outputs[(r, a)]['c2_estimate']

    # Plot
    plt.figure(figsize=(8,6))
    cmap = plt.get_cmap("Reds").copy()  # or whatever cmap you're using
    cmap.set_bad(color="0.5")  # "0.5" = grey
    im = plt.imshow(
        heatmap,
        origin="lower",
        extent=[r_vals[0], r_vals[-1], a_vals[0], a_vals[-1]],
        aspect="auto",
        cmap=cmap,
        #norm=divnorm
    )
    plt.colorbar(im)#, label="c2 estimate")
    plt.xlabel("r")
    plt.ylabel("a")
    #plt.title("Heatmap of c diff")
    #plt.show()
    plt.savefig(f"figures/cdiff.png", dpi=300, bbox_inches="tight")
    # make c1 heatmap

def generate_ivp_example_2():
    # time series for the two primary cases of interest
    import matplotlib.pyplot as plt
    a = 0.25
    r = 0.75
    map_data = classify_vortkamp_map(r, a)
    if not map_data['period'] == 2:
        raise ValueError()
    u_plus, u_minus = map_data['cycle']
    solution = solve_ide(
        growth_map = vortkamp_map(r, a),
        kernel_density_fn=laplace_kernel(),
        initial_data=lambda x: numpy.heaviside(-x, 1),
        xmin=-100,
        xmax=100,
        kernel_radius=4,
        dx=0.01,
        n_steps=100
    )
    # plot heatmap
    plt.figure()


    # # define the colormap function...
    # def rgb_func(u):
    #     r = u
    #     g = 1 - u
    #     b = 0.5*numpy.sin(4*numpy.pi*u) + 0.5
    #     return r, g, b

    # # Sample it
    # u = numpy.linspace(0,1,256)

    # Example usage:
    from matplotlib.colors import LinearSegmentedColormap

    points = [
        (0, (1, 1, 1)),   # red at 0.2
        #(a, (1, 0.5, 0)),   # red at 0.2
        (u_minus, (1, 0.5, 0)),
        (1, (1, 0, 0)),   # green at 0.5
        (u_plus, (0, 0, 1))    # blue at 0.8
    ]
    #print(points)
    u_min, u_max = points[0][0], points[-1][0]
    points_norm = [((u - u_min) / (u_max - u_min), c) for u, c in points]

    # Extract positions and colors
    us, cs = zip(*points_norm)

    # Build colormap
    cmap = LinearSegmentedColormap.from_list("linear_custom", list(zip(us, cs)), N=256)

    

    #cmap = LinearSegmentedColormap.from_list("my_map", colors, N=256)

    #plt.imshow(Z, cmap=cmap, origin="lower")
    #plt.text(0.1, 50, "≈ 1", color="red", fontsize=12, ha="left", va="center")
    #plt.text(0.8, 20, "≈ u+", color="blue", fontsize=12, ha="right", va="center")
    import matplotlib.patches as mpatches

    red_patch = mpatches.Patch(color=(1,0,0), label="u(x, t) ≈ 1")
    blue_patch = mpatches.Patch(color=(0,0,1), label="u(x, t) ≈ u+")
    white_patch = mpatches.Patch(color=(1, 1,1), label="u(x, t) ≈ 0")
    plt.legend(handles=[red_patch, blue_patch, white_patch], loc="upper right")
    

    im = plt.imshow(solution['solution'][::2,:], extent=[solution['domain'][0], solution['domain'][-1], len(solution['solution']), 0], cmap=cmap, aspect='auto', interpolation='none')
    cbar = plt.colorbar(im)
    #cbar.set_label("u", rotation=0)
    cbar.set_ticks([0, a, u_minus, 1, u_plus])
    cbar.set_ticklabels(["0", f"a = {a}", f"u- ≈ {u_minus:.2f}", "1", f"u+ ≈ {u_plus:.2f}"])
    plt.xlabel("location, x")
    plt.ylabel("time, t (even)", rotation=0, labelpad=20)
    plt.savefig(f"figures/example_spacetime2.png", dpi=300, bbox_inches="tight")

def continuation_problem():
    a = 0.1
    r = 0.23
    #a = 0.4
    #r = 1.5
    map_data = classify_vortkamp_map(r, a)
    if not map_data['period'] == 2:
        raise ValueError()
    u_plus, u_minus = map_data['cycle']
    T0 = 10
    T1 = 20
    T2 = 100
    solution = simulate_leading_wave_continuation(
        r, a, kernel_density_fn=laplace_kernel(), T0=T0, T1=T1, T2=T2,
        domain_radius=80,
        dx=0.01,
        kernel_radius=4,
    )
    # generate marks?
    u = solution['solution']
    plt.plot(numpy.diff(solution['marks']))
    plt.show()

    plt.figure()


    # # define the colormap function...
    # def rgb_func(u):
    #     r = u
    #     g = 1 - u
    #     b = 0.5*numpy.sin(4*numpy.pi*u) + 0.5
    #     return r, g, b

    # # Sample it
    # u = numpy.linspace(0,1,256)

    # Example usage:
    from matplotlib.colors import LinearSegmentedColormap

    import matplotlib.colors as mcolors

    points = [
        (0, (1, 1, 1)),
        (u_minus, (1, 0.5, 0)),
        (1, (1, 0, 0)),
        (u_plus, (0, 0, 1)),   # <-- top endpoint is u_plus
    ]

    # Normalize positions onto [0,1]
    u_min, u_max = 0, u_plus
    points_norm = [((u - u_min)/(u_max - u_min), c) for u, c in points]
    us, cs = zip(*points_norm)
    cmap = LinearSegmentedColormap.from_list("linear_custom", list(zip(us, cs)), N=256)

    # Important: tell matplotlib how to map your data values into [0,1]
    norm = mcolors.Normalize(vmin=0, vmax=u_plus)

    

    #cmap = LinearSegmentedColormap.from_list("my_map", colors, N=256)

    #plt.imshow(Z, cmap=cmap, origin="lower")
    #plt.text(0.1, 50, "≈ 1", color="red", fontsize=12, ha="left", va="center")
    #plt.text(0.8, 20, "≈ u+", color="blue", fontsize=12, ha="right", va="center")
    import matplotlib.patches as mpatches

    red_patch = mpatches.Patch(color=(1,0,0), label="u(x, t) ≈ 1")
    blue_patch = mpatches.Patch(color=(0,0,1), label="u(x, t) ≈ u+")
    white_patch = mpatches.Patch(color=(1, 1,1), label="u(x, t) ≈ 0")
    plt.legend(handles=[red_patch, blue_patch, white_patch], loc="upper right")
    

    # im = plt.imshow(solution['solution'][::2,:], extent=[solution['domain'][0], solution['domain'][-1], len(solution['solution']), 0], cmap=cmap, norm=norm, aspect='auto', interpolation='none')
    # cbar = plt.colorbar(im)
    # #cbar.set_label("u", rotation=0)
    # cbar.set_ticks([0, a, u_minus, 1, u_plus])
    # cbar.set_ticklabels(["0", f"a = {a}", f"u- ≈ {u_minus:.2f}", "1", f"u+ ≈ {u_plus:.2f}"])
    # plt.xlabel("location, x")
    # plt.ylabel("time, t (even)", rotation=0, labelpad=20)
    # plt.savefig(f"figures/continuation.png", dpi=300, bbox_inches="tight")

    nt = len(solution['solution'])  # number of time steps you plotted
    times = numpy.linspace(0, T2, nt)

    # Build r(t): constant at 0.2 until T0, linear up to 0.8 by T1, then constant
    rm = a/(1-a)
    r = r
    r_vals = numpy.piecewise(
        times,
        [times <= T0, (times > T0) & (times <= T1), times > T1],
        [
            rm,
            lambda t: rm + (r - rm) * (t - T0) / (T1 - T0),
            r,
        ],
    )

    fig, (ax_main, ax_r) = plt.subplots(
        1, 2, gridspec_kw={'width_ratios': [12, 4]}, sharey=True, figsize=(10, 6)
    )
    ax_r.set_ylim(0, nt)

    # --- main space–time plot ---
    im = ax_main.imshow(
        solution['solution'][::2, :],
        extent=[solution['domain'][0], solution['domain'][-1], nt, 0],
        cmap=cmap,
        norm=norm,
        aspect='auto',
        interpolation='none',
    )
    cbar = fig.colorbar(im, ax=ax_main)
    cbar.set_ticks([0, a, u_minus, 1, u_plus])
    cbar.set_ticklabels(["0", f"a = {a}", f"u- ≈ {u_minus:.2f}", "1", f"u+ ≈ {u_plus:.2f}"])
    ax_main.set_xlabel("location, x")
    ax_main.set_ylabel("time, t (even)", rotation=0, labelpad=40)

    # --- r(t) plot ---
    ax_r.plot(r_vals, numpy.arange(nt), color="black")
    ax_r.set_xlim(0, r*1.2)
    ax_r.set_xticks([rm, r])
    ax_r.set_xticklabels([f"r_m(a) ≈ {rm:.2f}", f"r ≈ {r:.2f}"])
    ax_r.set_xlabel("growth rate, r(t)")
    ax_r.invert_yaxis()  # so time increases downward, consistent with main plot

    fig.tight_layout()
    plt.savefig("figures/continuation_with_r.png", dpi=300, bbox_inches="tight")


    # generate the time series plots?
    plt.figure()
    #plt.plot(solution['solution'][10])
    #plt.plot(solution['solution'][20])
    plt.plot(solution['solution'][T2])
    plt.savefig("figures/continuation_timeseries.png", dpi=300, bbox_inches="tight")
    





if __name__ == "__main__":

    #generate_growth_map_figure()

    #generate_timeseries_figures()

    #generate_leading_trailing_heatmap()
    
    # test some plots...
    #generate_ivp_example_2()
    continuation_problem()
    # a = 0.1
    # #r = 0.62
    # for r in numpy.arange(0, 2, 0.01):
    #     data = classify_vortkamp_map(r, a)
        
    #     if not data['period'] == 2:
    #         continue
    #     u_plus = data['cycle'][0]
    #     soln = solve_ide(
    #         growth_map=vortkamp_map(r,a),
    #         kernel_density_fn=laplace_kernel(),
    #         initial_data=lambda x:numpy.heaviside(-x,1),#*u_plus,
    #         xmin=-5,
    #         xmax=20,
    #         kernel_radius=8,
    #         dx=0.002,
    #         n_steps=30,
    #     )
    #     plt.imshow(soln['solution'],aspect='auto')
    #     plt.show()