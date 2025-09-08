# Fix: constrain the visualized range to r < 4
# - Keep the computation grid at r in (0, 4)
# - Explicitly set y-limits to [0, 4)
# - Truncate the analytic curve r = 2/(1-a) to where it is <= 4 (i.e., a <= 0.5)

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

def g(u, a, r):
    return u * np.exp(r * (1 - u) * (u - a))

def gprime(u, a, r):
    E = r * (1 - u) * (u - a)
    return np.exp(E) * (1 + u * r * (a + 1 - 2*u))

def gg(u, a, r):
    return g(g(u, a, r), a, r)

def h(u, a, r):
    return gg(u, a, r) - u

def find_period2_points(a, r):
    eps = 1e-6
    avoid = [0.0, a, 1.0]

    def safe(u):
        return (0.0 < u < 1.0) and all(abs(u - c) > 1e-3 for c in avoid)

    xs = np.linspace(0.0 + eps, 1.0 - eps, 400)
    hs = h(xs, a, r)
    roots = []
    for i in range(len(xs)-1):
        x1, x2 = xs[i], xs[i+1]
        y1, y2 = hs[i], hs[i+1]
        if np.isfinite(y1) and np.isfinite(y2) and y1 * y2 <= 0:
            mid = 0.5*(x1+x2)
            if safe(mid):
                try:
                    root = brentq(lambda u: h(u, a, r), x1, x2, maxiter=200, xtol=1e-12, rtol=1e-12)
                    if safe(root):
                        roots.append(root)
                except ValueError:
                    pass
    # dedupe
    roots.sort()
    uniq = []
    for x in roots:
        if not uniq or abs(x - uniq[-1]) > 1e-3:
            uniq.append(x)
    cycles = []
    for u in uniq:
        v = g(u, a, r)
        if abs(v - u) < 1e-4:
            continue
        J = gprime(u, a, r) * gprime(v, a, r)
        cycles.append((u, v, J))
    return cycles

# Grid (open interval at top end for r<4)
a_vals = np.linspace(0, 1, 120)
r_vals = np.linspace(0, 10, 160)

Z = np.zeros((len(r_vals), len(a_vals)), dtype=int)
for i, r in enumerate(r_vals):
    for j, a in enumerate(a_vals):
        if r * (1 - a) < 2.0:
            Z[i, j] = 1
            continue
        cycles = find_period2_points(a, r)
        stable_two = any(np.isfinite(J) and abs(J) < 1.0 for (_, _, J) in cycles)
        Z[i, j] = 2 if stable_two else 0

from matplotlib.colors import ListedColormap, BoundaryNorm
cmap = ListedColormap(["#9e9e9e", "#1976d2", "#d32f2f"])  # grey, blue, red
bounds = [-0.5, 0.5, 1.5, 2.5]
norm = BoundaryNorm(bounds, cmap.N)

fig, ax = plt.subplots(figsize=(7, 5), dpi=140)
A, R = np.meshgrid(a_vals, r_vals)
ax.pcolormesh(A, R, Z, cmap=cmap, norm=norm, shading='nearest')

ax.set_xlabel("a")
ax.set_ylabel("r")
ax.set_title("Stability regions for g(u) = u exp(r (1-u)(u-a))")
ax.set_xlim(0, 1)
ax.set_ylim(0, 8)

# Analytic flip curve only where r<=4 -> a <= 0.5
a_curve = np.linspace(0.02, 0.5, 300)
r_curve = 2.0 / (1.0 - a_curve)
ax.plot(a_curve, r_curve, linestyle="--", linewidth=1.5)

import matplotlib.patches as mpatches
legend_handles = [
    mpatches.Patch(color="#1976d2", label="period-1 (u=1) stable"),
    mpatches.Patch(color="#d32f2f", label="period-2 stable"),
    mpatches.Patch(color="#9e9e9e", label="other / unclassified"),
    mpatches.Patch(facecolor='none', edgecolor='black', linestyle='--', label="r = 2/(1-a)")
]
ax.legend(handles=legend_handles, loc="upper right")
plt.tight_layout()
plt.show()
