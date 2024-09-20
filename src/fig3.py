
# Figure 2: dynamical stabilization with Ricker growth function


import os
import matplotlib.pyplot as plt
from scipy import stats

dir = os.path.dirname(os.path.abspath(__file__))

from integrodifference import *

f = VortcampMap_v2(a=0.25, r=3.0)
k = LaplaceKernel()


U = np.arange(0, 1.5, 0.01)
plt.plot(U, f(U), color="black")
plt.plot(U, U, color="black", linestyle=":")

plt.gca().set_aspect('equal')

plt.savefig(os.path.join(dir, '../figures/fig3_growth1.png'))

# estimate fixed point
u0 = 1 + 0.01
for _ in range(1000):
    u0 = f(u0)
u0 = max(u0, f(u0))

soln = simulate_integrodifference(
    growth_func = f,
    kernel_func = k,
    initial_cond = lambda x: u0 * np.heaviside(x, 1),
    n_steps=100,
    xmin=-100.0,
    xmax=100.0,
    dx=0.1,
    kernel_radius=4.0,
)

plt.imshow(soln, aspect="auto", interpolation="nearest")

plt.xticks(np.linspace(0, soln.shape[1], 11), np.linspace(-100, 100, 11).astype(int))
plt.yticks(np.linspace(0, soln.shape[0] - 1, 11), np.linspace(0, soln.shape[0] - 1, 11).astype(int))

plt.colorbar()
plt.xlabel("x")
plt.ylabel("t", rotation=0)

plt.savefig(os.path.join(dir, '../figures/fig3a.png'))

plt.clf()

tmax = soln.shape[0]
for t in range(0, tmax, 10):
  plt.plot(soln[t], color=(t/tmax, 0, 1 - t/tmax))

plt.xticks(np.linspace(0, soln.shape[1], 11), np.linspace(-100, 100, 11).astype(int))
plt.xlabel("x")
plt.ylabel("u_t(x)", rotation=0)

plt.savefig(os.path.join(dir, '../figures/fig3b.png'))


plt.clf()

tmax = soln.shape[0]
for t in range(5, tmax, 10):
  plt.plot(soln[t], color=(t/tmax, 0, 1 - t/tmax))

plt.xticks(np.linspace(0, soln.shape[1], 11), np.linspace(-100, 100, 11).astype(int))

plt.xlabel("x")
plt.ylabel("u_t(x)", rotation=0)

plt.savefig(os.path.join(dir, '../figures/fig3c.png'))


plt.clf()

f = VortcampMap_v2(a=0.1, r=2.3)
k = LaplaceKernel()

U = np.arange(0, 1.5, 0.01)
plt.plot(U, f(U), color="black")
plt.plot(U, U, color="black", linestyle=":")

plt.gca().set_aspect('equal')

plt.savefig(os.path.join(dir, '../figures/fig3_growth2.png'))

# estimate fixed point
u0 = 1 + 0.01
for _ in range(1000):
    u0 = f(u0)
u0 = max(u0, f(u0))

soln = simulate_integrodifference(
    growth_func = f,
    kernel_func = k,
    initial_cond = lambda x: u0 * np.heaviside(x, 1),
    n_steps=100,
    xmin=-100.0,
    xmax=100.0,
    dx=0.1,
    kernel_radius=4.0,
)

plt.imshow(soln, aspect="auto", interpolation="nearest")

plt.xticks(np.linspace(0, soln.shape[1], 11), np.linspace(-100, 100, 11).astype(int))
plt.yticks(np.linspace(0, soln.shape[0] - 1, 11), np.linspace(0, soln.shape[0] - 1, 11).astype(int))

plt.colorbar()
plt.xlabel("x")
plt.ylabel("t", rotation=0)

plt.savefig(os.path.join(dir, '../figures/fig3d.png'))

plt.clf()

tmax = soln.shape[0]
for t in range(0, tmax, 10):
  plt.plot(soln[t], color=(t/tmax, 0, 1 - t/tmax))

plt.xticks(np.linspace(0, soln.shape[1], 11), np.linspace(-100, 100, 11).astype(int))
plt.xlabel("x")
plt.ylabel("u_t(x)", rotation=0)

plt.savefig(os.path.join(dir, '../figures/fig3e.png'))


plt.clf()

tmax = soln.shape[0]
for t in range(5, tmax, 10):
  plt.plot(soln[t], color=(t/tmax, 0, 1 - t/tmax))

plt.xticks(np.linspace(0, soln.shape[1], 11), np.linspace(-100, 100, 11).astype(int))

plt.xlabel("x")
plt.ylabel("u_t(x)", rotation=0)

plt.savefig(os.path.join(dir, '../figures/fig3f.png'))