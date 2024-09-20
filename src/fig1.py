import os
import matplotlib.pyplot as plt

from integrodifference import *

dir = os.path.dirname(os.path.abspath(__file__))

a = 0.25
r = 1.0

print("instability at u=1 satisfied:", r < 2/(1-a))


f = VortcampMap(a=a, r=r)

U = np.arange(0, 1.5, 0.01)
plt.plot(U, f(U), color="black")
plt.plot(U, U, color="black", linestyle=":")

plt.gca().set_aspect('equal')

plt.savefig(os.path.join(dir, '../figures/fig1a.png'))