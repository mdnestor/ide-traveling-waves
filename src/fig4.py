
# Figure 4: non-spatial period map

# for a given a, determine the periodicity of the stable cycle
from integrodifference import *
import numpy as np

def VortcampMap_v3(r, a):
  return lambda u: u * np.exp((u-a) * (u-1) / np.log(r))

def guess_periodicity(a: float, r: float):

  f = VortcampMap_v3(r, a)
  u0 = 1 + 0.01
  for _ in range(1000):
    u0 = f(u0)
    #print(u0)
  tol = 0.001
  n = 1
  v0 = f(u0)
  #print(u0, v0)
  while np.abs(u0 - v0) >= tol:
      v0 = f(v0)
      n += 1
  return n

for a in np.linspace(0.3, 0.7, 5):
  for r in np.linspace(0, 1, 40):
    print(a, r, guess_periodicity(r, a))
  