# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.ion()
plt.show()
plt.rcParams["image.cmap"]="gray"

nx = 32
ny = 32
N = nx * ny

x = np.linspace(-nx/2,nx/2-1,nx)
y = np.linspace(-ny/2,ny/2-1,ny)
X,Y = np.meshgrid(y,x)

Z = np.abs(X) + np.abs(Y)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X,Y,Z,rstride=5,cstride=5,linewidth=1)
plt.show()