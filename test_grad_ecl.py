# -*- coding: utf-8 -*-

import os
os.chdir("/home/rosaell/Documents/2017_2018/Stage/stageL3_photoclinometrie")

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from libSFS import *

plt.ion()
plt.show()
plt.rcParams["image.cmap"]="gray"

nx = 32
ny = 32
N = nx * ny

pi = np.pi

theta = pi / 9
phi = 0
theta_obs = 0
phi_obs = 0
lV = direction_eclairement((theta, phi), (theta_obs, phi_obs))
(alpha,beta,gamma) = lV
x = np.linspace(-nx/2,nx/2-1,nx)
y = np.linspace(-ny/2,ny/2-1,ny)
X,Y = np.meshgrid(y,x)

pente1 = gamma * (1-gamma**2)**.5 + (1-gamma**2)*(1+gamma*2)**.5 / gamma
pente2 = gamma * (1-gamma**2)**.5 - (1-gamma**2)*(1+gamma*2)**.5 / gamma

print(pente1)
print(pente2)


Z1 = pente1 * Y

E1 = eclairement(Z1,lV,np.gradient)

fig = plt.figure(1)
ax = fig.gca(projection='3d')
ax.plot_surface(X,Y,Z1,rstride=2,cstride=2,linewidth=1,color='b')

plt.figure()
plt.imshow(E1,vmin=0,vmax=1)

if pente2 > 0:

	Z2 = pente2 * Y

	E2 = eclairement(Z2,lV,np.gradient)

	fig = plt.figure(1)
	ax = fig.gca(projection='3d')
	ax.plot_surface(X,Y,Z2,rstride=2,cstride=2,linewidth=1,color='r')

	plt.figure()
	plt.imshow(E2,vmin=0,vmax=1)


plt.show()