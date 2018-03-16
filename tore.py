# -*- coding: utf-8 -*-

import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from time import clock
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from libSFS import *

nx = 64
ny = 64
N = nx * ny

theta = np.pi / 3
phi = np.pi / 3
theta_obs = 0
phi_obs = 0
lV = direction_eclairement((theta, phi), (theta_obs, phi_obs))

alpha, beta, gamma = lV

eps = 0.1

dx = 1
dy = 1

nb_it = 1

x = np.linspace(-nx / 2, nx / 2 - 1, nx)
y = np.linspace(-ny / 2, ny / 2 - 1, ny)
X, Y = np.meshgrid(y, x)

# gradient selon x

M_dx = sp.lil_matrix((N + 1, N + 1))
M_dx.setdiag(-1)
M_dx.setdiag(1, ny)
M_dx.setdiag(1, -((nx - 1) * ny - 1))
M_dx[-1, -1] = 0
M_dx[(nx - 1) * ny, -1] = 0
M_dx[-1, ny + 1] = 0
M_dx = (1 / dx) * M_dx
M_dx = M_dx.tocsr()

# gradient selon y

M_dy = sp.lil_matrix((N + 1, N + 1))
M_dy.setdiag(-1)
M_dy.setdiag(1, 1)
M_dy[-2, 0] = 1
M_dy[-1, -1] = 0
M_dy[-2, -1] = 0
M_dy = (1 / dy) * M_dy
M_dy = M_dy.tocsr()

# laplacien

M_lap = M_dx.T.dot(M_dx) + M_dy.T.dot(M_dy)

# matrice finale

M = eps * M_lap + alpha * M_dx + beta * M_dy

# remplissage de la derniere ligne

M[-1, 0] = 1
M[-1, -1] = -1

print("matrice formee")

grad = lambda U : (M_dx.dot(U), M_dy.dot(U))

# Z,E,V = generer_surface(Nx=nx, Ny=ny, forme=('volcan',20,20,0.5,0.2,0.5), reg = 0, lV=(theta,phi),obV=(0,0))
# Z,E,V = generer_surface(Nx=nx, Ny=ny, forme=('trap',80,80,1,0.5), reg=0, lV=(theta,phi),obV=(0,0))
Z_mat = generer_surface(Nx=nx, Ny=ny, forme=('cone', 20, 1), reg=0)
# Z,E,V = generer_surface(Nx=nx, Ny=ny, forme=('plateau',20,20,1), reg = 0, lV=(theta,phi),obV=(0,0))

Z = np.reshape(Z_mat, N)

Z = np.concatenate((Z, [0]))

E = eclairement(Z, lV, grad)

E_cp = E.copy()
E_cp_mat = np.reshape(E_cp[:-1], (nx, ny))

V = np.sum(Z[:-1])

print("surface generee")

compt = 0

Z_appr = np.zeros(N + 1)

plt.figure(-5)
plt.imshow(E_cp_mat, cmap='gray')

while compt < nb_it:

	compt += 1

	Z_gradx, Z_grady = grad(Z_appr)
	corr = np.sqrt(1 + Z_gradx**2 + Z_grady**2)
	E = E_cp * corr - gamma
	E[-1] = 0
	    
	# F = np.reshape(E, N)

	Z_appr = spsolve(M,E)
	E_appr = eclairement(Z_appr, lV, grad)
	Z_appr_mat = np.reshape(Z_appr[:-1], (nx, ny))
	E_appr_mat = np.reshape(E_appr[:-1], (nx, ny))

	fig = plt.figure(10 * compt)
	ax = fig.gca(projection='3d')
	ax.plot_surface(X, Y, Z_appr_mat, rstride=2, cstride=2, linewidth=1)
	ax.plot_wireframe(X, Y, Z_mat, rstride=2, cstride=2, linewidth=1, color='r')

	plt.figure(10 * compt + 1)
	plt.imshow(E_appr_mat, cmap='gray')

	print(comparer_eclairement(E_cp[:-1], E_appr[:-1]))
	V_appr = np.sum(Z_appr[:-1])
	print(V, V_appr, np.abs(V - V_appr) / V)

plt.show()
