# -*- coding: utf-8 -*-

import scipy.sparse as sp
from scipy.sparse.linalg import spsolve, eigs, norm
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from libSFS import *
import scipy.misc as io
from numpy.linalg import solve

nx = 212
ny = 251
N = nx * ny

theta = np.pi / 4
phi = np.pi / 2.7
theta_obs = 0
phi_obs = 0
lV = direction_eclairement((theta, phi), (theta_obs, phi_obs))
alpha, beta, gamma = lV

eps = 5

dx = 1
dy = 1

nb_it = 10



x = np.linspace(0, np.pi, nx)
y = np.linspace(0, np.pi, ny)
X, Y = np.meshgrid(y, x)

Dx_ii = sp.lil_matrix((ny, ny))
Dx_ii.setdiag(-1.0 / dx)
Dx_ii.setdiag(1.0 / dx, 1)
Dx_ii = Dx_ii.tocsr()

Kx_ii = sp.eye(nx)
Kx_ii = Kx_ii.tocsr()

Dx = sp.kron(Kx_ii, Dx_ii)
Dx = Dx.tocsr()

Dy_ii = sp.eye(ny) * (-1 / dy)
Dy_ii = Dy_ii.tocsr()

Ky_ii = sp.eye(nx)
Ky_ii = Ky_ii.tocsr()

Dy_ij = sp.eye(ny) / dy
Dy_ij = Dy_ij.tocsr()

Ky_ij = sp.lil_matrix((nx, nx))
Ky_ij.setdiag(1, 1)
Ky_ij = Ky_ij.tocsr()

Dy = sp.kron(Ky_ii, Dy_ii) + sp.kron(Ky_ij, Dy_ij)
Dy = Dy.tocsr()


# Lap_ii = sp.lil_matrix((ny, ny))
# Lap_ii.setdiag(-2 * (1 / dx ** 2 + 1 / dy ** 2))
# Lap_ii.setdiag(1 / dx ** 2, 1)
# Lap_ii.setdiag(1 / dx ** 2, -1)
# Lap_ii = Lap_ii.tocsr()

# Klap_ii = sp.eye(nx)
# Klap_ii = Klap_ii.tocsr()

# Lap_ij = sp.eye(ny) / (dy ** 2)
# Lap_ij = Lap_ij.tocsr()

# Klap_ij = sp.lil_matrix((nx, nx))
# Klap_ij.setdiag(1, 1)
# Klap_ij.setdiag(1, -1)
# Klap_ij = Klap_ij.tocsr()

# Lap = sp.kron(Klap_ii, Lap_ii) + sp.kron(Klap_ij, Lap_ij)

Lap = -(Dx.T.dot(Dx) + Dy.T.dot(Dy))
Lap = Lap.tocsr()

M = alpha * Dx + beta * Dy + eps * Lap
M = M.tocsr()


def grad(U): return (Dx.dot(U), Dy.dot(U))

# # Z_mat = generer_surface(Nx=nx, Ny=ny, forme=('volcan',50,50,0.5,0.2,0.5), reg = 0)
# # Z_mat = generer_surface(Nx=nx, Ny=ny, forme=('trap',30,100,1,0.5), reg=0)
# Z_mat = generer_surface(Nx=nx, Ny=ny, forme=('cone', 50, 10), reg=0)
# # Z_mat = generer_surface(Nx=nx, Ny=ny, forme=('plateau',20,20,1), reg = 0)



# Z = np.reshape(Z_mat, N)

# E = eclairement(Z, lV, grad)
# # E = bruit_gaussien(E, 0.2)
# # E = bruit_selpoivre(E, 0.01)

# # E = simul_camera(E, (nx, ny), 6)


# E_cp = E.copy()
# E_cp_mat = np.reshape(E_cp, (nx, ny))

# V = np.sum(Z)

# print(2**.5*np.max(np.abs(E)) / ((alpha + beta + (2*eps/(np.pi/128)))))

E_cp_mat = io.imread('crop.png', 'L')
E_cp_mat = (E_cp_mat - np.min(E_cp_mat))/(np.max(E_cp_mat) - np.min(E_cp_mat))
E = np.reshape(E_cp_mat, N)
E_cp = E.copy()

compt = 0

Z_appr = np.zeros(N)
# Z_appr = Z

err_T = []

plt.figure(-5)
plt.imshow(E_cp_mat, cmap='gray')

while compt < nb_it:
	eps = eps * 0.7
	M = alpha * Dx + beta * Dy + eps * Lap

	compt += 1

	Z_gradx, Z_grady = grad(Z_appr)
	corr = np.sqrt(1 + Z_gradx**2 + Z_grady**2)
	E = E_cp * corr - gamma

	# for i in range(ny):
	#     E[i] = 0
	#     E[-(i + 1)] = 0

	# for j in range(nx):
	#     E[j * ny] = 0
	#     E[(j + 1) * ny - 1] = 0
	# if compt == 5:
	# 	M = alpha * Dx + beta * Dy
	Z_appr = spsolve(M.T.dot(M), M.T.dot(E).T)

	# print(np.max(np.abs(M.T.dot(M).dot(Z_appr)-M.T.dot(E).T)))

	E_appr = eclairement(Z_appr, lV, grad)
	Z_appr_mat = np.reshape(Z_appr, (nx, ny))
	E_appr_mat = np.reshape(E_appr, (nx, ny))

	fig = plt.figure(10 * compt)
	ax = fig.gca(projection='3d')
	ax.plot_surface(X, Y, Z_appr_mat, rstride=5, cstride=5, linewidth=1)
	# ax.plot_wireframe(X, Y, Z_mat, rstride=5, cstride=5, linewidth=1, color='r')


	# fig = plt.figure(10 * compt + 1)
	# ax = fig.gca(projection='3d')
	# ax.plot_surface(X,Y,Z_appr_n - Z_appr,rstride=2,cstride=2,linewidth=1)

	# plt.figure(10 * compt + 2)
	# plt.imshow(E_appr_mat, cmap='gray')

	# plt.figure()
	# plt.imshow(np.abs(E_appr_mat - E_cp_mat), cmap='gray', vmin = 0, vmax = 1)

	# V_appr = np.sum(Z_appr)
	# err = np.abs(V - V_appr) / V
	# err_T.append(err)

	# print(comparer_eclairement(E_cp, E_appr))
	# print(err)
	print(compt)
	# print(np.sum((Z - Z_appr)**2)**.5/np.sum(Z))

plt.show()