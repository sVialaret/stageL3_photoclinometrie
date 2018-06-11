# -*- coding: utf-8 -*-

from ad import adnumber
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from numpy.linalg import solve
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from libSFS import *
import numpy.random as rd

nx = 32
ny = 32
N = nx * ny

theta = np.pi / 3
phi = np.pi / 3
theta_obs = 0
phi_obs = 0
lV = direction_eclairement((theta, phi), (theta_obs, phi_obs))
alpha, beta, gamma = lV

a = 1.0

eps = 0

dx = 1
dy = 1

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

Lap = -(Dx.T.dot(Dx) + Dy.T.dot(Dy))

# Dx = adnumber(Dx)
# Dy = adnumber(Dy)
# Lap = adnumber(Lap)

M = alpha * Dx + beta * Dy + eps * Lap

def grad(U): return (Dx.dot(U), Dy.dot(U))

# Z_mat = generer_surface(Nx=nx, Ny=ny, forme=('volcan',10,10,0.5,0.2,0.5), reg = 0)
# Z_mat = generer_surface(Nx=nx, Ny=ny, forme=('trap',7,10,1,0.5), reg=0)
Z_mat = generer_surface(Nx=nx, Ny=ny, forme=('cone', 10, 0.00001), reg=0)
# Z_mat = generer_surface(Nx=nx, Ny=ny, forme=('plateau',20,20,1), reg = 0)

Z = np.reshape(Z_mat, N)

E = eclairement(Z, lV, grad)
E_cp = E.copy()
E_cp_mat = np.reshape(E_cp, (nx, ny))

# F = lambda X : E*np.sqrt(1 + (Dx.dot(X))**2 + (Dy.dot(X))**2) - (alpha * Dx.dot(X) + beta * Dy.dot(X) + gamma + eps * Lap.dot(X))
# jacF = lambda X : ((Dx.T * (Dx.dot(X) * E)/np.sqrt(1 + Dx.dot(X)**2 + Dy.dot(X)**2)).T + (Dy.T * (Dy.dot(X) * (E - eps * Lap.dot(X)))).T) - alpha * Dx - beta * Dy

F = lambda X : spsolve(M, (E/a) * np.sqrt(1 + Dx.dot(X)**2 + Dy.dot(X)**2) - gamma)
# jacF = lambda X : (Dx.T * (Dx.dot(X) * E)/np.sqrt(1 + Dx.dot(X)**2 + Dy.dot(X)**2)).T + (Dy.T * (Dy.dot(X) * (E - eps * Lap.dot(X)))).T
# jacF = lambda X : spsolve(M,(Dx.T * (Dx.dot(X) * E)/np.sqrt(1 + Dx.dot(X)**2 + Dy.dot(X)**2)).T + (Dy.T * (Dy.dot(X) * (E - eps * Lap.dot(X)))).T) / a - np.eye(len(X))


def jacF(X):
	tmp = sp.lil_matrix((N,N))
	indx1, indx2 = Dx.nonzero()
	indy1, indy2 = Dy.nonzero()
	dxx = Dx.dot(X)
	dyx = Dy.dot(X)
	for i,j in zip(indx1,indx2):
		tmp[i,j] = tmp[i,j] + (E[i] / a) * Dx[i,j] * dxx[i] / ((1 + dxx[i] ** 2 + dyx[i] ** 2))
	for i,j in zip(indy1,indy2):
		tmp[i,j] = tmp[i,j] + (E[i] / a) * Dy[i,j] * dyx[i] / ((1 + dxx[i] ** 2 + dyx[i] ** 2))
	return spsolve(M,tmp)

Z0 = np.zeros(N)
Z1 = Z + rd.randn(N) / 100
# Z1 = np.zeros(N)

Z1_mat = np.reshape(Z1, (nx, ny))

# fig = plt.figure(1000000)
# ax = fig.gca(projection='3d')
# ax.plot_surface(X, Y, Z1_mat, rstride=5, cstride=5, linewidth=1)
# ax.plot_wireframe(X, Y, Z_mat, rstride=5, cstride=5, linewidth=1, color='r')

nb_it = 10

for i in range(nb_it):
	print(i)
	Z0 = Z1
	Z1 = Z0 + spsolve(jacF(Z0),-F(Z0))
	Z1_mat = np.reshape(Z1, (nx, ny))

	E_appr = eclairement(Z1, lV, grad)
	E_appr_mat = np.reshape(E_appr, (nx, ny))

	print(comparer_eclairement(E_cp, E_appr))


fig = plt.figure(10*i)
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z1_mat, rstride=5, cstride=5, linewidth=1)
ax.plot_wireframe(X, Y, Z_mat, rstride=5, cstride=5, linewidth=1, color='r')

plt.show()