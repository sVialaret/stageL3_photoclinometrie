# -*- coding: utf-8 -*-

import scipy.sparse as sp
from scipy.sparse.linalg import spsolve, eigs
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from libSFS import *
import scipy.misc as io

nx = 256
ny = 256
N = nx * ny

theta = np.pi / 3
phi = np.pi / 4
theta_obs = 0
phi_obs = 0
lV = direction_eclairement((theta, phi), (theta_obs, phi_obs))

alpha, beta, gamma = lV

eps = 0.1

dx = 1.0
dy = 1.0

nb_it = 1

CB = sp.lil_matrix(N)
# for i in range(ny):
#     CB[i] = 0
#     CB[-(i + 1)] = 0

# for j in range(nx):
#     CB[j * ny] = 0
#     CB[(j + 1) * ny - 1] = 0


x = np.linspace(-nx / 2, nx / 2 - 1, nx)
y = np.linspace(-ny / 2, ny / 2 - 1, ny)
X, Y = np.meshgrid(y, x)

K_id = sp.csr_matrix((nx, nx))
K_id[0, 0] = 1
K_id[-1, -1] = 1

Dx_ii = sp.lil_matrix((ny, ny))
Dx_ii.setdiag(-1.0 / dx)
Dx_ii.setdiag(1.0 / dx, 1)
Dx_ii = Dx_ii.tocsr()
# Dx_ii[0, 0] = 1
# Dx_ii[0, 1] = 0
# Dx_ii[-1, -1] = 1

Kx_ii = sp.eye(nx)
Kx_ii = Kx_ii.tocsr()
# Kx_ii[0, 0] = 0
# Kx_ii[-1, -1] = 0

Dx = sp.kron(Kx_ii, Dx_ii) #+ sp.kron(K_id, sp.eye(ny))


Dy_ii = sp.eye(ny) * (-1 / dy)
Dy_ii = Dy_ii.tocsr()
# Dy_ii[0, 0] = 1
# Dy_ii[-1, -1] = 1

Ky_ii = sp.eye(nx)
Ky_ii = Ky_ii.tocsr()
# Ky_ii[0, 0] = 0
# Ky_ii[-1, -1] = 0

Dy_ij = sp.eye(ny) / dy
Dy_ij = Dy_ij.tocsr()
# Dy_ij[0, 0] = 0
# Dy_ij[-1, -1] = 0

Ky_ij = sp.lil_matrix((nx, nx))
Ky_ij.setdiag(1, 1)
Ky_ij = Ky_ij.tocsr()
# Ky_ij[0, 1] = 0

Dy = sp.kron(Ky_ii, Dy_ii) + sp.kron(Ky_ij, Dy_ij) #+ sp.kron(K_id, sp.eye(ny))


Lap_ii = sp.lil_matrix((ny, ny))
Lap_ii.setdiag(-2 * (1 / dx ** 2 + 1 / dy ** 2))
Lap_ii.setdiag(1 / dx ** 2, 1)
Lap_ii.setdiag(1 / dx ** 2, -1)
Lap_ii = Lap_ii.tocsr()
# Lap_ii[0, 0] = 1
# Lap_ii[0, 1] = 0
# Lap_ii[-1, -1] = 1
# Lap_ii[-1, -2] = 0

Klap_ii = sp.eye(nx)
Klap_ii = Klap_ii.tocsr()
# Klap_ii[0, 0] = 0
# Klap_ii[-1, -1] = 0

Lap_ij = sp.eye(ny) / (dy ** 2)
Lap_ij = Lap_ij.tocsr()
# Lap_ij[0, 0] = 0
# Lap_ij[-1, -1] = 0

Klap_ij = sp.lil_matrix((nx, nx))
Klap_ij.setdiag(1, 1)
Klap_ij.setdiag(1, -1)
Klap_ij = Klap_ij.tocsr()
# Klap_ij[0, 1] = 0
# Klap_ij[-1, -2] = 0

Lap = sp.kron(Klap_ii, Lap_ii) + sp.kron(Klap_ij, Lap_ij) #+ sp.kron(K_id, sp.eye(ny))

M = alpha * Dx + beta * Dy + eps * Lap

# print(eigs(M)[0])

# eig = []

# C1 = -(alpha / dx + beta / dy + 2 * (1 / dx ** 2 + 1 / dy ** 2) * eps)
# C2 = -2 * (eps * alpha / (dx ** 3) + (eps ** 2) / (dx ** 4)) ** 0.5
# C3 = -2 * (eps * beta / (dy ** 3) + (eps ** 2) / (dy ** 4)) ** 0.5
# m = 100

# for l in range(1, nx + 1):
# 	for k in range(1, ny + 1):
# 		eig.append(C1 + C2 * np.cos(k * np.pi / (ny + 1)) + C3 * np.cos(l * np.pi / (nx + 1)))
# 		if np.abs(eig[-1]) < m:
# 			print(eig[-1], k, l)
# eig.sort()

# plt.plot(range(N), eig)
# plt.show()



# B = np.ones(N)

# for i in range(ny):
#     B[i] = 0
#     B[-(i + 1)] = 0

# for j in range(nx):
#     B[j * ny] = 0
#     B[(j + 1) * ny - 1] = 0

# X_sol = spsolve(Dx, B)

# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_surface(X, Y, X_sol.reshape((nx, ny)), rstride=2, cstride=2, linewidth=1)
# plt.show()

# plt.figure()
# plt.imshow(Dx.A, cmap='gray')
# plt.title('dx')
# plt.figure()
# plt.imshow(Dy.A, cmap='gray')
# plt.title('dy')
# plt.figure()
# plt.imshow(Lap.A, cmap='gray')
# plt.title('lap')
# plt.figure()
# plt.imshow(M.A, cmap='gray')
# plt.title('M')
# plt.show()

def grad(U): return (Dx.dot(U), Dy.dot(U))


# F = np.ones(N)

# for i in range(ny):
#         F[i] = 0
#         F[-(i + 1)] = 0

# for j in range(nx):
#     F[j * ny] = 0
#     F[(j + 1) * ny - 1] = 0

# Fx, Fy = grad(F)
# Flap = Lap.dot(F)

# plt.figure(1)
# plt.imshow(Fx.reshape((nx, ny)), cmap='gray')
# plt.figure(2)
# plt.imshow(Fy.reshape((nx, ny)), cmap='gray')
# plt.figure(3)
# plt.imshow(Flap.reshape((nx, ny)), cmap='gray')

# plt.show()


Z_mat = generer_surface(Nx=nx, Ny=ny, forme=('volcan',100,100,0.5,0.2,0.5), reg = 1)
# Z_mat = generer_surface(Nx=nx, Ny=ny, forme=('trap',30,100,1,0.5), reg=0)
# Z_mat = generer_surface(Nx=nx, Ny=ny, forme=('cone', 100, 10), reg=0)
# Z_mat = generer_surface(Nx=nx, Ny=ny, forme=('plateau',20,20,1), reg = 0)

Z = np.reshape(Z_mat, N)

E = eclairement(Z, lV, grad)
# E = bruit_gaussien(E, 0.1)
# E = bruit_selpoivre(E, 0.01)

E_cp = E.copy()
E_cp_mat = np.reshape(E_cp, (nx, ny))

V = np.sum(Z)

# E_cp_mat = io.imread('img/lune.jpeg', 'L')
# E_cp_mat = (E_cp_mat - np.min(E_cp_mat))/(np.max(E_cp_mat) - np.min(E_cp_mat))
# E = np.reshape(E_cp_mat, N)
# E_cp = E.copy()

compt = 0

Z_appr = np.zeros(N)

plt.figure(-5)
plt.imshow(E_cp_mat, cmap='gray')

while compt < nb_it:

    compt += 1

    Z_gradx, Z_grady = grad(Z_appr)
    corr = np.sqrt(1 + Z_gradx**2 + Z_grady**2)
    E = E_cp * corr - gamma - CB

    # for i in range(ny):
    #     E[i] = 0
    #     E[-(i + 1)] = 0

    # for j in range(nx):
    #     E[j * ny] = 0
    #     E[(j + 1) * ny - 1] = 0

    Z_appr = spsolve(M.T.dot(M), M.T.dot(E).T)

    # print(np.max(np.abs(M.T.dot(M).dot(Z_appr)-M.T.dot(E).T)))

    E_appr = eclairement(Z_appr, lV, grad)
    Z_appr_mat = np.reshape(Z_appr, (nx, ny))
    E_appr_mat = np.reshape(E_appr, (nx, ny))

    fig = plt.figure(10 * compt)
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z_appr_mat, rstride=5, cstride=5, linewidth=1)
    ax.plot_wireframe(X, Y, Z_mat, rstride=5, cstride=5, linewidth=1, color='r')

    # fig = plt.figure(10 * compt + 1)
    # ax = fig.gca(projection='3d')
    # ax.plot_surface(X,Y,Z_appr_n - Z_appr,rstride=2,cstride=2,linewidth=1)

    # plt.figure(10 * compt + 2)
    # plt.imshow(E_appr_mat, cmap='gray')

    # plt.figure()
    # plt.imshow(np.abs(E_appr_mat - E_cp_mat), cmap='gray', vmin = 0, vmax = 1)

    print(comparer_eclairement(E_cp, E_appr))
    V_appr = np.sum(Z_appr)
    print(np.abs(V - V_appr) / V)


plt.show()
