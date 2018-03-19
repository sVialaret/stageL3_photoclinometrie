# -*- coding: utf-8 -*-

import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from libSFS import *

t1 = clock()

nx = 64
ny = 64
N = nx * ny

theta = np.pi / 5
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

# gradient decentre

# Matrice du gradient selon x

M_ii_dx = sp.lil_matrix((ny, ny))
M_ii_dx.setdiag(-1)
M_ii_dx[0, 0] = 1
M_ii_dx[-1, -1] = 1
M_ii_dx = M_ii_dx.tocsr()

M_ij_dx = sp.lil_matrix((ny, ny))
M_ij_dx.setdiag(1)
M_ij_dx[0, 0] = 0
M_ij_dx[-1, -1] = 0
M_ij_dx = M_ij_dx.tocsr()

K_ii_dx = sp.lil_matrix((nx, nx))
K_ii_dx.setdiag(1)
K_ii_dx[0, 0] = 0
K_ii_dx[-1, -1] = 0
K_ii_dx = K_ii_dx.tocsr()

K_ij_dx = sp.lil_matrix((nx, nx))
K_ij_dx.setdiag(1, 1)
K_ij_dx[0, 1] = 0
K_ij_dx = K_ij_dx.tocsr()

K_id = sp.lil_matrix((nx, nx))
K_id[0, 0] = 1
K_id[-1, -1] = 1

M_dx = (sp.kron(K_id, sp.eye(ny)) + sp.kron(K_ii_dx, M_ii_dx) + sp.kron(K_ij_dx, M_ij_dx)) / dx

# Matrice du gradient selon y

M_ii_dy = sp.lil_matrix((ny, ny))
M_ii_dy.setdiag(1, 1)
M_ii_dy.setdiag(-1)
M_ii_dy[0, 0] = 1
M_ii_dy[0, 1] = 0
M_ii_dy[1, 0] = 0
M_ii_dy[-1, -1] = 1
M_ii_dy[-1, -2] = 0
M_ii_dy[-2, -1] = 0
M_ii_dy = M_ii_dy.tocsr()

K_ii_dy = sp.lil_matrix((nx, nx))
K_ii_dy.setdiag(1)
K_ii_dy[0, 0] = 0
K_ii_dy[-1, -1] = 0
K_ii_dy = K_ii_dy.tocsr()

M_dy = (sp.kron(K_id, sp.eye(ny)) + sp.kron(K_ii_dy, M_ii_dy)) / dy

# Matrice du laplacien

M_lap = M_dx.transpose().dot(M_dx) + M_dy.T.dot(M_dy)

# Matrice finale

M = eps * M_lap + alpha * M_dx + beta * M_dy

print("matrice formee")


def grad(U): return (M_dx.dot(U), M_dy.dot(U))


# Z,E,V = generer_surface(Nx=nx, Ny=ny, forme=('volcan',20,20,0.5,0.2,0.5), reg = 0, lV=(theta,phi),obV=(0,0))
# Z,E,V = generer_surface(Nx=nx, Ny=ny, forme=('trap',80,80,1,0.5), reg=0, lV=(theta,phi),obV=(0,0))
Z_mat = generer_surface(Nx=nx, Ny=ny, forme=('cone', 30, 5), reg=0)
# Z,E,V = generer_surface(Nx=nx, Ny=ny, forme=('plateau',20,20,1), reg = 0, lV=(theta,phi),obV=(0,0))

Z = np.reshape(Z_mat, N)

E = eclairement(Z, lV, grad)

E_cp = E.copy()
E_cp_mat = np.reshape(E_cp, (nx, ny))

V = np.sum(Z)

print("surface generee")

compt = 0

Z_appr = np.zeros(N)

plt.figure(-5)
plt.imshow(E_cp_mat, cmap='gray')

while compt < nb_it:

    compt += 1

    Z_gradx, Z_grady = grad(Z_appr)
    corr = np.sqrt(1 + Z_gradx**2 + Z_grady**2)
    E = E_cp * corr - gamma

    for i in range(ny):
        E[i] = 0
        E[-(i + 1)] = 0

    for j in range(nx):
        E[j * ny] = 0
        E[(j + 1) * ny - 1] = 0

    Z_appr = spsolve(M, E)

    E_appr = eclairement(Z_appr, lV, grad)
    Z_appr_mat = np.reshape(Z_appr, (nx, ny))
    E_appr_mat = np.reshape(E_appr, (nx, ny))

    fig = plt.figure(10 * compt)
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z_appr_mat, rstride=2, cstride=2, linewidth=1)
    ax.plot_wireframe(X, Y, Z_mat, rstride=2, cstride=2, linewidth=1, color='r')

    # fig = plt.figure(10 * compt + 1)
    # ax = fig.gca(projection='3d')
    # ax.plot_surface(X,Y,Z_appr_n - Z_appr,rstride=2,cstride=2,linewidth=1)

    plt.figure(10 * compt + 2)
    plt.imshow(E_appr_mat, cmap='gray')

    print(comparer_eclairement(E_cp, E_appr))
    V_appr = np.sum(Z_appr)
    print(np.abs(V - V_appr) / V)


plt.show()
