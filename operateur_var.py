import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs
from numpy.linalg import norm
from libSFS import *
from math import sqrt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import quad

N = 32
eps = 1

nb_vp = 6
N_vp = 32

n_mail = 128

theta = np.pi / 3
phi = np.pi / 4
theta_obs = 0
phi_obs = 0
lV = direction_eclairement((theta, phi), (theta_obs, phi_obs))
alpha, beta, gamma = lV

dx = 1
dy = 1

Dx_ii = sp.lil_matrix((n_mail, n_mail))
Dx_ii.setdiag(-1.0 / dx)
Dx_ii.setdiag(1.0 / dx, 1)
Dx_ii = Dx_ii.tocsr()

Kx_ii = sp.eye(n_mail)
Kx_ii = Kx_ii.tocsr()

Dx = sp.kron(Kx_ii, Dx_ii)

Dy_ii = sp.eye(n_mail) * (-1 / dy)
Dy_ii = Dy_ii.tocsr()

Ky_ii = sp.eye(n_mail)
Ky_ii = Ky_ii.tocsr()

Dy_ij = sp.eye(n_mail) / dy
Dy_ij = Dy_ij.tocsr()

Ky_ij = sp.lil_matrix((n_mail, n_mail))
Ky_ij.setdiag(1, 1)
Ky_ij = Ky_ij.tocsr()

Dy = sp.kron(Ky_ii, Dy_ii) + sp.kron(Ky_ij, Dy_ij)


def grad(U): return (Dx.dot(U), Dy.dot(U))


Z_mat = generer_surface(Nx=n_mail, Ny=n_mail, forme=('cone', 10, 1), reg=0)
Z = np.reshape(Z_mat, n_mail**2)

E = eclairement(Z, lV, grad)
E_mat = np.reshape(E, (n_mail, n_mail))

F_mat = E_mat - gamma

# A = sp.csr_matrix((N ** 2, N ** 2))

# for i in range(N):
# 	print(i)
# 	for j in range(N):
# 		for k in range(N):
# 			for l in range(N):
# 				if i == k and j == l:
# 					A[N * j + i, N * l + k] = -np.pi * \
# 					eps / (1 + i ** 2 + j ** 2)
# 				elif i == k and (j-l)%2 == 1:
# 					A[N * j + i, N * l + k] = 2 * beta * j * l / (sqrt(1 + i ** 2 + j ** 2)*sqrt(1+k**2 + l**2)*(l-j)*(l+j))
# 				elif j == l and (i-k)%2 == 1:
# 					A[N * j + i, N * l + k] = 2 * alpha * i * k / (sqrt(1 + i ** 2 + j ** 2)*sqrt(1+k**2 + l**2)*(k-i)*(k+i))

# print("A formee")

val_p_T, vect_p_T = eigs(A, k=nb_vp, which='LM')

x = np.linspace(0, np.pi, n_mail)
y = np.linspace(0, np.pi, n_mail)
X, Y = np.meshgrid(y, x)
# surf_p_T = []

Z_appr = np.zeros((n_mail, n_mail))
for i in range(nb_vp):
	vect_p = vect_p_T[:,i]
	surf_p = np.zeros((n_mail,n_mail))
	for n in range(1, N_vp):
		for m in range(1, N_vp):
			surf_p = surf_p + vect_p[N*m + n] * np.sin(n*X) * np.sin(m*Y) * (2 / (np.pi * sqrt(1+n**2 + m **2)))

	print(np.sum(np.abs(surf_p**2)))
	# surf_p = np.abs(surf_p)
	# surf_p_T.append(surf_p_T)

	# int_p = np.sum(F_mat * surf_p) * np.pi / N**2
	# print(norm(F_mat * surf_p))

	# print(int_p, val_p_T[i], int_p / val_p_T[i])

	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.plot_surface(X, Y, surf_p, rstride=5, cstride=5, linewidth=1)
	plt.title(np.abs(val_p_T[i]))

	# mod = norm(vect_p)
	# plt.title(str(val_p_T[i]) + ' ' + str(mod))
	# Z_appr = Z_appr + (int_p / val_p_T[i]) * np.sin(n*X) * np.sin(m*Y) * (2 / (np.pi * sqrt(1+n**2 + m **2)))
	# fig = plt.figure()
	# ax = fig.gca(projection='3d')
	# ax.plot_surface(X, Y, Z_appr, rstride=5, cstride=5, linewidth=1)

# fig = plt.figure('resultat')
# ax = fig.gca(projection='3d')
# ax.plot_surface(X, Y, Z_appr, rstride=5, cstride=5, linewidth=1)

# plt.plot(np.absolute(val_p_T))
# plt.show()

plt.show()