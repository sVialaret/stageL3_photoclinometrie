import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs
from libSFS import *
from math import sqrt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

N = 64
eps = 1

nb_vp = 10
N_vp = 64

theta = np.pi / 5
phi = np.pi / 4
theta_obs = 0
phi_obs = 0
lV = direction_eclairement((theta, phi), (theta_obs, phi_obs))
alpha, beta, gamma = lV

Z_mat = generer_surface(Nx=N, Ny=N, forme=('cone', 30, 1), reg=0)
Z = np.reshape(Z_mat, N**2)
E = eclairement(Z, lV, np.gradient)
E_mat = np.reshape(E, (N, N))

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

x = np.linspace(0, np.pi, N)
y = np.linspace(0, np.pi, N)
X, Y = np.meshgrid(y, x)

# surf_p_T = []

for i in range(nb_vp):
	vect_p = vect_p_T[:,i]
	# print(val_p_T[i])
	surf_p = np.zeros((N,N))
	for n in range(N_vp):
		for m in range(N_vp):
			surf_p = surf_p + vect_p[N*m + n] * np.sin(n*X) * np.sin(m*Y)
	# surf_p_T.append(surf_p_T)


	# fig = plt.figure()
	# ax = fig.gca(projection='3d')
	# ax.plot_surface(X, Y, surf_p, rstride=5, cstride=5, linewidth=1)


plt.show()