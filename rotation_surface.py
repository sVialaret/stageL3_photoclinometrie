# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from libSFS import direction_eclairement, generer_surface, eclairement
from time import clock

plt.ion()

plt.show()
plt.rcParams["image.cmap"]="gray"

## Fonctions utiles

def g(a,b,c,d): # gradient
    return np.sqrt(max(a,b,0)**2+max(c,d,0)**2)

def f(U,n): # on cherche z(x,y) tel que le gradient obtenu (par rapport à l'ancienne surface) convienne
    v=min(U)
    c=g(v-U[0],v-U[1],v-U[2],v-U[3])-n
    while abs(c)>delta :
        v=v-c
        c=g(v-U[0],v-U[1],v-U[2],v-U[3])-n
    return v

def Psi(x,y,theta,z):
    r=np.sqrt(z[x,y]**2+(x-nx/2+1)**2)
    if r>0:
        sigma=np.arccos((x-nx/2+1)/r)
    else:
        sigma=0
    return int(r*np.cos(theta-sigma)+nx/2*np.cos(theta))
    # return x * np.cos(theta) - z[x,y] * np.sin(theta)

def rotation_Z(Z, theta):
    Z_rot = np.zeros((3,nx,ny))

    for i in range(nx):
        for j in range(ny):
            Z_rot[0][i,j] = Z[0][i,j] * np.cos(theta) - Z[2][i,j] * np.sin(theta)
            Z_rot[1][i,j] = Z[1][i,j]
            Z_rot[2][i,j] = Z[0][i,j] * np.sin(theta) + Z[2][i,j] * np.cos(theta)
    return Z_rot

def regularisation_maillage(X,Y,Z):
    minMeshX = np.min(X)
    maxMeshX = np.max(X)
    minMeshY = np.min(Y)
    maxMeshY = np.max(Y)

    x_mesh_reg = np.linspace(minMeshX, maxMeshX, nx)
    y_mesh_reg = np.linspace(minMeshY, maxMeshY, ny)
    X_reg,Y_reg = np.meshgrid(x_mesh_reg, y_mesh_reg)

    Z_reg = np.zeros((nx, ny))
    I_reg = np.zeros((nx, ny))

    # print(np.abs(X-X_reg))

    for i in range(nx):
        for j in range(ny):
            distArray = (X_reg - X[i,j]) ** 2 + (Y_reg - Y[i,j]) ** 2 
            ind = np.unravel_index(np.argmin(distArray, axis = None), (nx,ny))
            # if (i,j) != ind:
            #     print(i,j,ind)
            # print('==============')
            # print(distArray[i,j])
            # print(np.min(distArray))
            # print(np.unravel_index(np.argmin((X_reg - X[i,j]) ** 2 + (Y_reg - Y[i,j]) ** 2, axis=None), (nx,ny)))
            Z_reg[i,j] = Z[np.unravel_index(np.argmin((X_reg - X[i,j]) ** 2 + (Y_reg - Y[i,j]) ** 2), (nx,ny))]
            # Z_reg[i,j] = Z[i,j]
    return X_reg, Y_reg, Z_reg




## Parametres du probleme

nx = 64
ny = 64

theta = np.pi / 10
phi = 0
theta_obs = 0
phi_obs = 0
lV = direction_eclairement((theta, phi), (theta_obs, phi_obs))
(alpha, beta, gamma) = lV
CB=np.array([np.zeros(nx),np.zeros(nx),np.zeros(ny),np.zeros(ny)]) # Conditions de bord du problème

x_mesh = np.linspace(-5, 5, nx)
y_mesh = np.linspace(-5, 5, ny)
X,Y = np.meshgrid(x_mesh,y_mesh)

Z=generer_surface(Nx=nx, Ny=ny, forme=('cone',20,5), reg=0)
# Z=generer_surface(Nx=nx, Ny=ny, forme=('trap',10,50,1,1), reg = 0)
# Z = generer_surface(Nx=nx, Ny=ny, forme=('plateau',30,30,5), reg = 0)

V=np.sum(Z)
I=eclairement(Z,lV,np.gradient)

print(True in (I < (alpha ** 2 + beta ** 2)**.5))

Z_mesh = np.array([X,Y,Z])
Z_rot = rotation_Z(Z_mesh,theta)

# print(np.min(Z_rot[0]))


# I_rot=np.zeros((nx,ny))

# for x in range(nx):
#     for y in range(ny):
#         I_rot[Psi(x,y,theta,Z),y]=I[x,y]


# print(Z_rot[0][:,30])

# Z_retour = rotation_Z(Z_rot,theta)

# nx_rot, ny_rot = Z_rot.shape
# x_rot = np.linspace(-np.cos(theta), np.cos(theta), nx_rot)
# X_rot,Y_rot = np.meshgrid(y_mesh,x_rot)


# Z_retour = rotation_Z(Z_rot,np.cos(theta),-theta)
# nx_rot2, ny_rot2 = Z_retour.shape
# x_rot2 = np.linspace(-1, 1, nx_rot2)
# X_rot2,Y_rot2 = np.meshgrid(y_mesh,x_rot2)

# I_rot = eclairement(Z_rot[2], (0,0,1), np.gradient)
# I_retour = eclairement(Z_retour[2], lV, np.gradient)


X_reg, Y_reg, Z_reg = regularisation_maillage(Z_rot[0],Z_rot[1],Z_rot[2])


fig = plt.figure()
ax = fig.gca(projection='3d')
ax.axis('equal')
# ax.plot_wireframe(X,Y,Z,rstride=2,cstride=2,linewidth=1)
ax.plot_surface(Z_rot[0],Z_rot[1],Z_rot[2],rstride=2,cstride=2,linewidth=1, color='r')
# ax.plot_surface(X,Y,Z_rot[2],rstride=2,cstride=2,linewidth=1, color='r')

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X_reg, Y_reg, Z_reg,rstride=2,cstride=2,linewidth=1, color='r')



# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_surface(Z_retour[0],Z_retour[1],Z_retour[2],rstride=2,cstride=2,linewidth=1)
# ax.plot_wireframe(X, Y, Z, rstride=5, cstride=5, linewidth=1, color='r')


# plt.figure()
# plt.imshow(I)
# plt.figure(10)
# plt.imshow(I_rot)
# plt.figure()
# plt.imshow(I_retour)

# plt.figure()
# plt.imshow(np.abs(I - I_rot))


# print(np.max(np.abs(I - I_retour)))
# print(np.max(np.abs(I - I_rot)))