# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from libSFS import *
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

def reg(Z):
    r=0
    (Zx,Zy)=np.gradient(Z)
    Zxx=np.gradient(Zx)[0]
    Zyy=np.gradient(Zy)[0]
    for x in range(nx):
        for y in range(ny):
            r = r + Zxx[x,y]**2 + Zyy[x,y]**2
    return r

def Psi(x,y,theta,z):
    r=np.sqrt(z[x,y]**2+(x-nx/2+1)**2)
    if r>0:
        sigma=np.arccos((x-nx/2+1)/r)
    else:
        sigma=0
    return int(r*np.cos(theta-sigma)+nx/2*np.cos(theta))
    # return x * np.cos(theta) - z[x,y] * np.sin(theta)

def Psi_2(x,y,theta,z):
    r=np.sqrt(z[x,y]**2+(x-nx/2+1)**2)
    if r>0:
        sigma=np.arccos((x-nx/2+1)/r)
    else: 
        sigma=0
    return r*np.sin(theta-sigma)+ny/2
    # return x * np.sin(theta) + z[x,y] * np.cos(theta)

# def rotation_Z(Z, longueur, theta):
#     nx, ny = Z.shape
#     # x_mesh_tmp = np.linspace(-longueur, longueur, nx)
#     # y_mesh_tmp = np.linspace(-longueur, longueur, ny)
#     x_mesh_tmp = np.linspace(0, nx - 1, nx)
#     y_mesh_tmp = np.linspace(0, ny - 1, ny)
#     X_tmp, Y_tmp = np.meshgrid(y_mesh_tmp, x_mesh_tmp)

#     # nx_rot = int(np.max(np.abs(X_tmp * np.cos(theta) - Z * np.sin(theta)))) + 2
#     # index_non_assign = np.ones((nx_rot, ny))

#     # for x in x_mesh_tmp:
#     #     for y in y_mesh_tmp:
#     #         x_new = x * np.cos(theta) - Z[x,y] * np.sin(theta)
#     #         Z_rot[round(x_new), y] = x_new * np.sin(theta) + Z[x,y] * np.cos(theta)
#     #         index_non_assign[round(x_new), y] = 0

#     # ## on remplit les trous
#     # index_missing = np.where(index_non_assign == 1)

#     # nb_trou = len(index_missing[0])

#     # for x,y in zip(index_missing[0], index_missing[1]):
#     #     Z_rot[x,y] = np.sum(Z_rot[x-1:x+2,y-1:y+2]) / np.sum((Z_rot[x-1:x+2,y-1:y+2] != 0))
#     # return Z_rot

#     Z_rot = np.zeros((nx, ny))

#     for x in x_mesh_tmp:
#         for y in y_mesh_tmp:
#             Z_rot[x,y] = (x - nx/2) * np.sin(theta) + Z[x,y] * np.cos(theta)

#     # longueur = longueur * np.cos(theta)

#     return Z_rot

def rotation_Z(Z, theta):
    x_mesh_tmp = np.linspace(0, nx - 1, nx)
    y_mesh_tmp = np.linspace(0, ny - 1, ny)
    X_tmp, Y_tmp = np.meshgrid(y_mesh_tmp, x_mesh_tmp)
    Z_rot = np.zeros((3,nx,ny), dtype = object)

    for i in range(nx):
        for j in range(ny):
            Z_rot[0][i,j] = Z[0][i,j] * np.cos(theta) - Z[2][i,j] * np.sin(theta)
            Z_rot[1][i,j] = Z[1][i,j]
            Z_rot[2][i,j] = Z[0][i,j] * np.sin(theta) + Z[2][i,j] * np.cos(theta)
    return Z_rot


## Paramètres du problème

nx = 32
ny = 32

theta = np.pi / 10
phi = 0
theta_obs = 0
phi_obs = 0
lV = direction_eclairement((theta, phi), (theta_obs, phi_obs))
(alpha, beta, gamma) = lV
CB=np.array([np.zeros(nx),np.zeros(nx),np.zeros(ny),np.zeros(ny)]) # Conditions de bord du problème


x_mesh = np.linspace(-5, 5, nx)
y_mesh = np.linspace(-5, 5, ny)
X,Y = np.meshgrid(y_mesh,x_mesh)

Z=generer_surface(Nx=nx, Ny=ny, forme=('cone',10,5), reg=0)
V=np.sum(Z)
I=eclairement(Z,lV,np.gradient)

print(True in (I < (alpha ** 2 + beta ** 2)))

Z_mesh = np.array([[[[X[i,j], Y[i,j]], Z[i,j]] for i in range(nx)] for j in range(ny)])

Z_mesh = np.array([X,Y,Z])


Z_rot = rotation_Z(Z_mesh,theta)
Z_retour = rotation_Z(Z_rot,-theta)

# nx_rot, ny_rot = Z_rot.shape
# x_rot = np.linspace(-np.cos(theta), np.cos(theta), nx_rot)
# X_rot,Y_rot = np.meshgrid(y_mesh,x_rot)


# Z_retour = rotation_Z(Z_rot,np.cos(theta),-theta)
# nx_rot2, ny_rot2 = Z_retour.shape
# x_rot2 = np.linspace(-1, 1, nx_rot2)
# X_rot2,Y_rot2 = np.meshgrid(y_mesh,x_rot2)

# I_rot = eclairement(Z_retour, (0,0,1), np.gradient)
# I_retour = eclairement(Z_retour, lV, np.gradient)


fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X,Y,Z,rstride=2,cstride=2,linewidth=1)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(Z_rot[0],Z_rot[1],Z_rot[2],rstride=2,cstride=2,linewidth=1)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(Z_retour[0],Z_retour[1],Z_retour[2],rstride=2,cstride=2,linewidth=1)
ax.plot_wireframe(X, Y, Z, rstride=5, cstride=5, linewidth=1, color='r')


# plt.figure()
# plt.imshow(I)
# plt.figure()
# plt.imshow(I_rot)
# plt.figure()
# plt.imshow(I_retour)


# print(np.max(np.abs(I - I_retour)))