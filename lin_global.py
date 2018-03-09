# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from fonctions_utiles import *
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from time import clock

t1 = clock()

nx = 128
ny = 128
N = nx*ny

theta = np.pi/5
phi = np.pi/4
alpha = np.sin(theta)*np.cos(phi)
beta = np.sin(theta)*np.sin(phi)
gamma = np.cos(theta)
lV = np.array([alpha,beta,gamma])

eps = 0

dx = 1
dy = 1

x = np.linspace(-nx/2,nx/2-1,nx)
y = np.linspace(-ny/2,ny/2-1,ny)
X,Y = np.meshgrid(y,x)

Z,E,V = generer_surface(Nx=nx, Ny=ny, forme=('volcan',40,40,0.5,0.3,0.4), reg = 2, lV=(theta,phi),obV=(0,0))
# Z,E,V = generer_surface(Nx=nx, Ny=ny, forme=('trap',80,80,1,0.5), reg=0, lV=(theta,phi),obV=(0,0))
# Z,E,V = generer_surface(Nx=nx, Ny=ny, forme=('cone',40,10), reg = 0, lV=(theta,phi),obV=(0,0))
# Z,E,V = generer_surface(Nx=nx, Ny=ny, forme=('plateau',20,20,1), reg = 5, lV=(theta,phi),obV=(0,0))

E_cp = E.copy()

t2 = clock()

print("surface generee")
print(t2-t1)

t1 = clock()

M_lapx = sp.lil_matrix((N,N))
M_lapx.setdiag(2)
M_lapx.setdiag(-1,1)
M_lapx.setdiag(-1,-1)
M_lapx = -1/(dx**2) * M_lapx

M_lapy = sp.lil_matrix((N,N))
M_lapy.setdiag(2)
M_lapy.setdiag(-1,ny)
M_lapy.setdiag(-1,-ny)
M_lapy = -1/(dy**2) * M_lapy

M_lap = M_lapx + M_lapy

M_dx = sp.lil_matrix((N,N))

M_dx.setdiag(1)
M_dx.setdiag(-1,1)
M_dx = (-1/dx) * M_dx


M_dy = sp.lil_matrix((N,N))

M_dy.setdiag(1)
M_dy.setdiag(-1,ny)
M_dy = (-1/(dy)) * M_dy


M = eps*M_lap + alpha*M_dx + beta*M_dy

M[:ny,:ny] = sp.eye(ny)
M[-ny:,-ny:] = sp.eye(ny)

for i in range(1,nx-1):
    M[i*ny,i*ny] = 1
    M[(i+1)*ny -1,(i+1)*ny -1] = 1
    # M[i*ny +1,i*ny] = 0
    M[i*ny,i*ny +1] = 0
    M[(i+1)*ny -1,(i+1)*ny -2] = 0
    # M[(i+1)*ny -2,(i+1)*ny -1] = 0
    M[i*ny,(i+1)*ny] = 0
    M[(i+1)*ny,i*ny] = 0


t2 = clock()

print("matrice formee")
print(t2-t1)

t1 = clock()

nb_it = 5

compt = 0

Z_appr = np.zeros((nx,ny))

# plt.figure(-5)
# plt.imshow(E_cp,cmap='gray')

while compt < nb_it:
    
    compt+=1
    
    Z_gradx, Z_grady = np.gradient(Z_appr)
    corr = np.sqrt(1 + Z_gradx**2 + Z_grady**2)
    E = E_cp*corr - gamma
        
    F = np.reshape(E,N)
    
    F[:ny] = [0] * ny
    F[-ny:] = [0] * ny
    
    for i in range(1,nx-1):
        F[ny*i] = 0
        F[ny*(i+1) -1] = 0
    
    Z_appr_vect = spsolve(M,F)
    
    Z_appr_n = np.reshape(Z_appr_vect,(nx,ny))
    
    E_appr = eclairement(Z_appr_n,lV)
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X,Y,Z_appr_n,rstride=2,cstride=2,linewidth=1)
    ax.plot_wireframe(X,Y,Z,rstride=2,cstride=2,linewidth=1,color='r')
    plt.title(compt)
    
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.plot_surface(X,Y,Z_appr_n - Z_appr,rstride=2,cstride=2,linewidth=1)
    # plt.title(10*compt +1)
    
    # plt.figure(100*compt)
    # plt.imshow(E_appr,cmap='gray')
    
    print(comparer_eclairement(E_cp,E_appr))
    Z_appr = Z_appr_n
    V_appr = np.sum(Z_appr)
    print(V,V_appr,np.abs(V-V_appr)/V)

t2 = clock()

print("affichage ok")
print(t2-t1)

plt.show()