# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from libSFS import *
from time import clock

plt.ion()
plt.show()
plt.rcParams["image.cmap"]="gray"

nx = 128
ny = 128

## Paramètres du problème

theta = np.pi/4
phi = 0
theta_obs = 0
phi_obs = 0
lV = direction_eclairement((theta, phi), (theta_obs, phi_obs))
CB=np.array([np.zeros(nx),np.zeros(nx),np.zeros(ny),np.zeros(ny)]) # Conditions de bord du problème
x = np.linspace(-nx/2,nx/2-1,nx)
y = np.linspace(-ny/2,ny/2-1,ny)
X,Y = np.meshgrid(y,x)

Z=generer_surface(Nx=nx, Ny=ny, forme=('cone',50,20), reg = 0)
V=sum(sum(Z))
I=eclairement(Z,lV,np.gradient)
delta=0.001 # précision de la méthode
J=0.99

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

def Psi_2(x,y,theta,z):
    r=np.sqrt(z[x,y]**2+(x-nx/2+1)**2)
    if r>0:
        sigma=np.arccos((x-nx/2+1)/r)
    else: 
        sigma=0
    return r*np.sin(theta-sigma)+ny/2

nx_tilde=int(nx*np.cos(theta))+1
x_tilde = np.linspace(-nx_tilde/2,nx_tilde/2-1,nx_tilde)
X_tilde,Y_tilde = np.meshgrid(y,x_tilde)

I_test=np.zeros(nx)
I_tilde=np.zeros((nx_tilde,ny))
Z_tilde=np.zeros((nx_tilde,ny))

for x in range(nx):
    for y in range(ny):
        I_tilde[Psi(x,y,theta,Z),y]=I[x,y]
        
for x in range(nx):
    for y in range(ny):
        Z_tilde[Psi(x,y,theta,Z),y]=Psi_2(x,y,theta,Z)+Z[x,y]

plt.figure(27)
plt.imshow(I_tilde)
plt.figure(17)
plt.plot(I_test)

n=np.zeros((nx,ny)) # n(x,y) est la fonction qui est égale à la norme du gradient
for x in range(1,nx-1):
    for y in range(1,ny-1):
        n[x,y]=np.sqrt(1/I[x,y]**2-1)

z0=np.zeros((nx,ny))
z=np.zeros((nx,ny))

z0[:,0]=CB[0] # on impose les conditions de bord a priori
z0[:,ny-1]=CB[1]
z0[0,:]=CB[2]
z0[nx-1,:]=CB[3]
z[:,0]=CB[0]
z[:,ny-1]=CB[1]
z[0,:]=CB[2]
z[nx-1,:]=CB[3]

fig = plt.figure(1)
ax = fig.gca(projection='3d')
ax.plot_surface(X,Y,Z,rstride=2,cstride=2,linewidth=1)

fig = plt.figure(15)
ax = fig.gca(projection='3d')
ax.plot_surface(X_tilde,Y_tilde,Z_tilde,rstride=2,cstride=2,linewidth=1)

I_ttt=eclairement(Z_tilde,lV,np.gradient)

plt.figure(2)
plt.imshow(I)
plt.figure(3)
plt.imshow(I_ttt)
plt.figure(4)
plt.imshow(eclairement(z,lV,np.gradient))