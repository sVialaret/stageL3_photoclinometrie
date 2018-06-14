# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from libSFS import *
from time import clock

plt.ion()
plt.show()
plt.rcParams["image.cmap"]="gray"

#t1 = clock()

nx = 64
ny = 64
N = nx * ny

theta = 0
phi = 0
theta_obs = 0
phi_obs = 0
lV = direction_eclairement((theta, phi), (theta_obs, phi_obs))
CB=np.array([np.zeros(nx),np.zeros(nx),np.zeros(ny),np.zeros(ny)]) # Conditions de bord du problème
# CB=np.array([30*np.ones(nx),30*np.ones(nx),30*np.ones(ny),30*np.ones(ny)]) # Conditions de bord du problème
x = np.linspace(-nx/2,nx/2-1,nx)
y = np.linspace(-ny/2,ny/2-1,ny)
X,Y = np.meshgrid(y,x)

# Z=generer_surface(Nx=nx, Ny=ny, forme=('trap',60,20,20,10), reg = 0)
Z = generer_surface(Nx=nx, Ny=ny, forme=('cone', 30, 100), reg=0)
V=sum(sum(Z))
I=eclairement(Z,lV,np.gradient)
nb=61 # nombre d'itérations nécessaires pour déterminer la surface
m=5
delta=0.01 # précision de la méthode

if True in (I<0):
    print("occlusion")
else:

    def g(a,b,c,d): # gradient
        return np.sqrt(max(max(a,0),max(b,0))**2+max(max(c,0),max(d,0))**2)

    def f(U,n): # on cherche z(x,y) tel que le gradient obtenu (par rapport à l'ancienne surface) convienne
        v=min(U)
        while abs(g(v-U[0],v-U[1],v-U[2],v-U[3])-n)>delta :
            v=v-g(v-U[0],v-U[1],v-U[2],v-U[3])+n
        return v
        
    def h(x,y):
        C=[y,ny-y,x,nx-x]
        c=np.argmin(C)
        if c==0:
            H=CB[0,x]
            for j in range(y):
                H+=n[x,j]
        elif c==1:
            H=CB[1,x]
            for j in range(ny-y):
                H+=n[x,j+y]
        elif c==2:
            H=CB[2,y]
            for i in range(x):
                H+=n[i,y]
        else:
            H=CB[3,y]
            for i in range(nx-x):
                H+=n[i+x,y]
        return H
        
    n=np.zeros((nx,ny)) # n(x,y) est la fonction qui est égale à la norme du gradient
    for x in range(1,nx-1):
        for y in range(1,ny-1):
            n[x,y]=np.sqrt(1/I[x,y]**2-1)

    z0=np.zeros((nx,ny))
    z=np.zeros((nx,ny))

    Q=points_critiques(I)

    z0[:,0]=CB[0] # on impose les conditions de bord a priori
    z0[:,ny-1]=CB[1]
    z0[0,:]=CB[2]
    z0[ny-1,:]=CB[3]
    z[:,0]=CB[0]
    z[:,ny-1]=CB[1]
    z[0,:]=CB[2]
    z[ny-1,:]=CB[3]

    for x in range(nx):
        for y in range(ny):
            if Q[x,y]==1:
                z0[x,y]=h(x,y)
                z[x,y]=z0[x,y]

    for i in range(nb):
        if i==0:
            t1=clock()
        for x in range(1,nx-1):
            for y in range(1,ny-1):
                if Q[x,y]==0:
                    U=[z0[x-1,y],z0[x+1,y],z0[x,y-1],z0[x,y+1]]
                    z[x,y]=f(U,n[x,y])
        for x in range(1,nx-1):
            for y in range(1,ny-1):
                if Q[x,y]==0:
                    z0[x,y]=z[x,y]
        if i==m:
            t2=clock()
            print((t2-t1)*nb/m)
        if i % 10 == 0:
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.plot_surface(X,Y,z,rstride=2,cstride=2,linewidth=1)

    # fig = plt.figure(1)
    # ax = fig.gca(projection='3d')
    # ax.plot_surface(X,Y,Z,rstride=2,cstride=2,linewidth=1)
    # plt.figure(2)
    # plt.imshow(I)
    # plt.figure(3)
    # plt.imshow(n)
    # plt.figure(4)
    # plt.imshow(eclairement(z,lV,np.gradient))
    # plt.figure(5)
    # plt.imshow(Q)
    # F=frontiere(Q)
    # plt.figure(6)
    # plt.imshow(F)

    V=np.sum(Z)
    v=np.sum(z)
    print(abs(v-V)/V)