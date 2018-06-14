# -*- coding: utf-8 -*-

import os
os.chdir("/home/rosaell/Documents/2017_2018/Stage/stageL3_photoclinometrie")


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from libSFS import *
from time import clock
import numpy.random as rd

plt.ion()
plt.show()
plt.rcParams["image.cmap"]="gray"

#t1 = clock()

nx = 64
ny = 64
N = nx * ny

pi = np.pi

theta = pi / 10
phi = 0
theta_obs = 0
phi_obs = 0
lV = direction_eclairement((theta, phi), (theta_obs, phi_obs))
(alpha,beta,gamma) = lV
S = lV[:-1]
CB=np.array([np.zeros(nx),np.zeros(nx),np.zeros(ny),np.zeros(ny)]) # Conditions de bord du problème
x = np.linspace(-nx/2,nx/2-1,nx)
y = np.linspace(-ny/2,ny/2-1,ny)
X,Y = np.meshgrid(y,x)

# Z = generer_surface(Nx=nx, Ny=ny, forme=('trap',6,25,20,20), reg = 0)
Z = generer_surface(Nx=nx, Ny=ny, forme=('cone', 20, 10), reg=0)
# Z = generer_surface(Nx=nx, Ny=ny, forme=('volcan',10,10,10,7,0.5), reg = 2)
# Z = generer_surface(Nx=nx, Ny=ny, forme=('plateau',20,20,1), reg = 1)
# Z = np.zeros((nx,ny))

V=sum(sum(Z))
I=eclairement(Z,lV,np.gradient)
nb=100 # nombre d'itérations nécessaires pour déterminer la surface
m=5
delta=0.01 # précision de la méthode

# Q = points_critiques(I)
# print(1 in Q)
# plt.figure()
# plt.imshow(I)
# plt.figure()
# plt.imshow(Q)
# plt.show()


if True in (I<0):
    print("occlusion")
else:

    def g(a,b,c,d): # gradient
        return np.sqrt(max(max(a,0),max(b,0))**2+max(max(c,0),max(d,0))**2)

    def f(U,n): # on cherche z(x,y) tel que le gradient obtenu (par rapport à l'ancienne surface) convienne
        v=min(U)
        compt = 0
        while abs(g(v-U[0],v-U[1],v-U[2],v-U[3])-n)>delta and compt < 500 :
            v=v-g(v-U[0],v-U[1],v-U[2],v-U[3])+n
            compt = compt + 1
        if compt == 1000:
            print("satur")
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

    def ind(SscalU):

        delta = I ** 2 - SscalU ** 2

        delta_mask = (delta == 0)

        rac = (gamma * SscalU) / delta  + np.sqrt((I**2 / delta **2)*np.maximum(gamma**2 - delta, np.zeros(SscalU.shape)))

        return np.maximum(rac, np.zeros(SscalU.shape))

    def vectUnitGrad(z):
        gradZx, gradZy = np.gradient(z)
        dirUnitZ = np.zeros((nx,ny,2))

        for i in range(nx):
            for j in range(ny):
                if (gradZx[i,j]**2 + gradZy[i,j]**2) != 0:
                    dirUnitZ[i,j] = [gradZx[i,j], gradZy[i,j]] / (gradZx[i,j]**2 + gradZy[i,j]**2)**.5

        return dirUnitZ

    nb_it = 100

    # dirUnitZ = vectUnitGrad(Z)
    # bruitDir = rd.randn(nx,ny,2)
    # dirUnitZ = bruitDir + dirUnitZ

    # dirUnitZ = np.zeros((nx,ny,2))

    # for i in range(nx):
    #     for j in range(ny):
    #         if (X[i,j]**2 + Y[i,j]**2) != 0:
    #             dirUnitZ[i,j] = [-10*X[i,j], -Y[i,j]] / (X[i,j]**2 + Y[i,j]**2)**.5

    dirUnitZ = np.zeros((nx,ny,2))
    for i in range(nx):
        for j in range(ny):
            if (X[i,j]**2 + Y[i,j]**2) != 0:
                dirUnitZ[i,j] = S / (alpha ** 2 + beta ** 2)

    for i in range(nx):
        for j in range(ny):
            if (X[i,j]**2 + Y[i,j]**2) != 0:
                dirUnitZ[i,j] = -S / (alpha ** 2 + beta ** 2)**0.5

    for it in range(nb_it):

        print("======="+str(it)+"=======")
        # if it % 4 == 0:
        # plt.figure()
        # plt.quiver(X,Y,dirUnitZ[:,:,0],dirUnitZ[:,:,1],angles='xy')

        z0=np.zeros((nx,ny))
        z=np.zeros((nx,ny))

        z0[:,0]=CB[0] # on impose les conditions de bord a priori
        z0[:,ny-1]=CB[1]
        z0[0,:]=CB[2]
        z0[ny-1,:]=CB[3]
        z[:,0]=CB[0]
        z[:,ny-1]=CB[1]
        z[0,:]=CB[2]
        z[ny-1,:]=CB[3]

        # z0[nx/2,ny/2] = 5
        # z[nx/2,ny/2] = 5

        SscalU = np.array([[S.dot(dirUnitZ[i,j]) for j in range(ny)] for i in range(nx)])

        n = ind(SscalU)
        if True in (n < 0):
            print("n<0")
        for i in range(nb):
            # print(i)
            if i==0:
                t1=clock()
            for x in range(1,nx-1):
                for y in range(1,ny-1):
                    # if x != nx / 2 or y != ny / 2:
                    if SscalU[x,y] ** 2 > I[x,y] ** 2 - gamma ** 2:
                        U=[z0[x-1,y],z0[x+1,y],z0[x,y-1],z0[x,y+1]]
                        z[x,y]=f(U,n[x,y])
                    else:
                        U=[z0[x-1,y],z0[x+1,y],z0[x,y-1],z0[x,y+1]]
                        n_xy = gamma * (alpha ** 2 + beta ** 2)**.5 / (I[x,y] ** 2 - (alpha ** 2 + beta ** 2)) + (I[x,y] ** 2 * (1 - I[x,y] ** 2) / ((I[x,y] ** 2 - (alpha ** 2 + beta ** 2))**2))
                        if n_xy < 0:
                            print("n<0")
                        z[x,y]=f(U,n_xy)

            for x in range(1,nx-1):
                for y in range(1,ny-1):
                    z0[x,y]=z[x,y]
            if i==m:
                t2=clock()
                print((t2-t1)*nb/m)
        I_new = eclairement(z,lV,np.gradient)
        if True in (I_new <= 0):
            print("----------OCCLUSION----------")
        dirUnitZ = vectUnitGrad(z)
        print(comparer_eclairement(I,I_new))
        V_appr = np.sum(z)
        print(abs(V-V_appr)/V)

        # plt.figure()
        # plt.quiver(X,Y,dirUnitZ[:,:,1],dirUnitZ[:,:,0],angles='xy')
#         if it % 5 == 0:
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_surface(X,Y,z,rstride=2,cstride=2,linewidth=1)
# ax.plot_wireframe(X,Y,Z,rstride=2,cstride=2,linewidth=1,color='r')

#             plt.figure()
#             plt.imshow(SscalU ** 2 < I ** 2 - gamma ** 2)




    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.plot_surface(X,Y,z,rstride=2,cstride=2,linewidth=1)
    # plt.figure()
    # plt.imshow(eclairement(z,lV,np.gradient))
    # plt.figure()
    # plt.quiver(X,Y,dirUnitZ[:,:,0],dirUnitZ[:,:,1],angles='xy')

    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.plot_surface(X,Y,z,rstride=2,cstride=2,linewidth=1)
    # ax.plot_wireframe(X,Y,Z,rstride=2,cstride=2,linewidth=1,color='r')
# plt.figure(2)
# plt.imshow(I)
# plt.figure(3)
# plt.imshow(eclairement(z,lV,np.gradient))
