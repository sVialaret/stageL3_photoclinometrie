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
N = nx * ny

theta = 0
phi = 0
theta_obs = 0
phi_obs = 0
lV = direction_eclairement((theta, phi), (theta_obs, phi_obs))
CB=np.array([np.zeros(nx),np.zeros(nx),np.zeros(ny),np.zeros(ny)]) # Conditions de bord du problème
x = np.linspace(-nx/2,nx/2-1,nx)
y = np.linspace(-ny/2,ny/2-1,ny)
X,Y = np.meshgrid(y,x)

Z=generer_surface(Nx=nx, Ny=ny, forme=('cone',50,50), reg = 0)
#Z=generer_surface(Nx=nx, Ny=ny, forme=('trap',60,20,20,10), reg = 0)
#Z=generer_surface(Nx=nx, Ny=ny, forme=('volcan',50,50,10,7,0.5), reg = 2)
V=sum(sum(Z))
I=eclairement(Z,lV,np.gradient)
delta=0.001 # précision de la méthode
J=0.99

def g(a,b,c,d): # gradient
    return np.sqrt(max(a,b,0)**2+max(c,d,0)**2)

def f(U,n): # on cherche z(x,y) tel que le gradient obtenu (par rapport à l'ancienne surface) convienne
    v=min(U)
    c=g(v-U[0],v-U[1],v-U[2],v-U[3])-n
    while abs(c)>delta :
        v=v-c
        c=g(v-U[0],v-U[1],v-U[2],v-U[3])-n
    return v
    
def R(p,q):
    return 1/np.sqrt(1+p**2+q**2)
    
def Rp(p,q):
    return -p*(1+p**2+q**2)**(-1.5)
    
def Rq(p,q):
    return -q*(1+p**2+q**2)**(-1.5)
    
def reg(Z):
    r=0
    (Zx,Zy)=np.gradient(Z)
    Zxx=np.gradient(Zx)[0]
    Zyy=np.gradient(Zy)[0]
    for x in range(nx):
        for y in range(ny):
            r = r + Zxx[x,y]**2 + Zyy[x,y]**2
    return r

n=np.empty((nx,ny)) # n(x,y) est la fonction qui est égale à la norme du gradient
for x in range(1,nx-1):
    for y in range(1,ny-1):
        n[x,y]=np.sqrt(1/I[x,y]**2-1)

z0=np.zeros((nx,ny))
z=np.zeros((nx,ny))
z0[nx//3:2*nx//3,ny//3:2*ny//3]=30
z[nx//3:2*nx//3,ny//3:2*ny//3]=30

Q=points_critiques(I)
Q_bis=deepcopy(Q)
Q_bis[:,0]=np.ones(nx)
Q_bis[:,ny-1]=np.ones(nx)
Q_bis[0,:]=np.ones(ny)
Q_bis[nx-1,:]=np.ones(nx)
CC=comp_connexes(Q)
P=np.ones(len(CC))
P[0]=1
P[1]=1
# P[2]=-1

z0[:,0]=CB[0] # on impose les conditions de bord a priori
z0[:,ny-1]=CB[1]
z0[0,:]=CB[2]
z0[nx-1,:]=CB[3]
z[:,0]=CB[0]
z[:,ny-1]=CB[1]
z[0,:]=CB[2]
z[nx-1,:]=CB[3]

V=np.zeros(len(CC))
CC=rearrange(CC)

for i in range(len(CC)):
    plt.figure(20+i)
    plt.imshow(CC[i],vmin=0,vmax=1)

for i in range(len(CC)): # on impose la hauteur sur chaque plateau
    V[i]=height(i,Q,V,CC,n,CB,P)

# 1ère étape

T=deepcopy(Q_bis)
for i in range(len(CC)): # on impose la hauteur sur chaque plateau
    for x in range(nx):
        for y in range(ny):
            if CC[i,x,y]==1:
                z0[x,y]=V[i]
                z[x,y]=z0[x,y]
t1=clock()
T=deepcopy(Q_bis)
i=0
while (T!=1).any(): # génération de la solution
    if i%10==0:
        fig = plt.figure(30+i//10)
        ax = fig.gca(projection='3d')
        ax.plot_surface(X,Y,z,rstride=2,cstride=2,linewidth=1)
    for x in range(1,nx-1):
        for y in range(1,ny-1):
            if T[x,y]==0:
                U=[z0[x-1,y],z0[x+1,y],z0[x,y-1],z0[x,y+1]]
                z[x,y]=f(U,n[x,y])
            if z[x,y]==z0[x,y]:
                T[x,y]=1
    for x in range(1,nx-1):
        for y in range(1,ny-1):
            if T[x,y]==0:
                z0[x,y]=z[x,y]
    i=i+1
t2=clock()
print(t2-t1)
        
REG=reg(z)
print(REG)
Vol=np.sum(Z)
v=np.sum(z)
print(abs(v-Vol)/Vol)

fig = plt.figure(15)
ax = fig.gca(projection='3d')
ax.plot_surface(X,Y,z,rstride=2,cstride=2,linewidth=1)

# 2ème étape

# j=0
# while reg(z)<=REG :
#     j=j+1
#     print(J**j)
#     print("-")
#     REG=reg(z)
#     for i in range(len(CC)):
#         for x in range(nx):
#             for y in range(ny):
#                 if CC[i,x,y]==1:
#                     z0[x,y]=V[i]*J**j
#                     z[x,y]=z0[x,y]
# 
#     T=deepcopy(Q_bis)
#     while (T!=1).any(): # génération de la solution
#         for x in range(1,nx-1):
#             for y in range(1,ny-1):
#                 if T[x,y]==0:
#                     U=[z0[x-1,y],z0[x+1,y],z0[x,y-1],z0[x,y+1]]
#                     z[x,y]=f(U,n[x,y])
#                 if z[x,y]==z0[x,y]:
#                     T[x,y]=1
#         for x in range(1,nx-1):
#             for y in range(1,ny-1):
#                 if T[x,y]==0:
#                     z0[x,y]=z[x,y]
#             
#     print(reg(z))
#     v=np.sum(z)
#     print(abs(v-Vol)/Vol)
# 
# fig = plt.figure(16)
# ax = fig.gca(projection='3d')
# ax.plot_surface(X,Y,z,rstride=2,cstride=2,linewidth=1)
# 
# # 3ème étape
# 
# z0=np.zeros((nx,ny))
# z=np.zeros((nx,ny))
# 
# z0[:,0]=CB[0]
# z0[:,ny-1]=CB[1]
# z0[0,:]=CB[2]
# z0[nx-1,:]=CB[3]
# z[:,0]=CB[0]
# z[:,ny-1]=CB[1]
# z[0,:]=CB[2]
# z[nx-1,:]=CB[3]
# 
# 
# for i in range(len(CC)): # on impose la hauteur sur chaque plateau
#     for x in range(nx):
#         for y in range(ny):
#             if CC[i,x,y]==1:
#                 z0[x,y]=V[i]*J**(j-1)
#                 z[x,y]=z0[x,y]
# 
# t1=clock()
# T=deepcopy(Q_bis)
# while (T!=1).any(): # génération de la solution
#     for x in range(1,nx-1):
#         for y in range(1,ny-1):
#             if T[x,y]==0:
#                 U=[z0[x-1,y],z0[x+1,y],z0[x,y-1],z0[x,y+1]]
#                 z[x,y]=f(U,n[x,y])
#             if z[x,y]==z0[x,y]:
#                 T[x,y]=1
#     for x in range(1,nx-1):
#         for y in range(1,ny-1):
#             if T[x,y]==0:
#                 z0[x,y]=z[x,y]
#         
# REG=reg(z)
# print(REG)
# v=np.sum(z)
# print(abs(v-Vol)/Vol)
# 
# fig = plt.figure(17)
# ax = fig.gca(projection='3d')
# ax.plot_surface(X,Y,z,rstride=2,cstride=2,linewidth=1)

fig = plt.figure(1)
ax = fig.gca(projection='3d')
ax.plot_surface(X,Y,Z,rstride=2,cstride=2,linewidth=1)
plt.figure(2)
plt.imshow(I)
plt.figure(3)
plt.imshow(Q)
# plt.figure(4)
# plt.imshow(eclairement(z,lV,np.gradient))

# H2

# values=[]
# 
# values=height_2(I,Q,values,CC)
# values=np.array(values)
# 
# distance=[]
# for i in range(len(CC)):
#     distance.append([])
#     for j in range(len(CC)):
#         distance[i].append([])
#         for k in range(len(values)):
#             if values[k,0]==i and values[k,1]==j:
#                 distance[i][j].append(values[k,2])
# 
# #print(distance)
# plt.figure(30)
# A1=sorted(distance[0][1])
# plt.plot(sorted(distance[0][1]))
# plt.figure(31)
# A2=sorted(distance[1][0])
# plt.plot(sorted(distance[1][0]))
#                 
# XX=[int(min(A1))+i for i in range(int(max(A1))-int(min(A1)))]
# B=np.zeros(int(max(A1))-int(min(A1)))
# for i in A:
#     B[int(i)-int(min(A1))-1]=B[int(i)-int(min(A1))-1]+1
# plt.figure(60)
# plt.plot(XX,B)
# XX=[int(min(A2))+i for i in range(int(max(A2))-int(min(A2)))]
# B=np.zeros(int(max(A2))-int(min(A2)))
# for i in A:
#     B[int(i)-int(min(A2))-1]=B[int(i)-int(min(A2))-1]+1
# plt.figure(61)
# plt.plot(XX,B)


# def height_2(I,Q,V,CC):
#     """
#         calcule la hauteur des ensembles CC[i] en suivant une ligne caractéristique 
#     """
#     ds=0.5
#     (Ix,Iy)=np.gradient(I)
#     (nx,ny)=CC[0].shape
#     F=[]
#     for k in range(len(CC)):
#         F.append(frontiere(CC[k]))
#     F=np.array(F)
#     L=[]
#     H=[]
#     LC=[]
#     for k in range(len(CC)):
#         L.append([])
#         LC.append([])
#         for x in range(nx):
#             for y in range(ny):
#                 if CC[k,x,y]:
#                     LC[k].append([x,y])
#                 if F[k,x,y]:
#                     L[k].append([x,y])
#     g=0
#     T=[np.zeros((nx,ny)),np.zeros((nx,ny)),np.zeros((nx,ny)),np.zeros((nx,ny)),np.zeros((nx,ny)),np.zeros((nx,ny)),np.zeros((nx,ny)),np.zeros((nx,ny)),np.zeros((nx,ny)),np.zeros((nx,ny))]
#     B=np.zeros((nx,ny))
#     for k in range(len(CC)):
#         for (x,y) in L[k]:
#             i=x
#             j=y
#             u=0
#             t1=clock()
#             p=0
#             q=0
#             S=[[]]
#             while i>0 and i<nx-1 and j>0 and j<ny-1 and Q[int(i),int(j)]==0:
#                 t2=clock()
#                 if t2-t1>10:
#                     print([x,y])
#                 # if Rp(p,q)==0 and Rq(p,q)==0:
#                 #     ds=1
#                 # else:
#                 #     ds=1/max(abs(Rp(p,q)),abs(Rq(p,q)))
#                 (i,j,u,p,q)=(i + Rp(p,q)*ds , j  + Rq(p,q)*ds , u + (p*Rp(p,q) + q*Rq(p,q))*ds , p + Ix[int(i),int(j)]*ds , q + Iy[int(i),int(j)]*ds)
#                 S.append([int(i),int(j)])
#             for a in range(nx):
#                 for b in range(ny):
#                     if [a,b] in S:
#                         T[g][a,b]=1
#             g=g+1
#             if g==10:
#                 g=0
#             for l in range(len(CC)):
#                 if [int(i),int(j)] in LC[l]:
#                     V.append([k,l,u])
#                 # +rajouter le cas où on arrive au bord
#             if u<-50:
#                 
#                 for a in range(nx):
#                     for b in range(ny):
#                         if [a,b] in S:
#                             B[a,b]=1
#                     plt.figure(70)
#                     plt.imshow(B)
#     for g in range(10):
#         plt.figure(40+g)
#         plt.imshow(T[g])
#     return V