import numpy as np
import pylab as plt
import scipy.misc as imageio
from libSFS import *
from libFourier import *
from scipy.linalg import inv
from scipy import dot
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams["image.cmap"]="gray"
plt.ion()
plt.show()

## Paramètres de la simulation

nx=30
ny=30
# Paramètres du rayon incident
theta=np.pi/3  # Angle avec la normale -pi/2 < theta < pi/2
phi=np.pi/4
alpha=np.sin(theta)*np.cos(phi)
beta=np.sin(theta)*np.sin(phi)
gamma=np.cos(theta)
epsilon=0.1
nb_it=0

Z=generer_surface(Nx=nx, Ny=ny, forme=('cone',10,5), reg = 0)
lV=direction_eclairement((theta,phi),(0,0))
I=eclairement(Z,lV,np.gradient)
N=I.shape[0]

## Résolution du problème

A_alpha=np.zeros((N*N,N*N))
A_beta=np.zeros((N*N,N*N))
A_epsilon=np.zeros((N*N,N*N))
for i in range(N):
    for j in range(N):
        for k in range(N):
            for m in range(N):
                A_alpha[i*N+j,k*N+m]=np.pi/(N+1)*(k+1)*np.cos(np.pi/(N+1)*(i+1)*(k+1))*np.sin(np.pi/(N+1)*(j+1)*(m+1))
                A_beta[i*N+j,k*N+m]=np.pi/(N+1)*(m+1)*np.sin(np.pi/(N+1)*(i+1)*(k+1))*np.cos(np.pi/(N+1)*(j+1)*(m+1))
                A_epsilon[i*N+j,k*N+m]=-np.pi**2/(N+1)**2*((k+1)**2+(m+1)**2)*np.sin(np.pi/(N+1)*(i+1)*(k+1))*np.sin(np.pi/(N+1)*(j+1)*(m+1))
          
A=alpha*A_alpha+beta*A_beta+epsilon*A_epsilon
A_inv=inv(A)

B=np.zeros(N*N)
for i in range(N):
    for j in range(N):
        B[i*N+j]=I[i,j]-gamma

print("Matrices créées")

ZDST=dot(A_inv,B)
print("Coefficients calculés")
Zsq=np.zeros((N,N))
for k in range(N):
    for m in range(N):
        Zsq[k,m]=ZDST[k*N+m]
z=dst2(Zsq)

x = np.linspace(-nx/2,nx/2-1,nx)
y = np.linspace(-ny/2,ny/2-1,ny)
X,Y = np.meshgrid(y,x)

for h in range(nb_it):
    fig = plt.figure(17+h)
    ax = fig.gca(projection='3d')
    ax.plot_surface(X,Y,z,rstride=2,cstride=2,linewidth=1)
    Ibis=I.copy()
    Zx,Zy=gradient_tfd2(z)
    Ibis = Ibis*np.sqrt(1+Zx**2+Zy**2)
    for i in range(N):
        for j in range(N):
            B[i*N+j]=Ibis[i,j]-gamma
    ZDST=dot(A_inv,B)
    Zsq=np.zeros((N,N))
    for k in range(N):
        for m in range(N):
            Zsq[k,m]=ZDST[k*N+m]
    z=dst2(Zsq)
    print(h)

    
Ez=eclairement(z,lV,gradient_tfd2)
V=np.sum(Z)
v=np.sum(z)
print(abs((V-v)/V))
print(comparer_eclairement(I,Ez))

print("Affichage des figures")

## Affichage des résultats

x = np.linspace(-nx/2,nx/2-1,nx)
y = np.linspace(-ny/2,ny/2-1,ny)
X,Y = np.meshgrid(y,x)

fig = plt.figure(1)
ax = fig.gca(projection='3d')
ax.plot_surface(X,Y,Z,rstride=2,cstride=2,linewidth=1)

plt.figure(2)
plt.imshow(I)

fig = plt.figure(3)
ax = fig.gca(projection='3d')
ax.plot_surface(X,Y,z,rstride=2,cstride=2,linewidth=1)

plt.figure(4)
plt.imshow(Ez)

print("OK")