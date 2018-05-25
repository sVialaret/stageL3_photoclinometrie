import numpy as np
import pylab as plt
import scipy.misc as imageio
from libSFS import *
from libFourier import *
from scipy.signal import convolve2d
from mpl_toolkits.mplot3d import Axes3D
from numpy.fft import fftshift,fft2,ifft2
from time import clock

plt.rcParams["image.cmap"]="gray"
plt.ion()
plt.show()

## Paramètres de la simulation

nx=256
ny=256
# Paramètres du rayon incident
theta=np.pi/4 # Angle avec la normale -pi/2 < theta < pi/2
phi=np.pi/4
lV=direction_eclairement((theta,phi),(0,0))
(alpha,beta,gamma)=lV
delta=alpha/beta
epsilon=0.1
nb_it=0

Z=generer_surface(Nx=nx, Ny=ny, forme=('cone',100,1), reg = 0)
V=sum(sum(Z))
I=eclairement(Z,lV,np.gradient)
#I=imageio.imread("copernicus.png")[:,:,0]/256.
N=I.shape[0]
nx,ny=I.shape

i=fft2(I)
plt.figure(17)
i=fftshift(i)
plt.imshow(np.log(1+abs(i)))

## Résolution du problème

# z2=inv_cl2(I,alpha,beta,gamma,epsilon) #solution corrigée (voir plus bas)
# 
# for y0 in range(1,N):
#     i=0
#     debut=z2[0,y0]
#     if int(y0+delta*(N-1))<N:
#         fin=z2[N-1,int(y0+delta*(N-1))]
#         while i<N :
#             z2[i,int(y0+delta*i)]=z2[i,int(y0+delta*i)]-(debut+(fin-debut)*(i/(N-1)))
#             i=i+1
#     else :
#         n=int((N-y0)/delta)
#         fin=z2[n,N-1]
#         while int(y0+delta*i)<N :
#             z2[i,int(y0+delta*i)]=z2[i,int(y0+delta*i)]-(debut+(fin-debut)*(i/n))
#             i=i+1
# 
# for x0 in range(1,N):
#     i=0
#     debut=z2[x0,0]
#     if int(x0+(N-1)/delta)<N:
#         fin=z2[N-1,int(x0+(N-1)/delta)]
#         while i<N :
#             z2[int(x0+i/delta),i]=z2[int(x0+i/delta),i]-(debut+(fin-debut)*(i/(N-1)))
#             i=i+1
#     else :
#         n=int(delta*(N-x0))
#         fin=z2[N-1,n]
#         if n>0 :
#             while int(x0+i/delta)<N :
#                 z2[int(x0+i/delta),i]=z2[int(x0+i/delta),i]-(debut+(fin-debut)*(i/n))
#                 i=i+1

t1=clock()
z1=inv_cl2(I,alpha,beta,gamma,epsilon) #solution brute
t2=clock()
print("Solution calculée en :")
print(t2-t1)
# Zfft=np.zeros((N,N))
# Zfft[0,0]=1
# Zfft[0,1]=1
# Zfft[1,0]=1
# Zfft[1,1]=1
# z1=np.real(ifft2(Zfft))


x = np.linspace(-nx/2,nx/2-1,nx)
y = np.linspace(-ny/2,ny/2-1,ny)
X,Y = np.meshgrid(y,x)

S=0
for i in range(N):
    S+=z1[0,i]
    S+=z1[-1,i]
    S+=z1[i,0]
    S+=z1[i,-1]
S=S/(4*N)
    
for i in range(N):
    for j in range(N):
        z1[i,j]-=S

Ibis=I.copy()
for k in range(nb_it):
    (ii, jj) = build_centered_indices(N,N)
    # Dx = (2*1j*np.pi)*(ii/N)
    # Dx[0,0]=1
    # Dy = (2*1j*np.pi)*(jj/N)
    # Dy[0,0]=1
    # Zx = np.real(ifft2(fft2(z1)/Dx))
    # Zy = np.real(ifft2(fft2(z1)/Dy))
    Zx,Zy=np.gradient(z1)
    Ibis=I.copy()
    Ibis = Ibis*np.sqrt(1+Zx**2+Zy**2)
    # epsilon*=0.9
    z1 = inv_cl2(Ibis,alpha,beta,gamma,epsilon)
    S=0
    for m in range(N):
        S+=z1[0,m]
        S+=z1[-1,m]
        S+=z1[m,0]
        S+=z1[m,-1]
    S=S/(4*N)
    for m in range(N):
        for j in range(N):
            z1[m,j]-=S
    
    # fig = plt.figure(k)
    # ax = fig.gca(projection='3d')
    # ax.plot_surface(X,Y,z1,rstride=2,cstride=2,linewidth=1)
t1=clock()
E1=eclairement(z1,[alpha,beta,gamma],np.gradient)
t2=clock()
V1=np.sum(z1)
print("Eclairement calculé en :")
print(t2-t1)

#print(V)
print(V1)
#print(np.abs((V1-V)/V))

## Affichage des résultats

x = np.linspace(-nx/2,nx/2-1,nx)
y = np.linspace(-ny/2,ny/2-1,ny)
X,Y = np.meshgrid(y,x)

# fig = plt.figure(71)
# ax = fig.gca(projection='3d')
# ax.plot_surface(X,Y,Z,rstride=2,cstride=2,linewidth=1)

fig = plt.figure(72)
ax = fig.gca(projection='3d')
ax.plot_surface(X,Y,z1,rstride=2,cstride=2,linewidth=1)

# fig = plt.figure(3)
# ax = fig.gca(projection='3d')
# ax.plot_surface(X,Y,z2,rstride=2,cstride=2,linewidth=1)

plt.figure(74)
plt.imshow(I)

plt.figure(75)
plt.imshow(E1)

# plt.figure(6)
# plt.imshow(E2)

# E2=I

print(comparer_eclairement(I,E1))
# print(comparer_eclairement(I,E2))