import numpy as np
import pylab as plt
import scipy.misc as imageio
from libSFS import *
from libFourier import *
from scipy.signal import convolve2d
from mpl_toolkits.mplot3d import Axes3D
from numpy.fft import fftshift,fft2,ifft2
from scipy.fftpack import dct,dst,idct,idst

plt.rcParams["image.cmap"]="gray"
plt.ion()
plt.show()

## Paramètres de la simulation

nx=128
ny=128
# Paramètres du rayon incident
theta=np.pi/4 # Angle avec la normale -pi/2 < theta < pi/2
phi=np.pi/4
lV=direction_eclairement((theta,phi),(0,0))
(alpha,beta,gamma)=lV
delta=alpha/beta
epsilon=0

Z=generer_surface(Nx=nx, Ny=ny, forme=('cone',50,5), reg = 0)
V=sum(sum(Z))
I=eclairement(Z,lV,np.gradient)
N=I.shape[0]

i=fft2(I)
plt.figure(17)
i=fftshift(i)
plt.imshow(np.log(1+abs(i)))

## Résolution du problème

z1=inv_cl2(I,alpha,beta,gamma,epsilon) #solution brute
z2=inv_cl2(I,alpha,beta,gamma,epsilon) #solution corrigée (voir plus bas)

for y0 in range(1,N):
    i=0
    debut=z2[0,y0]
    if int(y0+delta*(N-1))<N:
        fin=z2[N-1,int(y0+delta*(N-1))]
        while i<N :
            z2[i,int(y0+delta*i)]=z2[i,int(y0+delta*i)]-(debut+(fin-debut)*(i/(N-1)))
            i=i+1
    else :
        n=int((N-y0)/delta)
        fin=z2[n,N-1]
        while int(y0+delta*i)<N :
            z2[i,int(y0+delta*i)]=z2[i,int(y0+delta*i)]-(debut+(fin-debut)*(i/n))
            i=i+1

for x0 in range(1,N):
    i=0
    debut=z2[x0,0]
    if int(x0+(N-1)/delta)<N:
        fin=z2[N-1,int(x0+(N-1)/delta)]
        while i<N :
            z2[int(x0+i/delta),i]=z2[int(x0+i/delta),i]-(debut+(fin-debut)*(i/(N-1)))
            i=i+1
    else :
        n=int(delta*(N-x0))
        fin=z2[N-1,n]
        if n>0 :
            while int(x0+i/delta)<N :
                z2[int(x0+i/delta),i]=z2[int(x0+i/delta),i]-(debut+(fin-debut)*(i/n))
                i=i+1
                
z2[0:N,0]=0
z2[0:N,N-1]=0
z2[0,0:N]=0
z2[N-1,0:N]=0        

E1=eclairement(z1,[alpha,beta,gamma],np.gradient)
E2=eclairement(z2,[alpha,beta,gamma],np.gradient)

V1=np.sum(z1)
V2=np.sum(z2)

print(V)
print(V1)
print(V2)
print(np.abs((V1-V)/V))
print(np.abs((V2-V)/V))
    
## Affichage des résultats

x = np.linspace(-nx/2,nx/2-1,nx)
y = np.linspace(-ny/2,ny/2-1,ny)
X,Y = np.meshgrid(y,x)

fig = plt.figure(1)
ax = fig.gca(projection='3d')
ax.plot_surface(X,Y,Z,rstride=2,cstride=2,linewidth=1)

fig = plt.figure(2)
ax = fig.gca(projection='3d')
ax.plot_surface(X,Y,z1,rstride=2,cstride=2,linewidth=1)

fig = plt.figure(3)
ax = fig.gca(projection='3d')
ax.plot_surface(X,Y,z2,rstride=2,cstride=2,linewidth=1)

plt.figure(4)
plt.imshow(I)

plt.figure(5)
plt.imshow(E1)

plt.figure(6)
plt.imshow(E2)