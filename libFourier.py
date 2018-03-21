import numpy as np
from numpy.fft import fftshift,fft2,ifft2
from scipy.fftpack import dst

def build_centered_indices(M,N):
    i = M//2 - (M//2 - np.arange(0,M)) % M  # (0, 1, ..., M/2, -M/2, ..., -1)
    j = N//2 - (N//2 - np.arange(0,N)) % N  # (0, 1, ..., M/2, -M/2, ..., -1)
    return np.meshgrid(i, j)
    
def gradient_tfd2(u):
    M,N = u.shape
    U = fft2(u);
    (ii, jj) = build_centered_indices(M, N)
    dx = (2j * np.pi / M) * ii
    dy = (2j * np.pi / N) * jj
    dxU = dx * U
    dyU = dy * U
    dxu = np.real(ifft2(dxU))
    dyu = np.real(ifft2(dyU))
    return np.array([dxu, dyu])

def inv_cl2(u,alpha,beta,gamma,epsilon):
    u=np.array(u)
    M,N = u.shape
    U = fft2(u)
    U[0,0]=U[0,0]-gamma
    (ii, jj) = build_centered_indices(N,M)
    D = (2*1j*np.pi)*((ii/N)*alpha + (jj/M)*beta) - 4*epsilon*np.pi**2*((ii/N)**2+(jj/N)**2)                   
    X=np.zeros((N,N),dtype=complex)
    for h in range(N):
        for k in range(N):
            if (h,k)==(0,0) or ii[0,h]==-jj[k,0]:
                X[h,k]=0
            else :
                X[h,k]=U[h,k]/D[h,k]
    return np.real(ifft2(X))
    
def eclairement_lin(Z, lV, grad):
    """
        renvoie la carte d'eclairement correspondant à la surface Z eclairee dans la direction lV, selon le gradient grad

        entree :
            Z : surface d'entree sous forme vectorielle (ie Z_{i,j} i < nx, j < ny --->  Z_j*nx + i, i < nx, j < ny)
            lV : vecteur unitaire dont la direction est celle du rayon incident
            grad : fonction calculant les composantes du gradient (sous forme vectorielle) de la matrice représentée par Z

        sortie :
            E : carte d'eclairement sous forme vectorielle
    """

    gradZx, gradZy = grad(Z)
    E = (lV[2] + lV[0] * gradZx + lV[1] * gradZy)
    # E = (lV[2] + lV[0]*gradZx +lV[1]*gradZy)
    
    return E
    
def dst2(u):
    f=[]
    M=len(u)
    for i in range(M):
        f.append(dst(u[i],1)/2)
    f=np.array(f)
    U=[]
    for i in range(M):
        U.append(dst(f[:,i],1)/2)
    return np.array(U)