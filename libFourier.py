import numpy as np
from numpy.fft import fftshift,fft2,ifft2

def build_centered_indices(M,N):
    i = M//2 - (M//2 - np.arange(0,M)) % M  # (0, 1, ..., M/2, -M/2, ..., -1)
    j = N//2 - (N//2 - np.arange(0,N)) % N  # (0, 1, ..., M/2, -M/2, ..., -1)
    return np.meshgrid(i, j)

def inv_cl2(u,alpha,beta,gamma,epsilon):
    u=np.array(u)
    M,N = u.shape
    U = fft2(u)
    U[0,0]=U[0,0]-gamma
    (ii, jj) = build_centered_indices(N,M)
    D = (2*1j*np.pi)*((ii/N)*alpha + (jj/M)*beta) + epsilon*(-4*(np.pi)**2*((ii/N)**2+(jj/M)**2))
    D[0,0]=1
    for h in range(N):
        for k in range(N):
            if D[h,k]==0:
                D[h,k]=1
    X = U/D
    return np.real(ifft2(X))