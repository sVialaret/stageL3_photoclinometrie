# -*- coding: utf-8 -*-

import numpy as np
import pylab as plt
import scipy.misc as imageio
from scipy.signal import convolve2d
from numpy.fft import fftshift,fft2,ifft2

def build_centered_indices(M,N):
    i = M//2 - (M//2 - np.arange(0,M)) % M  # (0, 1, ..., M/2, -M/2, ..., -1)
    j = N//2 - (N//2 - np.arange(0,N)) % N  # (0, 1, ..., M/2, -M/2, ..., -1)
    return np.meshgrid(i, j)

def gradient_tfd2(u): # 2D -> 2D
    M,N = u.shape
    U = fft2(u);
    (ii, jj) = build_centered_indices(M,N)
    dx = (2j * np.pi / M) * ii
    dy = (2j * np.pi / N) * jj
    dxU = dx * U
    dyU = dy * U
    dxu = np.real(ifft2(dxU))
    dyu = np.real(ifft2(dyU))
    return np.array([dxu, dyu])
    
def integr_x(u): # 2D -> 2D
    u=np.array(u)
    M,N = u.shape
    U = fft2(u);
    (ii, jj) = build_centered_indices(M,N)
    dx = (2j * np.pi / M) * ii
    dx[:,0]=1
    dxU =  U / dx
    dxu = np.real(ifft2(dxU))
    return dxu
    
def integr_y(u): # 2D -> 2D
    u=np.array(u)
    M,N = u.shape
    U = fft2(u);
    (ii, jj) = build_centered_indices(M,N)
    dy = (2j * np.pi / N) * jj
    dy[0,:]=1
    dyU = U / dy
    dyu = np.real(ifft2(dyU))
    return dyu

def inv_cl(u,alpha,beta,gamma):
    u=np.array(u)
    M,N = u.shape
    U = fft2(u)
    U[0,0]=U[0,0]-gamma
    (ii, jj) = build_centered_indices(N,M)
    D = (2*1j*np.pi)*((ii/N)*alpha + (jj/M)*beta)
    D[0,0]=1
    for h in range(N):
        for k in range(N):
            if D[h,k]==0:
                D[h,k]=1
    X = U/D
    return np.real(ifft2(X))
    
def creer_cl(u,alpha,beta,gamma):
    u=np.array(u)
    M,N = u.shape
    U = fft2(u)
    U[0,0]=U[0,0]-gamma
    (ii, jj) = build_centered_indices(N,M)
    X = U*(2*1j*np.pi)*((ii/N)*alpha + (jj/M)*beta)
    return np.real(ifft2(X))
    
def laplacien_per_dft2(u):
    M,N = u.shape
    U = fft2(u);
    (ii, jj) = build_centered_indices(N,M)
    dx = (2j * np.pi / N) * ii
    dy = (2j * np.pi / M) * jj
    dxU = dx**2 * U
    dyU = dy**2 * U
    Du = np.real(ifft2(dxU+dyU))
    return Du
    
def laplacien_sym_dft2(u):
    M,N = u.shape
    x=miroir(u)
    Dx=laplacien_per_dft2(x)
    return Dx[0:M,0:N]

def inv_cl2(u,alpha,beta,gamma,epsilon):
    u=np.array(u)
    M,N = u.shape
    U = fft2(u)
    print(np.absolute(U) < 10**-5)
    print(U)
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

def inv_cl_real(u,alpha,beta,gamma):
    u=np.array(u)
    M,N = u.shape
    U = fft2(u)
    U[0,0]=U[0,0]-gamma
    (ii, jj) = build_centered_indices(N,M)
    D = (2*1j*np.pi)*((ii/N)*alpha + (jj/M)*beta)
    D[0,0]=1
    for h in range(N):
        for k in range(N):
            if D[h,k]==0:
                D[h,k]=1
    X = U/D
    X=np.real(X)
    return np.real(ifft2(X))
    
def inv_cl_imag(u,alpha,beta,gamma):
    u=np.array(u)
    M,N = u.shape
    U = fft2(u)
    U[0,0]=U[0,0]-gamma
    (ii, jj) = build_centered_indices(N,M)
    D = (2*1j*np.pi)*((ii/N)*alpha + (jj/M)*beta)
    D[0,0]=1
    for h in range(N):
        for k in range(N):
            if D[h,k]==0:
                D[h,k]=1
    X = U/D
    X=1j*np.imag(X)
    return np.real(ifft2(X))
    
def estimation_volume(I,z,alpha,beta,gamma):
    V=-gamma*(alpha+beta)/(2*(alpha**2+beta**2))
    L=laplacien_per_dft2(z)
    for k in range(N):
        for m in range(N):
            V+=((alpha*(1-k/N)+beta*(1-m/N))*I[k,m]-(1-k/N)*(1-m/N)*L[k,m])/(alpha**2+beta**2)
    return V

def generer_surface(Nx = 64, Ny = 64, forme = ('plateau', 16, 16, 1), reg = 0, lV = (0,0), obV = (0,0)):
    
    """
        genere une surface centree du type specifie, renvoie le profil de hauteur, la carte d'intensite sour un eclairement d'angle ang_ecl, et le volume de l'objet
        
        entree :
            Nx,Ny : dimension de l'image generee
            forme : (type, caracteristiques...) ; par defaut : plateau
                - ('plateau', sigx, sigy, H) : fonction plateau de hauteur H, de dimension 2sigx*2sigy
                - ('cone', r, h) : cone de base de rayon r, de hauteur h
                - ('trap', L, r, H, h) : trapezoide conique de base de dimension (L,l), de facteur de reduction k, de hauteur h 
                - ('volcan', sigx, sigy, H, p, k) : volcan de base d'ecart type sigx, sigy, de hauteur H, de profondeur p, de trou = base k-contractee
                
            reg : sigma : faut-il regulariser la surface ? (convolution par une gaussienne centree d'ecart-type reg) si reg = 0 : pas de regularisation
            lV : (theta,phi) : coordonnees spheriques d'un vecteur unitaire donnant la direction de la lumière
            obV : (theta_o, phi_o) : coordonnees spheriques d'un vecteur unitaire donnant la direction de l'observateur
        
        sortie :
            Z : profil de hauteur
            E : carte d'eclairement
            V : volume du tas
    """
    
    # Generation de la surface
    
    xt = np.linspace(-Nx/2,Nx/2-1,Nx)
    yt = np.linspace(-Ny/2,Ny/2-1,Ny)
    X,Y = np.meshgrid(yt,xt)
    Z = np.zeros((Nx,Ny))

    
    if forme[0] == 'plateau':
        sigx,sigy,H = forme[1:]
        T = (np.abs(X) < sigx) * (np.abs(Y) < sigy)
        if 2*sigx > Nx or 2*sigy > Ny:
            raise ValueError('Surface trop large')
        # Z = H*np.exp(-(1/(sigx**2-X**2))-(1/(sigy**2-Y**2)))*T
        Z[Nx/2-sigx+1:Nx/2+sigx-1,Ny/2 - sigy+1:Ny/2 + sigy-1] = H*np.exp(-(1/(sigx**2-X[Nx/2-sigx+1:Nx/2+sigx-1,Ny/2 - sigy+1:Ny/2 + sigy-1]**2))-(1/(sigy**2-Y[Nx/2-sigx+1:Nx/2+sigx-1,Ny/2 - sigy+1:Ny/2 + sigy-1]**2)))
        
    
    elif forme[0] == 'cone':
        r,h = forme[1:]
        if 2*r > min(Nx,Ny):
            raise ValueError('Surface trop large')
        Z = h*(r-(X**2+Y**2)**.5)/r
        Z = Z * (Z >0)
    
    elif forme[0] == 'trap':
        L,r,H,h = forme[1:]
        if L+2*r > min(Nx,Ny):
            raise ValueError('Surface trop large')
        
        rb = r
        ru = rb*(H-h)/H
        Z_tmp = np.zeros((Nx,Ny))
        
        Z = H*(rb-((X+L/2)**2+Y**2)**.5)/rb
        Z = Z * (Z>0)
        Z[:,Ny/2-L/2:] = 0
        Z_tmp = H*(rb-((X-L/2)**2+Y**2)**.5)/rb
        Z_tmp = Z_tmp * (Z_tmp>0)
        Z_tmp[:,:Ny/2+L/2] = 0
        
        Z = Z + Z_tmp
        
        Z[Nx/2:Nx/2+rb,Ny/2 - L/2:Ny/2 + L/2] = H*(rb-Y[Nx/2:Nx/2+rb,Ny/2 - L/2:Ny/2 + L/2])/rb
        Z[Nx/2-rb:Nx/2,Ny/2 - L/2:Ny/2 + L/2] = H*(Y[Nx/2:Nx/2+rb,Ny/2 - L/2:Ny/2 + L/2])/rb
        
        Z = Z - (Z-h) * (Z>=h)
    
    elif forme[0] == 'volcan':
        sigx,sigy,H,p,k = forme[1:]
        Z_trou = np.zeros((Nx,Ny))
        T = (np.abs(X) < sigx) * (np.abs(Y) < sigy)
        Tk = (np.abs(X) < k*sigx) * (np.abs(Y) < k*sigy)
        if 2*sigx > Nx or 2*sigy > Ny:
            raise ValueError('Surface trop large')
        # Z = H*np.exp(-(1/(sigx**2-X**2))-(1/(sigy**2-Y**2)))*T
        
        Z[Nx/2-sigx+1:Nx/2+sigx-1,Ny/2 - sigy+1:Ny/2 + sigy-1] = H*np.exp(-(1/(sigx**2-X[Nx/2-sigx+1:Nx/2+sigx-1,Ny/2 - sigy+1:Ny/2 + sigy-1]**2))-(1/(sigy**2-Y[Nx/2-sigx+1:Nx/2+sigx-1,Ny/2 - sigy+1:Ny/2 + sigy-1]**2)))
        
        Z_trou[Nx/2-k*sigx+1:Nx/2+k*sigx-1,Ny/2 - k*sigy+1:Ny/2 + k*sigy-1] = p*np.exp(-(1/((k*sigx)**2-X[Nx/2-k*sigx+1:Nx/2+k*sigx-1,Ny/2 - k*sigy+1:Ny/2 + k*sigy-1]**2))-(1/((k*sigy)**2-Y[Nx/2-k*sigx+1:Nx/2+k*sigx-1,Ny/2 - k*sigy+1:Ny/2 + k*sigy-1]**2)))
        
        # Z_trou = p*np.exp(-(1/((k*sigx)**2-X**2))-(1/((k*sigy)**2-Y**2)))*Tk
        Z = Z-Z_trou
        
        
        
    
    # Regularisation eventuelle
    
    if reg != 0:
        G = 1/(2*np.pi*reg)**.5 * np.exp(-((X)**2 + (Y)**2)/(2*reg**2))
        Z = convolve2d(Z,G,'same')
    
    
    # Generation de la carte d'eclairement
    
    th, ph = lV
    thO, phO = obV
    lV = np.array([np.sin(th)*np.cos(ph),np.sin(th)*np.sin(ph), np.cos(th)])
    
    c = np.cross(obV,(0,0,1))
    lVapp = np.cos(thO)*lV + (1-np.cos(thO))*(np.dot(lV,c))*c + np.sin(thO)*(np.cross(c,lV))
    
    E = eclairement(Z,lVapp)
    
    # Calcul du volume
    
    V = np.sum(Z)
    
    return Z,E,V



def eclairement(Z,lV):
    gradZx,gradZy = np.gradient(Z)
    E = (lV[2] + lV[0]*gradZx +lV[1]*gradZy)/(1+gradZx**2 + gradZy**2)
    # E = (lV[2] + lV[0]*gradZx +lV[1]*gradZy)
    
    return E


def comparer_eclairement(E1,E2):
    Gx, Gy = np.gradient(E1-E2)
    Gzx, Gzy = np.gradient(E1)
    N1ref = np.sum(np.abs(E1))
    N2ref = np.sum(np.abs(E1**2))**.5
    N1 = np.sum(np.abs(E1-E2))/N1ref
    N2 = ((np.sum(np.abs((E1-E2))**2))**(1.0/2))/N2ref
    sig2 = np.abs(N2**2 - N1**2)**.5
    S = np.sum(np.abs(E1)) + np.sum(np.abs(Gzx)) + np.sum(np.abs(Gzy))
    S1 = (np.sum(np.abs(E1-E2)) + np.sum(np.abs(Gx)) + np.sum(np.abs(Gy)))/S
    
    return N1,N2,sig2,S1
