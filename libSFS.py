# -*- coding: utf-8 -*-

import numpy as np
import numpy.random as rand
from scipy.signal import convolve2d
from copy import deepcopy


def direction_eclairement(angLum, angObs):

    thL, phiL = angLum
    thO, phiO = angObs
    lV = np.array([np.sin(thL) * np.cos(phiL), np.sin(thL)
                   * np.sin(phiL), np.cos(thL)])

    c = np.cross(angObs, (0, 0, 1))
    lVapp = np.cos(thO) * lV + (1 - np.cos(thO)) * \
        (np.dot(lV, c)) * c + np.sin(thO) * (np.cross(c, lV))

    return lVapp


def generer_surface(Nx=64, Ny=64, forme=('plateau', 16, 16, 1), reg=0):
    """
        genere une surface centree du type specifie et en renvoie le profil de hauteur

        entree :
            Nx,Ny : dimension de l'image generee
            forme : (type, caracteristiques...) ; par defaut : plateau
                - ('plateau', sigx, sigy, H) : fonction plateau de hauteur H, de dimension 2sigx*2sigy
                - ('cone', r, h) : cone de base de rayon r, de hauteur h
                - ('trap', L, r, H, h) : trapezoide conique de base de dimension (L,l), de facteur de reduction k, de hauteur h 
                - ('volcan', sigx, sigy, H, p, k) : volcan de base d'ecart type sigx, sigy, de hauteur H, de profondeur p, de trou = base k-contractee

            reg : sigma : faut-il regulariser la surface ? (convolution par une gaussienne centree d'ecart-type reg) si reg = 0 : pas de regularisation


        sortie :
            Z : profil de hauteur
    """

    # Generation de la surface

    xt = np.linspace(-Nx / 2, Nx / 2 - 1, Nx)
    yt = np.linspace(-Ny / 2, Ny / 2 - 1, Ny)
    X, Y = np.meshgrid(yt, xt)
    Z = np.zeros((Nx, Ny))

    if forme[0] == 'plateau':
        sigx, sigy, H = forme[1:]
        T = (np.abs(X) < sigx) * (np.abs(Y) < sigy)
        if 2 * sigx > Nx or 2 * sigy > Ny:
            raise ValueError('Surface trop large')
        # Z = H*np.exp(-(1/(sigx**2-X**2))-(1/(sigy**2-Y**2)))*T
        Z[Nx // 2 - sigx + 1:Nx // 2 + sigx - 1, Ny // 2 - sigy + 1:Ny // 2 + sigy - 1] = H * np.exp(-(1 / (sigx**2 - X[Nx // 2 - sigx + 1:Nx // 2 + sigx - 1, Ny // 2 - sigy + 1:Ny // 2 + sigy - 1]**2)) - (
            1 / (sigy**2 - Y[Nx // 2 - sigx + 1:Nx // 2 + sigx - 1, Ny // 2 - sigy + 1:Ny // 2 + sigy - 1]**2)))

    elif forme[0] == 'cone':
        r, h = forme[1:]
        if 2 * r > min(Nx, Ny):
            raise ValueError('Surface trop large')
        Z = h * (r - (X**2 + Y**2)**.5) / r
        Z = Z * (Z > 0)

    elif forme[0] == 'trap':
        L, r, H, h = forme[1:]
        if L + 2 * r > min(Nx, Ny):
            raise ValueError('Surface trop large')

        rb = r

        ru = rb * (H - h) / H
        Z_tmp = np.zeros((Nx, Ny))

        Z = H * (rb - ((X + L / 2)**2 + Y**2)**.5) / rb
        Z = Z * (Z > 0)
        Z[:, Ny / 2 - L / 2:] = 0
        Z_tmp = H * (rb - ((X - L / 2)**2 + Y**2)**.5) / rb
        Z_tmp = Z_tmp * (Z_tmp > 0)
        Z_tmp[:, :Ny / 2 + L / 2] = 0

        Z = Z + Z_tmp

        Z[Nx / 2:Nx / 2 + rb, Ny / 2 - L / 2:Ny / 2 + L / 2] = H * \
            (rb - Y[Nx / 2:Nx / 2 + rb, Ny / 2 - L / 2:Ny / 2 + L / 2]) / rb
        Z[Nx / 2 - rb:Nx / 2, Ny / 2 - L / 2:Ny / 2 + L / 2] = H * \
            (Y[Nx / 2:Nx / 2 + rb, Ny / 2 - L / 2:Ny / 2 + L / 2]) / rb

        Z = Z - (Z - h) * (Z >= h)

    elif forme[0] == 'volcan':
        sigx, sigy, H, p, k = forme[1:]
        Z_trou = np.zeros((Nx, Ny))
        T = (np.abs(X) < sigx) * (np.abs(Y) < sigy)
        Tk = (np.abs(X) < k * sigx) * (np.abs(Y) < k * sigy)
        if 2 * sigx > Nx or 2 * sigy > Ny:
            raise ValueError('Surface trop large')
        # Z = H*np.exp(-(1/(sigx**2-X**2))-(1/(sigy**2-Y**2)))*T

        Z[Nx / 2 - sigx + 1:Nx / 2 + sigx - 1, Ny / 2 - sigy + 1:Ny / 2 + sigy - 1] = H * np.exp(-(1 / (sigx**2 - X[Nx / 2 - sigx + 1:Nx / 2 + sigx - 1, Ny / 2 - sigy + 1:Ny / 2 + sigy - 1]**2)) - (
            1 / (sigy**2 - Y[Nx / 2 - sigx + 1:Nx / 2 + sigx - 1, Ny / 2 - sigy + 1:Ny / 2 + sigy - 1]**2)))

        Z_trou[Nx / 2 - k * sigx + 1:Nx / 2 + k * sigx - 1, Ny / 2 - k * sigy + 1:Ny / 2 + k * sigy - 1] = p * np.exp(-(1 / ((k * sigx)**2 - X[Nx / 2 - k * sigx + 1:Nx / 2 + k * sigx - 1, Ny / 2 - k * sigy + 1:Ny / 2 + k * sigy - 1]**2)) - (
            1 / ((k * sigy)**2 - Y[Nx / 2 - k * sigx + 1:Nx / 2 + k * sigx - 1, Ny / 2 - k * sigy + 1:Ny / 2 + k * sigy - 1]**2)))

        # Z_trou = p*np.exp(-(1/((k*sigx)**2-X**2))-(1/((k*sigy)**2-Y**2)))*Tk
        Z = Z - Z_trou

    # Regularisation eventuelle

    if reg != 0:
        G = 1 / (2 * np.pi * reg)**.5 * \
            np.exp(-((X)**2 + (Y)**2) / (2 * reg**2))
        Z = convolve2d(Z, G, 'same')

    return Z


def eclairement(Z, lV, grad):
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
    E = (lV[2] + lV[0] * gradZx + lV[1] * gradZy) / np.sqrt(1 + gradZx ** 2 + gradZy ** 2)
    # E = (lV[2] + lV[0]*gradZx +lV[1]*gradZy)

    return E


def comparer_eclairement(E1, E2):
    """
        renvoie la difference en norme 1, en norme 2 et en norme uniforme entre E1 et E2, avec E1 comme reference
    """
    N1ref = np.sum(np.abs(E1))
    N2ref = np.sum(np.abs(E1 ** 2)) ** .5
    N1 = np.sum(np.abs(E1 - E2)) / N1ref
    N2 = ((np.sum(np.abs((E1 - E2)) ** 2)) ** (1.0 / 2)) / N2ref
    N_uni = np.max(np.abs(E1 - E2)) / np.max(np.abs(E1))

    return N1, N2, N_uni


def bruit_gaussien(I, sigma):

    bruit = sigma * rand.standard_normal(I.shape)
    return I + bruit

def bruit_selpoivre(I, freq):

    bruit = rand.randint(int(1/freq) + 1, size = I.shape)
    indexP = np.where(bruit == 0)
    indexS = np.where(bruit == int(1/freq) + 1)
    I[indexP] = 0
    I[indexS] = 1
    return I

def simul_camera(I, (nx, ny), patch):

    I_mat = np.reshape(I, (nx, ny))

    for i in range(nx // patch):
        for j in range(ny // patch):
            m = np.sum(I_mat[patch*i:patch*(i+1), patch*j:patch*(j+1)]) / (patch **2)
            # print(I_mat[patch*i:patch*(i+1), patch*j:patch*(j+1)])
            # print(patch**2)
            # print(m)
            for k in range(patch):
                for l in range(patch):
                    I_mat[patch * i + k, patch*j + l] = m
    return np.reshape(I_mat, nx*ny)

    
def points_critiques(E):
    """
        renvoie un tableau de bouléens qui indiquent les points critiques associés à la carte d'éclairement E 
    """
    (nx,ny)=E.shape
    P=np.zeros((nx,ny))
    for i in range(1,nx-1):
        for j in range(1,ny-1):
            if E[i,j]==1:
                P[i,j]=1
    return P
    
def comp_connexes(P):
    """
        renvoie la liste des composantes connexes de {(x,y)|P[x,y]=1}
    """
    R=deepcopy(P)
    Q=[]
    L=[]
    (nx,ny)=R.shape
    for i in range(nx):
        for j in range(ny):
            if R[i,j]==1:
                L.append([i,j])
    h=0
    while len(L)>0:
        C=[]
        (x,y)=L[0]
        while (frontiere(C)>frontiere(R)).any():
            
        for (i,j) in Q[h]:
            R[i,j]=0
        L=[]
        for i in range(nx):
            for j in range(ny):
                if R[i,j]==1:
                    L.append([i,j])
        h+=1
    return np.array(Q)
    
def frontiere(Q):
    """
        renvoie les coordonnées des points de la frontière de Q
    """
    L=[]
    (nx,ny)=Q.shape
    for i in range(nx):
        for j in range(ny):
            if Q[i,j]==1:
                L.append([i,j])
    F=[]
    for i in range(nx):
        for j in range(ny):
            if ([i,j] not in L) and ([i-1,j] in L or [i-1,j-1] in L or [i,j-1] in L or [i+1,j-1] in L or [i+1,j] in L or [i+1,j+1] in L or [i,j+1] in L or [i-1,j+1] in L):
                F.append([i,j])
    G=np.zeros((nx,ny))
    for i in range(nx):
        for j in range(ny):
            if [i,j] in F:
                G[i,j]=1
    return G
    
def voisinage(P,Q,x,y):
    """
        ajoute à Q les points voisins de [x,y] qui appartiennent à P
    """
    (nx,ny)=P.shape
    if x>0 and P[x-1,y]==1 and [x-1,y] not in Q:
        Q.append([x-1,y])
        Q=voisinage(P,Q,x-1,y)
    if x>0 and y>0 and P[x-1,y-1]==1 and [x-1,y-1] not in Q:
        Q.append([x-1,y-1])
        Q=voisinage(P,Q,x-1,y)
    if y>0 and P[x,y-1]==1 and [x,y-1] not in Q:
        Q.append([x,y-1])
        Q=voisinage(P,Q,x,y-1)
    if x<nx-1 and y>0 and P[x+1,y-1]==1 and [x+1,y-1] not in Q:
        Q.append([x+1,y-1])
        Q=voisinage(P,Q,x+1,y-1)
    if x<nx-1 and P[x+1,y]==1 and [x+1,y] not in Q:
        Q.append([x+1,y])
        Q=voisinage(P,Q,x+1,y)
    if x<nx-1 and y<ny-1 and P[x+1,y+1]==1 and [x+1,y+1] not in Q:
        Q.append([x+1,y+1])
        Q=voisinage(P,Q,x+1,y+1)
    if y<ny-1 and P[x,y+1]==1 and [x,y+1] not in Q:
        Q.append([x,y+1])
        Q=voisinage(P,Q,x,y+1)
    if x>0 and y<ny-1 and P[x-1,y+1]==1 and [x-1,y+1] not in Q:
        Q.append([x-1,y+1])
        Q=voisinage(P,Q,x-1,y+1)
    return Q