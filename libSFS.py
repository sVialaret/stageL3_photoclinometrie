# -*- coding: utf-8 -*-

import numpy as np
import numpy.random as rand
from scipy.signal import convolve2d
from copy import deepcopy
import scipy.sparse as sp


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

    xt = np.linspace(-Nx // 2, Nx // 2 - 1, Nx)
    yt = np.linspace(-Ny // 2, Ny // 2 - 1, Ny)
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
        Z[:, Ny // 2 - L // 2:] = 0
        Z_tmp = H * (rb - ((X - L / 2)**2 + Y**2)**.5) / rb
        Z_tmp = Z_tmp * (Z_tmp > 0)
        Z_tmp[:, :Ny // 2 + L // 2] = 0

        Z = Z + Z_tmp

        Z[Nx // 2:Nx // 2 + rb, Ny // 2 - L // 2:Ny // 2 + L // 2] = H * \
            (rb - Y[Nx // 2:Nx // 2 + rb, Ny // 2 - L // 2:Ny // 2 + L // 2]) / rb
        Z[Nx // 2 - rb:Nx // 2, Ny // 2 - L // 2:Ny // 2 + L // 2] = H * \
            (Y[Nx // 2:Nx // 2 + rb, Ny // 2 - L // 2:Ny // 2 + L // 2]) / rb

        Z = Z - (Z - h) * (Z >= h)

    elif forme[0] == 'volcan':
        sigx, sigy, H, p, k = forme[1:]
        Z_trou = np.zeros((Nx, Ny))
        T = (np.abs(X) < sigx) * (np.abs(Y) < sigy)
        Tk = (np.abs(X) < k * sigx) * (np.abs(Y) < k * sigy)
        if 2 * sigx > Nx or 2 * sigy > Ny:
            raise ValueError('Surface trop large')
        # Z = H*np.exp(-(1/(sigx**2-X**2))-(1/(sigy**2-Y**2)))*T

        Z[Nx // 2 - sigx + 1:Nx // 2 + sigx - 1, Ny // 2 - sigy + 1:Ny // 2 + sigy - 1] = H * np.exp(-(1 / (sigx**2 - X[Nx // 2 - sigx + 1:Nx // 2 + sigx - 1, Ny // 2 - sigy + 1:Ny // 2 + sigy - 1]**2)) - (
            1 / (sigy**2 - Y[Nx // 2 - sigx + 1:Nx // 2 + sigx - 1, Ny // 2 - sigy + 1:Ny // 2 + sigy - 1]**2)))

        Z_trou[Nx // 2 - int(k * sigx) + 1:Nx // 2 + int(k * sigx) - 1, Ny // 2 - int(k * sigy) + 1:Ny // 2 + int(k * sigy) - 1] = p * np.exp(-(1 / (int(k * sigx)**2 - X[Nx // 2 - int(k * sigx) + 1:Nx // 2 + int(k * sigx) - 1, Ny // 2 - int(k * sigy) + 1:Ny // 2 + int(k * sigy) - 1]**2)) - (
            1 / (int(k * sigy)**2 - Y[Nx // 2 - int(k * sigx) + 1:Nx // 2 + int(k * sigx) - 1, Ny // 2 - int(k * sigy) + 1:Ny // 2 + int(k * sigy) - 1]**2)))

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

# def simul_camera(I, (nx, ny), patch):
#     
#     I_mat = np.reshape(I, (nx, ny))
# 
#     for i in range(nx // patch):
#         for j in range(ny // patch):
#             m = np.sum(I_mat[patch*i:patch*(i+1), patch*j:patch*(j+1)]) / (patch **2)
#             # print(I_mat[patch*i:patch*(i+1), patch*j:patch*(j+1)])
#             # print(patch**2)
#             # print(m)
#             for k in range(patch):
#                 for l in range(patch):
#                     I_mat[patch * i + k, patch*j + l] = m
#     return np.reshape(I_mat, nx*ny)

    
def points_critiques(E):
    """
        renvoie un tableau de booléens qui indiquent les points critiques associés à la carte d'éclairement E
    """
    (nx,ny)=E.shape
    P=np.zeros((nx,ny))
    for i in range(1,nx-1):
        for j in range(1,ny-1):
            if abs(E[i,j]-1) < 0.0001:
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
        C0=np.zeros((nx,ny))
        C=np.zeros((nx,ny))
        (x,y)=L[0]
        C[x,y]=1
        while (frontiere(C)>frontiere(R)).any():
            S=C-C0
            C0=deepcopy(C)
            M=[]
            (nx,ny)=R.shape
            for i in range(nx):
                for j in range(ny):
                    if S[i,j]==1:
                        M.append([i,j])
            for (x,y) in M:
                C=voisinage(R,C,x,y)
            Q.append(C)
               
            for (i,j) in Q[h]:
                R[i,j]=0
            L=[]
            for i in range(nx):
                for j in range(ny):
                    if R[i,j]==1:
                        L.append([i,j])
            h+=1
    return np.array(Q)
    
def frontiere(K):
    """
        renvoie la carte de la frontière de K
    """
    (nx,ny)=K.shape
    
    Dx1 = sp.lil_matrix((nx, nx))
    Dx1.setdiag(-1)
    Dx1.setdiag(1,1)
    Dx1 = Dx1.toarray()
    
    Dx2 = sp.lil_matrix((nx, nx))
    Dx2.setdiag(1)
    Dx2.setdiag(-1,-1)
    Dx2 = Dx2.toarray()
    
    Dy1 = sp.lil_matrix((ny, ny))
    Dy1.setdiag(-1)
    Dy1.setdiag(1,1)
    Dy1 = Dy1.toarray()
    
    Dy2 = sp.lil_matrix((ny, ny))
    Dy2.setdiag(1)
    Dy2.setdiag(-1,-1)
    Dy2 = Dy2.toarray()
    
    U1=(np.dot(Dx1,K)>0)
    U2=(np.dot(Dx2,K)<0)
    
    V1=(np.dot(K,Dy1)>0)
    V2=(np.dot(K,Dy2)<0)
    
    return U1+U2+V1+V2
    
def voisinage(P,C,x,y):
    """
        ajoute à C les points voisins de [x,y] qui appartiennent à P
    """
    (nx,ny)=P.shape
    i=x
    while i>0 and P[i-1,y]==1 and C[i-1,y]==0:
        C[i-1,y]=1
        i=i-1
    j=y
    while j>0 and P[x,j-1]==1 and C[x,j-1]==0:
        C[x,j-1]=1
        j=j-1
    i=x
    while i<nx-1 and P[i+1,y]==1 and C[i+1,y]==0:
        C[i+1,y]=1
        i=i+1
    j=y
    while j<ny-1 and P[x,j+1]==1 and C[x,j+1]==0:
        C[x,j+1]=1
        j=j+1
    return C
    
def rearrange(CC):
    """
        classe les composantes connexes par ordre de proximité au bord
    """
    if len(CC)==0:
        return CC
    D=[]
    (nx,ny)=CC[0].shape
    C=np.zeros((len(CC),nx,ny))
    for i in range(len(CC)):
        L=[]
        F=frontiere(CC[i])
        for x in range(nx):
            for y in range(ny):
                if F[x,y]:
                    L.append([x,y])
        l=[]
        for (x,y) in L:
            l.append(x)
            l.append(nx-x)
            l.append(y)
            l.append(ny-y)
        D.append(min(l))
    for i in range(len(CC)):
        B=max(D)+1
        j=np.argmin(D)
        D[j]=B
        C[i]=CC[j]
    return C
    
def h(x,y,c,Q,V,CC,n,CB,P,p):
    """
        calcule la hauteur du point (x,y)
    """
    (nx,ny)=CC[0].shape
    M=0
    i=x
    j=y
    if c==0:
        while Q[x,j]==0 and j>0:
            M+=n[x,j]
            j=j-1
        if j==0:
            M+=CB[0][x]
        else:
            k=0
            while CC[k,x,j]==0:
                k=k+1
            M-=V[k]*P[k]
    elif c==1:
        while Q[x,j]==0 and j<ny-1:
            M+=n[x,j]
            j=j+1
        if j==ny-1:
            M+=CB[1][x]
        else:
            k=0
            while CC[k,x,j]==0:
                k=k+1
            M-=V[k]*P[k]
    elif c==2:
        while Q[i,y]==0 and i>0:
            M+=n[i,y]
            i=i-1
        if i==0:
            M+=CB[2][y]
        else:
            k=0
            while CC[k,i,y]==0:
                k=k+1
            M-=V[k]*P[k]
    else:
        while Q[i,y]==0 and i<nx-1:
            M+=n[i,y]
            i=i+1
        if i==nx-1:
            M+=CB[3][y]
        else:
            k=0
            while CC[k,i,y]==0:
                k=k+1
            M-=V[k]*P[k]
    return p*M
    
    
def height(i,Q,V,CC,n,CB,P):
    """
        calcule la hauteur de l'ensemble CC[i]
    """
    p=P[i]
    K=CC[i]
    (nx,ny)=K.shape
    F=frontiere(K)
    L=[]
    H=[]
    R=deepcopy(Q)
    for j in range(i):
        for x in range(nx):
            for y in range(ny):
                if CC[j,x,y]:
                    R[x,y]=0
    for x in range(nx):
        for y in range(ny):
            if F[x,y]:
                L.append([x,y])
    for (x,y) in L:
        if (R[x,0:y]==np.zeros(y)).all():
            H.append(h(x,y,0,Q,V,CC,n,CB,P,p))
        if (R[x,y:ny]==np.zeros(ny-y)).all():
            H.append(h(x,y,1,Q,V,CC,n,CB,P,p))
        if (R[0:x,y]==np.zeros(x)).all():
            H.append(h(x,y,2,Q,V,CC,n,CB,P,p))
        if (R[x:nx,y]==np.zeros(nx-x)).all():
            H.append(h(x,y,3,Q,V,CC,n,CB,P,p))
    if p==1:
        return min(H)
    else:
        return max(H)