import numpy as np
from numpy.linalg import norm

def frobenius(X):
    X = np.array(X)
    return np.sum(X**2)

def poweriteration(M, thres=1e-3):
    M = np.array(M)

    while abs(np.sum(xk1 - xk)) < thres:
        mul = np.matmul(M, xk)
        xk, xk1 = xk1.copy(), xk1 = mul / frobenius(mul)

def CUR_decomposition(M, r):
    p = (np.sum(M**2, axis=1)/norm(M)**2).flatten() # distribution of i
    q = (np.sum(M**2, axis=0)/norm(M)**2).flatten() # distribution of j
    
    i = np.random.choice(len(p), size=r, replace=True, p=p)
    j = np.random.choice(len(q), size=r, replace=True, p=q)

    C = M[:, j]
    R = M[i, :]

    C = C / np.sqrt(r*q[j][np.newaxis, ...])
    R = R / np.sqrt(r*p[i][..., np.newaxis])

    W = M[i, :][:, j]

    X, s, Yh = np.linalg.svd(W)
    S = np.diagflat(s)
    Sp = np.linalg.pinv(S)
    U = np.matmul(np.matmul(Yh.T, np.matmul(Sp, Sp)), X.T)
    
    return C, U, R