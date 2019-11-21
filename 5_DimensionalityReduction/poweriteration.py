import numpy as np

def frobenius(X):
    X = np.array(X)
    return np.sum(X**2)

def poweriteration(M, thres=1e-3):
    M = np.array(M)

    while abs(np.sum(xk1 - xk)) < thres:
        mul = np.matmul(M, xk)
        xk, xk1 = xk1.copy(), xk1 = mul / frobenius(mul)

M = np.array([[1, 1, 1], [1, 2, 3], [1, 3, 6]])
x0 = np.array([1, 1, 1])

