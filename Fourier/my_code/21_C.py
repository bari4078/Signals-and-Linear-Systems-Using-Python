import numpy as np
import matplotlib.pyplot as plt

fx=100
t = np.linspace(0,2,100)

y_original = 2*np.sin(14*np.pi*t)-np.sin(2*np.pi*t)*(4*np.sin(2*np.pi*t)*np.sin(14*np.pi*t)-1)

y_sum = np.sin(2*np.pi*t) + np.sin(10*np.pi*t) + np.sin(18*np.pi*t)

def manual_dft(x):
    N = len(x)
    X = np.zeros(N,dtype = complex)
    for k in range(N):
        n = np.arange(N)
        X[k] = np.sum(x*np.exp(-2*1j*k*n/N))
    return X

X_k = manual_dft(y_original)
frequencies = np.arange(len(X_k)) * 