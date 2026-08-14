import numpy as np
import matplotlib.pyplot as plt

class SignalGenerator:

    def __init__(self,t):
        self.t = t

    def square(self, freq =1, duty =0.5):
        return np.sign(np.sin(2*np.pi*freq*self.t))
    
    def triangle(self, freq =1):
        return (2/np.pi) * np.arcsin(np.sin(2*np.pi*freq*self.t))
    def gaussian(self, a=1.0):
        return np.exp(-a*self.t**2)

    def time_shift(self,x,t0):
        return np.interp(self.t-t0,self.t,x,left=0,right=0)

    def time_scale(self, x,a):
        return np.interp(a*self.t,self.t,x,left=0,right=0)

    def modulate(self,x,f0):
        return x* np.exp(1j*2*np.pi*f0*self.t)

class CFTAnalyzer:
    def __init__(self,t,x):
        self.t=t
        self.x = x

    def transform(self, f_axis):
        X = np.zeros(len(f_axis),dtype=complex)

        for i,f in enumerate(f_axis):
            e_power_thingy = np.exp(-1j*2*np.pi*f*self.t)
            real = np.trapezoid(np.real(self.x*e_power_thingy),self.t)
            imag = np.trapezoid(np.imag(self.x*e_power_thingy),self.t)
            X[i] = real + 1j * imag
        return X

    @staticmethod
    def magnitude(X):
        return np.abs(X)

    @staticmethod
    def phase(X):
        return np.angle(X)
    
    @staticmethod
    def mse(a,b):
        return np.mean((a-b)**2)