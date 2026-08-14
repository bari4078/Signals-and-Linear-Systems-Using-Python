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

t = np.linspace(-5,5,4000)
a = 1
generator = SignalGenerator(t)
x = generator.gaussian(a)

t0 = 1
y = generator.time_shift(x,t0)

f_axis = np.linspace(-10,10,2000)
cft_x = CFTAnalyzer(t,x)
X = cft_x.transform(f_axis=f_axis)
cft_y = CFTAnalyzer(t,y)
Y = cft_y.transform(f_axis)


threshold = 1e-5
X[np.abs(X) < threshold] = 0
Y[np.abs(Y) < threshold] = 0

phase_Y = np.unwrap(CFTAnalyzer.phase(Y))
phase_X = np.unwrap(CFTAnalyzer.phase(X))
mag_mse = CFTAnalyzer.mse(CFTAnalyzer.magnitude(Y),CFTAnalyzer.magnitude(X))
phase_mse = CFTAnalyzer.mse(phase_Y ,phase_X - 2*np.pi*f_axis*t0)

print("Magnitude mse: ",mag_mse)
print("Phase mse: ",phase_mse)

plt.figure()
plt.plot(f_axis,CFTAnalyzer.magnitude(X), label="|X(f)|")
plt.plot(f_axis,CFTAnalyzer.magnitude(Y), "--",label="|Y(f)|")
plt.legend()
plt.title("Magnitude is unchanged by time shift")
plt.show()

plt.figure()
plt.plot(f_axis,CFTAnalyzer.phase(X), label="<X(f)")
plt.plot(f_axis,CFTAnalyzer.phase(Y), "--",label="<Y(f)")
plt.legend()
plt.title("Phase is ∠Y(f) = ∠X(f)-2πft0 after time shift")
plt.show()