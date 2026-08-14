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

generator = SignalGenerator(t)

x = generator.square(freq = 1) + generator.triangle(freq=1)
a0, f0 = 10, 10
x_scaled = generator.time_scale(x,a0)
y = generator.modulate(x_scaled,f0)

f = np.linspace(-10,10,2000)

cft_x = CFTAnalyzer(t,x)
X = cft_x.transform(f)

cft_y = CFTAnalyzer(t,y)
Y = cft_y.transform(f)

cft_x_theory = CFTAnalyzer(t,x)
X_theory = cft_x_theory.transform((f-f0)/a0) / abs(a0)

mse_mag = CFTAnalyzer.mse( CFTAnalyzer.magnitude(Y), CFTAnalyzer.magnitude(X_theory))
mse_phase = CFTAnalyzer.mse( CFTAnalyzer.phase(Y), CFTAnalyzer.phase(X_theory))

print("MSE Magnitude:", mse_mag)
print("MSE Phase: ", mse_phase)

plt.figure()
plt.plot(f, np.abs(Y), label="|Y(f)|")
plt.plot(f, np.abs(X_theory), '--', label="(1/|a|)|X((f-f0)/a)|")
plt.legend(); plt.xlabel("f"); plt.title("Magnitude comparison")
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(f, CFTAnalyzer.phase(Y), label="∠Y(f)")
plt.plot(f, CFTAnalyzer.phase(X_theory), '--', label="∠X((f-f0)/a)")
plt.legend()
plt.xlabel("Frequency (f)")
plt.title("Phase Comparison")
plt.show()