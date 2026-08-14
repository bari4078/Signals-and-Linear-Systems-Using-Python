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

    def base_signal(self):
        return 0.5*np.cos(5*self.t) + 0.5*np.sin(6*self.t)

    def first_derivative(self):
        return -2.5*np.sin(5*self.t) + 3*np.cos(6*self.t)

    def second_derivative(self):
        return -12.5*np.cos(5*self.t) - 18*np.sin(6*self.t)
    
    def third_derivative(self):
        return 62.5*np.sin(5*self.t) - 108*np.cos(6*self.t)

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
f = np.linspace(-2,2,2000)

j2pif = 1j * 2 * np.pi * f

generator = SignalGenerator(t)
x_t = generator.base_signal()
derivatives = [generator.first_derivative(), generator.second_derivative(),generator.third_derivative()]

X = CFTAnalyzer(t,x_t).transform(f)

for n in range(3):
    order = n+1

    Y_theoretical = (j2pif ** order) * X

    Y_calc = CFTAnalyzer(t, derivatives[n]).transform(f)

    threshold = 1e-5
    Y_theoretical[np.abs(Y_theoretical) < threshold] = 0
    Y_calc[np.abs(Y_calc) < threshold] = 0
    
    phase_theory = np.unwrap(CFTAnalyzer.phase(Y_theoretical))
    phase_practical = np.unwrap(CFTAnalyzer.phase(Y_calc))

    mse_mag = CFTAnalyzer.mse(CFTAnalyzer.magnitude(Y_theoretical), CFTAnalyzer.magnitude(Y_calc))
    mse_phase = CFTAnalyzer.mse(phase_theory, phase_practical)


    print("Mse Mag: ",mse_mag)
    print("MSE phase: ",mse_phase)

    # Plot Magnitude
    plt.figure(figsize=(10, 4))
    plt.plot(f, CFTAnalyzer.magnitude(Y_theoretical), label=f"Theoretical |(j2πf)^{order} X(f)|", linewidth=3)
    plt.plot(f, CFTAnalyzer.magnitude(Y_calc), '--', label=f"Practical |Y_{order}(f)|")
    plt.title(f"Derivative {order} - Magnitude Comparison")
    plt.legend()
    plt.xlabel("Frequency (f)")
    plt.show()

    # Plot Phase
    plt.figure(figsize=(10, 4))
    plt.plot(f, phase_theory, label=f"Theoretical Phase", linewidth=3)
    plt.plot(f, phase_practical, '--', label=f"Practical Phase")
    plt.title(f"Derivative {order} - Phase Comparison")
    plt.legend()
    plt.xlabel("Frequency (f)")
    plt.show()