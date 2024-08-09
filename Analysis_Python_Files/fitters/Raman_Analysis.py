import numpy as np
from scipy.optimize import curve_fit
from .. import Constants as cs

def double_gaussian(x, a1, b1, c1, a2, b2, c2,d):
    return a1 * np.exp(-(x - b1)**2 / (2 * c1**2)) + a2 * np.exp(-(x - b2)**2 / (2 * c2**2))+d

def T_nbar(trap_freq,nbar):
    w = 2*np.pi*trap_freq
    T = cs.hbar*w/cs.k_B *1/(np.log(1/nbar+1))
    return T

def get_nbar(x_data,y_data, guess = (0.8, -130, 20, 0.5, 130, 20, 0.1)):
    initial_guess = list(guess)
    popt, pcov = curve_fit(double_gaussian, x_data, y_data, p0=initial_guess)
    x_fit = np.linspace(min(x_data), max(x_data), 1000)
    y_fit = double_gaussian(x_fit, *popt)
    R = popt[3]/popt[0]
    max_y_index = np.argmax(y_fit)
    max_x_value = x_fit[max_y_index]
    x_min = 0 ## left of RSB
    x_max = 200 ## right of RSB
    indices = np.where((x_fit >= x_min) & (x_fit <= x_max))
    x_range = x_fit[indices]
    y_range = y_fit[indices]
    max_y_index = np.argmax(y_range)
    max_x_value2 = x_range[max_y_index]
    print(max_x_value,max_x_value2)
    nbar = R/(1-R)            
    trap_freq = (max_x_value2-max_x_value)/2*1e3
    T = T_nbar(trap_freq,nbar)
    return x_fit,y_fit, nbar, trap_freq, T
             

    
    

