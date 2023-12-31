import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import matplotlib.pyplot as plt
import scipy.special as special
import scipy.constants as const
import numpy as np
from arc import *
import random
from scipy import optimize
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.optimize import fmin
from scipy.optimize import brentq
from scipy.optimize import minimize
from scipy.optimize import curve_fit
import Analysis_Python_Files.Constants as cs
mass_Rb = 87*const.u        
        
def P_survival_lin_heating(t, alpha,U0,T0):
    tau = 5e3 # vacuum lifetime in ms
    P_vacuum = np.exp(-t/tau) # t in ms
    P_heat = 1- np.exp(- U0/(T0 + alpha*t) ) * ( 1+ U0/(T0 + alpha*t) + U0**2/(2*(T0 + alpha*t)**2) )
    return P_vacuum*P_heat

def linearHeatingFit(x_data,y_data,U0, T0,alpha_guess=135e-6,num_points=100,plot=True):
    params, covariance = curve_fit(lambda x_data, alpha: P_survival_lin_heating(x_data, alpha,U0,T0), x_data, y_data,p0=alpha_guess)
    optimal_alpha = params[0]
    alpha_err = covariance[0][0]*1e6
    x_fit = np.linspace(min(x_data), max(x_data), num_points)
    y_fit = P_survival_lin_heating(x_fit, *params,U0,T0)
    if plot:
        if optimal_alpha < 1e-6:
            plt.plot(x_fit, y_fit, label=f'$\\alpha$ ={optimal_alpha:.2e} $\\pm$ {alpha_err:.2e} $\\mathrm{{ Ks^{{-1}}}}$',linestyle= 'none')
        else:
            plt.plot(x_fit, y_fit, label=f'$\\alpha$ ={optimal_alpha:.2e} $\\pm$ {alpha_err:.2e} $\\mathrm{{ Ks^{{-1}}}}$')
    return optimal_alpha
    
def decay_exponential(x, A, k, C):
    return A * np.exp(-k * x) + C

def decay_fit(x_data,y_data,num_points=100,plot=True,color='tab:blue',manual_xmax = False,xmax_val = 10):
    initial_guess = [max(y_data), 0.1, min(y_data)]
    if manual_xmax:
        popt, pcov = curve_fit(decay_exponential, x_data, y_data, p0=initial_guess)
        x_fit = np.linspace(min(x_data), xmax_val, num_points)
        y_fit = decay_exponential(x_fit, *popt)
    popt, pcov = curve_fit(decay_exponential, x_data, y_data, p0=initial_guess)
    x_fit = np.linspace(min(x_data), max(x_data), num_points)
    y_fit = decay_exponential(x_fit, *popt)
    decay_constant = popt[1]
    decay_constant_uncertainty = np.sqrt(pcov[1, 1])
    one_over_e_time = 1 / decay_constant
    one_over_e_time_uncertainty = decay_constant_uncertainty / decay_constant
    decay_constant_str = f"{decay_constant:.3f}"
    decay_constant_uncertainty_str = f"{decay_constant_uncertainty:.3f}"
    legend_label = f'$\\tau^{{-1}}$ = {decay_constant_str} ± {decay_constant_uncertainty_str} ms'
    if plot:
        plt.plot(x_fit, y_fit, color=color,label=legend_label)
    return decay_constant, x_fit, y_fit, decay_constant_uncertainty

def beta_fit_guess(x_data,load_one_y_data,load_two_y_data,U0=1e-3,T0=30e-6,alpha_guess=1e-5,num_points=100,plot_alpha=True,plot_tau=True,color='tab:blue'):
    alpha = linearHeatingFit(x_data,load_one_y_data,U0, T0,alpha_guess,num_points,plot=plot_alpha)
    decay_constant,x_fit,y_fit, unc = decay_fit(x_data,load_two_y_data,num_points,plot=plot_tau,color=color)
    return alpha, decay_constant
                   

def trap_volume(T0, omega_ax, omega_rad, mass=mass_Rb):
    """
    T0 initial temperature in K
    omega_ax (rad) axial (radial) trap fequencies
    return in m^3
    """
    omega = (omega_ax * omega_rad**2)**(1/3)
    return (2*np.pi*const.k*T0/ (mass* omega**2))**(3/2)

def calc_beta(beta_prime,V):
    """
    beta_prime is the fitted two-body loss decay rate. 
    V trap volume
    return in SI units m^3/s
    """
    return beta_prime * 2* np.sqrt(2) * V

def dq_two_body(beta_prime, N, dt):
    return beta_prime * N*(N-1)*dt/2

def dq_one_body(gamma, N, dt):
    return gamma * N *dt

def P1(t,U, T0, alpha):
    """ Survival probability of an atom due to heating. Assuming a harmonic trap and a Boltzmann energy distribution
    U: trap depth in J
    T0 initial temperature
    alpha: heating rate"""
    return 1- np.exp(- U/(T0 + alpha*t) ) * ( 1+ U/(T0 + alpha*t) + U**2/(2*(T0 + alpha*t)**2) )

def gamma(t,U, T0, alpha):
    P1 = 1- np.exp(- U/(T0 + alpha*t) ) * ( 1+ U/(T0 + alpha*t) + U**2/(2*(T0 + alpha*t)**2) )
    P1dot_term1 = -np.exp(-U / (alpha * t + T0)) * (-alpha * U / (alpha * t + T0)**2 - alpha * U**2 / (alpha * t + T0)**3)
    P1dot_term2 = -(alpha * np.exp(-U / (alpha * t + T0)) * U * (1 + U / (alpha * t + T0) + U**2 / (2 * (alpha * t + T0)**2))) / (alpha * t + T0)**2
    P1_dot = P1dot_term1 + P1dot_term2
    return float(-P1_dot/P1  )

def number_of_lost_atoms(key_name):
    if key_name == 'no_loss':
            return 0 # no loss
    elif key_name== 'one_loss':
            return 1 # one loss
    elif key_name== 'two_loss':
            return 2 # two loss

def two_body_loss_sim(ts, beta_prime,alpha, U=1e-3,T0=30e-6, start = 0.85, asymp = 0.2):
    trajectories = 700
    dt_step = ts[1] - ts[0]
    surv_sum_traj = np.zeros(len(ts))

    for j in np.arange(trajectories):
        surv_dts = np.zeros(len(ts)) 
        surv_dts[0] = 2    # Initial number of atoms = 2
        for i in range(len(ts)):
            # print("\033[1m trajectory # \033[0m",j,"\033[1m time step # \033[0m",i)
            if i == 0:
                num_atoms_at_this_step = surv_dts[0]  # fixes inital atom number at N=2
            else:
                num_atoms_at_this_step = surv_dts[i - 1] # selects matrix elemenet
            P_two_body_loss = dq_two_body(beta_prime, num_atoms_at_this_step, dt_step)
            P_one_body_loss = dq_one_body(gamma(dt_step, U,T0,alpha), num_atoms_at_this_step, dt_step)
            prob_no_loss = (1 - P_one_body_loss) * (1 - P_two_body_loss)
            prob_one_loss = P_one_body_loss * (1 - P_two_body_loss)
            prob_two_loss = (1 - P_one_body_loss) * P_two_body_loss

            prob_dict = {'no_loss': prob_no_loss, 'one_loss': prob_one_loss, 'two_loss': prob_two_loss}
            
            # Define a probability order (smallest prob to largest prob)
            ordered_dict = sorted(prob_dict.items(), key=lambda x: x[1]) 
            
            # Choose which decay channel will happen using Monte Carlo
            p_rand = random.random() # random number between 0 and 1
            p_rand_20_21 = random.random()
            # print('p_rand=',p_rand)
            # print('probabilites=',ordered_dict)
            if p_rand < ordered_dict[0][1]: # if random number is less than the smallest loss probability of the three loss options
                if surv_dts[i] == 2 and p_rand_20_21 < asymp:
                    surv_dts[i] -= 1
                else: 
                    surv_dts[i] = num_atoms_at_this_step - number_of_lost_atoms(ordered_dict[0][0]) # then the # of remaining atoms at this time step is determined by key name of smallest survival probability
                # print('key, atoms lost=',ordered_dict[0][0],',',number_of_lost_atoms(ordered_dict[0][0]))
                # print('remaining atoms at this time=',surv_dts[i])
            elif ordered_dict[0][1] < p_rand < ordered_dict[1][1]: # if random number is between smallest loss and second smallest loss probability 
                if surv_dts[i] == 2 and p_rand_20_21 < asymp:
                    surv_dts[i] -= 1  
                else:
                    surv_dts[i] = num_atoms_at_this_step - number_of_lost_atoms(ordered_dict[1][0])
                # print('key, atoms lost=',ordered_dict[1][0],',',number_of_lost_atoms(ordered_dict[1][0]))
                # print('remaining atoms at this time=',surv_dts[i])
            else: # if random number is greater than both of the 2 smallest probabilities
                if surv_dts[i] == 2 and p_rand_20_21 < asymp:
                    surv_dts[i] -= 1                
                else:
                    surv_dts[i] = num_atoms_at_this_step - number_of_lost_atoms(ordered_dict[2][0]) # the # of atoms remaining = inital atom # - value for the key with the largest probability
                # print('key, atoms lost=',ordered_dict[2][0],',',number_of_lost_atoms(ordered_dict[2][0]))
                # print('remaining atoms at this time=',surv_dts[i])
                
        # apply a 2->1 collision (80% probability 2-1) and image (10% prob of 2-1)
        for k in range(len(surv_dts)):
            p_rand_LAC_21 = random.random()
            p_rand_image_21 = random.random()
            if surv_dts[k] == 2 and p_rand_LAC_21 <= start: #2-1 pulse
                surv_dts[k] -= 1   
            if surv_dts[k] == 2 and p_rand_image_21 <= 0.1: #imaging pulse
                surv_dts[k] -= 1            
            if surv_dts[k] == 2:         
                surv_dts[k] = 0
                
        surv_sum_traj += surv_dts
        # print('surv_sum_traj',surv_sum_traj)

    surv_average = surv_sum_traj / trajectories
    # print(surv_average)
    return surv_average  

########## Used for the fitting
def residual_alpha(alpha,x_pts, data, uncertainties):
    model = P_survival_lin_heating(x_pts, betaPrime,alpha,U=1e-3,T0=30e-6)
    return (((model-data)/uncertainties)**2).sum()

def residual_beta(betaPrime,x_pts, data, uncertainties,alpha):
    model = two_body_loss_sim(x_pts, betaPrime,alpha,U=1e-3,T0=30e-6)
    return (((model-data)/uncertainties)**2).sum()
                   
def get_smoothed_residual_f(x, residual):
    _sg = savgol_filter(residual, window_length=7, polyorder=3) # window size 51, polynomial order 3
    _f = interp1d(x, _sg, kind='cubic')
    return _f
                   
def get_minimum(_f, x0):
    _xmin,_fmin, _,_,warnflag = fmin(_f, 
        x0, disp=False,full_output=True)
    if warnflag:
        print(f"{warnflag:d} ---  1 : Maximum number of function evaluations made. 2 : Maximum number of iterations reached.")
    return _xmin, _fmin
                   
def get_root(_f, y0, xbound0, xbound1):
    _f_offset = lambda x : (_f(x) - y0)
    root = brentq(_f_offset, xbound0, xbound1, disp=False)
    return root

def beta_prime_fit(decay_constant,alpha,uncertainties,x_fit,y_fit,range_lim=2,range_step=0.1):
    out, fout, grid, fgrid = optimize.brute(residual_beta,(slice((decay_constant-range_lim), (decay_constant+range_lim), range_step),),
    args=(x_fit, y_fit, uncertainties,alpha),full_output=True,finish='leastsq')
    f_loss_smooth = get_smoothed_residual_f(grid, fgrid)
    grid_min, loss_min = get_minimum(f_loss_smooth, out)
    grid_min_err_low = get_root(f_loss_smooth, y0=loss_min*1.1, xbound0=grid.min(), xbound1=grid_min)
    grid_min_err_high = get_root(f_loss_smooth, y0=loss_min*1.1, xbound0=grid_min, xbound1=grid.max())
    low_err=grid_min[0]-grid_min_err_low # those turns out to be array with only one element
    high_err=grid_min_err_high-grid_min[0]
    popt_min=grid_min[0]
    loss_mins=loss_min
    return popt_min,high_err,low_err


def betaCalc(beta_prime,trap_depth,T,beta_prime_unc,trap_depth_unc,T_unc): 
    KelvinToJoules = 1.380648780669e-23
    lmbda = 850e-9
    w_0 = 0.7e-6 # beam waist (radius)
    U = trap_depth*KelvinToJoules # depth in Joules
    w_r = np.sqrt(4*U/(cs.Rb87_M*w_0**2))# Hz
    w_a = w_r / (np.sqrt(2)*w_0*np.pi/lmbda) # Hz
    # w_r = 135000*(2*np.pi)
    # w_a = 25000*(2*np.pi)
    w = (w_r**2 * w_a)**(1/3)
    V = ((2*np.pi*cs.k_B * T/(cs.Rb87_M*w**2))**(3/2))*1e6
    beta = beta_prime*1e3*2*np.sqrt(2)*V
    unc = np.sqrt(beta_prime_unc**2+trap_depth_unc**2+T_unc**2)
    total_unc = unc*1e3*2*np.sqrt(2)*V
    # print('beta',beta,w_r/(2*np.pi),w_a/(2*np.pi))
    return beta,total_unc

def trapVolume(trap_depth,T): 
    KelvinToJoules = 1.380648780669e-23
    lmbda = 850e-9
    w_0 = 0.7e-6 # beam waist (radius)
    U = trap_depth*KelvinToJoules # depth in Joules
    w_r = np.sqrt(4*U/(cs.Rb87_M*w_0**2))# Hz
    w_a = w_r / (np.sqrt(2)*w_0*np.pi/lmbda) # Hz
    # w_r = 135000*(2*np.pi)
    # w_a = 25000*(2*np.pi)
    w = (w_r**2 * w_a)**(1/3)
    V = ((2*np.pi*cs.k_B * T/(cs.Rb87_M*w**2))**(3/2))*1e6
    # print('beta',beta,w_r/(2*np.pi),w_a/(2*np.pi))
    return V

def count_potentials(energy_curves, interatomic_distance, energy_bin,plot=False):
    ## energy in GHz, distance in nm
    intersections_count = 0

    for energy_curve in energy_curves:
        x_values = energy_curve[:, 0]
        y_values = energy_curve[:, 1]
        if plot:
            plt.plot(x_values, y_values, label='Energy Curve')
            plt.ylim(energy_bin)
            plt.xlim(interatomic_distance - (interatomic_distance) / 3, interatomic_distance + (interatomic_distance) / 3)
            plt.axvline(x=interatomic_distance, color='black', linestyle='--', label='Interatomic Distance')

        # Check for intersections with the vertical line (interatomic distance) within the energy bin
        for i in range(len(x_values) - 1):
            x1, y1 = x_values[i], y_values[i]
            x2, y2 = x_values[i + 1], y_values[i + 1]

            # Check if the line segment is within the specified energy bin
            if min(y1, y2) <= energy_bin[1] and max(y1, y2) >= energy_bin[0]:
                # Check if the line segment intersects the vertical line (interatomic distance)
                if min(x1, x2) <= interatomic_distance <= max(x1, x2):
                    # Calculate the y-coordinate of the intersection
                    if x1 != x2:  # Avoid division by zero
                        y_intersection = y1 + (interatomic_distance - x1) * (y2 - y1) / (x2 - x1)

                        # Check if the intersection is within the y-range of the energy bin and the vertical line
                        if min(y1, y2) <= y_intersection <= max(y1, y2) and energy_bin[0] <= y_intersection <= energy_bin[1]:
                            intersections_count += 1

    return intersections_count

def get_hyperfine_potentials(label='D1_potentials_F22.csv',):
    hyperfine_potential = np.loadtxt(label, delimiter=',', skiprows=1) 
    atomic_distances = hyperfine_potential[:, 0]
    potentials = hyperfine_potential[:, 1:].T
    energy_curves = [np.column_stack((atomic_distances, potentials[i])) for i in range(potentials.shape[0])]
    return energy_curves

def calculate_average_inelastic_probability(inelastic_counts, reached_counts, num_simulations):
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        # Specify the output array and its data type (float)
        average_probabilities = np.divide(inelastic_counts, reached_counts, out=np.zeros_like(inelastic_counts, dtype=float), where=(reached_counts != 0))
    average_probabilities[np.isnan(average_probabilities)] = 0  # Set NaN values to 0
    return np.sum(average_probabilities) / num_simulations

def rabiRate(mu,I): 
    E = np.sqrt(2*I/(cs.c*cs.epsilon0))
    omega = (mu/cs.hbar)*E
    return omega/(2*np.pi)

def rabiRateGen(mu,I,delta): 
    E = np.sqrt(2*I/(cs.c*cs.epsilon0))
    omega = (mu/cs.hbar)*E
    rabi_gen = np.sqrt(omega**2+(delta*1e6)**2)
    return rabi_gen/(2*np.pi)

def P_inelastic_red(v,I,mu,delta):
    Omega = rabiRate(mu,I)
    PLZ = np.exp((-2 * np.pi * Omega**2) / (3 * v * delta*1e6) * ((cs.Rb87_C3 / (cs.hbar * delta*1e6))**(1 / 3))) 
    P_in = 1-PLZ/(2-PLZ)
    return P_in

def P_inelastic_blue(v,I,mu,delta):
    Omega = rabiRate(mu,I)
    PLZ = np.exp((-2 * np.pi * Omega**2) / (3 * v * delta*1e6) * ((cs.Rb87_C3 / (cs.hbar * delta*1e6))**(1 / 3))) 
    P_in = 2*PLZ*(1-PLZ)
    return P_in