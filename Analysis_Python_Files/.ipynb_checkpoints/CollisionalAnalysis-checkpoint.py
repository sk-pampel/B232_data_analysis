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
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter
from scipy.optimize import fmin
from scipy.optimize import brentq
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline
import pandas as pd
import Analysis_Python_Files.Constants as cs
from statsmodels.stats.proportion import proportion_confint as confidenceInterval
        
    
def rabiRate(mu,I): 
    E = np.sqrt(2*I/(cs.c*cs.epsilon0))
    omega = (mu/cs.hbar)*E
    return omega

# def rabiRate(mu,I,delta): 
#     E = np.sqrt(2*I/(cs.c*cs.epsilon0))
#     omega = (mu/cs.hbar)*E
#     return np.sqrt(omega**2+delta**2)

def P_inelastic_red(v,I,mu,delta,C3_factor=1,Omega_factor=1):
    Omega = rabiRate(mu,I)*Omega_factor
    PLZ = np.exp((-2 * np.pi * cs.hbar * Omega**2) / (3 * v) * ((cs.Rb87_C3*C3_factor / (cs.h * delta*1e6)**4)**(1 / 3))) 
    P_in = 1-PLZ/(2-PLZ)
    return P_in

def P_inelastic_red_R(v,I,mu,R,C3_factor=1,Omega_factor=1):
    Omega = rabiRate(mu,I)*Omega_factor
    PLZ = np.exp((-2 * np.pi * Omega**2 * cs.hbar * R**4) / (3 * v * cs.Rb87_C3*C3_factor)) 
    P_in = 1-PLZ/(2-PLZ)
    return P_in

def P_inelastic_red_slope(alpha,v,I,mu,Omega_factor=1):
    Omega = rabiRate(mu,I)*Omega_factor
    PLZ = np.exp((-2 * np.pi * Omega**2 * cs.hbar ) / (alpha * v )) 
    P_in = 1-PLZ/(2-PLZ)
    return P_in 



def P_inelastic_blue(v,I,mu,delta,C3_factor=1,Omega_factor=1):
    Omega = rabiRate(mu,I)*Omega_factor
    PLZ = np.exp((-2 * np.pi * cs.hbar * Omega**2) / (3 * v) * ((cs.Rb87_C3 *C3_factor/ (cs.h*delta*1e6)**4)**(1 / 3))) 
    P_in = 2*PLZ*(1-PLZ)
    return P_in

def P_inelastic_blue_R(v,I,mu,R,C3_factor=1,Omega_factor=1):
    Omega = rabiRate(mu,I)*Omega_factor
    PLZ = np.exp((-2 * np.pi * Omega**2 * cs.hbar * R**4) / (3 * v * cs.Rb87_C3*C3_factor)) 
    P_in = 2*PLZ*(1-PLZ)
    return P_in 

def P_inelastic_blue_slope(alpha,v,I,mu,Omega_factor=1): # Omega_factor=1 gives Omega/2pi = 8 MHz
    Omega = rabiRate(mu,I)*Omega_factor
    PLZ = np.exp((-2 * np.pi * Omega**2 * cs.hbar ) / (alpha * v )) 
    P_in = 2*PLZ*(1-PLZ)
    return P_in 


def P_LZ_slope(alpha,v,I,mu,Omega_factor=1): # Omega_factor=1 gives Omega/2pi = 8 MHz
    Omega = rabiRate(mu,I)*Omega_factor
    PLZ = np.exp((-2 * np.pi * Omega**2 * cs.hbar ) / (alpha * v )) 
    return PLZ

def P_LZ_tolerance(alpha,v,I,mu,Omega_factor=1): # Omega_factor=1 gives Omega/2pi = 8 MHz
    Omega = rabiRate(mu,I)*Omega_factor
    delta = np.exp((-2 * np.pi * Omega**2 * cs.hbar ) / ((alpha+delta_alpha) * v )) 
    return PLZ

def P_survival_lin_heating(t, alpha,U0,T0):
    tau = 5e3 # vacuum lifetime in ms
    P_vacuum = np.exp(-t/tau) # t in ms
    P_heat = 1- np.exp(- U0/(T0 + alpha*t) ) * ( 1+ U0/(T0 + alpha*t) + U0**2/(2*(T0 + alpha*t)**2) )
    return P_vacuum*P_heat

def linearHeatingFit(x_data,y_data,U0, T0,alpha_guess=135e-6,num_points=100,color = 'tab:green',plot=True):
    params, covariance = curve_fit(lambda x_data, alpha: P_survival_lin_heating(x_data, alpha,U0,T0), x_data, y_data,p0=alpha_guess)
    optimal_alpha = params[0]
    alpha_err = covariance[0][0]*1e6
    x_fit = np.linspace(min(x_data), max(x_data), num_points)
    y_fit = P_survival_lin_heating(x_fit, *params,U0,T0)
    if plot:
        if optimal_alpha < 1e-6:
            plt.plot(x_fit, y_fit, label=f'$\\alpha$ ={optimal_alpha:.2e} $\\pm$ {alpha_err:.2e} $\\mathrm{{ Ks^{{-1}}}}$')
        else:
            plt.plot(x_fit, y_fit, label=f'$\\alpha$ ={optimal_alpha:.2e} $\\pm$ {alpha_err:.2e} $\\mathrm{{ Ks^{{-1}}}}$',color=color)
            
            # plt.plot(x_fit, y_fit, label=f'$\\alpha$ ={optimal_alpha:.2e} $\\pm$ {alpha_err:.2e} $\\mathrm{{ Ks^{{-1}}}}$')
    return optimal_alpha
    
def decay_exponential(x, A, B, C):
    return A * np.exp(-B * x) + C

def decay_fit(x_data,y_data,num_points=100,plot=True,color='tab:blue',manual_xmax = False,xmax_val = 10,decay_guess = 0.1):
    initial_guess = [max(y_data), decay_guess, min(y_data)]
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

def beta_fit_guess(x_data,load_one_y_data,load_two_y_data,U0=1e-3,T0=30e-6,alpha_guess=1e-5,decay_guess = 0.1, num_points=100,plot_alpha=True,plot_tau=True,color_tau='tab:blue',color_alpha = 'tab:green'):
    alpha = linearHeatingFit(x_data,load_one_y_data,U0, T0,alpha_guess,num_points,plot=plot_alpha,color=color_alpha)
    decay_constant,x_fit,y_fit, unc = decay_fit(x_data,load_two_y_data,num_points,plot=plot_tau,color=color_tau,decay_guess=decay_guess)
    return alpha, decay_constant
                   

def trap_volume(T0, omega_ax, omega_rad, mass=cs.Rb87_M):
    """
    T0 initial temperature in K
    omega_ax (rad) axial (radial) trap fequencies 
    return in m^3
    don't forget the factor of 2 pi!
    """
    omega = (omega_ax * omega_rad**2)**(1/3)
    return (2*np.pi*const.k*T0/ (mass* omega**2))**(3/2)

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




def betaCalc(beta_prime, beta_prime_unc, w, w_unc, T, T_unc): 
    V = ((2*np.pi*cs.k_B * T/(cs.Rb87_M*w**2))**(3/2))*1e6
    beta = beta_prime * 1e3 * 2 * np.sqrt(2) * V
    unc_beta_prime = beta_prime_unc/beta_prime * beta
    unc_w = (3/2)*w_unc/w * beta
    unc_T = T_unc/T * beta
    total_unc = np.sqrt(unc_beta_prime**2 + unc_w**2 + unc_T**2)
    return beta , total_unc

def betaCalcNoUnc(beta_prime, beta_prime_unc, w, w_unc, T, T_unc): 
    V = ((2*np.pi*cs.k_B * T/(cs.Rb87_M*w**2))**(3/2))*1e6
    beta = beta_prime * 1e3 * 2 * np.sqrt(2) * V
    unc_beta_prime = beta_prime_unc/beta_prime * beta
    unc_w = (3/2)*w_unc/w * beta
    unc_T = T_unc/T * beta
    total_unc = np.sqrt(unc_beta_prime**2 + unc_w**2 + unc_T**2)
    return beta 

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
    V = ((2*np.pi*cs.k_B * T/(cs.Rb87_M*w**2))**(3/2))
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

def get_hyperfine_potentials(FS,HFS): # form (D2, F23)
    filename = str(FS) + '_potentials_' + str(HFS) +'.csv'
    filepath = '/Users/stevenpampel/Documents/B232_Data_Analysis/Molecular_Potentials/' + filename
    hyperfine_potential = np.loadtxt(filepath, delimiter=',', skiprows=1) 
    # hyperfine_potential = np.loadtxt('LZ_potential.csv', delimiter=',', skiprows=1)
    atomic_distances = hyperfine_potential[:, 0]
    # energy_curves = hyperfine_potential[:, 1:] 
    potentials_T = hyperfine_potential[:, 1:].T
    potentials = hyperfine_potential[:, 1:]   
    energy_curves = [np.column_stack((atomic_distances, potentials_T[i])) for i in range(potentials_T.shape[0])]
    return atomic_distances, energy_curves, potentials

def get_hyperfine_potentials_plot(FS,HFS): # form (D2, F23)
    filename = str(FS) + '_potentials_' + str(HFS) +'.csv'
    filepath = '/Users/stevenpampel/Documents/B232_Data_Analysis/Molecular_Potentials/' + filename
    hyperfine_potential = np.loadtxt(filepath, delimiter=',', skiprows=1) 
    # hyperfine_potential = np.loadtxt('LZ_potential.csv', delimiter=',', skiprows=1)
    atomic_distances = hyperfine_potential[:, 0]
    # energy_curves = hyperfine_potential[:, 1:] 
    potentials = hyperfine_potential[:, 1:]
    return atomic_distances, potentials


def find_closest_value(value, values, tolerance):
    closest = min(values, key=lambda x: abs(x - value))
    if abs(closest - value) <= tolerance:
        return closest
    return None

def get_HFS_density(potentials,distance, detuning, tolerance_distance=5, tolerance_detuning=5):
    df = pd.read_csv(potentials, index_col='Detuning')
    df.columns = df.columns.astype(float)
    closest_distance = find_closest_value(distance, df.columns, tolerance_distance)
    closest_detuning = find_closest_value(detuning, df.index, tolerance_detuning)
    if closest_distance is not None and closest_detuning is not None:
        return df.loc[closest_detuning, closest_distance]
    return None

def find_Condon_radius(energy_curves, energy_level, distance_bin,plot=False):
    intersections = []

    for energy_curve in energy_curves:
        x_values = energy_curve[:, 0]
        y_values = energy_curve[:, 1]
        if plot:
            plt.plot(x_values, y_values, label='Energy Curve')
            plt.xlim(distance_bin)
        # Check for intersections with the horizontal line (energy level) within the distance bin
        for i in range(len(x_values) - 1):
            x1, y1 = x_values[i], y_values[i]
            x2, y2 = x_values[i + 1], y_values[i + 1]

            # Check if the line segment is within the specified distance bin
            if min(x1, x2) <= distance_bin[1] and max(x1, x2) >= distance_bin[0]:
                # Check if the line segment intersects the horizontal line (energy level)
                if min(y1, y2) <= energy_level <= max(y1, y2):
                    # Calculate the x-coordinate of the intersection
                    if y1 != y2:  # Avoid division by zero
                        x_intersection = x1 + (energy_level - y1) * (x2 - x1) / (y2 - y1)

                        # Check if the intersection is within the x-range of the line segment and within the distance bin
                        if min(x1, x2) <= x_intersection <= max(x1, x2) and distance_bin[0] <= x_intersection <= distance_bin[1]:
                            intersections.append(x_intersection)

    return intersections

def get_intersections(potentials, delta_min, delta_max, step,distance_bin = (10e-9,100e-9) ):
    potential = np.loadtxt(potentials, delimiter=',', skiprows=1) 
    x_values = potential[:, 0]
    curve_data_transposed = potential[:, 1:].T
    energy_curves_data = [np.column_stack((x_values, curve_data_transposed[i])) for i in range(curve_data_transposed.shape[0])]
    intersections_dict = {}
    for detuning in np.arange(delta_min,delta_max,step):  
        intersections = find_Condon_radius(energy_curves_data, detuning, distance_bin)
        intersections = sorted(intersections, reverse=True)
        intersections_dict[detuning] = intersections

    return intersections_dict

def get_intersections_list(potentials, detunings ,distance_bin = (10e-9,100e-9) ):
    potential = np.loadtxt(potentials, delimiter=',', skiprows=1) 
    x_values = potential[:, 0]
    curve_data_transposed = potential[:, 1:].T
    energy_curves_data = [np.column_stack((x_values, curve_data_transposed[i])) for i in range(curve_data_transposed.shape[0])]
    intersections_dict = {}
    for detuning in detunings:  
        intersections = find_Condon_radius(energy_curves_data, detuning, distance_bin)
        intersections = sorted(intersections, reverse=True)
        intersections_dict[detuning] = intersections

    return intersections_dict


def get_slope_parameter(potentials_file, target_distance, target_energy,plot=False): # expects potentials in GHz
    energy_matrix = np.loadtxt(potentials_file, delimiter=',', skiprows=1)
    distances = energy_matrix[:, 0]
    energy_curves = energy_matrix[:, 1:]
    column_labels = np.genfromtxt(potentials_file, delimiter=',', max_rows=1, dtype=str)

    # Create CubicSpline objects for each energy curve
    smoothed_curves = [CubicSpline(distances, curve) for curve in energy_curves.T]

    # Find the index of the closest distance
    closest_distance_index = np.argmin(np.abs(distances - target_distance))

    # Calculate the differences between the energy curves and the target energy at the closest distance
    energy_diffs = energy_curves[closest_distance_index] - target_energy

    # Find the index of the column with the overall minimum difference
    closest_energy_index = np.argmin(np.abs(energy_diffs))

    # Retrieve the nearest potential
    nearest_potential = smoothed_curves[closest_energy_index]
    nearest_potential_label = column_labels[closest_energy_index + 1]
    
    nearest_potential_array = np.array([[nearest_potential(distance)* cs.h * 1e9]  for distance in distances]) # convert to SI

    derivative_at_distance = nearest_potential.derivative()(target_distance)* cs.h * 1e9
    energy_at_large_distance = nearest_potential(distances[-1])

    # Plot all potentials with the nearest potential in black
    if plot:
        plt.subplots(figsize=(10,6))
        for i, potential in enumerate(smoothed_curves):
            if i == closest_energy_index:
                plt.plot(distances*1e9, potential(distances), color='black',linewidth=5, label=f'Nearest Potential (Derivative: {derivative_at_distance:.4f})')

            else:
                plt.plot(distances*1e9, potential(distances), color='grey',label=f'Potential {i + 1}')
        plt.plot(target_distance * 1e9, nearest_potential(target_distance), 'r*', markersize=15, label='Target Distance')

        plt.xlabel('Distance')
        plt.ylabel('Potential')
        plt.ylim(-1,1)
        plt.xlim(10,50)
        plt.show()
        
    return distances,nearest_potential_array, derivative_at_distance,nearest_potential_label, energy_at_large_distance 

def get_deltaR_C3(detuning, Omega, resonance):
    delta = 2 * np.pi * detuning * 1e9*cs.h
    res = 2 * np.pi * resonance * 1e9*cs.h    
    R_plus = (cs.Rb87_C3/(cs.hbar*(delta - res) + cs.hbar*Omega/2))**(1/3) 
    R_minus = (cs.Rb87_C3/(cs.hbar*(delta - res) + cs.hbar*Omega/2))**(1/3) 
    return abs(R_plus-R_minus)
               
def get_deltaR(x_values, curve_data, detuning, Omega, resonance, target_distance):
    delta = 2 * np.pi * detuning * 1e9
    res = 2 * np.pi * resonance * 1e9
               
    energy1 = (cs.hbar * (delta - res) + cs.hbar * Omega / 2) / (cs.h * 1e9) 
    energy2 = (cs.hbar * (delta - res) - cs.hbar * Omega / 2) / (cs.h * 1e9) 
    
    intersections1 = []
    intersections2 = []
    potential = curve_data.flatten() / (cs.h * 1e9) - resonance # convert from J to oGHz
    
    # Iterate through the curve data to find where the curve crosses the horizontal lines
    for i in range(len(x_values) - 1):
        x1, x2 = x_values[i], x_values[i + 1]
        y1, y2 = potential[i], potential[i + 1]

        # Check if the segment crosses the first horizontal line
        if (y1 - energy1) * (y2 - energy1) <= 0:
            if y1 != y2:  # Avoid division by zero
                x_intersect = x1 + (x2 - x1) * (energy1 - y1) / (y2 - y1)
                intersections1.append(x_intersect)
            else:
                intersections1.append(x1)  # Both points are on the line

        # Check if the segment crosses the second horizontal line
        if (y1 - energy2) * (y2 - energy2) <= 0:
            if y1 != y2:  # Avoid division by zero
                x_intersect = x1 + (x2 - x1) * (energy2 - y1) / (y2 - y1)
                intersections2.append(x_intersect)
            else:
                intersections2.append(x1)  # Both points are on the line

    # Find the intersection closest to the target distance
    def closest_intersection(intersections, target_distance):
        if intersections:
            closest = min(intersections, key=lambda x: abs(x - target_distance))
            return closest
        return None

    closest_intersection1 = closest_intersection(intersections1, target_distance)
    closest_intersection2 = closest_intersection(intersections2, target_distance)
    
    # Handle cases where no intersections are found 
    if closest_intersection1 is None or closest_intersection2 is None:
        return 1e-9

    # Calculate the absolute difference between the closest intersections
    abs_difference = abs(closest_intersection1 - closest_intersection2)
    
    return abs_difference

def diff_eq_collisionLZ(t, r, C3_factor):
    drdt = np.zeros(2)
    drdt[0] = r[1]
    drdt[1] = C3_factor * 3 * cs.Rb87_C3 / (cs.Rb87_M * r[0]**4)
    return drdt


def get_collisional_energyLZ(R, V, C3_factor):
    dt = 27e-9
    r0 = [R, -V]
    t_span = [0, dt]
    sol = solve_ivp(lambda t, r: diff_eq_collisionLZ(t, r, C3_factor), t_span, r0, t_eval=np.linspace(0, dt, 1000))
    velocities = sol.y[1]
    E_final_MHz = 3 / 8 * cs.Rb87_M * velocities[-1]**2 / cs.hbar * 1e-6
    return E_final_MHz 

def get_acceleration(distances, energy_curve, position):
    # Calculate acceleration based on the energy curve
    derivatives_interp = interp1d(distances, np.gradient(energy_curve, distances, axis=0),
                                  kind='linear', axis=0, fill_value='extrapolate')
    acc = -1 * derivatives_interp(position) / cs.Rb87_M
    return acc

def diff_eq_collision_num(t, r, distances, energy_curve):
    drdt = np.zeros(2)
    drdt[0] = r[1]
    drdt[1] = get_acceleration(distances, energy_curve, r[0])
    return drdt

def get_collisional_energy(distances, energy_curve, initial_position, initial_velocity,sym_label,detuning,mu,I_0, I_sat, Gamma_0,alpha=1):
    # dt_atomic = 27e-9 
    # dt_atomic_rand = abs(np.random.default_rng().exponential(dt_atomic))
    dt_eff = effective_lifetime(mu, I_0, I_sat, detuning*1e9, Gamma_0,alpha=alpha)
    dt_eff_rand = abs(np.random.default_rng().exponential(dt_eff))
    r0 = [initial_position, -initial_velocity]
    t_span = [0, dt_eff_rand]
    times = np.linspace(0, dt_eff_rand, 1000)
    sol = solve_ivp(diff_eq_collision_num, t_span, r0, t_eval=times, args=(distances, energy_curve))
    
    interp_position = interp1d(sol.t, sol.y[0], kind='cubic', fill_value='extrapolate')
    interp_velocity = interp1d(sol.t, sol.y[1], kind='cubic', fill_value='extrapolate')
    
    R_final = interp_position(times[-1])
    V_final = interp_velocity(times[-1])
    # U_initial = energy_curve[np.argmin(np.abs(initial_position - distances))]/cs.h*1e-6  # Get energy at initial position
    # U_final = energy_curve[np.argmin(np.abs(R_final - distances))]/cs.h*1e-6  # Get energy at final position
    if np.isnan(R_final) or np.isnan(V_final):
        return 0,dt_eff
    if np.all(energy_curve >= 0): # if repulsive potential
        if V_final < 0: # if atom doesn't reach turning 
            return 0,dt_eff # ignore  
        else:
            return V_final,dt_eff
    else: # attractive potential
        if abs(V_final) > 10: # avoid infinite energy collision
            V_final = -10 
            return V_final,dt_eff
        else:
            return V_final,dt_eff

def jeffreyInterval(m,num,alpha):
    # sigma = 1-0.6827 gives the standard "1 sigma" intervals.
    i1, i2 = confidenceInterval(round(m*num), num, method='jeffreys',alpha=alpha) # alpha=1-0.6827)
    return (m - i1, i2 - m)

def getCollisionStats(tferList,alpha):
    # Take the previous data, which includes entries when there was no atom in the first picture, and convert it to
    # an array of just loaded and survived or loaded and died.
    transferErrors = np.zeros([2])
    tferVarList = np.array([x for x in tferList if x != -1])
    if tferVarList.size == 0:
        # catch the case where there's no relevant data, typically if laser becomes unlocked.
        transferErrors = [0,0]
        transferAverages = 0
    else:
        # normal case
        transferAverages = np.average(tferVarList)
        transferErrors = jeffreyInterval(transferAverages, len(tferVarList),alpha)
    return transferAverages, transferErrors

# def effective_lifetime(mu, I_0, delta_nu, Gamma_0,alpha=1):
#     """
#     Calculate the effective lifetime of an excited state considering stimulated emission.
    
#     Parameters:
#     tau_0 (float): Natural lifetime of the excited state (in seconds)
#     mu (float): Transition dipole moment (in Coulomb-meters)
#     I_nu (float): Intensity of the incident radiation (in W/m^2/Hz)
#     delta_nu (float): Detuning from resonance (in Hz)
#     gamma (float): Natural linewidth (FWHM) of the transition (in Hz)
    
#     Returns:
#     float: Effective lifetime of the excited state (in seconds)
#     """
    
#     # Spontaneous emission rate
    
#     I = I_0*1/((2*delta_nu/Gamma_0)**2+1)
    
#     # Stimulated emission term
#     Gamma_stim = alpha*(mu**2 * I) / (6 * cs.epsilon0 * cs.hbar**2 * cs.c) * (1 / (1 + (2 * delta_nu / Gamma_0)**2))
    
#     # Total emission rate
#     Gamma_eff = Gamma_0 + Gamma_stim
    
#     # Effective lifetime
#     tau_eff = 1 / Gamma_eff
    
#     return tau_eff

def effective_lifetime(mu,I_0, I_sat, delta, Gamma_0,alpha=1):
    # Define the constant A
    A = (np.pi*mu**2) / (3*cs.c*cs.hbar**2*cs.epsilon0) 
    I_w = I_0/np.pi*((Gamma_0/2)/((delta)**2+(Gamma_0/2)**2))
    Gamma_stim = alpha*A*(I_w/(1+I_w/I_sat))     
    Gamma_eff = 2*Gamma_0 + Gamma_stim
    tau_eff = 1 / Gamma_eff
    
    return tau_eff