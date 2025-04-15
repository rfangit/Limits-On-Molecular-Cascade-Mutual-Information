import numpy as np
from scipy.integrate import solve_ivp

# Compute time derivatives for a two-state Markov system without detection events
def two_state_system_dt(t, p, k_off, k_on):
    dp0dt = -k_on * p[0] + k_off * p[1]
    dp1dt = k_on * p[0] - k_off * p[1]
    return np.array([dp0dt, dp1dt])

# Compute time derivatives for a two-state Markov system with detection events
def two_state_system_detection_dt(t, p, k_off, k_on, alpha):
    dp0dt = -k_on * p[0] + k_off * p[1]
    dp1dt = k_on * p[0] - k_off * p[1] - alpha*p[1]
    return np.array([dp0dt, dp1dt])

# Compute the analytical solution for a two-state Markov system with detection events
# Returns: Tuple of arrays (P_on, P_off) representing probabilities at each time point
# Specifically, P_on[time_index] refers to the probability that given an initial detection event
# at t = 0, the system has yet to produce a new detection event, and is in the on-state currently.
def two_state_analytical_solution(k_on, k_off, alpha, t_eval):
    # Calculate eigenvalues
    a = alpha + k_off + k_on
    D = np.sqrt(a**2 - 4*alpha*k_on)
    lambda1 = (-a + D)/2
    lambda2 = (-a - D)/2
    
    denominator = lambda2 - lambda1
    
    # Calculate probabilities
    P_on = ((alpha + k_off + lambda2) * np.exp(lambda1 * t_eval) - 
            (alpha + k_off + lambda1) * np.exp(lambda2 * t_eval)) / denominator
    P_off = -k_off * (np.exp(lambda1 * t_eval) - np.exp(lambda2 * t_eval)) / denominator
    return P_on, P_off

# Compute the analytical distribution of detection event times
# Returns an array, denoting the probability that the next detection event occurs
# at a time t_eval[time_index] after the last detection event
def event_time_dist(k_on, k_off, alpha, t_eval):
    # Calculate eigenvalues
    a = alpha + k_off + k_on
    D = np.sqrt(a**2 - 4*alpha*k_on)
    lambda1 = (-a + D)/2
    lambda2 = (-a - D)/2
    denominator = lambda2 - lambda1
    
    numerator = (alpha + lambda1)*lambda2*np.exp(lambda2*t) - (alpha + lambda2)*lambda1*np.exp(lambda1*t)
    return numerator / denominator

# Compute steady-state distribution of system states.
def x_dist(k_on, k_off):
    P_on = k_on/(k_on + k_off)
    return np.array([P_on, 1 - P_on])

# Generates normalized probability distributions for the two-state system
# The joint probability distribution has indices [current_x, time_index]
# where time index refers to the time since the last detection event occured
# and current_x the x value at the current time.
# The distribution of survival times marginalizes over the current_x.
def two_state_markov_pdfgen(k_on, k_off, alpha, t_eval):
    dt = t_eval[1] - t_eval[0]
    # Obtain the probability arrays
    P_on, P_off = two_state_analytical_solution(k_on, k_off, alpha, t_eval)    
    joint_prob = np.vstack((P_off, P_on))
    survival_time_prob = P_on + P_off
    
    norm = np.sum(joint_prob)*dt
    norm2 = np.sum(survival_time_prob)*dt
    joint_prob = joint_prob/norm
    survival_time_prob = survival_time_prob/norm
    
    x_steady_probs = x_dist(k_on, k_off)
    return joint_prob, survival_time_prob, x_steady_probs

# Calculate entropies from probability distributions
# Uses numerical thresholds to avoid issues with log(0)
def entropy_calcs(joint_prob_calc, survival_time_calc, x_steady_probs, dt, eps=1e-10):
    # Entropy calculations (ignoring near-zero probabilities)
    mask = joint_prob_calc > eps
    joint_entropy = -np.sum(joint_prob_calc[mask] * np.log2(joint_prob_calc[mask])) * dt
    mask = survival_time_calc > eps
    survival_time_entropy = -np.sum(survival_time_calc[mask] * np.log2(survival_time_calc[mask])) * dt
    
    x_entropy = -np.sum(x_steady_probs * np.log2(x_steady_probs))
    return joint_entropy, survival_time_entropy, x_entropy
    
# Calculates entropies and mutual information between system state and survival times
# k_on, k_off, alpha, are system parameters, t_eval, dt are temporal resolution parameters
# Returns tuple with mutual information, joint entropy, survival time entropy, and x entropy
def mutual_information_calc(k_on, k_off, alpha, t_eval, dt):
    joint_prob, survival_time_prob, x_steady_probs = two_state_markov_pdfgen(k_on, k_off, alpha, t_eval)
    joint_entropy, survival_time_entropy, x_entropy = entropy_calcs(joint_prob, survival_time_prob, x_steady_probs, dt)
    mutual_information = survival_time_entropy + x_entropy - joint_entropy
    return mutual_information, joint_entropy, survival_time_entropy, x_entropy

# Converts effective parameters N, A to rate constants
# Assumes k_off = 1.
# Since mutual information is scale invariant to rates, this is allowed.
def rates_from_params(N, A):
    k_off = 1  # arbitrary scale
    alpha = A
    k_on = N/(1 - N)
    return k_on, k_off, alpha

# Computes the steady state solution of a diffeq
# Inputs: steady state diffeq, initial state, t_max, dt, rate params
# Output: steady state solution at the end of t_max
def steady_state_sim(steady_state_diffeq, initial_p, t_max, dt, k_on, k_off):
    t_eval = np.arange(0, t_max, dt)
    solution = solve_ivp(two_state_system_dt, [0, t_max], initial_p, 
                    t_eval=t_eval, method='RK45', args=(k_off, k_on))
    steady_state = solution.y[:, -1]
    return steady_state

# Computes the initial distribution (a steady state distribution conditioned on an event occuring)
# This involves reweighting the steady state distribution by the probability each state produces
# A detectable event. 
def steady_state_when_event_occurs(steady_state):
    state_indices = np.arange(steady_state.shape[0])  # [0, 1, 2, ...]
    weighted_init_norm_factors = np.sum(steady_state * state_indices, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        normalized_init_state = (steady_state * state_indices) / weighted_init_norm_factors
        normalized_init_state = np.nan_to_num(normalized_init_state)  # Convert NaN/inf to 0
    return normalized_init_state

# Computes the probability that given an initial state
# no detectable event has happened since, for all initial states possible
# Returns a matrix P_survival[initial_condition, time_index, current_state]
def survival_probabilities_given_initial_state(survival_diffeq, num_states, t_max, dt,
                                              k_off, k_on, alpha):
    t_eval = np.arange(0, t_max, dt)
    num_time_points = len(t_eval)
    solution = np.zeros((num_states, num_time_points, num_states))
    for initial_state in range(num_states):
        initial_p = np.zeros(num_states)
        initial_p[initial_state] = 1.0
        sol = solve_ivp(two_state_system_detection_dt, [0, t_max], initial_p, 
                   t_eval=t_eval, method='RK45', args=(k_off, k_on, alpha))
        # Store solution in array (solution[initial_state, time_index, state_value])
        solution[initial_state, :, :] = sol.y.T
    return solution

# Computes the probability of observing a given time since the last event
# which is a normalized version of the survival probabilities.
def compute_prob_time_since_last_event(solution, dt):
    prob_time_since_last_event = np.zeros_like(solution)
    for initial_state in range(solution.shape[0]):
        # Sum over states for each time point (keep dimensions for broadcasting)
        norm_factors = np.sum(solution[initial_state]*dt)
        prob_time_since_last_event[initial_state] = solution[initial_state] / norm_factors
    return prob_time_since_last_event

# Computes the probability that a detection event happens at
# a specific time index
def death_event_probs(solution, dt):
    # Create marginalized array by summing over x values (axis=2)
    marginalized = np.sum(solution, axis=2)  # Shape: [initial_conditions, time_points]
    # Compute death probability density (negative time derivative of S)
    death_prob = -np.diff(marginalized, axis=1) / dt  # Finite difference derivative
    # Pad with zero to match original array size
    death_prob = np.pad(death_prob, ((0,0),(0,1)), mode='constant') #Shape: [initial_conditions, time_points]
    return death_prob

def prob_x_given_detection_t_and_init(solution):
    # Create state indices array for weighting
    state_indices = np.arange(solution.shape[2])  # [0, 1, 2, ...]
    # Compute weighted sum (expectation value), to normalize by the probability of the current x given t and init
    weighted_norm_factors = np.sum(solution * state_indices, axis=2, keepdims=True)
    # Safe division (skip where denominator is zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        prob_x_given_t_and_init = (solution * state_indices) / weighted_norm_factors
        prob_x_given_t_and_init = np.nan_to_num(prob_x_given_t_and_init)  # Convert NaN/inf to 0
    return prob_x_given_t_and_init

# Obtain the numerical PDF
def numerical_prob_x_t_xi(init_cond, time_index, current_x, steady_state_with_detection, time_since_last_event):
    return steady_state_with_detection[init_cond] * time_since_last_event[init_cond, time_index, current_x]
    
# Imagine we want to the probability distribution of (init cond, time index, time index2, current state)
# Then we need steady state with detection[init cond] * death_probs[init cond, time index2] * 
# sum over intermediate x of prob_x_given_t_and_init[init cond, time index2, intermediate x]*
# time_since_last_event[intermediate x, time index, current state]
def numerical_prob_x_t_t2_xi(init_cond, time_index, time_index2, current_x, steady_state_with_detection,
                             time_since_last_event, death_probs, prob_x_given_t_and_init, num_x_states):
    prob = steady_state_with_detection[init_cond]*death_probs[init_cond, time_index2]
    intermediate_sum = 0
    for intermediate_value in range(1, num_x_states):
        intermediate_sum += prob_x_given_t_and_init[init_cond, time_index2, intermediate_value]*time_since_last_event[intermediate_value, time_index, current_x]
    return prob*intermediate_sum

# Compute the numerical mutual information given the time resolution, initial steady state probabilities
# and the time since last event
def numerical_mutual_info_onestep(t_eval, dt, steady_state_with_detection, time_since_last_event):
    joint_entropy = 0
    time_entropy = 0
    num_of_states = len(steady_state_with_detection)

    for time_index in range(len(t_eval)):
        prob_t = 0
        for current_x in range(num_of_states):
            prob_x_t = 0
            for init_cond in range(1, num_of_states):
                prob_x_t_xi = numerical_prob_x_t_xi(init_cond, time_index, current_x, steady_state_with_detection, time_since_last_event)
                prob_x_t += prob_x_t_xi #marginalizing out the initial
            if prob_x_t > 0:
                joint_entropy += -prob_x_t * np.log2(prob_x_t)*dt
            prob_t += prob_x_t
        if prob_t > 0:
            time_entropy += -prob_t * np.log2(prob_t)*dt
    return joint_entropy, time_entropy

# Compute the numerical mutual information given the time resolution, initial steady state probabilities
# and the time since last event, where the mutual information has knowledge of the exact initial state
def numerical_mutual_info_onestep_with_init(t_eval, dt, steady_state_with_detection, time_since_last_event):
    joint_entropy_with_init = 0
    time_entropy_with_init = 0
    num_of_states = len(steady_state_with_detection)

    for time_index in range(len(t_eval)):
        for init_cond in range(1, num_of_states):
            prob_t_xi = 0
            for current_x in range(num_of_states):
                prob_x_t_xi = numerical_prob_x_t_xi(init_cond, time_index, current_x, steady_state_with_detection, time_since_last_event)
                if prob_x_t_xi > 0:
                    joint_entropy_with_init += -prob_x_t_xi * np.log2(prob_x_t_xi)*dt
                prob_t_xi += prob_x_t_xi
            if prob_t_xi > 0:
                time_entropy_with_init += -prob_t_xi * np.log2(prob_t_xi)*dt
    return joint_entropy_with_init, time_entropy_with_init

# Compute the numerical mutual information for two reaction events given the time resolution, initial steady state probabilities
# and the time since last event
def numerical_mutual_info_twostep(t_eval, dt, steady_state_with_detection, prob_time_since_last_event, death_probs, prob_x_given_t_and_init):
    joint_entropy = 0
    time_entropy = 0
    num_of_states = len(steady_state_with_detection)

    for time_index in range(len(t_eval)):
        for time_index2 in range(len(t_eval)):
            prob_t1t2 = 0
            for current_x in range(num_of_states):
                prob_x_t1t2 = 0
                for init_cond in range(1, num_of_states):
                    prob_x_t1t2_xi = numerical_prob_x_t_t2_xi(init_cond, time_index, time_index2, current_x, steady_state_with_detection,
                                 prob_time_since_last_event, death_probs, prob_x_given_t_and_init, num_of_states)
                    prob_x_t1t2 += prob_x_t1t2_xi #marginalizing out the initial condition
                if prob_x_t1t2 > 0:
                    joint_entropy += -prob_x_t1t2 * np.log2(prob_x_t1t2)*dt*dt
                prob_t1t2 += prob_x_t1t2
            if prob_t1t2 > 0:
                time_entropy += -prob_t1t2 * np.log2(prob_t1t2)*dt*dt
        #if time_index %500 == 0:
        #    print ("Time index ", time_index)
    return joint_entropy, time_entropy

# Compute the numerical mutual information for two reaction events given the time resolution, initial steady state probabilities
# and the time since last event, where the mutual information has knowledge of the exact initial state
def numerical_mutual_info_twostep_with_init(t_eval, dt, steady_state_with_detection, prob_time_since_last_event, death_probs, prob_x_given_t_and_init):
    joint_entropy_with_init = 0
    time_entropy_with_init = 0
    num_of_states = len(steady_state_with_detection)

    for time_index in range(len(t_eval)):
        for time_index2 in range(len(t_eval)):
            for init_cond in range(1, num_of_states):
                prob_t1t2_xi = 0
                for current_x in range(num_of_states):
                    prob_x_t1t2_xi = numerical_prob_x_t_t2_xi(init_cond, time_index, time_index2, current_x, steady_state_with_detection,
                                 prob_time_since_last_event, death_probs, prob_x_given_t_and_init, num_of_states)
                    if prob_x_t1t2_xi > 0:
                        joint_entropy_with_init += -prob_x_t1t2_xi * np.log2(prob_x_t1t2_xi)*dt*dt
                    # Marginalize out x
                    prob_t1t2_xi += prob_x_t1t2_xi
                if prob_t1t2_xi > 0:
                    time_entropy_with_init += -prob_t1t2_xi * np.log2(prob_t1t2_xi)*dt*dt
        #if time_index %500 == 0:
        #    print ("Time index ", time_index)
    return joint_entropy_with_init, time_entropy_with_init