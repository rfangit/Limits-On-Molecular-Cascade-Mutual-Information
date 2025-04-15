import numpy as np

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