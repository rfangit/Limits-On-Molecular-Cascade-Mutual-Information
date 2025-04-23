import numpy as np

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

## Fast Vectorized Computation Methods
def numerical_prob_x_t_xi_vectorized(steady_state_with_detection, time_since_last_event):
    """
    Vectorized version that computes all prob_x_t_xi at once
    Returns a 3D array of shape (num_init_conds, num_time_indices, num_current_x)
    """
    # Expand dimensions for broadcasting
    steady_state_exp = steady_state_with_detection[:, np.newaxis, np.newaxis]
    return steady_state_exp * time_since_last_event

def numerical_mutual_info_onestep_vectorized(t_eval, dt, steady_state_with_detection, time_since_last_event):
    """
    Vectorized version of numerical_mutual_info_onestep
    """
    num_of_states = len(steady_state_with_detection)
    
    # Compute all probabilities at once
    prob_x_t_xi = numerical_prob_x_t_xi_vectorized(steady_state_with_detection, time_since_last_event)
    
    # Marginalize out initial conditions (sum over axis 0)
    prob_x_t = np.sum(prob_x_t_xi[1:, :, :], axis=0)  # Skip init_cond=0 as in original
    
    # Compute joint entropy
    mask = prob_x_t > 0
    joint_entropy = -np.sum(prob_x_t[mask] * np.log2(prob_x_t[mask])) * dt
    
    # Compute time entropy
    prob_t = np.sum(prob_x_t, axis=1)  # Sum over current_x
    mask_t = prob_t > 0
    time_entropy = -np.sum(prob_t[mask_t] * np.log2(prob_t[mask_t])) * dt
    
    return joint_entropy, time_entropy

def numerical_mutual_info_onestep_vectorized_with_init(t_eval, dt, steady_state_with_detection, time_since_last_event):
    """
    Vectorized version of numerical_mutual_info_onestep_with_init
    """
    num_of_states = len(steady_state_with_detection)
    
    # Compute all probabilities at once
    prob_x_t_xi = numerical_prob_x_t_xi_vectorized(steady_state_with_detection, time_since_last_event)
    
    # Compute joint entropy with init
    mask = prob_x_t_xi > 0
    joint_entropy_with_init = -np.sum(prob_x_t_xi[mask] * np.log2(prob_x_t_xi[mask])) * dt
    
    # Compute time entropy with init
    prob_t_xi = np.sum(prob_x_t_xi, axis=2)  # Sum over current_x
    mask_t = prob_t_xi > 0
    time_entropy_with_init = -np.sum(prob_t_xi[mask_t] * np.log2(prob_t_xi[mask_t])) * dt
    
    return joint_entropy_with_init, time_entropy_with_init

## Multi-event Vectorization
def fast_numerical_prob_x_t_t2_xi_vectorized(steady_state_with_detection, death_probs, 
                                            prob_x_given_t_and_init, time_since_last_event):
    """
    Fast vectorized computation of probabilities with proper broadcasting
    """
    # Compute the first part: steady_state * death_probs
    part1 = steady_state_with_detection[:, np.newaxis] * death_probs  # shape: (init_cond, time_index2)
    
    # Compute the intermediate sum using einsum
    intermediate_sum = np.einsum('itx,xjc->itjc', 
                               prob_x_given_t_and_init, 
                               time_since_last_event,
                               optimize=True)
    
    # Multiply with part1, ensuring proper broadcasting
    part1_reshaped = part1[:, :, np.newaxis, np.newaxis]  # shape: (init_cond, time_index2, 1, 1)
    return part1_reshaped * intermediate_sum  # shape: (init_cond, time_index2, time_index, current_x)

def fast_numerical_mutual_info_twostep(t_eval, dt, steady_state_with_detection, 
                                      time_since_last_event, death_probs, 
                                      prob_x_given_t_and_init, chunk_size=1000):
    """
    Fast vectorized version of the two-step mutual information calculation
    """
    num_time_points = len(t_eval)
    joint_entropy = 0
    time_entropy = 0
    
    for chunk_start in range(0, num_time_points, chunk_size):
        chunk_end = min(chunk_start + chunk_size, num_time_points)
        #print(f"Processing time_index2 {chunk_start} to {chunk_end-1}")
        
        # Get chunk of data
        death_probs_chunk = death_probs[:, chunk_start:chunk_end]
        prob_x_given_chunk = prob_x_given_t_and_init[:, chunk_start:chunk_end, :]
        
        # Compute probabilities for this chunk
        prob_x_t1t2_xi = fast_numerical_prob_x_t_t2_xi_vectorized(
            steady_state_with_detection,
            death_probs_chunk,
            prob_x_given_chunk,
            time_since_last_event
        )
        
        # Marginalize out initial condition (skip init_cond=0)
        prob_x_t1t2 = np.sum(prob_x_t1t2_xi[1:, :, :, :], axis=0)
        
        # Compute entropies
        mask = prob_x_t1t2 > 0
        joint_entropy += -np.sum(prob_x_t1t2[mask] * np.log2(prob_x_t1t2[mask])) * dt**2
        
        prob_t1t2 = np.sum(prob_x_t1t2, axis=2)  # sum over current_x
        mask_t = prob_t1t2 > 0
        time_entropy += -np.sum(prob_t1t2[mask_t] * np.log2(prob_t1t2[mask_t])) * dt**2
    
    return joint_entropy, time_entropy

def fast_numerical_mutual_info_twostep_with_init(t_eval, dt, steady_state_with_detection, 
                                               time_since_last_event, death_probs, 
                                               prob_x_given_t_and_init, chunk_size=1000):
    """
    Fast vectorized version with initial condition knowledge
    """
    num_time_points = len(t_eval)
    joint_entropy_with_init = 0
    time_entropy_with_init = 0
    
    for chunk_start in range(0, num_time_points, chunk_size):
        chunk_end = min(chunk_start + chunk_size, num_time_points)
        #print(f"Processing time_index2 {chunk_start} to {chunk_end-1}")
        
        # Get chunk of data
        death_probs_chunk = death_probs[:, chunk_start:chunk_end]
        prob_x_given_chunk = prob_x_given_t_and_init[:, chunk_start:chunk_end, :]
        
        # Compute probabilities for this chunk
        prob_x_t1t2_xi = fast_numerical_prob_x_t_t2_xi_vectorized(
            steady_state_with_detection,
            death_probs_chunk,
            prob_x_given_chunk,
            time_since_last_event
        )
        
        # Compute entropies
        mask = prob_x_t1t2_xi > 0
        joint_entropy_with_init += -np.sum(prob_x_t1t2_xi[mask] * np.log2(prob_x_t1t2_xi[mask])) * dt**2
        
        prob_t1t2_xi = np.sum(prob_x_t1t2_xi, axis=3)  # sum over current_x
        mask_t = prob_t1t2_xi > 0
        time_entropy_with_init += -np.sum(prob_t1t2_xi[mask_t] * np.log2(prob_t1t2_xi[mask_t])) * dt**2
    
    return joint_entropy_with_init, time_entropy_with_init