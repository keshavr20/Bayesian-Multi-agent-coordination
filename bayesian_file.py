import numpy as np

# Assuming the following variables are defined and initialized elsewhere
# ta_t = task allocation set at time t whose preconditions have been satisfied
# V_T, Q_T = value and quality functions , which will be found using BRTDP
# A = set of all possible actions
# st_1 = previous state
# at_1 = previous action
# beta = temperature parameter(our case its 1.3)
# P = belief distribution over task allocations
# ta_star = maximum a posteriori task allocation
#s_t = current state
#a_t = current action

# Initialize beliefs on the first time step or when task allocation set is updated.
P = {ta: 1 / V_T[s_t] for ta in ta_t}

# Update beliefs based on action likelihoods.
for ta in ta_t:
    # Calculate likelihood
    likelihood = np.prod([np.exp(1.3 * Q_T[st_1, at_1]) / np.sum(np.exp(beta * Q_T[st_1, a])) for a in A])
    
    # Update posterior
    P[ta] = P[ta] * likelihood

# Normalizing P(ta)
total_P = sum(P.values())
P = {ta: P[ta] / total_P for ta in P}

# Pick the maximum a posteriori task allocation (ta*).
ta_star = max(P, key=P.get)
