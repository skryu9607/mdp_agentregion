import numpy as np
import time
from func_mdp_agentnregion import *

# Arguments
REWARD = -0.01 # constant reward for non-terminal states
DISCOUNT = 0.8
MAX_ERROR = 10**(-2)

# Set up the initial environment
N_region = 5
N_agent = 2

Info = info(REWARD,DISCOUNT,MAX_ERROR,N_region,N_agent)
Val_mat = np.zeros(((N_region+1)**N_agent, 2**N_region))

tic = time.time()
Val_mat = valueIteration(Val_mat, Info)
toc = time.time()
print(toc - tic,'seconds')
policy = getOptimalPolicy(Val_mat, Info)
toc = time.time()
print(toc - tic,'seconds')

simulate(policy, Info, np.zeros(N_agent), np.ones(N_region))

