import numpy as np
import time
import random
import csv
import matplotlib.pyplot as plt
from func_mdp_agentnregion import *

np.random.seed(1)
# Arguments
REWARD = -0.01 # constant reward for non-terminal states
DISCOUNT = 0.8
MAX_ERROR = 10**(-2)

# Set up the initial environment
N_region = 5
N_agent = 4

region = np.random.random((N_region+1,2))*10
region[-1] = [0,0]

# region_ = np.transpose(region)
# plt.plot(region_[0],region_[1],'o')
# plt.show()

Info = info(REWARD,DISCOUNT,MAX_ERROR,N_region,N_agent)
Val_mat = np.zeros(((N_region+1)**N_agent, 2**N_region))

tic = time.time()
Val_mat = valueIteration(Val_mat, Info)
toc = time.time()
print(toc - tic,'seconds')
policy = getOptimalPolicy(Val_mat, Info)
toc = time.time()
print(toc - tic,'seconds')

file_name = 't5a4.csv'
with open(file_name, 'w', newline='') as f:
    # using csv.writer method from CSV package
    write = csv.writer(f)
    write.writerows(policy)
f.close()
#
# simulate(policy, Info, np.zeros(N_agent), np.ones(N_region))