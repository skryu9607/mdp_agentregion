import numpy as np
import matplotlib.pyplot as plt
import csv
np.random.seed(1)

# Set up the initial environment
N_region = 5
N_agent = 4

region = np.random.random((N_region+1,2))*10
region[-1] = [0,0]

file_name = 't5a1.csv'
f = open(file_name, 'r')
rdr = csv.reader(f)
policy_t5a1 = []
cnt = 0
for line in rdr:
    policy_t5a1.append([])
    for i in range(len(line)):
        policy_t5a1[cnt].append(float(line[i]))
    cnt += 1
f.close()

file_name = 't5a2.csv'
f = open(file_name, 'r')
rdr = csv.reader(f)
policy_t5a2 = []
cnt = 0
for line in rdr:
    policy_t5a2.append([])
    for i in range(len(line)):
        policy_t5a2[cnt].append(float(line[i]))
    cnt += 1
f.close()

file_name = 't5a3.csv'
f = open(file_name, 'r')
rdr = csv.reader(f)
policy_t5a3 = []
cnt = 0
for line in rdr:
    policy_t5a3.append([])
    for i in range(len(line)):
        policy_t5a3[cnt].append(float(line[i]))
    cnt += 1
f.close()

# Plotting of regions
init_agent = np.zeros(N_agent)
init_unc = np.ones(N_region)

region_ = np.transpose(region)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(region_[0],region_[1],"o")

# Red circles is the agents.
aa = []
tt = []
for i in range(N_agent):
    aa.append([])
    tt.append([])
    aa[i], = ax.plot(0,0,'ro')
    tt[i] = plt.text(0,0,str(i))

init_agent_ = 0
for i in range(N_agent):
    init_agent_ += init_agent[i] * (N_region ** (N_agent - i - 1))

Fuel_max = 5
fuel_state = []
agent_state = []
for i in range(N_agent):
    fuel_state.append(Fuel_max-i)
    agent_state.append(0)

init_unc_ = 0
for i in range(N_region):
    init_unc_ += init_unc[i] * (2 ** (N_region - i - 1))
cnt = 0

agent_moving = []

while cnt < 30:
    cnt += 1
    for i in range(N_agent):
        if fuel_state[i] == 0 and i in agent_moving:
            agent_moving.pop(agent_moving.index(i)) # agent i가 refuel이 필요하여 refueling하러 떠남.
        elif fuel_state[i] > 0 and i not in agent_moving:
            agent_moving.insert(-1, i)
            agent_moving.sort()

    action_out = [N_region,N_region,N_region]

    if len(agent_moving) == 3:
        agent_ = 0
        for i in range(N_agent):
            #agent_ += agent_state[i] * (N_region ** (N_agent - i - 1))
            agent_ += agent_state[i] * (N_region ** (N_agent - i -1))
        action = policy_t5a3[int(agent_)][int(init_unc_)]
        action_ = np.zeros(len(agent_moving))
        for i in range(len(agent_moving)):
            if not i == 0:
                action = action % (N_region ** (N_agent - i))
            action_[i] = int(action / ((N_region) ** (N_agent - i - 1)))

        action_out[0] = action_[0]
        action_out[1] = action_[1]
        action_out[2] = action_[2] 

    elif len(agent_moving) == 2:
        agent_ = 0
        for i in range(len(agent_moving)):
            agent_ += agent_state[agent_moving[i]] * (N_region ** (len(agent_moving) - i - 1))

        action = policy_t5a2[int(agent_)][int(init_unc_)]
        action_ = np.zeros(len(agent_moving))
        for i in range(len(agent_moving)):
            if not i == 0:
                action = action % (N_region ** (N_agent  - i))
            action_[i] = int(action / ((N_region) ** (N_agent  - i - 1)))

        action_out[agent_moving[0]] = action_[0]
        action_out[agent_moving[1]] = action_[1]

    elif len(agent_moving) == 1:
        agent_ = 0
        for i in range(len(agent_moving)):
            agent_ += agent_state[agent_moving[i]] * (N_region ** (len(agent_moving) - i - 1))
        action = policy_t5a1[int(agent_)][int(init_unc_)]
        action_ = np.zeros(len(agent_moving))
        for i in range(len(agent_moving)):

            if not i == 0:
                action = action % (N_region ** (N_agent  - i))
            action_[i] = int(action / ((N_region) ** (N_agent - i - 1)))

        action_out[agent_moving[0]] = action_[0]

    for i in range(N_agent):
        print(i)
        if action_out[i] < N_region:
            print(action_out)
            init_unc[int(action_out[i])] = 0

    init_unc_ = 0
    for i in range(N_region):
        init_unc_ += init_unc[i] * (2 ** (N_region - i - 1))
    if init_unc_ == 0:
        init_unc = np.ones(N_region)
        for i in range(N_agent):
            init_unc[int(action_out[i])] = 0
        init_unc_ = 0
        for i in range(N_region):
            init_unc_ += init_unc[i] * (2 ** (N_region - i - 1))

    for i in range(N_agent):
        if action_out[i] == N_region:
            fuel_state[i] = Fuel_max
        else:
            fuel_state[i] = max(fuel_state[i]-1, 0)

    action_state = action_out
    print(fuel_state,action_out)
    for i in range(N_agent):
        aa[i].set_xdata(region[int(action_out[i])][0])
        aa[i].set_ydata(region[int(action_out[i])][1])
        tt[i].set_position((region[int(action_out[i])][0], region[int(action_out[i])][1]))

    print(cnt, action_out)
    plt.pause(0.5)