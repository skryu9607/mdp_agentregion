import numpy as np
import matplotlib.pyplot as plt
import csv
np.random.seed(1)

# Set up the initial environment
N_region = 4
N_agent = 2

region = np.random.random((N_region+1,2))*10
region[-1] = [0,0]

file_name = 't4a1.csv'
f = open(file_name, 'r')
rdr = csv.reader(f)
policy_t4a1 = []
cnt = 0
for line in rdr:
    policy_t4a1.append([])
    for i in range(len(line)):
        policy_t4a1[cnt].append(float(line[i]))
    cnt += 1
f.close()

file_name = 't4a2.csv'
f = open(file_name, 'r')
rdr = csv.reader(f)
policy_t4a2 = []
cnt = 0
for line in rdr:
    policy_t4a2.append([])
    for i in range(len(line)):
        policy_t4a2[cnt].append(float(line[i]))
    cnt += 1
f.close()
#
init_agent = np.zeros(N_agent)
init_unc = np.ones(N_region)

region_ = np.transpose(region)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(region_[0],region_[1],"o")

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

Fuel_max = 6
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

while cnt < 20:
    cnt += 1
    for i in range(N_agent):
        if fuel_state[i] == 0 and i in agent_moving:
            agent_moving.pop(agent_moving.index(i))
        elif fuel_state[i] > 0 and i not in agent_moving:
            agent_moving.insert(-1, i)
            agent_moving.sort()

    action_out = [float(N_region), float(N_region)]
    if len(agent_moving) == 2:
        agent_ = 0
        for i in range(N_agent):
            agent_ += agent_state[i] * ((N_region+1) ** (N_agent - i - 1))

        action = policy_t4a2[int(agent_)][int(init_unc_)]
        # print("aa", agent_state, agent_, init_unc_, action)
        action_ = np.zeros(len(agent_moving))
        for i in range(len(agent_moving)):
            if not i == 0:
                action = action % ((N_region+1) ** (N_agent - i))
            action_[i] = int(action / ((N_region+1) ** (N_agent - i - 1)))

        action_out[0] = action_[0]
        action_out[1] = action_[1]

    elif len(agent_moving) == 1:
        agent_ = 0
        for i in range(len(agent_moving)):
            agent_ += agent_state[agent_moving[i]] * ((N_region+1) ** (len(agent_moving) - i - 1))

        action = policy_t4a1[int(agent_)][int(init_unc_)]
        action_ = np.zeros(len(agent_moving))
        for i in range(len(agent_moving)):
            if not i == 0:
                action = action % ((N_region+1) ** (len(agent_moving) - i))
            action_[i] = int(action / ((N_region+1) ** (len(agent_moving) - i - 1)))
        # print(agent_moving[0],action_[0])
        action_out[agent_moving[0]] = action_[0]

    # init_unc = np.ones(N_region)
    # for i in range(N_agent):
    #     if action_out[i] < N_region:
    #         init_unc[int(action_out[i])] = 0
    #
    # init_unc_ = 0
    # for i in range(N_region):
    #     init_unc_ += init_unc[i] * (2 ** (N_region - i - 1))

    for i in range(N_agent):
        if action_out[i] < N_region:
            init_unc[int(action_out[i])] = 0

    init_unc_ = 0
    for i in range(N_region):
        init_unc_ += init_unc[i] * (2 ** (N_region - i - 1))

    if init_unc_ == 0:
        init_unc = np.ones(N_region)
        for i in range(N_agent):
            if action_out[i] < N_region:
                init_unc[int(action_out[i])] = 0
        init_unc_ = 0
        for i in range(N_region):
            init_unc_ += init_unc[i] * (2 ** (N_region - i - 1))

        for i in range(N_agent):
            if action_out[i] == N_region:
                fuel_state[i] = Fuel_max-1
            else:
                fuel_state[i] = max(fuel_state[i] - 1, 0)

    for i in range(N_agent):
        if action_out[i] == N_region:
            fuel_state[i] = Fuel_max
        else:
            fuel_state[i] = max(fuel_state[i]-1, 0)

    agent_state[0] = action_out[0]
    agent_state[1] = action_out[1]
    print(fuel_state,action_out,init_unc)
    for i in range(N_agent):
        aa[i].set_xdata(region[int(action_out[i])][0])
        aa[i].set_ydata(region[int(action_out[i])][1])
        tt[i].set_position((region[int(action_out[i])][0], region[int(action_out[i])][1]))

    print(cnt, action_out)
    plt.pause(0.2)