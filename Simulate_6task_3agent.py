import numpy as np
import matplotlib.pyplot as plt
import csv
np.random.seed(1)

# Set up the initial environment
N_region = 6
N_agent = 3

region = np.random.random((N_region+1,2))*10
region[-1] = [0,0]

file_name = 't6a1.csv'
f = open(file_name, 'r')
rdr = csv.reader(f)
policy_t6a1 = []
cnt = 0
for line in rdr:
    policy_t6a1.append([])
    for i in range(len(line)):
        policy_t6a1[cnt].append(float(line[i]))
    cnt += 1
f.close()

file_name = 't6a2.csv'
f = open(file_name, 'r')
rdr = csv.reader(f)
policy_t6a2 = []
cnt = 0
for line in rdr:
    policy_t6a2.append([])
    for i in range(len(line)):
        policy_t6a2[cnt].append(float(line[i]))
    cnt += 1
f.close()

file_name = 't6a3.csv'
f = open(file_name, 'r')
rdr = csv.reader(f)
policy_t6a3 = []
cnt = 0
for line in rdr:
    policy_t6a3.append([])
    for i in range(len(line)):
        policy_t6a3[cnt].append(float(line[i]))
    cnt += 1
f.close()

#
init_agent = np.zeros(N_agent)
init_unc = np.ones(N_region)

region_ = np.transpose(region)
_,axes = plt.subplots(1,3)
ax = axes[0]
ax2 = axes[1]
ax3 = axes[2]
ax.plot(region_[0],region_[1],"ko")
aa = []
tt = []
for i in range(N_agent):
    aa.append([])
    tt.append([])
    aa[i], = ax.plot(0,0,'bo')
    tt[i] = ax.text(0,0,str(i))
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
unc =  [1] * N_region
while cnt < 20:
    cnt += 1
    for i in range(N_agent):
        if fuel_state[i] == 0 and i in agent_moving:
            agent_moving.pop(agent_moving.index(i))
        elif fuel_state[i] > 0 and i not in agent_moving:
            agent_moving.insert(-1, i)
            agent_moving.sort()

    action_out = []
    for i in range(N_agent):
        action_out.append(float(N_region))

    if len(agent_moving) > 0:
        agent_ = 0
        for i in range(len(agent_moving)):
            agent_ += agent_state[agent_moving[i]] * ((N_region + 1) ** (len(agent_moving) - i - 1))
        if len(agent_moving) == 3:
            ax.plot(0,0,'ko')
            action = policy_t6a3[int(agent_)][int(init_unc_)]
        elif len(agent_moving) == 2:
            action = policy_t6a2[int(agent_)][int(init_unc_)]
            refueling, = ax.plot(0,0,'ro')
        elif len(agent_moving) == 1:
            action = policy_t6a1[int(agent_)][int(init_unc_)]
            refueling, = ax.plot(0,0,'ro')
        else:
            action = -1
        if action == -1:
            print("ERROR------------------------")
        action_ = np.zeros(len(agent_moving))
        for i in range(len(agent_moving)):
            if not i == 0:
                action = action % ((N_region + 1) ** (len(agent_moving) - i))
            action_[i] = int(action / ((N_region + 1) ** (len(agent_moving) - i - 1)))

        for i in range(len(agent_moving)):
            action_out[agent_moving[i]] = action_[i]

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
            fuel_state[i] = max(fuel_state[i]-1, 0)

    agent_state[0] = action_out[0]
    agent_state[1] = action_out[1]
    # there is no uncertainties in all regions, uncertainties get back to all zeros. -> We want continuously discover the area.
    if unc == [0] * N_region:
        unc = [1] * N_region
    for i in range(len(unc)):
        if unc[i] == 1 and init_unc[i] == 1:
            unc[i] = 1
        else:
            unc[i] = 0
    ax2.bar(range(N_region),init_unc,color='b')        
    ax3.bar(range(N_region),unc,color='r')
    print(fuel_state,action_out,init_unc,unc)
    for i in range(N_agent):
        aa[i].set_xdata(region[int(action_out[i])][0])
        aa[i].set_ydata(region[int(action_out[i])][1])
        tt[i].set_position((region[int(action_out[i])][0], region[int(action_out[i])][1]))

    print(cnt, action_out)
    plt.pause(0.8)
    ax2.clear()
    ax3.clear()