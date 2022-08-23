import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
import time

class info:
    def __init__(self, REWARD, DISCOUNT, MAX_ERROR, N_region, N_agent):
        self.REWARD = REWARD
        self.DISCOUNT = DISCOUNT
        self.MAX_ERROR = MAX_ERROR
        self.N_region = N_region
        self.N_agent = N_agent
        self.region = np.random.random((N_region+1,2))*10 # <-?
        self.region[-1] = [0,0]
        self.mat = np.zeros((N_region+1,N_region+1)) # Cost matrix
        for i in range(N_region+1):
            for j in range(N_region+1):
                self.mat[i][j] = LA.norm(self.region[i] - self.region[j]) # Numpy linear algebra and norm
        # print(self.mat)

def prac(): # Agent를 표현하는 state value. 
    agent = 32142
    N = 5
    aa = np.zeros(N)
    for i in range(N):
        if not i == 0:
            agent = agent % (10**(N - i)) # 나머지를 주는 것. 하나씩 자리수가 없어진다. 
        aa[i] = int(agent / (10**(N - i-1))) # 가장 처음 것이 없어진다. 
    print(aa)

    bb = 0
    for i in range(N):
        bb += aa[i]*(10**(N-i-1))
    print(bb)
    return 0
def has_duplicates(seq): # sequence가 없어진 것.
    return len(seq) != len(set(seq))

def getU(agent, unc, action, Val_mat, Info):
    u = -0.01
    agent_ = np.zeros(Info.N_agent)
    unc_ = np.zeros(Info.N_region) # uncertainty 
    action_ = np.zeros(Info.N_agent)
    for i in range(Info.N_agent):
        if not i == 0:
            agent = agent % ((Info.N_region+1)**(Info.N_agent - i))
        agent_[i] = int(agent / ((Info.N_region+1)**(Info.N_agent - i - 1)))
    for i in range(Info.N_region):
        if not i == 0:
            unc = unc % (2**(Info.N_region - i))
        unc_[i] = int(unc / (2**(Info.N_region - i - 1)))
    for i in range(Info.N_agent):
        if not i == 0:
            action = action % ((Info.N_region+1) ** (Info.N_agent - i))
        action_[i] = int(action / ((Info.N_region+1) ** (Info.N_agent - i - 1)))
        # agent와 unc와 action 를 expression one single value -> single list
    flag = True
    # reward - penalty 계산
    for i in range(Info.N_agent):
        if action_[i] < Info.N_region: # 이게 아닌 경우는 무엇일까?, 예외처리.
            unc__ = unc_
            if unc__[int(action_[i])] == 1:
                u += 0.1
                unc__[int(action_[i])] = 0 # 같은 곳을 방문하면 점수 없음
        u -= Info.mat[int(agent_[i])][int(action_[i])]/150 # Normalization

    # 특수한 상황에서의 Value
    cnt = 0 # Count ? 
    for i in range(Info.N_region):
        if i in action_: 
            if unc_[i] == 1:
                cnt += 1
        else:
            if unc_[i] == 0:
                cnt += 1

    if cnt == Info.N_region: # 도든 지역을 다 돌았다면,
        u += Info.DISCOUNT * 1
    else: # 모든 지역을 다 돌기 전에는 
        u += Info.DISCOUNT * Val_mat[agent][unc]  # Val_mat : valueItreation(Val_mat,Info)
    
    return u

def cal_val(agent, unc, action, Val_mat, Info):
    u = 0
    u += getU(agent, unc, action, Val_mat, Info)
    # 가장 가까운 동네를 Action 으로 줄 수 있게 코딩 필요
    # u += 0.9 * Info.DISCOUNT * getU(agent, unc, action, Val_mat, Info)
    # u += 0.1 * Info.DISCOUNT * getU(agent, unc, action, Val_mat, Info)
    return u

def valueIteration(Val_mat,Info):
    print("During the value iteration:\n")
    N_agent = Info.N_agent
    N_region = Info.N_region
    N_action = (N_region+1) ** N_agent
    cnt = 0
    tic = time.time()
    while True:
        Next_val_mat = np.zeros(((N_region+1)**N_agent, 2**N_region))
        error = 0
        for agent in range((N_region+1)**N_agent):
            for unc in range(2**N_region):
                print(cal_val(agent, unc, 0, Val_mat, Info))
                Next_val_mat[agent][unc] = max([cal_val(agent, unc, action, Val_mat, Info) for action in range(N_action)]) # Bellman update
                error = max(error, abs(Next_val_mat[agent][unc]-Val_mat[agent][unc]))
            toc = time.time()
            print(toc - tic, 'seconds')
        Val_mat = Next_val_mat
        cnt += 1
        toc = time.time()
        print(toc - tic, 'seconds')
        print(cnt)
        if error < Info.MAX_ERROR * (1-Info.DISCOUNT) / Info.DISCOUNT:
            print(cnt)
            break
    return Val_mat

def getOptimalPolicy(Val_mat, Info): # ACTION들을 다 따져서 최고의 BEST POLICY만 가져가는 것.
    N_agent = Info.N_agent
    N_region = Info.N_region
    N_action = (N_region+1) ** N_agent
    policy = np.zeros(((N_region+1)** N_agent, 2 ** N_region))
    for agent in range((N_region+1) ** N_agent):
        for unc in range(2 ** N_region):
            # Choose the action that maximizes the utility
            maxAction, maxU = None, -float("inf")
            for action in range(N_action):
                u = cal_val(agent, unc, action, Val_mat, Info)
                if u > maxU:
                    maxAction, maxU = action, u
            policy[agent][unc] = maxAction
    return policy

def simulate(policy, Info, init_agent, init_unc):
    region = np.transpose(Info.region)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(region[0],region[1],"o")

    aa = []
    tt = []
    for i in range(Info.N_agent):
        aa.append([])
        tt.append([])
        aa[i], = ax.plot(0,0,'ro')
        tt[i] = plt.text(0,0,str(i))
    # 여기 부분이 새로운 formulation. 진법으로 표현하는 것.
    init_agent_ = 0
    for i in range(Info.N_agent):
        init_agent_ += init_agent[i] * (Info.N_region ** (Info.N_agent - i - 1))
    init_unc_ = 0
    for i in range(Info.N_region):
        init_unc_ += init_unc[i] * (2 ** (Info.N_region - i - 1))
    cnt = 0

    while cnt < 30:
        cnt += 1
        action = policy[int(init_agent_)][int(init_unc_)]

        action_ = np.zeros(Info.N_agent)
        for i in range(Info.N_agent):
            if not i == 0:
                action = action % ((Info.N_region+1) ** (Info.N_agent - i))
            action_[i] = int(action / ((Info.N_region+1) ** (Info.N_agent - i - 1)))
        for i in range(Info.N_agent):
            if action_[i] < Info.N_region:
                init_unc[int(action_[i])] = 0
        init_unc_ = 0
        for i in range(Info.N_region):
            init_unc_ += init_unc[i] * (2 ** (Info.N_region - i - 1))
        if init_unc_ == 0:
            init_unc = np.ones(Info.N_region)
            for i in range(Info.N_agent):
                init_unc[int(action_[i])] = 0
            init_unc_ = 0
            for i in range(Info.N_region):
                init_unc_ += init_unc[i] * (2 ** (Info.N_region - i - 1))

        init_agent_ = action

        for i in range(Info.N_agent):
            aa[i].set_xdata(Info.region[int(action_[i])][0])
            aa[i].set_ydata(Info.region[int(action_[i])][1])
            tt[i].set_position((Info.region[int(action_[i])][0],Info.region[int(action_[i])][1]))

        print(cnt,action_)
        plt.pause(0.3)

    return 0