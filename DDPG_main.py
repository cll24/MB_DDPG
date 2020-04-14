import numpy as np
from DDPG import *
import DDPG_env


MAX_EPISODES = 100
state_dim = (DDPG_env.N_SERVICES+1) * DDPG_env.N_EDGENODES
action_dim = DDPG_env.N_SERVICES * DDPG_env.N_EDGENODES

response_time_threshold = 50

average_time_state = 100000
min_average_time = average_time_state
max_average_time = 0

var = np.ones(24)

ddpg = DDPG(state_dim, action_dim)

min_average_time = np.ones(24)
min_average_time = np.multiply(min_average_time,1000000)
time_all = []
for i in range(MAX_EPISODES):
    state = DDPG_env.initial(0)

    print("迭代 %d ----- " %(i))

    for t in range(24):

        ep_reward = 0
        count = 0
        average_time_state = DDPG_env.compute_system_response(t,state)
        print("**********************************")
        print(average_time_state)
        print("**********************************")
        temp = 1
        while average_time_state > response_time_threshold and count < 1000:

            action = ddpg.choose_action(state)
            action = np.clip(np.random.normal(action, var[t]), -1, 1)
            state_ = DDPG_env.change_state(state, action)

            average_time_state, average_time_state_, reward = DDPG_env.evaluate_state(t,state, state_)
            if average_time_state_ <= min_average_time[t] :
                min_average_time[t] = average_time_state_
                print("////////////////////////////////////////",t)
                print(average_time_state_)


            if average_time_state > max_average_time:
                max_average_time = average_time_state



            if i>=1 and average_time_state_<=min_average_time[t] + 10:
                reward = 1 + (min_average_time[t] - average_time_state_)/10
            elif i==0:
                reward = (average_time_state - average_time_state_) / 10
            else:
                reward = (min_average_time[t] - average_time_state_) / 100


            # 进行存储
            ddpg.store_tracsition(state, action, reward, state_,t)

            state = state_
            average_time_state = average_time_state_
            ep_reward += reward
            count += 1

            var[t] *= .9995

            if i>=1 and average_time_state < min_average_time[t] + 5:
                # print("！！！！！！！！！！！！！！！")
                break


            if ddpg.pointer[t] > MEMORY_CAPACITY:
                print("----------------------------------------------------Learning！！！！！！！！！！！！")
                ddpg.learn(t)

        state = np.reshape(state, ((DDPG_env.N_SERVICES + 1), DDPG_env.N_EDGENODES))
        for k in range(DDPG_env.N_EDGENODES):
            state[DDPG_env.N_SERVICES][k] = t + 1
        state = np.reshape(state, (1, (DDPG_env.N_SERVICES + 1) * DDPG_env.N_EDGENODES))

        print('Episode/Step:%s/%s'%(i, t), ' Reward: %f'%(ep_reward), 'Explore: %.2f'%var[t], "average_time_state is ", average_time_state)
        # print("-------------action-----------")
        # print(action)
        print("-------------state-----------")
        print(state)
        time_all.append(average_time_state)

ddpg.plot_cost()

print(time_all)