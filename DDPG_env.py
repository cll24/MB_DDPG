import numpy as np
from User_Edge import *

N_APPLICATIONS = 20
Application_service = get_Application_service()
Application_lists = get_Application_miroservice()
USER_REQUESTS = get_user_request()

N_USER = 47

N_SERVICES = 15

N_EDGENODES = 4

EDGE_POSITION, USER_POSITION = get_edge_users_location()
EDGE_RADIUS = [150, 70, 130, 80]

EDGE_BANDWIDTH = [
    [0, 100, 200, 100, 1000],
    [100, 0, 150, 400, 1000],
    [200, 150, 0, 100, 1000],
    [100, 400, 100, 0, 1000]
]
EDGE_CAPACITY = [[1800, 1300], [600, 900], [1500, 1800], [1400, 1300]]
SERVICE_SET = [[15, 10], [5, 8], [15, 26], [4, 8],[5,9],
               [10, 12], [5, 6], [11, 11],[4,6],[3,7],
               [8, 9], [14, 16], [10,10],[4,9],[2,2]]

user_edge_v = 1000

user_cloud_v = 10
user_cloud_time = 1000

edge_cloud_v =500
edge_cloud_time = 500

request_over_time = get_request_over_time()


def initial(t):
    state = np.zeros(shape=(N_SERVICES + 1, N_EDGENODES))
    for i in range(N_SERVICES):
        np.random.seed(i + 10)
        j = np.random.randint(0, 4)
        state[i][j] = 1
    for i in range(N_EDGENODES):
        state[N_SERVICES][i] = t
    state = np.reshape(state, (1, (N_SERVICES + 1) * N_EDGENODES))
    return state

def change_state(state, action):
    state1 = np.reshape(state, ((N_SERVICES + 1, N_EDGENODES)))
    state_t = state1[N_SERVICES]
    state_s = state1[0:N_SERVICES]
    action = np.reshape(action, (N_SERVICES, N_EDGENODES))

    new_state = 0.5 * state_s + action
    indexs = np.argmax(new_state, axis=1)
    new_state = np.zeros(shape=(N_SERVICES + 1, N_EDGENODES))
    for i, j in enumerate(indexs):
        new_state[i][j] = 1
    new_state[N_SERVICES] = state_t
    new_state = np.reshape(new_state, (1, (N_SERVICES + 1) * N_EDGENODES))
    return new_state


def evaluate_state(t,state, state_):
    state = np.reshape(state, (N_SERVICES+1, N_EDGENODES))
    state_ = np.reshape(state_, (N_SERVICES+1, N_EDGENODES))
    average_time_state = compute_system_response(t,state)
    average_time_state_ = compute_system_response(t,state_)
    reward = (average_time_state - average_time_state_) / 10 + 100/average_time_state_
    return average_time_state,average_time_state_,reward

def compute_system_response(t,state):
    avail_resource = [[1800, 1300], [600, 900], [1500, 1800], [1400, 1300]]
    all_response_time = 0
    user_requests = USER_REQUESTS[t]
    user_positions = USER_POSITION[t]
    for j in range(N_USER):
        avail_edge = -1
        min_distance = 10000000
        user_position = user_positions[j]
        user_request = user_requests[j]
        user_data = 10000

        if len(user_request) == 0:
            break

        for i in range(N_EDGENODES):
            distance = np.sqrt(np.square(user_position[0]-EDGE_POSITION[i][0]) + np.square(user_position[1]-EDGE_POSITION[i][1]))
            if (distance < EDGE_RADIUS[i]) and (distance < min_distance):
                min_distance = distance
                avail_edge = i

        if avail_edge == -1:


            for i in range(len(user_request)):
                t_cloud_up = user_cloud_time
                t_cloud_ex = 0
                t_cloud_down = user_cloud_time
                all_response_time += t_cloud_up + t_cloud_ex + t_cloud_down
                # print(all_response_time)
        else:
            for i in range(len(user_request)):

                t_edge_up = min_distance / user_edge_v
                t_edge_down = min_distance / user_edge_v
                all_response_time += t_edge_up + t_edge_down

                edge_index = []
                service_number = Application_service[user_request[i]]
                service_index = Application_lists[user_request[i]]


                for z in range(service_number):
                    edge_index.append(search_edge_index(state, service_index[z]))  # 得到的是service


                avail_resource_edge = avail_resource[edge_index[0]]

                if SERVICE_SET[0][0] < avail_resource_edge[0] and SERVICE_SET[0][1] < avail_resource_edge[1] :


                    avail_resource[edge_index[0]] = np.subtract(avail_resource[edge_index[0]], SERVICE_SET[0])

                    if EDGE_BANDWIDTH[avail_edge][edge_index[0]] == 0:
                        t_edge_transport = 0
                    else:
                        t_edge_transport = user_data / EDGE_BANDWIDTH[avail_edge][edge_index[0]]

                    t_edge_ex = user_data / 1000

                    user_data = 0.5 * user_data
                    all_response_time += t_edge_transport + t_edge_ex

                    for z in range(1, service_number):
                        avail_resource_edge = avail_resource[edge_index[z]]

                        if SERVICE_SET[z][0] <= avail_resource_edge[0] and SERVICE_SET[z][1] <= avail_resource_edge[1]:


                            avail_resource[edge_index[z]] = np.subtract(avail_resource[edge_index[z]], SERVICE_SET[z])

                            if EDGE_BANDWIDTH[edge_index[z - 1]][edge_index[z]] == 0:
                                t_edge_transport = 0
                            else:
                                t_edge_transport = user_data / EDGE_BANDWIDTH[edge_index[z - 1]][edge_index[z]]

                            t_edge_ex = user_data / 1000
                            user_data = 0.5 * user_data
                            all_response_time += t_edge_transport + t_edge_ex
                        else:

                            t_cloud_up = edge_cloud_time
                            t_cloud_down = edge_cloud_time
                            t_cloud_ex = 0
                            all_response_time += t_cloud_up + t_cloud_down + t_cloud_ex
                            break



                else:

                    t_cloud_up = edge_cloud_time
                    t_cloud_down = edge_cloud_time
                    t_cloud_ex = 0

                    all_response_time += t_cloud_up + t_cloud_down + t_cloud_ex

    average_time = all_response_time / len(request_over_time[t])

    return average_time



def search_edge_index(state, service_index):
    state = np.reshape(state, (N_SERVICES+1, N_EDGENODES))
    for i,j in enumerate(state[service_index]):
        if j==1:
            return i