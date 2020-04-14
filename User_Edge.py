## This is a Data processing problem
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import random

import matplotlib
matplotlib.rcParams['backend'] = 'SVG'


def get_requestnumber_over_time():
    requestnumber_over_time = [60, 50, 35, 23, 18, 24, 38, 59, 70, 65, 62, 50, 49, 54, 60, 65, 78, 95, 110, 120, 115,
                               110, 85,
                               70]
    requestnumber_over_time = np.multiply(requestnumber_over_time,1)
    m = 0
    plt.figure(figsize=(10, 6), dpi=80)
    # plt.subplot(1, 1, 1)
    N = len(requestnumber_over_time)
    values = requestnumber_over_time
    index = np.arange(N)
    p2 = plt.bar(index, values, label="num", color="#87CEFA")
    plt.xlabel('Time Slot in One Day')
    plt.ylabel('Number of Requests')
    plt.title('Requests Distribution over Time')
    plt.xticks(index)
    # plt.yticks(np.arange(0, 10000, 10))
    plt.legend(loc="upper right")

    plt.savefig("D:\study\IoTJ论文准备\图片\请求图.eps", format = 'eps')
    plt.show()

    return requestnumber_over_time



def get_edge_users_location():
    user_matrix_over_time=[]
    edge_matrix = np.loadtxt(open('mature_edges.csv'), delimiter=' ')
    user_matrix = np.loadtxt(open('mature_user.csv'), delimiter=' ')
    min_latitude, min_longitude = 0, 150
    max_latitude, max_longitude = -40, 0
    for i in user_matrix:
        if i[0] < min_latitude:
            min_latitude = i[0]
        if i[0] > max_latitude:
            max_latitude = i[0]
        if i[1] < min_longitude:
            min_longitude = i[1]
        if i[1] > max_longitude:
            max_longitude = i[1]

    for j in edge_matrix:
        if j[0] < min_latitude:
            min_latitude = j[0]
        if j[1] < min_longitude:
            min_longitude = j[1]

    t = 0
    while t < 24:
        user_location = np.zeros((47, 2))
        random.seed(t)
        for m in range(47):

            user_location[m][0] = random.uniform(min_latitude, max_latitude)
            user_location[m][1] = random.uniform(min_longitude, max_longitude)
        user_matrix_over_time.append(user_location)
        t += 1

    t = 0
    while t < 24:
        user_matrix = user_matrix_over_time[t]
        edge_matrix = np.loadtxt(open('mature_edges.csv'), delimiter=' ')
        # print(user_matrix_t)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.xlabel('x')
        plt.ylabel('y')

        edge_radius = [150, 70, 130, 80]
        for i in (range(len(edge_matrix))):
            edge_matrix[i][0] = (edge_matrix[i][0] - min_latitude) * 111110
            edge_matrix[i][1] = (edge_matrix[i][1] - min_longitude) * 111110
            random.seed(2 * i)  # 141待选

            circle = plt.Circle((edge_matrix[i][0], edge_matrix[i][1]), radius=edge_radius[i], alpha=0.5)
            ax.plot(edge_matrix[i][0], edge_matrix[i][1], 'darkblue')
            ax.add_patch(circle)

            ax.plot(edge_matrix[i][0], edge_matrix[i][1], 'ro')
            plt.axis('equal')


        for i in (range(len(user_matrix))):
            user_matrix[i][0] = (user_matrix[i][0] - min_latitude) * 111110
            user_matrix[i][1] = (user_matrix[i][1] - min_longitude) * 111110
            ax.plot(user_matrix[i][0], user_matrix[i][1], 'darkblue')
            plt.scatter(user_matrix[i][0], user_matrix[i][1], alpha=0.5)
        # plt.show()
        plt.savefig("D:\study\IoTJ论文准备\图片\\" + str(t) + ".eps", format = 'eps')
        t += 1

    return edge_matrix,user_matrix_over_time





def get_Application_service():
    i = 0
    Application_service = {}
    while i < 20:
        i += 1
        n = 'application' + str(i)
        if i <= 2:
            Application_service[n] = 2
        elif i <= 12:
            Application_service[n] = 3
        elif i <= 18:
            Application_service[n] = 4
        else:
            Application_service[n] = 5
    return Application_service



def get_request_over_time():
    Request_over_time = []
    t = 0
    requestnumber = get_requestnumber_over_time()
    Application_service = get_Application_service()
    while t < 24:
        user_to_application =[]
        n = requestnumber[t]
        numbers = int(n*0.8)
        i = 0
        random.seed(t)
        app = [random.randint(0,19) for _ in range(4)]
        while i < n:
            if i < numbers:
                random.seed(i+10)
                rand = random.choice(app)
                name = 'application' + str(rand + 1)
            else:
                random.seed(i+10)
                rand = random.randint(0,100) % 20
                name = 'application' + str(rand+1)
            user_to_application.append(name)
            i+=1
        t +=1
        Request_over_time.append(user_to_application)
    return Request_over_time




def get_Application_miroservice():
    Application_lists = {}
    i = 0
    while i < 20:
        Application_i = []
        if i < 2:
            i += 1
            Application_i.append(i-1)
            random.seed(i)
            Application_i.append(random.randint(0,150)%15)
            # Application_lists.append(Application_i)
            n = 'application' + str(i)
            Application_lists[n] = Application_i
        elif i <12:
            i += 1
            Application_i.append(i-1)
            random.seed(i+34)
            Application_i.append(random.randint(0,150)%15)
            random.seed(i+100)
            Application_i.append(random.randint(0,150)%15)
            # Application_lists.append(Application_i)
            n = 'application' + str(i)
            Application_lists[n] = Application_i
        elif i<18:
            i += 1
            random.seed(i+99)
            Application_i.append(random.randint(0,150)%15)
            random.seed(i+199)
            Application_i.append(random.randint(0,150)%15)
            random.seed(i+299)
            Application_i.append(random.randint(0,150) % 15)
            random.seed(i+399)
            Application_i.append(random.randint(0,150) % 15)
            # Application_lists.append(Application_i)
            n = 'application' + str(i)
            Application_lists[n] = Application_i
        elif i<20:
            i += 1
            random.seed(i+6)
            Application_i.append(random.randint(0,150)%15)
            random.seed(i+7)
            Application_i.append(random.randint(0,150)%15)
            random.seed(i+8)
            Application_i.append(random.randint(0,150) % 15)
            random.seed(i+9)
            Application_i.append(random.randint(0,150) % 15)
            random.seed(i+10)
            Application_i.append(random.randint(0,150) % 15)
            # Application_lists.append(Application_i)
            n = 'application' + str(i)
            Application_lists[n] = Application_i

    return Application_lists

def get_user_request():
    user_number = 47
    user_requeset = []
    request_over_time = get_request_over_time()
    for t in range(24):
        user_requeset_t = [[] for _ in range(user_number)]
        request_over_time_t = request_over_time[t]
        for i in range(len(request_over_time_t)):
            user_index = i%47
            user_requeset_t[user_index].append(request_over_time_t[i])
        user_requeset.append(user_requeset_t)
    return user_requeset





if __name__ == '__main__':
    # User_to_Application = user_to_Applicaiton() #47个用户，对Application的访问的一个list
    # edge_matrix, user_matrix, _, _ , _ , _ = get_edge_users_location() #用户和edge坐标矩阵
    # application_lists = get_Application_miroservice()
    # print(application_lists)
    # print(len(application_lists))

    # edge_matrix, user_matrix, edge_matrix_x, edge_matrix_y, user_matrix_x, user_matrix_y = get_edge_users_location()
    # plt.scatter(edge_matrix_x, edge_matrix_y, c='dodgerblue', label='Edge Nodes')
    # plt.scatter(user_matrix_x, user_matrix_y, c='red', label='Users')
    # plt.legend()
    # plt.show()
    #get_edge_users_location()
    # Request_over_time = get_request_over_time()
    # for i in Request_over_time:
    #     print(i)
    print(get_user_request())
