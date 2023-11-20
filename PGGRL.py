from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation
import random as rand
# import seaborn as sns
import math
import numpy as np
import networkx as nx
import pandas as pd
from tqdm import trange
import random
from numba import jit
import matplotlib
from tqdm import tqdm


def lattice():
    G = nx.grid_2d_graph(int(math.sqrt(size)), int(math.sqrt(size)), periodic=True)
    neighborsList = []
    neighborsArray = []
    for i in G.nodes():
        temp = []
        for j in list(G.adj[i]):
            temp.append(j[0] * L + j[1])
        neighborsList.append(temp)
        neighborsArray.append(temp)
    neighborsArray = np.asarray(neighborsArray, dtype='int32')
    return neighborsArray



@jit(nopython=True)
def initialization_strategy(I_c, total_attribute):
    for i in range(size):
        temp = rand.random()
        if temp <= I_c:
            total_attribute[0][i] = 0  # cooperation
        elif I_c * 2 >= temp > I_c:
            total_attribute[0][i] = 1  # defect
        else:
            total_attribute[0][i] = 2  # loners



@jit(nopython=True)
def initialization_payoff(total_attribute, neighborsArray, delta, r):
    for i in range(size):
        update_payoff(i, total_attribute, neighborsArray, delta, r)



@jit(nopython=True)
def update_payoff(i, total_attribute, neighborsArray, delta, r):
    if total_attribute[0][i] == 2:
        total_attribute[1][i] = delta * (r - 1)
    else:
        total_payoff = 0
        cooperation_num = 0
        defect_num = 0
        if total_attribute[0][i] == 0:
            cooperation_num += 1
        else:
            defect_num += 1
        for j in neighborsArray[i]:
            if total_attribute[0][j] == 0:
                cooperation_num += 1
            elif total_attribute[0][j] == 1:
                defect_num += 1
        if total_attribute[0][i] == 0:
            total_payoff += (cooperation_num / (cooperation_num + defect_num)) * r - 1
        elif total_attribute[0][i] == 1:
            total_payoff += (cooperation_num / (cooperation_num + defect_num)) * r
        for j in neighborsArray[i]:
            cooperation_num = 0
            defect_num = 0
            if total_attribute[0][j] == 0:
                cooperation_num += 1
            elif total_attribute[0][j] == 1:
                defect_num += 1
            elif total_attribute[0][j] == 2:
                continue
            for k in neighborsArray[j]:
                if total_attribute[0][k] == 0:
                    cooperation_num += 1
                elif total_attribute[0][k] == 1:
                    defect_num += 1
            if total_attribute[0][i] == 0:
                total_payoff += (cooperation_num / (cooperation_num + defect_num)) * r - 1
            elif total_attribute[0][i] == 1:
                total_payoff += (cooperation_num / (cooperation_num + defect_num)) * r

        total_attribute[1][i] = total_payoff


@jit(nopython=True)
def get_reward(i, state, total_attribute, delta, r):
    total_payoff = 0
    if state == 2:
        total_payoff = delta * (r - 1)
    else:
        cooperation_num = 0
        defect_num = 0
        if state == 0:
            cooperation_num += 1
        else:
            defect_num += 1
        for j in neighborsArray[i]:
            if total_attribute[0][j] == 0:
                cooperation_num += 1
            elif total_attribute[0][j] == 1:
                defect_num += 1
        if state == 0:
            total_payoff += (cooperation_num / (cooperation_num + defect_num)) * r - 1
        elif state == 1:
            total_payoff += (cooperation_num / (cooperation_num + defect_num)) * r
        for j in neighborsArray[i]:
            cooperation_num = 0
            defect_num = 0
            if total_attribute[0][j] == 0:
                cooperation_num += 1
            elif total_attribute[0][j] == 1:
                defect_num += 1
            elif total_attribute[0][j] == 2:
                continue
            for k in neighborsArray[j]:
                if total_attribute[0][k] == 0:
                    cooperation_num += 1
                elif total_attribute[0][k] == 1:
                    defect_num += 1
            if state == 0:
                total_payoff += (cooperation_num / (cooperation_num + defect_num)) * r - 1
            elif state == 1:
                total_payoff += (cooperation_num / (cooperation_num + defect_num)) * r

    return total_payoff


@jit(nopython=True)
def number_of_cooperation(total_attribute):
    number = 0
    for i in range(size):
        if total_attribute[0][i] == 0:
            number += 1
    return number


@jit(nopython=True)
def number_of_defect(total_attribute):
    number = 0
    for i in range(size):
        if total_attribute[0][i] == 1:
            number += 1
    return number


@jit(nopython=True)
def number_of_loners(total_attribute):
    number = 0
    for i in range(size):
        if total_attribute[0][i] == 2:
            number += 1
    return number


@jit(nopython=True)
def choose_action(i, state, q_table, epsilon):
    if np.random.rand() < epsilon or np.all(q_table[i][int(state)] == 0):
        # if np.random.rand() < epsilon:
        action = random.randrange(3)
    else:
        q_values = q_table[i][int(state)]
        action = np.argmax(q_values)  # based on the Q table choose the best action
    return action


@jit(nopython=True)
def monte_carlo_simulation(total_attribute, cooperation, defect, loners, alpha, gamma, epsilon, delta, r, states,
                           actions):
    initialization_strategy(I_c, total_attribute)
    q_table = np.zeros((size, len(states), len(actions)))
    for epoch in range(max_epoch):
        cooperation[epoch] += number_of_cooperation(total_attribute) / size / average
        defect[epoch] += number_of_defect(total_attribute) / size / average
        loners[epoch] += number_of_loners(total_attribute) / size / average
        for timestep in range(size):
            i = rand.randrange(size)
            state = total_attribute[0][i]
            action = choose_action(i, state, q_table, epsilon)
            reward = get_reward(i, action, total_attribute, delta, r)
            next_state = action
            next_q_value = q_table[i][int(next_state)]
            next_max_q_value = np.max(next_q_value)
            q_table[i][int(state), action] += alpha * (
                    reward + gamma * next_max_q_value - q_table[i][int(state), action])
            total_attribute[0][i] = next_state


def is_plot_cooperation_ratio(cooperation, defect, loners):
    plt.figure()
    plt.ylim(0, 1.05)
    plt.plot(range(1, 1 + len(cooperation)), cooperation, label='C')
    plt.plot(range(1, 1 + len(defect)), defect, label='D')
    plt.plot(range(1, 1 + len(loners)), loners, label='L')
    plt.ylabel('proportion')
    plt.xlabel('time step')
    plt.legend(loc='upper right')
    # plt.savefig('result\\delta={}_r={}.jpg'.format(delta, r))


def SaveData(data, rowname, colname, allname, filename):
    df = pd.DataFrame(data, columns=colname)
    df.insert(0, allname, rowname)
    df.to_csv(filename, index=False)


if __name__ == "__main__":
    # the setting of parameters
    I_c = 1 / 3
    max_epoch = 100000
    r = 4.6
    delta = 2
    accumulation = 500
    alpha = 0.8  # learning rate
    gamma = 0.8  # discount factor
    epsilon = 0.02  # ε-greedy
    states = [0, 1, 2]  # 0:cooperation,1:defect,2:loner
    states = np.asarray(states, dtype='int32')
    actions = [0, 1, 2]
    actions = np.asarray(actions, dtype='int32')

    # initialize the network structure
    average = 1
    L = 100
    size = pow(L, 2)

    # results
    cooperation = np.zeros(max_epoch, dtype=np.float32)
    defect = np.zeros(max_epoch, dtype=np.float32)
    loners = np.zeros(max_epoch, dtype=np.float32)

    # 0:strategy, 1:payoff
    total_attribute = np.zeros((2, size), dtype=np.float32)

    # lattice network
    neighborsArray = lattice()

    for avg in trange(average):
        # MC
        monte_carlo_simulation(total_attribute, cooperation, defect, loners, alpha, gamma, epsilon, delta, r, states,
                               actions)

    # visualization
    is_plot_cooperation_ratio(cooperation, defect, loners)
    plt.show()
