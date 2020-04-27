'''
hw1.py
Author: Jiawei Wang

Tufts COMP 135 Intro ML

'''


import numpy as np
import math


#question 1
def split_into_train_and_test(x_all_LF, frac_test=0.5, random_state=None):
    copy = x_all_LF.copy()
    train_len = math.floor(len(copy) * (1-frac_test))

    if random_state is None:
        np.random.shuffle(copy)
    elif isinstance(random_state, int):
        np.random.RandomState(random_state).shuffle(copy)
    else:
        random_state.shuffle(copy)
    return copy[:train_len], copy[train_len:]


#question 2
def calc_k_nearest_neighbors(data_NF, query_QF, K=1):
    list = []

    for v2 in query_QF:
        dict = {}
        temp = []
        for v1 in data_NF:
            dist = distance(v1,v2)
            if dist in dict:
                dict[dist].append(v1)
            else:
                dict[dist] = [v1]
        for i in range(K):
            curr = min(dict.keys())
            temp.append(dict[curr].pop())
            if len(dict[curr]) == 0:
                dict.pop(curr)
        list.append(temp)
    return np.array(list)

def distance(v1,v2):
    sum = 0

    for i in range(len(v1)):
        sum += (v1[i] - v2[i]) * (v1[i] - v2[i])
    return math.sqrt(sum)