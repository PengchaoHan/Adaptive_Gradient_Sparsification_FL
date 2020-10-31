import random
import copy
import numpy as np


class CONTINUOUS_BANDIT:
    def __init__(self, dim, max, min, T):
        self.delta = 1/ np.power(T,1/4)
        self.eta = (max-min+1)/dim/np.power(T,3/4)
        self.dim = dim  # Dimension of actions
        self.max_action = max
        self.min_action = min
        self.max_delta_action = self.max_action-self.delta
        self.min_delta_action = self.min_action+self.delta
        self.action = (self.max_action + self.min_action) / 200  # Initial action point
        u_ = random.uniform(0, 1)
        if u_ <= 0.5:
            self.u = 1  # Initial unit vector
        else:
            self.u = -1

    def get_initial_action(self):
        action = self.action + self.delta * self.u
        pro_action = self.project_action(action)
        return self.stochasitic_rouding(pro_action)

    def project_action(self, action):
        pro_action = copy.deepcopy(action)
        for i in range(self.dim):  # Project action
            if pro_action[i] > self.max_delta_action[i]:
                pro_action[i] = self.max_delta_action[i]
            elif pro_action[i] < self.min_delta_action[i]:
                pro_action[i] = self.min_delta_action[i]
        return pro_action

    def get_next_action(self, loss):
        if loss == None:
            action = self.action
        else:
            grad = self.dim / self.delta * loss * self.u  # Estimate gradient
            a = self.action - self.eta * grad  # Update action
            self.action = self.project_action(a)
            u_ = random.uniform(0, 1)
            if u_ <= 0.5:
                self.u = 1
            else:
                self.u = -1
            action = self.action + self.delta * self.u  # Project the point x onto the nearest point in convex set
        return self.stochasitic_rouding(action)

    def stochasitic_rouding(self, x):
        floor_x = int(np.floor(x))
        prob = random.random()
        if prob < x - floor_x:
            x = floor_x + 1
        else:
            x = floor_x
        return x
