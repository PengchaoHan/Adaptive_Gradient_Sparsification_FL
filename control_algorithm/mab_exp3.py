# Ep3 algorithm for multi-armed bandit (MAB)
import numpy as np
import random


class EXP3:
    def __init__(self, dim, T, choices):
        self.gamma = np.sqrt(np.log(dim)/dim/T)
        self.weights = np.ones([len(choices)])
        self.prob = (1-self.gamma)*self.weights/sum(self.weights) + self.gamma/len(choices)
        self.choices = choices

    def pick_choice(self):
        x = random.uniform(0, 1)
        cumulative_probability = 0.0
        for item, item_probability in zip(self.choices, self.prob):
            cumulative_probability += item_probability
            if x < cumulative_probability:
                break
        return item

    def weight_update(self, choice, reward):
        reward_ = np.zeros([len(self.choices)])
        idx = self.choices.index(choice)
        reward_[idx] = reward/self.prob[idx]
        self.weights = self.weights*np.exp(self.gamma * reward_ / len(self.choices))
        self.prob = (1 - self.gamma) * self.weights / sum(self.weights) + self.gamma / len(self.choices)

    def step(self, choice, reward):
        self.weight_update(choice, reward)
        choice_nxt = self.pick_choice()
        return choice_nxt, self.prob




