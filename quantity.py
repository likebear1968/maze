import numpy as np
import itertools
from board import Board

class StochasticPolicyGradient():
    def __init__(self, beta=1.0, eta=0.1):
        self.beta, self.eta = beta, eta
        self.theta, self.pi = None, None
        self.init_theta()

    def init_theta(self):
        # 行は状態、列はアクション[↑、→、↓、←]
        self.theta = np.array([
            [np.nan, 1,      1,      np.nan],  # s0
            [np.nan, 1,      np.nan, 1],       # s1
            [np.nan, np.nan, 1,      1],       # s2
            [np.nan, np.nan, 1,      np.nan],  # s3
            [1,      1,      1,      np.nan],  # s4
            [np.nan, np.nan, 1,      1],       # s5
            [1,      1     , np.nan, np.nan],  # s6
            [1,      np.nan, 1,      1],       # s7
            [1,      np.nan, 1,      np.nan],  # s8
            [1,      1,      np.nan, np.nan],  # s9
            [np.nan, 1,      np.nan, 1],       # s10
            [1,      np.nan, 1     , 1],       # s11
            [1,      1     , np.nan, np.nan],  # s12
            [np.nan, 1     , np.nan, 1],       # s13
            [np.nan, 1     , np.nan, 1],       # s14
            [1,      np.nan, np.nan, 1],       # s15
        ])
        return self.theta

    def init_quantity(self):
        self.init_theta()
        self.pi = self.to_probability(self.theta, self.beta)
        return self.pi

    def to_probability(self, target, beta):
        tmp = target.copy()
        #tmp -= np.nanmax(tmp, axis=1, keepdims=True)
        tmp = np.exp(tmp ** beta)
        tmp /= np.nansum(tmp, axis=1, keepdims=True)
        return np.nan_to_num(tmp)

    def next_state(self, action, state):
        return state + Board.DIRECTION[action]

    def next_behavior(self, state):
        action = np.random.choice(range(self.pi.shape[1]), p=self.pi[state])
        return self.next_state(action, state), action

    def update(self, history):
        step = len(history) - 1
        delta = self.theta.copy()
        R, C = self.theta.shape
        for s, a in itertools.product(range(R), range(C)):
            if np.isnan(self.theta[s, a]): continue
            # 状態sの総行動数
            scount = len([h for h in history if h[0] == s])
            # 状態sで行動aをとった回数
            acount = len([h for h in history if h == [s, a]])
            delta[s, a] = (acount + self.pi[s, a] * scount) / step
        self.theta += self.eta * delta
        self.pi = self.to_probability(self.theta, self.beta)
        return self.pi

class QLearning(StochasticPolicyGradient):
    def __init__(self, beta=1.0, eta=0.1, epsilon=0.5, gamma=0.9):
        super().__init__(beta, eta)
        self.epsilon, self.gamma = epsilon, gamma
        self.Q = None

    def init_quantity(self, alfa=0.1):
        self.init_theta()
        self.Q = np.random.rand(*self.theta.shape) * self.theta * alfa
        return self.Q

    def next_behavior(self, state):
        if np.random.rand() < self.epsilon:
            p = self.to_probability(self.Q, beta=1.0)
            action = np.random.choice(range(self.Q.shape[1]), p=p[state])
        else:
            action = range(self.Q.shape[1])[np.nanargmax(self.Q[state])]
        return self.next_state(action, state), action

    def update(self, state, action, reward, nstate):
        if nstate == Board.GOAL:
            self.Q[state, action] += self.eta * (reward - self.Q[state, action])
        else:
            self.Q[state, action] += self.eta * (reward + self.gamma * np.nanmax(self.Q[nstate]) - self.Q[state, action])
        return self.Q

class Sarsa(QLearning):
    def __init__(self, beta=1.0, eta=0.1, epsilon=0.5, gamma=0.9):
        super().__init__(beta, eta, epsilon, gamma)

    def update(self, state, action, reward, nstate, naction):
        if nstate == Board.GOAL: return super().update(state, action, reward, nstate)
        self.Q[state, action] += self.eta * (reward + self.gamma * self.Q[nstate, naction] - self.Q[state, action])
        return self.Q
