import numpy as np
from board import Board
from abc import ABCMeta, abstractmethod
from enum import Enum

class QTYPE(Enum):
    SPG = 1
    QL = 2
    SARSA = 3

class Player(metaclass=ABCMeta):
    @abstractmethod
    def play(self, quantity, state=0):
        pass

class Player_spg(Player):
    def play(self, quantity, state=0):
        history = [[state, np.nan]]
        while True:
            state, action = quantity.next_behavior(state)
            history[-1][1] = action
            history.append([state, np.nan])
            if state == Board.GOAL: break
        return history

class Player_ql(Player):
    def play(self, quantity, state=0):
        history = [[state, np.nan]]
        while True:
            nstate, action = quantity.next_behavior(state)
            history[-1][1] = action
            history.append([nstate, np.nan])
            if nstate == Board.GOAL:
                quantity.update(state, action, 1, nstate)
                break
            else:
                quantity.update(state, action, 0, nstate)
                state = nstate
        return history

class Player_sarsa(Player):
    def play(self, quantity, state=0):
        history = [[state, np.nan]]
        while True:
            nstate, action = quantity.next_behavior(state)
            history[-1][1] = action
            history.append([nstate, np.nan])
            if nstate == Board.GOAL:
                quantity.update(state, action, 1, nstate, None)
                break
            else:
                _, naction = quantity.next_behavior(nstate)
                quantity.update(state, action, 0, nstate, naction)
                state = nstate
        return history

class PFactory():
    def __init__(self, qtype):
        self.qtype = qtype

    def create(self):
        if self.qtype == QTYPE.SPG: return Player_spg()
        if self.qtype == QTYPE.QL: return Player_ql()
        if self.qtype == QTYPE.SARSA: return Player_sarsa()
        return None
