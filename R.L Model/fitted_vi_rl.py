import sys
import os
sys.path.append(os.path.abspath("../NS-Model"))
from ns_pred import NextStateModel
import reward

import numpy as np
import einops as eops

class RL():
    def __init__(self, _nstates :int, _nactions :int, rewardFn_):
        self.theta = np.zeros(shape=(1, _nstates))
        self._nState = _nstates
        self._nActions = _nactions
        self.reward_fn = lambda state : rewardFn_(state)

    def runSGD(self, y_i :np.ndarray, _state :np.ndarray):
        diff = 2 * self.theta @ _state - y_i
        grad = diff * _state
        self.theta -= self.lr*grad

    #Follows Fitted value iteration
    def train(self, States :np.ndarray, Actions :np.ndarray, lr :float=1e-2, epochs :int=1000, gamma :float=0.99):
        ns_pred_model = NextStateModel(self._nState, self._nActions)
        ns_pred_model.train(States, Actions, epochs)
        
        if States.ndim != 2 or Actions.ndim != 2:
            print("Dimension mismatch")
            return
            
        _nState, _nSample = States.shape
        _nAction, _ = Actions.shape


        if _nState!=self._nState or _nAction != self._nAction:
            print(f"Req:({self._nState},{self._nAction}) provided:({_nState},{_nAction})")
            return

        theta = self.theta
        self.lr = lr
        
        for iter_ in range(epochs):
            for i in range(_nSample):
                state = States[:, i:i+1]
                
                ns_pred = np.array([model.__predict__(state, action).flatten() for action in np.arange(-3, 3.25, 0.25)])
                rewards = self.reward_fn(state)
                
                print(ns_pred.shape)
                
                q_a = np.array([rewards+gamma*(theta@ns) for ns in ns_pred])
                print(q_a.shape)
                y_i = np.max(q_a)
                
                self.runSGD(y_i, state)