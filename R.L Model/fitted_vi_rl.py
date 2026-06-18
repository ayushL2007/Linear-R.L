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
        self._nAction = _nactions
        self.reward_fn = lambda state, action : rewardFn_(state, action)

    def runSGD(self, y_i :np.ndarray, _state :np.ndarray, theta):
        diff = 2 * (theta @ _state) - y_i
        grad = diff * _state
        # print(diff.shape, _state.shape, y_i.shape, grad.shape)
        self.theta -= self.lr*grad.T
        return theta
        
        

    #Follows Fitted value iteration
    def train(self, States :np.ndarray, Actions :np.ndarray, lr :float=1e-4, epochs :int=1000, gamma :float=0.99):
        self.ns_pred_model = lr_model
        model = self.ns_pred_model
        
        if States.ndim != 2 or Actions.ndim != 2:
            print("Dimension mismatch")
            return
            
        _nState, _nSample = States.shape
        _nAction, _ = Actions.shape


        if _nState!=self._nState or _nAction != self._nAction:
            print(f"Req:({self._nState},{self._nAction}) provided:({_nState},{_nAction})")
            return

        
        self.lr = lr

        for iter_ in range(epochs):
            s_i = (_nSample//epochs)*iter_
            print(f"running {iter_+1}th epoch at loss:{self.acc(States[:,0:1], Actions[:, 0:1])}")
            print(f"Updating theta from {self.theta}", end="")
            theta = self.theta.copy()
            
            skipped=0
            for i in range(_nSample):
                pass_or_not = np.random.rand(1)
                if pass_or_not>0.5:
                    skipped+=1
                    continue
                state = States[:, i:i+1]
                actions = np.arange(-3.0, 4, 1).reshape(7,1)
                
                ns_pred = np.array([model.__predict__(state, actions[action][:].reshape(1,1)) for action in range(actions.shape[0])])
                
                rewards = self.reward_fn(state, Actions[:,i:i+1])
                ns_pred = eops.rearrange(ns_pred, "r c e-> c (r e)")
                q_a = np.array([rewards+gamma*(self.theta@ns_pred[:,ns:ns+1]) for ns in range(ns_pred.shape[1])])
                y_i = np.max(q_a)
                
                self.runSGD(y_i, state, theta)

            theta = self.theta.copy()
            
            print(f" to {self.theta}")
            print(f"skipped: {skipped} data pts")



    def acc(self, state, action):
        pred_action = self.predOptimalAction(state)
        diff = np.abs(pred_action-action)
        acc = (diff**2).mean()
        return acc

    def predOptimalState(self, State :np.ndarray):
        if State.shape[0]!=self._nState:
            print("Mismatching Dimension")
            return

        return self.theta@State
                                   
    def predOptimalAction(self, State :np.ndarray):
        possible_actions = np.arange(-3.0, 4, 1).reshape(7,1)
        
        possible_ns_ = [self.ns_pred_model.__predict__(State, possible_actions[i:i+1, :]) 
                        for i in range(possible_actions.shape[0])]
        
        # Calculate the Value of each predicted next state
        values = [self.theta @ ns for ns in possible_ns_]

        print(values)
        # The optimal action is the one that leads to the highest future value
        best_action_idx = np.argmax(values)
        return possible_actions[best_action_idx]
