import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import einops as eops


class NextStateModel():
    def __init__(self, _nState :int, _nAction :int):
        self._nState = _nState
        self._nAction = _nAction
        self.A = np.eye(_nState)
        self.B = np.zeros(shape=(_nState, _nAction))

    def train(self, States :np.ndarray, Actions :np.ndarray, lr :float=1e-2, epochs :int=1000):
        if States.ndim != 2 or Actions.ndim != 2:
            print("Dimension mismatch")
            return
            
        _nState, _nSample = States.shape
        _nAction, _ = Actions.shape


        if _nState!=self._nState or _nAction != self._nAction:
            print(f"Req:({self._nState},{self._nAction}) provided:({_nState},{_nAction})")
            return


        A=self.A.copy()
        B=self.B.copy()

        for j in range(epochs):
            prev_state = States[:, 0:1]
            action = Actions[:, 0:1]

            dL_dA = 0
            dL_dB = 0
            err_total = 0
            for i in range(1, _nSample):
                curr_state = States[:, i:i+1]
    
                pred = A@prev_state + B@action
                err = curr_state - pred

                err_total += err.mean()*100
                
                dL_dA -= (2*err) @ prev_state.T
                dL_dB -= (2*err) @ action.T

                prev_state = curr_state
                action = Actions[:, i:i+1]
                
            dL_dA /= _nSample
            dL_dB /= _nSample

            err_total /= 699
            print(f"Err standing at {abs(err_total)}% at {j}th epochs")

            A -= lr*dL_dA
            B -= lr*dL_dB


        self.A = A
        self.B = B

    def __predict__(self, state :np.ndarray, action :np.ndarray) -> np.ndarray:
        if state.shape[0] != self._nState or action.shape[0] != self._nAction:
            print("Dim mismatch")
            return None

        return (self.A@state + self.B@action)