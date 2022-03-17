"""
Class to determine wind velocity at any given moment,
calculates a steady wind speed and uses a stochastic
process to represent wind gusts. (Follows section 4.4 in uav book)
"""
import sys
sys.path.append('..')
from tools.transfer_function import transferFunction
import numpy as np


class WindSimulation:
    def __init__(self, Ts):
        # steady state wind defined in the inertial frame
        self._steady_state = np.array([[0., 0., 0.]]).T
        # self._steady_state = np.array([[0., 5., 0.]]).T

        #   Dryden gust model parameters (section 4.4 UAV book)
        Va =  5.   # must set Va to a constant value
        Lu = 533.   # 200., 533.
        Lv = 533.   # 200., 533.
        Lw = 533.    # 50., 533.
        gust_flag = True
        if gust_flag==True:
            sigma_u = 3.0  # 1.06, 2.12, 1.5, 3.0
            sigma_v = 3.0  # 1.06, 2.12, 1.5, 3.0
            sigma_w = 3.0   # 0.7, 1.4, 1.5, 3.0
        else:
            sigma_u = 0.
            sigma_v = 0.
            sigma_w = 0.

        # Dryden transfer functions (section 4.4 UAV book)
        b0_u = sigma_u*np.sqrt(2*Va/Lu)
        # a1_u = 1
        # a0_u = Va/Lu
        self.u_w = transferFunction(num=np.array([[b0_u]]), den=np.array([[1, Va/Lu]]),Ts=Ts)
        b1_v = sigma_v*np.sqrt(3*Va/Lv)
        b0_v = sigma_v*np.sqrt(3*Va/Lv)*Va/(np.sqrt(3)*Lv)
        # a2_v = 1
        a1_v = 2*Va/Lv
        a0_v = (Va/Lv)**2
        self.v_w = transferFunction(num=np.array([[b1_v, b0_v]]), den=np.array([[1, a1_v, a0_v]]),Ts=Ts)
        b1_w = sigma_w*np.sqrt(3*Va/Lw)
        b0_w = sigma_w*np.sqrt(3*Va/Lw)*Va/(np.sqrt(3)*Lw)
        a1_w = 2*Va/Lw
        a0_w = (Va/Lw)**2
        self.w_w = transferFunction(num=np.array([[b1_w, b0_w]]), den=np.array([[1, a1_w, a0_w]]),Ts=Ts)
        self._Ts = Ts

    def update(self):
        # returns a six vector.
        #   The first three elements are the steady state wind in the inertial frame
        #   The second three elements are the gust in the body frame
        gust = np.array([[self.u_w.update(np.random.randn())],
                         [self.v_w.update(np.random.randn())],
                         [self.w_w.update(np.random.randn())]])
        #gust = np.array([[0.],[0.],[0.]])
        return np.concatenate(( self._steady_state, gust ))

