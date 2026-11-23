import numpy as np

def solidity(inlet_angle,exit_angle):
    inlet_angle = 90 - inlet_angle
    exit_angle = 90 - exit_angle
    s_c_0 = 0.427 + exit_angle / 58 - (exit_angle / 93)**2
    s_c_1 = 0.224 + (1.575 - exit_angle / 90) * exit_angle / 90
    zeta = (90 - inlet_angle) / (90 - exit_angle)
    s_c_opt = s_c_0 + (s_c_1 - s_c_0) * np.abs(zeta) * zeta

    return s_c_opt