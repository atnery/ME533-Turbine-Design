import numpy as np
from CoolProp.CoolProp import PropsSI

phi_arr = np.arange(0.1, 0.5)
psi_arr = np.arange(0.5, 2)
R_arr = np.arange(0, 0.8)

def main():
    # Operating conditions
    fluid = "air"
    T01 = 1850 # K
    P01 = 20 * c.BAR2PA # Pa
    P0_Pe = 4.1
    mdot = 30 # kg/s

    cp = PropsSI('Cp', 'T', T01, 'P', P01, fluid)
    gamma = PropsSI('Cpmolar', 'T', T01, 'P', P01, fluid) / PropsSI('Cvmolar', 'T', T01, 'P', P01, fluid)

    dh0_is = cp * T01 * (1 - (1 / P0_Pe) ** ((gamma - 1) / gamma))

