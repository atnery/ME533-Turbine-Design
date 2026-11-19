import numpy as np
from CoolProp.CoolProp import PropsSI
import constants as c

phi_arr = np.arange(0.1, 0.5)
psi_arr = np.arange(0.5, 2)
R_arr = np.arange(0, 0.8)

def main():
    # Operating conditions
    fluid = "air"
    T01 = 1850 # K
    P01 = 20 * c.BAR2PA # Pa
    P0_Pe = 4.1
    Pe = P01 / P0_Pe
    mdot = 30 # kg/s

    cp = PropsSI('Cp', 'T', T01, 'P', P01, fluid)
    gamma = PropsSI('Cpmolar', 'T', T01, 'P', P01, fluid) / PropsSI('Cvmolar', 'T', T01, 'P', P01, fluid)

    h01 = PropsSI('H', 'T', T01, 'P', P01, fluid)
    s01 = PropsSI('S', 'T', T01, 'P', P01, fluid)
    h_exit = PropsSI('H', 'P', Pe, 'S', s01, fluid)
    dh0_is = h01 - h_exit

