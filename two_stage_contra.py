import numpy as np
from CoolProp.CoolProp import PropsSI
import constants as c

def props(T, P, fluid):
    cp = PropsSI('Cpmass', 'T', T, 'P', P, fluid)
    gamma = PropsSI('Cpmass', 'T', T, 'P', P, fluid) / PropsSI('Cvmass', 'T', T, 'P', P, fluid)
    R = PropsSI('Cpmass', 'T', T, 'P', P, fluid) - PropsSI('Cvmass', 'T', T, 'P', P, fluid)
    return cp, gamma, R

def two_stage_contra(phi1, psi1, R1, R2, exit_swirl, rpm1, T01, P01, mdot, P01_Pe, fluid):
    # Velocity triangles
    beta2 = np.arctan((0.5*psi1 - R1) / phi1)
    alpha2 = np.arctan((1 + 0.5*psi1 - R1) / phi1)
    beta3 = np.arctan(-(0.5*psi1 + R1) / phi1)
    alpha3 = np.arctan((1 - 0.5*psi1 - R1) / phi1)
    alpha4 = exit_swirl
    phi2 = 2 * (1 - R2) / (np.tan(alpha3) + np.tan(alpha4))
    psi2 = phi2 * (np.tan(alpha3) - np.tan(alpha4))
    beta3_2 = np.arctan(np.tan(alpha3) - 1/phi2)
    beta4 = np.arctan(np.tan(alpha4) - 1/phi2)
    s = phi1 / phi2

    # Efficiency loop
    P_exit = P01 / P01_Pe
    h01 = PropsSI('H', 'T', T01, 'P', P01, fluid)
    s01 = PropsSI('S', 'T', T01, 'P', P01, fluid)
    h_exit = PropsSI('H', 'P', P_exit, 'S', s01, fluid)
    dh_is = h01 - h_exit

    eta_ts = 0.8    
    err = 1

    ref = None

    while err > 1e-4:
        dh = eta_ts * dh_is
        U1 = np.sqrt(dh / (psi1 + s**2 * psi2))
        U2 = s * U1

        Cx = phi1 * U1
        C2 = Cx / np.cos(alpha2)
        W2 = Cx / np.cos(beta2)
        
        C3 = Cx / np.cos(alpha3)
        W3 = Cx / np.cos(beta3)
        W3_2 = Cx / np.cos(beta3_2)

        C4 = Cx / np.cos(alpha4)
        W4 = Cx / np.cos(beta4)

        # KO losses
        Yp_s = 0.01 # Stator loss coefficient
        Yp_r1 = 0.01 # Rotor 1 loss coefficient
        Yp_r2 = 0.01 # Rotor 2 loss coefficient

        cp, gamma, R = props(*(ref or (T01, P01)), fluid)

        # Station 2 (Stator exit / rotor 1 inlet)
        h02 = h01
        h2 = h02 - C2**2 / 2
        h02r = h2 + W2**2 / 2
        
        T02 = T01
        T2 = T02 - (h02 - h2) / cp
        T02r = T2  + (h02r - h2) / cp
        a2 = np.sqrt(gamma * R * T2)
        M2 = C2 / a2
        M2r = W2 / a2
        P02_P2 = (T02 / T2)**(gamma / (gamma - 1))
        P02 = P01 / (1 + Yp_s * (1 - 1 / P02_P2))
        P2 = P02 / P02_P2
        P02r = P2 * (T02r / T2)**(gamma / (gamma - 1))

        # Station 3.1 (Rotor 1 exit)
        h03r = h02r
        h3 = h03r - W3**2 / 2
        h03 = h3 + C3**2 / 2

        T03r = T02r
        T3 = T03r - W3**2 / (2 * cp)
        T03 = T3 + C3**2 / (2 * cp)
        a3 = np.sqrt(gamma * R * T3)
        M3 = C3 / a3
        M3r = W3 / a3
        P03r_P3 = (T03r / T3)**(gamma / (gamma - 1))
        P03r = P02r / (1 + Yp_r1 * (1 - 1 / P03r_P3))
        P3 = P03r / P03r_P3
        P03 = P3 * (T03 / T3)**(gamma / (gamma - 1))

        # Station 3.2 (Rotor 2 inlet)
        h03r_2 = h3 + W3_2**2 / 2
        
        T03r_2 = T3 + W3_2**2 / (2 * cp)
        P03r_2 = P3 * (T03r_2 / T3)**(gamma / (gamma - 1))

        # Station 4 (Rotor 2 exit)
        h04r = h03r_2
        h4 = h04r - W4**2 / 2
        h04 = h4 + C4**2 / 2

        T04r = T03r_2
        T4 = T04r - W4**2 / (2 * cp)
        T04 = T4 + C4**2 / (2 * cp)
        a4 = np.sqrt(gamma * R * T4)
        M4 = C4 / a4
        M4r = W4 / a4
        P04r_P4 = (T04r / T4)**(gamma / (gamma - 1))
        P04r = P03r_2 / (1 + Yp_r2 * (1 - 1 / P04r_P4))
        P4 = P04r / P04r_P4
        P04 = P4 * (T04 / T4)**(gamma / (gamma - 1))

        ref = 0.5*(T01 + T4), 0.5*(P01 + P4)

        # Calculated work
        dh0_r1 = h02 - h03
        dh0_r2 = h03 - h04

        # New T-s efficiency
        h_exit = PropsSI('H', 'P', P4, 'S', s01, fluid)
        dh_is_calc = h01 - h_exit

        eta_ts_new  = (dh0_r1 + dh0_r2) / dh_is_calc
        err = abs(eta_ts_new - eta_ts) / eta_ts
        eta_ts = eta_ts_new


    omega1 = rpm1 * c.RPM2RAD
    r = U1 / omega1
    omega2 = U2 / r
    rpm2 = omega2 / c.RPM2RAD

