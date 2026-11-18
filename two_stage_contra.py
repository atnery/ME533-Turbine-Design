import numpy as np
from CoolProp.CoolProp import PropsSI
import constants as c

def two_stage_contra(phi1, psi1, R1, R2, exit_swirl, rpm1, T01, P01, mdot, P_ratio, dh0_is, fluid):
    P01_P3 = P_ratio
    P3 = P01 / P01_P3

    eta_ts = 0.8
    dh0 = eta_ts * dh0_is

    omega1 = rpm1 * c.RPM2RAD

    # Velocity triangles
    beta2 = np.arctan((0.5*psi1 + R1) / phi1)
    alpha2 = np.arctan((1 + 0.5*psi1 - R1) / phi1)
    beta3 = np.arctan(-(0.5*psi1 + R1) / phi1)
    alpha3 = np.arctan((1 - 0.5*psi1 - R1) / phi1)

    alpha4 = alpha3 # Statorless contra-rotating rotor
    alpha5 = exit_swirl
    phi2 = 2 * (1 - R2) / (np.tan(alpha4) + np.tan(alpha5))
    psi2 = phi2 * (np.tan(alpha4) - np.tan(alpha5))
    beta4 = np.arctan(np.tan(alpha4) - 1/phi2)
    beta5 = np.arctan(np.tan(alpha5) - 1/phi2)

    s = phi1 / phi2

    U1 = np.sqrt(dh0 / (psi1**2 + s**2 * phi2**2))
    U2 = s * U1

    r = U1 / omega1
    omega2 = U2 / r
    rpm2 = omega2 / c.RPM2RAD

    Cx = phi1 * U1
    C2 = Cx / np.cos(alpha2)
    W2 = Cx / np.cos(beta2)
    
    C3 = Cx / np.cos(alpha3)
    W3 = Cx / np.cos(beta3)

    C4 = C3
    W4 = Cx / np.cos(beta4)

    C5 = Cx / np.cos(alpha5)
    W5 = Cx / np.cos(beta5)

    # KO losses
    Yp_s = 0.01 # Stator loss coefficient
    Yp_r1 = 0.01 # Rotor 1 loss coefficient
    Yp_r2 = 0.01 # Rotor 2 loss coefficient

    # Thermodynamics
    # Station 2 (stator exit, rotor 1 inlet)
    cp1 = PropsSI('Cpmass', 'T', T01, 'P', P01, fluid)
    gamma1 = PropsSI('Cpmass', 'T', T01, 'P', P01, fluid) / PropsSI('Cvmass', 'T', T01, 'P', P01, fluid)
    R1 = PropsSI('Cpmass', 'T', T01, 'P', P01, fluid) - PropsSI('Cvmass', 'T', T01, 'P', P01, fluid)

    T02 = T01
    T2 = T02 - W2**2 / (2 * cp1)
    T02r = T2 + W2**2 / (2 * cp1)
    a2 = np.sqrt(gamma1 * R1 * T2)
    M2 = C2 / a2
    M2r = W2 / a2
    P02_P2 = (T02 / T2)**(gamma1 / (gamma1 - 1))
    P02 = P01 / (1 + Yp_s * (1 - 1 / P02_P2))
    P2 = P02 / P02_P2
    P02r = P2 * (T02r / T2)**(gamma1 / (gamma1 - 1))

    # Station 3 (rotor 1 exit, pseudo stator 2 inlet)
    cp2 = PropsSI('Cpmass', 'T', T2, 'P', P2, fluid)
    gamma2 = PropsSI('Cpmass', 'T', T2, 'P', P2, fluid) / PropsSI('Cvmass', 'T', T2, 'P', P2, fluid)
    R2 = PropsSI('Cpmass', 'T', T2, 'P', P2, fluid) - PropsSI('Cvmass', 'T', T2, 'P', P2, fluid)

    T03r = T02r
    T3 = T03r - W3**2 / (2 * cp2)
    T03 = T3 + C3**2 / (2 * cp2)
    a3 = np.sqrt(gamma2 * R2 * T3)
    M3 = C3 / a3
    M3r = W3 / a3
    P03r_P3 = (T03r / T3)**(gamma2 / (gamma2 - 1))
    P03r = P02r / (1 + Yp_r1 * (1 - 1 / P03r_P3))
    P3 = P03r / P03r_P3
    P03 = P3 * (T03 / T3)**(gamma2 / (gamma2 - 1))

    # Station 4 (psuedo stator 2 exit, rotor 2 inlet)
    cp3 = PropsSI('Cpmass', 'T', T3, 'P', P3, fluid)
    gamma3 = PropsSI('Cpmass', 'T', T3, 'P', P3, fluid) / PropsSI('Cvmass', 'T', T3, 'P', P3, fluid)
    R3 = PropsSI('Cpmass', 'T', T3, 'P', P3, fluid) - PropsSI('Cvmass', 'T', T3, 'P', P3, fluid)

    T04 = T03
    T4 = T3
    P4 = P3
    T04r = T4 + W4**2 / (2 * cp3)
    P04r = P4 * (T04r / T4)**(gamma3 / (gamma3 - 1))

    # Station 5 (rotor 2 exit)
    cp4 = cp3
    gamma4 = gamma3
    R4 = R3

    T05r = T04r
    T5 = T05r - W5**2 / (2 * cp4)
    T05 = T5 + C5**2 / (2 * cp4)
    a5 = np.sqrt(gamma4 * R4 * T5)
    M5 = C5 / a5
    M5r = W5 / a5
    P05r_P5 = (T05r / T5)**(gamma4 / (gamma4 - 1))
    P05r = P04r / (1 + Yp_r2 * (1 - 1 / P05r_P5))
    P5 = P05r / P05r_P5
    P05 = P5 * (T05 / T5)**(gamma4 / (gamma4 - 1))

    cp5 = PropsSI('Cpmass', 'T', T5, 'P', P5, fluid)
    gamma5 = PropsSI('Cpmass', 'T', T5, 'P', P5, fluid) / PropsSI('Cvmass', 'T', T5, 'P', P5, fluid)
    R5 = PropsSI('Cpmass', 'T', T5, 'P', P5, fluid) - PropsSI('Cvmass', 'T', T5, 'P', P5, fluid)

    # Work
    dh0_r1 = 0.5 * (cp2 + cp3) * (T03 - T02)
    dh0_r2 = 0.5 * (cp4 + cp5) * (T05 - T04)

    eta_ts  = (dh0_r1 + dh0_r2) / dh0_is


    