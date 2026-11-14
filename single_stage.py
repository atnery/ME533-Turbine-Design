# Library imports

import numpy as np
from CoolProp.CoolProp import PropsSI
from KO_losses import ko_losses

# Function to calculate the exit flow conditions of a single-stage turbine using the Kacker-Okapuu loss model

def single_stage_turbine(T01,P01,P3,mdot_core,mdot_cool,reac,r_hub,U):
    """
    Calculate the exit flow conditions of a single-stage turbine using the Kacker-Okapuu loss model.

    Parameters:
    T01 : float
        Total temperature at the turbine inlet (K)
    P01 : float
        Total pressure at the turbine inlet (Pa)
    P3 : float
        Pressure at the turbine exit (Pa)
    mdot_core : float
        Mass flow rate of the core flow (kg/s)
    mdot_cool : float
        Mass flow rate of the cooling flow (kg/s)
    reac : float
        Reaction degree (dimensionless)
    r_hub : float
        Hub radius of the turbine (m)
    U : float
        Stage blade speed (m/s)

    Returns:
    T03 : float
        Total temperature at the turbine exit (K)
    P03 : float
        Total pressure at the turbine exit (Pa)
    eta_tt : float
        Total-to-total efficiency of the turbine stage
    eta_ts : float
        Total-to-static efficiency of the turbine stage
    W : float
        Specific work extracted by the turbine stage (J/kg)
    A2 : float
        Annular flow area at the the stator exit (m^2)
    A3 : float
        Annular flow area at the the rotor exit (m^2)
    blade_h2 : float
        Stator blade height (m)
    blade_h3 : float
        Rotor blade height (m)
    """

    # Define constants and initial conditions
    fld = 'Air'  # Working fluid
    R = 287.05   # Specific gas constant for air (J/(kg*K))
    Cp1 = PropsSI('C', 'T', T01, 'P', P01, fld)  # Specific heat at constant pressure (J/(kg*K))
    gamma = Cp1 / (Cp1 - R)  # Ratio of specific heats
    
    # Calculate total mass flow rate
    mdot_total = mdot_core - mdot_cool
    
    # Calculate inlet velocity and pressure conditions
    M_inlet = 0.15 # Inlet Mach number (assumed)
    T1 = T01 / (1 + (gamma - 1) / 2 * M_inlet ** 2)  # Static temperature at the stator inlet (K)
    P1 = P01 / (1 + (gamma - 1) / 2 * M_inlet ** 2) ** (gamma / (gamma - 1))  # Static pressure at the stator inlet (Pa)
    V1 = M_inlet * np.sqrt(gamma * R * T1)  # Inlet stator velocity (m/s)
    alpha1 = 0.0  # Inlet stator flow angle (deg)

    # Flow Velocity assumptions --> Constant axial velocity through the stage
    V2x = V1
    W2x = V2x
    V3x = V2x
    W3x = V3x

    # Degree of Reaction Calcs
    T02 = T01 # Total temperature at the adiabatic stator exit (K)
    T3 = T01 * (P01 / P3) ** ((gamma - 1) / gamma)  # Isentropic exit temperature (K)
    n_th = 0.80 # Assumed turbine stage efficiency
    Cp3 = PropsSI('C', 'T', T3, 'P', P3, fld)  # Specific heat at constant pressure at exit (J/(kg*K))
    delh_is =  Cp1*T1 - Cp3*T3  # Isentropic enthalpy drop (J/kg)
    delh = delh_is * n_th  # Actual enthalpy drop (J/kg)
    Cp_avg = (Cp1 + Cp3) / 2  # Average specific heat (J/(kg*K))
    T2 = (delh * reac + Cp3*T3)/Cp_avg  # Stator exit temperature (K)
    Cp2 = PropsSI('C', 'T', T2, 'P', P1, fld)  # Specific heat at constant pressure at stator exit (J/(kg*K))

    # Stator Analysis
    V2 = ((T02 - T2)*2*Cp2)**0.5 # Stator exit velocity (m/s)
    V2t = (V2**2 - V2x**2)**0.5 # Stator exit tangential velocity (m/s)
    alpha2 = np.arctan(V2t/V2x) * 180/np.pi # Stator exit flow angle (deg)
    W2t = V2t - U # Stator exit relative tangential velocity (m/s)
    W2 = (W2t**2 + W2x**2)**0.5 # Stator exit relative velocity (m/s)
    beta2 = np.arctan(W2t/W2x) * 180/np.pi # Stator exit relative flow angle (deg)

## WORK IN PROGRESS: The following section is incomplete and requires further development to calculate the rotor exit conditions and losses using the Kacker-Okapuu model.
    
    # Calculate outlet velocity and pressure conditions using isentropic relations
    T02s = T01 * (P02 / P01) ** ((gamma - 1) / gamma)  # Isentropic exit temperature (K)
    
    # Calculate losses using Kacker-Okapuu model
    V = [in_speed, angle_in, P01, P01, in_speed, angle_in, P02, P02]
    G = [0.1, 0.05, 0.5, 1.0, 0.2, 0.3, 0.01]  # Example geometry parameters
    F = [0.1, 0.1, PropsSI('D', 'T', T02s, 'P', P02, 'Air'), PropsSI('V', 'T', T02s, 'P', P02, 'Air'), 9.81]
    B = ['stator', False]
    E = [0.5, 0.5, 0.1, 0.1, 0.1]
    
    loss_select = [1, 2, 3, 4]