def efficiency(T01, T03, T03ss, T3ss, Cp1, Cp3):

    """
    Calculate the efficiencies of a turbine stage.

    Parameters:
    T01 : float
        Total temperature at the inlet (K).
    T03 : float
        Total temperature at the exit (K).
    T03ss : float
        Total temperature at the exit for a single-stage turbine (K).
    T3ss : float
        Static temperature at the exit for a single-stage turbine (K).
    Cp1 : float
        Specific heat capacity at constant pressure for the inlet (J/(kg*K)).
    Cp3 : float
        Specific heat capacity at constant pressure for the exit (J/(kg*K)).

    Returns:
    eta_ts : float
        Total to static efficiency of the turbine stage
    eta_tt : float
        Total to total efficiency of the turbine stage
    """

    # Calculate efficiencies
    eta_ts = (Cp1*T01 - Cp3*T03) / (Cp1*T01 - Cp3*T3ss)
    eta_tt = (Cp1*T01 - Cp3*T03) / (Cp1*T01 - Cp3*T03ss)
  
    return eta_ts, eta_tt