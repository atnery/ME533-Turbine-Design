"""Kacker-Okapuu turbine loss model (converted from MATLAB).

Units: angles [rad], lengths [m], velocities [m/s], pressures [Pa],
density [kg/m^3], viscosity [Pa*s].
"""
import math
import numpy as np


def ko_losses(V, G, F, B, E, loss_select):
    """Kacker-Okapuu turbine loss model.

    Each input is a list

    V  = [in_speed, angle_in, in_p0, in_p,
          out_speed, angle_out, out_p0, out_p] # Velocity and pressure conditions
    G  = [chord, tMax, rHub, rTip, axialChord, height, tClearance] # Geometry of blade
    F  = [aIn, aOut, rhoOut, muOut, g] # Flow properties
    B  = [type, shrouded]            # type: 'stator' or 'rotor'; shrouded: bool # Type of blade and whether it is shrouded
    E  = [YpReaction, YpImpulse, fHub, dPhi2Reaction, dPhi2Impulse] # Empirical coefficients for loss correlations
    loss_select = list containing any of 1, 2, 3, 4
                  (1 profile, 2 secondary, 3 tip clearance, 4 trailing edge) # What losses to consider

    Units: angles [deg], lengths [m], velocities [m/s], pressures [Pa],
    density [kg/m^3], viscosity [Pa*s].

    Returns
    -------
    (Y_total, Y_profile, Y_secondary, Y_clearance, Y_trailing_edge)

    Blade Geometry Convention:
    - Inlet angles are positive in the counter-clockwise (CCW) direction from the inlet horizontal axis
    - Exit angles are positive in the clockwise (CW) direction from the exit horizontal axis
    - Chord is defined from leading edge stagnation point to trailing edge
    - Stagger angle is defined clockwise (CW) from the inlet horizontal axis toward the leading edge stagnation point
    - For rotor analysis, PLEASE USE RELATIVE QUANTITIES (i.e., relative velocity, relative angles, etc.)

    Notes:
    - When inputting angles, please use degrees. The function will convert to radians internally.
    - Angle_in, Angle_out are the absolute angles for stators and the relative angles for rotors.

    References
    ----------
    Kacker, S. C., & Okapuu, V. (1984).

    """
    # ---- Unpack lists -------------------------------------------------
    (in_speed, angle_in, in_p0, in_p,
     out_speed, angle_out, out_p0, out_p) = V
    (chord, tMax, rHub, rTip, axialChord, height, tClearance) = G
    aIn, aOut, rhoOut, muOut, g = F
    b_type, shrouded = B
    (YpReaction, YpImpulse, fHub, dPhi2Reaction, dPhi2Impulse) = E

    # ---- Initialize outputs -------------------------------------------
    Y_profile = 0.0
    Y_secondary = 0.0
    Y_clearance = 0.0
    Y_trailing_edge = 0.0

    # ---- Validate inputs ----------------------------------------------
    if len(loss_select) == 0 or any(s not in (1, 2, 3, 4) for s in loss_select):
        raise ValueError("loss_select must contain values from 1 to 4.")
    b_type = str(b_type).lower()
    if b_type not in ("stator", "rotor"):
        raise ValueError("B type must be 'stator' or 'rotor'.")

    # ---- Mach numbers -------------------------------------------------
    M_in = in_speed / aIn
    M_out = out_speed / aOut

    # ---- Reynolds number correction -----------------------------------
    Re = rhoOut * out_speed * chord / muOut
    if Re < 2e5:
        fRe = (Re / 2e5) ** (-0.4)
    elif Re <= 1e6:
        fRe = 1.0
    else:
        fRe = (Re / 1e6) ** (-0.2)

    # ---- Mach number corrections --------------------------------------
    if M_out < 0.2:
        K1 = 1.0
    elif M_out <= 1:
        K1 = 1 - 1.25 * (M_out - 0.2)
    else:
        K1 = 0.0
    K2 = (M_in / M_out) ** 2
    Kp = 1 - K2 * (1 - K1)

    # Supersonic exit correction
    fMa = 1.0 if M_out <= 1 else 1 + 60 * (M_out - 1) ** 2

    # ---- Blade loading parameters --------------------------------------
    alpha1 = math.radians(angle_in) # Whatever angle the gas enters the row (abs. for stator, relative for rotor)
    alpha2 = math.radians(angle_out) # Whatever angle the gas exits the row (abs. for stator, relative for rotor)
    beta1 = alpha1 # Assuming zero incidence for blades
    alphaMean = math.atan(0.5 * (math.tan(alpha1) + math.tan(alpha2)))
    CL_sc = 2 * math.cos(alphaMean) * (math.tan(alpha1) + math.tan(alpha2))
    Z = CL_sc ** 2 * math.cos(alpha2) ** 2 / math.cos(alphaMean) ** 3

    # ---- 1. Profile loss ----------------------------------------------
    if 1 in loss_select:
        ratio = beta1 / alpha2
        YpPrime = ((YpReaction + ratio * abs(ratio) * (YpImpulse - YpReaction))
                   * ((tMax / chord) / 0.20) ** (ratio))

        # Shock loss
        shockTerm = 0.75 * (fHub * M_in - 0.40) ** 1.75
        if np.real(shockTerm) > 0 and shockTerm.imag == 0:
            qIn = in_p0 - in_p
            qOut = out_p0 - out_p
            if qIn <= 0 or qOut <= 0:
                raise ValueError("Invalid stagnation-to-static pressure difference.")
            Yshock = shockTerm  * (rHub / rTip) * (qIn / qOut)
        else:
            Yshock = 0.0

        Y_profile = 0.914 * ((2 / 3) * YpPrime * Kp + Yshock)
        # Reynolds and Mach corrections apply to profile loss
        Y_profile = fRe * fMa * Y_profile

    # ---- 2. Secondary loss --------------------------------------------
    if 2 in loss_select:
        K3 = (axialChord / height) ** 2
        Ks = 1 - K3 * (1 - Kp)
        AR = height / chord
        if AR < 2:
            fAR = (1 - 0.25 * math.sqrt(2 - AR)) / AR
        else:
            fAR = 1 / AR
        Y_S_AMDC = 0.0334 * fAR * Z * math.cos(alpha2) / math.cos(beta1)
        Y_secondary = 1.2 * Ks * Y_S_AMDC

    # ---- 3. Tip clearance loss ----------------------------------------
    if 3 in loss_select:
        if b_type == "stator":
            Bclearance = 0.0
        elif shrouded:
            Bclearance = 0.37
        else:
            # KO recommendation for plain-tip rotor (kept from original)
            Bclearance = 0.0
        Y_clearance = (Bclearance * Z * (chord / height)
                       * (tClearance / height) ** 0.78)

    # ---- 4. Trailing edge loss ----------------------------------------
    if 4 in loss_select:
        ratio = beta1 / alpha2
        dPhi2 = dPhi2Reaction + ratio * abs(ratio) * (dPhi2Impulse - dPhi2Reaction)
        if dPhi2 < 0 or dPhi2 >= 1:
            raise ValueError("Invalid trailing-edge empirical coefficient.")
        num = (1 - (g-1)/2 * M_out**2 * (1 / (1 - dPhi2) - 1))**(-g/(g-1)) - 1
        den = 1 - (1 + (g-1)/2 * M_out**2)**(-g/(g-1))
        Y_trailing_edge = num/den

    # ---- Total loss ---------------------------------------------------
    Y_total = Y_profile + Y_secondary + Y_clearance + Y_trailing_edge
    return Y_total, Y_profile, Y_secondary, Y_clearance, Y_trailing_edge