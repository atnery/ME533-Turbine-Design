"""Kacker-Okapuu turbine loss model (converted from MATLAB).

Units: angles [deg], lengths [m], velocities [m/s], pressures [Pa],
density [kg/m^3], viscosity [Pa*s].
"""
import numpy as np

def aungier_fits(exit_angle, s_c):
    alpha2p = 90 - exit_angle

    # ---- Nozzle blade, beta1 = 0 from axial (Aungier Fig. 4-3, Eqs. 4-22 to 4-31) ----
    sc_min_n = 0.46 + alpha2p / 77.0 if alpha2p <= 30.0 else 0.614 + alpha2p / 130.0
    X_n = s_c - sc_min_n
    n_n = 1.0 + alpha2p / 30.0
    A_n = 0.025 + (27.0 - alpha2p) / 530.0 if alpha2p <= 27.0 else 0.025 + (27.0 - alpha2p) / 3085.0
    B_n = 0.1583 - alpha2p / 1640.0
    C_n = 0.08 * ((alpha2p / 30.0)**2 - 1.0)
    Yp_nozzle = A_n + B_n * X_n**2 + C_n * X_n**3 if alpha2p <= 30.0 else A_n + B_n * abs(X_n)**n_n

    # ---- Impulse blade, beta1 = alpha2 (Aungier Fig. 4-4, Eqs. 4-32 to 4-38) ----
    sc_min_i = 0.224 + 1.575 * (alpha2p / 90.0) - (alpha2p / 90.0)**2
    X_i = s_c - sc_min_i
    A_i = 0.242 - alpha2p / 151.0 + (alpha2p / 127.0)**2
    B_i = 0.3 + (30.0 - alpha2p) / 50.0 if alpha2p <= 30.0 else 0.3 + (30.0 - alpha2p) / 275.0
    C_i = 0.88 - alpha2p / 42.4 + (alpha2p / 72.8)**2
    Yp_impulse = A_i + B_i * X_i**2 - C_i * X_i**3

    return Yp_nozzle, Yp_impulse


def ko_losses(V, G, F, B, E, loss_select):
    """Kacker-Okapuu turbine loss model.

    Each input is a list

    V  = [in_speed, angle_in, in_p0, in_p,
          out_speed, angle_out, out_p0, out_p] # Velocity and pressure conditions
    G  = [chord, tMax, rHub, rTip, axialChord, height, tClearance, s_c, t_o] # Geometry of blade (tClearance = k' for shrouded, Eq. 21)
    F  = [aIn, aOut, rhoOut, muOut, g] # Flow properties
    B  = [type, shrouded]            # type: 'stator' or 'rotor'; shrouded: bool # Type of blade and whether it is shrouded
    E  = dh0 # Row work (Eq. 19, unshrouded rotors)
    loss_select = list containing any of 1, 2, 3, 4
                  (1 profile, 2 secondary, 3 tip clearance, 4 trailing edge) # What losses to consider

    Units: angles [deg], lengths [m], velocities [m/s], pressures [Pa],
    density [kg/m^3], viscosity [Pa*s].

    Returns
    -------
    (Y_total, Y_profile, Y_secondary, Y_clearance, Y_trailing_edge, Y_shock) # Y_shock is included in Y_profile

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
    (chord, tMax, rHub, rTip, axialChord, height, tClearance, s_c, t_o) = G
    aIn, aOut, rhoOut, muOut, g = F
    b_type, shrouded = B
    dh0 = E

    # ---- Initialize outputs -------------------------------------------
    Y_profile = 0.0
    Y_secondary = 0.0
    Y_clearance = 0.0
    Y_trailing_edge = 0.0
    Y_shock = 0.0

    # ---- Validate inputs ----------------------------------------------
    if len(loss_select) == 0 or any(s not in (1, 2, 3, 4) for s in loss_select):
        raise ValueError("loss_select must contain values from 1 to 4.")
    b_type = str(b_type).lower()
    if b_type not in ("stator", "rotor"):
        raise ValueError("B type must be 'stator' or 'rotor'.")

    # ---- Mach numbers -------------------------------------------------
    M_in = in_speed / aIn
    M_out = out_speed / aOut
    M_out_sub = min(M_out, 1.0) # Corrections held at M2 = 1 for supersonic exit

    # ---- Reynolds number correction -----------------------------------
    Re = rhoOut * out_speed * chord / muOut
    if Re < 2e5:
        fRe = (Re / 2e5) ** (-0.4)
    elif Re <= 1e6:
        fRe = 1.0
    else:
        fRe = (Re / 1e6) ** (-0.2)

    # ---- Mach number corrections --------------------------------------
    if M_out_sub < 0.2:
        K1 = 1.0
    else:
        K1 = 1 - 1.25 * (M_out_sub - 0.2)
    K2 = min(M_in / M_out_sub, 1.0) ** 2
    Kp = 1 - K2 * (1 - K1)

    # Supersonic exit correction
    fMa = 1.0 if M_out <= 1 else 1 + 60 * (M_out - 1) ** 2

    # ---- Blade loading parameters --------------------------------------
    alpha1 = np.radians(angle_in) # Whatever angle the gas enters the row (abs. for stator, relative for rotor)
    alpha2 = np.radians(angle_out) # Whatever angle the gas exits the row (abs. for stator, relative for rotor)
    beta1 = alpha1 # Assuming zero incidence for blades
    alphaMean = np.arctan(0.5 * (np.tan(alpha1) - np.tan(alpha2)))
    CL_sc = 2 * np.cos(alphaMean) * (np.tan(alpha1) + np.tan(alpha2))
    Z = CL_sc ** 2 * np.cos(alpha2) ** 2 / np.cos(alphaMean) ** 3

    # ---- 1. Profile loss ----------------------------------------------
    if 1 in loss_select:
        ratio = max(beta1 / alpha2, 0.0)
        
        Yp_nozzle, Yp_impulse = aungier_fits(angle_out, s_c)

        YpPrime = ((Yp_nozzle + ratio * abs(ratio) * (Yp_impulse - Yp_nozzle))
                   * ((tMax / chord) / 0.20) ** (ratio))

        # Shock loss
        rh_rt = max(rHub / rTip, 0.5) # Fig. 6 data range
        if b_type == "rotor":
            fHub = 1 + 6.0979 * max(0.9425 - rh_rt, 0.0) ** 2 # Fig. 6 fit
        else:
            fHub = 1 + 3.6491 * max(0.7723 - rh_rt, 0.0) ** 1.6809
        shockTerm = 0.75 * max(fHub * M_in - 0.40, 0.0) ** 1.75
        if shockTerm > 0:
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
        Y_shock = fRe * fMa * 0.914 * Yshock

    # ---- 2. Secondary loss --------------------------------------------
    if 2 in loss_select:
        K3 = (axialChord / height) ** 2
        Ks = 1 - K3 * (1 - Kp)
        AR = height / chord
        if AR < 2:
            fAR = (1 - 0.25 * np.sqrt(2 - AR)) / AR
        else:
            fAR = 1 / AR
        Y_S_AMDC = 0.0334 * fAR * Z * np.cos(alpha2) / np.cos(beta1)
        Y_secondary = 1.2 * Ks * Y_S_AMDC

    # ---- 3. Tip clearance loss ----------------------------------------
    if 3 in loss_select:
        if b_type == "stator":
            Bclearance = 0.0
        elif shrouded:
            Bclearance = 0.37
        else:
            Bclearance = 0.0
        Y_clearance = (Bclearance * Z * (chord / height)
                       * (tClearance / chord) ** 0.78)
        if b_type == "rotor" and not shrouded: # Eq. 19 efficiency debit -> equivalent Y
            dEta = 0.93 * tClearance / (height * np.cos(alpha2)) * rTip / (0.5 * (rHub + rTip))
            Y_clearance = out_p0 * (np.exp(dEta * dh0 * rhoOut / out_p) - 1) / (out_p0 - out_p)

    # ---- 4. Trailing edge loss ----------------------------------------
    if 4 in loss_select:
        M_te = min(M_out, 1.0) # Held at M=1 when supersonic, fMa adds the rest
        # Kacker–Okapuu Fig. 14: trailing-edge energy loss coefficient vs t/o (valid 0 <= t/o <= 0.4)
        dPhi2Reaction = 1.5702*t_o**2 - 2.5076*t_o**3 + 1.9376*t_o**4   # axial-entry nozzle (beta1 = 0)
        dPhi2Impulse  = 0.8385*t_o**2 - 1.4096*t_o**3 + 1.1127*t_o**4   # impulse blading (beta1 = alpha2)

        ratio = beta1 / alpha2
        dPhi2 = dPhi2Reaction + ratio * abs(ratio) * (dPhi2Impulse - dPhi2Reaction)
        if dPhi2 < 0 or dPhi2 >= 1:
            raise ValueError("Invalid trailing-edge empirical coefficient.")
        num = (1 - (g-1)/2 * M_te**2 * (1 / (1 - dPhi2) - 1))**(-g/(g-1)) - 1
        den = 1 - (1 + (g-1)/2 * M_te**2)**(-g/(g-1))
        Y_trailing_edge = num/den
    
    # ---- Total loss ---------------------------------------------------
    Y_total = Y_profile + Y_secondary + Y_clearance + Y_trailing_edge
    return Y_total, Y_profile, Y_secondary, Y_clearance, Y_trailing_edge, Y_shock