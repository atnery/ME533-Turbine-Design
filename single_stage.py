# Library imports

import traceback
import numpy as np
from CoolProp.CoolProp import PropsSI
from KO_losses import ko_losses
from scipy.optimize import root_scalar

MAX_ITER = 100      # Max iterations for BOTH loops
TOL_P3 = 1e-4       # Relative tolerance on exit static pressure match (outer loop)
TOL_P2 = 1e-6       # Relative tolerance on stator exit pressure (inner loop)
AN2_MAX_DEFAULT = 3.2e7   # Structural limit on A*N^2 at stator inlet, stator exit AND rotor exit [m^2 rpm^2]

# Order of the values returned by single_stage_turbine (last entry is the full results dict)
OUTPUT_NAMES = [
    "T03", "P03", "eta_tt", "eta_ts", "W",
    "A2", "A3", "h2", "h3",
    "Y_s", "Y_r", "alpha2", "beta2", "beta3",
    "M2", "M2R", "M3", "M3R",
    "P2", "T2", "T3", "P02", "T02", "alpha3",
    "N_rpm",
    "h2_geom", "h3_geom",
    "r_hub2", "r_tip2",
    "r_hub3", "r_tip3",
    "A1", "AN2_1", "AN2_2", "AN2_3", "sc_op_s", "sc_opt_r", "sol"
]

# Blade geometry NOT supplied by the caller is derived from these ratios (preliminary design values)
DEFAULT_GEOM = dict(
    h_over_c=1.5,           # blade height / chord
    tmax_over_c=0.20,       # max thickness / chord
    ax_over_c=0.80,         # axial chord / chord
    clearance_over_h=0.015, # rotor tip clearance / blade height
    shrouded=True,         # rotor shroud
    s_over_c=0.80,          # pitch / chord (used by the Aungier profile-loss fits)  <-- placeholder, set yours
    te_over_o=0.05,         # trailing-edge thickness / throat opening (t/o), valid 0-0.4  <-- placeholder, set yours
)

# NOTE: ko_losses now computes fHub itself, and its 'E' input is the row work dh0 [J/kg] (used for the unshrouded
# rotor tip-clearance loss, Eq. 19). single_stage_turbine therefore has no E argument: dh0 is taken from the solution.
DEFAULT_LOSSES = [1, 2, 3, 4] # Consider all losses by default: 1 profile, 2 secondary, 3 tip clearance, 4 trailing edge


def _ko_Y(row, V_in, ang_in, V_out, ang_out, P0_in, P_in, P0_out, P_out,
          T_in, T_out, rho_out, mu_out, height, r_tip, r_hub, gamma, R, geom, dh0, loss_select):
    """
    Wrapper around ko_losses. Angles in degrees; for a rotor pass RELATIVE velocities/angles/pressures.
    Returns the ko_losses tuple (Y_total, Y_profile, Y_secondary, Y_clearance, Y_trailing_edge, Y_shock).
    """
    chord = height / geom['h_over_c']
    V = [V_in, ang_in, P0_in, P_in, V_out, ang_out, P0_out, P_out]
    G = [chord, geom['tmax_over_c'] * chord, r_hub, r_tip,
         geom['ax_over_c'] * chord, height, geom['clearance_over_h'] * height, geom['s_over_c'], geom['te_over_o']]
    F = [np.sqrt(gamma * R * T_in), np.sqrt(gamma * R * T_out), rho_out, mu_out, gamma]
    B = [row, geom['shrouded']]
    return tuple(float(y) for y in ko_losses(V, G, F, B, dh0, loss_select))

def blade_dim_func(h,r_mean):
    # Function to compute row cross sectional area given blade half height
    return (((r_mean + h)**2 - r_mean**2) + (r_mean**2 - (r_mean - h)**2))*np.pi

def sc_opt_func(angle_in, angle_out):
    # Implements the Ainley-Mathieson formula for optimal blade s/c ratio

    # NOTE: BLADE ANGLES HERE ARE DEFINED RELATIVE TO TANGENTIAL DIRECTION
    # FOR ROTORS JUST INPUT BETA 2 and BETA 3
    # FOR STATORS JUST INPUT ALPHA 1 AND ALPHA 2

    # Computes angles as properly defined for the correlation
    beta1 = 90 - abs(angle_in)
    alpha2 = 90 - abs(angle_out)

    sc_0 = 0.427 + alpha2/58 - (alpha2/93)**2
    sc_1 = 0.224 + (1.575 - alpha2/90)*(alpha2/90)
    xi = (90 - beta1)/(90 - alpha2)
    sc_opt = sc_0 + (sc_1 - sc_0)*abs(xi)*xi # Optimum sc_ratio given blade inlet and exit angles
    return sc_opt

def _evaluate(n_th, T01, P01, P3, reac, U, gamma, R, Cp1, Cp3, T3s, V1, P1, fld,
              mdot, r_mean, geom, loss_select):
    """
    One full pass of the stage for a given assumed efficiency n_th.
    Contains the INNER loop (stator exit pressure P2 with KO stator losses).
    Returns a dict of results, or None if the pass is non-physical / not converged.
    """

    g = gamma / (gamma - 1) # Compute g, common exponent for isentropic relations
    Cp_avg = (Cp1 + Cp3) / 2 # Average specific heat for the stage
    V2x = V3x = V1 # Constant axial velocity across blade rows
    W2x = V2x 
    T02 = T01 # Adiabatic stator exit total temperature (K)

    # Actual work from assumed total-static efficiency (n_th relative to isentropic drop T01 -> T3s)
    T03 = T01 - n_th * (T01 - T3s)
    W = Cp_avg * (T01 - T03)

    # Static exit temperature depends on exit swirl (Euler work) -> small fixed-point on T3
    T3 = T3s
    for _ in range(MAX_ITER): # Outer loop to converge T3 based on Euler work and reaction

        T2 = T3 + reac * W / Cp_avg                  # stator exit static T from reaction
        if T2 >= T02:
            return None
        Cp2 = PropsSI('C', 'T', T2, 'P', P1, fld) # Update stator exit specific heat

        # Stator kinematics
        V2 = (2 * Cp2 * (T02 - T2)) ** 0.5 # Total velocity at stator exit
        if V2 <= V2x:
            return None
        V2t = (V2**2 - V2x**2) ** 0.5 # Tangential velocity at stator exit
        alpha2 = np.degrees(np.arctan(V2t / V2x)) # Stator exit flow angle
        W2t = V2t - U # Relative tangential velocity at stator exit
        W2 = (W2t**2 + W2x**2) ** 0.5 # Relative velocity at stator exit
        beta2 = np.degrees(np.arctan(W2t / W2x)) # Relative flow angle at stator exit

        # Rotor exit kinematics from Euler work: W = U*(V2t - V3t)
        V3t = V2t - W / U # Tangential velocity at rotor exit
        V3 = (V3x**2 + V3t**2) ** 0.5  # Total velocity at rotor exit
        W3t = V3t - U # Relative tangential velocity at rotor exit
        W3 = (W3t**2 + V3x**2) ** 0.5  # Relative velocity at rotor exit
        beta3 = np.degrees(np.arctan(W3t / V3x)) # Relative flow angle at rotor exit
        alpha3 = np.degrees(np.arctan(V3t / V3x)) # Absolute flow angle at rotor exit
        T3_new = T03 - V3**2 / (2 * Cp3) 
        if abs(T3_new - T3)/T3 < 1e-6: # Convergence check for T3
            T3 = T3_new
            break
        T3 = T3_new
    else:
        return None
    T2 = T3 + reac * W / Cp_avg # Update T2 based on the converged T3

    sc_opt_1 = sc_opt_func(0,alpha2) # Optimum stator blade pitch / chord ratio
    geom_s = geom
    geom_s["s_over_c"] = sc_opt_1

    sc_opt_2 = sc_opt_func(beta2,beta3) # Optimum rotor blade pitch / chord ratio
    geom_r = geom
    geom_r["s_over_c"] = sc_opt_2

    # ---------------- INNER LOOP: stator exit pressure P2 with KO losses ----------------
    P2 = (P1 + P3)/2  # initial guess
    converged = False
    for _ in range(MAX_ITER): # Inner loop to converge P2 based on KO stator losses
        rho2 = PropsSI('D', 'T', T2, 'P', P2, fld) # Density at stator exit
        mu2 = PropsSI('V', 'T', T2, 'P', P2, fld) # Viscosity at stator exit
        A2 = mdot / (rho2 * V2x) # Annular flow area at stator exit

        # Stator blade sizing
        residual2 = lambda x: blade_dim_func(x,r_mean) - A2
        sol = root_scalar(residual2,bracket=(0,0.20),method='brentq') # Solves for half height of blade given A2
        h2 = 2*sol.root
        
        r_tip2 = r_mean + h2/2
        r_hub2 = r_mean - h2/2
        T1 = T01 - V1**2 / (2 * Cp1) # Static temperature at stator inlet, K

        P02_guess = P2 * (T02 / T2) ** g # Stagnation pressure at stator exit (guess)
        Y_s_all = _ko_Y('stator', V1, 0.0, V2, alpha2, P01, P1, P02_guess, P2,
                        T1, T2, rho2, mu2, h2, r_tip2, r_hub2,
                        gamma, R, geom_s, 0.0, loss_select) #  KO stator loss coefficient (dh0 unused for a stator)
        Y_s = Y_s_all[0]
        P02 = (P01 + Y_s * P2) / (1 + Y_s)           # from Y = (P01-P02)/(P02-P2)
        P2_new = P02 / ((T02 / T2) ** g)                # static P2 consistent with actual V2
        if abs(P2_new - P2) / P2 < TOL_P2:
            P2 = P2_new
            converged = True
            break
        P2 = P2_new
    if not converged:
        return None

    # Rotor: relative total conditions and KO rotor losses (relative frame)
    T02R = T2 + W2**2 / (2 * Cp2)
    P02R = P2 * (T02R / T2) ** g
    T03R = T3 + W3**2 / (2 * Cp3)
    P03R_guess = P3 * (T03R / T3) ** g

    rho3 = PropsSI('D', 'T', T3, 'P', P3, fld)
    mu3 = PropsSI('V', 'T', T3, 'P', P3, fld)
    A3 = mdot / (rho3 * V3x)

    # Rotor blade sizing
    residual3 = lambda x: blade_dim_func(x,r_mean) - A3
    sol = root_scalar(residual3,bracket=(0,0.20),method='brentq') # Solves for half height of blade given A3
    h3 = 2*sol.root

    r_tip3 = r_mean + h3/2
    r_hub3 = r_mean - h3/2

    # KO angle convention: inlet CCW+, exit CW+ -> rotor exit relative angle enters with flipped sign
    Y_r_all = _ko_Y('rotor', W2, beta2, W3, -beta3, P02R, P2, P03R_guess, P3,
                    T2, T3, rho3, mu3, h3, r_tip3, r_hub3, gamma, R, geom_r, W, loss_select)   # dh0 = rotor work = stage work W
    Y_r = Y_r_all[0]
    P03R = (P02R + Y_r * P3) / (1 + Y_r)
    P3_calc = P03R / (T03R / T3) ** g                 # exit static pressure implied by losses

    # Mach numbers (absolute and relative) at stator exit (station 2) and rotor exit (station 3)
    a2 = np.sqrt(gamma * R * T2)   # speed of sound at station 2
    a3 = np.sqrt(gamma * R * T3)   # speed of sound at station 3
    M2, M2R = V2 / a2, W2 / a2     # stator exit: absolute, relative
    M3, M3R = V3 / a3, W3 / a3     # rotor exit: absolute, relative

    return dict(T03=T03, W=W, T2=T2, T3=T3, P2=P2, P02=P02, P3_calc=P3_calc,
                A2=A2, A3=A3, h2=h2, h3=h3, Y_s=Y_s, Y_r=Y_r,
                alpha2=alpha2, beta2=beta2, beta3=beta3,
                M2=M2, M2R=M2R, M3=M3, M3R=M3R, T02=T02, alpha3=alpha3,
                P02R=P02R, T02R=T02R, P03R=P03R, T03R=T03R, Y_s_all=Y_s_all, Y_r_all=Y_r_all,sc_opt_s=sc_opt_1,sc_opt_r=sc_opt_2)

# Function to calculate the exit flow conditions of a single-stage turbine using the Kacker-Okapuu loss model

def single_stage_turbine(T01, P01, P3, mdot_core, mdot_cool, reac, r_mean, U,
                         geom=None, loss_select=None, phi=None, M_inlet = 0.25, debug=False, AN2_max=AN2_MAX_DEFAULT):
    """
    Calculate the exit flow conditions of a single-stage turbine using the Kacker-Okapuu loss model.

    OUTER loop (max 100 it): bisection on the assumed stage efficiency n_th until the exit
        static pressure implied by the KO losses matches the specified P3 (reaction is
        imposed through T2 on every pass).
    INNER loop (max 100 it): stator exit pressure P2 with KO stator losses.

    Optional: geom (dict, see DEFAULT_GEOM), loss_select,
    debug (True -> print the traceback when the KO model raises, instead of silently returning NaN).
    Axial velocity: set by M_inlet unless phi (flow coefficient Vx/U) is given.
    AN2_max: structural limit on A*N^2 [m^2 rpm^2] (default 3.2e7, None = do not enforce). It is checked at
        station 1 (stator inlet), station 2 (stator exit) and station 3 (rotor exit); exceeding it at ANY of them
        omits the design (all NaN). A1 = mdot/(rho1*Vx), A2, A3 are the annulus areas. One shaft speed N applies to
        all three, found from U at the rotor exit MEAN radius: N = 60*U/(2*pi*r_mean3), r_mean3 = (r_hub + r_tip3)/2.

    Returns a tuple in the order of OUTPUT_NAMES:
        T03, P03, eta_tt, eta_ts, W, A2, A3, h2, h3,           (exit total T/P, efficiencies, work, areas, heights)
        Y_s, Y_r, alpha2, beta2, beta3,                         (KO loss coefficients, flow angles in deg)
        M2, M2R, M3, M3R,                                       (Mach: stator exit abs/rel, rotor exit abs/rel)
        P2, T2, T3, P02, T02,                                   (static P2, T2, T3; total P02, T02)
        alpha3,                                                 (absolute flow angle at rotor exit, deg)
        N_rpm, r_mean3, A1,                                     (shaft speed [rpm], rotor exit mean radius [m], stator inlet annulus area [m^2])
        AN2_1, AN2_2, AN2_3,                                    (A*N^2 at stator inlet, stator exit, rotor exit [m^2 rpm^2])
        sol                                                     (dict with every intermediate result)
    If either loop fails to converge within 100 iterations, or AN2 > AN2_max at any station, every numeric entry is NaN and sol is {}.
    """
    nan_out = (np.nan,) * (len(OUTPUT_NAMES) - 1) + ({},)
    geom = {**DEFAULT_GEOM, **(geom or {})}
    loss_select = DEFAULT_LOSSES if loss_select is None else loss_select

    # Inlet fluid properties and definitions
    fld = 'Air'
    R = 287.05
    Cp1 = PropsSI('C', 'T', T01, 'P', P01, fld) # Specific heat at constant pressure (J/(kg*K))
    gamma = Cp1 / (Cp1 - R) # Ratio of specific heats

    mdot_total = mdot_core - mdot_cool

    # Inlet conditions
    if phi is not None:
        V1 = phi * U
        T1 = T01 - V1**2 / (2 * Cp1)
        if T1 <= 0:
            return nan_out
        M_inlet = V1 / np.sqrt(gamma * R * T1)
    T1 = T01 / (1 + (gamma - 1) / 2 * M_inlet ** 2)
    P1 = P01 / (1 + (gamma - 1) / 2 * M_inlet ** 2) ** (gamma / (gamma - 1))
    V1 = M_inlet * np.sqrt(gamma * R * T1)

    # Isentropic exit temperature
    T3s = T01 * (P3 / P01) ** ((gamma - 1) / gamma)
    Cp3 = PropsSI('C', 'T', T3s, 'P', P3, fld)

    args = (T01, P01, P3, reac, U, gamma, R, Cp1, Cp3, T3s, V1, P1, fld,
            mdot_total, r_mean, geom, loss_select) # Aruments for _evaluate

    def err(n): # Evaluate the relative error in exit static pressure for a given assumed efficiency n_th
        try:
            res = _evaluate(n, *args)
        except Exception:      # KO model raised (invalid inputs) -> treat as non-physical
            if debug:          # debug=True prints the real error instead of silently returning NaN
                traceback.print_exc()
            return None, None
        if res is None:
            return None, None
        return (res['P3_calc'] - P3) / P3, res

    # ---------------- OUTER LOOP: bisection on n_th ----------------
    lo, hi = 0.50, 0.99 # Minimum and maximum stage total - static efficiency to consider
    e_lo, _ = err(lo)
    e_hi, _ = err(hi)
    if e_lo is None or e_hi is None or e_lo * e_hi > 0:
        return nan_out

    sol = None
    for _ in range(MAX_ITER): # Outer loop to converge on the assumed stage efficiency n_th
        mid = 0.5 * (lo + hi)
        e_mid, res = err(mid)
        if e_mid is None:
            return nan_out
        if abs(e_mid) < TOL_P3:
            sol = res
            break
        if e_lo * e_mid < 0:
            hi = mid
        else:
            lo, e_lo = mid, e_mid
    if sol is None:
        return nan_out

    # ---------------- AN^2 structural constraint at stations 1, 2 and 3 ----------------
    rho1 = PropsSI('D', 'T', T1, 'P', P1, fld)       # stator inlet static density
    A1 = mdot_total / (rho1 * V1)                    # stator inlet annulus area
    A2, A3 = sol['A2'], sol['A3']                    # stator exit / rotor exit annulus areas
    h2, h3 = sol['h2'], sol['h3']                    # stator / rotor blade heights
    
    r_tip3 = r_mean + h3/2                           # rotor tip radius
    r_hub3 = r_mean - h3/2                           # rotor hub radius

    r_tip2 = r_mean + h2/2                           # stator tip radius
    r_hub2 = r_mean - h2/2                           # stator hub radius

    N_rpm = 60.0 * U / (2 * np.pi * r_mean)         # single shaft speed [rpm]
    AN2_1, AN2_2, AN2_3 = A1 * N_rpm**2, A2 * N_rpm**2, A3 * N_rpm**2   # [m^2 rpm^2]
    sol.update(A1=A1, N_rpm=N_rpm, AN2_1=AN2_1, AN2_2=AN2_2, AN2_3=AN2_3, T1=T1, P1=P1, V1=V1)
    if AN2_max is not None:
        over = [(name, val) for name, val in (("stator inlet", AN2_1), ("stator exit", AN2_2), ("rotor exit", AN2_3))
                if val > AN2_max]
        if over:
            if debug:
                print("Design omitted: AN2 limit " + f"{AN2_max:.3e}" + " exceeded at " +
                      ", ".join(f"{n} ({v:.3e})" for n, v in over))
            return nan_out

    # ---------------- Outputs ----------------
    T03, W, T3 = sol['T03'], sol['W'], sol['T3']
    Cp_avg = (Cp1 + Cp3) / 2
    P03 = P3 * (T03 / T3) ** (gamma / (gamma - 1))
    sol['P03'] = P03   # keep the exit total pressure in the results dict too

    eta_tt = W / (Cp_avg * T01 * (1 - (P03 / P01) ** ((gamma - 1) / gamma)))
    eta_ts = W / (Cp_avg * T01 * (1 - (P3 / P01) ** ((gamma - 1) / gamma)))

    return (T03, P03, eta_tt, eta_ts, W, sol['A2'], sol['A3'], sol['h2'], sol['h3'],
            sol['Y_s'], sol['Y_r'], sol['alpha2'], sol['beta2'], sol['beta3'],
            sol['M2'], sol['M2R'], sol['M3'], sol['M3R'],
            sol['P2'], sol['T2'], sol['T3'], sol['P02'], sol['T02'], sol['alpha3'],
            N_rpm, h2, h3, r_hub2, r_tip2, r_hub3, r_tip3, A1, AN2_1, AN2_2, AN2_3, sol['sc_opt_s'], sol['sc_opt_r'], sol)