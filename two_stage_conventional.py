import numpy as np
import CoolProp
from scipy.optimize import brentq

RPM2RAD = 2 * np.pi / 60  # rpm -> rad/s
Y_VANE = 0.06  # placeholder vane loss coefficient
Y_ROTOR = 0.10  # placeholder rotor loss coefficient


def ko_losses(V, G, F, B, E, loss_select):
    """
    Placeholder for the Kacker-Okapuu loss function. Same inputs and outputs as the real one:
    V = [in_speed, angle_in, in_p0, in_p, out_speed, angle_out, out_p0, out_p]
    G = [chord, tMax, rHub, rTip, axialChord, height, tClearance]
    F = [aIn, aOut, rhoOut, muOut, gamma]
    B = [type ('stator' or 'rotor'), shrouded]
    E = KO empirical coefficients, loss_select = loss terms to include
    Returns (Y_total, Y_profile, Y_secondary, Y_clearance, Y_trailing_edge).
    Currently returns a constant Y per row type, all booked as profile loss.
    To use the real model, delete this function and add: from KO_losses import ko_losses
    """
    Y = Y_VANE if B[0] == 'stator' else Y_ROTOR
    return Y, Y, 0.0, 0.0, 0.0



def meanline_two_stage(phi1, phi2, x1, R1, alpha_exit, U, N_rpm, T01, P01, mdot, PR_ts, fluid,
                       alpha_in, AN2_max, h_over_c, tmax_over_c, bx_over_c, k_over_h, shrouded,
                       E, loss_select, tol, max_iter):
    """
    Meanline analysis of a conventionaltwo-stage axial turbine
    All inputs are passed in from the main faunction

    Stations: 1 vane-1 inlet, 2 vane-1 exit / rotor-1 inlet, 3 rotor-1 exit / vane-2 inlet,
    4 vane-2 exit / rotor-2 inlet, 5 rotor-2 exit (turbine exit)
    Also constant blade speed for each stage

    Inputs
    phi1, phi2      flow coefficients Vx/U of stage 1 and 2
    x1              fraction of total work done by stage 1
    R1              stage-1 reaction
    alpha_exit      turbine exit absolute flow angle [deg]
    U               mean blade speed [m/s]
    N_rpm           shaft speed [rpm]; None -> sized from AN2_max
    T01, P01        turbine inlet total temperature [K] and pressure [Pa]
    mdot            mass flow [kg/s]
    PR_ts           total-to-static pressure ratio P01/P_exit
    fluid           CoolProp fluid name ('Air')
    alpha_in        vane-1 inlet flow angle [deg]
    AN2_max         rotor exit annulus area * N^2 limit [m^2 rpm^2]
    h_over_c, tmax_over_c, bx_over_c   aspect ratio, max thickness/chord, axial chord/chord (all rows)
    k_over_h        rotor tip clearance / blade height
    shrouded        rotor shroud flag passed to the KO function
    E               KO empirical coefficients [YpReaction, YpImpulse, fHub, dPhi2Reaction, dPhi2Impulse]
    loss_select     KO loss terms to include (1 profile, 2 secondary, 3 tip, 4 trailing edge)
    tol, max_iter   inner-loop tolerance and iteration limit

    Returns (same first five outputs as single_stage_turbine)
    T05             turbine exit total temperature [K]
    P05             turbine exit total pressure [Pa]
    eta_tt, eta_ts  total-to-total and total-to-static efficiency
    W               specific work dh0 [J/kg]
    A               exit annulus area of each row [vane1, rotor1, vane2, rotor2] = stations 2-5 [m^2]
    h_blade         exit blade height of each row, same order [m]
    res             dict with everything (below)
    If no solution is found, all outputs are NaN and res['converged'] is False.

    res dict:
    converged, message
    eta_tt, eta_ts, dh0, dh_is_ts, power, leaving_loss
    N_rpm, r_m, U, AN2 (rotor 1 exit, rotor 2 exit)
    psi1, psi2, phi1, phi2, R1, R2, R1_check, R2_check (reaction recomputed from static temperatures)
    stations: list of 5 dicts (T0, P0, T, P, rho, cp, gamma, V, Vx, Vt, alpha, M,
              W, Wt, beta, Mr, T0r, P0r, A, h_blade, r_hub, r_tip, hub_tip). 
    rows: list of 4 dicts (name, Y, Y_profile, Y_secondary, Y_tip, Y_te, turning, chord, Re)
    warnings: list of constraint violations
    """
    # CoolProp state for air
    AS = CoolProp.AbstractState('HEOS', fluid)
    E = list(E)
    loss_select = list(loss_select)

    # Grab fluid properties
    def props(T, P):
        AS.update(CoolProp.PT_INPUTS, P, T)
        cp, cv = AS.cpmass(), AS.cvmass()
        return cp, cp / cv, cp - cv, AS.viscosity()

    # Total-to-static pressure ratio
    def p_ratio(T0, T, g):
        return (T0 / T) ** (g / (g - 1))

    # Build full state at station
    def station(T0, P0, T, P, Vx, Vt, cp, g, R, mu, r_m):

        Wt = Vt - U  # relative swirl
        V = np.hypot(Vx, Vt)
        W = np.hypot(Vx, Wt)
        a = np.sqrt(g * R * T)

        rho = P / (R * T)
        T0r = T + W**2 / (2 * cp)  # relative total temperature
        P0r = P * p_ratio(T0r, T, g)

        A = mdot / (rho * Vx)  # annulus area from mdot eq
        hb = A / (2 * np.pi * r_m)  # blade height at constant mean radius

        return dict(T0=T0, P0=P0, T=T, P=P, rho=rho, cp=cp, gamma=g, R=R, a=a, mu=mu,
                    V=V, Vx=Vx, Vt=Vt, alpha=np.degrees(np.arctan2(Vt, Vx)), M=V / a,
                    W=W, Wt=Wt, beta=np.degrees(np.arctan2(Wt, Vx)), Mr=W / a,
                    T0r=T0r, P0r=P0r, A=A, h_blade=hb, r_hub=r_m - hb / 2, r_tip=r_m + hb / 2,
                    hub_tip=(r_m - hb / 2) / (r_m + hb / 2))


    def blade_row(kind, si, Vx_out, Vt_out, r_m):

        Wt_out = Vt_out - U
        V_out = np.hypot(Vx_out, Vt_out)
        W_out = np.hypot(Vx_out, Wt_out)

        if kind == 'vane':
            c_in, c_out = si['V'], V_out
            ang_in, ang_out = -si['alpha'], np.degrees(np.arctan2(Vt_out, Vx_out))
        else:
            c_in, c_out = si['W'], W_out
            ang_in, ang_out = si['beta'], -np.degrees(np.arctan2(Wt_out, Vx_out))
        cp, g, R, mu = props(si['T'], si['P'])  # first guess: row inlet properties
        T_out, P_out = si['T'], si['P']
        Y = 0.0

        for _ in range(max_iter):
            # Total temps conserved
            Tt = si['T'] + c_in**2 / (2 * cp)
            Pt_in = si['P'] * p_ratio(Tt, si['T'], g)
            T_new = Tt - c_out**2 / (2 * cp)  # exit static T from the velocity triangle

            if T_new <= 0:
                raise ValueError('negative static temperature')
            
            pr = p_ratio(Tt, T_new, g)  # exit total/static ratio
            Pt_out = Pt_in / (1 + Y * (1 - 1 / pr))  # from Y = (Pt_in - Pt_out) / (Pt_out - P_out)
            P_new = Pt_out / pr

            #properties at the mean static state
            cp_n, g_n, R_n, mu = props(0.5 * (si['T'] + T_new), 0.5 * (si['P'] + P_new))
            rho_out = P_new / (R_n * T_new)
            a_out = np.sqrt(g_n * R_n * T_new)

            hb_out = mdot / (rho_out * Vx_out) / (2 * np.pi * r_m)
            hb = 0.5 * (si['h_blade'] + hb_out)  # row mean blade height
            chord = hb / h_over_c

            G = [chord, tmax_over_c * chord, r_m - hb_out / 2, r_m + hb_out / 2,
                 bx_over_c * chord, hb, k_over_h * hb if kind == 'rotor' else 0.0]
            
            F = [si['a'], a_out, rho_out, mu, g_n]

            B = ['stator' if kind == 'vane' else 'rotor', shrouded]

            ko = ko_losses([c_in, ang_in, Pt_in, si['P'], c_out, ang_out, Pt_out, P_new],
                           G, F, B, E, loss_select)
            Y_new = float(ko[0])

            done = (abs(T_new - T_out) / T_out < tol and abs(P_new - P_out) / P_out < tol
                    and abs(Y_new - Y) < tol and abs(cp_n - cp) / cp < tol)
            T_out, P_out, Y, cp, g, R = T_new, P_new, Y_new, cp_n, g_n, R_n

            if done:
                break
        else:
            raise RuntimeError(kind + ' loss loop did not converge')
        
        # Absolute exit totals from the absolute velocity
        T0_out = T_out + V_out**2 / (2 * cp)
        P0_out = P_out * p_ratio(T0_out, T_out, g)
        so = station(T0_out, P0_out, T_out, P_out, Vx_out, Vt_out, cp, g, R, mu, r_m)
        info = dict(Y=Y, Y_profile=float(ko[1]), Y_secondary=float(ko[2]), Y_tip=float(ko[3]),
                    Y_te=float(ko[4]), turning=abs(ang_in + ang_out), chord=chord,
                    Re=rho_out * c_out * chord / mu, cp=cp)
        return so, info

    # One pass through all four rows for a given total work and mean radius
    def run_pass(dh0, r_m):
        psi1 = x1 * dh0 / U**2
        psi2 = (1 - x1) * dh0 / U**2
        R2 = 1 - psi2 / 2 - phi2 * np.tan(np.radians(alpha_exit))  # stage 2 reaction set by exit swirl
        Vx = [phi1 * U] * 3 + [phi2 * U] * 2

        # Swirl at stations 1-5 from the (phi, psi, R) triangle relations
        Vt = [Vx[0] * np.tan(np.radians(alpha_in)),
              U * (1 - R1 + psi1 / 2), U * (1 - R1 - psi1 / 2),
              U * (1 - R2 + psi2 / 2), U * (1 - R2 - psi2 / 2)]
        
        # Vane 1 inlet static state (properties at inlet total state)
        cp, g, R, mu = props(T01, P01)
        T1 = T01 - (Vx[0]**2 + Vt[0]**2) / (2 * cp)
        P1 = P01 / p_ratio(T01, T1, g)
        st = [station(T01, P01, T1, P1, Vx[0], Vt[0], cp, g, R, mu, r_m)]

        # Go through rows
        rows = []
        for i, kind in enumerate(['vane', 'rotor', 'vane', 'rotor']):
            so, info = blade_row(kind, st[-1], Vx[i + 1], Vt[i + 1], r_m)
            info['name'] = kind + ('1' if i < 2 else '2')
            st.append(so)
            rows.append(info)

        return dict(st=st, rows=rows, psi1=psi1, psi2=psi2, R2=R2)

    # Mean radius: fixed by N if given, otherwise iterated so AN2 at rotor 2 exit hits the limit
    def solve_geometry(dh0):
        if N_rpm is not None:
            r_m = U / (N_rpm * RPM2RAD)
            return run_pass(dh0, r_m), r_m
        rho5 = P5_target / (R_m * T5s)  # first guess from isentropic exit state
        r_m = U / RPM2RAD * np.sqrt(mdot / (rho5 * phi2 * U) / AN2_max)
        for _ in range(max_iter):
            res = run_pass(dh0, r_m)
            # AN2 = A5 * N^2 with N = U / (r_m * RPM2RAD)  ->  r_m = U / RPM2RAD * sqrt(A5 / AN2)
            r_new = U / RPM2RAD * np.sqrt(res['st'][-1]['A'] / AN2_max)
            if abs(r_new - r_m) / r_m < 1e-6:
                return res, r_new
            r_m = r_new

        raise RuntimeError('mean radius loop did not converge')

    # Isentropic total-to-static work, with cp and gamma at the mean of inlet and isentropic exit
    P5_target = P01 / PR_ts
    T5s = T01 * (1 / PR_ts) ** (0.3 / 1.3)

    # Fixed iteration for cp so that it is a little more accurate since T5s depends on gamma
    for _ in range(20):
        cp_m, g_m, R_m, _mu = props(0.5 * (T01 + T5s), 0.5 * (P01 + P5_target))
        T5s = T01 * (1 / PR_ts) ** ((g_m - 1) / g_m)
    dh_is = cp_m * (T01 - T5s)

    # find the work dh0 = eta * dh_is that gives the target exit static pressure
    # resid > 0 means too little work assumed, resid < 0 too much

    cache = {} # store results for reuse

    def resid(eta):
        try:
            res, r_m = solve_geometry(eta * dh_is)
        except Exception:
            return None
        cache[eta] = (res, r_m)
        return res['st'][-1]['P'] - P5_target

    out = dict(converged=False, message='', dh_is_ts=dh_is)
    # Scan downward from high efficiency to bracket the sign change, then refine with brentq
    etas = np.linspace(0.98, 0.50, 25)
    bracket = None
    e_prev, f_prev = None, None
    for e in etas:
        f = resid(e)
        if f is None:
            continue
        if f_prev is not None and f_prev * f < 0:
            bracket = (e, e_prev)
            break
        e_prev, f_prev = e, f
    if bracket is None:
        out['message'] = 'no solution: could not bracket exit pressure (check inputs / non-physical triangles)'
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.full(4, np.nan), np.full(4, np.nan), out

    eta_sol = brentq(lambda e: np.nan if resid(e) is None else cache[e][0]['st'][-1]['P'] - P5_target,
                     bracket[0], bracket[1], xtol=1e-9)
    res, r_m = cache[eta_sol] if eta_sol in cache else solve_geometry(eta_sol * dh_is)
    st, rows = res['st'], res['rows']
    dh0 = eta_sol * dh_is
    N = U / (r_m * RPM2RAD)
    s5 = st[4]

    k = (g_m - 1) / g_m
    out.update(converged=True, message='ok',
               eta_ts=dh0 / (cp_m * T01 * (1 - (s5['P'] / P01) ** k)),
               eta_tt=dh0 / (cp_m * T01 * (1 - (s5['P0'] / P01) ** k)),
               dh0=dh0, power=mdot * dh0, leaving_loss=s5['V']**2 / 2 / dh_is,
               N_rpm=N, r_m=r_m, U=U, AN2=[st[2]['A'] * N**2, st[4]['A'] * N**2],
               psi1=res['psi1'], psi2=res['psi2'], phi1=phi1, phi2=phi2, R1=R1, R2=res['R2'],
               R1_check=rows[1]['cp'] * (st[1]['T'] - st[2]['T']) / (x1 * dh0),
               R2_check=rows[3]['cp'] * (st[3]['T'] - st[4]['T']) / ((1 - x1) * dh0),
               stations=[{k2: v for k2, v in s.items() if k2 not in ('a', 'mu', 'R')} for s in st],
               rows=rows)

    # Constraint checks
    w = []
    if U > 500:
        w.append('U above 500 m/s')
    for i, a in enumerate(out['AN2']):
        if a > AN2_max * 1.0001:
            w.append(f'rotor {i + 1} AN2 above limit')
    for i, s in enumerate(st):
        if not 0.70 <= s['hub_tip'] <= 0.94:
            w.append(f'station {i + 1} hub/tip {s["hub_tip"]:.3f} outside AMBRA range')
    if out['R2'] < 0 or R1 < 0:
        w.append('negative mean reaction')
    for r in rows:
        if r['turning'] > 120:
            w.append(f'{r["name"]} turning {r["turning"]:.0f} deg')
    out['warnings'] = w

    A = np.array([s['A'] for s in st[1:]])
    h_blade = np.array([s['h_blade'] for s in st[1:]])
    return s5['T0'], s5['P0'], out['eta_tt'], out['eta_ts'], dh0, A, h_blade, out


def print_meanline(res):
    """Print a summary of the res dict returned by meanline_two_stage()."""
    if not res['converged']:
        print(res['message'])
        return
    print(f"eta_ts = {res['eta_ts']:.4f}   eta_tt = {res['eta_tt']:.4f}")
    print(f"dh0 = {res['dh0'] / 1e3:.1f} kJ/kg   power = {res['power'] / 1e6:.2f} MW   leaving loss = {res['leaving_loss']:.3f}")
    print(f"N = {res['N_rpm']:.0f} rpm   r_m = {res['r_m']:.4f} m   AN2 = {res['AN2'][0]:.3e}, {res['AN2'][1]:.3e}")
    print(f"psi = {res['psi1']:.3f}, {res['psi2']:.3f}   R = {res['R1']:.3f}, {res['R2']:.3f}   (check {res['R1_check']:.3f}, {res['R2_check']:.3f})")
    print()
    cols = ['T0', 'P0', 'T', 'P', 'cp', 'gamma', 'V', 'alpha', 'M', 'W', 'beta', 'Mr', 'T0r', 'h_blade', 'hub_tip']
    print('st ' + ' '.join(f'{col:>9}' for col in cols))
    for i, s in enumerate(res['stations']):
        vals = []
        for col in cols:
            v = s[col]
            if col in ('P0', 'P'):
                v = v / 1e5
            elif col == 'h_blade':
                v = v * 1e3
            vals.append(f'{v:9.4g}')
        print(f'{i + 1:<3}' + ' '.join(vals))
    print('(P in bar, h_blade in mm, angles in deg)')
    print()
    for r in res['rows']:
        print(f"{r['name']:<7} Y = {r['Y']:.4f}  (p {r['Y_profile']:.4f}, s {r['Y_secondary']:.4f}, tip {r['Y_tip']:.4f}, te {r['Y_te']:.4f})  turning = {r['turning']:.1f} deg  Re = {r['Re']:.2e}")
    if res['warnings']:
        print()
        print('Warnings:')
        for w in res['warnings']:
            print(' -', w)