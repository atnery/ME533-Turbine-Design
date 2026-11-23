import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
from CoolProp.CoolProp import PropsSI
import constants as c
from KO_losses import ko_losses
from solidity import solidity

def props(T, P, fluid):
    cp = PropsSI('Cpmass', 'T', T, 'P', P, fluid)
    gamma = cp / PropsSI('Cvmass', 'T', T, 'P', P, fluid)
    R = PropsSI('gas_constant', fluid) / PropsSI('molar_mass', fluid)
    return cp, gamma, R

def two_stage_contra(phi1, psi1, R1, R2, exit_swirl, rpm1, geom, T01, P01, mdot, P01_Pe, fluid, angles_only=False):
    # Unpack geometry
    h_c, tmax_c, ax_c, clr_h, shrouded, t_o = geom

    loss_select = [1, 2, 3, 4] # Losses to consider from KO losses

    # Dimensionless velocity triangles
    beta2 = np.arctan((0.5*psi1 - R1) / phi1)
    alpha2 = np.arctan((1 + 0.5*psi1 - R1) / phi1)
    beta3 = np.arctan(-(0.5*psi1 + R1) / phi1)
    alpha3 = np.arctan((1 - 0.5*psi1 - R1) / phi1)
    alpha4 = exit_swirl
    if np.tan(alpha3) + np.tan(alpha4) >= 0:
        raise ValueError("No usable counter-swirl from rotor 1")
    phi2 = 2 * (1 - R2) / (np.tan(alpha3) + np.tan(alpha4))
    psi2 = phi2 * (np.tan(alpha3) - np.tan(alpha4))
    if psi2 <= 0:
        raise ValueError("Rotor 2 work not positive")
    beta3_2 = np.arctan(np.tan(alpha3) - 1/phi2)
    beta4 = np.arctan(np.tan(alpha4) - 1/phi2)
    s = phi1 / phi2

    ang_in = np.array([0.0, np.rad2deg(beta2), np.rad2deg(-beta3_2)]) # KO convention, stator/r1/r2
    ang_out = np.array([np.rad2deg(alpha2), np.rad2deg(-beta3), np.rad2deg(beta4)])
    if angles_only:
        return dict(xi=ang_in / ang_out, turning=ang_in + ang_out, exit_deg=ang_out)

    # Station 1
    

    # Thermo loop
    P_exit = P01 / P01_Pe
    h01 = PropsSI('H', 'T', T01, 'P', P01, fluid)
    s01 = PropsSI('S', 'T', T01, 'P', P01, fluid)
    h_exit = PropsSI('H', 'P', P_exit, 'S', s01, fluid)
    dh_is = h01 - h_exit

    eta_ts = 0.8    
    err = 1

    ref = None
    Yp_s = Yp_r1 = Yp_r2 = 0.05

    for _ in range(100):
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

        # Gas properties
        cp, gamma, R = props(*(ref or (T01, P01)), fluid)

        # Station 1
        T1 = T01 - Cx**2 / (2 * cp)
        P1 = P01 * (T1 / T01)**(gamma / (gamma - 1))
        a1 = np.sqrt(gamma * R * T1)
        rho1 = P1 / (R * T1)

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
        rho2 = P2 / (R * T2)

        mu2 = PropsSI('V', 'T', T2, 'P', P2, fluid)

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
        rho3 = P3 / (R * T3)

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
        rho4 = P4 / (R * T4)

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

        # Annulus areas and blade heights
        A1 = mdot / (rho1 * Cx) 
        A2 = mdot / (rho2 * Cx)
        A3 = mdot / (rho3 * Cx)
        A4 = mdot / (rho4 * Cx)

        height1 = A1 / (2 * np.pi * r)
        height2 = A2 / (2 * np.pi * r)
        height3 = A3 / (2 * np.pi * r)
        height4 = A4 / (2 * np.pi * r)

        # Stator geometry
        s_c_s = solidity(0, np.rad2deg(alpha2))
        height_avg_s = 0.5 * (height1 + height2)
        chord_s = height_avg_s / h_c
        tmax_s = tmax_c * chord_s
        rhub_s = r - height_avg_s / 2
        rtip_s = r + height_avg_s / 2
        axial_chord_s = ax_c * chord_s
        tip_clr_s = clr_h * height_avg_s

        # Stator losses
        Vs  = [Cx, 0, P01, P1, C2, np.rad2deg(alpha2), P02, P2]
        Gs  = [chord_s, tmax_s, rhub_s, rtip_s, axial_chord_s, height_avg_s, tip_clr_s, s_c_s, t_o]
        Fs  = [a1, a2, rho2, mu2, gamma]
        Bs  = ['stator', shrouded]
        Es  = 0.0
        Y_s = ko_losses(Vs, Gs, Fs, Bs, Es, loss_select)
        Yp_s = Y_s[0]

        # Rotor 1 geometry
        s_c_r1 = solidity(np.rad2deg(beta2), np.rad2deg(-beta3))
        height_avg_r1 = 0.5 * (height2 + height3)
        chord_r1 = height_avg_r1 / h_c
        tmax_r1 = tmax_c * chord_r1
        rhub_r1 = r - height_avg_r1 / 2
        rtip_r1 = r + height_avg_r1 / 2
        axial_chord_r1 = ax_c * chord_r1
        tip_clr_r1 = clr_h * height_avg_r1
        mu3 = PropsSI('V', 'T', T3, 'P', P3, fluid)

        # Rotor 1 losses
        Vr1  = [W2, np.rad2deg(beta2), P02r, P2, W3, np.rad2deg(-beta3), P03r, P3]
        Gr1  = [chord_r1, tmax_r1, rhub_r1, rtip_r1, axial_chord_r1, height_avg_r1, tip_clr_r1, s_c_r1, t_o]
        Fr1  = [a2, a3, rho3, mu3, gamma]
        Br1  = ['rotor', shrouded]
        Er1  = dh0_r1
        Y_r1 = ko_losses(Vr1, Gr1, Fr1, Br1, Er1, loss_select)
        Yp_r1 = Y_r1[0]

        # Rotor 2 geometry
        s_c_r2 = solidity(np.rad2deg(-beta3_2), np.rad2deg(beta4))
        height_avg_r2 = 0.5 * (height3 + height4)
        chord_r2 = height_avg_r2 / h_c
        tmax_r2 = tmax_c * chord_r2
        rhub_r2 = r - height_avg_r2 / 2
        rtip_r2 = r + height_avg_r2 / 2
        axial_chord_r2 = ax_c * chord_r2
        tip_clr_r2 = clr_h * height_avg_r2
        mu4 = PropsSI('V', 'T', T4, 'P', P4, fluid)

        # Rotor 2 losses
        Vr2  = [W3_2, np.rad2deg(-beta3_2), P03r_2, P3, W4, np.rad2deg(beta4), P04r, P4]
        Gr2  = [chord_r2, tmax_r2, rhub_r2, rtip_r2, axial_chord_r2, height_avg_r2, tip_clr_r2, s_c_r2, t_o]
        Fr2  = [a3, a4, rho4, mu4, gamma]
        Br2  = ['rotor', shrouded]
        Er2  = dh0_r2
        Y_r2 = ko_losses(Vr2, Gr2, Fr2, Br2, Er2, loss_select)
        Yp_r2 = Y_r2[0]

        if err < 1e-6:
            break
    else:
        raise RuntimeError("Efficiency loop did not converge")

    h1 = h01 - Cx**2 / 2
    R1_static = (h2 - h3) / (h1 - h3) # Stage 1 static reaction
    dh_split = np.array([h1 - h2, h2 - h3, h3 - h4]) / (h1 - h4) # Static drop fraction: stator/r1/r2

    AN2_r1 = A3 * rpm1**2 # m^2 rpm^2
    AN2_r2 = A4 * rpm2**2
    eta_tt = (dh0_r1 + dh0_r2) / (h01 - PropsSI('H', 'P', P04, 'S', s01, fluid))

    # Report tables (SI units)
    nan = np.nan
    stations = dict(P0=[P01, P02, P03, P04], P=[P1, P2, P3, P4], T0=[T01, T02, T03, T04], T=[T1, T2, T3, T4],
                    rho=[rho1, rho2, rho3, rho4], C=[Cx, C2, C3, C4], Cx=[Cx] * 4,
                    Ctheta=[0.0, Cx * np.tan(alpha2), Cx * np.tan(alpha3), Cx * np.tan(alpha4)],
                    alpha=np.rad2deg([0.0, alpha2, alpha3, alpha4]), M=[Cx / a1, M2, M3, M4],
                    W_r1=[nan, W2, W3, nan], beta_r1=[nan, np.rad2deg(beta2), np.rad2deg(beta3), nan],
                    Mrel_r1=[nan, M2r, M3r, nan], P0rel_r1=[nan, P02r, P03r, nan], T0rel_r1=[nan, T02r, T03r, nan],
                    W_r2=[nan, nan, W3_2, W4], beta_r2=[nan, nan, np.rad2deg(beta3_2), np.rad2deg(beta4)],
                    Mrel_r2=[nan, nan, W3_2 / a3, M4r], P0rel_r2=[nan, nan, P03r_2, P04r], T0rel_r2=[nan, nan, T03r_2, T04r],
                    A=[A1, A2, A3, A4], h=[height1, height2, height3, height4])
    rows = dict(h_in=[height1, height2, height3], h_out=[height2, height3, height4],
                chord=[chord_s, chord_r1, chord_r2], ax_chord=[axial_chord_s, axial_chord_r1, axial_chord_r2],
                s_c=[s_c_s, s_c_r1, s_c_r2], tmax=[tmax_s, tmax_r1, tmax_r2], clr=[nan, tip_clr_r1, tip_clr_r2],
                ang_in=ang_in, ang_out=ang_out, xi=ang_in / ang_out, turning=ang_in + ang_out,
                M_in=[Cx / a1, M2r, W3_2 / a3], M_out=[M2, M3r, M4r], Y=[Y_s, Y_r1, Y_r2],
                dh_split=dh_split, rpm=[0.0, rpm1, rpm2], AN2=[nan, AN2_r1, AN2_r2])
    stage = dict(phi1=phi1, psi1=psi1, R1=R1, R1_static=R1_static, phi2=phi2, psi2=psi2, R2=R2, s=s,
                 U1=U1, U2=U2, r=r, dh0_r1=dh0_r1, dh0_r2=dh0_r2, eta_ts=eta_ts, eta_tt=eta_tt,
                 power=mdot * (dh0_r1 + dh0_r2), PR=P01_Pe, mdot=mdot, T01=T01, P01=P01)

    return dict(eta_ts=eta_ts, U1=U1, U2=U2, rpm2=rpm2, r=r, P4=P4, Yp_s=Yp_s, Yp_r1=Yp_r1, Yp_r2=Yp_r2,
                heights=(height1, height2, height3, height4), AN2_r1=AN2_r1, AN2_r2=AN2_r2,
                hub_tip=tuple((r - h / 2) / (r + h / 2) for h in (height1, height2, height3, height4)),
                xi=ang_in / ang_out, turning=ang_in + ang_out, exit_deg=ang_out,
                M_in=(Cx / a1, M2r, W3_2 / a3), M_out=(M2, M3r, M4r), R1_static=R1_static, dh_split=dh_split,
                report=dict(stations=stations, rows=rows, stage=stage))