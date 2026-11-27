import numpy as np
from scipy.optimize import brentq

def gauging_angle(exit_angle, M_exit, P0_exit, P_exit, rho_inlet, rho_exit, Y_TE):
    exit_angle = 90 - exit_angle # Convert to tangential angle convention
    
    # Solve for gauging angle
    def deviation(Bg, M_exit):
        o_s = np.sin(np.radians(Bg))
        d0 = np.degrees(np.arcsin(o_s * (1 + (1 - o_s) * (Bg / 90)**2))) - Bg
        if M_exit <= 0.5:
            d = d0
        else:
            X = 2 * M_exit - 1
            d = d0 * (1 - 10*X**3 + 15*X**4 - 6*X**5)
        return d

    Bg = brentq(lambda Bg: deviation(Bg, M_exit) + Bg - exit_angle, 1, 89)

    # o/s correction (Aungier eqns 4-15 - 4-17, assuming constant Cx)
    o_s = np.sin(np.radians(Bg))
    F_AR = 0.8 + 0.2 * (rho_exit / rho_inlet)
    F_TE = 1 + Y_TE * (P0_exit - P_exit) / P0_exit
    o_s_id = o_s / (F_AR * F_TE)    

    Bg_blade = 90 - np.degrees(np.arcsin(o_s_id)) # Convert back to axial angle convention

    return Bg_blade, o_s_id
