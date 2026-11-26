import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

def main():
    # Inputs
    B1, B2 = 40, 78
    Bg = 76.14
    s, c = 1.486, 1.858
    gamma = 50
    R1, dB1 = 0.17, 30
    R2, dB2 = 0.02, 4
    Gam = 15
    S2, S4 = None, None #(0.5, 1.32980), (1.0, 0.70650)
    P2, P3 = None, None #(0.5, 0.93957), (1.0, 0.52264)

    B1, B2, Bg, gamma = 90 - B1, 90 - B2, 90 - Bg, 90 - gamma # Convert to tangential angle convention
    B1, B2, Bg, gamma, dB1, dB2, Gam = np.radians([B1, B2, Bg, gamma, dB1, dB2, Gam])
    
    bz = c * np.sin(gamma) + R1 * (1 - np.sin(B1)) + R2 * (1 - np.sin(B2))
    H = c * np.cos(gamma) + R1 * np.cos(B1) - R2 * np.cos(B2)
    o = s * np.sin(Bg)

    x1 = R1 * (1 - np.sin(B1))
    y1 = H - R1 * np.cos(B1)
    x2 = bz - R2 * (1 - np.sin(B2))
    y2 = -R2 * np.cos(B2)

    P1 = R1 * (1 + np.cos(B1 + 0.5 * dB1)), H - R1 * np.sin(B1 + 0.5 * dB1), np.tan(0.5 * np.pi - B1 - 0.5 * dB1)
    P4 = bz - R2 * (1 + np.cos(B2 + 0.5 * dB2)), -R2 * np.sin(B2 + 0.5 * dB2), np.tan(B2 + 0.5 * dB2 - 0.5 * np.pi)
    S1 = R1 * (1 - np.cos(B1 - 0.5 * dB1)), H + R1 * np.sin(B1 - 0.5 * dB1), np.tan(0.5 * np.pi - B1 + 0.5 * dB1)
    S3 = P4[0] - o * np.cos(Gam + B2 - 0.5 * dB2), P4[1] + s - o * np.sin(Gam + B2 - 0.5 * dB2), np.tan(Gam + B2 - 0.5 * dB2 - 0.5 * np.pi)
    S5 = bz - R2 * (1 - np.cos(B2 - 0.5 * dB2)), R2 * np.sin(B2 - 0.5 * dB2), np.tan(B2 - 0.5 * dB2 - 0.5 * np.pi)

    suction = [S1, S2, S3, S4, S5]
    pressure = [P1, P2, P3, P4]

    suction_pts = [p for p in suction if p is not None]
    x_suction = [p[0] for p in suction_pts]
    y_suction = [p[1] for p in suction_pts]

    pressure_pts = [p for p in pressure if p is not None]
    x_pressure = [p[0] for p in pressure_pts]
    y_pressure = [p[1] for p in pressure_pts]

    cubic_S = CubicSpline(x_suction, y_suction, bc_type=((1, S1[2]), (1, S5[2])))
    cubic_P = CubicSpline(x_pressure, y_pressure, bc_type=((1, P1[2]), (1, P4[2])))
    
    x_S = np.linspace(S1[0], S5[0], 100)
    y_S = cubic_S(x_S)
    x_P = np.linspace(P1[0], P4[0], 100)
    y_P = cubic_P(x_P)

    phi_LE = np.linspace(B1 + 0.5 * dB1 - np.pi, B1 - 0.5 * dB1, 50)
    x_LE = R1 * (1 - np.cos(phi_LE))
    y_LE = H + R1 * np.sin(phi_LE)

    phi_TE = np.linspace(B2 + 0.5 * dB2 - np.pi, B2 - 0.5 * dB2, 50)
    x_TE = bz - R2 * (1 - np.cos(phi_TE))
    y_TE = R2 * np.sin(phi_TE)

    plt.plot(x_S, y_S, label="Suction")
    plt.plot(x_P, y_P, label="Pressure")
    plt.plot(x_LE, y_LE, label="LE")
    plt.plot(x_TE, y_TE, label="TE")
    plt.axis("equal")
    plt.grid()
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()

