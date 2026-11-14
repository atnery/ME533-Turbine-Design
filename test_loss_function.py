import os
import subprocess
import sys
import urllib.request
from KO_losses import ko_losses

## REQUIREMENTS FOLDER HANDLING

# 1. Define the file name and optional raw URL for requirements.txt
REQ_FILE = "requirements.txt"

# 2. Install all packages into the active Python environment
print("Checking and installing required libraries...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", REQ_FILE])

## CODE HERE TO TEST YOUR FUNCTION

"""Sweep stator exit angle (alpha2) and plot Kacker-Okapuu losses.

Put this file in the same folder as ko_losses.py and run it.
All input values are plausible placeholders for a gas-turbine stator.
"""
"""Rotor inputs and an alpha2 sweep for the Kacker-Okapuu loss model.

All quantities for the rotor are RELATIVE (relative speeds, relative
total pressures, relative flow angles). Angles in degrees.

Underlying velocity triangle (mean line), used to get the numbers:
    Cx = 200 m/s, U = 250 m/s
    Stator exit absolute angle = 60 deg  -> V_theta2 = 346 m/s
    Rotor relative inlet:  W_theta = 346 - 250 = 96 m/s
        -> relative inlet angle = atan(96/200) = 25.7 deg
        -> W2 = sqrt(200^2 + 96^2) = 222 m/s
    Rotor relative exit angle = 60 deg (opposite side of axial)
        -> W3 = 200 / cos(60 deg) = 400 m/s
Both angles are entered as POSITIVE numbers (paper Fig. 3 convention,
which is what the tan(a1) + tan(a2) loading term in ko_losses expects).
Zero incidence: beta1 = alpha1 (done inside ko_losses).
"""
import numpy as np
import matplotlib.pyplot as plt

# V = [in_speed, angle_in, in_p0, in_p, out_speed, angle_out, out_p0, out_p]
V = [222.0,    # m/s   relative inlet velocity    (M_in  ~ 0.33)
     25.7,     # deg   relative inlet flow angle
     0.914e6,  # Pa    relative inlet total pressure
     0.850e6,  # Pa    inlet static pressure
     400.0,    # m/s   relative exit velocity     (M_out ~ 0.61)
     60.0,     # deg   relative exit flow angle (overwritten in sweep)
     0.889e6,  # Pa    relative exit total pressure
     0.700e6]  # Pa    exit static pressure

# G = [chord, tMax, rHub, rTip, axialChord, height, tClearance]
G = [0.040,    # m   true chord
     0.008,    # m   max thickness (t/c = 0.20)
     0.190,    # m   hub radius
     0.260,    # m   tip radius (hub/tip ~ 0.73)
     0.030,    # m   axial chord
     0.070,    # m   blade height (h/c = 1.75)
     0.0005]   # m   tip clearance (see note on k' in the reply)

# F = [aIn, aOut, rhoOut, muOut, g]
F = [680.0,    # m/s    speed of sound, inlet
     660.0,    # m/s    speed of sound, exit
     2.5,      # kg/m^3 exit density
     5.0e-5,   # Pa*s   exit viscosity
     1.33]     # gamma

# B = [type, shrouded]  (shrouded=True so the clearance term is non-zero)
B = ["rotor", True]

loss_select = [1, 2, 3, 4]


# ROUGH eyeball read-offs at s/c ~ 0.8. Replace with your own fits.
def yp_beta0_rough(a2):        # paper Fig. 1 (beta1 = 0)
    return float(np.interp(a2, [40, 50, 60, 65, 70, 75],
                               [0.025, 0.026, 0.030, 0.036, 0.042, 0.050]))


def yp_impulse_rough(a2):      # paper Fig. 2 (beta1 = alpha2)
    return float(np.interp(a2, [40, 50, 55, 60, 65, 70],
                               [0.045, 0.055, 0.065, 0.080, 0.105, 0.130]))


alpha2_vals = np.linspace(50, 70, 21)
res = {"total": [], "profile": [], "secondary": [],
       "clearance": [], "trailing": []}

for a2 in alpha2_vals:
    V_i = V.copy()
    V_i[5] = a2                        # relative exit angle

    # E = [YpReaction (beta1=0), YpImpulse (beta1=a2), fHub,
    #      dPhi2 (beta1=0), dPhi2 (beta1=a2)]
    E_i = [yp_beta0_rough(a2),
           yp_impulse_rough(a2),
           1.4,      # fHub, Fig. 6 rotor curve at hub/tip ~ 0.73
           0.02,     # Fig. 14 axial-entry nozzle value, placeholder
           0.01]     # Fig. 14 impulse value, placeholder

    Yt, Yp, Ys, Yc, Yte = ko_losses(V_i, G, F, B, E_i, loss_select)
    res["total"].append(Yt)
    res["profile"].append(Yp)
    res["secondary"].append(Ys)
    res["clearance"].append(Yc)
    res["trailing"].append(Yte)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(alpha2_vals, res["total"], "k-", lw=2.5, label="Total")
ax.plot(alpha2_vals, res["profile"], "--", label="Profile")
ax.plot(alpha2_vals, res["secondary"], "--", label="Secondary")
ax.plot(alpha2_vals, res["trailing"], "--", label="Trailing edge")
ax.plot(alpha2_vals, res["clearance"], "--", label="Tip clearance")
ax.set_xlabel(r"Rotor relative exit angle $\alpha_2$ (deg)")
ax.set_ylabel("Loss coefficient Y")
ax.set_title("Kacker-Okapuu rotor losses vs exit angle")
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()