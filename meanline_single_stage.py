# Meanline Single Stage Analysis

# Author: Rugved Dikay
# Date: 11/24/2026

# Description: For a given blade speed, this script analyzes the velocity triangles of a single stage turbine design 
#              and produces an output CSV file with relevant information


### Imports

import numpy as np
from single_stage import single_stage_turbine, DEFAULT_GEOM, DEFAULT_LOSSES, AN2_MAX_DEFAULT
import matplotlib.pyplot as plt

### Chosen Design Point & Boundary Conditions

# Boundaries
mdot_core = 30 # Core engine mdot, kg/s
mdot_cool = 0 # No cooling flow rate in initial analysis, kg/s
P01 = 20*(10**5) # Inlet stag. pressure, Pa
T01 = 1850 # Inlet stag. temperature, K
r_mean = 0.275 # Arbritaraily chosen value (will change this to approach more reasonable design)
PR = 4.1                                        # P01/P3 (held fixed)
AN2_LIMIT = 3.2e7                               # structural limit on A*N^2 [m^2 rpm^2], applied at stations 1, 2 and 3

# Target Speed
U_target = 450 # Target blade rotational speed, m/s

### Computations

### Computations

# Sweep ranges
reac_vec = np.linspace(0.1, 0.9, 21)
phi_vec = np.linspace(0.2, 0.8, 21)

# Generate all design point combinations
Reac, Phi = np.meshgrid(reac_vec, phi_vec)

# Number of design points
n_points = Reac.size

# Store numerical outputs (exclude final sol dictionary)
output_names = [
    "T03", "P03", "eta_tt", "eta_ts", "W",
    "A2", "A3", "h2", "h3",
    "Y_s", "Y_r", "alpha2", "beta2", "beta3",
    "M2", "M2R", "M3", "M3R",
    "P2", "T2", "T3", "P02", "T02", "alpha3",
    "N_rpm",
    "h2_geom", "h3_geom",
    "r_hub2", "r_tip2",
    "r_hub3", "r_tip3",
    "A1", "AN2_1", "AN2_2", "AN2_3"
]

# Preallocate storage
results = np.full((n_points, len(output_names)), np.nan)

# Store full solution dictionaries separately
solutions = []

# Store corresponding design inputs
design_points = np.column_stack((Reac.ravel(), Phi.ravel()))

# Run sweep
for k, (reac, phi) in enumerate(design_points):

    output = single_stage_turbine(
        T01, P01, P01/PR,
        mdot_core, mdot_cool,
        reac, r_mean, U_target,
        geom=DEFAULT_GEOM,
        loss_select=DEFAULT_LOSSES,
        phi=phi,
        debug=False,
        AN2_max=AN2_LIMIT
    )

    # Store numerical outputs
    results[k, :] = output[:-1]

    # Store full solution dictionary
    solutions.append(output[-1])

# Convert results into a dictionary for easier access
results_dict = {
    name: results[:, i]
    for i, name in enumerate(output_names)
}

# Add design inputs
results_dict["reaction"] = design_points[:, 0]
results_dict["phi"] = design_points[:, 1]

print(f"Completed {n_points} design points.")
print(f"Valid designs: {np.sum(np.isfinite(results[:, 0]))}")
print(f"Invalid designs: {np.sum(~np.isfinite(results[:, 0]))}")

### Efficiency Contour Plots

eta_tt = results_dict["eta_tt"]
eta_ts = results_dict["eta_ts"]

# Reshape results back into the meshgrid dimensions
ETA_TT = eta_tt.reshape(Reac.shape)
ETA_TS = eta_ts.reshape(Reac.shape)

fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# Total-to-total efficiency
c1 = ax[0].contourf(
    Reac, Phi, ETA_TT,
    levels=20,
    cmap="viridis"
)
fig.colorbar(c1, ax=ax[0], label="Total-to-Total Efficiency")

ax[0].set_title("Total-to-Total Efficiency")
ax[0].set_xlabel("Degree of Reaction")
ax[0].set_ylabel("Flow Coefficient ($V_x/U$)")

# Total-to-static efficiency
c2 = ax[1].contourf(
    Reac, Phi, ETA_TS,
    levels=20,
    cmap="viridis"
)
fig.colorbar(c2, ax=ax[1], label="Total-to-Static Efficiency")

ax[1].set_title("Total-to-Static Efficiency")
ax[1].set_xlabel("Degree of Reaction")
ax[1].set_ylabel("Flow Coefficient ($V_x/U$)")

plt.tight_layout()
plt.show()

valid = np.isfinite(results_dict["eta_tt"]).reshape(Reac.shape)

fig, ax = plt.subplots(figsize=(7, 5))

c = ax.contourf(
    Reac, Phi, valid.astype(int),
    levels=[-0.5, 0.5, 1.5],
    colors=["lightcoral", "lightgreen"]
)

ax.set_title("Feasible Design Space")
ax.set_xlabel("Degree of Reaction")
ax.set_ylabel("Flow Coefficient ($V_x/U$)")

plt.tight_layout()
plt.show()


import csv
from CoolProp.CoolProp import PropsSI

# ---------------------------------------------------------
# 1. Identify the highest-efficiency feasible design
# ---------------------------------------------------------

valid = np.isfinite(results_dict["eta_ts"])

if not np.any(valid):
    raise ValueError("No feasible turbine designs found.")

valid_indices = np.where(valid)[0]
best_idx = valid_indices[np.argmax(results_dict["eta_ts"][valid])]

# Recover the selected design
best_output = single_stage_turbine(
    T01, P01, P01/PR,
    mdot_core, mdot_cool,
    design_points[best_idx, 0], r_mean, U_target,
    geom=DEFAULT_GEOM,
    loss_select=DEFAULT_LOSSES,
    phi=design_points[best_idx, 1],
    debug=True,
    AN2_max=AN2_LIMIT
)

best = dict(zip(output_names, best_output[:-1]))
sol = best_output[-1]

if not sol:
    raise ValueError("Selected design could not be recomputed.")

# ---------------------------------------------------------
# 2. Extract design parameters
# ---------------------------------------------------------

reaction = design_points[best_idx, 0]
phi = design_points[best_idx, 1]

N = best["N_rpm"]
W = best["W"]

h2 = best["h2"]
h3 = best["h3"]

A1 = best["A1"]
A2 = best["A2"]
A3 = best["A3"]

# Radii (m)
rh2 = best["r_hub2"]
rt2 = best["r_tip2"]
rh3 = best["r_hub3"]
rt3 = best["r_tip3"]

# Geometry conversions
M_TO_IN = 39.3700787
M2_TO_IN2 = M_TO_IN**2

# Blade geometry
geom = DEFAULT_GEOM

chord_s = h2 / geom["h_over_c"]
chord_r = h3 / geom["h_over_c"]

# ---------------------------------------------------------
# 3. Calculate station properties
# ---------------------------------------------------------

# Static temperatures and pressures
T = [sol["T1"], sol["T2"], sol["T3"]]
P = [sol["P1"], sol["P2"], P01 / PR]

# Total temperatures and pressures
T0 = [T01, sol["T02"], best["T03"]]
P0 = [P01, sol["P02"], best["P03"]]

# Density and speed of sound
rho = [
    PropsSI("D", "T", T[i], "P", P[i], "Air")
    for i in range(3)
]

a = [
    PropsSI("A", "T", T[i], "P", P[i], "Air")
    for i in range(3)
]

# Absolute velocity
V1 = sol["V1"]
V2 = best["M2"] * a[1]
V3 = best["M3"] * a[2]

V = [V1, V2, V3]

# Axial velocity (constant through the stage)
Vx = [V1, V1, V1]

# Tangential velocity
Vtheta = [
    0.0,
    np.sqrt(max(V2**2 - V1**2, 0)),
    V3 * np.sin(np.radians(best["alpha3"]))
]

# Absolute flow angles
alpha = [0.0, best["alpha2"], best["alpha3"]]

# Mach numbers
Mach = [V[i] / a[i] for i in range(3)]

# Relative velocity and angles
Wrel = [
    None,
    best["M2R"] * a[1],
    best["M3R"] * a[2]
]

beta = [
    None,
    best["beta2"],
    best["beta3"]
]

Mrel = [None, best["M2R"], best["M3R"]]

# Relative total conditions
P0rel = [None, sol["P02R"], sol["P03R"]]
T0rel = [None, sol["T02R"], sol["T03R"]]

# Annulus areas
area = [A1, A2, A3]

# Blade heights
height = [None, h2, h3]

# Mean, hub and tip diameters
Dmean = [2*r_mean, 2*r_mean, 2*r_mean]

Dhub = [
    None,
    2*rh2,
    2*rh3
]

Dtip = [
    None,
    2*rt2,
    2*rt3
]

hub_tip = [
    None,
    rh2 / rt2,
    rh3 / rt3
]

AN2 = [best["AN2_1"], best["AN2_2"], best["AN2_3"]]

# ---------------------------------------------------------
# 4. Helper for formatting CSV values
# ---------------------------------------------------------

def fmt(x):
    if x is None:
        return ""
    if not np.isfinite(x):
        return ""
    return f"{x:.6g}"


# ---------------------------------------------------------
# 5. Create station results
# ---------------------------------------------------------

station_rows = [
    ["STATION RESULTS", "UNITS", "1", "2", "3"],

    ["P0", "PA", *[fmt(x) for x in P0]],
    ["P", "PA", *[fmt(x) for x in P]],
    ["T0", "K", *[fmt(x) for x in T0]],
    ["T", "K", *[fmt(x) for x in T]],
    ["RHO", "KG/M3", *[fmt(x) for x in rho]],
    ["C", "M/S", *[fmt(x) for x in V]],
    ["CX", "M/S", *[fmt(x) for x in Vx]],
    ["CTHETA", "M/S", *[fmt(x) for x in Vtheta]],
    ["ALPHA", "DEG", *[fmt(x) for x in alpha]],
    ["MACH", "-", *[fmt(x) for x in Mach]],
    ["W", "M/S", *[fmt(x) for x in Wrel]],
    ["BETA", "DEG", *[fmt(x) for x in beta]],
    ["MREL", "-", *[fmt(x) for x in Mrel]],
    ["P0REL", "PA", *[fmt(x) for x in P0rel]],
    ["T0REL", "K", *[fmt(x) for x in T0rel]],
    ["AREA", "IN2", *[fmt(x*M2_TO_IN2) for x in area]],
    ["HEIGHT", "IN", *[fmt(x*M_TO_IN if x is not None else None) for x in height]],
    ["D HUB", "IN", *[fmt(x*M_TO_IN if x is not None else None) for x in Dhub]],
    ["D MEAN", "IN", *[fmt(x*M_TO_IN) for x in Dmean]],
    ["D TIP", "IN", *[fmt(x*M_TO_IN if x is not None else None) for x in Dtip]],
    ["HUB/TIP", "-", *[fmt(x) for x in hub_tip]],
    ["AN2", "IN2RPM2", *[fmt(x*M2_TO_IN2) for x in AN2]],
]

# ---------------------------------------------------------
# 6. Create blade row results
# ---------------------------------------------------------

# KO loss outputs:
# [total, profile, secondary, clearance, trailing edge, shock]

Ys = sol["Y_s_all"]
Yr = sol["Y_r_all"]

# Stator and rotor values
blade_rows = [
    ["BLADE ROW RESULTS", "UNITS", "STATOR", "ROTOR"],

    ["H INLET", "IN", "", fmt(h2*M_TO_IN)],
    ["H EXIT", "IN", fmt(h2*M_TO_IN), fmt(h3*M_TO_IN)],

    ["CHORD", "IN", fmt(chord_s*M_TO_IN), fmt(chord_r*M_TO_IN)],
    ["AX CHORD", "IN",
     fmt(chord_s*geom["ax_over_c"]*M_TO_IN),
     fmt(chord_r*geom["ax_over_c"]*M_TO_IN)],

    ["PITCH", "IN",
     fmt(chord_s*geom["s_over_c"]*M_TO_IN),
     fmt(chord_r*geom["s_over_c"]*M_TO_IN)],

    ["S/C", "-", fmt(geom["s_over_c"]), fmt(geom["s_over_c"])],

    ["BLADES", "-", "", ""],

    ["TMAX", "IN",
     fmt(chord_s*geom["tmax_over_c"]*M_TO_IN),
     fmt(chord_r*geom["tmax_over_c"]*M_TO_IN)],

    ["TIP CLR", "IN", "",
     fmt(geom["clearance_over_h"]*h3*M_TO_IN)],

    ["ANG IN", "DEG", "0", fmt(best["beta2"])],
    ["ANG OUT", "DEG", fmt(best["alpha2"]), fmt(-best["beta3"])],

    ["TURNING", "DEG",
     fmt(abs(best["alpha2"])),
     fmt(abs(-best["beta3"] - best["beta2"]))],

    ["XI", "-", "", ""],

    ["M IN", "-", fmt(Mach[0]), fmt(best["M2R"])],
    ["M OUT", "-", fmt(best["M2"]), fmt(best["M3R"])],

    ["Y PROFILE", "-", fmt(Ys[1]), fmt(Yr[1])],
    ["  SHOCK", "-", fmt(Ys[5]), fmt(Yr[5])],
    ["Y SECOND", "-", fmt(Ys[2]), fmt(Yr[2])],
    ["Y CLEAR", "-", fmt(Ys[3]), fmt(Yr[3])],
    ["Y TE", "-", fmt(Ys[4]), fmt(Yr[4])],
    ["Y TOTAL", "-", fmt(Ys[0]), fmt(Yr[0])],

    ["RPM", "RPM", "0", fmt(N)],
]

# ---------------------------------------------------------
# 7. Create stage results
# ---------------------------------------------------------

mdot_stage = mdot_core - mdot_cool

U_tip = U_target * rt3 / r_mean

stage_rows = [
    ["STAGE RESULTS", "UNITS", "VALUE"],

    ["REACTION", "-", fmt(reaction)],
    ["PHI", "-", fmt(phi)],
    ["PSI", "-", fmt(W / U_target**2)],

    ["U", "M/S", fmt(U_target)],
    ["U TIP", "M/S", fmt(U_tip)],

    ["DH0", "J/KG", fmt(W)],
    ["POWER", "KW", fmt(mdot_stage*W/1000)],

    ["MDOT CORE", "KG/S", fmt(mdot_core)],
    ["MDOT COOL", "KG/S", fmt(mdot_cool)],
    ["MDOT STAGE", "KG/S", fmt(mdot_stage)],

    ["T01", "K", fmt(T01)],
    ["P01", "PA", fmt(P01)],
    ["P01/P3", "-", fmt(PR)],

    ["ETA TS", "-", fmt(best["eta_ts"])],
    ["ETA TT", "-", fmt(best["eta_tt"])],
]

# ---------------------------------------------------------
# 8. Write CSV
# ---------------------------------------------------------

filename = "best_single_stage_turbine_design.csv"

with open(filename, "w", newline="") as file:
    writer = csv.writer(file)

    writer.writerows(station_rows)
    writer.writerow([])

    writer.writerows(blade_rows)
    writer.writerow([])

    writer.writerows(stage_rows)

print(f"\nBest design exported to: {filename}")
print(f"Reaction: {reaction:.4f}")
print(f"Flow coefficient: {phi:.4f}")
print(f"Total-to-static efficiency: {best['eta_ts']:.4f}")
print(f"Total-to-total efficiency: {best['eta_tt']:.4f}")