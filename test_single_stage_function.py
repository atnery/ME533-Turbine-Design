"""
Parametric test + plots for single_stage_turbine().
Sweeps: degree of reaction, blade speed U, flow coefficient phi (= Vx/U); pressure ratio held fixed.
Each sweep produces five figure grids:
    1) performance  : eta_tt, eta_ts, psi, T03, h2, h3
    2) Mach numbers : absolute and relative, stator exit (2) and rotor exit (3)
    3) thermo state : P2, P02, P03 (top row) and T2, T3, T02, T03 (bottom row)
    4) angles/losses: absolute flow angles alpha2, alpha3 and KO loss coefficients Y_s, Y_r
    5) structural   : N [rpm], r_mean3, annulus areas A1/A2/A3 and AN^2 at stator inlet (1), stator exit (2), rotor exit (3)
Designs with AN^2 above AN2_LIMIT at ANY of the three stations, mean blade speed above U_MAX or hub/tip outside HUB_TIP
are omitted (NaN) from every plot when ENFORCE_AN2 = True;
the AN^2 panels still show the raw values (red dashed = limit) so you can see where each station crosses it.
Non-converged cases come back as NaN and show up as gaps (line plots) or grey cells (maps).
Two more analyses at the end of the script:
    6) peak eta_ts   : for each blade speed U in PEAK_U_LIST, search reaction x phi for the highest total-to-static
                       efficiency (AN^2-feasible designs only) and report the optimum reaction, phi, psi and Y_s, Y_r
                       at that point (printed table + figure)
    7) Smith diagram : scatter of every tested design in the (phi, psi) plane, coloured by eta_ts / eta_tt / reaction,
                       with the peak-eta_ts point of each U marked
Run:  python test_turbine.py   (needs single_stage.py and KO_losses.py in the same folder)
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from single_stage import single_stage_turbine, OUTPUT_NAMES, DEFAULT_GEOM

SAVE_FIGS = False   # set True to also write every figure to a PNG

# ------------------------- Baseline inputs -------------------------
BASE = dict(T01=1850.0, P01=20*(10**5), mdot_core=30.0, mdot_cool=0.0, reac=0.6, r_hub=0.30, U=450.0, phi=0.5) # Baseline design inputs
PR = 4.1                                        # P01/P3 (held fixed)
AN2_LIMIT = 3.2e7                               # structural limit on A*N^2 [m^2 rpm^2], applied at stations 1, 2 and 3
RAW_KEYS = ["AN2_1_raw", "AN2_2_raw", "AN2_3_raw"]   # unmasked AN^2 values kept for the AN^2 panels
ENFORCE_AN2 = True                              # True -> omit designs outside any limit (AN^2, blade speed, hub/tip)
U_MAX = 500.0                                   # mean blade speed limit [m/s]
HUB_TIP = (0.7, 0.94)                           # hub/tip ratio range, applied at stations 1, 2 and 3
SCALARS = [n for n in OUTPUT_NAMES if n != "sol"]

# Preflight: run the baseline once with debug=True so a failing KO call prints its real error
_p = BASE
_pre = single_stage_turbine(_p["T01"], _p["P01"], _p["P01"] / PR, _p["mdot_core"], _p["mdot_cool"],
                            _p["reac"], _p["r_hub"], _p["U"], phi=_p["phi"], debug=True, AN2_max=None)
if np.isnan(_pre[0]):
    print("WARNING: baseline case did not converge (NaN) - check the traceback above / the baseline inputs.")


def run(**over):
    """Run one case. Returns dict of all scalar outputs (NaN if not converged). Adds psi = W/U^2."""
    p = {**BASE, **over}
    # Call with AN2_max=None to get the raw design, then apply the AN^2 constraint here so the violators can be shown
    out = single_stage_turbine(p["T01"], p["P01"], p["P01"] / PR, p["mdot_core"], p["mdot_cool"],
                               p["reac"], p["r_hub"], p["U"], phi=p["phi"], AN2_max=None)
    r = dict(zip(OUTPUT_NAMES, out))
    r.pop("sol")
    r["psi"] = r["W"] / p["U"] ** 2
    for i in (1, 2, 3):
        r[f"AN2_{i}_raw"] = r[f"AN2_{i}"]                       # unmasked, for the AN^2 panels
    an2_max = np.nanmax([r["AN2_1"], r["AN2_2"], r["AN2_3"]]) if not np.isnan(r["AN2_3"]) else np.nan
    r_tip = np.sqrt(np.array([r["A1"], r["A2"], r["A3"]]) / np.pi + p["r_hub"]**2)
    r["HT_min"], r["HT_max"] = p["r_hub"] / r_tip.max(), p["r_hub"] / r_tip.min()
    outside = an2_max > AN2_LIMIT or p["U"] > U_MAX or not (HUB_TIP[0] <= r["HT_min"] and r["HT_max"] <= HUB_TIP[1])
    # status: 0 = no solution, 1 = feasible, 2 = converged but outside a limit (AN^2, blade speed, hub/tip)
    r["status"] = 0 if np.isnan(an2_max) else (2 if outside else 1)
    if ENFORCE_AN2 and r["status"] == 2:
        for k in r:
            if k not in RAW_KEYS and k != "status":
                r[k] = np.nan                                   # omit the design
    return r


def sweep(xname, xs, **fixed):
    """Sweep one input over xs (other inputs from BASE, overridden by `fixed`). Returns {output: array}."""
    res = {k: np.full(len(xs), np.nan) for k in SCALARS + ["psi"] + RAW_KEYS}
    for i, x in enumerate(xs):
        r = run(**{**fixed, xname: x})
        for k in res:
            res[k][i] = r[k]
    return res


# ------------------------- Panel definitions: (key, title, scale) -------------------------
PANELS_PERF = [("eta_tt", "Total-to-total efficiency", 1),
               ("eta_ts", "Total-to-static efficiency", 1),
               ("psi", "Loading coefficient  psi = W/U^2", 1),
               ("T03", "Exit total temperature T03 [K]", 1),
               ("h2", "Stator exit blade height h2 [mm]", 1e3),
               ("h3", "Rotor exit blade height h3 [mm]", 1e3)]

PANELS_MACH = [("M2",  "Stator exit  -  absolute Mach  M2", 1),
               ("M2R", "Stator exit  -  relative Mach  M2R", 1),
               ("M3",  "Rotor exit  -  absolute Mach  M3", 1),
               ("M3R", "Rotor exit  -  relative Mach  M3R", 1)]

PANELS_STATE = [("P2",  "Stator exit static pressure P2 [bar]", 1e-5),
                ("P02", "Stator exit total pressure P02 [bar]", 1e-5),
                ("P03", "Rotor exit total pressure P03 [bar]", 1e-5),
                None,                                                   # blank cell
                ("T2",  "Stator exit static temperature T2 [K]", 1),
                ("T3",  "Rotor exit static temperature T3 [K]", 1),
                ("T02", "Stator exit total temperature T02 [K]", 1),
                ("T03", "Rotor exit total temperature T03 [K]", 1)]

PANELS_ANGLES_LOSS = [("alpha2", "Stator exit absolute flow angle alpha2 [deg]", 1),
                      ("alpha3", "Rotor exit absolute flow angle alpha3 [deg]", 1),
                      ("Y_s", "Stator KO loss coefficient Y_s", 1),
                      ("Y_r", "Rotor KO loss coefficient Y_r", 1)]

PANELS_STRUCT = [("N_rpm", "Shaft speed N [rpm]", 1),
                 ("A1", "Stator inlet annulus area A1 [m^2]", 1),
                 ("A2", "Stator exit annulus area A2 [m^2]", 1),
                 ("A3", "Rotor exit annulus area A3 [m^2]", 1),
                 ("r_mean3", "Rotor exit mean radius [m]", 1),
                 ("AN2_1_raw", "AN^2 stator inlet [1e7 m^2 rpm^2]  (dashed = limit)", 1e-7, AN2_LIMIT * 1e-7),
                 ("AN2_2_raw", "AN^2 stator exit [1e7 m^2 rpm^2]  (dashed = limit)", 1e-7, AN2_LIMIT * 1e-7),
                 ("AN2_3_raw", "AN^2 rotor exit [1e7 m^2 rpm^2]  (dashed = limit)", 1e-7, AN2_LIMIT * 1e-7)]


def grid_plot(data, xs, xlabel, fam_label, panels, nrows, ncols, title, figsize, fname, mach=False):
    """One figure grid. data = {family value: sweep result}; one curve per family value in every panel."""
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    panels = list(panels) + [None] * (nrows * ncols - len(panels))
    first = None
    for a, panel in zip(ax.flat, panels):
        if panel is None:
            a.axis("off")
            continue
        key, t, scale, *extra = panel               # optional 4th item = horizontal limit line
        ymax = np.nan
        for fv, res in data.items():
            a.plot(xs, res[key] * scale, "o-", ms=3, label=fam_label(fv))
            if not np.all(np.isnan(res[key])):
                ymax = np.nanmax([ymax, np.nanmax(res[key] * scale)])
        if extra:
            a.axhline(extra[0], color="r", ls="--", lw=1.0)
        if mach and ymax > 0.8:                         # sonic reference line when it is relevant
            a.axhline(1.0, color="k", ls="--", lw=0.8)
        a.set_title(t, fontsize=10); a.set_xlabel(xlabel); a.grid(alpha=0.3)
        first = first or a
    first.legend()
    fig.suptitle(title)
    fig.tight_layout()
    if SAVE_FIGS:
        fig.savefig(fname, dpi=150)


# ------------------------- Sweep definitions -------------------------
SWEEPS = [
    dict(tag="reaction", xname="reac", xs=np.linspace(0.1, 0.9, 25), xlabel="Degree of reaction",
         fam_name="phi", fam_vals=[0.3, 0.5, 0.7], fam_label=lambda v: f"phi = {v}",
         note=f"U = {BASE['U']:.0f} m/s"),
    dict(tag="blade_speed", xname="U", xs=np.linspace(150, 500, 25), xlabel="Blade speed U [m/s]",
         fam_name="reac", fam_vals=[0.3, 0.5, 0.7], fam_label=lambda v: f"reaction = {v}",
         note=f"phi = {BASE['phi']} fixed, so Vx scales with U"),
    dict(tag="flow_coefficient", xname="phi", xs=np.linspace(0.2, 1.0, 25), xlabel="Flow coefficient phi = Vx/U",
         fam_name="reac", fam_vals=[0.3, 0.5, 0.7], fam_label=lambda v: f"reaction = {v}",
         note=f"U = {BASE['U']:.0f} m/s"),
]

for sw in SWEEPS:
    # Run the sweep once per family value, then reuse the data for all three figure grids
    data = {fv: sweep(sw["xname"], sw["xs"], **{sw["fam_name"]: fv}) for fv in sw["fam_vals"]}
    head = f"{sw['tag'].replace('_', ' ').title()} sweep ({sw['note']}; PR = {PR}, T01 = {BASE['T01']:.0f} K)"
    common = dict(data=data, xs=sw["xs"], xlabel=sw["xlabel"], fam_label=sw["fam_label"])

    grid_plot(panels=PANELS_PERF, nrows=2, ncols=3, figsize=(15, 8), title=head,
              fname=f"sweep_{sw['tag']}.png", **common)
    grid_plot(panels=PANELS_MACH, nrows=2, ncols=2, figsize=(11, 8), title=head + " - Mach numbers",
              fname=f"sweep_{sw['tag']}_mach.png", mach=True, **common)
    grid_plot(panels=PANELS_STATE, nrows=2, ncols=4, figsize=(17, 8), title=head + " - pressures and temperatures",
              fname=f"sweep_{sw['tag']}_state.png", **common)
    grid_plot(panels=PANELS_ANGLES_LOSS, nrows=2, ncols=2, figsize=(11, 8), title=head + " - flow angles and loss coefficients",
              fname=f"sweep_{sw['tag']}_angles_loss.png", **common)
    grid_plot(panels=PANELS_STRUCT, nrows=2, ncols=4, figsize=(17, 8), title=head + " - shaft speed, areas and AN^2 at stations 1, 2, 3",
              fname=f"sweep_{sw['tag']}_structural.png", **common)

# ------------------------- Map: reaction vs flow coefficient -------------------------
from matplotlib.colors import ListedColormap
reg = np.linspace(0.1, 0.9, 17)
phg = np.linspace(0.2, 1.0, 17)
eta = np.full((len(phg), len(reg)), np.nan)          # total-to-static
eta_tt = np.full_like(eta, np.nan)                    # total-to-total
psi = np.full_like(eta, np.nan)
status = np.zeros_like(eta)
an2_max = np.full_like(eta, np.nan)          # highest AN^2 of the three stations for every converged case (raw, unmasked)
for i, ph in enumerate(phg):
    for j, rc in enumerate(reg):
        r = run(reac=rc, phi=ph)
        eta[i, j], eta_tt[i, j], psi[i, j], status[i, j] = r["eta_ts"], r["eta_tt"], r["psi"], r["status"]
        raw = [r[k] for k in RAW_KEYS]
        if not np.all(np.isnan(raw)):
            an2_max[i, j] = np.nanmax(raw)

fig, ax = plt.subplots(2, 3, figsize=(20, 10))
cmap = plt.get_cmap("viridis").copy(); cmap.set_bad("lightgrey")     # NaN -> grey
# Row 1: total-to-total efficiency, total-to-static efficiency, loading coefficient
for a, data_, lab in zip(ax[0], [eta_tt, eta, psi],
                         ["Total-to-total efficiency", "Total-to-static efficiency", "Loading coefficient psi"]):
    im = a.pcolormesh(reg, phg, np.ma.masked_invalid(data_), cmap=cmap, shading="nearest")
    fig.colorbar(im, ax=a)
    a.set_title(f"{lab}  (grey = omitted / no solution)")
    a.set_xlabel("Degree of reaction"); a.set_ylabel("Flow coefficient phi")

# Row 2: status map, highest AN^2 over stations 1, 2, 3, (blank)
ax[1, 0].pcolormesh(reg, phg, status, cmap=ListedColormap(["#d9534f", "#5cb85c", "#f0ad4e"]), vmin=-0.5, vmax=2.5, shading="nearest")
ax[1, 0].set_title("Red = no solution, green = feasible, orange = outside limits")

# Highest AN^2 over stations 1, 2, 3 (actual values, including the designs that violate the limit)
cmap_an = plt.get_cmap("plasma").copy(); cmap_an.set_bad("lightgrey")
an2_plot = np.ma.masked_invalid(an2_max) * 1e-7
im = ax[1, 1].pcolormesh(reg, phg, an2_plot, cmap=cmap_an, shading="nearest")
fig.colorbar(im, ax=ax[1, 1], label="max AN^2 [1e7 m^2 rpm^2]")
lim7 = AN2_LIMIT * 1e-7
if np.nanmin(an2_max) * 1e-7 < lim7 < np.nanmax(an2_max) * 1e-7:      # draw the limit contour when the map crosses it
    cs = ax[1, 1].contour(reg, phg, an2_plot, levels=[lim7], colors="k", linestyles="--", linewidths=1.5)
    ax[1, 1].clabel(cs, fmt=f"limit = {lim7:.1f}")
ax[1, 1].set_title("Highest AN^2 of stations 1, 2, 3 (grey = no solution)")

for a in ax[1, :2]:
    a.set_xlabel("Degree of reaction"); a.set_ylabel("Flow coefficient phi")
ax[1, 2].axis("off")
fig.tight_layout()
if SAVE_FIGS:
    fig.savefig("map_reaction_vs_phi.png", dpi=150)

n_all = status.size
print(f"Limits: AN^2 <= {AN2_LIMIT:.1e} and hub/tip {HUB_TIP[0]}-{HUB_TIP[1]} at stations 1, 2, 3; mean blade speed <= {U_MAX:.0f} m/s")
print(f"Map ({n_all} cases): {int(np.sum(status == 1))} feasible, "
      f"{int(np.sum(status == 2))} outside limits, {int(np.sum(status == 0))} no solution")
print(f"Peak eta_tt on the map: {np.nanmax(eta_tt):.3f}   Peak eta_ts: {np.nanmax(eta):.3f}")
print(f"Highest AN^2 on the map: {np.nanmax(an2_max):.3e}  (limit {AN2_LIMIT:.1e})")

# ======================= Peak total-to-static efficiency vs blade speed =======================
PEAK_U_LIST = np.linspace(300, 600, 7)          # blade speeds U [m/s] to search
PEAK_REAC = np.linspace(0.1, 0.9, 13)           # reaction search range (coarse grid)
PEAK_PHI = np.linspace(0.2, 1.0, 13)            # flow-coefficient search range (coarse grid)
REFINE_PEAK = True                              # True -> refine each coarse peak on a finer local grid
REFINE_N = 9                                    # points per direction in the refinement grid
PEAK_KEYS = ("eta_ts", "eta_tt", "psi", "Y_s", "Y_r")


def eval_points(U, reacs, phis):
    """Run every (reaction, phi) pair at blade speed U. Keeps only converged designs within all limits."""
    pts = []
    for ph in phis:
        for rc in reacs:
            r = run(U=U, reac=rc, phi=ph)
            if not np.isnan(r["eta_ts"]):
                pts.append(dict(U=U, reac=rc, phi=ph, **{k: r[k] for k in PEAK_KEYS}))
    return pts


def refine_grid(best):
    """Finer grid of +/- one coarse step around the coarse peak, clipped to the search range."""
    dr, dp = PEAK_REAC[1] - PEAK_REAC[0], PEAK_PHI[1] - PEAK_PHI[0]
    reacs = np.linspace(max(best["reac"] - dr, PEAK_REAC[0]), min(best["reac"] + dr, PEAK_REAC[-1]), REFINE_N)
    phis = np.linspace(max(best["phi"] - dp, PEAK_PHI[0]), min(best["phi"] + dp, PEAK_PHI[-1]), REFINE_N)
    return reacs, phis


def on_edge(p):
    """True if the peak sits on the boundary of the search range (the true optimum may lie outside it)."""
    return bool(np.isclose(p["reac"], [PEAK_REAC[0], PEAK_REAC[-1]]).any()
                or np.isclose(p["phi"], [PEAK_PHI[0], PEAK_PHI[-1]]).any())


smith_pts = []      # every converged, feasible design tested above (feeds the Smith diagram)
peaks = []          # best (highest eta_ts) design for each U
for U_i in PEAK_U_LIST:
    pts = eval_points(U_i, PEAK_REAC, PEAK_PHI)
    if pts and REFINE_PEAK:
        pts += eval_points(U_i, *refine_grid(max(pts, key=lambda p: p["eta_ts"])))
    smith_pts += pts
    if pts:
        peaks.append(max(pts, key=lambda p: p["eta_ts"]))
    else:                                                       # nothing converged / feasible at this U
        peaks.append(dict(U=U_i, reac=np.nan, phi=np.nan, **{k: np.nan for k in PEAK_KEYS}))

print(f"\nPeak total-to-static efficiency for each blade speed "
      f"(reaction {PEAK_REAC[0]:.2f}-{PEAK_REAC[-1]:.2f}, phi {PEAK_PHI[0]:.2f}-{PEAK_PHI[-1]:.2f}, limits "
      f"{'enforced' if ENFORCE_AN2 else 'not enforced'})")
print(f"{'U [m/s]':>8} {'reaction':>9} {'phi':>6} {'psi':>6} {'eta_ts':>7} {'eta_tt':>7} {'Y_s':>7} {'Y_r':>7}")
for p in peaks:
    flag = " *" if (not np.isnan(p["eta_ts"]) and on_edge(p)) else ""
    print(f"{p['U']:8.0f} {p['reac']:9.3f} {p['phi']:6.3f} {p['psi']:6.3f} {p['eta_ts']:7.4f} "
          f"{p['eta_tt']:7.4f} {p['Y_s']:7.4f} {p['Y_r']:7.4f}{flag}")
if any((not np.isnan(p["eta_ts"])) and on_edge(p) for p in peaks):
    print("* = peak sits on the edge of the search range -> widen PEAK_REAC / PEAK_PHI")

# ======================= Optimum design CSV =======================
M2IN = 1 / 0.0254   # m -> in


def ff(x, d):
    """Fixed-point string with d decimals, blank for NaN."""
    if x is None or not np.isfinite(x):
        return ""
    return f"{0.0 if round(x, d) == 0 else x:.{d}f}"


def write_optimum_csv(best, fname="single_stage_optimum.csv"):
    """Re-run the peak eta_ts design and write station, blade row and stage tables (dimensions in inches)."""
    p = {**BASE, "U": best["U"], "reac": best["reac"], "phi": best["phi"]}
    out = single_stage_turbine(p["T01"], p["P01"], p["P01"] / PR, p["mdot_core"], p["mdot_cool"],
                               p["reac"], p["r_hub"], p["U"], phi=p["phi"], AN2_max=None)
    r = dict(zip(OUTPUT_NAMES, out))
    s, gm, nan = r["sol"], DEFAULT_GEOM, np.nan
    mdot = p["mdot_core"] - p["mdot_cool"]                 # stage flow, as in single_stage_turbine
    U, Vx, N, P3 = p["U"], s["V1"], r["N_rpm"], p["P01"] / PR

    # Stations 1, 2, 3 (constant hub radius)
    A = np.array([r["A1"], r["A2"], r["A3"]])
    rh = p["r_hub"]
    rt = np.sqrt(A / np.pi + rh**2)
    h, rm = rt - rh, (rh + rt) / 2
    alpha = np.array([0.0, r["alpha2"], r["alpha3"]])
    C = Vx / np.cos(np.radians(alpha))
    T = np.array([s["T1"], r["T2"], r["T3"]])
    a = C[1] / r["M2"] * np.sqrt(T / T[1])                 # speed of sound, single gamma as in the model
    Wrel = Vx / np.cos(np.radians([r["beta2"], r["beta3"]]))
    stations = [("P0", "PA", [p["P01"], r["P02"], r["P03"]], 0), ("P", "PA", [s["P1"], r["P2"], P3], 0),
                ("T0", "K", [p["T01"], r["T02"], r["T03"]], 1), ("T", "K", T, 1),
                ("RHO", "KG/M3", mdot / (A * Vx), 4),
                ("C", "M/S", C, 1), ("CX", "M/S", [Vx] * 3, 1), ("CTHETA", "M/S", Vx * np.tan(np.radians(alpha)), 1),
                ("ALPHA", "DEG", alpha, 2), ("MACH", "-", C / a, 4),
                ("W", "M/S", [nan, *Wrel], 1), ("BETA", "DEG", [nan, r["beta2"], r["beta3"]], 2),
                ("MREL", "-", [nan, r["M2R"], r["M3R"]], 4),
                ("P0REL", "PA", [nan, s["P02R"], s["P03R"]], 0), ("T0REL", "K", [nan, s["T02R"], s["T03R"]], 1),
                ("AREA", "IN2", A * M2IN**2, 2), ("HEIGHT", "IN", h * M2IN, 3),
                ("D HUB", "IN", 2 * np.full(3, rh) * M2IN, 3), ("D MEAN", "IN", 2 * rm * M2IN, 3),
                ("D TIP", "IN", 2 * rt * M2IN, 3),
                ("HUB/TIP", "-", rh / rt, 4),
                ("AN2", "IN2RPM2", np.array([r["AN2_1"], r["AN2_2"], r["AN2_3"]]) * M2IN**2, 0)]

    # Blade rows: chord from the exit height, as passed to the KO model
    chord = h[1:] / gm["h_over_c"]
    pitch = gm["s_over_c"] * chord
    ang_in, ang_out = np.array([0.0, r["beta2"]]), np.array([r["alpha2"], -r["beta3"]])
    Y = np.array([s["Y_s_all"], s["Y_r_all"]])           # cols: total, profile, secondary, clearance, TE, shock
    rows = [("H INLET", "IN", h[:2] * M2IN, 3), ("H EXIT", "IN", h[1:] * M2IN, 3),
            ("CHORD", "IN", chord * M2IN, 3), ("AX CHORD", "IN", gm["ax_over_c"] * chord * M2IN, 3),
            ("PITCH", "IN", pitch * M2IN, 3), ("S/C", "-", [gm["s_over_c"]] * 2, 4),
            ("BLADES", "-", np.ceil(2 * np.pi * rm[1:] / pitch), 0),
            ("TMAX", "IN", gm["tmax_over_c"] * chord * M2IN, 3),
            ("TIP CLR", "IN", [nan, gm["clearance_over_h"] * h[2] * M2IN], 4),
            ("ANG IN", "DEG", ang_in, 2), ("ANG OUT", "DEG", ang_out, 2), ("TURNING", "DEG", ang_in + ang_out, 2),
            ("XI", "-", ang_in / ang_out, 4), ("M IN", "-", [C[0] / a[0], r["M2R"]], 4), ("M OUT", "-", [r["M2"], r["M3R"]], 4),
            ("Y PROFILE", "-", Y[:, 1], 5), ("  SHOCK", "-", Y[:, 5], 5), ("Y SECOND", "-", Y[:, 2], 5),
            ("Y CLEAR", "-", Y[:, 3], 5), ("Y TE", "-", Y[:, 4], 5), ("Y TOTAL", "-", Y[:, 0], 5),
            ("RPM", "RPM", [0.0, N], 0)]

    stage = [("REACTION", "-", p["reac"], 4), ("PHI", "-", p["phi"], 4), ("PSI", "-", r["W"] / U**2, 4),
             ("U", "M/S", U, 2), ("U TIP", "M/S", 2 * np.pi * N / 60 * rt[1:].max(), 2),
             ("DH0", "J/KG", r["W"], 0), ("POWER", "KW", mdot * r["W"] / 1e3, 1),
             ("MDOT CORE", "KG/S", p["mdot_core"], 3), ("MDOT COOL", "KG/S", p["mdot_cool"], 3),
             ("MDOT STAGE", "KG/S", mdot, 3), ("T01", "K", p["T01"], 1), ("P01", "PA", p["P01"], 0),
             ("P01/P3", "-", PR, 3), ("ETA TS", "-", r["eta_ts"], 4), ("ETA TT", "-", r["eta_tt"], 4)]

    lines = [["STATION RESULTS", "UNITS", "1", "2", "3"]]
    lines += [[n, u] + [ff(v, d) for v in vals] for n, u, vals, d in stations]
    lines += [[], ["BLADE ROW RESULTS", "UNITS", "STATOR", "ROTOR"]]
    lines += [[n, u] + [ff(v, d) for v in vals] for n, u, vals, d in rows]
    lines += [[], ["STAGE RESULTS", "UNITS", "VALUE"]]
    lines += [[n, u, ff(v, d)] for n, u, v, d in stage]
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), fname)
    with open(path, "w", newline="") as f:
        f.write("\r\n".join(",".join(l) for l in lines) + "\r\n")
    print(f"Optimum design (U = {U:.0f} m/s, reaction = {p['reac']:.3f}, phi = {p['phi']:.3f}) written to {fname}")


best_pk = max((p for p in peaks if not np.isnan(p["eta_ts"])), key=lambda p: p["eta_ts"], default=None)
if best_pk is not None:
    write_optimum_csv(best_pk)

# ---- Figure: peak conditions vs U ----
pk = {k: np.array([p[k] for p in peaks]) for k in ("U", "reac", "phi", "psi", "eta_ts", "eta_tt", "Y_s", "Y_r")}
edge = np.array([(not np.isnan(p["eta_ts"])) and on_edge(p) for p in peaks])
fig, ax = plt.subplots(2, 3, figsize=(16, 8))
panels_pk = [(("eta_ts", "eta_tt"), "Peak efficiency"),
             (("reac",), "Reaction at peak eta_ts"),
             (("phi",), "Flow coefficient phi at peak eta_ts"),
             (("psi",), "Loading coefficient psi at peak eta_ts"),
             (("Y_s",), "Stator loss coefficient Y_s at peak"),
             (("Y_r",), "Rotor loss coefficient Y_r at peak")]
for a, (keys, title) in zip(ax.flat, panels_pk):
    for k in keys:
        a.plot(pk["U"], pk[k], "o-", label={"eta_ts": "eta_ts (peak)", "eta_tt": "eta_tt (same design)"}.get(k))
    if edge.any():
        a.plot(pk["U"][edge], pk[keys[0]][edge], "o", mfc="none", mec="r", ms=12, mew=1.5,
               label="peak on edge of search range")
    a.set_title(title); a.set_xlabel("Blade speed U [m/s]"); a.grid(alpha=0.3)
ax[0, 0].legend(fontsize=8)
fig.suptitle(f"Peak total-to-static efficiency vs blade speed (PR = {PR}, T01 = {BASE['T01']:.0f} K)")
fig.tight_layout()
if SAVE_FIGS:
    fig.savefig("peak_eta_ts_vs_U.png", dpi=150)

# ---- Figure: Smith-diagram-type scatter (phi vs psi) of every tested design ----
if smith_pts:
    sp = {k: np.array([p[k] for p in smith_pts]) for k in ("U", "reac", "phi", "psi", "eta_ts", "eta_tt")}
    fig, ax = plt.subplots(1, 3, figsize=(20, 6))
    for a, key, lab, cm in zip(ax, ("eta_ts", "eta_tt", "reac"),
                               ("Total-to-static efficiency", "Total-to-total efficiency", "Degree of reaction"),
                               ("viridis", "viridis", "coolwarm")):
        order = np.argsort(sp[key])                              # draw the highest values on top
        sc = a.scatter(sp["phi"][order], sp["psi"][order], c=sp[key][order], s=22, cmap=cm)
        fig.colorbar(sc, ax=a, label=lab)
        a.set_xlabel("Flow coefficient phi = Vx/U"); a.set_ylabel("Loading coefficient psi = W/U^2")
        a.set_title(f"Smith-type diagram - {lab}"); a.grid(alpha=0.3)
    for p in peaks:                                              # mark the peak eta_ts design of each U
        if not np.isnan(p["eta_ts"]):
            for a in ax[:2]:
                a.plot(p["phi"], p["psi"], "k*", ms=13, mec="w")
                a.annotate(f"U={p['U']:.0f}", (p["phi"], p["psi"]), textcoords="offset points",
                           xytext=(6, 6), fontsize=8)
    fig.suptitle(f"{len(smith_pts)} converged, feasible designs, U = {PEAK_U_LIST[0]:.0f}-{PEAK_U_LIST[-1]:.0f} m/s "
                 f"(stars = peak eta_ts for each U)")
    fig.tight_layout()
    if SAVE_FIGS:
        fig.savefig("smith_diagram.png", dpi=150)

plt.show()