"""
Parametric test + plots for single_stage_turbine().
Sweeps: degree of reaction, blade speed U, flow coefficient phi (= Vx/U); pressure ratio held fixed.
Each sweep produces five figure grids:
    1) performance  : eta_tt, eta_ts, psi, T03, h2, h3
    2) Mach numbers : absolute and relative, stator exit (2) and rotor exit (3)
    3) thermo state : P2, P02, P03 (top row) and T2, T3, T02, T03 (bottom row)
    4) angles/losses: absolute flow angles alpha2, alpha3 and KO loss coefficients Y_s, Y_r
    5) structural   : N [rpm], r_mean3, annulus areas A1/A2/A3 and AN^2 at stator inlet (1), stator exit (2), rotor exit (3)
Designs with AN^2 above AN2_LIMIT at ANY of the three stations are omitted (NaN) from every plot when ENFORCE_AN2 = True;
the AN^2 panels still show the raw values (red dashed = limit) so you can see where each station crosses it.
Non-converged cases come back as NaN and show up as gaps (line plots) or grey cells (maps).
Run:  python test_turbine.py   (needs single_stage.py and KO_losses.py in the same folder)
"""
import numpy as np
import matplotlib.pyplot as plt
from single_stage import single_stage_turbine, OUTPUT_NAMES

SAVE_FIGS = False   # set True to also write every figure to a PNG

# ------------------------- Baseline inputs -------------------------
BASE = dict(T01=1850.0, P01=20*(10**5), mdot_core=30.0, mdot_cool=2.0, reac=0.6, r_hub=0.30, U=300.0, phi=0.5) # Baseline design inputs
PR = 4.1                                        # P01/P3 (held fixed)
AN2_LIMIT = 3.2e7                               # structural limit on A*N^2 [m^2 rpm^2], applied at stations 1, 2 and 3
RAW_KEYS = ["AN2_1_raw", "AN2_2_raw", "AN2_3_raw"]   # unmasked AN^2 values kept for the AN^2 panels
ENFORCE_AN2 = True                              # True -> omit designs that exceed AN2_LIMIT
FHUB = 1.0                                      # <-- E in ko_losses is now just fHub (single number)
SCALARS = [n for n in OUTPUT_NAMES if n != "sol"]

# Preflight: run the baseline once with debug=True so a failing KO call prints its real error
_p = BASE
_pre = single_stage_turbine(_p["T01"], _p["P01"], _p["P01"] / PR, _p["mdot_core"], _p["mdot_cool"],
                            _p["reac"], _p["r_hub"], _p["U"], E=FHUB, phi=_p["phi"], debug=True, AN2_max=None)
if np.isnan(_pre[0]):
    print("WARNING: baseline case did not converge (NaN) - check the traceback above / the baseline inputs.")


def run(**over):
    """Run one case. Returns dict of all scalar outputs (NaN if not converged). Adds psi = W/U^2."""
    p = {**BASE, **over}
    # Call with AN2_max=None to get the raw design, then apply the AN^2 constraint here so the violators can be shown
    out = single_stage_turbine(p["T01"], p["P01"], p["P01"] / PR, p["mdot_core"], p["mdot_cool"],
                               p["reac"], p["r_hub"], p["U"], E=FHUB, phi=p["phi"], AN2_max=None)
    r = dict(zip(OUTPUT_NAMES, out))
    r.pop("sol")
    r["psi"] = r["W"] / p["U"] ** 2
    for i in (1, 2, 3):
        r[f"AN2_{i}_raw"] = r[f"AN2_{i}"]                       # unmasked, for the AN^2 panels
    an2_max = np.nanmax([r["AN2_1"], r["AN2_2"], r["AN2_3"]]) if not np.isnan(r["AN2_3"]) else np.nan
    # status: 0 = no solution, 1 = feasible, 2 = converged but AN^2 > limit at any station
    r["status"] = 0 if np.isnan(an2_max) else (2 if an2_max > AN2_LIMIT else 1)
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
eta = np.full((len(phg), len(reg)), np.nan)
psi = np.full_like(eta, np.nan)
status = np.zeros_like(eta)
an2_max = np.full_like(eta, np.nan)          # highest AN^2 of the three stations for every converged case (raw, unmasked)
for i, ph in enumerate(phg):
    for j, rc in enumerate(reg):
        r = run(reac=rc, phi=ph)
        eta[i, j], psi[i, j], status[i, j] = r["eta_ts"], r["psi"], r["status"]
        raw = [r[k] for k in RAW_KEYS]
        if not np.all(np.isnan(raw)):
            an2_max[i, j] = np.nanmax(raw)

fig, ax = plt.subplots(2, 2, figsize=(14, 10))
cmap = plt.get_cmap("viridis").copy(); cmap.set_bad("lightgrey")     # NaN -> grey
for a, data_, lab in zip(ax.flat[:2], [eta, psi], ["Total-to-static efficiency", "Loading coefficient psi"]):
    im = a.pcolormesh(reg, phg, np.ma.masked_invalid(data_), cmap=cmap, shading="nearest")
    fig.colorbar(im, ax=a)
    a.set_title(f"{lab}  (grey = omitted / no solution)")
    a.set_xlabel("Degree of reaction"); a.set_ylabel("Flow coefficient phi")

ax[1, 0].pcolormesh(reg, phg, status, cmap=ListedColormap(["#d9534f", "#5cb85c", "#f0ad4e"]), vmin=-0.5, vmax=2.5, shading="nearest")
ax[1, 0].set_title("Red = no solution, green = feasible, orange = AN^2 > limit")

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

for a in ax.flat[2:]:
    a.set_xlabel("Degree of reaction"); a.set_ylabel("Flow coefficient phi")
fig.tight_layout()
if SAVE_FIGS:
    fig.savefig("map_reaction_vs_phi.png", dpi=150)

n_all = status.size
print(f"AN^2 limit applied at stator inlet, stator exit and rotor exit.")
print(f"Map ({n_all} cases): {int(np.sum(status == 1))} feasible, "
      f"{int(np.sum(status == 2))} violate AN^2 <= {AN2_LIMIT:.1e}, {int(np.sum(status == 0))} no solution")
print(f"Highest AN^2 on the map: {np.nanmax(an2_max):.3e}  (limit {AN2_LIMIT:.1e})")
plt.show()