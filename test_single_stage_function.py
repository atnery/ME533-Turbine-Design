"""
Parametric test + plots for single_stage_turbine().
Place next to single_stage_turbine.py and KO_losses.py, then run:  python test_turbine.py
Non-converged cases come back as NaN and show up as gaps (line plots) or grey cells (maps).
"""
import numpy as np
import matplotlib.pyplot as plt
from single_stage import single_stage_turbine

# ------------------------- Baseline inputs -------------------------
BASE = dict(T01=1850.0, P01=20*(10**5), mdot_core=30.0, mdot_cool=2.0, reac=0.6, r_hub=0.30, U=300.0, phi=0.5) # Baseline design inputs
PR = 4.1                                  # P01/P3
E_COEFFS = [0.04, 0.04, 1.0, 0.02, 0.02]        # <-- replace with your KO chart values
OUT_NAMES = ["T03", "P03", "eta_tt", "eta_ts", "W", "A2", "A3", "h2", "h3"] 


def run(**over):
    """Run one case. Returns dict of outputs (NaN if not converged). Adds psi = W/U^2."""
    p = {**BASE, **over}
    out = single_stage_turbine(p["T01"], p["P01"], p["P01"] / PR, p["mdot_core"], p["mdot_cool"],
                               p["reac"], p["r_hub"], p["U"], E=E_COEFFS, phi=p["phi"])
    r = dict(zip(OUT_NAMES, out))
    r["psi"] = r["W"] / p["U"] ** 2
    return r


def sweep(xname, xs, **fixed):
    res = {k: np.full(len(xs), np.nan) for k in OUT_NAMES + ["psi"]}
    for i, x in enumerate(xs):
        r = run(**{**fixed, xname: x})
        for k in res:
            res[k][i] = r[k]
    return res


PANELS = [("eta_tt", "Total-to-total efficiency", 1),
          ("eta_ts", "Total-to-static efficiency", 1),
          ("psi", "Loading coefficient  psi = W/U^2", 1),
          ("T03", "Exit total temperature T03 [K]", 1),
          ("h2", "Stator exit blade height h2 [mm]", 1e3),
          ("h3", "Rotor exit blade height h3 [mm]", 1e3)]


def family_plot(xname, xs, xlabel, fam_name, fam_vals, fam_label, fname, title):
    """One figure: sweep xname, one curve per value of the family variable."""
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    for fv in fam_vals:
        res = sweep(xname, xs, **{fam_name: fv})
        for a, (key, _, scale) in zip(ax.flat, PANELS):
            a.plot(xs, res[key] * scale, "o-", ms=3, label=fam_label(fv))
    for a, (_, t, _) in zip(ax.flat, PANELS):
        a.set_title(t); a.set_xlabel(xlabel); a.grid(alpha=0.3)
    ax[0, 0].legend()
    fig.suptitle(f"{title}   (PR = {PR}, T01 = {BASE['T01']:.0f} K)")
    fig.tight_layout()
    fig.savefig(fname, dpi=150)


# ------------------------- 1. Degree of reaction (family: flow coefficient) -------------------------
reacs = np.linspace(0.1, 0.9, 25)
family_plot("reac", reacs, "Degree of reaction", "phi", [0.3, 0.5, 0.7],
            lambda v: f"phi = {v}", "sweep_reaction.png",
            f"Reaction sweep (U = {BASE['U']:.0f} m/s)")

# ------------------------- 2. Blade speed U (family: reaction; phi held fixed) -------------------------
Us = np.linspace(150, 500, 25)
family_plot("U", Us, "Blade speed U [m/s]", "reac", [0.3, 0.5, 0.7],
            lambda v: f"reaction = {v}", "sweep_blade_speed.png",
            f"Blade-speed sweep (phi = {BASE['phi']} fixed, so Vx scales with U)")

# ------------------------- 3. Flow coefficient (family: reaction; U held fixed) -------------------------
phis = np.linspace(0.2, 1.0, 25)
family_plot("phi", phis, "Flow coefficient phi = Vx/U", "reac", [0.3, 0.5, 0.7],
            lambda v: f"reaction = {v}", "sweep_flow_coefficient.png",
            f"Flow-coefficient sweep (U = {BASE['U']:.0f} m/s)")

# ------------------------- 4. Map: reaction vs flow coefficient -------------------------
reg = np.linspace(0.1, 0.9, 17)
phg = np.linspace(0.2, 1.0, 17)
eta = np.full((len(phg), len(reg)), np.nan)
psi = np.full_like(eta, np.nan)
for i, ph in enumerate(phg):
    for j, rc in enumerate(reg):
        r = run(reac=rc, phi=ph)
        eta[i, j], psi[i, j] = r["eta_ts"], r["psi"]

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
cmap = plt.get_cmap("viridis").copy(); cmap.set_bad("lightgrey")     # NaN -> grey
for a, data, lab in zip(ax, [eta, psi], ["Total-to-static efficiency", "Loading coefficient psi"]):
    im = a.pcolormesh(reg, phg, np.ma.masked_invalid(data), cmap=cmap, shading="nearest")
    fig.colorbar(im, ax=a)
    a.set_title(f"{lab}  (grey = no solution)")
    a.set_xlabel("Degree of reaction"); a.set_ylabel("Flow coefficient phi")
fig.tight_layout()
fig.savefig("map_reaction_vs_phi.png", dpi=150)

n_ok = int(np.sum(~np.isnan(eta)))
print(f"Map: {n_ok}/{eta.size} cases converged ({100 * n_ok / eta.size:.0f}%)")
print("Saved: sweep_reaction.png, sweep_blade_speed.png, sweep_flow_coefficient.png, map_reaction_vs_phi.png")
plt.show()