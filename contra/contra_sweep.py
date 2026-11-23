import os
import numpy as np
from CoolProp.CoolProp import PropsSI
import constants as c
from two_stage_contra import two_stage_contra
from report import print_report, write_csv
import matplotlib.pyplot as plt
from contra_plots import plot_vs
from baseline import BASELINE, OPERATING, GEOM, LIMITS, R1_LINES
from multiprocessing import Pool
from functools import partial
from itertools import product

def evaluate(x, geom, T01, P01, mdot, P0_Pe, fluid):
    try:
        return two_stage_contra(*x, geom, T01, P01, mdot, P0_Pe, fluid) # x = (phi1, psi1, R1, R2, exit, rpm1)
    except (ValueError, RuntimeError):
        return None

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__))) # Write outputs next to this script
    # Operating conditions and geometry (baseline.py)
    fluid, T01, P01, P0_Pe, mdot = (OPERATING[k] for k in ("fluid", "T01", "P01", "P0_Pe", "mdot"))
    DEFAULT_GEOM = GEOM

    # Two stage contra-rotating sweep
    phi1_range = np.linspace(0.3, 0.9, 5)
    psi1_range = np.linspace(1.0, 3.0, 6)
    R1_range = np.linspace(0.0, 1.0, 11)
    R2_range = np.linspace(0.0, 1.0, 11)
    exit_range = np.radians(np.linspace(0, 0, 1))
    rpm1_range = np.linspace(14000, 24000, 6)

    # Valid ranges (baseline.py)
    XI_RANGE, TURN_RANGE, EXIT_RANGE = LIMITS["xi"], LIMITS["turning"], LIMITS["exit"]
    M_OUT_MAX, M_IN_MAX, AN2_MAX = LIMITS["M_out"], LIMITS["M_in"], LIMITS["AN2"]
    U_MAX, HT_RANGE = LIMITS["U_mean"], LIMITS["hub_tip"]

    def angles_ok(o):
        return (np.all((o["xi"] >= XI_RANGE[0]) & (o["xi"] <= XI_RANGE[1]))
                and np.all((o["turning"] >= TURN_RANGE[0]) & (o["turning"] <= TURN_RANGE[1]))
                and np.all((o["exit_deg"] >= EXIT_RANGE[0]) & (o["exit_deg"] <= EXIT_RANGE[1])))

    def limits_ok(o):
        return (max(o["M_out"]) <= M_OUT_MAX and max(o["M_in"]) <= M_IN_MAX
                and max(o["AN2_r1"], o["AN2_r2"]) <= AN2_MAX and max(abs(o["U1"]), abs(o["U2"])) <= U_MAX
                and HT_RANGE[0] <= min(o["hub_tip"]) and max(o["hub_tip"]) <= HT_RANGE[1])

    # Pre-filter on velocity-triangle angles (no thermodynamics needed)
    grid = list(product(phi1_range, psi1_range, R1_range, R2_range, exit_range))
    cands = []
    n_fail = 0
    n_invalid = 0
    for x in grid:
        try:
            ang = two_stage_contra(*x, rpm1_range[0], DEFAULT_GEOM, T01, P01, mdot, P0_Pe, fluid, angles_only=True)
        except (ValueError, ZeroDivisionError):
            n_fail += 1
            continue
        if angles_ok(ang):
            cands.append(x)
        else:
            n_invalid += 1
    print(f"{len(grid)} grid points, {len(cands)} pass angle checks")
    cands = [x + (rpm1,) for x in cands for rpm1 in rpm1_range] # angles do not depend on rpm

    # Full evaluation in parallel
    with Pool() as pool:
        results = pool.map(partial(evaluate, geom=DEFAULT_GEOM, T01=T01, P01=P01, mdot=mdot,
                                   P0_Pe=P0_Pe, fluid=fluid), cands)

    best = None
    records = [] # (x, out, valid) for plots
    for (phi1, psi1, R1, R2, exit_swirl, rpm1), out in zip(cands, results):
        if out is None:
            n_fail += 1
            continue
        valid = limits_ok(out)
        records.append(((phi1, psi1, R1, R2, np.degrees(exit_swirl), rpm1), out, valid and np.isfinite(out["eta_ts"])))
        if not valid:
            n_invalid += 1
            continue
        if np.isfinite(out["eta_ts"]) and (best is None or out["eta_ts"] > best["eta_ts"]):
            best = dict(out, phi1=phi1, psi1=psi1, R1=R1, R2=R2, exit_deg_swirl=np.degrees(exit_swirl), rpm1=rpm1)

    print(f"Infeasible/failed points: {n_fail}")
    print(f"Outside valid range: {n_invalid}")
    if best is None:
        print("No valid design point found")
        return
    print_report(best["report"])
    write_csv(best["report"], "two_stage_contra.csv")

    # Fine one-variable sweeps around the baseline, one line per R1
    def fine_sweep(pts):
        with Pool() as pool:
            res = pool.map(partial(evaluate, geom=DEFAULT_GEOM, T01=T01, P01=P01, mdot=mdot,
                                   P0_Pe=P0_Pe, fluid=fluid), pts)
        recs = []
        for x, out in zip(pts, res):
            if out is None or not np.isfinite(out["eta_ts"]):
                continue
            ok = angles_ok(out) and limits_ok(out)
            recs.append(((x[0], x[1], x[2], x[3], np.degrees(x[4]), x[5]), out, ok))
        return recs

    b = BASELINE
    ex = np.radians(b["exit_swirl_deg"])
    cond = f"T01 = {T01:.0f} K, P01/Pe = {P0_Pe}"

    base = fine_sweep([(b["phi1"], b["psi1"], b["R1"], b["R2"], ex, b["rpm1"])])
    if not base or not base[0][2]:
        print("WARNING: baseline design is infeasible or outside limits")
    else:
        write_csv(base[0][1]["report"], "two_stage_contra_baseline.csv")
        print(f"Baseline eta_ts = {base[0][1]['eta_ts']:.4f}  (optimum {best['eta_ts']:.4f})")

    phi_fine = np.linspace(0.3, 1.0, 29)
    recs = fine_sweep([(p, b["psi1"], r, b["R2"], ex, b["rpm1"]) for r in R1_LINES for p in phi_fine])
    plot_vs(recs, 0, "Flow coefficient φ1", b["phi1"], M_OUT_MAX, M_IN_MAX, AN2_MAX, tag="phi", x_opt=best["phi1"], u_max=U_MAX,
            title=f"Flow coefficient sweep (baseline ψ1 = {b['psi1']:.2f}, R2 = {b['R2']:.2f}, rpm1 = {b['rpm1']:.0f}, {cond})")

    rpm_fine = np.linspace(10000, 26000, 33)
    recs = fine_sweep([(b["phi1"], b["psi1"], r, b["R2"], ex, n) for r in R1_LINES for n in rpm_fine])
    plot_vs(recs, 5, "Rotor 1 speed [rpm]", b["rpm1"], M_OUT_MAX, M_IN_MAX, AN2_MAX, tag="rpm", x_opt=best["rpm1"], u_max=U_MAX,
            title=f"Speed sweep (baseline φ1 = {b['phi1']:.2f}, ψ1 = {b['psi1']:.2f}, R2 = {b['R2']:.2f}, {cond})")
    plt.show()


if __name__ == "__main__":
    main()