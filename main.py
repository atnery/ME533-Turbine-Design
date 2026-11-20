import numpy as np
from CoolProp.CoolProp import PropsSI
import constants as c
from two_stage_contra import two_stage_contra
from report import print_report, write_csv
from multiprocessing import Pool
from functools import partial
from itertools import product

def evaluate(x, rpm1, geom, T01, P01, mdot, P0_Pe, fluid):
    try:
        return two_stage_contra(*x, rpm1, geom, T01, P01, mdot, P0_Pe, fluid)
    except (ValueError, RuntimeError):
        return None

def main():
    # Operating conditions
    fluid = "air"
    T01 = 1850 # K
    P01 = 20 * c.BAR2PA # Pa
    P0_Pe = 4.1
    Pe = P01 / P0_Pe
    mdot = 30 # kg/s

    DEFAULT_GEOM = (1.5, 0.2, 0.8, 0.01, False, 0.05) # h/c, tmax/c, ax/c, clr/h, shrouded, t/o

    # Two stage contra-rotating sweep
    rpm1 = 20000

    phi1_range = np.linspace(0.3, 0.9, 5)
    psi1_range = np.linspace(1.0, 3.0, 6)
    R1_range = np.linspace(0.0, 1.0, 11)
    R2_range = np.linspace(0.0, 1.0, 11)
    exit_range = np.radians(np.linspace(0, 0, 1))

    # Valid ranges for the AMDC/KO correlations (per blade row)
    XI_RANGE = (-0.4, 1.0)       # beta1/alpha2, KO convention
    TURN_RANGE = (40.0, 120.0)   # deg
    EXIT_RANGE = (40.0, 75.0)    # deg from axial, extent of the AM charts
    M_OUT_MAX = 1.17             # exit (relative) Mach, KO data up to 1.17
    M_IN_MAX = 0.8               # inlet (relative) Mach
    AN2_MAX = 3.2e7              # m^2 rpm^2, both rotors

    def angles_ok(o):
        return (np.all((o["xi"] >= XI_RANGE[0]) & (o["xi"] <= XI_RANGE[1]))
                and np.all((o["turning"] >= TURN_RANGE[0]) & (o["turning"] <= TURN_RANGE[1]))
                and np.all((o["exit_deg"] >= EXIT_RANGE[0]) & (o["exit_deg"] <= EXIT_RANGE[1])))

    # Pre-filter on velocity-triangle angles (no thermodynamics needed)
    grid = list(product(phi1_range, psi1_range, R1_range, R2_range, exit_range))
    cands = []
    n_fail = 0
    n_invalid = 0
    for x in grid:
        try:
            ang = two_stage_contra(*x, rpm1, DEFAULT_GEOM, T01, P01, mdot, P0_Pe, fluid, angles_only=True)
        except (ValueError, ZeroDivisionError):
            n_fail += 1
            continue
        if angles_ok(ang):
            cands.append(x)
        else:
            n_invalid += 1
    print(f"{len(grid)} grid points, {len(cands)} pass angle checks")

    # Full evaluation in parallel
    with Pool() as pool:
        results = pool.map(partial(evaluate, rpm1=rpm1, geom=DEFAULT_GEOM, T01=T01, P01=P01, mdot=mdot,
                                   P0_Pe=P0_Pe, fluid=fluid), cands)

    best = None
    for (phi1, psi1, R1, R2, exit_swirl), out in zip(cands, results):
        if out is None:
            n_fail += 1
            continue
        valid = (max(out["M_out"]) <= M_OUT_MAX and max(out["M_in"]) <= M_IN_MAX
                 and max(out["AN2_r1"], out["AN2_r2"]) <= AN2_MAX)
        if not valid:
            n_invalid += 1
            continue
        if np.isfinite(out["eta_ts"]) and (best is None or out["eta_ts"] > best["eta_ts"]):
            best = dict(out, phi1=phi1, psi1=psi1, R1=R1, R2=R2, exit_deg_swirl=np.degrees(exit_swirl))

    print(f"Infeasible/failed points: {n_fail}")
    print(f"Outside valid range: {n_invalid}")
    if best is None:
        print("No valid design point found")
        return
    print_report(best["report"])
    write_csv(best["report"], "two_stage_contra.csv")


if __name__ == "__main__":
    main()