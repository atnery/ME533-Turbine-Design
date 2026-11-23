import csv
import numpy as np

M2IN = 1 / 0.0254  # m -> in

def ff(x, d, w=12):
    """Fixed-point format with d decimals. NaN prints as a dash."""
    if x is None or not np.isfinite(x):
        return "-".rjust(w) if w else ""
    if round(x, d) == 0:
        x = 0.0  # avoid "-0.00"
    return f"{x:{w}.{d}f}"

def build_tables(rep):
    st, rw, sg = rep["stations"], rep["rows"], rep["stage"]
    r = sg["r"]
    h = np.array(st["h"])

    stations = [
        ("P0", "PA", st["P0"], 0), ("P", "PA", st["P"], 0),
        ("T0", "K", st["T0"], 1), ("T", "K", st["T"], 1),
        ("RHO", "KG/M3", st["rho"], 4),
        ("C", "M/S", st["C"], 1), ("CX", "M/S", st["Cx"], 1), ("CTHETA", "M/S", st["Ctheta"], 1),
        ("ALPHA", "DEG", st["alpha"], 2), ("MACH", "-", st["M"], 4),
        ("W R1", "M/S", st["W_r1"], 1), ("BETA R1", "DEG", st["beta_r1"], 2), ("MREL R1", "-", st["Mrel_r1"], 4),
        ("P0REL R1", "PA", st["P0rel_r1"], 0), ("T0REL R1", "K", st["T0rel_r1"], 1),
        ("W R2", "M/S", st["W_r2"], 1), ("BETA R2", "DEG", st["beta_r2"], 2), ("MREL R2", "-", st["Mrel_r2"], 4),
        ("P0REL R2", "PA", st["P0rel_r2"], 0), ("T0REL R2", "K", st["T0rel_r2"], 1),
        ("AREA", "IN2", np.array(st["A"]) * M2IN**2, 2), ("HEIGHT", "IN", h * M2IN, 3),
        ("D HUB", "IN", (2 * r - h) * M2IN, 3), ("D MEAN", "IN", [2 * r * M2IN] * 4, 3), ("D TIP", "IN", (2 * r + h) * M2IN, 3),
    ]

    chord = np.array(rw["chord"])
    pitch = np.array(rw["s_c"]) * chord
    Y = np.array(rw["Y"])  # rows: stator/r1/r2; cols: total, profile, secondary, clearance, TE, shock
    rows = [
        ("H INLET", "IN", np.array(rw["h_in"]) * M2IN, 3), ("H EXIT", "IN", np.array(rw["h_out"]) * M2IN, 3),
        ("CHORD", "IN", chord * M2IN, 3), ("AX CHORD", "IN", np.array(rw["ax_chord"]) * M2IN, 3),
        ("PITCH", "IN", pitch * M2IN, 3), ("S/C", "-", rw["s_c"], 4),
        ("BLADES", "-", np.ceil(2 * np.pi * r / pitch), 0),
        ("TMAX", "IN", np.array(rw["tmax"]) * M2IN, 3), ("TIP CLR", "IN", np.array(rw["clr"]) * M2IN, 4),
        ("ANG IN", "DEG", rw["ang_in"], 2), ("ANG OUT", "DEG", rw["ang_out"], 2),
        ("TURNING", "DEG", rw["turning"], 2), ("XI", "-", rw["xi"], 4),
        ("M IN", "-", rw["M_in"], 4), ("M OUT", "-", rw["M_out"], 4),
        ("Y PROFILE", "-", Y[:, 1], 5), ("  SHOCK", "-", Y[:, 5], 5), ("Y SECOND", "-", Y[:, 2], 5), ("Y CLEAR", "-", Y[:, 3], 5),
        ("Y TE", "-", Y[:, 4], 5), ("Y TOTAL", "-", Y[:, 0], 5),
        ("DH STATIC", "FRAC", rw["dh_split"], 4),
        ("RPM", "RPM", rw["rpm"], 0), ("AN2", "IN2RPM2", np.array(rw["AN2"]) * M2IN**2, 0),
    ]

    stage = [
        ("PHI1", "-", sg["phi1"], 4), ("PSI1", "-", sg["psi1"], 4),
        ("R1 (DH0)", "-", sg["R1"], 4), ("R1 STATIC", "-", sg["R1_static"], 4),
        ("PHI2", "-", sg["phi2"], 4), ("PSI2", "-", sg["psi2"], 4), ("R2 (DH0)", "-", sg["R2"], 4),
        ("U2/U1", "-", sg["s"], 4), ("U1", "M/S", sg["U1"], 2), ("U2", "M/S", sg["U2"], 2),
        ("DH0 R1", "J/KG", sg["dh0_r1"], 0), ("DH0 R2", "J/KG", sg["dh0_r2"], 0),
        ("POWER R1", "KW", sg["mdot"] * sg["dh0_r1"] / 1e3, 1), ("POWER R2", "KW", sg["mdot"] * sg["dh0_r2"] / 1e3, 1),
        ("POWER TOT", "KW", sg["power"] / 1e3, 1),
        ("WORK FR R1", "-", sg["dh0_r1"] / (sg["dh0_r1"] + sg["dh0_r2"]), 4),
        ("WORK FR R2", "-", sg["dh0_r2"] / (sg["dh0_r1"] + sg["dh0_r2"]), 4),
        ("MDOT", "KG/S", sg["mdot"], 3),
        ("T01", "K", sg["T01"], 1), ("P01", "PA", sg["P01"], 0), ("P01/PE", "-", sg["PR"], 3),
        ("ETA TS", "-", sg["eta_ts"], 4), ("ETA TT", "-", sg["eta_tt"], 4),
    ]
    return stations, rows, stage

def print_report(rep):
    stations, rows, stage = build_tables(rep)
    bar = " " + "*" * 70
    print(bar)
    print("        TWO-STAGE CONTRA-ROTATING TURBINE  -  MEANLINE RESULTS")
    print(bar)

    print("\n STATION RESULTS  (ANGLES: GLOBAL SIGNED, + IN ROTOR 1 DIRECTION)")
    print(f" {'':10s}{'UNITS':>8s}" + "".join(f"{s:>12s}" for s in ("1", "2", "3", "4")))
    for name, unit, vals, d in stations:
        print(f" {name:10s}{unit:>8s}" + "".join(ff(v, d) for v in vals))

    print("\n BLADE ROW RESULTS  (ANGLES: KO CONVENTION)")
    print(f" {'':10s}{'UNITS':>8s}" + "".join(f"{s:>12s}" for s in ("STATOR", "ROTOR 1", "ROTOR 2")))
    for name, unit, vals, d in rows:
        print(f" {name:10s}{unit:>8s}" + "".join(ff(v, d) for v in vals))

    print("\n STAGE RESULTS")
    for name, unit, val, d in stage:
        print(f" {name:10s}{unit:>8s}{ff(val, d)}")
    print(bar)

def write_csv(rep, filename):
    stations, rows, stage = build_tables(rep)
    with open(filename, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["STATION RESULTS", "UNITS", "1", "2", "3", "4"])
        for name, unit, vals, d in stations:
            w.writerow([name, unit] + [ff(v, d, 0) for v in vals])
        w.writerow([])
        w.writerow(["BLADE ROW RESULTS", "UNITS", "STATOR", "ROTOR 1", "ROTOR 2"])
        for name, unit, vals, d in rows:
            w.writerow([name, unit] + [ff(v, d, 0) for v in vals])
        w.writerow([])
        w.writerow(["STAGE RESULTS", "UNITS", "VALUE"])
        for name, unit, val, d in stage:
            w.writerow([name, unit, ff(val, d, 0)])