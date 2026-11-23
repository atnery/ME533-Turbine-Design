"""Baseline design and run definition. Change these deliberately - sweeps are plotted around them."""

# Baseline design point (fine sweeps vary one variable at a time around this)
BASELINE = dict(phi1=0.50, psi1=2.2, R1=0.65, R2=0.60, exit_swirl_deg=0.0, rpm1=16000)

# Operating conditions
OPERATING = dict(fluid="air", T01=1850.0, P01=20e5, P0_Pe=4.1, mdot=30.0)  # K, Pa, -, kg/s

# Blade geometry ratios: h/c, tmax/c, ax/c, clr/h, shrouded, t/o
GEOM = (1.5, 0.2, 0.8, 0.01, False, 0.05)

# Validity limits (per blade row)
LIMITS = dict(
    xi=(-0.4, 1.0),          # beta1/alpha2, KO convention
    turning=(40.0, 120.0),   # deg
    exit=(40.0, 75.0),       # deg from axial, extent of the AM charts
    M_out=1.17,              # exit (relative) Mach, KO data up to 1.17
    M_in=0.8,                # inlet (relative) Mach
    AN2=3.2e7,               # m^2 rpm^2, both rotors
    U_mean=500.0,            # m/s, mean blade speed, both rotors
    hub_tip=(0.7, 0.94),     # all stations
)

# Reaction values drawn as lines in the fine sweeps
R1_LINES = (0.3, 0.5, 0.65, 0.8)