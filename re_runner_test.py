import numpy as np
from two_stage_contra import two_stage_contra
from radial_equilibrium import two_stage_contra_RE, print_RE, plot_RE

# Meanline inputs
phi1 = 0.6
psi1 = 2.2
R1 = 0.4
R2 = 0.4
exit_swirl = np.radians(0.0)  # rotor 2 exit absolute angle [rad]
rpm1 = 9000

# Geometry: h/c, tmax/c, ax/c, clearance/h, shrouded, t/o
geom = (1.5, 0.20, 0.80, 0.015, True, 0.05)

T01 = 1200.0     # K
P01 = 1.0e6      # Pa
mdot = 20.0      # kg/s
P01_Pe = 3.0     # total-to-exit pressure ratio
fluid = 'Air'

n_span = 11      # number of span points hub to tip

# Meanline
res = two_stage_contra(phi1, psi1, R1, R2, exit_swirl, rpm1, geom, T01, P01, mdot, P01_Pe, fluid)

print("Meanline: eta_ts = %.4f, eta_tt = %.4f, rpm2 = %.1f, r_mean = %.4f m" % (
    res['eta_ts'], res['report']['stage']['eta_tt'], res['rpm2'], res['r']))
print("Meanline dh split (stator, rotor1, rotor2):", res['dh_split'])

# Radial equilibrium
re = two_stage_contra_RE(res, n_span)
print_RE(re)
plot_RE(re)