#dailey m* method for internal cooling. Film cooling isn't explicitly considered but it has an assumed presence
# Stations: 1 stator inlet, 2 stator exit / rotor inlet, 3 rotor exit


import math
from CoolProp.CoolProp import PropsSI

CSV_FILE = "single_stage_optimum.csv"
IN = 0.0254 # in2m

# read meanline CSV -------------------------------
station = {} # [st1, st2, st3]
blade = {} # [stator, rotor]
stage = {} # single value
section = ""

f = open(CSV_FILE)
for line in f:
    parts = line.strip().split(",")
    if len(parts) < 2:
        continue
    if parts[0] == "STATION RESULTS" or parts[0] == "BLADE ROW RESULTS" or parts[0] == "STAGE RESULTS":
        section = parts[0]
        continue
    label = parts[0].strip()
    vals = []
    for p in parts[2:]:
        if p == "":
            vals.append(None)
        else:
            vals.append(float(p))
    if section == "STATION RESULTS":
        station[label] = vals
    elif section == "BLADE ROW RESULTS":
        blade[label] = vals
    elif section == "STAGE RESULTS":
        stage[label] = vals[0]
f.close()

# from meanline ----------------
T01 = stage["T01"] # stage inlet T0 [K]
P01 = stage["P01"] # stage inlet P0 [Pa]
mdot = stage["MDOT CORE"] # core flow [kg/s]
U = stage["U"] # mean blade speed [m/s]
dh0 = stage["DH0"] # stage work [J/kg]
T03 = station["T0"][2] # rotor exit T0 [K]
cp_g = dh0 / (T01 - T03) # gas cp matching meanline work
Ctheta2 = station["CTHETA"][1] # rotor inlet swirl [m/s]
eta_uncooled = stage["ETA TT"] # uncooled eta_tt

# cycle assumptions ---------------- i dontt fuckin know man...
P_amb = 101325.0 # [Pa]
T_amb = 288.15 # [K]
dP_comb = 0.04 # combustor P0 loss fraction
eta_poly = 0.90 # compressor polytropic eff
P_cd = P01 / (1.0 - dP_comb) # compressor delivery P0
T_cd = T_amb * (P_cd / P_amb) ** (0.4 / (1.4 * eta_poly)) # compressor delivery T0 = coolant source

OTDF = 0.30 # stator peak factor, dailey: 0.25-0.50
RTDF = 0.20 # rotor radial factor, dailey: ~0.20

# assume we don't have pre swirl for rotor cooling
Up = 0.0 # disc speed at pre-swirl radius [m/s], 0 = none
Cth_p = 0.0 # pre-swirl nozzle swirl [m/s]

# per row [stator, rotor] ----------------
name = ["Stator (NGV)", "Rotor"]
N = [blade["BLADES"][0], blade["BLADES"][1]]
chord = [blade["CHORD"][0] * IN, blade["CHORD"][1] * IN]
span = [0.5 * (blade["H INLET"][0] + blade["H EXIT"][0]) * IN,
        0.5 * (blade["H INLET"][1] + blade["H EXIT"][1]) * IN] # mean of inlet/exit height
V_exit = [station["C"][1], station["W"][2]] # stator C2, rotor W3
rho_ex = [station["RHO"][1], station["RHO"][2]]
T_ex = [station["T"][1], station["T"][2]] # static, for properties
P_ex = [station["P"][1], station["P"][2]]
M_exit = [station["MACH"][1], station["MREL"][2]]

perim_c = [2.3, 2.3] # perimeter/chord, assumed
Tm = [1250.0, 1200.0] # allowable metal T [K]
eta_c = [0.6, 0.6] # cooling efficiency, Fig 25
t_tbc = [0.0, 0.0] # TBC thickness [m], 0 = none
k_tbc = [1.0, 1.0] # TBC conductivity [W/mK]
VcV0 = [0.5, 0.5] # coolant jet / freestream velocity
alpha = [35.0, 35.0] # injection angle [deg]

# Fig 15 coolant share by region 1-7 (% core) and local Mach
share = [[2.5, 0.75, 1.75, 0.75, 1.5, 1.375, 2.75],
         [1.0, 0.6, 0.0, 0.0, 0.45, 0.45, 1.05]]
Mreg = [[0.075, 0.275, 0.5, 0.95, 0.1, 0.3, 0.875],
        [0.075, 0.275, 0.5, 0.95, 0.3, 0.3, 0.875]]

# calculation ----------------
dT_comb = T01 - T_cd
cp_c = PropsSI("C", "T", T_cd, "P", P_cd, "Air")
mc_row = [0.0, 0.0]
deta_row = [0.0, 0.0]
Tci_row = [T_cd, T_cd]
T02mix = T01 # stator adiabatic

print("Coolant source: P_cd = %.0f Pa, T_cd = %.1f K, cp_c = %.0f J/kgK" % (P_cd, T_cd, cp_c))
print("Gas cp from meanline work = %.0f J/kgK" % cp_g)

for i in range(2):

    # Step 1: design gas T
    if i == 0:
        Tg = T01 + OTDF * dT_comb
    else:
        T0R2 = T02mix - (2.0 * U * Ctheta2 - U**2) / (2.0 * cp_g) # relative T0
        Tg = T0R2 + RTDF * dT_comb

    # Step 2: coolant supply T
    if i == 0:
        Tci = T_cd
    else:
        Tci = T_cd - (2.0 * Up * Cth_p - Up**2) / (2.0 * cp_c) # pre-swirl
    Tci_row[i] = Tci

    # Step 3: required effectiveness (Eq 5)
    eps = (Tg - Tm[i]) / (Tg - Tci)
    if eps <= 0.0:
        print(name[i], ": Tg below Tm, row can be uncooled")
        continue
    if eps >= 1.0:
        print(name[i], ": Tm below coolant T, impossible")
        continue

    # Step 4: external htc and wetted area
    mu = PropsSI("V", "T", T_ex[i], "P", P_ex[i], "Air")
    kg = PropsSI("L", "T", T_ex[i], "P", P_ex[i], "Air")
    Pr = PropsSI("Prandtl", "T", T_ex[i], "P", P_ex[i], "Air")
    Re = rho_ex[i] * V_exit[i] * chord[i] / mu
    hg = 0.037 * Re**0.8 * Pr**(1.0/3.0) * kg / chord[i] # turbulent flat plate
    h_eff = hg / (1.0 + hg * t_tbc[i] / k_tbc[i]) # Eq 2, TBC
    A = perim_c[i] * chord[i] * span[i]

    # Step 5: m* and coolant flow (Eqs 4, 6)
    mstar = eps / (eta_c[i] * (1.0 - eps))
    mc_one = mstar * h_eff * A / cp_c
    mc_row[i] = mc_one * N[i]

    # Step 6: coolant exit T (Eq 3)
    Tce = Tci + eta_c[i] * (Tm[i] - Tci)

    # Step 7: Hartsel loss (Fig 16) over Fig 15 regions
    m_main = mdot
    T0row = T01
    if i == 1:
        m_main = mdot + mc_row[0]
        T0row = T02mix
    bracket = 1.0 + Tce / T0row - 2.0 * VcV0[i] * math.cos(math.radians(alpha[i]))
    total_share = 0.0
    for r in range(7):
        total_share = total_share + share[i][r]
    dPP = 0.0
    deta_rel = 0.0
    for r in range(7):
        mfrac = (share[i][r] / total_share) * mc_row[i] / m_main
        dPP = dPP + 0.7 * Mreg[i][r]**2 * mfrac * bracket # gamma/2 ~ 0.7
        deta_rel = deta_rel + (Mreg[i][r] / M_exit[i])**2 * mfrac * bracket
    deta_row[i] = deta_rel * eta_uncooled

    # Step 8: mix stator coolant before rotor
    if i == 0:
        T02mix = (mdot * cp_g * T01 + mc_row[0] * cp_c * Tci) / ((mdot + mc_row[0]) * cp_g)

    print("----", name[i], "----")
    print("  Tg design = %.1f K" % Tg)
    print("  Tci = %.1f K" % Tci)
    print("  eps0 = %.3f" % eps)
    print("  Re_c = %.3e" % Re)
    print("  hg / h_eff = %.0f / %.0f W/m2K" % (hg, h_eff))
    print("  A per aerofoil = %.5f m2" % A)
    print("  m* = %.3f" % mstar)
    print("  mc per aerofoil = %.5f kg/s" % mc_one)
    print("  mc row = %.3f kg/s (%.2f %% of core)" % (mc_row[i], 100.0 * mc_row[i] / mdot))
    print("  Tce = %.1f K" % Tce)
    print("  Hartsel dP/P0 = %.4f" % dPP)
    print("  d(eta) row = %.4f" % deta_row[i])

# Step 9: stage totals
m_after_stator = mdot + mc_row[0]
m_exit = m_after_stator + mc_row[1]
T03mix = (m_after_stator * cp_g * T03 + mc_row[1] * cp_c * Tci_row[1]) / (m_exit * cp_g)
deta_stage = 0.5 * (deta_row[0] + deta_row[1]) # stage ~ half of row, Fig 16

print("---- STAGE ----")
print("  total coolant = %.3f kg/s (%.2f %% of core)" % (mc_row[0] + mc_row[1], 100.0 * (mc_row[0] + mc_row[1]) / mdot))
print("  mdot rotor in = %.3f kg/s" % m_after_stator)
print("  T02 mixed = %.1f K" % T02mix)
print("  mdot exit = %.3f kg/s" % m_exit)
print("  T03 mixed = %.1f K" % T03mix)
print("  d(eta) stage = %.4f" % deta_stage)
print("  eta_tt cooled = %.4f (uncooled %.4f)" % (eta_uncooled - deta_stage, eta_uncooled))