import numpy as np

R_AIR = 287.05


def station_span(r_hub, r_tip, r_ref, Cx, Ct_ref, T0, P0, cp, gamma, R, n):
    # Free vortex at one station: r*Ct = const, Cx, T0, P0 constant along the span
    r = np.zeros(n)
    Ct = np.zeros(n)
    C = np.zeros(n)
    alpha = np.zeros(n)
    T = np.zeros(n)
    P = np.zeros(n)
    a = np.zeros(n)
    M = np.zeros(n)

    for i in range(n):
        r[i] = r_hub + (r_tip - r_hub) * i / (n - 1)
        Ct[i] = Ct_ref * r_ref / r[i]
        C[i] = np.sqrt(Cx**2 + Ct[i]**2)
        alpha[i] = np.degrees(np.arctan(Ct[i] / Cx))
        T[i] = T0 - C[i]**2 / (2 * cp)
        if T[i] <= 0:
            raise ValueError("Negative static temperature at r = " + str(r[i]))
        
        P[i] = P0 * (T[i] / T0) ** (gamma / (gamma - 1))
        a[i] = np.sqrt(gamma * R * T[i])
        M[i] = C[i] / a[i]

    return dict(r=r, Cx=Cx, Ct=Ct, C=C, alpha=alpha, T=T, P=P, a=a, M=M)


def blade_row(name, st_in, st_out, omega, n):
    # Velocity triangles at each span point; omega = 0 for a stator
    ang_in = np.zeros(n)
    ang_out = np.zeros(n)
    M_in = np.zeros(n)
    M_out = np.zeros(n)
    work = np.zeros(n)
    dh = np.zeros(n)

    for i in range(n):
        U_in = omega * st_in['r'][i]
        U_out = omega * st_out['r'][i]
        Wt_in = st_in['Ct'][i] - U_in
        Wt_out = st_out['Ct'][i] - U_out

        if omega == 0:
            ang_in[i] = st_in['alpha'][i]
            ang_out[i] = st_out['alpha'][i]
            M_in[i] = st_in['M'][i]
            M_out[i] = st_out['M'][i]
        else:
            ang_in[i] = np.degrees(np.arctan(Wt_in / st_in['Cx']))
            ang_out[i] = np.degrees(np.arctan(Wt_out / st_out['Cx']))
            M_in[i] = np.sqrt(st_in['Cx']**2 + Wt_in**2) / st_in['a'][i]
            M_out[i] = np.sqrt(st_out['Cx']**2 + Wt_out**2) / st_out['a'][i]

        work[i] = U_in * st_in['Ct'][i] - U_out * st_out['Ct'][i]  # Euler work
        dh[i] = work[i] + (st_out['C'][i]**2 - st_in['C'][i]**2) / 2  # static enthalpy drop

    return dict(name=name, omega=omega, ang_in=ang_in, ang_out=ang_out,
                M_in=M_in, M_out=M_out, work=work, dh=dh)


def finish(stations, rows, n):
    # Static enthalpy drop fraction per row 
    for i in range(n):
        total = 0
        for row in rows:
            total = total + row['dh'][i]
        for row in rows:
            if i == 0:
                row['dh_frac'] = np.zeros(n)
            row['dh_frac'][i] = row['dh'][i] / total

    for row in rows:
        if row['omega'] != 0 and row['dh_frac'][0] < 0:
            print("WARNING: " + row['name'] + " has negative reaction at the hub")
        if max(row['M_in']) > 1 or max(row['M_out']) > 1:
            print("WARNING: " + row['name'] + " is supersonic somewhere along the span")

    return dict(stations=stations, rows=rows)


def single_stage_RE(sol, r_hub, P01, n=11):
    # sol = last entry returned by single_stage_turbine
    if len(sol) == 0:
        raise ValueError("single_stage_turbine did not converge")

    g = np.log(sol['P02'] / sol['P2']) / np.log(sol['T02'] / sol['T2'])  # gamma/(gamma-1)
    gamma = g / (g - 1)

    V2 = sol['M2'] * np.sqrt(gamma * R_AIR * sol['T2'])
    Vx = V2 * np.cos(np.radians(sol['alpha2']))
    V2t = V2 * np.sin(np.radians(sol['alpha2']))
    V3 = sol['M3'] * np.sqrt(gamma * R_AIR * sol['T3'])
    V3t = Vx * np.tan(np.radians(sol['alpha3']))

    cp1 = gamma * R_AIR / (gamma - 1)
    cp2 = V2**2 / (2 * (sol['T02'] - sol['T2']))
    cp3 = V3**2 / (2 * (sol['T03'] - sol['T3']))

    r_tip1 = np.sqrt(sol['A1'] / np.pi + r_hub**2)
    r_tip2 = np.sqrt(sol['A2'] / np.pi + r_hub**2)
    r_tip3 = np.sqrt(sol['A3'] / np.pi + r_hub**2)
    r_ref = sol['r_mean3']  # radius where the meanline U is defined
    omega = sol['N_rpm'] * 2 * np.pi / 60

    st1 = station_span(r_hub, r_tip1, r_ref, Vx, 0.0, sol['T02'], P01, cp1, gamma, R_AIR, n)
    st2 = station_span(r_hub, r_tip2, r_ref, Vx, V2t, sol['T02'], sol['P02'], cp2, gamma, R_AIR, n)
    st3 = station_span(r_hub, r_tip3, r_ref, Vx, V3t, sol['T03'], sol['P03'], cp3, gamma, R_AIR, n)

    stator = blade_row('stator', st1, st2, 0.0, n)
    rotor = blade_row('rotor', st2, st3, omega, n)
    return finish([st1, st2, st3], [stator, rotor], n)


def two_stage_contra_RE(res, n=11):
    # res = dict returned by two_stage_contra
    st = res['report']['stations']
    r = res['r']

    stations = []
    for k in range(4):
        T0 = st['T0'][k]
        T = st['T'][k]
        P0 = st['P0'][k]
        P = st['P'][k]
        g = np.log(P0 / P) / np.log(T0 / T)
        gamma = g / (g - 1)
        cp = st['C'][k]**2 / (2 * (T0 - T))
        R = P / (st['rho'][k] * T)
        h = st['h'][k]
        stations.append(station_span(r - h / 2, r + h / 2, r, st['Cx'][k], st['Ctheta'][k],
                                     T0, P0, cp, gamma, R, n))

    stator = blade_row('stator', stations[0], stations[1], 0.0, n)
    rotor1 = blade_row('rotor1', stations[1], stations[2], res['U1'] / r, n)
    rotor2 = blade_row('rotor2', stations[2], stations[3], res['U2'] / r, n)
    return finish(stations, [stator, rotor1, rotor2], n)


def print_RE(out):
    n = len(out['stations'][0]['r'])
    for row in out['rows']:
        print("\n" + row['name'] + "   work = " + str(round(row['work'][0] / 1000, 2)) + " kJ/kg")
        print("span   ang_in   ang_out   M_in    M_out   dh_frac")
        for i in range(n):
            print("%4.2f  %7.2f  %8.2f  %6.3f  %6.3f  %7.3f" % (
                i / (n - 1), row['ang_in'][i], row['ang_out'][i],
                row['M_in'][i], row['M_out'][i], row['dh_frac'][i]))


def plot_RE(out):
    import matplotlib.pyplot as plt
    n = len(out['stations'][0]['r'])
    span = np.zeros(n)
    for i in range(n):
        span[i] = i / (n - 1)

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    for row in out['rows']:
        ax[0].plot(row['ang_in'], span, label=row['name'] + ' in')
        ax[0].plot(row['ang_out'], span, '--', label=row['name'] + ' out')
        ax[1].plot(row['M_in'], span, label=row['name'] + ' in')
        ax[1].plot(row['M_out'], span, '--', label=row['name'] + ' out')
        ax[2].plot(row['dh_frac'], span, label=row['name'])
    ax[0].set_xlabel('Flow angle [deg] (relative for rotors)')
    ax[1].set_xlabel('Mach (relative for rotors)')
    ax[2].set_xlabel('Static enthalpy drop fraction')
    ax[0].set_ylabel('Span (0 = hub)')
    for k in range(3):
        ax[k].grid()
        ax[k].legend(fontsize=7)
    plt.tight_layout()
    plt.show()