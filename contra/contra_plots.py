import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROW_NAMES = ("Stator", "Rotor 1", "Rotor 2")
ROW_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")
ROW_MARKERS = ("o", "s", "^")
GRAY = "#b4b2a9"


def _style(ax):
    ax.grid(True, color="#e6e5e0", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#888780")
    ax.tick_params(colors="#5f5e5a", labelsize=9)


M2IN = 1 / 0.0254
ROWS = ("Stator", "Rotor 1", "Rotor 2")


def _perf_panels(m_out_max, m_in_max, an2_max, u_max=None):
    st = lambda o: o["report"]["stage"]
    sn = lambda o: o["report"]["stations"]
    return [
        ("Total-to-total efficiency", lambda o: st(o)["eta_tt"], None),
        ("Total-to-static efficiency", lambda o: o["eta_ts"], None),
        ("Specific work [kJ/kg]", lambda o: (st(o)["dh0_r1"] + st(o)["dh0_r2"]) / 1e3, None),
        ("Max rotor tip speed [m/s]", lambda o: max(o["U_tip"]), u_max),
        ("Stator exit blade height h2 [in]", lambda o: sn(o)["h"][1] * M2IN, None),
        ("Rotor 2 exit blade height h4 [in]", lambda o: sn(o)["h"][3] * M2IN, None),
        ("Mean diameter [in]", lambda o: 2 * st(o)["r"] * M2IN, None),
        ("Max AN² [m²·rpm²]", lambda o: max(o["AN2_r1"], o["AN2_r2"]), an2_max),
        ("Stator exit Mach", lambda o: o["M_out"][0], m_out_max),
        ("Rotor 1 exit relative Mach", lambda o: o["M_out"][1], m_out_max),
        ("Rotor 2 exit relative Mach", lambda o: o["M_out"][2], m_out_max),
        ("Rotor 1 inlet relative Mach", lambda o: o["M_in"][1], m_in_max),
    ]


def _loss_panels():
    Y = lambda o, row, col: o["report"]["rows"]["Y"][row][col]  # cols: total, profile, secondary, clearance, TE, shock
    panels = []
    for row, name in enumerate(ROWS):
        for col, comp in ((0, "total Y"), (1, "profile (incl. shock)"), (2, "secondary"), (3, "tip clearance"), (4, "trailing edge")):
            if row == 0 and col == 3:
                panels.append(None)  # stator has no tip clearance loss
                continue
            panels.append((f"{name}: {comp}", lambda o, r=row, c=col: Y(o, r, c), None))
    return panels


def _line_figure(records, xi, xlabel, x_best, panels, shape, title, fname, r1_values=None, x_opt=None):
    """One line per R1 vs variable xi. At each (x, R1) the best valid design is plotted (filled marker);
    if none is valid, the best converged design is plotted with a hollow marker."""
    X = np.array([r[0] for r in records], dtype=float)
    out = [r[1] for r in records]
    eta = np.array([o["eta_ts"] for o in out])
    valid = np.array([r[2] for r in records])
    xs = np.unique(X[:, xi])
    R1s = np.unique(X[:, 2]) if r1_values is None else np.array(r1_values, dtype=float)

    fig, axes = plt.subplots(*shape, figsize=(4.5 * shape[1], 3.6 * shape[0] + 0.6), constrained_layout=True)
    k = 0
    for R1 in R1s:
        pick = []  # (x, index, is_valid)
        for xv in xs:
            m = (X[:, xi] == xv) & np.isclose(X[:, 2], R1)
            for ok in (True, False):
                idx = np.where(m & (valid == ok))[0]
                if len(idx):
                    pick.append((xv, idx[np.argmax(eta[idx])], ok))
                    break
        if not pick:
            continue
        color = f"C{k}"
        k += 1
        x = [p for p, _, _ in pick]
        for ax, panel in zip(axes.flat, panels):
            if panel is None:
                continue
            f = panel[1]
            y = [f(out[i]) for _, i, _ in pick]
            ax.plot(x, y, color=color, linewidth=1.5, label=f"reaction = {R1:g}")
            ax.scatter(x, y, s=16, zorder=3, color=[color if ok else "white" for _, _, ok in pick],
                       edgecolors=color, linewidths=1.2)
    for ax, panel in zip(axes.flat, panels):
        if panel is None:
            ax.set_visible(False)
            continue
        name, _, lim = panel
        if lim is not None:
            ax.axhline(lim, color="k", linewidth=1, linestyle="--", alpha=0.6)
        ax.axvline(x_best, color="k", linewidth=1, linestyle=":", alpha=0.6)
        if x_opt is not None:
            ax.axvline(x_opt, color="k", linewidth=1, linestyle="-.", alpha=0.6)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.set_title(name)
        ax.set_xlabel(xlabel)
        ax.grid(True, alpha=0.3)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color="gray", linewidth=0, marker="o", markerfacecolor="white"))
    labels.append("outside limits")
    handles.append(plt.Line2D([], [], color="k", linewidth=1, linestyle=":", alpha=0.6))
    labels.append("baseline")
    if x_opt is not None:
        handles.append(plt.Line2D([], [], color="k", linewidth=1, linestyle="-.", alpha=0.6))
        labels.append("sweep optimum")
    fig.legend(handles, labels, loc="outside lower center", ncol=len(labels))
    fig.suptitle(title, fontsize=13)
    fig.savefig(fname, dpi=150)


def plot_vs(records, xi, xlabel, x_best, m_out_max, m_in_max, an2_max, title="", tag="phi", r1_values=None, x_opt=None, u_max=None):
    """Performance and loss figures vs swept variable xi (index into the record x tuple)."""
    _line_figure(records, xi, xlabel, x_best, _perf_panels(m_out_max, m_in_max, an2_max, u_max), (3, 4),
                 title, f"sweep_vs_{tag}_performance.png", r1_values, x_opt)
    _line_figure(records, xi, xlabel, x_best, _loss_panels(), (3, 5),
                 title, f"sweep_vs_{tag}_losses.png", r1_values, x_opt)