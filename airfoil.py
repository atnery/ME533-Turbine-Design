import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
from matplotlib.widgets import Button
from datetime import datetime
import json

def airfoil(B1, B2, Bg, s, c, gamma, R1, dB1, R2, dB2, S2=None, S4=None, P2=None, P3=None):
    B1, B2, Bg, gamma = 90 - B1, 90 - B2, 90 - Bg, 90 - gamma # Convert to tangential angle convention
    B1, B2, Bg, gamma, dB1, dB2 = np.radians([B1, B2, Bg, gamma, dB1, dB2]) # Convert to radians
    
    bz = c * np.sin(gamma) + R1 * (1 - np.sin(B1)) + R2 * (1 - np.sin(B2)) # Axial chord
    H = c * np.cos(gamma) + R1 * np.cos(B1) - R2 * np.cos(B2) # Tangential height
    o = s * np.sin(Bg) # Throat

    # Defining points
    P1 = R1 * (1 + np.cos(B1 + 0.5 * dB1)), H - R1 * np.sin(B1 + 0.5 * dB1), np.tan(0.5 * np.pi - B1 - 0.5 * dB1)
    P4 = bz - R2 * (1 + np.cos(B2 + 0.5 * dB2)), -R2 * np.sin(B2 + 0.5 * dB2), np.tan(B2 + 0.5 * dB2 - 0.5 * np.pi)
    S1 = R1 * (1 - np.cos(B1 - 0.5 * dB1)), H + R1 * np.sin(B1 - 0.5 * dB1), np.tan(0.5 * np.pi - B1 + 0.5 * dB1)
    S5 = bz - R2 * (1 - np.cos(B2 - 0.5 * dB2)), R2 * np.sin(B2 - 0.5 * dB2), np.tan(B2 - 0.5 * dB2 - 0.5 * np.pi)

    # Pressure surface
    x_pressure, y_pressure = zip(*[p[:2] for p in [P1, P2, P3, P4] if p is not None])
    cubic_P = CubicSpline(x_pressure, y_pressure, bc_type=((1, P1[2]), (1, P4[2])))
    x_P = np.linspace(P1[0], P4[0], 200)
    y_P = cubic_P(x_P)

    # Suction surface
    Gam = np.radians(20)
    err = 1
    n = 1
    while err > 1e-6:
        S3 = P4[0] - o * np.cos(Gam + B2 - 0.5 * dB2), P4[1] + s - o * np.sin(Gam + B2 - 0.5 * dB2), np.tan(Gam + B2 - 0.5 * dB2 - 0.5 * np.pi)
        
        x_suction, y_suction = zip(*[p[:2] for p in [S1, S2, S3, S4, S5] if p is not None])
        cubic_S = CubicSpline(x_suction, y_suction, bc_type=((1, S1[2]), (1, S5[2])))
        x_S = np.linspace(S1[0], S5[0], 100)
        y_S = cubic_S(x_S)

        width = lambda z: np.hypot(z - P4[0], cubic_S(z) - P4[1] - s)
        i = width(x_S).argmin()
        x_t = minimize_scalar(width, bounds=(x_S[max(i - 1, 0)], x_S[min(i + 1, len(x_S) - 1)]), method="bounded").x
        Gam_act = np.arctan(cubic_S(x_t, 1)) + 0.5 * np.pi - B2 + 0.5 * dB2

        err = abs(Gam_act - Gam)
        Gam = Gam_act if Gam_act >= 0 else 0.5 * Gam
        Gam = min(Gam, 0.475 * np.pi - B2 + 0.5 * dB2)
        
        n += 1
        if n > 100:
            raise RuntimeError("Failed to converge on uncovered turning in less than 100 iterations.")

    # Leading edge radius
    phi_LE = np.linspace(B1 + 0.5 * dB1 - np.pi, B1 - 0.5 * dB1, 50)
    x_LE = R1 * (1 - np.cos(phi_LE))
    y_LE = H + R1 * np.sin(phi_LE)

    # Trailing edge radius
    phi_TE = np.linspace(B2 + 0.5 * dB2 - np.pi, B2 - 0.5 * dB2, 50)
    x_TE = bz - R2 * (1 - np.cos(phi_TE))
    y_TE = R2 * np.sin(phi_TE)

    # Throat location and width
    throat_S = (x_t, cubic_S(x_t))
    throat_P = (P4[0], P4[1] + s)      
    o_act = width(x_t)

    # Passage width
    x_f = np.linspace(S1[0], S5[0], 2000)
    d = np.hypot(x_P[:, None] - x_f[None, :], y_P[:, None] + s - cubic_S(x_f)[None, :])
    passage_width = d.min(axis=1)

    return x_S, y_S, x_P, y_P, x_LE, y_LE, x_TE, y_TE, S3, throat_S, throat_P, o, passage_width

B1, B2 = 63.1207, 74.1048
Bg = 72.796
s, c = 1.0035, 1.90741
gamma = 20
R1, dB1 = 0.05, 15
R2, dB2 = 0.015, 8
pts = {"S2": [0.7665, 1.8885], "S4": [1.7436, 0.2564], "P2": [0.7983, 1.2062]} # "P3": [1.5, 0.5],  

x_S, y_S, x_P, y_P, x_LE, y_LE, x_TE, y_TE, S3, throat_S, throat_P, o, passage_width = airfoil(B1, B2, Bg, s, c, gamma, R1, dB1, R2, dB2, **pts)
    
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.subplots_adjust(left=0.25)

lines = [ax1.plot(x_S, y_S, "tab:blue")[0], ax1.plot(x_S, y_S + s, "tab:blue")[0],
         ax1.plot(x_P, y_P, "tab:red")[0], ax1.plot(x_P, y_P + s, "tab:red")[0],
         ax1.plot(x_LE, y_LE, "tab:purple")[0], ax1.plot(x_LE, y_LE + s, "tab:purple")[0],
         ax1.plot(x_TE, y_TE, "tab:purple")[0], ax1.plot(x_TE, y_TE + s, "tab:purple")[0],
         ax1.plot([throat_S[0], throat_P[0]], [throat_S[1], throat_P[1]], "tab:green", linestyle="--")[0]]
markers = {n: ax1.plot(*p, "ko", ms=7, picker=True)[0] for n, p in pts.items()}
labels = {n: ax1.annotate(n, p, xytext=(5, 5), textcoords="offset points") for n, p in pts.items()}
ax1.set_aspect("equal")
ax1.grid()
ax1.set_title("Blade airfoil")
ax1.set_xlabel("Axial coordinate [in]")
ax1.set_ylabel("Tangential coordinate [in]")

width_line = ax2.plot(x_P, passage_width)
ax2.set_title("Passage width")
ax2.set_xlabel("Axial coordinate [in]")
ax2.set_ylabel("Passage width [in]")
ax2.grid()

dragging = None

def update(text=None):
    try:
        vals = {arg: float(box.text) for arg, box in text_boxes.items()}
    except ValueError:
        ax1.set_title("Not a number")
        fig.canvas.draw_idle() 
        return
    
    try:
        x_S, y_S, x_P, y_P, x_LE, y_LE, x_TE, y_TE, S3, throat_S, throat_P, o, passage_width = airfoil(**vals, **pts)
    except ValueError:
        ax1.set_title("Invalid: points out of order") 
        fig.canvas.draw_idle()
    except RuntimeError:
        ax1.set_title("Invalid: failed to converge on uncovered turning")
        return

    s = vals["s"]
    lines[0].set_data(x_S, y_S)
    lines[1].set_data(x_S, y_S + s)
    lines[2].set_data(x_P, y_P)
    lines[3].set_data(x_P, y_P + s)
    lines[4].set_data(x_LE, y_LE)
    lines[5].set_data(x_LE, y_LE + s)
    lines[6].set_data(x_TE, y_TE)
    lines[7].set_data(x_TE, y_TE + s)
    lines[8].set_data([throat_S[0], throat_P[0]], [throat_S[1], throat_P[1]])

    ax1.set_title("Blade airfoil")
    #ax1.relim()
    #ax1.autoscale_view()

    width_line[0].set_data(x_P, passage_width)
    ax2.relim()
    ax2.autoscale_view()

    fig.canvas.draw_idle()

def on_press(event):
    global dragging
    
    if event.inaxes is not ax1:
        return
    
    for name, m in markers.items():
        if m.contains(event)[0]:
            dragging = name
            return

def on_move(event):
    global last_update

    if dragging is None or event.inaxes is not ax1:
        return
    
    pts[dragging] = [event.xdata, event.ydata]
    markers[dragging].set_data([event.xdata], [event.ydata])
    labels[dragging].xy = (event.xdata, event.ydata)

    update()

def on_release(event):
    global dragging
    if dragging is not None:
        sync_point_boxes(dragging)
        update()
    dragging = None

def set_point(name, k, text):
    try:
        pts[name][k] = float(text)
    except ValueError:
        ax1.set_title("Not a number")
        fig.canvas.draw_idle()
        return
    markers[name].set_data([pts[name][0]], [pts[name][1]])
    labels[name].xy = tuple(pts[name])
    update()

def sync_point_boxes(name):
    for k, box in enumerate(point_boxes[name]):
        box.eventson = False
        box.set_val(f"{pts[name][k]:.4f}")
        box.eventson = True

def save_inputs(event):
    design = {
        "Inputs": {arg: float(box.text) for arg, box in text_boxes.items()},
        "Points": {name: [float(x), float(y)] for name, (x, y) in pts.items()}
    }
    name = f"blade_design_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(name, "w") as f:
        json.dump(design, f, indent=2)

    ax1.set_title(f"Design saved to {name}")
    fig.canvas.draw_idle()




inputs = {"B1": ("Inlet angle [deg]    ", B1),
          "B2": ("Exit angle [deg]    ", B2),
          "Bg": ("Gauging angle [deg]    ", Bg),
          "s": ("Pitch [in]    ", s),
          "c": ("Chord [in]    ", c),
          "gamma": ("Stagger angle [deg]    ", gamma),
          "R1": ("LE radius [in]    ", R1),
          "dB1": ("LE wedge [deg]   ", dB1),
          "R2": ("TE radius [in]    ", R2),
          "dB2": ("TE wedge [deg]   ", dB2)}

top, h, gap = 0.9, 0.04, 0.015
text_boxes = {}
for n, (arg, (label, val)) in enumerate(inputs.items()):
    bottom = top - n * (h + gap)
    text_boxes[arg] = TextBox(fig.add_axes([0.1, bottom, 0.04, h]), label, initial=str(val))
    text_boxes[arg].on_submit(update)

point_boxes = {}
row_top = bottom - 2 * (h + gap)
fig.text(0.12, row_top + h + 0.01, "x", ha="center")
fig.text(0.17, row_top + h + 0.01, "y", ha="center")

for n, (name, (x, y)) in enumerate(pts.items()):
    b = row_top - n * (h + gap)
    box_x = TextBox(fig.add_axes([0.10, b, 0.04, h]), name + "    ", initial=f"{x:.4f}")
    box_y = TextBox(fig.add_axes([0.15, b, 0.04, h]), "", initial=f"{y:.4f}")
    box_x.on_submit(lambda text, name=name: set_point(name, 0, text))
    box_y.on_submit(lambda text, name=name: set_point(name, 1, text))
    point_boxes[name] = (box_x, box_y)

save_button = Button(fig.add_axes([0.1, 0.05, 0.04, 0.04]), "Save")
save_button.on_clicked(save_inputs)

fig.canvas.mpl_connect("button_press_event", on_press)
fig.canvas.mpl_connect("motion_notify_event", on_move)
fig.canvas.mpl_connect("button_release_event", on_release)

plt.show()