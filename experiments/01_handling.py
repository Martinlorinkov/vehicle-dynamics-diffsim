# milestone 1: does the car model behave like a car?
#
#   fig 1  tire curves (pacejka vs the linear approximation)
#   fig 2  steady state yaw rate gain vs speed, sim vs closed form theory
#   fig 3  handling diagram from a slow ramp steer at constant speed
#
# usage: python experiments/01_handling.py

import math
import os
import sys

import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vdsim.params import car_m340i, G
from vdsim.car import dynamic_derivs, pacejka_tires, rollout, steady_yaw_gain, lateral_accel
from vdsim.tires import pacejka, pacejka_cornering_stiffness, peak_slip_angle

torch.set_default_dtype(torch.float64)
FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
DT = 0.01

p = car_m340i()
tire_f, tire_r = pacejka_tires(p)

Fzf0 = p["m"] * G * p["b"] / p["L"]
Fzr0 = p["m"] * G * p["a"] / p["L"]
Cf = pacejka_cornering_stiffness(Fzf0, p["mu"], *p["tire_f"][:2])
Cr = pacejka_cornering_stiffness(Fzr0, p["mu"], *p["tire_r"][:2])
K = p["m"] / p["L"] * (p["b"] / Cf - p["a"] / Cr)

print(f"static axle loads   front {Fzf0:.0f} N   rear {Fzr0:.0f} N")
print(f"cornering stiffness front {Cf/1000:.1f} kN/rad   rear {Cr/1000:.1f} kN/rad")
print(f"understeer gradient {math.degrees(K * G):.2f} deg/g")
print(f"characteristic speed {math.sqrt(p['L'] / K) * 3.6:.0f} km/h (yaw gain peaks here)")
print(f"peak slip angle     front {math.degrees(peak_slip_angle(*p['tire_f'])):.1f} deg   rear {math.degrees(peak_slip_angle(*p['tire_r'])):.1f} deg")


def speed_hold(v_target):
    def f(x, u):
        drag = 0.5 * 1.225 * p["CdA"] * x[..., 3] ** 2 + p["Crr"] * p["m"] * G
        Fx = 3000.0 * (v_target - x[..., 3]) + drag
        uu = torch.stack([u[..., 0], Fx], dim=-1)
        return dynamic_derivs(x, uu, p, tire_f, tire_r)
    return f


# ---- fig 1, tire curves
a = torch.linspace(-0.35, 0.35, 701)
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(torch.rad2deg(a), pacejka(a, 1.0, p["mu"], *p["tire_f"]), label="front (pacejka)")
ax.plot(torch.rad2deg(a), pacejka(a, 1.0, p["mu"], *p["tire_r"]), label="rear (pacejka)")
ax.plot(torch.rad2deg(a), p["tire_f"][0] * p["tire_f"][1] * a, "k--", lw=1, label="front, linear")
ax.set_ylim(-1.3, 1.3)
ax.set_xlabel("slip angle (deg)")
ax.set_ylabel("Fy / Fz")
ax.set_title("axle tire curves")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIG, "01_tire_curves.png"), dpi=150)

# ---- fig 2, yaw gain vs speed
# two sim runs:
#   tiny steer (0.05 deg) with drag turned off -> should match linear theory exactly
#   0.5 deg steer with drag on -> what the real car does, the gap from theory is
#   tire nonlinearity + load transfer from the drag force (more on that in the readme)
speeds = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]


def yaw_gain_run(delta, with_drag):
    q = dict(p)
    if not with_drag:
        q["CdA"], q["Crr"] = 0.0, 0.0
    tf, tr = pacejka_tires(q)
    out = []
    for v in speeds:
        def f(x, u):
            drag = 0.5 * 1.225 * q["CdA"] * x[..., 3] ** 2 + q["Crr"] * q["m"] * G
            Fx = 3000.0 * (v - x[..., 3]) + drag
            return dynamic_derivs(x, torch.stack([u[..., 0], Fx], dim=-1), q, tf, tr)
        x0 = torch.tensor([0, 0, 0, float(v), 0, 0])
        us = torch.tensor([[delta, 0.0]]).repeat(1200, 1)
        xs = rollout(f, x0, us, DT)
        out.append(xs[-1, 5].item() / delta)
    return out


gain_lin = yaw_gain_run(math.radians(0.05), False)
gain_real = yaw_gain_run(math.radians(0.5), True)

vv = torch.linspace(1, 55, 200)
theory = [steady_yaw_gain(p, Cf, Cr, v.item()) for v in vv]
neutral = [v.item() / p["L"] for v in vv]

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(vv * 3.6, theory, label="linear theory (understeer gradient)")
ax.plot(vv * 3.6, neutral, "k:", lw=1, label="neutral steer (K = 0)")
ax.plot([s * 3.6 for s in speeds], gain_lin, "o", label="sim, 0.05 deg steer, no drag")
ax.plot([s * 3.6 for s in speeds], gain_real, "s", mfc="none", label="sim, 0.5 deg steer, with drag")
ax.set_ylim(0, 12)
ax.set_xlabel("speed (km/h)")
ax.set_ylabel("yaw rate / steer angle (1/s)")
ax.set_title("steady state yaw rate gain")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIG, "01_yaw_gain.png"), dpi=150)

for name, gains in [("tiny steer, no drag", gain_lin), ("0.5 deg, with drag", gain_real)]:
    errs = [(g - steady_yaw_gain(p, Cf, Cr, float(v))) / steady_yaw_gain(p, Cf, Cr, float(v)) for g, v in zip(gains, speeds)]
    print(f"yaw gain vs theory ({name}): worst {100 * max(abs(e) for e in errs):.2f}%, at 180 km/h {100 * errs[-1]:+.2f}%")

# ---- fig 3, ramp steer at 25 m/s (90 km/h), 0 -> 10 deg over 40 s
v = 25.0
T = 4000
t = torch.arange(T) * DT
steer = torch.deg2rad(torch.linspace(0, 10, T))
us = torch.stack([steer, torch.zeros(T)], dim=-1)
x0 = torch.tensor([0, 0, 0, v, 0, 0])
f = speed_hold(v)
xs = rollout(f, x0, us, DT)[:-1]
d = f(xs, us)
ay = lateral_accel(xs, d)

# understeer angle = steer minus what a neutral car would need for the same path
radius_term = p["L"] * xs[:, 5] / xs[:, 3]
us_angle = torch.rad2deg(steer - radius_term)
ok = ay > 0.05 * G

lin = torch.linspace(0, 1.0, 50)
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot((ay / G)[ok], us_angle[ok], label="simulation, ramp steer")
ax.plot(lin, math.degrees(K * G) * lin, "k--", lw=1, label=f"linear, K = {math.degrees(K * G):.2f} deg/g")
ax.set_xlabel("lateral acceleration (g)")
ax.set_ylabel("understeer angle  delta - L/R  (deg)")
ax.set_title("handling diagram at 90 km/h")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIG, "01_handling_diagram.png"), dpi=150)

print(f"max steady lateral accel in ramp steer: {(ay.max() / G).item():.2f} g")
idx = torch.argmax(ay).item()
print(f"  reached at {math.degrees(steer[idx].item()):.1f} deg steer, after that more steer = less grip (front washes out)")
