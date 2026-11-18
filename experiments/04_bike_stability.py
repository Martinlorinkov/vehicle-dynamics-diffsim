# motorcycle / bicycle self-stability with the linearized whipple model
#
#   fig 1  eigenvalues vs speed for the benchmark bicycle (matches the paper)
#   fig 2  same thing for the sport bike estimate
#   fig 3  how trail and rake move the self-stable speed range
#
# usage: python experiments/04_bike_stability.py

import math
import os
import sys

import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vdsim.params import bike_benchmark, bike_sport
from vdsim.whipple import eig_sweep, stable_speed_range, eigenvalues

torch.set_default_dtype(torch.float64)
FIG = os.path.join(os.path.dirname(__file__), "..", "figures")


def eig_plot(p, vmax, title, fname):
    speeds = torch.linspace(0.0, vmax, 500)
    ev = eig_sweep(p, speeds)
    v_w, v_c = stable_speed_range(p, vmax=vmax)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for k in range(4):
        ax.plot(speeds, ev[:, k].real, "C0.", ms=2)
        ax.plot(speeds, ev[:, k].imag.abs(), "C1.", ms=1)
    ax.plot([], [], "C0.", label="real part (growth rate)")
    ax.plot([], [], "C1.", label="|imag part| (oscillation, rad/s)")
    ax.axhline(0, color="k", lw=0.8)
    if v_w is not None and v_c is not None:
        ax.axvspan(v_w, v_c, color="g", alpha=0.1, label=f"self-stable {v_w:.2f}-{v_c:.2f} m/s")
    ax.set_ylim(-10, 10)
    ax.set_xlabel("forward speed (m/s)")
    ax.set_ylabel("eigenvalue (1/s)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fname), dpi=150)
    return v_w, v_c


v_w, v_c = eig_plot(bike_benchmark(), 10.0, "benchmark bicycle (Meijaard et al. 2007)", "04_eig_benchmark.png")
print(f"benchmark bicycle: weave speed {v_w:.4f} m/s, capsize speed {v_c:.4f} m/s  (paper: 4.2924, 6.0243)")

sp = bike_sport()
v_w, v_c = eig_plot(sp, 40.0, "sport bike estimate (whipple model, no tire slip)", "04_eig_sportbike.png")
if v_w is None or v_c is None:
    print(f"sport bike: weave {v_w}, capsize {v_c}  (no full stable window found)")
else:
    print(f"sport bike: weave speed {v_w:.2f} m/s ({v_w * 3.6:.0f} km/h), capsize speed {v_c:.2f} m/s ({v_c * 3.6:.0f} km/h)")

# capsize mode growth rate above the capsize speed: how slow the "fall" is.
# a big time constant means the rider barely notices it and corrects without thinking
for v in [20.0, 30.0]:
    ev = eigenvalues(sp, v)
    real_only = ev[ev.imag.abs() < 1e-6].real
    if real_only.numel() > 0 and real_only.max() > 0:
        print(f"  at {v * 3.6:.0f} km/h the capsize mode grows with time constant {1.0 / real_only.max().item():.1f} s")

# ---- trail / rake sweep for the sport bike
trails = torch.linspace(0.02, 0.16, 29)
rakes = torch.linspace(15, 35, 21)
weave = torch.full((len(rakes), len(trails)), float("nan"))
width = torch.full((len(rakes), len(trails)), float("nan"))
for i, rk in enumerate(rakes):
    for j, tr in enumerate(trails):
        q = dict(sp)
        q["c"] = tr.item()
        q["lam"] = math.radians(rk.item())
        vw, vc = stable_speed_range(q, vmax=60.0, n=240)
        if vw is not None:
            weave[i, j] = vw * 3.6
            if vc is not None:
                width[i, j] = (vc - vw) * 3.6
            else:
                width[i, j] = 60.0 * 3.6 - vw * 3.6

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
im0 = axes[0].pcolormesh(trails * 1000, rakes, weave, shading="auto", cmap="viridis")
fig.colorbar(im0, ax=axes[0], label="km/h")
axes[0].set_title("weave speed (bike starts to self-stabilize)")
im1 = axes[1].pcolormesh(trails * 1000, rakes, width, shading="auto", cmap="magma")
fig.colorbar(im1, ax=axes[1], label="km/h")
axes[1].set_title("width of self-stable speed window")
for ax in axes:
    ax.plot(sp["c"] * 1000, math.degrees(sp["lam"]), "w*", ms=14, mec="k")
    ax.set_xlabel("trail (mm)")
    ax.set_ylabel("rake / steer axis tilt (deg)")
fig.suptitle("sport bike geometry sweep (star = current estimate)")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "04_geometry_sweep.png"), dpi=150)
print("saved figures")
