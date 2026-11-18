# racing line optimization by gradient descent on lap time
#
# the line is the centerline pushed sideways by an offset at every point. lap time
# comes out of the QSS lap sim, which is differentiable, so d(lap time)/d(offset)
# says which way to move every point of the line. adam does the rest.
#
# runs it for the car and for the bike on the same track and compares them.
#
# usage: python experiments/05_racing_line.py [track] [iters]
#        python experiments/05_racing_line.py Nuerburgring 300

import os
import sys
import time

import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vdsim.track import load_track, path_points, curvature, segment_lengths, smooth_periodic
from vdsim.lapsim import qss_car, qss_bike, speed_profile, lap_time, accelerations

torch.set_default_dtype(torch.float64)
FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
DATA = os.path.join(os.path.dirname(__file__), "..", "data")

TRACK  = sys.argv[1] if len(sys.argv) > 1 else "Nuerburgring"
ITERS  = int(sys.argv[2]) if len(sys.argv) > 2 else 300
DS     = 6.0          # m between points
SIGMA  = 3.0          # smoothing on the line, in points (3 * 6 m = 18 m)
SMOOTH = 2.0          # extra penalty on wiggly lines, the curvature estimate gets noisy without it
MARGIN = 0.3          # m kept from the track edge
LR     = 0.05

path = os.path.join(DATA, TRACK + ".csv")
if not os.path.exists(path):
    print(f"no {path}, run: python data/get_tracks.py {TRACK}")
    sys.exit(1)

track = load_track(path, ds=DS)
n = len(track["x"])
print(f"{TRACK}: {track['length'] / 1000:.2f} km, {n} points")


def evaluate(offset, veh):
    x, y = path_points(track, offset)
    k = curvature(x, y)
    ds = segment_lengths(x, y)
    v = speed_profile(k, ds, veh)
    return lap_time(v, ds), v, k, ds, x, y


def optimize(veh):
    # the width data jumps around a bit point to point, smooth it so the
    # "middle of the track" line isn't wiggly to begin with
    lo = smooth_periodic(-torch.as_tensor(track["w_right"]), SIGMA) + veh["half_width"] + MARGIN
    hi = smooth_periodic(torch.as_tensor(track["w_left"]), SIGMA) - veh["half_width"] - MARGIN
    # sigmoid keeps the line inside the track no matter what adam does, and the
    # blur on z means adam can't move single points on their own (that just
    # makes curvature noise, which the lap sim reads as tiny corners)
    z = torch.zeros(n, requires_grad=True)
    opt = torch.optim.Adam([z], lr=LR)

    with torch.no_grad():
        mid = lo + (hi - lo) * 0.5
        t_center = evaluate(mid, veh)[0].item()
    print(f"\n{veh['name']}: centerline lap {t_center:.2f} s")

    hist = []
    t0 = time.time()
    for it in range(ITERS):
        offset = lo + (hi - lo) * torch.sigmoid(smooth_periodic(z, SIGMA))
        T, v, k, ds, x, y = evaluate(offset, veh)
        wiggle = torch.sum((torch.roll(offset, -1) - 2 * offset + torch.roll(offset, 1)) ** 2)
        loss = T + SMOOTH * wiggle / n
        opt.zero_grad()
        loss.backward()
        opt.step()
        hist.append(T.item())
        if it % 25 == 0 or it == ITERS - 1:
            print(f"  it {it:4d}  lap {T.item():7.2f} s   ({time.time() - t0:.0f} s)")

    with torch.no_grad():
        offset = lo + (hi - lo) * torch.sigmoid(smooth_periodic(z, SIGMA))
        T, v, k, ds, x, y = evaluate(offset, veh)
    return {"veh": veh, "offset": offset, "T": T.item(), "T_center": t_center, "v": v,
            "k": k, "ds": ds, "x": x, "y": y, "hist": hist}


def fmt(t):
    return f"{int(t // 60)}:{t % 60:06.3f}"


res = [optimize(qss_car()), optimize(qss_bike())]

print("\nresults")
for r in res:
    gain = r["T_center"] - r["T"]
    vmax = r["v"].max().item() * 3.6
    vmin = r["v"].min().item() * 3.6
    print(f"  {r['veh']['name']:28s} centerline {fmt(r['T_center'])}   optimized {fmt(r['T'])}   "
          f"(-{gain:.2f} s)   speed {vmin:.0f}-{vmax:.0f} km/h")

# ---- fig 1, full track with both lines colored by speed
fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
lx = track["x"] + track["w_left"] * track["nx"]
ly = track["y"] + track["w_left"] * track["ny"]
rx = track["x"] - track["w_right"] * track["nx"]
ry = track["y"] - track["w_right"] * track["ny"]
for ax, r in zip(axes, res):
    ax.plot(lx, ly, "k", lw=0.6)
    ax.plot(rx, ry, "k", lw=0.6)
    sc = ax.scatter(r["x"], r["y"], c=r["v"] * 3.6, s=3, cmap="turbo")
    fig.colorbar(sc, ax=ax, label="speed (km/h)", shrink=0.8)
    ax.set_aspect("equal")
    ax.set_title(f"{r['veh']['name']}  {fmt(r['T'])}")
    ax.axis("off")
fig.suptitle(f"{TRACK}: optimized racing lines")
fig.tight_layout()
fig.savefig(os.path.join(FIG, f"05_{TRACK}_lines.png"), dpi=150)

# ---- fig 2, speed traces
fig, ax = plt.subplots(figsize=(12, 4))
for r in res:
    s = torch.cumsum(r["ds"], 0) - r["ds"]
    ax.plot(s / 1000, r["v"] * 3.6, label=r["veh"]["name"])
ax.set_xlabel("distance (km)")
ax.set_ylabel("speed (km/h)")
ax.grid(alpha=0.3)
ax.legend()
ax.set_title(f"{TRACK}: speed over one lap")
fig.tight_layout()
fig.savefig(os.path.join(FIG, f"05_{TRACK}_speed.png"), dpi=150)

# ---- fig 3, g-g diagrams + where the two lines differ most
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
for r, c in zip(res, ["C0", "C1"]):
    ax_, ay_ = accelerations(r["v"], r["k"], r["ds"])
    axes[0].plot(ay_ / 9.81, ax_ / 9.81, ".", ms=2, color=c, label=r["veh"]["name"])
axes[0].set_xlabel("lateral (g)")
axes[0].set_ylabel("longitudinal (g)")
axes[0].set_title("g-g diagram")
axes[0].set_aspect("equal")
axes[0].grid(alpha=0.3)
axes[0].legend(markerscale=5)

diff = (res[0]["offset"] - res[1]["offset"]).abs()
i = int(torch.argmax(diff).item())
lo_i, hi_i = max(0, i - 40), min(n, i + 40)
ax = axes[1]
ax.plot(lx[lo_i:hi_i], ly[lo_i:hi_i], "k", lw=0.8)
ax.plot(rx[lo_i:hi_i], ry[lo_i:hi_i], "k", lw=0.8)
for r, c in zip(res, ["C0", "C1"]):
    ax.plot(r["x"][lo_i:hi_i], r["y"][lo_i:hi_i], color=c, lw=2, label=r["veh"]["name"])
ax.set_aspect("equal")
ax.set_title("corner where the lines differ the most")
ax.legend(fontsize=8)
ax.axis("off")

for r, c in zip(res, ["C0", "C1"]):
    axes[2].plot(r["hist"], color=c, label=r["veh"]["name"])
axes[2].set_xlabel("iteration")
axes[2].set_ylabel("lap time (s)")
axes[2].set_title("optimizer progress")
axes[2].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG, f"05_{TRACK}_analysis.png"), dpi=150)
print("saved figures")
