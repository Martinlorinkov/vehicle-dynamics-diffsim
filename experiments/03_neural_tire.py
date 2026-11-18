# learning the tire curve from driving data, with NO force measurements
#
# the pacejka tires get swapped for two small neural nets (front and rear axle).
# the only training signal is yaw rate + lateral accel from the trajectory, the
# gradient has to flow back through the whole rollout into the nets. if it works,
# the nets should end up looking like the pacejka curves the data came from.
#
# things to look at in the figures:
#   - where the data actually has slip angles, the learned curve should match
#   - outside that range the net is just guessing (no data = no info)
#   - the learned model gets tested on a maneuver it never saw (double lane change)
#
# usage: python experiments/03_neural_tire.py

import math
import os
import sys
import time

import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vdsim.params import car_m340i, G
from vdsim.car import dynamic_derivs, rollout, lateral_accel, pacejka_tires, slip_angles
from vdsim.tires import NeuralTire, pacejka

torch.set_default_dtype(torch.float64)
torch.manual_seed(3)
FIG = os.path.join(os.path.dirname(__file__), "..", "figures")

DT        = 0.01
WIN       = 150       # 1.5 s windows, shorter = easier for the nets early on
N_WIN     = 24
V         = 22.0
GYRO_STD  = 0.005
ACC_STD   = 0.15
ITERS     = 600

p = car_m340i()


def speed_hold(tf, tr, v_target=V):
    def f(x, u):
        drag = 0.5 * 1.225 * p["CdA"] * x[..., 3] ** 2 + p["Crr"] * p["m"] * G
        Fx = 3000.0 * (v_target - x[..., 3]) + drag
        uu = torch.stack([u[..., 0], Fx], dim=-1)
        return dynamic_derivs(x, uu, p, tf, tr)
    return f


def steering_inputs(amp_deg, n_win, win):
    t = torch.arange(win) * DT
    out = []
    for i in range(n_win):
        s = torch.zeros(win)
        for k in range(3):
            fr = 0.2 + 1.3 * torch.rand(1).item()
            ph = 2 * math.pi * torch.rand(1).item()
            s = s + torch.sin(2 * math.pi * fr * t + ph)
        s = s / s.abs().max()
        # vary the amplitude between windows so the data covers small AND big slip
        scale = 0.3 + 0.7 * torch.rand(1).item()
        out.append(math.radians(amp_deg) * scale * s)
    us = torch.stack(out, dim=1)
    return torch.stack([us, torch.zeros_like(us)], dim=-1)


tf_true, tr_true = pacejka_tires(p)
f_true = speed_hold(tf_true, tr_true)

# training data
us = steering_inputs(5.0, N_WIN, WIN)
x0 = torch.zeros(N_WIN, 6)
x0[:, 3] = V
with torch.no_grad():
    xs_true = rollout(f_true, x0, us, DT)[:-1]
    ay_true = lateral_accel(xs_true, f_true(xs_true, us))
    af_true, ar_true = slip_angles(xs_true, us[..., 0], p)
r_meas = xs_true[..., 5] + GYRO_STD * torch.randn_like(xs_true[..., 5])
ay_meas = ay_true + ACC_STD * torch.randn_like(ay_true)
print(f"training data: peak |ay| {ay_true.abs().max().item() / G:.2f} g, "
      f"front slip up to {math.degrees(af_true.abs().max().item()):.1f} deg, "
      f"rear up to {math.degrees(ar_true.abs().max().item()):.1f} deg")

nn_f = NeuralTire().double()
nn_r = NeuralTire().double()
vy0 = torch.zeros(N_WIN, requires_grad=True)
opt = torch.optim.Adam(list(nn_f.parameters()) + list(nn_r.parameters()) + [vy0], lr=3e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ITERS)
f_nn = speed_hold(nn_f, nn_r)

losses = []
t0 = time.time()
for it in range(ITERS):
    z = torch.zeros_like(vy0)
    x0_fit = torch.stack([z, z, z, z + V, vy0, r_meas[0]], dim=-1)
    xs = rollout(f_nn, x0_fit, us, DT)[:-1]
    ay = lateral_accel(xs, f_nn(xs, us))
    loss = torch.mean(((xs[..., 5] - r_meas) / GYRO_STD) ** 2) + torch.mean(((ay - ay_meas) / ACC_STD) ** 2)
    opt.zero_grad()
    loss.backward()
    opt.step()
    sched.step()
    losses.append(loss.item())
    if it % 100 == 0 or it == ITERS - 1:
        print(f"it {it:5d}  loss {loss.item():10.4f}  ({time.time() - t0:.0f} s)")
# a perfect model would sit at a loss of ~2 (it's normalized by the noise)

# ---- fig 1: learned vs true tire curves
a = torch.linspace(-0.25, 0.25, 501)
with torch.no_grad():
    lf = nn_f(a, torch.ones_like(a))
    lr_ = nn_r(a, torch.ones_like(a))
af_max = af_true.abs().max().item()
ar_max = ar_true.abs().max().item()

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, learned, tire, amax, title in [(axes[0], lf, p["tire_f"], af_max, "front axle"),
                                       (axes[1], lr_, p["tire_r"], ar_max, "rear axle")]:
    ax.axvspan(-math.degrees(amax), math.degrees(amax), color="g", alpha=0.08, label="slip range in training data")
    ax.plot(torch.rad2deg(a), pacejka(a, 1.0, p["mu"], *tire), "k", label="true (pacejka)")
    ax.plot(torch.rad2deg(a), learned, "C1--", lw=2, label="learned (neural net)")
    ax.set_title(title)
    ax.set_xlabel("slip angle (deg)")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("Fy / Fz")
axes[0].legend(loc="lower right")
fig.suptitle("tire curves learned only from yaw rate + lateral accel")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "03_learned_tires.png"), dpi=150)

inside_f = a.abs() <= af_max
inside_r = a.abs() <= ar_max
err_f = (lf - pacejka(a, 1.0, p["mu"], *p["tire_f"]))[inside_f].abs().max().item()
err_r = (lr_ - pacejka(a, 1.0, p["mu"], *p["tire_r"]))[inside_r].abs().max().item()
print(f"max |Fy/Fz| error inside the data range: front {err_f:.3f}, rear {err_r:.3f}")

# ---- fig 2: generalization on a maneuver it never trained on.
# ISO style double lane change at 80 km/h, steering built from two smoothed pulses
T = 600
t = torch.arange(T) * DT


def pulse(t, t0, width, amp):
    return amp * torch.exp(-((t - t0) / width) ** 2)


steer = pulse(t, 1.0, 0.35, 0.06) - pulse(t, 2.0, 0.35, 0.075) + pulse(t, 3.2, 0.35, 0.03)
us_test = torch.stack([steer, torch.zeros(T)], dim=-1)
x0_test = torch.tensor([0, 0, 0, V, 0, 0])
with torch.no_grad():
    xs_t = rollout(f_true, x0_test, us_test, DT)
    xs_n = rollout(f_nn, x0_test, us_test, DT)

fig, axes = plt.subplots(2, 1, figsize=(9, 6))
axes[0].plot(xs_t[:, 0], xs_t[:, 1], "k", label="true car")
axes[0].plot(xs_n[:, 0], xs_n[:, 1], "C1--", lw=2, label="learned model")
axes[0].set_xlabel("x (m)")
axes[0].set_ylabel("y (m)")
axes[0].set_title("double lane change (never seen in training)")
axes[0].legend()
axes[0].grid(alpha=0.3)
tt = torch.arange(T + 1) * DT
axes[1].plot(tt, xs_t[:, 5], "k")
axes[1].plot(tt, xs_n[:, 5], "C1--", lw=2)
axes[1].set_xlabel("time (s)")
axes[1].set_ylabel("yaw rate (rad/s)")
axes[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "03_lane_change.png"), dpi=150)

pos_err = torch.hypot(xs_t[:, 0] - xs_n[:, 0], xs_t[:, 1] - xs_n[:, 1]).max().item()
print(f"lane change: max position error {pos_err:.2f} m over {xs_t[-1, 0].item():.0f} m")
