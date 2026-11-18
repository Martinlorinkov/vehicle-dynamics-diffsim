# system identification by differentiating through the simulator
#
# idea: record what a phone would see while driving (yaw rate from the gyro,
# lateral accel from the accelerometer, speed from gps), then find the car
# parameters that make the sim reproduce it. here the "recording" comes from the
# sim itself with the true params + sensor noise, so we know the right answer and
# can check whether the fit actually finds it.
#
# two datasets:
#   gentle   - normal driving, never over ~0.35 g
#   spirited - steering sweeps that get up to ~0.85 g
# the point is to show that mu (peak grip) can't be identified from gentle
# driving, the tires never get close to the limit so the data doesn't care.
#
# usage: python experiments/02_system_id.py

import math
import os
import sys
import time

import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vdsim.params import car_m340i, G
from vdsim.car import dynamic_derivs, rollout, lateral_accel
from vdsim.tires import pacejka
from torch.func import jacfwd

torch.set_default_dtype(torch.float64)
torch.manual_seed(7)
FIG = os.path.join(os.path.dirname(__file__), "..", "figures")

DT        = 0.01
WIN       = 250       # steps per window (2.5 s)
N_WIN     = 12
V         = 22.0      # m/s, ~80 km/h
GYRO_STD  = 0.005     # rad/s, roughly a phone gyro after some filtering
ACC_STD   = 0.15      # m/s^2
LM_IT     = 25
NAMES     = ["Iz", "Bf", "Br", "mu"]


def make_model(p, Iz, Bf, Br, mu):
    q = dict(p)
    q["Iz"] = Iz
    q["mu"] = mu
    Cf_, Ef = p["tire_f"][1], p["tire_f"][2]
    Cr_, Er = p["tire_r"][1], p["tire_r"][2]

    def tf(a, Fz):
        return pacejka(a, Fz, mu, Bf, Cf_, Ef)

    def tr(a, Fz):
        return pacejka(a, Fz, mu, Br, Cr_, Er)

    def f(x, u):
        drag = 0.5 * 1.225 * q["CdA"] * x[..., 3] ** 2 + q["Crr"] * q["m"] * G
        Fx = 3000.0 * (V - x[..., 3]) + drag
        uu = torch.stack([u[..., 0], Fx], dim=-1)
        return dynamic_derivs(x, uu, q, tf, tr)
    return f


def steering_inputs(amp_deg, n_win, win):
    # each window is a random mix of sines between 0.2 and 1.5 hz, like
    # someone weaving down an empty road / a slalom
    t = torch.arange(win) * DT
    out = []
    for i in range(n_win):
        s = torch.zeros(win)
        for k in range(3):
            fr = 0.2 + 1.3 * torch.rand(1).item()
            ph = 2 * math.pi * torch.rand(1).item()
            s = s + torch.sin(2 * math.pi * fr * t + ph)
        s = s / s.abs().max()
        out.append(math.radians(amp_deg) * s)
    us = torch.stack(out, dim=1)                     # [T, N]
    return torch.stack([us, torch.zeros_like(us)], dim=-1)


def simulate(f, us):
    x0 = torch.zeros(us.shape[1], 6)
    x0[:, 3] = V
    xs = rollout(f, x0, us, DT)[:-1]
    d = f(xs, us)
    return xs, lateral_accel(xs, d)


def fit(data_name, us, r_meas, ay_meas, p, true_vals):
    # levenberg-marquardt (damped gauss-newton) on the 4 log params.
    #
    # first version of this used adam, and it stalled about 5% off the true
    # values: at small slip the tire force is B*C*mu*Fz*alpha, so the data mostly
    # sees the PRODUCT B*mu, and adam crawls along that long narrow valley. LM
    # uses the jacobian of the residuals, so it sees the valley shape and walks
    # straight down it. it's a least squares problem anyway, which is exactly
    # what LM is for.
    #
    # the jacobian comes from forward mode autodiff (torch.func.jacfwd) through
    # the whole rollout, one pass per parameter.
    #
    # windows start from driving straight (vy = 0), yaw rate comes from the gyro.
    init = {"Iz": 2000.0, "Bf": 16.0, "Br": 10.0, "mu": 0.7}    # 30-40% off
    theta = torch.tensor([math.log(init[k]) for k in NAMES])

    def residuals(th):
        Iz, Bf, Br, mu = torch.exp(th)
        f = make_model(p, Iz, Bf, Br, mu)
        x0 = torch.zeros(us.shape[1], 6)
        x0[:, 3] = V
        x0[:, 5] = r_meas[0]
        xs = rollout(f, x0, us, DT)[:-1]
        ay = lateral_accel(xs, f(xs, us))
        return torch.cat([((xs[..., 5] - r_meas) / GYRO_STD).reshape(-1),
                          ((ay - ay_meas) / ACC_STD).reshape(-1)])

    hist = {k: [] for k in NAMES}
    hist["loss"] = []
    N = r_meas.numel()
    lam = 1e-2
    t0 = time.time()
    for it in range(LM_IT):
        r = residuals(theta)
        J = jacfwd(residuals)(theta)
        JtJ = J.T @ J
        g = J.T @ r
        cost = 0.5 * (r @ r)

        vals = torch.exp(theta)
        for i, k in enumerate(NAMES):
            hist[k].append(vals[i].item())
        hist["loss"].append(cost.item() / N)     # ~1.0 means the fit is down at the noise level
        s = "  ".join(f"{k}={vals[i].item():8.3f}" for i, k in enumerate(NAMES))
        print(f"[{data_name}] it {it:3d}  loss {cost.item() / N:10.4f}  lam {lam:.0e}  {s}")

        # try a step, if it makes things worse damp harder and try again
        while True:
            step = torch.linalg.solve(JtJ + lam * torch.diag(torch.diag(JtJ)), -g)
            r_new = residuals(theta + step)
            if 0.5 * (r_new @ r_new) < cost:
                theta = theta + step
                lam = max(lam / 3.0, 1e-7)
                break
            lam *= 5.0
            if lam > 1e6:
                break
        if step.abs().max() < 1e-7 or lam > 1e6:
            break
    print(f"[{data_name}] took {time.time() - t0:.0f} s")

    # error bars. cost = 0.5 * sum(r^2) is the negative log likelihood (residuals
    # are already divided by the noise), so J^T J is the fisher information and
    # its inverse is the covariance of the log params. sqrt of the diagonal is a
    # 1 sigma error in log space, which is basically a relative (%) error
    J = jacfwd(residuals)(theta)
    cov = torch.linalg.inv(J.T @ J)
    sigma = torch.sqrt(torch.diag(cov))
    hist["sigma"] = {k: 100 * sigma[i].item() for i, k in enumerate(NAMES)}
    hist["corr"] = cov / torch.outer(sigma, sigma)
    print(f"[{data_name}] 1-sigma (%): " + "  ".join(f"{k} {hist['sigma'][k]:.2f}" for k in NAMES))

    vals = torch.exp(theta)
    f = make_model(p, *vals)
    x0 = torch.zeros(us.shape[1], 6)
    x0[:, 3] = V
    x0[:, 5] = r_meas[0]
    with torch.no_grad():
        xs = rollout(f, x0, us, DT)[:-1]
        ay = lateral_accel(xs, f(xs, us))
    return hist, xs, ay


p = car_m340i()
true_vals = {"Iz": p["Iz"], "Bf": p["tire_f"][0], "Br": p["tire_r"][0], "mu": p["mu"]}
f_true = make_model(p, **true_vals)

results = {}
for name, amp in [("gentle", 1.2), ("spirited", 4.2)]:
    us = steering_inputs(amp, N_WIN, WIN)
    with torch.no_grad():
        xs, ay = simulate(f_true, us)
    r_meas = xs[..., 5] + GYRO_STD * torch.randn_like(xs[..., 5])
    ay_meas = ay + ACC_STD * torch.randn_like(ay)
    print(f"\n{name}: peak |ay| in data = {ay.abs().max().item() / G:.2f} g")
    hist, xs_fit, ay_fit = fit(name, us, r_meas, ay_meas, p, true_vals)
    results[name] = (hist, r_meas, xs_fit, ay_meas, ay_fit)

print("\nfinal relative error, +/- 1 sigma (%)")
print(f"{'':10s}" + "".join(f"{k:>17s}" for k in NAMES))
for name in results:
    hist = results[name][0]
    row = "".join(f"{100 * (hist[k][-1] - true_vals[k]) / true_vals[k]:8.2f} +/-{hist['sigma'][k]:5.2f}" for k in NAMES)
    print(f"{name:10s}{row}")
    c = hist["corr"]
    print(f"{'':10s}correlation Bf-mu {c[1, 3].item():+.3f}   Br-mu {c[2, 3].item():+.3f}   Bf-Br {c[1, 2].item():+.3f}")

# ---- plots
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, name in zip(axes, results):
    hist = results[name][0]
    for k in NAMES:
        err = [100 * (h - true_vals[k]) / true_vals[k] for h in hist[k]]
        ax.plot(err, label=k)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title(f"{name} driving")
    ax.set_xlabel("LM iteration")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("parameter error (%)")
axes[1].legend()
fig.suptitle("recovering car parameters with levenberg-marquardt through the simulator")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "02_sysid_convergence.png"), dpi=150)

hist, r_meas, xs_fit, ay_meas, ay_fit = results["spirited"]
t = torch.arange(WIN) * DT
fig, axes = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)
axes[0].plot(t, r_meas[:, 0], ".", ms=2, alpha=0.5, label="measured (noisy)")
axes[0].plot(t, xs_fit[:, 0, 5], label="fitted model")
axes[0].set_ylabel("yaw rate (rad/s)")
axes[0].legend()
axes[1].plot(t, ay_meas[:, 0] / G, ".", ms=2, alpha=0.5)
axes[1].plot(t, ay_fit[:, 0] / G)
axes[1].set_ylabel("lateral accel (g)")
axes[1].set_xlabel("time (s)")
for ax in axes:
    ax.grid(alpha=0.3)
fig.suptitle("one window of the spirited dataset after fitting")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "02_sysid_fit.png"), dpi=150)
