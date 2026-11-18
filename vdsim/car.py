# planar car models, written in torch so the whole rollout is differentiable.
#
# dynamic bicycle model state:  [X, Y, psi, vx, vy, r]
#   X, Y   position in the world frame (m)
#   psi    heading (rad)
#   vx, vy velocity in the BODY frame (m/s), vy positive to the left
#   r      yaw rate (rad/s), positive = turning left
# inputs:  [delta, Fx]
#   delta  front wheel steer angle (rad)
#   Fx     longitudinal force at the rear axle (N), the cars here are RWD
#
# kinematic model state: [X, Y, psi, v], inputs [delta, accel]
# the kinematic one has no tires at all, it's only right at low speed. it's in
# here mostly as a sanity check for the dynamic one.

import torch

from vdsim.params import G, RHO_AIR
from vdsim.tires import pacejka


def pacejka_tires(p):
    # builds the two axle force functions from a param dict
    Bf, Cf, Ef = p["tire_f"]
    Br, Cr, Er = p["tire_r"]
    mu = p["mu"]

    def front(alpha, Fz):
        return pacejka(alpha, Fz, mu, Bf, Cf, Ef)

    def rear(alpha, Fz):
        return pacejka(alpha, Fz, mu, Br, Cr, Er)

    return front, rear


def linear_tires(Cf, Cr):
    def front(alpha, Fz):
        return Cf * alpha

    def rear(alpha, Fz):
        return Cr * alpha

    return front, rear


def axle_loads(p, Fx):
    # static split plus longitudinal load transfer. uses Fx/m as the accel,
    # which ignores drag, but drag is small next to the tire forces
    m, a, b, L, h = p["m"], p["a"], p["b"], p["L"], p["h"]
    ax = Fx / m
    Fzf = m * (G * b - h * ax) / L
    Fzr = m * (G * a + h * ax) / L
    return Fzf, Fzr


def slip_angles(state, delta, p):
    vx, vy, r = state[..., 3], state[..., 4], state[..., 5]
    alpha_f = delta - torch.atan2(vy + p["a"] * r, vx)
    alpha_r = -torch.atan2(vy - p["b"] * r, vx)
    return alpha_f, alpha_r


def dynamic_derivs(state, u, p, tire_f, tire_r, ellipse=True):
    psi, vx, vy, r = state[..., 2], state[..., 3], state[..., 4], state[..., 5]
    delta, Fx = u[..., 0], u[..., 1]

    Fzf, Fzr = axle_loads(p, Fx)
    alpha_f, alpha_r = slip_angles(state, delta, p)

    Fyf = tire_f(alpha_f, Fzf)
    Fyr = tire_r(alpha_r, Fzr)

    if ellipse:
        # friction ellipse on the rear: the more Fx you ask for, the less
        # lateral grip is left. clamp keeps sqrt away from 0 so grads don't blow up
        frac = (Fx / (p["mu"] * Fzr)) ** 2
        Fyr = Fyr * torch.sqrt(torch.clamp(1.0 - frac, min=1e-3))

    drag = 0.5 * RHO_AIR * p["CdA"] * vx * vx + p["Crr"] * p["m"] * G

    m, Iz, a, b = p["m"], p["Iz"], p["a"], p["b"]
    cd, sd = torch.cos(delta), torch.sin(delta)

    dvx = (Fx - Fyf * sd - drag) / m + vy * r
    dvy = (Fyr + Fyf * cd) / m - vx * r
    dr = (a * Fyf * cd - b * Fyr) / Iz

    dX = vx * torch.cos(psi) - vy * torch.sin(psi)
    dY = vx * torch.sin(psi) + vy * torch.cos(psi)

    return torch.stack([dX, dY, r, dvx, dvy, dr], dim=-1)


def kinematic_derivs(state, u, p):
    psi, v = state[..., 2], state[..., 3]
    delta, acc = u[..., 0], u[..., 1]
    # beta = slip angle at the cg
    beta = torch.atan(p["b"] / p["L"] * torch.tan(delta))
    dX = v * torch.cos(psi + beta)
    dY = v * torch.sin(psi + beta)
    dpsi = v * torch.cos(beta) * torch.tan(delta) / p["L"]
    return torch.stack([dX, dY, dpsi, acc], dim=-1)


def rk4_step(f, x, u, dt):
    k1 = f(x, u)
    k2 = f(x + 0.5 * dt * k1, u)
    k3 = f(x + 0.5 * dt * k2, u)
    k4 = f(x + dt * k3, u)
    return x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def rollout(f, x0, us, dt):
    # us is [T, ..., n_inputs], returns states [T+1, ..., n_states]
    xs = [x0]
    x = x0
    for t in range(us.shape[0]):
        x = rk4_step(f, x, us[t], dt)
        xs.append(x)
    return torch.stack(xs)


def lateral_accel(state, deriv):
    # ay in the body frame = dvy/dt + vx * r (what an IMU / phone would read)
    return deriv[..., 4] + state[..., 3] * state[..., 5]


def understeer_gradient(p, Cf, Cr):
    # rad per (m/s^2). positive = understeer
    return p["m"] / p["L"] * (p["b"] / Cf - p["a"] / Cr)


def steady_yaw_gain(p, Cf, Cr, vx):
    # r / delta at steady state for the linear model
    K = understeer_gradient(p, Cf, Cr)
    return vx / (p["L"] + K * vx * vx)
