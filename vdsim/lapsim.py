# quasi steady state (QSS) lap time simulator
#
# the vehicle is a point mass with a g-g diagram (ellipse). for a given path:
#   1. every point gets a max cornering speed from its curvature
#   2. forward pass: accelerate as hard as power/traction allow
#   3. backward pass: brake as late as possible
#   4. speed = min of all three
# this is the same idea commercial lap sims started from. it's written with torch
# min/sqrt so lap time is differentiable w.r.t. the path, which is what lets
# experiments/05_racing_line.py push the line around with gradient descent.

import torch

from vdsim.params import G, RHO_AIR, car_m340i, bike_sport
from vdsim.whipple import mass_center


def qss_car():
    p = car_m340i()
    v = {}
    v["name"]       = "BMW M340i (est.)"
    v["m"]          = p["m"] + 80.0           # + driver
    v["P"]          = 285e3 * 0.85            # 382 hp at the crank, ~15% drivetrain loss
    v["CdA"]        = p["CdA"]
    v["ay_max"]     = p["mu"] * G             # lateral grip limit
    v["ax_brake"]   = 1.0 * G
    v["ax_accel"]   = 0.65 * G                # traction limited at low speed (0-60 in ~4.2 s)
    v["v_top"]      = 250.0 / 3.6             # electronically limited
    v["half_width"] = 0.93
    return v


def qss_bike():
    p = bike_sport()
    mT, xcg, hcg = mass_center(p)
    v = {}
    v["name"]       = "600cc sport bike (est.)"
    v["m"]          = mT
    v["P"]          = 75e3                    # ~120 hp at the crank, peaky, so use an average
    v["CdA"]        = 0.35                    # rider tucked in
    v["ay_max"]     = 1.1 * G                 # tire limit, lean angle ~48 deg
    # pitch limits fall straight out of where the cg is:
    #   stoppie when decel * h > g * (distance from cg to front contact)
    #   wheelie when accel * h > g * (distance from cg to rear contact)
    v["ax_brake"]   = min(1.1 * G, G * (p["w"] - xcg) / hcg)
    v["ax_accel"]   = min(1.1 * G, G * xcg / hcg)
    v["v_top"]      = 260.0 / 3.6
    v["half_width"] = 0.40
    return v


def _ellipse_left(v2, kappa, veh):
    # fraction of longitudinal grip left after cornering takes its share
    ay = v2 * torch.abs(kappa)
    frac = (ay / veh["ay_max"]) ** 2
    return torch.sqrt(torch.clamp(1.0 - frac, min=1e-4))


def speed_profile(kappa, ds, veh):
    n = kappa.shape[0]
    m = veh["m"]

    v_corner2 = veh["ay_max"] / (torch.abs(kappa) + 1e-6)
    v_lim2 = torch.clamp(v_corner2, max=veh["v_top"] ** 2)

    # start both passes at the slowest corner, that speed is fixed no matter what
    start = int(torch.argmin(v_lim2).item())
    idx = [(start + i) % n for i in range(n)]

    # forward pass (accelerating)
    fwd = [None] * n
    fwd[idx[0]] = v_lim2[idx[0]]
    for k in range(1, n):
        i, j = idx[k - 1], idx[k]
        vi2 = fwd[i]
        vi = torch.sqrt(vi2 + 1e-6)
        drag = 0.5 * RHO_AIR * veh["CdA"] * vi2 / m
        a_trac = torch.minimum(veh["P"] / (m * vi), torch.as_tensor(veh["ax_accel"], dtype=vi.dtype))
        a = a_trac * _ellipse_left(vi2, kappa[i], veh) - drag
        fwd[j] = torch.minimum(v_lim2[j], torch.clamp(vi2 + 2.0 * a * ds[i], min=1.0))

    # backward pass (braking), walking the lap in reverse
    bwd = [None] * n
    bwd[idx[0]] = fwd[idx[0]]
    for k in range(n - 1, 0, -1):
        j = idx[(k + 1) % n]
        i = idx[k]
        vj2 = bwd[j]
        drag = 0.5 * RHO_AIR * veh["CdA"] * vj2 / m
        a = veh["ax_brake"] * _ellipse_left(vj2, kappa[j], veh) + drag
        bwd[i] = torch.minimum(fwd[i], vj2 + 2.0 * a * ds[i])

    v2 = torch.stack(bwd)
    return torch.sqrt(v2 + 1e-9)


def lap_time(v, ds):
    v_next = torch.roll(v, -1)
    return torch.sum(2.0 * ds / (v + v_next))


def accelerations(v, kappa, ds):
    # for g-g plots
    ay = v * v * kappa
    ax = (torch.roll(v, -1) ** 2 - v * v) / (2.0 * ds)
    return ax, ay
