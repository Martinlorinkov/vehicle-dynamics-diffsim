# fakes a Sensor Logger export from the sim with the phone mounted at a random
# angle, then checks the loader gets yaw rate and lateral accel back out

import math
import os

import numpy as np
import torch

from vdsim.params import car_m340i, G
from vdsim.car import dynamic_derivs, pacejka_tires, rollout, lateral_accel
from vdsim.telemetry import load_sensor_logger


def random_rotation(seed):
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def write_csv(path, t, cols):
    with open(path, "w") as f:
        f.write("time,seconds_elapsed," + ",".join(cols.keys()) + "\n")
        for i in range(len(t)):
            f.write(f"{int(t[i] * 1e9)},{t[i]}," + ",".join(f"{c[i]}" for c in cols.values()) + "\n")


def test_phone_frame_recovery(tmp_path):
    torch.set_default_dtype(torch.float64)
    p = car_m340i()
    tf, tr = pacejka_tires(p)
    dt = 0.01

    # sit still 4 s, accelerate, then weave around
    T = 3000
    t = np.arange(T) * dt
    steer = np.where(t > 12, 0.04 * np.sin(2 * math.pi * 0.4 * t), 0.0)
    Fx = np.where((t > 4) & (t < 12), 6000.0, 600.0)
    Fx = np.where(t < 4, 0.0, Fx)
    us = torch.tensor(np.stack([steer, Fx], axis=1))

    f = lambda x, u: dynamic_derivs(x, u, p, tf, tr)
    x0 = torch.tensor([0, 0, 0, 0.5, 0, 0.0])
    with torch.no_grad():
        xs = rollout(f, x0, us, dt)[:-1]
        d = f(xs, us)
    xs, d = xs.numpy(), d.numpy()
    ax = d[:, 3] - xs[:, 4] * xs[:, 5]
    ay = d[:, 4] + xs[:, 3] * xs[:, 5]
    r = xs[:, 5]
    speed = np.hypot(xs[:, 3], xs[:, 4])

    # body frame is x fwd, y left, z up. accelerometer reads specific force = a + g*up
    acc_body = np.stack([ax, ay, np.full(T, G)], axis=1)
    gyro_body = np.stack([np.zeros(T), np.zeros(T), r], axis=1)
    R = random_rotation(4)          # columns = phone axes expressed in body frame
    acc_phone = acc_body @ R
    gyro_phone = gyro_body @ R

    write_csv(tmp_path / "Gyroscope.csv", t, {"z": gyro_phone[:, 2], "y": gyro_phone[:, 1], "x": gyro_phone[:, 0]})
    write_csv(tmp_path / "TotalAcceleration.csv", t, {"z": acc_phone[:, 2], "y": acc_phone[:, 1], "x": acc_phone[:, 0]})
    tg = t[::100]
    write_csv(tmp_path / "Location.csv", tg, {"speed": speed[::100]})

    out = load_sensor_logger(str(tmp_path))
    n = len(out["t"])
    mid = slice(500, n - 100)
    assert np.max(np.abs(out["r"][mid] - r[:n][mid])) < 1e-3
    assert np.max(np.abs(out["ay"][mid] - ay[:n][mid])) < 0.05
