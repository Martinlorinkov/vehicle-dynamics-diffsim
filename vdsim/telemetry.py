# loading real phone data (milestone 5)
#
# written for the "Sensor Logger" app (iOS / android) csv export. turn on
# Accelerometer (the one WITH gravity, called "TotalAcceleration" in some versions),
# Gyroscope and Location, record, then export as csv. you get a folder with one
# csv per sensor. columns the code uses:
#   Gyroscope.csv / Accelerometer.csv / TotalAcceleration.csv : seconds_elapsed, x, y, z
#   Location.csv                                              : seconds_elapsed, speed
#
# NOT tested on real data yet, I don't have a recording. the frame math is the
# standard approach but expect to fiddle with it the first time.
#
# the phone can be mounted at any angle as long as it doesn't move during the
# recording. the frame gets figured out from the data:
#   - "down" = average accelerometer direction while sitting still
#   - "forward" = horizontal direction that best lines up with dv/dt from gps
#   - "left" = down x forward ... sign checked so left turns give positive yaw rate

import os

import numpy as np


def _load(folder, name):
    path = os.path.join(folder, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        header = f.readline().strip().split(",")
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    return {h: data[:, i] for i, h in enumerate(header)}


def load_sensor_logger(folder, dt=0.01, still_seconds=3.0):
    gyro = _load(folder, "Gyroscope.csv")
    acc = _load(folder, "TotalAcceleration.csv")
    if acc is None:
        acc = _load(folder, "Accelerometer.csv")
        print("warning: using Accelerometer.csv, on iOS this has gravity REMOVED and is in g, not m/s^2")
    gps = _load(folder, "Location.csv")
    if gyro is None or acc is None or gps is None:
        raise FileNotFoundError("need Gyroscope.csv, TotalAcceleration.csv (or Accelerometer.csv) and Location.csv")

    t0 = max(gyro["seconds_elapsed"][0], acc["seconds_elapsed"][0], gps["seconds_elapsed"][0])
    t1 = min(gyro["seconds_elapsed"][-1], acc["seconds_elapsed"][-1], gps["seconds_elapsed"][-1])
    t = np.arange(t0, t1, dt)

    # resample everything onto one clock
    w = np.stack([np.interp(t, gyro["seconds_elapsed"], gyro[k]) for k in "xyz"], axis=1)
    a = np.stack([np.interp(t, acc["seconds_elapsed"], acc[k]) for k in "xyz"], axis=1)
    speed = np.interp(t, gps["seconds_elapsed"], np.clip(gps["speed"], 0, None))

    # gravity direction from the first few seconds (car parked, engine on is fine)
    still = t < t0 + still_seconds
    g_vec = a[still].mean(axis=0)
    up = g_vec / np.linalg.norm(g_vec)       # accelerometer reads +g pointing UP when still

    # forward axis: least squares fit of horizontal accel against gps dv/dt
    a_h = a - np.outer(a @ up, up)
    dvdt = np.gradient(speed, dt)
    moving = speed > 3.0
    coef, *_ = np.linalg.lstsq(a_h[moving], dvdt[moving], rcond=None)
    fwd = coef - (coef @ up) * up
    fwd = fwd / np.linalg.norm(fwd)
    left = np.cross(up, fwd)

    out = {}
    out["t"]     = t - t0
    out["speed"] = speed
    out["r"]     = w @ up          # yaw rate, positive turning left
    out["ay"]    = a @ left        # lateral accel, positive to the left
    out["ax"]    = a @ fwd
    # quick check: in a steady turn ay should be about speed * yaw rate
    turning = moving & (np.abs(out["r"]) > 0.05)
    if turning.sum() > 100:
        ratio = np.median(out["ay"][turning] / (out["speed"][turning] * out["r"][turning]))
        out["ay_over_vr"] = ratio
        print(f"sanity check: median ay / (v * r) in turns = {ratio:.2f}  (should be close to 1)")
    return out
