# track loading + geometry
#
# tracks come from the TUM racetrack database (github.com/TUMFTM/racetrack-database),
# csv columns are x_m, y_m, w_tr_right_m, w_tr_left_m. run data/get_tracks.py first.

import numpy as np
import torch


def load_track(path, ds=5.0):
    raw = np.loadtxt(path, delimiter=",", comments="#")
    x, y, wr, wl = raw[:, 0], raw[:, 1], raw[:, 2], raw[:, 3]

    # close the loop, then resample so every point is ds apart
    x = np.append(x, x[0])
    y = np.append(y, y[0])
    wr = np.append(wr, wr[0])
    wl = np.append(wl, wl[0])
    seg = np.hypot(np.diff(x), np.diff(y))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]

    n = int(round(total / ds))
    s_new = np.linspace(0.0, total, n, endpoint=False)
    track = {}
    track["x"]      = np.interp(s_new, s, x)
    track["y"]      = np.interp(s_new, s, y)
    track["w_right"] = np.interp(s_new, s, wr)
    track["w_left"]  = np.interp(s_new, s, wl)
    track["s"]      = s_new
    track["length"] = total

    # unit normal pointing to the LEFT of the driving direction
    dx = np.roll(track["x"], -1) - np.roll(track["x"], 1)
    dy = np.roll(track["y"], -1) - np.roll(track["y"], 1)
    norm = np.hypot(dx, dy)
    track["nx"] = -dy / norm
    track["ny"] = dx / norm
    return track


def path_points(track, offset):
    # offset is a torch tensor, positive = left of centerline
    cx = torch.as_tensor(track["x"], dtype=torch.float64)
    cy = torch.as_tensor(track["y"], dtype=torch.float64)
    nx = torch.as_tensor(track["nx"], dtype=torch.float64)
    ny = torch.as_tensor(track["ny"], dtype=torch.float64)
    return cx + offset * nx, cy + offset * ny


def curvature(x, y):
    # signed curvature at each point from the circle through it and its two
    # neighbors (menger curvature). closed loop so roll handles the wrap around
    x0, y0 = torch.roll(x, 1), torch.roll(y, 1)
    x2, y2 = torch.roll(x, -1), torch.roll(y, -1)
    ax, ay = x - x0, y - y0
    bx, by = x2 - x, y2 - y
    cross = ax * by - ay * bx
    la = torch.sqrt(ax * ax + ay * ay + 1e-12)
    lb = torch.sqrt(bx * bx + by * by + 1e-12)
    lc = torch.sqrt((x2 - x0) ** 2 + (y2 - y0) ** 2 + 1e-12)
    return 2.0 * cross / (la * lb * lc)


def segment_lengths(x, y):
    dx = torch.roll(x, -1) - x
    dy = torch.roll(y, -1) - y
    return torch.sqrt(dx * dx + dy * dy + 1e-12)


def smooth_periodic(a, sigma):
    # gaussian blur around a closed loop, sigma in number of points.
    # done with an fft so it's one op and autograd handles it fine
    n = a.shape[0]
    i = torch.arange(n, dtype=a.dtype)
    d = torch.minimum(i, n - i)
    g = torch.exp(-0.5 * (d / sigma) ** 2)
    g = g / g.sum()
    return torch.real(torch.fft.ifft(torch.fft.fft(a) * torch.fft.fft(g)))
