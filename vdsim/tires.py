# tire models. everything is written with torch ops so gradients flow through them.
#
# all of these return lateral force for a whole axle (bicycle model lumps the
# left and right tire together), given slip angle alpha in radians and the axle
# normal load Fz in newtons.

import torch
import torch.nn as nn


def pacejka(alpha, Fz, mu, B, C, E):
    # simplified "magic formula", Pacejka & Bakker. D = peak force = mu * Fz
    # B = stiffness factor, C = shape factor, E = curvature factor
    x = B * alpha
    return mu * Fz * torch.sin(C * torch.atan(x - E * (x - torch.atan(x))))


def pacejka_cornering_stiffness(Fz, mu, B, C):
    # slope at alpha = 0 is B*C*D
    return B * C * mu * Fz


def linear_tire(alpha, C_alpha):
    return C_alpha * alpha


def peak_slip_angle(B, C, E):
    # brute force, it's only used for plots and the readme
    a = torch.linspace(0.0, 0.5, 20001, dtype=torch.float64)
    f = pacejka(a, 1.0, 1.0, B, C, E)
    return a[torch.argmax(f)].item()


class NeuralTire(nn.Module):
    # small mlp that maps slip angle -> Fy / Fz
    # forced to be an odd function (f(-a) = -f(a)) since a tire pulls the same
    # either direction. without this the net happily learns an offset when the
    # data has more left turns than right turns
    def __init__(self, hidden=32, alpha_scale=0.1):
        super().__init__()
        self.alpha_scale = alpha_scale     # inputs come in as ~0.05 rad, scale them up so tanh isn't dead
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, alpha, Fz):
        x = (alpha / self.alpha_scale).unsqueeze(-1)
        out = 0.5 * (self.net(x) - self.net(-x))
        return Fz * out.squeeze(-1)
