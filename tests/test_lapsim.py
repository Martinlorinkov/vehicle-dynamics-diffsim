import math
import torch

from vdsim.lapsim import speed_profile, lap_time, qss_car


def test_circle_lap():
    # constant radius circle: should sit at sqrt(ay_max * R) the whole way
    veh = qss_car()
    R = 50.0
    n = 400
    kappa = torch.full((n,), 1.0 / R, dtype=torch.float64)
    ds = torch.full((n,), 2 * math.pi * R / n, dtype=torch.float64)
    v = speed_profile(kappa, ds, veh)
    v_exact = math.sqrt(veh["ay_max"] * R)
    assert torch.allclose(v, torch.full_like(v, v_exact), rtol=1e-3)
    assert abs(lap_time(v, ds).item() - 2 * math.pi * R / v_exact) < 1e-2
