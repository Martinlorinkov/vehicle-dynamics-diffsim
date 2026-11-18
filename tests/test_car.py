import torch

from vdsim.params import car_m340i
from vdsim.car import (dynamic_derivs, kinematic_derivs, linear_tires, pacejka_tires,
                       rollout, steady_yaw_gain, rk4_step)
from vdsim.tires import pacejka_cornering_stiffness
from vdsim.params import G


def hold_speed(p, tire_f, tire_r, v_target):
    # wraps the dynamics with a simple P controller on Fx so speed stays put
    def f(x, u):
        Fx = 3000.0 * (v_target - x[..., 3]) + 0.5 * 1.225 * p["CdA"] * x[..., 3] ** 2 + p["Crr"] * p["m"] * G
        uu = torch.stack([u[..., 0], Fx], dim=-1)
        return dynamic_derivs(x, uu, p, tire_f, tire_r, ellipse=False)
    return f


def test_linear_steady_state_matches_theory():
    p = car_m340i()
    Cf, Cr = 120000.0, 160000.0
    tf, tr = linear_tires(Cf, Cr)
    for v in [10.0, 20.0, 30.0]:
        f = hold_speed(p, tf, tr, v)
        delta = 0.01
        x0 = torch.tensor([0, 0, 0, v, 0, 0], dtype=torch.float64)
        us = torch.tensor([[delta, 0.0]], dtype=torch.float64).repeat(1500, 1)
        xs = rollout(f, x0, us, 0.01)
        r_sim = xs[-1, 5].item()
        r_theory = delta * steady_yaw_gain(p, Cf, Cr, v)
        assert abs(r_sim - r_theory) / r_theory < 0.01, (v, r_sim, r_theory)


def test_low_speed_matches_kinematic():
    p = car_m340i()
    tf, tr = pacejka_tires(p)
    v = 3.0
    delta = 0.05
    f = hold_speed(p, tf, tr, v)
    x0 = torch.tensor([0, 0, 0, v, 0, 0], dtype=torch.float64)
    us = torch.tensor([[delta, 0.0]], dtype=torch.float64).repeat(500, 1)
    r_dyn = rollout(f, x0, us, 0.01)[-1, 5].item()

    fk = lambda x, u: kinematic_derivs(x, u, p)
    xk0 = torch.tensor([0, 0, 0, v], dtype=torch.float64)
    xk = rk4_step(fk, xk0, torch.tensor([delta, 0.0], dtype=torch.float64), 0.01)
    r_kin = (xk[2] - xk0[2]).item() / 0.01
    assert abs(r_dyn - r_kin) / r_kin < 0.02


def test_gradient_matches_finite_difference():
    p = car_m340i()

    def final_yaw(Iz):
        p["Iz"] = Iz
        tf, tr = pacejka_tires(p)
        f = hold_speed(p, tf, tr, 20.0)
        x0 = torch.tensor([0, 0, 0, 20.0, 0, 0], dtype=torch.float64)
        t = torch.arange(100, dtype=torch.float64) * 0.01
        us = torch.stack([0.03 * torch.sin(6.0 * t), torch.zeros_like(t)], dim=-1)
        return rollout(f, x0, us, 0.01)[-1, 5]

    Iz = torch.tensor(2900.0, dtype=torch.float64, requires_grad=True)
    final_yaw(Iz).backward()
    h = 1.0
    fd = (final_yaw(torch.tensor(2900.0 + h, dtype=torch.float64)) -
          final_yaw(torch.tensor(2900.0 - h, dtype=torch.float64))) / (2 * h)
    assert abs(Iz.grad.item() - fd.item()) / abs(fd.item()) < 1e-4
