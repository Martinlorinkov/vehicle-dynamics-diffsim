# checks the whipple model against the benchmark in Meijaard et al. 2007 (table 2
# and the weave / capsize speeds quoted in section 6)

import torch

from vdsim.params import bike_benchmark
from vdsim.whipple import whipple_matrices, stable_speed_range, eigenvalues


def close(a, b, tol=1e-9):
    return abs(float(a) - float(b)) < tol


def test_benchmark_matrices():
    M, C1, K0, K2 = whipple_matrices(bike_benchmark())

    assert close(M[0, 0], 80.81722)
    assert close(M[0, 1], 2.31941332208709)
    assert close(M[1, 0], 2.31941332208709)
    assert close(M[1, 1], 0.29784188199686)

    assert close(K0[0, 0], -80.95)
    assert close(K0[0, 1], -2.59951685249872)
    assert close(K0[1, 1], -0.80329488458618)

    assert close(K2[0, 1], 76.59734589573222)
    assert close(K2[1, 1], 2.65431523794604)

    assert close(C1[0, 1], 33.86641391492494)
    assert close(C1[1, 0], -0.85035641456978)
    assert close(C1[1, 1], 1.68540397397560)


def test_benchmark_stable_range():
    v_weave, v_capsize = stable_speed_range(bike_benchmark())
    # paper: v_w = 4.292 m/s, v_c = 6.024 m/s
    assert abs(v_weave - 4.292) < 1e-3
    assert abs(v_capsize - 6.024) < 1e-3


def test_stable_inside_range():
    p = bike_benchmark()
    ev = eigenvalues(p, 5.0)
    assert ev.real.max().item() < 0      # self stable at 5 m/s
    ev = eigenvalues(p, 2.0)
    assert ev.real.max().item() > 0      # falls over at walking speed


def test_eigen_gradient():
    # making sure you can backprop through the eigenvalues wrt trail
    p = bike_benchmark()
    c = torch.tensor(0.08, dtype=torch.float64, requires_grad=True)
    p["c"] = c
    ev = eigenvalues(p, 5.0)
    ev.real.max().backward()
    assert c.grad is not None and torch.isfinite(c.grad)
