import torch

from vdsim.tires import pacejka, pacejka_cornering_stiffness, NeuralTire


def test_pacejka_slope_at_zero():
    Fz, mu, B, C, E = 8000.0, 1.0, 12.0, 1.35, -0.4
    a = torch.tensor(1e-5, dtype=torch.float64)
    slope = pacejka(a, Fz, mu, B, C, E) / a
    assert abs(slope.item() - pacejka_cornering_stiffness(Fz, mu, B, C)) / slope.item() < 1e-4


def test_pacejka_peak_is_mu_fz():
    a = torch.linspace(0, 0.5, 5001, dtype=torch.float64)
    f = pacejka(a, 8000.0, 1.0, 12.0, 1.35, -0.4)
    assert abs(f.max().item() - 8000.0) < 1.0


def test_pacejka_odd():
    a = torch.linspace(-0.3, 0.3, 101, dtype=torch.float64)
    f = pacejka(a, 5000.0, 1.0, 10.0, 1.4, 0.2)
    assert torch.allclose(f, -torch.flip(f, [0]))


def test_neural_tire_odd():
    torch.manual_seed(0)
    tire = NeuralTire()
    a = torch.linspace(-0.2, 0.2, 51)
    f = tire(a, torch.full_like(a, 5000.0))
    assert torch.allclose(f, -torch.flip(f, [0]), atol=1e-3)
