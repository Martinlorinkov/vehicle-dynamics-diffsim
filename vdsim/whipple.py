# linearized whipple bicycle model (lean + steer), straight from
# Meijaard et al. 2007, Proc. R. Soc. A 463, 1955-1982.
#
#   M q'' + v C1 q' + (g K0 + v^2 K2) q = f
#
# q = [phi, delta] = [lean angle, steer angle], f = [lean torque, steer torque]
#
# the matrices get built from the 25 physical parameters in params.py instead of
# being hard coded, so you can change trail/rake/masses and see what happens.
# tests/test_whipple.py checks them against the published benchmark numbers.
#
# everything is torch (float64) so you can backprop through the eigenvalues too.

import torch


def _t(x):
    return torch.as_tensor(x, dtype=torch.float64)


def whipple_matrices(p):
    w, c, lam, g = _t(p["w"]), _t(p["c"]), _t(p["lam"]), _t(p["g"])
    rR, mR, IRxx, IRyy = _t(p["rR"]), _t(p["mR"]), _t(p["IRxx"]), _t(p["IRyy"])
    xB, zB, mB = _t(p["xB"]), _t(p["zB"]), _t(p["mB"])
    IBxx, IBzz, IBxz = _t(p["IBxx"]), _t(p["IBzz"]), _t(p["IBxz"])
    xH, zH, mH = _t(p["xH"]), _t(p["zH"]), _t(p["mH"])
    IHxx, IHzz, IHxz = _t(p["IHxx"]), _t(p["IHzz"]), _t(p["IHxz"])
    rF, mF, IFxx, IFyy = _t(p["rF"]), _t(p["mF"]), _t(p["IFxx"]), _t(p["IFyy"])

    # wheels are symmetric so Izz = Ixx
    IRzz, IFzz = IRxx, IFxx
    sl, cl = torch.sin(lam), torch.cos(lam)

    # whole bike treated as one rigid body
    mT = mR + mB + mH + mF
    xT = (xB * mB + xH * mH + w * mF) / mT
    zT = (-rR * mR + zB * mB + zH * mH - rF * mF) / mT
    ITxx = IRxx + IBxx + IHxx + IFxx + mR * rR**2 + mB * zB**2 + mH * zH**2 + mF * rF**2
    ITxz = IBxz + IHxz - mB * xB * zB - mH * xH * zH + mF * w * rF
    ITzz = IRzz + IBzz + IHzz + IFzz + mB * xB**2 + mH * xH**2 + mF * w**2

    # front assembly (fork/bars + front wheel)
    mA = mH + mF
    xA = (xH * mH + w * mF) / mA
    zA = (zH * mH - rF * mF) / mA
    IAxx = IHxx + IFxx + mH * (zH - zA)**2 + mF * (rF + zA)**2
    IAxz = IHxz - mH * (xH - xA) * (zH - zA) + mF * (w - xA) * (rF + zA)
    IAzz = IHzz + IFzz + mH * (xH - xA)**2 + mF * (w - xA)**2

    # uA = how far the front assembly cg sits off the steer axis
    uA = (xA - w - c) * cl - zA * sl
    IAll = mA * uA**2 + IAxx * sl**2 + 2 * IAxz * sl * cl + IAzz * cl**2
    IAlx = -mA * uA * zA + IAxx * sl + IAxz * cl
    IAlz = mA * uA * xA + IAxz * sl + IAzz * cl

    mu = c / w * cl
    SR = IRyy / rR
    SF = IFyy / rF
    ST = SR + SF
    SA = mA * uA + mu * mT * xT

    M = torch.stack([
        torch.stack([ITxx, IAlx + mu * ITxz]),
        torch.stack([IAlx + mu * ITxz, IAll + 2 * mu * IAlz + mu**2 * ITzz]),
    ])
    K0 = torch.stack([
        torch.stack([mT * zT, -SA]),
        torch.stack([-SA, -SA * sl]),
    ])
    zero = torch.zeros((), dtype=torch.float64)
    K2 = torch.stack([
        torch.stack([zero, (ST - mT * zT) / w * cl]),
        torch.stack([zero, (SA + SF * sl) / w * cl]),
    ])
    C1 = torch.stack([
        torch.stack([zero, mu * ST + SF * cl + ITxz / w * cl - mu * mT * zT]),
        torch.stack([-(mu * ST + SF * cl), IAlz / w * cl + mu * (SA + ITzz / w * cl)]),
    ])
    return M, C1, K0, K2


def state_matrix(p, v, mats=None):
    # x = [phi, delta, phi_dot, delta_dot], x' = A x
    if mats is None:
        mats = whipple_matrices(p)
    M, C1, K0, K2 = mats
    v = _t(v)
    Minv = torch.linalg.inv(M)
    K = _t(p["g"]) * K0 + v**2 * K2
    top = torch.cat([torch.zeros(2, 2, dtype=torch.float64), torch.eye(2, dtype=torch.float64)], dim=1)
    bot = torch.cat([-Minv @ K, -Minv @ (v * C1)], dim=1)
    return torch.cat([top, bot], dim=0)


def eigenvalues(p, v, mats=None):
    return torch.linalg.eigvals(state_matrix(p, v, mats))


def eig_sweep(p, speeds):
    mats = whipple_matrices(p)
    out = []
    for v in speeds:
        ev = eigenvalues(p, float(v), mats)
        # sort by real part so the plot lines don't jump around as much
        ev = ev[torch.argsort(ev.real)]
        out.append(ev)
    return torch.stack(out)


def _max_real_osc(ev):
    # largest real part among the complex (oscillating) eigenvalues = weave mode
    osc = ev[ev.imag.abs() > 1e-6]
    if osc.numel() == 0:
        return None
    return osc.real.max().item()


def _max_real_nonosc(ev):
    # largest real part among the purely real eigenvalues = capsize mode
    re = ev[ev.imag.abs() <= 1e-6]
    if re.numel() == 0:
        return None
    return re.real.max().item()


def _bisect(fn, lo, hi, iters=60):
    flo = fn(lo)
    for i in range(iters):
        mid = 0.5 * (lo + hi)
        fm = fn(mid)
        if (fm > 0) == (flo > 0):
            lo, flo = mid, fm
        else:
            hi = mid
    return 0.5 * (lo + hi)


def stable_speed_range(p, vmax=20.0, n=400):
    # returns (weave speed, capsize speed). between the two the bike balances itself.
    # weave speed: weave mode real part goes from + to -
    # capsize speed: capsize mode real part goes from - to +
    mats = whipple_matrices(p)
    speeds = torch.linspace(0.05, vmax, n, dtype=torch.float64).tolist()

    def weave_fn(v):
        r = _max_real_osc(eigenvalues(p, v, mats))
        return 1.0 if r is None else r

    def capsize_fn(v):
        r = _max_real_nonosc(eigenvalues(p, v, mats))
        return -1.0 if r is None else r

    v_weave = None
    v_capsize = None
    prev_w, prev_c = weave_fn(speeds[0]), capsize_fn(speeds[0])
    for i in range(1, len(speeds)):
        cur_w, cur_c = weave_fn(speeds[i]), capsize_fn(speeds[i])
        if v_weave is None and prev_w > 0 and cur_w <= 0:
            v_weave = _bisect(weave_fn, speeds[i - 1], speeds[i])
        if v_weave is not None and v_capsize is None and prev_c < 0 and cur_c >= 0:
            v_capsize = _bisect(capsize_fn, speeds[i - 1], speeds[i])
        prev_w, prev_c = cur_w, cur_c
    return v_weave, v_capsize


def mass_center(p):
    # total cg (x forward from rear contact, height above ground), used by the lap sim
    mT = p["mR"] + p["mB"] + p["mH"] + p["mF"]
    xT = (p["xB"] * p["mB"] + p["xH"] * p["mH"] + p["w"] * p["mF"]) / mT
    zT = (-p["rR"] * p["mR"] + p["zB"] * p["mB"] + p["zH"] * p["mH"] - p["rF"] * p["mF"]) / mT
    return mT, xT, -zT
