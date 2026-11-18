# vehicle parameter sets
#
# the car numbers are rough, pulled from public spec sheets (curb weight, wheelbase,
# weight split) plus textbook guesses for the stuff BMW doesn't publish (yaw inertia,
# cg height, tire coefficients). good enough to get realistic behavior, not good
# enough to claim it IS an M340i. that's what experiments/03_system_id.py is for.

G = 9.81
RHO_AIR = 1.225


def car_m340i():
    p = {}
    p["m"]      = 1745.0    # kg, curb weight (xDrive is a bit heavier, this is close enough)
    p["L"]      = 2.851     # m, wheelbase (G20)
    p["a"]      = 1.42      # m, cg to front axle (roughly 50/50 split)
    p["b"]      = p["L"] - p["a"]
    p["h"]      = 0.50      # m, cg height, guess
    p["Iz"]     = 2900.0    # kg m^2, yaw inertia, guess using Iz ~ m*a*b*1.15
    p["CdA"]    = 0.60      # m^2, Cd 0.26 * ~2.3 m^2 frontal area
    p["Crr"]    = 0.012     # rolling resistance coefficient
    p["mu"]     = 1.0       # dry summer tires
    # pacejka lateral coefficients (B, C, E) per axle, peak force D = mu * Fz
    # rear is stiffer than the front so the car understeers a bit (~1 deg/g),
    # which is what every road car is tuned to do
    p["tire_f"] = (12.0, 1.35, -0.4)
    p["tire_r"] = (15.0, 1.35, -0.4)
    return p


def car_m240i():
    p = car_m340i()
    p["m"]  = 1690.0
    p["L"]  = 2.741     # G42 is shorter
    p["a"]  = 1.33
    p["b"]  = p["L"] - p["a"]
    p["Iz"] = 2650.0
    return p


# whipple / carvallo bicycle model parameters
# naming follows Meijaard, Papadopoulos, Ruina & Schwab (2007), "Linearized dynamics
# equations for the balance and steer of a bicycle: a benchmark and review"
#   R = rear wheel, B = rear frame (+ rider), H = front frame (fork + bars), F = front wheel
#   x is forward, z is DOWN (so heights are negative), origin at rear contact point


def bike_benchmark():
    p = {}
    p["w"]    = 1.02         # wheelbase
    p["c"]    = 0.08         # trail
    p["lam"]  = 3.14159265358979 / 10.0   # steer axis tilt from vertical (18 deg)
    p["g"]    = G

    p["rR"], p["mR"], p["IRxx"], p["IRyy"] = 0.3, 2.0, 0.0603, 0.12

    p["xB"], p["zB"], p["mB"] = 0.3, -0.9, 85.0
    p["IBxx"], p["IByy"], p["IBzz"], p["IBxz"] = 9.2, 11.0, 2.8, 2.4

    p["xH"], p["zH"], p["mH"] = 0.9, -0.7, 4.0
    p["IHxx"], p["IHyy"], p["IHzz"], p["IHxz"] = 0.05892, 0.06, 0.00708, -0.00756

    p["rF"], p["mF"], p["IFxx"], p["IFyy"] = 0.35, 3.0, 0.1405, 0.28
    return p


def bike_sport():
    # ~190 kg wet 600cc-ish sport bike with a 75 kg rider sitting upright.
    # NONE of these are measured, they're ballpark numbers from spec sheets
    # (wheelbase, rake, trail, wheel sizes) and rough guesses for the inertias.
    # the whipple model also ignores tire slip and frame flex, which matter a lot
    # more on a motorcycle than on a pushbike. treat results as qualitative.
    deg = 3.14159265358979 / 180.0
    p = {}
    p["w"]    = 1.40
    p["c"]    = 0.098
    p["lam"]  = 24.0 * deg
    p["g"]    = G

    # 180/55-17 rear, 120/70-17 front. wheel + tire + brake rotor
    p["rR"], p["mR"], p["IRxx"], p["IRyy"] = 0.315, 14.0, 0.40, 0.75
    p["rF"], p["mF"], p["IFxx"], p["IFyy"] = 0.300, 11.0, 0.30, 0.55

    # frame + engine + rider lumped together
    p["xB"], p["zB"], p["mB"] = 0.62, -0.72, 228.0
    p["IBxx"], p["IByy"], p["IBzz"], p["IBxz"] = 22.0, 45.0, 28.0, 4.0

    # fork, triple clamps, bars
    p["xH"], p["zH"], p["mH"] = 1.22, -0.65, 12.0
    p["IHxx"], p["IHyy"], p["IHzz"], p["IHxz"] = 0.80, 0.80, 0.25, -0.20
    return p
