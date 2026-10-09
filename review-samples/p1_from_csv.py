# Where do the gaps land? Four Garmin recordings, tsgap 0.7.0.
#
# Hide 20% of heart rate, MAR with speed as the driver (or MNAR "high"),
# and compare the missing share in the top and bottom quartile of the driver.
# MCAR is the control. Blocks and Markov bursts are about 60 s long.
# 20 seeds per row. Just an illustration, not a statistical test.

import numpy as np
from tsgap import simulate_missingness


def run(name, X, t, col, mech, **opts):
    dt = np.median(np.diff(t))
    d = X[:, col]
    top = d > np.quantile(d, 0.75)
    bottom = d < np.quantile(d, 0.25)
    patterns = {
        "pointwise": {},
        "block": {"block_len": round(60 / dt)},
        "markov": {"persist": 1 - dt / 60},
    }

    print(name)
    for pattern, kw in patterns.items():
        for m in [mech, "mcar"]:
            extra = opts if m == mech else {}
            hi, lo = [], []
            for seed in range(20):
                _, mask = simulate_missingness(X, m, 0.2, seed=seed, target=[0],
                                               pattern=pattern, **kw, **extra)
                miss = ~mask[:, 0]
                hi.append(miss[top].mean())
                lo.append(miss[bottom].mean())
            diff = np.array(hi) - np.array(lo)
            print(f"  {pattern:9} {m:4}  top {np.mean(hi):.2f}  bottom {np.mean(lo):.2f}"
                  f"  diff {diff.mean():+.2f} ± {diff.std():.2f}")
    print()


def read(path):
    return np.genfromtxt(path, delimiter=",", names=True)


ride = read("ride.csv")
run("ride, MAR on speed", np.column_stack([ride["heart_rate"], ride["speed_kmh"]]),
    ride["t_s"], 1, "mar", driver_dims=[1])

ride2 = read("trail_run.csv")
run("ride 2, MAR on speed", np.column_stack([ride2["heart_rate"], ride2["speed_kmh"]]),
    ride2["t_s"], 1, "mar", driver_dims=[1])

hike = read("hike.csv")
speed = np.r_[0, np.diff(hike["distance_m"]) / np.maximum(np.diff(hike["t_s"]), 1) * 3.6]
run("hike, MAR on speed", np.column_stack([hike["heart_rate"], speed]),
    hike["t_s"], 1, "mar", driver_dims=[1])

gym = read("strength.csv")
run("strength, MNAR high", gym["heart_rate"][:, None], gym["t_s"], 0, "mnar", mnar_mode="high")
