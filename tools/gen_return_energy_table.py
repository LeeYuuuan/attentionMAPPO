# tools/gen_return_energy_table.py
from __future__ import annotations
import os
import numpy as np

# IMPORTANT: keep consistent with EnergyModelConfig defaults
DEFAULT_OUT = "checkpoints/energy_tables/return_energy_table.npz"

def compute_return_energy_frac(
    d_xy: float,
    *,
    v_horiz: float,
    P_horiz: float,
    battery_capacity_Wh: float,
    airship_alt: float,
    uav_alt: float,
    P_climb: float | None = None,
) -> float:
    """
    Slanted return:
      d_3d = sqrt(d_xy^2 + dh^2)
      t = d_3d / v
      E = P * t
    Where P = P_climb if dh>0 and provided, else P_horiz.
    """
    d_xy = float(max(d_xy, 0.0))
    dh = float(airship_alt) - float(uav_alt)
    d_3d = float(np.sqrt(d_xy * d_xy + dh * dh))

    t_h = (d_3d / float(v_horiz)) / 3600.0
    if dh > 0.0 and P_climb is not None:
        P = float(P_climb)
    else:
        P = float(P_horiz)

    E_Wh = P * t_h
    return float(E_Wh / float(battery_capacity_Wh))


def main():
    # ---- You should keep these consistent with your EnergyModelConfig used in experiments ----
    P_horiz = 257.0
    v_horiz = 10.0
    battery_capacity_Wh = 125.0

    # altitude difference (set them to your paper/sim settings)
    airship_alt = 50.0
    uav_alt = 0.0

    # optional (if you want extra cost for climbing)
    P_climb = None  # e.g., 350.0

    # table range/resolution
    d_max = 600.0   # should cover your world; 400x400 => diagonal ~ 566
    d_step = 1.0    # 1 unit resolution; you can use 5.0 to make smaller file

    dist = np.arange(0.0, d_max + d_step, d_step, dtype=np.float32)
    frac = np.zeros_like(dist, dtype=np.float32)

    for i, d in enumerate(dist):
        frac[i] = compute_return_energy_frac(
            float(d),
            v_horiz=v_horiz,
            P_horiz=P_horiz,
            battery_capacity_Wh=battery_capacity_Wh,
            airship_alt=airship_alt,
            uav_alt=uav_alt,
            P_climb=P_climb,
        )

    meta = {
        "P_horiz": P_horiz,
        "v_horiz": v_horiz,
        "battery_capacity_Wh": battery_capacity_Wh,
        "airship_alt": airship_alt,
        "uav_alt": uav_alt,
        "P_climb": P_climb,
        "d_max": d_max,
        "d_step": d_step,
        "note": "return energy fraction lookup vs horizontal distance; slanted path includes altitude difference",
    }

    out_path = DEFAULT_OUT
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, dist=dist, frac=frac, meta=np.array(meta, dtype=object))
    print("[OK] saved:", out_path)
    print("  dist:", dist.shape, "frac:", frac.shape)
    print("  max_frac:", float(frac.max()))


if __name__ == "__main__":
    main()
