"""
Phenology layer: cumulative growing degree days -> crop stage.

Rule-based and deterministic on purpose. GDD staging is standard agronomy; an ML
stage classifier would add error without adding capability, and we have no
stage-labelled ground truth to train it on. This is a defensible choice, not a shortcut.
"""
from __future__ import annotations
import numpy as np


def daily_gdd(tmax, tmin, t_base, t_cap):
    tmax = np.minimum(tmax, t_cap)
    tmin = np.minimum(np.maximum(tmin, t_base), t_cap)
    tmax = np.maximum(tmax, t_base)
    return np.maximum((tmax + tmin) / 2.0 - t_base, 0.0)


def cumulative_gdd(tmax, tmin, cfg):
    p = cfg["phenology"]
    return np.cumsum(daily_gdd(np.asarray(tmax), np.asarray(tmin), p["t_base"], p["t_cap"]))


def stage_from_gdd(cgdd, cfg, gdd_total=None):
    """Return (stage_name, stage_index, kc, gdd_fraction) arrays."""
    p = cfg["phenology"]
    frac = np.asarray(cgdd) / (gdd_total or p["gdd_total"])
    names, idx, kcs = [], [], []
    for f in frac:
        hit = p["stages"][-1]
        hit_i = len(p["stages"]) - 1
        for i, s in enumerate(p["stages"]):
            if s["f_start"] <= f < s["f_end"]:
                hit, hit_i = s, i
                break
        names.append(hit["name"]); idx.append(hit_i); kcs.append(hit["kc"])
    return np.array(names), np.array(idx), np.array(kcs), frac
