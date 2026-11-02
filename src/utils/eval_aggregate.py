from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

_WEIGHTED_METRIC_KEYS = (
    "ekf_ate_3d",
    "ekf_rte_3d",
    "net_ate_3d",
    "net_rte_3d",
    "ekf_ate_2d",
    "ekf_rte_2d",
    "net_ate_2d",
    "net_rte_2d",
)


def gt_path_length_3d_m(positions: np.ndarray) -> float:
    if positions is None or len(positions) < 2:
        return 0.0
    p = np.asarray(positions, dtype=np.float64)
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def weighted_mean_by_gt_path(
    entries: Sequence[Mapping[str, float]],
    *,
    weight_key: str = "gt_path_m",
) -> Tuple[Optional[Dict[str, float]], float, int]:
    valid = [e for e in entries if float(e.get(weight_key, 0.0)) > 0]
    if not valid:
        return None, 0.0, 0

    total_w = sum(float(e[weight_key]) for e in valid)
    if total_w <= 0:
        return None, 0.0, len(valid)

    out: Dict[str, float] = {}
    for key in _WEIGHTED_METRIC_KEYS:
        if key not in valid[0]:
            continue
        out[key] = sum(float(e[key]) * float(e[weight_key]) for e in valid) / total_w

    return out, total_w, len(valid)


def group_metrics_for_summary(
    all_metrics: Iterable[Mapping],
) -> List[Tuple[str, str, List[Mapping]]]:
    groups: Dict[Tuple[str, str], List[Mapping]] = {}
    for m in all_metrics:
        key = (str(m["carrier"]), str(m.get("ckpt", "-")))
        groups.setdefault(key, []).append(m)
    return [(c, ck, groups[(c, ck)]) for c, ck in sorted(groups.keys())]


def format_path_weighted_summary_line(
    carrier: str,
    ckpt: str,
    averages: Dict[str, float],
    *,
    total_path_m: float,
    n_segments: int,
    ckpt_any: bool,
) -> str:
    label = f"{carrier}"
    if ckpt_any:
        label = f"{carrier} | {ckpt[:16]}"
    return (
        f"{label:<30} (path-weighted, n={n_segments}, GT path={total_path_m:.2f} m) | "
        f"EKF ATE {averages['ekf_ate_3d']:.4f} RTE {averages['ekf_rte_3d']:.4f} | "
        f"NET ATE {averages['net_ate_3d']:.4f} RTE {averages['net_rte_3d']:.4f}"
    )
