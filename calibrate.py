"""字幕时间轴 ↔ 音频时间轴 的偏移标定。

实测结论：两者只差一个固定偏移（斜率恒为 1.00000），所以标定一个 offset
即可。算法用「残差直方图求粗解 + 精细扫描」，复杂度 O(边界数 × 停顿数)。
"""

from __future__ import annotations

import bisect
from typing import List, Optional, Sequence, Tuple


def _nearest_dist(t: float, centers: Sequence[float]) -> float:
    if not centers:
        return 1e18
    i = bisect.bisect_left(centers, t)
    best = 1e18
    for j in (i - 1, i):
        if 0 <= j < len(centers):
            best = min(best, abs(centers[j] - t))
    return best


def _near(t: float, centers: Sequence[float], tol: float) -> bool:
    return _nearest_dist(t, centers) <= tol


def fit_offset(
    boundary_s: Sequence[float],
    silences: Sequence[Tuple[float, float]],
    budget: float = 30.0,
    bin_size: float = 0.02,
    tol: float = 0.12,
) -> Tuple[float, float]:
    """返回 (offset, 命中率)。音频时间 = 字幕时间 + offset。"""
    centers = sorted((s + e) / 2.0 for s, e in silences)
    if not centers or not boundary_s:
        return 0.0, 0.0

    hist = {}
    for b in boundary_s:
        for c in centers:
            r = c - b
            if -budget <= r <= budget:
                k = int(round(r / bin_size))
                hist[k] = hist.get(k, 0) + 1
    if not hist:
        return 0.0, 0.0
    coarse = max(hist, key=lambda k: hist[k]) * bin_size

    # 精细扫描用连续评分（距离越近分越高），避免命中数平台的平局任意取边值
    step = 0.002
    span = int(0.4 / step)
    best_off, best_score = coarse, -1.0
    for i in range(-span, span + 1):
        off = coarse + i * step
        score = 0.0
        for b in boundary_s:
            d = _nearest_dist(b + off, centers)
            if d < tol:
                score += 1.0 - d / tol
        if score > best_score:
            best_off, best_score = off, score
    hit = sum(1 for b in boundary_s if _nearest_dist(b + best_off, centers) <= tol)
    return best_off, hit / len(boundary_s)


def hit_rate(
    boundary_s: Sequence[float],
    silences: Sequence[Tuple[float, float]],
    offset: float,
    tol: float = 0.12,
) -> float:
    """字幕切口「落在停顿区间内（含 ±tol 余量）」的比例 —— 判断对齐好不好的实用指标。

    与 fit_offset 内部用的「距停顿中心距离」判据不同：长停顿的中心离切口本来就远，
    用中心判据会低估对齐质量，所以报告统一用这个区间判据。
    """
    if not boundary_s or not silences:
        return 0.0
    sil = sorted(silences)
    ok = 0
    j = 0
    for b in boundary_s:
        t = b + offset
        while j < len(sil) and sil[j][1] + tol < t:
            j += 1
        k = j
        while k < len(sil) and sil[k][0] - tol <= t:
            if t <= sil[k][1] + tol:
                ok += 1
                break
            k += 1
    return ok / len(boundary_s)


def chance_level(
    boundary_s: Sequence[float],
    silences: Sequence[Tuple[float, float]],
    avoid_s: float,
    trial_offsets: Sequence[float] = (-25, -20, -15, -11, -6, -3, 3, 6, 11, 15, 20, 25),
    tol: float = 0.12,
) -> float:
    """「对齐错了」时的命中率基准（取若干错误偏移的中位数）。

    有它才能说明标定结果不是巧合：真实偏移的命中率应该明显高于这个基准。
    """
    trials = []
    for o in trial_offsets:
        if abs(o - avoid_s) < 1.5:      # 别把真实偏移自己算进基准
            continue
        trials.append(hit_rate(boundary_s, silences, o, tol))
    if not trials:
        return 0.0
    trials.sort()
    return trials[len(trials) // 2]


def check_drift(
    boundary_s: Sequence[float],
    silences: Sequence[Tuple[float, float]],
    offset: float,
    tol: float = 0.12,
) -> Optional[float]:
    """用命中的边界做线性回归，返回斜率。接近 1.0 表示无漂移。

    返回 None 表示命中点太少，无法判断。
    """
    centers = sorted((s + e) / 2.0 for s, e in silences)
    if not centers or not boundary_s:
        return None
    xs, ys = [], []
    for b in boundary_s:
        t = b + offset
        i = bisect.bisect_left(centers, t)
        cands = [j for j in (i - 1, i) if 0 <= j < len(centers)]
        if not cands:
            continue
        j = min(cands, key=lambda j: abs(centers[j] - t))
        if abs(centers[j] - t) <= tol:
            xs.append(b)
            ys.append(centers[j])
    if len(xs) < 20:
        return None
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den else None
