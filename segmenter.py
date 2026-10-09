"""断点评分 + 动态规划切分。

评分由三部分构成：
  1. 停顿分 —— 切点区间 [上一字结束, 下一字开始] 是否覆盖到一个真实停顿
  2. 标点分 —— 是否落在原稿标点处
  3. 词边界分 —— 是否是 jieba 分出的词边界（防止切在「被|多收了钱」中间）
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .zh import NO_LINE_END, NO_LINE_START


@dataclass
class Config:
    max_chars: int = 16
    min_chars: int = 5
    min_dur: float = 0.8
    max_dur: float = 6.5
    target_chars: int = 12
    w_pause: float = 2.4
    w_dip: float = 1.0
    w_punct_strong: float = 1.0
    w_punct_weak: float = 0.5
    w_word: float = 0.5
    w_bad_end: float = 0.35
    w_bad_start: float = 1.2
    w_length: float = 4.0
    split_penalty: float = 0.0
    pause_full_ms: float = 300.0
    pause_match_tol_ms: float = 250.0
    min_flash_ms: float = 400.0


def pause_cut_scores(
    chars: Sequence[Dict],
    silences_ms: Sequence[Tuple[float, float]],
    cfg: Config,
) -> List[float]:
    """ps[k] = 在第 k 个字之后切，能吃到的停顿分（0~1）。

    切点 k 对应的时间区间是 [chars[k-1].end_ms, chars[k].start_ms]：
    只要这个区间覆盖到某个停顿，就认为「切在这里正好切在停顿上」。
    """
    n = len(chars)
    ps = [0.0] * (n + 1)
    if not silences_ms or n < 2:
        return ps
    lo = [chars[k - 1]["end_ms"] for k in range(1, n)]
    hi = [chars[k]["start_ms"] for k in range(1, n)]

    for s, e in silences_ms:
        j1 = bisect.bisect_left(hi, s)
        j2 = bisect.bisect_left(lo, e)
        cands = set()
        for j in range(max(0, j1 - 2), min(len(lo), j1 + 3)):
            cands.add(j)
        for j in range(max(0, j2 - 2), min(len(lo), j2 + 3)):
            cands.add(j)
        best_mm, best_k = None, None
        for j in cands:
            mm = max(0.0, s - hi[j]) + max(0.0, lo[j] - e)
            if best_mm is None or mm < best_mm:
                best_mm, best_k = mm, j + 1
        if best_k is None:
            continue
        strength = min(1.0, (e - s) / cfg.pause_full_ms) * max(
            0.0, 1.0 - best_mm / cfg.pause_match_tol_ms
        )
        if strength > ps[best_k]:
            ps[best_k] = strength
    return ps


def boundary_scores(
    chars: Sequence[Dict],
    pause_scores: Sequence[float],
    break_pos: Dict[int, int],
    word_cuts: Optional[Set[int]],
    cfg: Config,
    dip_scores: Optional[Sequence[float]] = None,
) -> List[float]:
    """bscore[k] = 在第 k 个字之后断开的得分（k=0 与 k=len 恒为 0）。"""
    n = len(chars)
    b = [0.0] * (n + 1)
    for k in range(1, n):
        level = break_pos.get(k, 0)
        if level >= 3:
            pv = cfg.w_punct_strong
        elif level >= 2:
            pv = cfg.w_punct_weak
        else:
            pv = 0.0
        wv = cfg.w_word if (word_cuts is None or k in word_cuts) else 0.0
        pen = cfg.w_bad_end if chars[k - 1]["ch"] in NO_LINE_END else 0.0
        if k < n and chars[k]["ch"] in NO_LINE_START:
            pen += cfg.w_bad_start
        dv = cfg.w_dip * dip_scores[k] if dip_scores is not None else 0.0
        b[k] = cfg.w_pause * pause_scores[k] + dv + pv + wv - pen
    return b


def _line_penalty(chars: Sequence[Dict], k: int, j: int, cfg: Config) -> float:
    """行长越偏离理想值扣分越多 —— 这一项决定「一共切几行」。"""
    n = j - k
    pen = -cfg.split_penalty
    dev = (n - cfg.target_chars) / float(cfg.target_chars)
    pen -= cfg.w_length * dev * dev
    dur_ms = chars[j - 1]["end_ms"] - chars[k]["start_ms"]
    lo = cfg.min_dur * 1000.0
    hi = cfg.max_dur * 1000.0
    if dur_ms < lo:
        pen -= (lo - dur_ms) / lo * 0.6
    if dur_ms > hi:
        pen -= (dur_ms - hi) / hi * 0.8
    return pen


def segment(
    chars: Sequence[Dict], bscore: Sequence[float], cfg: Config
) -> List[Tuple[int, int]]:
    """返回 [(起始字符下标, 结束字符下标), ...]（左闭右开）。"""
    n = len(chars)
    if n == 0:
        return []
    if n <= cfg.max_chars:
        return [(0, n)]

    NEG = -1e18
    dp = [NEG] * (n + 1)
    back = [-1] * (n + 1)
    dp[0] = 0.0
    for j in range(1, n + 1):
        hi = j - cfg.min_chars
        if hi < 0:
            continue
        lo = max(0, j - cfg.max_chars)
        for k in range(lo, hi + 1):
            if dp[k] <= NEG / 2:
                continue
            sc = dp[k] + bscore[k] + _line_penalty(chars, k, j, cfg)
            if sc > dp[j]:
                dp[j] = sc
                back[j] = k
    if back[n] < 0:
        return [(0, n)]

    lines: List[Tuple[int, int]] = []
    j = n
    while j > 0:
        k = back[j]
        lines.append((k, j))
        j = k
    lines.reverse()
    return lines
