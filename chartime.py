"""逐字时间轴：把每条字幕的文字，按行内「语音时长占比」铺开到音频时间上。

关键点：行内如果有停顿，停顿时间不参与分配字符时间，所以字符边界会天然
落在停顿的边缘上。
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from .textio import Cue

Interval = Tuple[float, float]


def subtract(
    a0: float, a1: float, silences_ms: Sequence[Interval]
) -> List[Interval]:
    """在 [a0, a1] 内挖掉所有静音段，返回剩下的语音段（单位 ms）。"""
    subs: List[Interval] = []
    cur = a0
    for s, e in silences_ms:
        if e <= a0:
            continue
        if s >= a1:
            break
        s2 = max(s, a0)
        e2 = min(e, a1)
        if s2 > cur:
            subs.append((cur, s2))
        cur = max(cur, e2)
    if cur < a1:
        subs.append((cur, a1))
    return [(s, e) for s, e in subs if e - s > 1e-6]


def build(
    cues: Sequence[Cue],
    silences: Sequence[Interval],
    offset_s: float,
) -> List[Dict]:
    """返回 [{ch, start_ms, end_ms, cue}, ...]，时间均为音频时间轴。"""
    sil_ms = sorted((s * 1000.0, e * 1000.0) for s, e in silences)
    off_ms = offset_s * 1000.0
    chars: List[Dict] = []

    for ci, cue in enumerate(cues):
        text_chars = [ch for ch in cue.text if not ch.isspace()]
        if not text_chars:
            continue
        a0 = cue.start_ms + off_ms
        a1 = cue.end_ms + off_ms
        if a1 <= a0:
            a1 = a0 + 1.0
        subs = subtract(a0, a1, sil_ms)
        if not subs:
            subs = [(a0, a1)]
        total = sum(e - s for s, e in subs)
        n = len(text_chars)

        def pos(frac: float) -> float:
            t = frac * total
            for s, e in subs:
                d = e - s
                if t <= d:
                    return s + t
                t -= d
            return subs[-1][1]

        for j, ch in enumerate(text_chars):
            chars.append(
                {
                    "ch": ch,
                    "start_ms": pos(j / n),
                    "end_ms": pos((j + 1) / n),
                    "cue": ci,
                }
            )
    return chars
