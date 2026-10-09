"""时间吸附。

原则：字幕时间以「逐字时间轴」为准，只向附近的停顿边缘做小幅吸附；
绝不用整段语音区域去回拉 / 延长（那会让字幕显示时间跑飞）。
"""

from __future__ import annotations

import bisect
from typing import Dict, List, Optional, Sequence, Tuple

from .segmenter import Config
from .textio import Cue

Interval = Tuple[float, float]


class EdgeIndex:
    """停顿开始点 / 结束点的最近邻查询。"""

    def __init__(self, silences_ms: Sequence[Interval]):
        self.starts = sorted(s for s, _ in silences_ms)
        self.ends = sorted(e for _, e in silences_ms)

    @staticmethod
    def _nearest(points: Sequence[float], t: float, window: float) -> Optional[float]:
        if not points:
            return None
        i = bisect.bisect_left(points, t)
        best = None
        for j in (i - 1, i):
            if 0 <= j < len(points) and abs(points[j] - t) <= window:
                if best is None or abs(points[j] - t) < abs(best - t):
                    best = points[j]
        return best

    def nearest_start(self, t: float, window: float) -> Optional[float]:
        """离 t 最近的一个「停顿开始」时刻（上一句语音结束处）。"""
        return self._nearest(self.starts, t, window)

    def nearest_end(self, t: float, window: float) -> Optional[float]:
        """离 t 最近的一个「停顿结束」时刻（下一句语音开始处）。"""
        return self._nearest(self.ends, t, window)


def snap(
    lines: Sequence[Tuple[int, int]],
    chars: Sequence[Dict],
    silences_ms: Sequence[Interval],
    offset_ms: float,
    duration_ms: float,
    cfg: Config,
    lead_in_ms: float = 80.0,
    lead_out_ms: float = 60.0,
    snap_window_ms: float = 250.0,
    gap_ms: float = 20.0,
) -> List[Cue]:
    edges = EdgeIndex(silences_ms)
    spans: List[List[float]] = []
    for a, b in lines:
        t0 = chars[a]["start_ms"]
        t1 = chars[b - 1]["end_ms"]
        s = t0 - lead_in_ms
        e = t1 + lead_out_ms
        pe = edges.nearest_end(s, snap_window_ms)
        if pe is not None:
            s = pe - lead_in_ms
        ps = edges.nearest_start(e, snap_window_ms)
        if ps is not None:
            e = ps + lead_out_ms
        s = max(0.0, s)
        e = min(duration_ms, e)
        if e <= s:
            e = s + 1.0
        spans.append([s, e])

    flash = cfg.min_flash_ms
    for i, sp in enumerate(spans):
        if i > 0 and sp[0] < spans[i - 1][1] + gap_ms:
            sp[0] = spans[i - 1][1] + gap_ms
        if sp[1] - sp[0] < flash:
            nxt = spans[i + 1][0] - gap_ms if i + 1 < len(spans) else duration_ms
            target = min(sp[0] + flash, max(nxt, sp[0] + flash))
            sp[1] = min(max(sp[1], target), duration_ms)

    cues: List[Cue] = []
    for i, (a, b) in enumerate(lines):
        text = "".join(chars[j]["ch"] for j in range(a, b))
        cues.append(
            Cue(
                index=1 + i,
                start_ms=spans[i][0] - offset_ms,
                end_ms=spans[i][1] - offset_ms,
                text=text,
            )
        )
    return cues
