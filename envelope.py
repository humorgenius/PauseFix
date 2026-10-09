"""音频能量包络（波形低谷）分析。

用 20ms 窗的短时 RMS 描述波形。切点打分不看"有没有静音"，而是看
"刀口那个瞬间是不是波形谷底"——这正是人对着波形手动断句时的判断依据。
"""

from __future__ import annotations

import subprocess
from typing import List, Sequence, Tuple

import numpy as np

from .pauses import find_ffmpeg

HOP_MS = 20.0


def load_envelope(audio: str, hop_ms: float = HOP_MS) -> Tuple[np.ndarray, float]:
    """返回 (rms, hop_ms)；rms 是线性幅度。"""
    exe = find_ffmpeg()
    cmd = [
        exe, "-hide_banner", "-nostdin", "-i", audio,
        "-ac", "1", "-ar", "8000", "-f", "s16le", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    a = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32)
    n = max(1, int(8000 * hop_ms / 1000.0))
    m = len(a) // n
    if m == 0:
        return np.zeros(1, dtype=np.float32), hop_ms
    a = a[: m * n].reshape(m, n)
    return np.sqrt((a ** 2).mean(axis=1)), hop_ms


class DipIndex:
    """波形低谷查询。"""

    def __init__(self, rms: np.ndarray, hop_ms: float = HOP_MS, half_ms: float = 500.0):
        self.rms = rms
        self.hop_ms = hop_ms
        self.half = max(1, int(half_ms / hop_ms))
        # 局部峰值：以 ±half 为窗的滑动最大值（用前缀/后缀最大值近似，够快）
        n = len(rms)
        self.locmax = np.empty(n, dtype=np.float32)
        # 分块算滑动最大，块长 = half
        step = self.half
        for start in range(0, n, step):
            end = min(n, start + 2 * step)
            self.locmax[start:end] = rms[max(0, start - step): min(n, end + step)].max()

    def _idx(self, t_ms: float) -> int:
        return int(min(max(0, round(t_ms / self.hop_ms)), len(self.rms) - 1))

    def dip_at(self, t_ms: float) -> float:
        """该时刻的能量 / 局部峰值，0~1，越小越接近谷底。"""
        i = self._idx(t_ms)
        m = float(self.locmax[i])
        if m <= 1e-6:
            return 0.0
        return float(self.rms[i]) / m

    def min_dip(self, lo_ms: float, hi_ms: float) -> float:
        """一个时间区间内的最佳（最小）谷底值。"""
        i0 = self._idx(lo_ms)
        i1 = self._idx(hi_ms)
        if i1 < i0:
            i0, i1 = i1, i0
        seg = self.rms[i0: i1 + 1]
        if len(seg) == 0:
            return 1.0
        j = i0 + int(np.argmin(seg))
        m = float(self.locmax[j])
        if m <= 1e-6:
            return 0.0
        return float(seg.min()) / m


def dip_cut_scores(
    chars: Sequence[dict],
    dip: DipIndex,
    full: float = 0.55,
) -> List[float]:
    """dscore[k] = 在第 k 个字之后切的「谷底分」（0~1，越大越是谷底）。"""
    n = len(chars)
    out = [0.0] * (n + 1)
    for k in range(1, n):
        lo = chars[k - 1]["end_ms"]
        hi = chars[k]["start_ms"]
        d = dip.min_dip(lo, hi)
        out[k] = max(0.0, 1.0 - d / full)
    return out
