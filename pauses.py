"""停顿（静音段）检测：调用 ffmpeg 的 silencedetect 滤波。"""

from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
from typing import List, Optional, Sequence, Tuple

_S_START = re.compile(r"silence_start:\s*(-?[\d.]+)")
_S_END = re.compile(r"silence_end:\s*(-?[\d.]+)")
_DUR = re.compile(r"Duration:\s*(\d+):(\d+):(\d+)\.(\d+)")

Interval = Tuple[float, float]


def find_ffmpeg() -> str:
    """按 环境变量 → 项目目录 → PATH → hermes 自带 的顺序找 ffmpeg。"""
    env = os.environ.get("PAUSEFIX_FFMPEG")
    if env and os.path.isfile(env):
        return env
    # 放在项目里最省事（换电脑不用配 PATH）：根目录 / tools 下 / ffmpeg*/bin 下都认
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for pat in (
        os.path.join(root, "ffmpeg.exe"),
        os.path.join(root, "ffmpeg", "ffmpeg.exe"),
        os.path.join(root, "ffmpeg", "bin", "ffmpeg.exe"),
        os.path.join(root, "tools", "ffmpeg.exe"),
        os.path.join(root, "tools", "ffmpeg*", "bin", "ffmpeg.exe"),
    ):
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[-1]
    found = shutil.which("ffmpeg")
    if found:
        return found
    patterns = [
        os.path.expanduser(
            r"~\AppData\Local\hermes\tools\ffmpeg-*\bin\ffmpeg.exe"
        ),
        r"E:\hermes-workspace\ffmpeg\bin\ffmpeg.exe",
        r"E:\hermes-workspace\tools\ffmpeg*\bin\ffmpeg.exe",
    ]
    for pat in patterns:
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[-1]
    raise FileNotFoundError(
        "找不到 ffmpeg。请把 ffmpeg.exe 放进 PATH，或设置环境变量 PAUSEFIX_FFMPEG。"
    )


def _parse_duration(log: str) -> Optional[float]:
    m = _DUR.search(log)
    if not m:
        return None
    h, mi, s, frac = m.groups()
    return int(h) * 3600 + int(mi) * 60 + int(s) + int(frac) / (10 ** len(frac))


def detect(
    audio: str,
    db: float = -35.0,
    min_silence: float = 0.10,
    ffmpeg: Optional[str] = None,
) -> Tuple[List[Interval], float]:
    """返回 (静音段列表, 音频总时长秒)。静音段为 [开始, 结束]，单位秒。"""
    exe = ffmpeg or find_ffmpeg()
    cmd = [
        exe, "-hide_banner", "-nostdin", "-i", audio,
        "-af", f"silencedetect=noise={db}dB:d={min_silence}",
        "-f", "null", "-",
    ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    log = proc.stderr or ""
    starts = [float(m.group(1)) for m in _S_START.finditer(log)]
    ends = [float(m.group(1)) for m in _S_END.finditer(log)]
    silences: List[Interval] = []
    for i, s in enumerate(starts):
        e = ends[i] if i < len(ends) else None
        if e is None:
            continue
        if e - s <= 0:
            continue
        silences.append((max(0.0, s), e))
    silences.sort()
    dur = _parse_duration(log)
    if dur is None:
        dur = silences[-1][1] if silences else 0.0
    return silences, dur


def speech_regions(silences: Sequence[Interval], duration: float) -> List[Interval]:
    """静音段的补集 = 语音段。"""
    regions: List[Interval] = []
    cur = 0.0
    for s, e in silences:
        s = max(0.0, min(s, duration))
        e = max(0.0, min(e, duration))
        if e <= s:
            continue
        if s > cur:
            regions.append((cur, s))
        cur = max(cur, e)
    if cur < duration:
        regions.append((cur, duration))
    return regions
