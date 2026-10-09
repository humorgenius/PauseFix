"""SRT 解析 / 写出，以及编码自适应的文本读取。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "gbk")


@dataclass
class Cue:
    index: int
    start_ms: float
    end_ms: float
    text: str

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms


def read_text(path: str) -> str:
    """按 utf-8-sig / utf-8 / gb18030 / gbk 依次尝试读取。"""
    last_err = None
    for enc in _ENCODINGS:
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except (UnicodeDecodeError, LookupError) as e:  # noqa: PERF203
            last_err = e
            continue
    # 最后兜底：不因编码问题整个失败
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


_TS = re.compile(
    r"(\d{1,2}):(\d{1,2}):(\d{1,2})[,.](\d{1,3})\s*-->\s*"
    r"(\d{1,2}):(\d{1,2}):(\d{1,2})[,.](\d{1,3})"
)


def _to_ms(h: str, m: str, s: str, ms: str) -> float:
    frac = ms.ljust(3, "0")[:3]
    return (int(h) * 3600 + int(m) * 60 + int(s)) * 1000 + int(frac)


def parse_srt(text: str, keep_lines: bool = False) -> List[Cue]:
    """解析 SRT / 剪映导出的 .txt 字幕（两者格式相同）。

    keep_lines=False（默认）：把多条正文行接成一行 —— 剪映导出的 .txt 会把长句折行，
    接起来才是原本的一句话。
    keep_lines=True：保留换行 —— .srt 里的换行是"一条字幕显示两行"，是刻意的，不能吞。
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    cues: List[Cue] = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = [ln for ln in block.split("\n") if ln.strip() != ""]
        if not lines:
            continue
        ts_at = next((i for i, ln in enumerate(lines) if "-->" in ln), None)
        if ts_at is None:
            continue
        m = _TS.search(lines[ts_at])
        if not m:
            continue
        g = m.groups()
        body = ("\n".join(lines[ts_at + 1:]) if keep_lines
                else "".join(lines[ts_at + 1:])).strip()
        cues.append(
            Cue(
                index=len(cues) + 1,
                start_ms=_to_ms(g[0], g[1], g[2], g[3]),
                end_ms=_to_ms(g[4], g[5], g[6], g[7]),
                text=body,
            )
        )
    return cues


def format_ts(ms: float) -> str:
    total = int(round(ms))
    if total < 0:
        total = 0
    h, rem = divmod(total, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, msec = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{msec:03d}"


def format_srt(cues: List[Cue]) -> str:
    out = []
    for i, c in enumerate(cues, 1):
        out.append(str(i))
        out.append(f"{format_ts(c.start_ms)} --> {format_ts(c.end_ms)}")
        out.append(c.text)
        out.append("")
    return "\n".join(out)


def visible_chars(s: str) -> str:
    """去掉所有空白，只留真正会被显示 / 计时的字符。"""
    return "".join(ch for ch in s if not ch.isspace())
