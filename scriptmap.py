"""把原稿里的标点位置，映射到字幕文字（无标点）的字符位置上。

返回 {break_pos: level}：
  break_pos = 这个断点前面有多少个字符（也就是「在第 k 个字之后断」）
  level: 3 = 强标点（。！？…），2 = 弱标点（，、；：）
"""

from __future__ import annotations

import difflib
from typing import Dict

STRONG = set("。！？!?…")
WEAK = set("，,、：:；;")


def _clean(s: str) -> str:
    return "".join(ch for ch in s if not ch.isspace())


def map_punct(srt_text: str, script_text: str) -> Dict[int, int]:
    a = _clean(srt_text)
    b = _clean(script_text)
    if not a or not b:
        return {}

    b2a: Dict[int, int] = {}
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                b2a[j1 + k] = i1 + k

    out: Dict[int, int] = {}
    for j, ch in enumerate(b):
        if ch in STRONG:
            level = 3
        elif ch in WEAK:
            level = 2
        else:
            continue
        # 找到标点前最后一个能对上字幕的字符
        jj = j - 1
        while jj >= 0 and jj not in b2a:
            jj -= 1
        if jj < 0:
            continue
        k = b2a[jj] + 1  # 断在字幕第 k 个字符之后
        if 0 < k < len(a):
            if out.get(k, 0) < level:
                out[k] = level
    return out
