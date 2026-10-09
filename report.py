"""指标统计与对照报告。"""

from __future__ import annotations

import bisect
import statistics
from typing import Dict, List, Sequence, Tuple

from .segmenter import Config
from .textio import Cue

Interval = Tuple[float, float]


def _nearest(t: float, centers: Sequence[float]) -> float:
    if not centers:
        return 1e18
    i = bisect.bisect_left(centers, t)
    best = 1e18
    for j in (i - 1, i):
        if 0 <= j < len(centers):
            best = min(best, abs(centers[j] - t))
    return best


def hit_rate(boundary_s: Sequence[float], silences: Sequence[Interval], tol: float = 0.12):
    centers = sorted((s + e) / 2.0 for s, e in silences)
    if not centers or not boundary_s:
        return 0, 0, 0.0
    hit = sum(1 for b in boundary_s if _nearest(b, centers) <= tol)
    return hit, len(boundary_s), hit / len(boundary_s)


def _pct(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    i = min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))
    return xs[i]


def classify(
    lines: Sequence[Tuple[int, int]], chars: Sequence[Dict]
) -> List[Tuple[str, object]]:
    cue_range: Dict[int, List[int]] = {}
    for idx, ch in enumerate(chars):
        c = ch["cue"]
        if c not in cue_range:
            cue_range[c] = [idx, idx]
        else:
            cue_range[c][1] = idx
    kinds: List[Tuple[str, object]] = []
    for a, b in lines:
        cues_in = sorted({chars[j]["cue"] for j in range(a, b)})
        if len(cues_in) == 1:
            r = cue_range[cues_in[0]]
            kinds.append(("keep" if (a, b) == (r[0], r[1] + 1) else "split", cues_in[0]))
        else:
            kinds.append(("merge", tuple(cues_in)))
    return kinds


def build_report(
    title: str,
    input_cues: Sequence[Cue],
    output_cues: Sequence[Cue],
    chars: Sequence[Dict],
    lines: Sequence[Tuple[int, int]],
    silences: Sequence[Interval],
    offset: float,
    cfg: Config,
    has_script: bool,
    punct_breaks: int,
    audio_duration: float,
    suspicious: Sequence[int],
) -> str:
    off = offset
    in_b = [c.end_ms / 1000.0 for c in input_cues[:-1]]
    # chars 里的时间已经是音频时间轴，不能再加偏移
    out_b = [
        (chars[a]["start_ms"] + chars[a - 1]["end_ms"]) / 2.0 / 1000.0
        for a, _ in lines[1:]
    ] if len(lines) > 1 else []
    in_b_audio = [b + off for b in in_b]
    out_b_audio = out_b

    ih, it, ir = hit_rate(in_b_audio, silences)
    oh, ot, orr = hit_rate(out_b_audio, silences)
    n_chars_in = sum(len([c for c in c.text if not c.isspace()]) for c in input_cues)
    n_chars_out = sum(len(c.text) for c in output_cues)
    lens = [len(c.text) for c in output_cues]
    durs = [c.duration_ms / 1000.0 for c in output_cues]
    kinds = classify(lines, chars)
    n_keep = sum(1 for k, _ in kinds if k == "keep")
    n_merge = sum(1 for k, _ in kinds if k == "merge")
    n_split = sum(1 for k, _ in kinds if k == "split")

    L: List[str] = []
    L.append(f"PauseFix 对照报告 — {title}")
    L.append("=" * 60)
    L.append("")
    L.append("【时间轴标定】")
    L.append(f"  音频时间 = 字幕时间 + ({off:+.2f}) s")
    L.append(f"  音频总长 {audio_duration:.1f}s；字幕覆盖末行 {input_cues[-1].end_ms/1000:.2f}s")
    L.append("")
    L.append("【断句质量（断句点落在真实停顿 ±120ms 的比例）】")
    L.append(f"  修改前：{ih}/{it} = {ir*100:.1f}%")
    L.append(f"  修改后：{oh}/{ot} = {orr*100:.1f}%")
    L.append("")
    L.append("【行结构】")
    L.append(f"  输入 {len(input_cues)} 行 → 输出 {len(output_cues)} 行")
    L.append(f"  沿用原行 {n_keep} ；合并 {n_merge} ；在行内另切 {n_split}")
    L.append(f"  原稿标点映射到 {punct_breaks} 个断点" + ("" if has_script else "（未提供原稿）"))
    L.append("")
    L.append("【每行字数】")
    L.append(
        f"  中位 {statistics.median(lens):.0f}；最小 {min(lens)}；最大 {max(lens)}；"
        f" 超过 {cfg.max_chars} 字的行 {sum(1 for x in lens if x > cfg.max_chars)} 条"
    )
    L.append("【每行时长(秒)】")
    L.append(
        f"  中位 {statistics.median(durs):.2f}；最短 {min(durs):.2f}；最长 {max(durs):.2f}"
    )
    L.append("")
    L.append("【文字一致性】")
    L.append(
        f"  输入字符 {n_chars_in} → 输出字符 {n_chars_out}："
        + ("一致 ✓" if n_chars_in == n_chars_out else "不一致 ✗")
    )
    L.append("")
    L.append(f"【需要人工看一眼的行（断点不在停顿上的）】共 {len(suspicious)} 条")
    for i in list(suspicious)[:40]:
        c = output_cues[i]
        L.append(
            f"  #{i+1} {c.start_ms/1000:7.2f}-{c.end_ms/1000:7.2f}s  {c.text}"
        )
    if len(suspicious) > 40:
        L.append(f"  ...（其余 {len(suspicious)-40} 条省略）")
    L.append("")
    L.append("【输出前 60 行预览】")
    for i, c in enumerate(output_cues[:60]):
        mark = "⚠" if i in set(suspicious) else " "
        L.append(
            f"  {mark} {i+1:4d} {c.start_ms/1000:7.2f} → {c.end_ms/1000:7.2f}"
            f" ({c.duration_ms/1000:.2f}s) {len(c.text):2d}字  {c.text}"
        )
    return "\n".join(L)
