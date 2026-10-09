"""提案引擎（v2）：只找出「错的那 15%」的断句，其余一律不动。

核心思路：
  1. 逐个已有刀口判断是否切错（词被劈开 / 行末是虚词 / 波形连续无停顿）
  2. 对切错的地方做**局部小手术**：只在那一两行的小窗口里重排，窗口外不碰
  3. 每条提案带理由与置信度，用户逐条确认后才导出

未提案的行，文字和时间戳逐字节沿用输入。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from . import calibrate, chartime, envelope, pauses, scriptmap, segmenter, textio, words
from .segmenter import Config

# ── 判据阈值 ──
PAUSE_ABSENT = 0.02      # 停顿分低于此值 = 刀口处没有停顿
DIP_CONTINUOUS = 0.42    # 谷底分高于此值 = 波形连续（切在字的中间）
SPLIT_PAUSE = 0.80       # 拆分要求的停顿分
SPLIT_MIN_LEN = 13       # 行字数达到此值才考虑拆
SPLIT_PART_MIN = 5       # 拆出来的每一截至少几个字
PAUSE_LEN_MIN = 0.40     # 拆分只认 0.40s 以上的停顿
LOCAL_MIN_CHARS = 4      # 局部重排时允许的最短行（「都不知道」这种 4 字行）

# 行末绝不能出现的虚词：出现即说明词被拆开了
# （用户给的实例都落在这里：被/多收了钱、走廊里朝/国王、这也是为/什么）
NO_LINE_END = set(
    "被把朝向为因由从跟同和与给让使对就才都也还只很最更太不没别请并而但若如之乎者以及或则且却又再倘假虽即便"
)
# 行首绝不能出现（切在它们之前 = 把词拆开）
NO_LINE_START = set("的了着过地得是之们个吧吗呢啊呀哦嗯")


@dataclass
class Proposal:
    pid: int
    kind: str                 # merge / resegment / split
    rows: List[int]           # 涉及的 1-based 行号
    before: List[str]
    after: List[str]
    reason: str
    confidence: float
    times: List[List[float]] = field(default_factory=list)   # after 每行的 [start_ms, end_ms]（字幕时间轴）
    evidence: Dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "pid": self.pid, "kind": self.kind, "rows": self.rows,
            "before": self.before, "after": self.after, "reason": self.reason,
            "confidence": round(self.confidence, 2), "times": self.times,
            "evidence": self.evidence,
        }


class Analyzer:
    """音频 + 字幕 + 原稿 → 可复用的中间结果。"""

    def __init__(self, audio: str, srt_path: str, script_path: Optional[str] = None,
                 db: float = -30.0, min_silence: float = 0.10):
        self.audio = audio
        self.srt_path = srt_path
        self.script_path = script_path
        self.cues = textio.parse_srt(textio.read_text(srt_path))
        if not self.cues:
            raise ValueError("字幕解析为空")
        self.sil, self.duration = pauses.detect(audio, db=db, min_silence=min_silence)
        self.sil_ms = [(s * 1000.0, e * 1000.0) for s, e in self.sil]
        self.offset, self.before_hit = calibrate.fit_offset(
            [c.end_ms / 1000.0 for c in self.cues[:-1]], self.sil
        )
        self.chars = chartime.build(self.cues, self.sil, self.offset)
        self.text = "".join(c["ch"] for c in self.chars)
        rms, hop = envelope.load_envelope(audio)
        self.dip_index = envelope.DipIndex(rms, hop)
        cfg = Config()
        self.ps = segmenter.pause_cut_scores(self.chars, self.sil_ms, cfg)
        self.ds = envelope.dip_cut_scores(self.chars, self.dip_index)

        # 词边界：用「词典里的词是否被刀口劈开」这个精确判据
        n = len(self.text)
        self.spanned: List[Optional[str]] = [None] * (n + 1)
        word_ok: Set[int] = set()
        if words.available():
            for k in range(1, n):
                w = words.cut_inside_word(self.text, k)
                self.spanned[k] = w
                if w is None:
                    word_ok.add(k)
        self.word_cuts = word_ok if words.available() else None

        self.break_pos: Dict[int, int] = {}
        if script_path:
            self.break_pos = scriptmap.map_punct(
                "".join(c.text for c in self.cues), textio.read_text(script_path)
            )
        self.cue_chars: List[Tuple[int, int]] = []
        p = 0
        for c in self.cues:
            m = len([ch for ch in c.text if not ch.isspace()])
            self.cue_chars.append((p, p + m))
            p += m
        self.bs = segmenter.boundary_scores(
            self.chars, self.ps, self.break_pos, self.word_cuts, cfg, self.ds
        )

    # ── 工具 ──
    def pause_len_at(self, t_ms: float) -> float:
        for s, e in self.sil_ms:
            if s - 60 <= t_ms <= e + 60:
                return e - s
        return 0.0

    def cut_time_srt(self, k: int) -> float:
        """第 k 个字之后那个刀口的时刻（换算回字幕时间轴）。"""
        t_audio = (self.chars[k - 1]["end_ms"] + self.chars[k]["start_ms"]) / 2.0
        return t_audio - self.offset * 1000.0

    def break_info(self, i: int) -> Dict:
        k = self.cue_chars[i + 1][0]
        return {
            "k": k,
            "pause": self.ps[k] if k < len(self.ps) else 0.0,
            "dip": self.ds[k] if k < len(self.ds) else 0.0,
            "word": self.spanned[k] if k < len(self.spanned) else None,
            "punct": self.break_pos.get(k, 0),
            "left": self.cues[i].text,
            "right": self.cues[i + 1].text,
        }

    def internal_split_point(self, i: int) -> Optional[Dict]:
        a, b = self.cue_chars[i]
        best = None
        for k in range(a + 3, b - 2):
            if self.word_cuts is not None and k not in self.word_cuts:
                continue
            p = self.ps[k] if k < len(self.ps) else 0.0
            if p < SPLIT_PAUSE:
                continue
            t = (self.chars[k - 1]["end_ms"] + self.chars[k]["start_ms"]) / 2.0
            length = self.pause_len_at(t)
            if length < PAUSE_LEN_MIN * 1000:
                continue
            if self.text[k - 1] in NO_LINE_END or self.text[k] in NO_LINE_START:
                continue
            score = p + (self.ds[k] if k < len(self.ds) else 0.0) * 0.4
            if best is None or score > best["score"]:
                best = {"k": k, "time_ms": t, "pause": p, "len_ms": length, "score": score}
        return best


def _break_confidence(bi: Dict) -> Tuple[float, str]:
    left = bi["left"]
    if bi["word"]:
        return 0.95, f"把词「{bi['word']}」劈开了"
    if left and left[-1] in NO_LINE_END:
        return 0.95, f"上一行以虚词「{left[-1]}」结尾，词被拆开"
    if bi["punct"] > 0:
        return 0.0, "原稿在此处有标点，判定本来是对的"
    if bi["pause"] <= PAUSE_ABSENT:
        if bi["dip"] < DIP_CONTINUOUS:
            return 0.78, "刀口处波形连续、没有停顿"
        return 0.60, "刀口处没有停顿（波形只有轻微起伏）"
    return 0.0, "刀口落在停顿上，判定本来是对的"


def _local_resegment(an: Analyzer, win: List[int], max_chars: int, max_dur: float):
    """在 win 这几行的小窗口里重排断句，返回 (分段下标列表, 置信提示) 或 None。"""
    a = an.cue_chars[win[0]][0]
    b = an.cue_chars[win[-1]][1]
    total = b - a
    if total <= max_chars:
        # 完全放得下一行 → 就是合并
        return [(0, total)], ""
    cfg2 = Config(
        max_chars=max_chars, min_chars=LOCAL_MIN_CHARS,
        target_chars=min(12, max(LOCAL_MIN_CHARS, total // 2 + 1)),
        min_dur=0.0, max_dur=max_dur, w_dip=1.0,
    )
    sub = an.chars[a:b]
    sub_bs = an.bs[a:b]
    seg = segmenter.segment(sub, sub_bs, cfg2)
    return seg, ""


def build_proposals(an: Analyzer, max_chars: int = 16, max_dur: float = 6.5) -> List[Proposal]:
    props: List[Proposal] = []
    n = len(an.cues)

    # ① 找坏刀口
    bad: Dict[int, Tuple[float, str, Dict]] = {}
    for i in range(n - 1):
        bi = an.break_info(i)
        conf, reason = _break_confidence(bi)
        if conf > 0:
            bad[i] = (conf, reason, bi)

    # ② 连续坏刀口合成一个窗口 → 局部重排
    used: Set[int] = set()
    for i in range(n - 1):
        if i in used or i not in bad:
            continue
        win = [i, i + 1]
        breaks = [i]
        while win[-1] in bad and win[-1] < n - 1:
            breaks.append(win[-1])
            win.append(win[-1] + 1)
        used.update(breaks)

        before = [an.cues[j].text for j in win]
        if sum(len(t) for t in before) > max_chars * len(win):
            continue
        res = _local_resegment(an, win, max_chars, max_dur)
        if res is None:
            continue
        seg, _ = res
        orig = [(an.cue_chars[j][0] - an.cue_chars[win[0]][0],
                 an.cue_chars[j][1] - an.cue_chars[win[0]][0]) for j in win]
        if list(seg) == [tuple(x) for x in orig]:
            continue                     # 重排结果和原来一样 → 不改
        a0 = an.cue_chars[win[0]][0]
        after = ["".join(an.chars[j]["ch"] for j in range(a0 + s, a0 + e)) for s, e in seg]
        # 时间：首行沿用原起点，末行沿用原终点，中间按停顿点切
        times = []
        for idx, (s, e) in enumerate(seg):
            st = an.cues[win[0]].start_ms if idx == 0 else an.cut_time_srt(a0 + s)
            en = an.cues[win[-1]].end_ms if idx == len(seg) - 1 else an.cut_time_srt(a0 + e)
            times.append([round(st, 1), round(en, 1)])
        conf = min(bad[k][0] for k in breaks)
        reason = bad[breaks[0]][1]
        if len(breaks) > 1:
            reason += f"（连续 {len(breaks)} 处刀口切错）"
        kind = "merge" if len(seg) == 1 else "resegment"
        if kind == "resegment":
            reason += "；合并后过长，改为在停顿处重新切开"
        props.append(Proposal(0, kind, [j + 1 for j in win], before, after, reason,
                              conf, times,
                              {"k": bad[breaks[0]][2]["k"],
                               "pause": bad[breaks[0]][2]["pause"],
                               "dip": bad[breaks[0]][2]["dip"]}))

    # ③ 漏断的长行 → 拆分
    for i in range(n):
        a, b = an.cue_chars[i]
        if b - a < SPLIT_MIN_LEN:
            continue
        sp = an.internal_split_point(i)
        if not sp:
            continue
        k = sp["k"]
        if k - a < SPLIT_PART_MIN or b - k < SPLIT_PART_MIN:
            continue
        left = an.cues[i].text[: k - a]
        right = an.cues[i].text[k - a:]
        t = an.cut_time_srt(k)
        conf = 0.85 if sp["len_ms"] >= 450 else 0.7
        if b - a > max_chars:
            conf = min(0.95, conf + 0.1)
        props.append(Proposal(
            0, "split", [i + 1], [an.cues[i].text], [left, right],
            f"行内有 {sp['len_ms']/1000:.2f}s 的明显停顿（该行 {b-a} 字），剪映漏断",
            conf, [[round(an.cues[i].start_ms, 1), round(t, 1)],
                   [round(t, 1), round(an.cues[i].end_ms, 1)]],
            {"k": k, "pause": round(sp["pause"], 2), "len_ms": round(sp["len_ms"])}))

    # ④ 同一行不能既合又拆：保留置信度高的
    by_row: Dict[int, List[Proposal]] = {}
    for p in props:
        for r in p.rows:
            by_row.setdefault(r, []).append(p)
    drop: Set[int] = set()
    for r, lst in by_row.items():
        if len(lst) > 1:
            lst.sort(key=lambda p: -p.confidence)
            for p in lst[1:]:
                drop.add(id(p))
    props = [p for p in props if id(p) not in drop]
    for idx, p in enumerate(props, 1):
        p.pid = idx
    props.sort(key=lambda p: (-p.confidence, p.rows[0]))
    for idx, p in enumerate(props, 1):
        p.pid = idx
    return props


def apply_proposals(an: Analyzer, accepted: Sequence[Proposal]) -> List[textio.Cue]:
    """只改被接受的提案，其余行逐字节沿用输入。"""
    mods: Dict[int, List[Proposal]] = {}
    for p in accepted:
        mods.setdefault(p.rows[0] - 1, []).append(p)

    out: List[textio.Cue] = []
    i = 0
    while i < len(an.cues):
        cands = mods.get(i)
        if cands:
            p = sorted(cands, key=lambda x: -x.confidence)[0]
            for text, (st, en) in zip(p.after, p.times):
                out.append(textio.Cue(1, st, en, text))
            i = p.rows[-1]
            continue
        c = an.cues[i]
        out.append(textio.Cue(1, c.start_ms, c.end_ms, c.text))
        i += 1
    for idx, c in enumerate(out, 1):
        c.index = idx
    return out


def format_list(props: Sequence[Proposal], name: str) -> str:
    L = [f"PauseFix 提案清单 — {name}", "=" * 76, "",
         f"共 {len(props)} 条提案："
         f"合并 {sum(1 for p in props if p.kind=='merge')} · "
         f"重排 {sum(1 for p in props if p.kind=='resegment')} · "
         f"拆分 {sum(1 for p in props if p.kind=='split')}", ""]
    label = {"merge": "合并", "resegment": "局部重排", "split": "拆分"}
    for p in props:
        star = "★" if p.confidence >= 0.9 else ("☆" if p.confidence >= 0.75 else "·")
        L.append(f"[{p.pid:3d}] {star} {p.confidence:.2f}  {label[p.kind]}  第 {p.rows} 行")
        for t in p.before:
            L.append(f"       现 │ {t}")
        for t in p.after:
            L.append(f"       改 │ {t}")
        L.append(f"       · {p.reason}")
        L.append("")
    return "\n".join(L)


def run(audio: str, srt_path: str, script_path: Optional[str], out_prefix: str,
        db: float = -30.0, min_silence: float = 0.10,
        max_chars: int = 16, max_dur: float = 6.5) -> Dict:
    an = Analyzer(audio, srt_path, script_path, db=db, min_silence=min_silence)
    props = build_proposals(an, max_chars=max_chars, max_dur=max_dur)
    name = os.path.basename(os.path.dirname(srt_path)) or os.path.basename(srt_path)
    payload = {
        "name": name, "srt_path": srt_path, "audio_path": audio,
        "script_path": script_path, "n_lines": len(an.cues),
        "n_proposals": len(props), "offset_ms": round(an.offset * 1000.0, 1),
        "before_hit": an.before_hit, "n_pauses": len(an.sil),
        "proposals": [p.to_dict() for p in props],
        "lines": [{"start_ms": round(c.start_ms, 1), "end_ms": round(c.end_ms, 1),
                   "text": c.text} for c in an.cues],
        "cue_chars": [list(x) for x in an.cue_chars],
        "pauses": [[round(s * 1000.0, 1), round(e * 1000.0, 1)] for s, e in an.sil],
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_prefix)) or ".", exist_ok=True)
    with open(out_prefix + ".json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    with open(out_prefix + ".txt", "w", encoding="utf-8") as f:
        f.write(format_list(props, name))
    return payload
