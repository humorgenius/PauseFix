"""PauseFix 命令行入口。

用法：
  python -m pausefix.cli --audio A.wav --srt 修复后的字幕.srt \
      --script 文稿文件.txt -o 断句修复.srt
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Tuple

from . import (
    calibrate,
    chartime,
    envelope,
    pauses,
    report,
    scriptmap,
    segmenter,
    snap,
    textio,
    words,
)
from .segmenter import Config


def _step(msg: str) -> None:
    print(f"[PauseFix] {msg}", flush=True)


def process(
    audio: str,
    srt_path: str,
    out_path: str,
    script_path: Optional[str] = None,
    cfg: Optional[Config] = None,
    db: float = -30.0,
    min_silence: float = 0.10,
    report_path: Optional[str] = None,
) -> Tuple[List[textio.Cue], str]:
    cfg = cfg or Config()
    _step(f"读取字幕：{os.path.basename(srt_path)}")
    input_cues = textio.parse_srt(textio.read_text(srt_path))
    if not input_cues:
        raise SystemExit("字幕文件解析为空，请检查格式。")
    _step(f"  共 {len(input_cues)} 行")

    _step(f"检测停顿：{os.path.basename(audio)}")
    silences, duration = pauses.detect(audio, db=db, min_silence=min_silence)
    _step(f"  停顿 {len(silences)} 个，音频 {duration:.1f}s")

    _step("标定字幕 ↔ 音频 偏移")
    in_boundaries = [c.end_ms / 1000.0 for c in input_cues[:-1]]
    offset, hit = calibrate.fit_offset(in_boundaries, silences)
    slope = calibrate.check_drift(in_boundaries, silences, offset)
    _step(
        f"  偏移 {offset:+.2f}s（原断句命中停顿 {hit*100:.0f}%"
        + (f"，斜率 {slope:.5f}" if slope is not None else "")
        + "）"
    )

    _step("构建逐字时间轴")
    chars = chartime.build(input_cues, silences, offset)
    _step(f"  字符 {len(chars)} 个")

    break_pos = {}
    if script_path:
        _step(f"读取原稿：{os.path.basename(script_path)}")
        script_text = textio.read_text(script_path)
        srt_text = "".join(c.text for c in input_cues)
        break_pos = scriptmap.map_punct(srt_text, script_text)
        _step(f"  映射到 {len(break_pos)} 个标点断点")

    sil_ms = [(s * 1000.0, e * 1000.0) for s, e in silences]
    _step("计算每个切点的停顿分")
    ps = segmenter.pause_cut_scores(chars, sil_ms, cfg)

    _step("计算每个切点的波形谷底分")
    rms, hop = envelope.load_envelope(audio)
    dip = envelope.DipIndex(rms, hop)
    ds = envelope.dip_cut_scores(chars, dip)

    word_cuts = None
    if words.available():
        word_cuts = words.cut_positions("".join(c["ch"] for c in chars))
        _step(f"  词边界 {len(word_cuts or ())} 个")
    else:
        _step("  jieba 不可用，跳过词边界约束")

    bscore = segmenter.boundary_scores(chars, ps, break_pos, word_cuts, cfg, ds)

    _step("动态规划重切断句")
    lines = segmenter.segment(chars, bscore, cfg)
    _step(f"  得到 {len(lines)} 行")

    output_cues = snap.snap(
        lines, chars, sil_ms, offset * 1000.0, duration * 1000.0, cfg
    )

    # 断点没有吃到停顿的行 = 需要人工看一眼
    suspicious = [i for i, (a, _) in enumerate(lines) if i > 0 and ps[a] <= 0.0]

    _step(f"写出 {out_path}")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(textio.format_srt(output_cues))

    rep = report.build_report(
        title=os.path.basename(srt_path),
        input_cues=input_cues,
        output_cues=output_cues,
        chars=chars,
        lines=lines,
        silences=silences,
        offset=offset,
        cfg=cfg,
        has_script=bool(script_path),
        punct_breaks=len(break_pos),
        audio_duration=duration,
        suspicious=suspicious,
    )
    if report_path is None:
        base, _ = os.path.splitext(out_path)
        report_path = base + "_对照报告.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(rep)
    _step(f"写出报告 {report_path}")
    return output_cues, rep


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="pausefix", description="用音频真实停顿修正字幕断句"
    )
    ap.add_argument("--audio", "-a", required=True, help="口播音频文件")
    ap.add_argument("--srt", "-s", required=True, help="已校对字幕（srt 或剪映 txt）")
    ap.add_argument("--script", "-t", default=None, help="原稿（带标点，可选但建议）")
    ap.add_argument("--output", "-o", required=True, help="输出 srt 路径")
    ap.add_argument("--report", "-r", default=None, help="报告输出路径")
    ap.add_argument("--max-chars", type=int, default=Config.max_chars)
    ap.add_argument("--min-chars", type=int, default=Config.min_chars)
    ap.add_argument("--min-dur", type=float, default=Config.min_dur)
    ap.add_argument("--max-dur", type=float, default=Config.max_dur)
    ap.add_argument("--db", type=float, default=-30.0, help="静音阈值 dB（越接近 0 越宽松）")
    ap.add_argument("--min-silence", type=float, default=0.10, help="最短静音秒")
    args = ap.parse_args(argv)

    cfg = Config(
        max_chars=args.max_chars,
        min_chars=args.min_chars,
        min_dur=args.min_dur,
        max_dur=args.max_dur,
    )
    cues, _ = process(
        audio=args.audio,
        srt_path=args.srt,
        out_path=args.output,
        script_path=args.script,
        cfg=cfg,
        db=args.db,
        min_silence=args.min_silence,
        report_path=args.report,
    )
    print(f"[PauseFix] 完成，共 {len(cues)} 行。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
