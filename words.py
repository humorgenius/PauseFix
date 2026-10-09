"""词边界：用 jieba 分词的结果标记出所有「词边界」字符位置。

jieba 不可用时返回 None，此时评分函数会退化为「不做词边界约束」。
"""

from __future__ import annotations

from typing import Optional, Set

_jieba = None
_tried = False


def _get_jieba():
    global _jieba, _tried
    if not _tried:
        _tried = True
        try:
            import jieba  # type: ignore

            jieba.setLogLevel(60)  # 关掉它自己的日志
            _jieba = jieba
        except Exception:
            _jieba = None
    return _jieba


def available() -> bool:
    return _get_jieba() is not None


def cut_positions(text: str) -> Optional[Set[int]]:
    """返回所有词边界位置（= 该位置前面有多少个字符）；末尾不算切点。"""
    jb = _get_jieba()
    if jb is None:
        return None
    pos: Set[int] = set()
    i = 0
    for tok in jb.lcut(text):
        i += len(tok)
        pos.add(i)
    pos.discard(len(text))
    pos.discard(0)
    return pos


def cut_inside_word(text: str, k: int, half: int = 10) -> Optional[str]:
    """刀口 k 是否落在一个词里面（返回该词）。

    用「局部窗口 + 词典模式(HMM=False)」分词：整篇是没标点的长串，HMM 会造出
    「事全」这类假词；限定在窗口里、关掉 HMM，分词结果才可靠。
    """
    jb = _get_jieba()
    if jb is None:
        return None
    lo = max(0, k - half)
    hi = min(len(text), k + half)
    seg = text[lo:hi]
    pos = lo
    for tok in jb.cut(seg, HMM=False):
        if pos < k < pos + len(tok):
            return tok
        pos += len(tok)
    return None


def word_spanned(text: str, k: int, max_span: int = 4) -> Optional[str]:
    """（保留）基于词典跨度的粗判据，精度低于 cut_inside_word。"""
    jb = _get_jieba()
    if jb is None:
        return None
    try:
        jb.initialize()
    except Exception:
        pass
    freq = jb.dt.FREQ
    best, best_score = None, 0.0
    for a in range(1, max_span):
        for b in range(1, max_span + 1 - a):
            if a + b < 2:
                continue
            if k - a < 0 or k + b > len(text):
                continue
            w = text[k - a:k + b]
            f = freq.get(w, 0.0)
            if f and len(w) >= 2:
                score = f * len(w)
                if score > best_score:
                    best_score, best = score, w
    return best
