"""GRPO 奖励：R = 3*CIDEr + 0.3*BERTScore - 0.4*IncompletenessPenalty + w_spec*Specificity。

Specificity = hyp 平均 unigram IDF（CiderGlobal.specificity，治熵坍缩/模板化：全功能词
的低分、图像特异词高分；仅当 --w_spec>0 参与）。

CIDEr 默认走 CiderGlobal（固定全局 IDF，0~1 尺度，与 utils.metrics 同公式）：df 在全量
COCO 参考语料上统计并固化，避免 batch 内 8~16 条参考的微型 IDF 强化通用模板（GRPO 坍缩
根因）。不传 cider_scorer 时降级 pycocoevalcap per-image 分数，再降级 2-gram F1 代理。
BERTScore 默认 0（--use_bertscore 才加载 roberta-large）。
"""
import re
from collections import Counter
from math import log

import torch

# 不完整功能词：BIT-LLM 里这些结尾词通常意味着生成被截断
INCOMPLETE_WORDS = {"and", "with", "of", "the", "a", "an", "to", "for",
                    "is", "in", "at", "on", "are", "was", "be"}

_bertscorer = None


def incompleteness_penalty(text):
    """生成以功能词结尾 → 1.0（惩罚），否则 0.0。"""
    words = re.findall(r"\w+", text.strip().lower())
    if not words:
        return 1.0
    return 1.0 if words[-1] in INCOMPLETE_WORDS else 0.0


def _tok(text):
    return re.findall(r"\w+", text.lower())


def _ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def ngram_f1(hyp, ref, n=4):
    """n-gram F1 代理。太短时退到 1-gram 集合交并比。"""
    h = re.findall(r"\w+", hyp.lower())
    r = re.findall(r"\w+", ref.lower())
    if not h or not r:
        return 0.0
    if len(h) < n or len(r) < n:
        inter = len(set(h) & set(r))
        return inter / max(len(set(h)), len(set(r)), 1)
    hg, rg = set(_ngrams(h, n)), set(_ngrams(r, n))
    if not hg or not rg:
        return 0.0
    inter = len(hg & rg)
    prec = inter / len(hg)
    rec = inter / len(rg)
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def _bertscore(hyp, ref):
    global _bertscorer
    if _bertscorer is None:
        try:
            from bert_score import BERTScorer
            _bertscorer = BERTScorer(lang="en", rescale_with_baseline=True)
        except ImportError:
            _bertscorer = None
    if _bertscorer is None:
        return 0.0
    p, r, f = _bertscorer.score([hyp], [ref])
    return float(f.item())


class CiderGlobal:
    """固定全局 IDF 的 CIDEr（0~1，与 utils.metrics._cider_score 同公式，df 固化）。

    GRPO 场景：batch 只有 B×G=8~16 条参考，常用词（man/standing/front）在微型语料里
    df=1 → IDF 虚高 → 被强化成通用模板（这就是 GRPO 后坍缩到 "A man is standing in
    front of a table..." 的根因）。改用全量 COCO 参考语料固定 df（log(N/df)），
    n=1..4 平均 TF-IDF 余弦，每图取 max over 多条参考。reward 只需组内相对序，
    绝对尺度无所谓。
    """

    def __init__(self, all_refs):
        """all_refs: list[list[str]]（每图的参考 caption 列表，即 build_caption_map 的值）。"""
        self.N = 0
        df = [Counter() for _ in range(4)]
        for rlist in all_refs:
            seen = [set() for _ in range(4)]
            for r in rlist:
                self.N += 1
                toks = _tok(r)
                for n in range(4):
                    for g in _ngrams(toks, n + 1):
                        if g not in seen[n]:
                            seen[n].add(g)
                            df[n][g] += 1
        # 固化 log(N/df)；只留语料中出现的 n-gram。OOV 的 hyp n-gram 按 df=1 取最大 IDF
        #（标准 IDF 语义：未见即最稀有），否则 _cos 里 norm 会 KeyError。
        self._log_df = [{g: log(self.N / (v + 1e-6)) for g, v in dfc.items()} for dfc in df]
        self._idf_default = log(self.N)

    def _cos(self, hc, rc, n):
        inter = set(hc) & set(rc)
        if not inter:
            return 0.0
        ld = self._log_df[n]
        dflt = self._idf_default
        th, tr = sum(hc.values()), sum(rc.values())
        num = sum(hc[g] / th * rc[g] / tr * ld[g] for g in inter)
        nh = sum((hc[g] / th) ** 2 * ld.get(g, dflt) for g in hc)
        nr = sum((rc[g] / tr) ** 2 * ld.get(g, dflt) for g in rc)
        return num / (nh * nr) ** 0.5 if nh and nr else 0.0

    def score(self, hyp, refs):
        """hyp vs 每条 ref 的固定 IDF CIDEr，n=1..4 平均后取 max（0~1）。"""
        ht = _tok(hyp)
        if not ht:
            return 0.0
        hc = [Counter(_ngrams(ht, n + 1)) for n in range(4)]
        best = 0.0
        for r in refs:
            rt = _tok(r)
            if not rt:
                continue
            c = sum(self._cos(hc[n], Counter(_ngrams(rt, n + 1)), n) for n in range(4)) / 4.0
            if c > best:
                best = c
        return best

    def specificity(self, hyp):
        """hyp 平均 unigram IDF（0~1）：图像特异词（corpus 稀见）高分、功能词簇低分。

        与 CIDEr 的 OOV 语义不同（那边取 max，奖励稀见词；这里 OOV 记 0，防 reward
        hacking——policy 乱造 corpus 外稀见词拿不到分）。归一化除 dflt=log(N) → 0~1。
        """
        toks = _tok(hyp)
        if not toks:
            return 0.0
        ld = self._log_df[0]  # unigram IDF，键是 1-tuple（如 ('a',)）
        return sum(ld.get((g,), 0.0) for g in toks) / len(toks) / self._idf_default


def compute_rewards(hypotheses, references, w_cider=3.0, w_bert=0.3, w_inc=0.4, w_spec=0.3,
                    use_bertscore=False, cider_scorer=None):
    """hypotheses: list[str]；references: list[list[str]]（每 hyp 的多条参考 caption）。
    返回 (N,) float32。

    cider_scorer：CiderGlobal 实例（固定全局 IDF）。有则优先用它（GRPO 用，避免
    batch 内微型 IDF 强化通用模板）；无则 pycoco per-image CIDEr，再降级 2-gram F1。
    use_bertscore=False 时 BERTScore 恒 0，不加载 roberta-large（省 1.3GB 下载）。
    w_spec>0 时加特异性项（cider_scorer.specificity，治熵坍缩）；无 scorer 时该项为 0。
    """
    n = len(hypotheses)
    if cider_scorer is not None:
        cider_vals = [cider_scorer.score(h, rl) for h, rl in zip(hypotheses, references)]
    else:
        try:
            from pycocoevalcap.cider.cider import Cider
            gts = {i: refs for i, refs in enumerate(references)}
            res = {i: [h] for i, h in enumerate(hypotheses)}
            scores, _ = Cider().compute_score(gts, res)
            if isinstance(scores, dict):
                scores = list(scores.values())
            cider_vals = [float(s) for s in scores]
        except Exception:
            # 降级代理：2-gram F1（对部分匹配敏感，避免 4-gram 对模板句恒 0 导致 reward 全 0）
            cider_vals = [max(ngram_f1(h, r, n=2) for r in rl)
                          for h, rl in zip(hypotheses, references)]
    out = []
    for i, (h, rl) in enumerate(zip(hypotheses, references)):
        bert = _bertscore(h, rl[0]) if use_bertscore else 0.0
        inc = incompleteness_penalty(h)
        spec = cider_scorer.specificity(h) if cider_scorer is not None else 0.0
        out.append(w_cider * cider_vals[i] + w_bert * bert - w_inc * inc + w_spec * spec)
    return torch.tensor(out, dtype=torch.float32)
