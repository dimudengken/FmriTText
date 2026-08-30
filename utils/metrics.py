"""文本生成评估指标：BLEU-1/2/3/4、ROUGE-L、METEOR、CIDEr、SPICE。

策略：pycocoevalcap 每项独立尝试（Java 依赖的 Meteor/SPICE 缺 Java 时单独失败，不拖累
Bleu/Rouge/Cider）；pycoco 缺失的项用纯 python 补齐（BLEU=nltk、ROUGE-L=LCS、CIDEr=标准
TF-IDF）；METEOR 需 Java(pycoco) 或 wordnet(nltk)，都没有则剔除。

尺度警告（CLAUDE.md）：CIDEr 有 0~1 归一与 ×10 原始两套尺度，跨表对比前先统一。
"""
import re
import shutil
import traceback
from collections import Counter
from math import log

_CIDER_NGRAM = 4
_HAS_JAVA = shutil.which("java") is not None  # Meteor/SPICE 依赖 Java


def _tok(text):
    return re.findall(r"\w+", text.lower())


def _ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def rouge_l(hyp, ref):
    """ROUGE-L：hyp/ref 的 LCS 长度 F1。"""
    h, r = _tok(hyp), _tok(ref)
    if not h or not r:
        return 0.0
    m, n = len(h), len(r)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if h[i - 1] == r[j - 1] \
                else max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    if lcs == 0:
        return 0.0
    prec, rec = lcs / m, lcs / n
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def _cider_score(hyps, refs):
    """标准 CIDEr（TF-IDF n-gram cosine，n=1..4 取均值），纯 python，0~1 尺度。

    df 在参考语料上统计；n-gram 向量用 TF 加权、IDF 用 log(N/df)。
    """
    hyp_toks = [_tok(h) for h in hyps]
    ref_toks = [[_tok(r) for r in rlist] for rlist in refs]
    N = len(hyps)

    df = [Counter() for _ in range(_CIDER_NGRAM)]
    for rlist in ref_toks:
        seen = [set() for _ in range(_CIDER_NGRAM)]
        for r in rlist:
            for n in range(_CIDER_NGRAM):
                for g in _ngrams(r, n + 1):
                    if g not in seen[n]:
                        seen[n].add(g)
                        df[n][g] += 1

    def tfidf(grams, n):
        c = Counter(grams)
        total = sum(c.values()) or 1
        d = df[n]
        return {g: (v / total) * log(N / (d.get(g, 0) + 1e-6)) for g, v in c.items()}

    def cos(a, b):
        inter = set(a) & set(b)
        if not inter:
            return 0.0
        num = sum(a[g] * b[g] for g in inter)
        na = sum(v * v for v in a.values()) ** 0.5
        nb = sum(v * v for v in b.values()) ** 0.5
        return num / (na * nb) if na and nb else 0.0

    if N == 0:
        return 0.0
    per = []
    for ht, rlt in zip(hyp_toks, ref_toks):
        c_n = []
        for n in range(_CIDER_NGRAM):
            h = tfidf(_ngrams(ht, n + 1), n)
            c_n.append(max(cos(h, tfidf(_ngrams(r, n + 1), n)) for r in rlt))
        per.append(sum(c_n) / _CIDER_NGRAM)
    return sum(per) / N


def _nltk_bleu(hyps, refs):
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    smooth = SmoothingFunction().method1
    acc = {f"BLEU-{i}": 0.0 for i in range(1, 5)}
    for h, rlist in zip(hyps, refs):
        hw, rws = _tok(h), [_tok(r) for r in rlist]
        for i in range(1, 5):
            w = [1.0 / i] * i + [0.0] * (4 - i)
            acc[f"BLEU-{i}"] += sentence_bleu(rws, hw, weights=w, smoothing_function=smooth)
    n = max(len(hyps), 1)
    return {k: v / n for k, v in acc.items()}


def _nltk_meteor(hyps, refs):
    """需要 nltk wordnet 语料；缺失时抛异常由调用方剔除。"""
    from nltk.translate.meteor_score import meteor_score
    total = sum(meteor_score([_tok(r) for r in rlist], _tok(h))
                for h, rlist in zip(hyps, refs))
    return total / max(len(hyps), 1)


def _extract(name, score):
    """把各 pycoco scorer 的返回归一成 dict——不同版本返回 dict 或 list（如 Bleu 返回
    [b1,b2,b3,b4] 或 {Bleu_1:...}），统一成 {指标名: 值}。"""
    if name == "Bleu":
        if isinstance(score, dict):
            return {f"BLEU-{int(k[-1])}": v for k, v in score.items()}
        return {f"BLEU-{i}": v for i, v in enumerate(score, start=1)}
    key = {"Rouge": "ROUGE-L", "Meteor": "METEOR", "Cider": "CIDEr"}[name]
    if isinstance(score, dict) and key in score:
        return {key: score[key]}
    if isinstance(score, (list, tuple)) and score:
        return {key: float(score[0])}
    return {key: float(score)}


def _pycoco_scores(hyps, refs, use_spice):
    """pycocoevalcap 逐 scorer 独立尝试；构造器也进 try（缺 Java 时 Meteor 单独失败）。"""
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge

    gts = {i: refs[i] for i in range(len(hyps))}
    res = {i: [hyps[i]] for i in range(len(hyps))}  # pycoco 的 res 值是 list[str]
    out = {}
    scorers = [("Bleu", lambda: Bleu(4)), ("Rouge", Rouge), ("Cider", Cider)]
    if _HAS_JAVA:
        scorers.insert(2, ("Meteor", Meteor))
    for name, ctor in scorers:
        try:
            scorer = ctor()
            score, _ = scorer.compute_score(gts, res)
            out.update(_extract(name, score))
        except Exception:
            print(f"[metrics] pycoco {name} 失败:\n{traceback.format_exc()}", flush=True)
            continue

    if use_spice and _HAS_JAVA:
        try:
            from pycocoevalcap.spice.spice import Spice
            score, _ = Spice().compute_score(gts, res)
            out["SPICE"] = score["SPICE"]
        except Exception as e:
            print(f"[metrics] SPICE 失败: {e}（需 Java，--no_spice 关闭）", flush=True)
    return out


def compute_text_metrics(hypotheses, references, use_spice=True):
    """hypotheses: list[str]；references: list[list[str]]。返回 dict[str, float]。"""
    scores = {}
    try:
        scores = _pycoco_scores(hypotheses, references, use_spice)
    except Exception as e:
        print(f"[metrics] pycocoevalcap 不可用，全部走纯 python 补齐: {e}", flush=True)

    # 补齐 pycoco 缺失的核心指标
    missing_bleu = {f"BLEU-{i}" for i in range(1, 5)} - set(scores)
    if missing_bleu:
        b = _nltk_bleu(hypotheses, references)
        scores.update({k: b[k] for k in missing_bleu})
    if "ROUGE-L" not in scores:
        scores["ROUGE-L"] = sum(max(rouge_l(h, r) for r in rl)
                                for h, rl in zip(hypotheses, references)) / max(len(hypotheses), 1)
    if "CIDEr" not in scores:
        scores["CIDEr"] = _cider_score(hypotheses, references)
    if "METEOR" not in scores:
        try:
            scores["METEOR"] = _nltk_meteor(hypotheses, references)
        except Exception as e:
            print(f"[metrics] METEOR 不可用（需 wordnet/Java）: {e}", flush=True)

    return {k: v for k, v in scores.items() if not (isinstance(v, float) and v != v)}
