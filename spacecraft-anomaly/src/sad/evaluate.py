"""Event-level evaluation, as in Hundman et al. 2018.

* A labeled anomaly sequence is a true positive if any predicted sequence overlaps it,
  otherwise a false negative.
* A predicted sequence that overlaps no labeled sequence is a false positive.

This avoids the "point-adjust" protocol, which is known to inflate F1 scores.
"""

from __future__ import annotations

from collections import defaultdict


def _overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def evaluate_channel(true_seqs: list, pred_seqs: list, classes: list | None = None) -> dict:
    true_seqs = [tuple(map(int, s)) for s in true_seqs]
    pred_seqs = [tuple(map(int, s)) for s in pred_seqs]
    hit = [any(_overlap(t, p) for p in pred_seqs) for t in true_seqs]
    fp = sum(not any(_overlap(p, t) for t in true_seqs) for p in pred_seqs)
    res = {"tp": sum(hit), "fn": len(hit) - sum(hit), "fp": fp}
    if classes is not None:
        by = defaultdict(lambda: [0, 0])
        for h, c in zip(hit, classes):
            by[c][0] += int(h)
            by[c][1] += 1
        res["by_class"] = {c: {"tp": v[0], "total": v[1]} for c, v in by.items()}
    return res


def prf(tp: int, fp: int, fn: int, beta: float = 1.0) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    b2 = beta ** 2
    denom = b2 * precision + recall
    f = (1 + b2) * precision * recall / denom if denom else 0.0
    return {"precision": precision, "recall": recall, f"f{beta:g}": f}


def aggregate(per_channel: list[dict]) -> dict:
    tp = sum(r["tp"] for r in per_channel)
    fp = sum(r["fp"] for r in per_channel)
    fn = sum(r["fn"] for r in per_channel)
    out = {"tp": tp, "fp": fp, "fn": fn, **prf(tp, fp, fn, 1.0)}
    out["f0.5"] = prf(tp, fp, fn, 0.5)["f0.5"]
    by = defaultdict(lambda: [0, 0])
    for r in per_channel:
        for c, v in r.get("by_class", {}).items():
            by[c][0] += v["tp"]
            by[c][1] += v["total"]
    if by:
        out["recall_by_class"] = {c: (v[0] / v[1] if v[1] else 0.0) for c, v in by.items()}
    return out
