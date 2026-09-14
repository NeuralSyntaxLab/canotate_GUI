"""Stage 3 — the experiment that decides whether this approach is viable.

Templates and test phrases come from **disjoint dates**.  Splitting by file
would still leak: phrases inside one recording session share bird state and
background noise, so a file-level split flatters the score.  The file-level
split is reported alongside as a robustness check — if the two disagree
substantially, session effects are carrying the result.

Duration is reported as context only, never as a feature.  At correction time
the segment boundaries are themselves the suspect quantity, so a detector that
leaned on duration would agree with the annotator precisely when the annotator
drew the boundaries wrong.

Run:  python experiment.py [--split date|file] [--n-templates 3]
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import scoring
import templates as templates_mod
from cache import Patches


# ── Splits ───────────────────────────────────────────────────────────

def split_groups(patches, by="date", seed=0):
    """Split patch indices into a template pool and a test pool by group.

    Returns ``(template_indices, test_indices, info)``.  Groups are dates or
    files; roughly a third of the groups become the template pool, which is
    ample for 30 candidates per label while leaving most data for testing.
    """
    keys = patches.date if by == "date" else np.array(
        [f"{d}/{f}" for d, f in zip(patches.date, patches.file_name)])
    groups = np.unique(keys)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(groups)
    n_template = max(1, int(round(len(groups) / 3)))
    template_groups = set(shuffled[:n_template])

    is_template = np.isin(keys, list(template_groups))
    info = {"by": by, "n_groups": len(groups),
            "template_groups": sorted(map(str, template_groups))}
    return (np.where(is_template)[0], np.where(~is_template)[0], info)


# ── Baselines ────────────────────────────────────────────────────────

def duration_baseline(patches, template_idx, test_idx):
    """Gaussian-on-duration classifier, fitted on the template split.

    Context only.  See the module docstring for why duration is excluded from
    the real score.
    """
    stats = {}
    for label in config.LABELS:
        d = np.array([patches.duration(i) for i in template_idx
                      if patches.label[i] == label])
        if len(d) >= 2 and d.std() > 0:
            stats[label] = (d.mean(), d.std())
    predicted = []
    for i in test_idx:
        best, best_label = None, None
        for label, (mu, sd) in stats.items():
            ll = -0.5 * ((patches.duration(i) - mu) / sd) ** 2 - np.log(sd)
            if best is None or ll > best:
                best, best_label = ll, label
        predicted.append(best_label)
    return np.array(predicted)


# ── Reporting ────────────────────────────────────────────────────────

def confusion(assigned, predicted, labels):
    m = np.zeros((len(labels), len(labels)), dtype=int)
    index = {l: k for k, l in enumerate(labels)}
    for a, p in zip(assigned, predicted):
        if a in index and p in index:
            m[index[a], index[p]] += 1
    return m


def print_confusion(matrix, labels, title):
    print(f"\n{title}")
    print("        " + "".join(f"{l:>6}" for l in labels) + "      n  recall")
    for r, label in enumerate(labels):
        n = matrix[r].sum()
        recall = matrix[r, r] / n * 100 if n else float("nan")
        print(f"  {label:>4}  " + "".join(f"{c:>6}" for c in matrix[r])
              + f" {n:>6}  {recall:5.1f}%")
    total = matrix.sum()
    correct = np.trace(matrix)
    print(f"  accuracy = {correct / total * 100:.1f}%  "
          f"({total - correct} errors of {total})")
    return correct / total if total else float("nan")


def print_hard_pairs(matrix, labels, n=6):
    pairs = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            pairs.append((matrix[i, j] + matrix[j, i], labels[i], labels[j]))
    pairs.sort(reverse=True)
    print("\n  hardest pairs (symmetric confusions):")
    for count, a, b in pairs[:n]:
        if count:
            print(f"    {a:>3} <-> {b:<3} {count:>5}")


def print_margin_histogram(margins, bins=25):
    """The shape of this histogram is the finding.

    Bimodal with a clean valley: the valley is the flagging threshold and the
    near-zero band is an honest 'uncertain' class.  Unimodal across zero: no
    discriminative power, and any threshold flags an arbitrary slice.
    """
    m = np.asarray([x for x in margins if np.isfinite(x)])
    if m.size == 0:
        print("\n  margin histogram: no finite margins")
        return
    counts, edges = np.histogram(m, bins=bins)
    peak = counts.max() or 1
    print(f"\n  margin histogram  (n={m.size}, "
          f"negative = method disagrees with annotator)")
    for c, lo, hi in zip(counts, edges[:-1], edges[1:]):
        bar = "#" * int(round(40 * c / peak))
        mark = " <-0" if lo <= 0 < hi else ""
        print(f"    {lo:+7.3f} {bar:<40} {c:>5}{mark}")
    print(f"    negative margins: {(m < 0).sum()} ({(m < 0).mean() * 100:.1f}%)"
          f"   |margin| < 0.02: {(np.abs(m) < 0.02).sum()}")


# ── Main ─────────────────────────────────────────────────────────────

def run(split_by=None, n_templates=None, global_band=None,
        shuffle_templates=False, tie_break_psr=False, verbose=True):
    split_by = split_by or config.SPLIT_BY
    if global_band is not None:
        # Control: force every label into the SAME band.  The per-label band is
        # derived from the label's own data and is applied to the query, so it
        # smuggles label identity into the comparison and puts each label's
        # peak on its own scale.  If accuracy survives a shared band, the
        # correlation itself is doing the work.
        config.GLOBAL_BAND = global_band
        print(f"[control] forcing a single band {global_band} Hz for all labels")
    patches = Patches()
    template_idx, test_idx, info = split_groups(
        patches, by=split_by, seed=config.RANDOM_SEED)

    print(f"=== split by {info['by']}: {len(info['template_groups'])} of "
          f"{info['n_groups']} groups to templates ===")
    print(f"    template pool {len(template_idx)} patches, "
          f"test pool {len(test_idx)} patches")

    pools = defaultdict(list)
    for i in template_idx:
        pools[int(patches.label[i])].append(int(i))
    specs = templates_mod.build(pools, patches=patches, n_keep=n_templates,
                                verbose=verbose)
    if not specs:
        raise RuntimeError("No templates could be built.")

    if shuffle_templates:
        # Negative control.  Re-attach each label's templates to a different
        # label.  A pipeline that still scores well here is not reading the
        # template at all, and every positive result from it is worthless.
        keys = sorted(specs)
        rng = np.random.default_rng(config.RANDOM_SEED + 1)
        perm = list(rng.permutation(keys))
        while any(a == b for a, b in zip(keys, perm)):
            perm = list(rng.permutation(keys))
        specs = {k: specs[int(v)] for k, v in zip(keys, perm)}
        print(f"[control] templates shuffled: "
              + ", ".join(f"{k}<-{int(v)}" for k, v in zip(keys, perm)))

    labels = [l for l in config.LABELS if l in specs]

    assigned, predicted, margins = [], [], []
    outcomes = Counter()
    rank_wins = Counter()
    records = []
    for n, i in enumerate(test_idx, 1):
        label = int(patches.label[i])
        if label not in specs:
            outcomes["no_template"] += 1
            continue
        scores, outcome = scoring.score_phrase(
            patches[i], patches.F, specs,
            min_support_ratio=config.MIN_SUPPORT_RATIO)
        outcomes[outcome] += 1
        if outcome == scoring.TOO_SHORT:
            continue
        pred, margin = scoring.decide(scores, label,
                                      tie_break_psr=tie_break_psr)
        assigned.append(label)
        predicted.append(pred)
        margins.append(margin)
        rank_wins[scores[pred]["template_rank"]] += 1
        records.append({
            "index": int(i), "date": str(patches.date[i]),
            "file_name": str(patches.file_name[i]),
            "t_start": float(patches.t_start[i]),
            "t_end": float(patches.t_end[i]),
            "assigned": label, "predicted": int(pred),
            "margin": float(margin),
            "scores": {str(l): {"peak": float(s["peak"]),
                                "psr": float(s["psr"]),
                                "lag": int(s["lag"])}
                       for l, s in scores.items()},
        })
        if verbose and n % 200 == 0:
            print(f"    scored {n}/{len(test_idx)}")

    assigned = np.array(assigned)
    predicted = np.array(predicted)

    xcorr_matrix = confusion(assigned, predicted, labels)
    accuracy = print_confusion(xcorr_matrix, labels,
                               "CROSS-CORRELATION (peak ZNCC, argmax over labels)")
    print_hard_pairs(xcorr_matrix, labels)
    print_margin_histogram(margins)

    dur_pred = duration_baseline(patches, template_idx,
                                 [r["index"] for r in records])
    dur_matrix = confusion(assigned, dur_pred, labels)
    dur_accuracy = print_confusion(
        dur_matrix, labels,
        "DURATION ONLY (context, not a target -- excluded from the score)")

    print(f"\noutcomes: {dict(outcomes)}")
    if len(rank_wins) > 1:
        print(f"winning template rank: {dict(rank_wins)}  "
              "(rank>0 wins indicate bimodal renditions)")
    print(f"\nsummary: cross-correlation {accuracy * 100:.1f}%  vs  "
          f"duration {dur_accuracy * 100:.1f}%")

    # Controls write to their own file: a control run must never overwrite the
    # real results that viz.py and any later analysis read.
    tag = split_by
    if shuffle_templates:
        tag += "_shuffled"
    if global_band is not None:
        tag += "_globalband"
    if n_templates:
        tag += f"_t{n_templates}"
    if tie_break_psr:
        tag += "_psrtie"
    out = os.path.join(config.CACHE_DIR, f"results_{tag}.json")
    with open(out, "w") as fh:
        json.dump({"split": info, "labels": labels,
                   "accuracy": accuracy, "duration_accuracy": dur_accuracy,
                   "confusion": xcorr_matrix.tolist(),
                   "records": records}, fh)
    print(f"per-case results -> {out}")
    return records, specs


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=["date", "file"], default=None)
    ap.add_argument("--n-templates", type=int, default=None)
    ap.add_argument("--tie-break-psr", action="store_true",
                    help="break near-tied peaks by PSR (see README)")
    ap.add_argument("--shuffle-templates", action="store_true",
                    help="negative control: give each label another's templates")
    ap.add_argument("--global-band", type=float, nargs=2, default=None,
                    metavar=("FMIN", "FMAX"),
                    help="control: score every label in this one band")
    args = ap.parse_args()
    run(split_by=args.split, n_templates=args.n_templates,
        global_band=tuple(args.global_band) if args.global_band else None,
        shuffle_templates=args.shuffle_templates,
        tie_break_psr=args.tie_break_psr)
