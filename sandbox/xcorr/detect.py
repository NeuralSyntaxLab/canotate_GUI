#!/usr/bin/env python
"""Production path: score every phrase in a corpus, streaming, with no cache.

The sandbox caches spectrogram patches because it re-scores the same phrases
dozens of times while parameters change.  A production scan touches each phrase
**once**, so caching it is pure cost: 3000 songs would be roughly 13 GB of
patches to write and read back for no reuse at all.  Here each file is read,
turned into a spectrogram, scored, and dropped before the next one is opened.
Peak memory is one file (~30 MB per worker) regardless of corpus size.

The only thing that persists is ``templates.npz`` — about 1 MB.

    python detect.py fit  --out templates.npz          # once, from verified data
    python detect.py scan --templates templates.npz \\
                          --annotations corpus.json --out flagged.json

Timing (measured, 8-core M-series, ~29 s of audio per file):

    read + spectrogram   146 ms/file      7.3 min / 3000 files, 1 core
    scoring               6 ms/phrase     2.4 min / 24000 phrases, 1 core
    ------------------------------------------------------------------
    total                                ~9.7 min on 1 core
                                         ~2.5 min on 4 workers
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

import config
import scoring
import spectro
from cannotate.cannotate_utils import read_wav

# One band for every label.  The sandbox's per-label bands measure 95.6% and a
# single shared band 95.7% — no loss — and sharing one band is what lets a
# phrase's Fourier transform be computed once and reused across all labels,
# which is an 11x speedup.  See README.md.
DEFAULT_BAND = (1000.0, 13000.0)

_WORKER = {}


# ── Templates ────────────────────────────────────────────────────────

def fit(annotation_file, audio_root, out_path, band=DEFAULT_BAND,
        n_candidates=None, n_keep=1, verbose=True):
    """Build and save the template bank.  Run once, on verified annotations."""
    import cache as cache_mod
    import templates as templates_mod

    n_candidates = n_candidates or config.N_CANDIDATES
    config.ANNOTATION_FILE = annotation_file
    config.AUDIO_ROOT = audio_root
    config.GLOBAL_BAND = band

    # Only the candidate phrases are ever read — at most n_candidates per
    # label, so fitting touches a small fraction of the corpus.
    tmp = os.path.join(os.path.dirname(os.path.abspath(out_path)),
                       "_fit_patches.npz")
    cache_mod.build(max_per_label=n_candidates, out_path=tmp, verbose=verbose)
    patches = cache_mod.Patches(tmp)

    pools = defaultdict(list)
    for i in range(len(patches)):
        pools[int(patches.label[i])].append(i)
    specs = templates_mod.build(pools, patches=patches, n_keep=n_keep,
                                verbose=verbose)

    labels = sorted(specs)
    arrays, index = [], []
    for label in labels:
        for template in specs[label]["templates"]:
            index.append((label, template.shape[1]))
            arrays.append(template.astype(np.float32))
    np.savez(out_path,
             data=np.concatenate(arrays, axis=1),
             cols=np.array([c for _, c in index], dtype=np.int64),
             label=np.array([l for l, _ in index], dtype=np.int64),
             band=np.array(band, dtype=np.float64),
             hop_seconds=np.array(config.HOP_SECONDS),
             nfft=np.array(config.NFFT, dtype=np.int64))
    os.remove(tmp)
    if verbose:
        print(f"[fit] {len(arrays)} template(s) for {len(labels)} label(s) "
              f"-> {out_path} ({os.path.getsize(out_path) / 1e6:.2f} MB)")
    return out_path


def load_bank(path):
    """Return ``(TemplateBank, band, hop_seconds)``."""
    with np.load(path) as z:
        data, cols, label = z["data"], z["cols"], z["label"]
        band = tuple(float(x) for x in z["band"])
        hop = float(z["hop_seconds"])
    by_label = defaultdict(list)
    start = 0
    for l, c in zip(label, cols):
        by_label[int(l)].append(data[:, start:start + c].astype(np.float64))
        start += c
    return scoring.TemplateBank(dict(by_label)), band, hop


# ── Scanning ─────────────────────────────────────────────────────────

def _init_worker(bank_path):
    bank, band, hop = load_bank(bank_path)
    _WORKER.update(bank=bank, band=band, hop=hop)


def _scan_file(job):
    """Score every phrase in one file, then discard its spectrogram."""
    path, date, file_name, segments = job
    bank, band = _WORKER["bank"], _WORKER["band"]
    try:
        fs, y = read_wav(path)
        S, F, T = spectro.compute(np.asarray(y, dtype=np.float64), fs)
    except Exception as exc:                                  # noqa: BLE001
        return [], [(file_name, repr(exc))]
    del y
    S, _ = scoring.band_limit(S, F, band[0], band[1])

    known = set(bank.labels)
    out = []
    for t0, t1, label in segments:
        if label not in known:
            # No template for this label, so there is nothing to agree or
            # disagree with.  Counting it as a disagreement would inflate the
            # flag rate with labels the bank was never fitted on.
            out.append({"date": date, "file_name": file_name,
                        "t_start": t0, "t_end": t1, "assigned": label,
                        "outcome": "no_template"})
            continue
        cols = np.where((T >= t0) & (T <= t1))[0]
        if cols.size < 2:
            out.append({"date": date, "file_name": file_name,
                        "t_start": t0, "t_end": t1, "assigned": label,
                        "outcome": "too_short"})
            continue
        surfaces = bank.surfaces(S[:, cols])
        scores = {}
        for candidate, stack in surfaces.items():
            best = None
            for k, surface in enumerate(stack):
                if surface is None:
                    continue
                m = bank.templates[candidate][k].shape[1]
                peak, lag, psr = scoring.peak_psr(surface, m)
                if best is None or peak > best[0]:
                    best = (peak, psr, lag)
            if best is not None:
                scores[candidate] = {"peak": best[0], "psr": best[1],
                                     "lag": best[2]}
        if not scores:
            out.append({"date": date, "file_name": file_name,
                        "t_start": t0, "t_end": t1, "assigned": label,
                        "outcome": "too_short"})
            continue
        predicted, margin = scoring.decide(scores, label)
        out.append({
            "date": date, "file_name": file_name,
            "t_start": t0, "t_end": t1,
            "assigned": label, "predicted": int(predicted),
            "margin": float(margin), "outcome": "scored",
            "peak": float(scores[predicted]["peak"]),
            "psr": float(scores[predicted]["psr"]),
        })
    del S
    return out, []


def scan(annotation_file, audio_root, bank_path, out_path, workers=None,
         limit=None, verbose=True):
    workers = workers or max(1, (os.cpu_count() or 2) // 2)
    with open(annotation_file) as fh:
        data = json.load(fh)

    jobs = []
    for date, entries in data.items():
        for entry in entries:
            segments = []
            for seg in entry.get("segments", []):
                if len(seg) < 3 or seg[2] is None:
                    continue
                try:
                    segments.append((float(seg[0]), float(seg[1]),
                                     int(seg[2])))
                except (TypeError, ValueError):
                    continue
            if segments:
                jobs.append((os.path.join(audio_root, date, config.WAV_SUBDIR,
                                          entry["file_name"]),
                             date, entry["file_name"], segments))
    if limit:
        jobs = jobs[:limit]

    n_phrases = sum(len(j[3]) for j in jobs)
    if verbose:
        print(f"[scan] {len(jobs)} files, {n_phrases} phrases, "
              f"{workers} worker(s)")

    started = time.time()
    records, failures = [], []
    with Pool(workers, initializer=_init_worker,
              initargs=(bank_path,)) as pool:
        for n, (got, failed) in enumerate(
                pool.imap_unordered(_scan_file, jobs, chunksize=4), 1):
            records.extend(got)
            failures.extend(failed)
            if verbose and (n % 50 == 0 or n == len(jobs)):
                rate = n / (time.time() - started)
                remaining = (len(jobs) - n) / rate if rate else 0
                print(f"[scan]   {n}/{len(jobs)} files "
                      f"({rate:.1f}/s, {remaining / 60:.1f} min left)")
    elapsed = time.time() - started

    scored = [r for r in records if r["outcome"] == "scored"]
    disagree = [r for r in scored if r["assigned"] != r["predicted"]]
    disagree.sort(key=lambda r: r["margin"])
    with open(out_path, "w") as fh:
        json.dump({"n_files": len(jobs), "n_phrases": len(records),
                   "n_scored": len(scored), "n_disagree": len(disagree),
                   "seconds": elapsed, "failures": failures,
                   "records": records}, fh)

    if verbose:
        print(f"\n[scan] {elapsed / 60:.1f} min for {len(jobs)} files "
              f"({len(jobs) / elapsed:.1f} files/s)")
        print(f"[scan] {len(scored)} scored, {len(disagree)} disagreements "
              f"({len(disagree) / max(1, len(scored)) * 100:.1f}%)")
        if failures:
            print(f"[scan] {len(failures)} file(s) failed: {failures[:3]}")
        print(f"[scan] -> {out_path} "
              f"({os.path.getsize(out_path) / 1e6:.1f} MB)")
        if len(jobs):
            per_file = elapsed / len(jobs)
            print(f"[scan] extrapolated to 3000 files: "
                  f"{per_file * 3000 / 60:.1f} min")
    return records


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    f = sub.add_parser("fit", help="build templates from verified annotations")
    f.add_argument("--annotations", default=config.ANNOTATION_FILE)
    f.add_argument("--audio-root", default=config.AUDIO_ROOT)
    f.add_argument("--out", default=os.path.join(HERE, "templates.npz"))
    f.add_argument("--n-keep", type=int, default=1)

    s = sub.add_parser("scan", help="score a corpus, streaming")
    s.add_argument("--annotations", default=config.ANNOTATION_FILE)
    s.add_argument("--audio-root", default=config.AUDIO_ROOT)
    s.add_argument("--templates", default=os.path.join(HERE, "templates.npz"))
    s.add_argument("--out", default=os.path.join(config.CACHE_DIR,
                                                 "flagged.json"))
    s.add_argument("--workers", type=int, default=None)
    s.add_argument("--limit", type=int, default=None,
                   help="scan only the first N files (for timing)")

    args = ap.parse_args()
    if args.command == "fit":
        fit(args.annotations, args.audio_root, args.out, n_keep=args.n_keep)
    else:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        scan(args.annotations, args.audio_root, args.templates, args.out,
             workers=args.workers, limit=args.limit)


if __name__ == "__main__":
    main()
