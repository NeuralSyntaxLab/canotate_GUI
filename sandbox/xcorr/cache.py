"""Stage 0 — turn annotations + audio into a single .npz of patches.

The external drive is the slow part of this sandbox, so it is read exactly
once.  Which instances are wanted is decided from the annotation metadata
*before* any audio is touched, so only files holding a wanted instance are
ever opened — the same discipline as
``dtw_suggestions.plan_instance_selection``, for the same reason.

Run:  python cache.py [--max-per-label N]
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# The sandbox reuses the project's channel-selection logic: these wavs are
# stereo (song + noise/trigger) and read_wav picks the song channel by
# spectral flatness.  Taking channel 0 blindly would sometimes analyse noise.
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)

import config
import spectro
from cannotate.cannotate_utils import read_wav


def load_annotations(path=None):
    """Return ``{(date, file_name): [(t0, t1, label), ...]}`` for labels in scope."""
    path = path or config.ANNOTATION_FILE
    with open(path) as fh:
        data = json.load(fh)
    wanted = set(config.LABELS) if config.LABELS is not None else None
    out = {}
    for date, entries in data.items():
        for entry in entries:
            segments = []
            for seg in entry.get("segments", []):
                if len(seg) < 3 or seg[2] is None:
                    continue
                try:
                    label = int(seg[2])
                    t0, t1 = float(seg[0]), float(seg[1])
                except (TypeError, ValueError):
                    continue
                if (wanted is None or label in wanted) and t1 > t0:
                    segments.append((t0, t1, label))
            if segments:
                out[(date, entry["file_name"])] = segments
    return out


def plan(annotations, max_per_label, seed=0):
    """Choose which instances to extract, without reading audio.

    Files are visited in a fixed random order and each label's quota is filled
    as it goes, so a common label does not pull in the whole corpus.  The
    random order keeps the sample unbiased across dates.
    """
    rng = np.random.default_rng(seed)
    keys = sorted(annotations)
    order = rng.permutation(len(keys))

    kept = defaultdict(int)
    plan_out = {}
    for i in order:
        key = keys[i]
        wanted = []
        for t0, t1, label in annotations[key]:
            if max_per_label and kept[label] >= max_per_label:
                continue
            kept[label] += 1
            wanted.append((t0, t1, label))
        if wanted:
            plan_out[key] = wanted
    return plan_out, dict(kept)


def wav_path(date, file_name):
    return os.path.join(config.AUDIO_ROOT, date, config.WAV_SUBDIR, file_name)


def build(max_per_label=None, out_path=None, verbose=True):
    max_per_label = (config.MAX_PER_LABEL if max_per_label is None
                     else max_per_label)
    out_path = out_path or config.PATCHES_NPZ
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    annotations = load_annotations()
    if config.LABELS is None:
        counts = defaultdict(int)
        for segs in annotations.values():
            for _, _, label in segs:
                counts[label] += 1
        config.LABELS = sorted(l for l, c in counts.items()
                               if c >= config.MIN_INSTANCES_PER_LABEL)
        if verbose:
            print(f"[cache] LABELS=all -> {len(config.LABELS)} labels with "
                  f">={config.MIN_INSTANCES_PER_LABEL} instances: "
                  f"{config.LABELS}")
        annotations = {k: [s for s in v if s[2] in set(config.LABELS)]
                       for k, v in annotations.items()}
        annotations = {k: v for k, v in annotations.items() if v}
    work, kept = plan(annotations, max_per_label, seed=config.RANDOM_SEED)
    if verbose:
        print(f"[cache] {len(annotations)} annotated files; "
              f"reading {len(work)} of them")
        print(f"[cache] per-label quota {max_per_label or 'unlimited'}: "
              f"{ {l: kept.get(l, 0) for l in config.LABELS} }")

    patches, meta = [], []
    F_band = None
    skipped = []
    start = time.time()
    for n, (key, wanted) in enumerate(sorted(work.items()), 1):
        date, file_name = key
        path = wav_path(date, file_name)
        try:
            fs, y = read_wav(path)
            y = np.asarray(y, dtype=np.float64)
            S, F, T = spectro.compute(y, fs)
        except Exception as exc:                      # noqa: BLE001
            skipped.append((path, repr(exc)))
            continue
        del y

        mask = (F >= config.CACHE_BAND[0]) & (F <= config.CACHE_BAND[1])
        S = S[mask, :]
        if F_band is None:
            F_band = F[mask]

        for t0, t1, label in wanted:
            cols = np.where((T >= t0) & (T <= t1))[0]
            if cols.size < 2:
                skipped.append((f"{file_name}@{t0:.3f}", "fewer than 2 columns"))
                continue
            patches.append(S[:, cols].astype(np.float32))
            meta.append((date, file_name, t0, t1, label, cols.size))
        del S

        if verbose and (n % 25 == 0 or n == len(work)):
            rate = n / (time.time() - start)
            print(f"[cache]   {n}/{len(work)} files "
                  f"({rate:.1f}/s, {len(patches)} patches)")

    if not patches:
        raise RuntimeError("No patches extracted — check AUDIO_ROOT.")

    lengths = np.array([p.shape[1] for p in patches], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(lengths)))
    stacked = np.concatenate(patches, axis=1)
    del patches

    # The patch data goes to its own .npy so it can be memory-mapped on read.
    # A .npz member cannot be mapped, so loading one would pull the whole
    # gigabyte into RAM to serve patches that are read a few columns at a time.
    data_path = os.path.splitext(out_path)[0] + "_data.npy"
    np.save(data_path, stacked)
    np.savez(
        out_path,
        offsets=offsets,
        F=F_band,
        date=np.array([m[0] for m in meta]),
        file_name=np.array([m[1] for m in meta]),
        t_start=np.array([m[2] for m in meta], dtype=np.float64),
        t_end=np.array([m[3] for m in meta], dtype=np.float64),
        label=np.array([m[4] for m in meta], dtype=np.int64),
        hop_seconds=np.array(config.HOP_SECONDS),
    )
    if verbose:
        size_mb = (os.path.getsize(out_path)
                   + os.path.getsize(data_path)) / 1e6
        print(f"[cache] wrote {len(meta)} patches "
              f"({stacked.shape[0]} bins x {stacked.shape[1]} cols, "
              f"{size_mb:.0f} MB) -> {out_path} (+ _data.npy, memory-mapped "
              f"on read)")
        if skipped:
            print(f"[cache] skipped {len(skipped)}: {skipped[:3]}")
    return out_path


class Patches:
    """Read-side view of the cache: indexable patches plus their metadata."""

    def __init__(self, path=None):
        path = path or config.PATCHES_NPZ
        # Mapped, not loaded: the sandbox reads a few hundred columns at a
        # time out of a multi-gigabyte array, so holding it resident is pure
        # waste and, on a smaller machine, fatal.
        self.data = np.load(os.path.splitext(path)[0] + "_data.npy",
                            mmap_mode="r")
        with np.load(path, allow_pickle=False) as z:
            self.offsets = z["offsets"]
            self.F = z["F"]
            self.date = z["date"]
            self.file_name = z["file_name"]
            self.t_start = z["t_start"]
            self.t_end = z["t_end"]
            self.label = z["label"]
            self.hop_seconds = float(z["hop_seconds"])
        # A cache may hold labels the current config does not list; the
        # experiment intersects them.
        if config.LABELS is None:
            config.LABELS = sorted(set(int(x) for x in np.unique(self.label)))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, i):
        # np.asarray materialises just this patch from the mapping.
        return np.asarray(self.data[:, self.offsets[i]:self.offsets[i + 1]])

    def indices_for(self, label):
        return np.where(self.label == label)[0]

    def duration(self, i):
        return float(self.t_end[i] - self.t_start[i])

    def cols_for_seconds(self, seconds):
        return max(1, int(round(seconds / self.hop_seconds)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-per-label", type=int, default=None,
                    help="0 keeps every instance")
    args = ap.parse_args()
    build(max_per_label=args.max_per_label)
