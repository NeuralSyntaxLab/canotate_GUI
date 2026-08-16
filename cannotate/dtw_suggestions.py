"""
DTW-based syllable suggestion helpers.

All spectrogram template building & matching logic lives here so
that both the viewer and the main GUI can call it without circular imports.
"""
import os
import json
import hashlib
import pickle
import random

import numpy as np
from scipy.signal import spectrogram, windows

from .cannotate_utils import read_wav


# Fixed number of time-columns every template is resampled to.
TEMPLATE_COLS = 50
# Maximum allowed relative duration difference between query and template.
MAX_DURATION_RATIO = 0.15
# Cap on instances used for the (O(n^2)) medoid DTW search; larger sets are
# randomly sub-sampled down to this many before the pairwise comparison.
MEDOID_MAX_INSTANCES = 50

# Per-label cap on the number of annotated instances used to build a template.
# The instances are chosen *up front* from the annotation metadata (see
# :func:`plan_instance_selection`), so only the wav files holding a chosen
# instance are ever read: neither memory nor startup time scales with the size
# of the annotated corpus. Kept above MEDOID_MAX_INSTANCES so the medoid search
# (which sub-samples to MEDOID_MAX_INSTANCES anyway) is statistically unaffected.
MAX_INSTANCES_PER_LABEL = 60

# ── Low-level helpers ────────────────────────────────────────────────

def extract_syllable_spectrogram(S_norm, T, t_start, t_end):
    """Extract normalised spectrogram columns for [t_start, t_end]."""
    if S_norm is None or T is None:
        return None
    indices = np.where((T >= t_start) & (T <= t_end))[0]
    if indices.size == 0:
        return None
    return S_norm[:, indices]


def resample_spectrogram_time(spec, target_cols=TEMPLATE_COLS):
    """Resample a spectrogram to *target_cols* time columns (linear interp)."""
    n_freq, n_time = spec.shape
    if n_time == 0:
        return spec
    x_old = np.linspace(0, 1, n_time)
    x_new = np.linspace(0, 1, target_cols)
    resampled = np.zeros((n_freq, target_cols))
    for i in range(n_freq):
        resampled[i, :] = np.interp(x_new, x_old, spec[i, :])
    return resampled


def dtw_distance(seq_a, seq_b):
    """DTW distance between two 2-D spectrograms (freq x time).

    Each column is a time-frame feature vector.  Uses a Sakoe-Chiba
    band constraint for speed.  Returns the normalised DTW distance
    (total cost / path length).
    """
    n = seq_a.shape[1]
    m = seq_b.shape[1]
    # Pre-compute pairwise Euclidean distances between columns
    a_T = seq_a.T  # (n, freq)
    b_T = seq_b.T  # (m, freq)
    diff = a_T[:, np.newaxis, :] - b_T[np.newaxis, :, :]  # (n, m, freq)
    cost = np.sqrt((diff ** 2).sum(axis=2))                # (n, m)

    w = max(5, int(0.15 * max(n, m)))  # Sakoe-Chiba band
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0
    for i in range(1, n + 1):
        j_lo = max(1, i - w)
        j_hi = min(m, i + w)
        for j in range(j_lo, j_hi + 1):
            dtw[i, j] = cost[i - 1, j - 1] + min(
                dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1]
            )
    path_len = n + m
    return dtw[n, m] / path_len if path_len > 0 else 0.0


def _subsample_for_medoid(items):
    """Randomly sub-sample *items* down to MEDOID_MAX_INSTANCES to bound the
    O(n^2) pairwise-DTW cost of the medoid search."""
    n = len(items)
    if n <= MEDOID_MAX_INSTANCES:
        return items
    return random.sample(items, MEDOID_MAX_INSTANCES)


def medoid_index(specs):
    """Index of the spectrogram with the smallest total DTW distance to all
    others (the medoid).  Returns 0 for trivial (<=1 element) inputs."""
    n = len(specs)
    if n <= 1:
        return 0
    total = np.zeros(n)
    for i in range(n):
        for j in range(i + 1, n):
            d = dtw_distance(specs[i], specs[j])
            total[i] += d
            total[j] += d
    return int(np.argmin(total))


def medoid_template(spec_dur_list):
    """Pick the *most representative real instance* for a syllable template.

    Given a list of ``(resampled_spec, duration)`` pairs, returns the
    ``(spec, duration)`` of the instance with the smallest total DTW
    distance to all other instances (the medoid).  This avoids the smearing
    produced by pixel-wise averaging of un-aligned spectrograms: the template
    is always a genuine, crisp example rather than a blurred mean.

    Returns ``(None, 0.0)`` for an empty list.
    """
    if not spec_dur_list:
        return None, 0.0
    items = _subsample_for_medoid(spec_dur_list)
    best = medoid_index([it[0] for it in items])
    return items[best][0], float(items[best][1])


def medoid_instance(instances):
    """Pick the medoid record from a list of instance dicts.

    Each dict must contain a ``"spec"`` key (a resampled spectrogram); other
    keys (e.g. provenance: ``full_path``, ``t_start``, ``t_end``) are carried
    through untouched.  Returns the chosen dict, or ``None`` for an empty list.
    """
    if not instances:
        return None
    items = _subsample_for_medoid(instances)
    best = medoid_index([it["spec"] for it in items])
    return items[best]


def compute_spectrogram_for_audio(y, fs, nfft=512):
    """Compute normalised spectrogram (same pipeline as *load_existing_data*).

    Returns (S_norm, T, F).
    """
    noverlap = nfft - int(0.001 * fs)
    NW = 4
    E = windows.dpss(nfft, NW, Kmax=2)
    w1, w2 = E[0], E[1]
    F, T, S1 = spectrogram(y, fs, window=w1, noverlap=noverlap,
                           nfft=nfft, nperseg=nfft)
    _, T, S2 = spectrogram(y, fs, window=w2, noverlap=noverlap,
                           nfft=nfft, nperseg=nfft)
    # Free each intermediate as soon as it is consumed and scale in place:
    # for a long recording every one of these arrays is hundreds of MB, so
    # holding them simultaneously is what drives the process out of memory.
    S_power = (np.abs(S1) + np.abs(S2)) / 2
    del S1, S2
    S_raw = np.log10(1 + S_power + 1e-12)
    del S_power
    eps = 1e-8
    minv, maxv = S_raw.min(), S_raw.max()
    S_raw -= minv
    S_raw /= (maxv - minv + eps)
    vmin = np.percentile(S_raw, 5) + eps
    vmax = np.percentile(S_raw, 97) + eps
    S_raw -= vmin
    S_raw /= (vmax - vmin)
    S_norm = np.clip(S_raw, 0, 1, out=S_raw)
    return S_norm, T, F


# ── Template building ────────────────────────────────────────────────

def plan_instance_selection(db_data, date_to_wavs, exclude_file=None,
                            max_instances_per_label=MAX_INSTANCES_PER_LABEL):
    """Choose which annotated instances to load — *without reading any audio*.

    Walks the annotation metadata only (already in memory as ``db_data``) and
    picks at most *max_instances_per_label* instances per syllable id.

    Selection is deliberately **file-aware**: the annotated files are visited in
    random order and the walk stops as soon as every label has filled its quota.
    Sampling instances independently at random would spread the picks across
    essentially every recording — capping the instances but still reading the
    whole corpus — whereas filling quotas file-by-file reads only as many wavs
    as the quotas actually require.  The random file order keeps the sample
    unbiased (no preference for the earliest dates); rare labels naturally pull
    in more files, since those are the only ones that contain them.

    *exclude_file* (a base file name) is skipped, e.g. to avoid double-counting
    the currently-loaded file when its segments are supplied separately.

    Returns ``{full_path: [(t_start, t_end, annot_id), ...]}`` — the per-file
    work list, so each needed wav is read exactly once.
    """
    if max_instances_per_label <= 0:
        return {}

    path_lookup = {}
    for d, paths in date_to_wavs.items():
        for fp in paths:
            path_lookup[(d, os.path.basename(fp))] = fp

    # 1) Metadata-only pass: what each annotated file offers, and how many
    #    appearances each label has in total. No audio is touched here.
    file_segs = []   # [(full_path, [(t0, t1, annot_id), ...]), ...]
    totals = {}      # annot_id -> total appearances in the corpus
    for date_str, entries in db_data.items():
        for entry in entries:
            fn = entry.get('file_name')
            if exclude_file is not None and fn == exclude_file:
                continue
            segs = entry.get('segments', [])
            if not segs:
                continue
            full_path = path_lookup.get((date_str, fn))
            if full_path is None:
                continue

            parsed = []
            for seg in segs:
                try:
                    s0, s1, lab = seg
                except Exception:
                    continue
                if lab is None:
                    continue
                try:
                    annot_id = int(lab)
                    t0, t1 = float(s0), float(s1)
                except (ValueError, TypeError):
                    continue
                parsed.append((t0, t1, annot_id))
                totals[annot_id] = totals.get(annot_id, 0) + 1
            if parsed:
                file_segs.append((full_path, parsed))

    if not file_segs:
        return {}

    # 2) Fill each label's quota from randomly-ordered files, stopping early.
    quota = {lab: min(max_instances_per_label, n) for lab, n in totals.items()}
    kept = {lab: 0 for lab in quota}
    unfilled = {lab for lab, q in quota.items() if q > 0}
    random.shuffle(file_segs)

    per_file = {}
    for full_path, parsed in file_segs:
        if not unfilled:
            break  # every label is satisfied — the remaining wavs are never read
        wanted = []
        for t0, t1, annot_id in parsed:
            if kept[annot_id] >= quota[annot_id]:
                continue
            kept[annot_id] += 1
            if kept[annot_id] >= quota[annot_id]:
                unfilled.discard(annot_id)
            wanted.append((t0, t1, annot_id))
        if wanted:
            per_file.setdefault(full_path, []).extend(wanted)
    return per_file


def collect_syllable_instances(db_data, date_to_wavs, nfft=512,
                               template_cols=TEMPLATE_COLS, exclude_file=None,
                               max_instances_per_label=MAX_INSTANCES_PER_LABEL,
                               verbose=False):
    """Gather annotated instances per syllable id, *with provenance*.

    Returns ``{annot_id: [instance, ...]}`` where each instance is a dict::

        {"spec": <resampled freq x template_cols array>,
         "duration": <float seconds>,
         "full_path": <source wav path>,
         "t_start": <float>, "t_end": <float>}

    Keeping the source file/time lets callers reproduce the *actual* audio
    and spectrogram of a chosen instance (e.g. the medoid template).

    Memory & time: the instances to keep are picked first from the annotation
    metadata (:func:`plan_instance_selection`, at most
    *max_instances_per_label* per syllable id), so **only the wav files holding
    a selected instance are read at all** — the rest of the corpus is never
    touched.  Those files are then processed one at a time, each file's
    spectrogram being released before the next is read, so peak memory is
    bounded by a single recording plus the (capped) set of kept patches.
    """
    per_file = plan_instance_selection(
        db_data, date_to_wavs, exclude_file=exclude_file,
        max_instances_per_label=max_instances_per_label)

    if verbose:
        n_sel = sum(len(v) for v in per_file.values())
        print(f"[templates] Selected {n_sel} instance(s) "
              f"(max {max_instances_per_label} per syllable) "
              f"from {len(per_file)} wav file(s)")

    instances = {}        # annot_id -> [instance dict, ...]  (bounded per label)

    for full_path, wanted in per_file.items():
        # Load audio & compute the full-file spectrogram for THIS file only.
        # S_n / T_ext are rebound (and freed) on the next iteration, so we
        # never hold more than one file's spectrogram at a time.
        try:
            fs_ext, y_ext = read_wav(full_path)
            if y_ext.ndim == 2:
                y_ext = y_ext.mean(axis=1)
            y_ext = y_ext.astype(np.float64)
            S_n, T_ext, _F_ext = compute_spectrogram_for_audio(
                y_ext, fs_ext, nfft=nfft)
        except Exception:
            continue
        del y_ext  # raw audio no longer needed; only the spectrogram is

        for t0, t1, annot_id in wanted:
            idx = np.where((T_ext >= t0) & (T_ext <= t1))[0]
            if idx.size == 0:
                continue
            patch = S_n[:, idx]
            instances.setdefault(annot_id, []).append({
                "spec": resample_spectrogram_time(
                    patch, template_cols).astype(np.float32),
                "duration": t1 - t0,
                "full_path": full_path,
                "t_start": t0,
                "t_end": t1,
            })

        del S_n, T_ext
    return instances


def build_global_syllable_templates(db_data, date_to_wavs, nfft=512,
                                    template_cols=TEMPLATE_COLS,
                                    max_instances_per_label=MAX_INSTANCES_PER_LABEL,
                                    verbose=False):
    """Build medoid spectrogram templates from a bounded sample of annotations.

    Called **once** at program startup from ``Cannotate()``.  Each template
    is the most representative real instance (the medoid by DTW distance),
    and carries the provenance of that instance so its audio/spectrogram can
    be reproduced on demand.

    Only up to *max_instances_per_label* randomly-chosen appearances of each
    syllable feed the medoid search, and only the wav files holding those
    appearances are read — the full corpus is never loaded.

    Returns ``{annot_id: {"spec", "avg_duration", "source_path",
    "source_start", "source_end"}}``.
    """
    instances = collect_syllable_instances(
        db_data, date_to_wavs, nfft=nfft, template_cols=template_cols,
        max_instances_per_label=max_instances_per_label, verbose=verbose)

    templates = {}
    for annot_id, inst_list in instances.items():
        inst = medoid_instance(inst_list)
        if inst is None:
            continue
        templates[annot_id] = {
            "spec": inst["spec"],
            "avg_duration": inst["duration"],
            "source_path": inst["full_path"],
            "source_start": inst["t_start"],
            "source_end": inst["t_end"],
        }
    return templates


# ── On-disk template cache ───────────────────────────────────────────

# Bump when the template format or build logic changes so that any stale cache
# on disk is transparently ignored (a mismatch triggers a rebuild, not a crash).
TEMPLATE_CACHE_VERSION = 1


def _templates_fingerprint(db_data, date_to_wavs, nfft, template_cols,
                           max_instances_per_label=MAX_INSTANCES_PER_LABEL):
    """Stable hash of everything that determines the built templates.

    Covers the annotations themselves, the identity of each resolved source
    wav (path + size + mtime, so replacing a wav on disk invalidates the
    cache), and the build parameters. Returns a hex-digest string.
    """
    path_lookup = {}
    for d, paths in date_to_wavs.items():
        for fp in paths:
            path_lookup[(d, os.path.basename(fp))] = fp

    wav_stats = []
    for date_str, entries in db_data.items():
        for entry in entries:
            if not entry.get('segments'):
                continue
            full_path = path_lookup.get((date_str, entry.get('file_name')))
            if full_path is None:
                continue
            try:
                st = os.stat(full_path)
                wav_stats.append([full_path, int(st.st_size), int(st.st_mtime_ns)])
            except OSError:
                wav_stats.append([full_path, -1, -1])
    wav_stats.sort()

    payload = {
        "version": TEMPLATE_CACHE_VERSION,
        "nfft": int(nfft),
        "template_cols": int(template_cols),
        "max_instances_per_label": int(max_instances_per_label),
        "annotations": db_data,
        "wav_stats": wav_stats,
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _load_templates_cache(cache_path, expected_fingerprint):
    """Return cached templates if present and matching, else ``None``.

    Any problem (missing file, corruption, version/fingerprint mismatch) yields
    ``None`` so the caller simply rebuilds — the cache is never allowed to break
    startup.
    """
    try:
        with open(cache_path, "rb") as f:
            blob = pickle.load(f)
    except (OSError, pickle.UnpicklingError, EOFError, ValueError):
        return None
    if not isinstance(blob, dict):
        return None
    if blob.get("cache_version") != TEMPLATE_CACHE_VERSION:
        return None
    if blob.get("fingerprint") != expected_fingerprint:
        return None
    templates = blob.get("templates")
    if not isinstance(templates, dict):
        return None
    return templates


def _save_templates_cache(cache_path, templates, fingerprint, nfft, template_cols):
    """Persist templates + fingerprint atomically. Failures are swallowed."""
    tmp_path = f"{cache_path}.tmp"
    blob = {
        "cache_version": TEMPLATE_CACHE_VERSION,
        "fingerprint": fingerprint,
        "nfft": int(nfft),
        "template_cols": int(template_cols),
        "templates": templates,
    }
    try:
        with open(tmp_path, "wb") as f:
            pickle.dump(blob, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, cache_path)  # atomic on the same filesystem
    except OSError:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass


def build_global_syllable_templates_cached(db_data, date_to_wavs, cache_path,
                                           nfft=512, template_cols=TEMPLATE_COLS,
                                           max_instances_per_label=MAX_INSTANCES_PER_LABEL,
                                           verbose=True):
    """Like :func:`build_global_syllable_templates`, backed by an on-disk cache.

    On a cache hit (identical annotations, source wavs and params) the templates
    are loaded directly and **no wav files are read** — startup becomes
    near-instant. On a miss the templates are built from a bounded random sample
    of each syllable's appearances (at most *max_instances_per_label*, reading
    only the wavs that hold them — see :func:`collect_syllable_instances`) and
    written back to *cache_path* for next time.
    """
    fingerprint = _templates_fingerprint(
        db_data, date_to_wavs, nfft, template_cols, max_instances_per_label)
    cached = _load_templates_cache(cache_path, fingerprint)
    if cached is not None:
        if verbose:
            print(f"[templates] Loaded {len(cached)} template(s) from cache: {cache_path}")
        return cached

    templates = build_global_syllable_templates(
        db_data, date_to_wavs, nfft=nfft, template_cols=template_cols,
        max_instances_per_label=max_instances_per_label, verbose=verbose)
    _save_templates_cache(cache_path, templates, fingerprint, nfft, template_cols)
    if verbose:
        print(f"[templates] Built {len(templates)} template(s); cached to: {cache_path}")
    return templates


def build_syllable_templates(S_norm, T, segments, db_data, date_to_wavs,
                             nfft=512, wav_filename=None,
                             template_cols=TEMPLATE_COLS,
                             max_instances_per_label=MAX_INSTANCES_PER_LABEL):
    """Rebuild medoid templates using *current file* segments + global annotations.

    Parameters mirror the viewer state that is needed.  Each template is the
    medoid instance and carries provenance (``source_path`` is ``None`` for
    instances taken from the currently-loaded file).

    The current file's segments are always used in full (its spectrogram is
    already in memory); the global side contributes at most
    *max_instances_per_label* sampled instances per syllable, so a rebuild never
    walks the whole corpus.
    """
    instances = {}

    # 1) Current file's segments (spectrogram already loaded). These have no
    #    on-disk path here, so source_path is None -> "the current file".
    if S_norm is not None and T is not None:
        for seg in segments:
            t_start, t_end, annot = seg[0], seg[1], seg[2]
            if annot is None:
                continue
            annot_id = int(annot)
            spec = extract_syllable_spectrogram(S_norm, T, t_start, t_end)
            if spec is not None and spec.shape[1] > 0:
                instances.setdefault(annot_id, []).append({
                    "spec": resample_spectrogram_time(spec, template_cols),
                    "duration": t_end - t_start,
                    "full_path": None,
                    "t_start": t_start,
                    "t_end": t_end,
                })

    # 2) Global annotations from every other file.
    if db_data and date_to_wavs:
        global_inst = collect_syllable_instances(
            db_data, date_to_wavs, nfft=nfft, template_cols=template_cols,
            exclude_file=wav_filename,
            max_instances_per_label=max_instances_per_label)
        for annot_id, inst_list in global_inst.items():
            instances.setdefault(annot_id, []).extend(inst_list)

    templates = {}
    for annot_id, inst_list in instances.items():
        inst = medoid_instance(inst_list)
        if inst is None:
            continue
        templates[annot_id] = {
            "spec": inst["spec"],
            "avg_duration": inst["duration"],
            "source_path": inst["full_path"],
            "source_start": inst["t_start"],
            "source_end": inst["t_end"],
        }
    return templates



def syllable_suggestions(S_norm, T, syllable_templates, t_start, t_end,
                         template_cols=TEMPLATE_COLS,
                         max_duration_ratio=MAX_DURATION_RATIO):
    """Return syllable-ID suggestions for the region [t_start, t_end].

    Templates whose average duration differs from the query duration by
    more than *max_duration_ratio* (default 15%) are skipped.

    Returns ``[(annot_id, distance), ...]`` sorted ascending by distance.
    """
    if not syllable_templates:
        return []

    query_spec = extract_syllable_spectrogram(S_norm, T, t_start, t_end)
    if query_spec is None or query_spec.shape[1] == 0:
        return []

    query_duration = t_end - t_start
    query_resampled = resample_spectrogram_time(query_spec, template_cols)

    results = []
    for annot_id, tmpl_data in syllable_templates.items():
        template_spec = tmpl_data["spec"]
        template_dur = tmpl_data.get("avg_duration", 0.0)

        # Skip templates whose duration is too different from the query
        if template_dur > 0 and query_duration > 0:
            longer = max(template_dur, query_duration)
            shorter = min(template_dur, query_duration)
            if (longer - shorter) / longer > max_duration_ratio:
                continue

        dist = dtw_distance(query_resampled, template_spec)
        results.append((annot_id, dist))
    results.sort(key=lambda x: x[1])
    return results
