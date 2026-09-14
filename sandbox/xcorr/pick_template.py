#!/usr/bin/env python
"""Hand-pick a label's template window, with the numbers that matter shown.

The automatic rule takes a *centred* 200 ms window from the most consistent
phrase.  That is what produces the 3 -> 26 failure: template 26 is only the
sustained harmonic stack, a strict subset of what a label-3 rendition contains,
so it wins on peak ZNCC against anything holding that stack.

Choosing by eye alone risks trading one confusion for another, so every
candidate window here is scored two ways:

    self   mean peak ZNCC against other phrases of its OWN label  (must stay high)
    other  mean peak ZNCC against phrases of the CONFUSABLE label (want low)
    gap    self - other                                           (maximise)

A window with a high gap is *specific*: it matches its own label and not the
neighbour.  That is the property the centred-window rule does not optimise for.

    python pick_template.py --label 3 --against 26      # browse, then open the page
    python pick_template.py --set 3=p41@0.32 26=p07@0.10
    python pick_template.py --show                      # current manual picks

Picks are written to manual_templates.json and honoured by templates.py.
"""
import argparse
import base64
import io
import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import config
import scoring
from cache import Patches

PICKS_FILE = os.path.join(HERE, "manual_templates.json")

# Candidates scored per TemplateBank.  A few hundred candidates in one bank
# would cache hundreds of MB of spectra; 32 keeps a chunk well under
# TemplateBank.MAX_SPECTRA_BYTES at any realistic phrase length.
CHUNK = 32


def load_picks():
    if os.path.exists(PICKS_FILE):
        with open(PICKS_FILE) as fh:
            return json.load(fh)
    return {}


def save_picks(picks):
    with open(PICKS_FILE, "w") as fh:
        json.dump(picks, fh, indent=2, sort_keys=True)
    return PICKS_FILE


def most_confused_with(label, split_by=None):
    """The label most often confused with *label* in the last experiment run.

    Saves having to remember which pair is the problem: the experiment already
    measured it.  Returns ``None`` when no results exist yet.
    """
    split_by = split_by or config.SPLIT_BY
    path = os.path.join(config.CACHE_DIR, f"results_{split_by}.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        records = json.load(fh)["records"]
    counts = defaultdict(int)
    for r in records:
        a, p = r.get("assigned"), r.get("predicted")
        if a == p:
            continue
        if a == label and p is not None:
            counts[p] += 1
        elif p == label and a is not None:
            counts[a] += 1
    return max(counts, key=counts.get) if counts else None


def parse_pick(pick_id):
    """``p41@0.32`` -> (41, 0.32): phrase index and window start in seconds."""
    phrase, _, offset = pick_id.partition("@")
    return int(phrase.lstrip("p")), float(offset)


def window_from_pick(patches, pick_id, cols, band):
    """Return the band-limited window a pick identifies."""
    index, offset = parse_pick(pick_id)
    start = int(round(offset / patches.hop_seconds))
    patch = patches[index]
    if start + cols > patch.shape[1]:
        raise ValueError(f"Pick {pick_id} runs past the end of phrase {index} "
                         f"({patch.shape[1]} cols).")
    window = patch[:, start:start + cols]
    return scoring.band_limit(window, patches.F, band[0], band[1])[0]


def _thumbnail(array, height=1.05, width=1.5):
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(array, aspect="auto", origin="lower", cmap="gray_r")
    ax.set_xticks([]); ax.set_yticks([])
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=96)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _phrase_strip(array, hop, marks, width=11.0):
    fig = plt.figure(figsize=(width, 1.5))
    ax = fig.add_axes([0.04, 0.22, 0.95, 0.72])
    ax.imshow(array, aspect="auto", origin="lower", cmap="gray_r",
              extent=[0, array.shape[1] * hop, 0, 1])
    for offset, cols, best in marks:
        ax.axvline(offset, color="tab:red" if best else "tab:blue",
                   lw=1.6 if best else 0.6,
                   alpha=1.0 if best else 0.45)
    ax.set_yticks([])
    ax.set_xlabel("time (s)", fontsize=8)
    ax.tick_params(labelsize=7)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=96)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def browse(label, against, n_phrases=8, stride_seconds=0.05,
           n_reference=25, band=None, cols=None, out_path=None,
           split_by=None):
    patches = Patches()
    band = band or (1000.0, 13000.0)
    cols = cols or patches.cols_for_seconds(config.TEMPLATE_MAX_SECONDS)

    # Candidates may only come from the template split.  Hand-picking a window
    # out of a phrase the experiment later scores would leak the answer into
    # the measurement, and the leak would be invisible in the confusion matrix.
    import experiment as experiment_mod
    split_by = split_by or config.SPLIT_BY
    allowed = set(int(i) for i in experiment_mod.split_groups(
        patches, by=split_by, seed=config.RANDOM_SEED)[0])

    rng = np.random.default_rng(config.RANDOM_SEED)
    own = [i for i in patches.indices_for(label)
           if patches[i].shape[1] >= cols and int(i) in allowed]
    rival = [i for i in patches.indices_for(against)
             if patches[i].shape[1] >= cols and int(i) in allowed]
    print(f"[pick] candidates restricted to the {split_by} template split: "
          f"{len(own)} label-{label} and {len(rival)} label-{against} phrases")
    if len(own) < 2 or not rival:
        raise SystemExit(f"Not enough phrases: label {label} has {len(own)}, "
                         f"label {against} has {len(rival)}.")

    source = list(rng.permutation(own)[:n_phrases])
    own_ref = [i for i in rng.permutation(own)[:n_reference] if i not in source]
    rival_ref = list(rng.permutation(rival)[:n_reference])

    # Every candidate window from every source phrase, at a regular stride.
    candidates = {}
    stride = max(1, int(round(stride_seconds / patches.hop_seconds)))
    for index in source:
        patch = scoring.band_limit(patches[index], patches.F, *band)[0]
        for start in range(0, patch.shape[1] - cols + 1, stride):
            pick = f"p{index}@{start * patches.hop_seconds:.2f}"
            candidates[pick] = patch[:, start:start + cols]
    if not candidates:
        raise SystemExit("No candidate windows — phrases are too short.")

    # Score candidates in chunks.  A bank holding all of them at once would
    # cache hundreds of MB of spectra; TemplateBank refuses that outright, and
    # chunking is the answer it asks for.  Each chunk still transforms every
    # reference phrase only once for all of its candidates.
    reference = []
    for group, indices in (("self", own_ref), ("other", rival_ref)):
        for i in indices:
            phrase = scoring.band_limit(patches[i], patches.F, *band)[0]
            if phrase.shape[1] >= cols:
                reference.append((group, phrase))

    sums = defaultdict(lambda: [0.0, 0, 0.0, 0])
    keys = list(candidates)
    for start in range(0, len(keys), CHUNK):
        chunk = keys[start:start + CHUNK]
        bank = scoring.TemplateBank({k: [candidates[k]] for k in chunk})
        for group, phrase in reference:
            for pick, stack in bank.surfaces(phrase).items():
                if stack[0] is None:
                    continue
                peak = float(stack[0].max())
                slot = sums[pick]
                if group == "self":
                    slot[0] += peak; slot[1] += 1
                else:
                    slot[2] += peak; slot[3] += 1
        del bank

    rows = []
    for pick, (s, sn, o, on) in sums.items():
        self_score = s / sn if sn else float("nan")
        other_score = o / on if on else float("nan")
        rows.append({"pick": pick, "self": self_score, "other": other_score,
                     "gap": self_score - other_score})
    rows.sort(key=lambda r: -r["gap"])
    best = {r["pick"] for r in rows[:3]}
    by_pick = {r["pick"]: r for r in rows}

    out_path = out_path or os.path.join(config.FIGURE_DIR,
                                        f"pick_{label}_vs_{against}.html")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    sections = []
    for index in source:
        patch = scoring.band_limit(patches[index], patches.F, *band)[0]
        picks = [p for p in candidates if p.startswith(f"p{index}@")]
        picks.sort(key=lambda p: parse_pick(p)[1])
        marks = [(parse_pick(p)[1], cols, p in best) for p in picks]
        strip = _phrase_strip(patch, patches.hop_seconds, marks)
        cells = []
        for p in picks:
            r = by_pick[p]
            cells.append(f"""
      <figure class="{'best' if p in best else ''}">
        <img src="data:image/png;base64,{_thumbnail(candidates[p])}">
        <figcaption><code>{p}</code><br>
          self {r['self']:.3f}<br>other {r['other']:.3f}<br>
          <b>gap {r['gap']:+.3f}</b></figcaption>
      </figure>""")
        sections.append(f"""
  <section>
    <h2>phrase {index} &mdash; {patches.date[index]}
        {patches.t_start[index]:.2f}&ndash;{patches.t_end[index]:.2f}s</h2>
    <img class=strip src="data:image/png;base64,{strip}">
    <div class=row>{''.join(cells)}</div>
  </section>""")

    table = "".join(
        f"<tr><td><code>{r['pick']}</code></td><td>{r['self']:.3f}</td>"
        f"<td>{r['other']:.3f}</td><td><b>{r['gap']:+.3f}</b></td></tr>"
        for r in rows[:15])

    html = f"""<!doctype html><meta charset=utf-8>
<title>pick template {label} vs {against}</title>
<style>
 body {{ font: 14px/1.5 -apple-system, system-ui, sans-serif; margin: 0 auto;
        padding: 24px; max-width: 1500px; color: #1a1a1a; }}
 h1 {{ font-size: 20px; margin: 0 0 4px; }}
 h2 {{ font-size: 14px; font-weight: 600; margin: 0 0 6px; color: #444; }}
 .lede {{ color: #555; max-width: 860px; }}
 section {{ border-top: 1px solid #ddd; padding: 16px 0; }}
 .strip {{ width: 100%; }}
 .row {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px; }}
 figure {{ margin: 0; width: 148px; }}
 figure img {{ width: 100%; border: 1px solid #e0e0e0; display: block; }}
 figure.best img {{ border: 2px solid #c0392b; }}
 figcaption {{ font-size: 10.5px; color: #555; line-height: 1.35;
               font-variant-numeric: tabular-nums; }}
 code {{ font-size: 10.5px; color: #1a1a1a; }}
 table {{ border-collapse: collapse; margin: 12px 0 8px; font-size: 13px;
          font-variant-numeric: tabular-nums; }}
 th, td {{ text-align: left; padding: 3px 16px 3px 0; }}
 th {{ border-bottom: 1px solid #ccc; }}
 pre {{ background: #f6f6f6; padding: 10px 12px; font-size: 12.5px;
        overflow-x: auto; }}
</style>
<h1>Template candidates for label {label}, against label {against}</h1>
<p class=lede>
  Every {cols * patches.hop_seconds * 1000:.0f} ms window from
  {len(source)} phrases of label {label}, at a
  {stride_seconds * 1000:.0f} ms stride.
  <b>self</b> is the mean peak ZNCC against {len(own_ref)} other label-{label}
  phrases (keep high); <b>other</b> is the mean against {len(rival_ref)}
  label-{against} phrases (want low); <b>gap</b> is the difference &mdash;
  how <i>specific</i> the window is, which is exactly what the automatic
  centred-window rule does not optimise for.
  The three best gaps are outlined in red.
</p>
<h2>best gaps</h2>
<table><tr><th>pick<th>self<th>other<th>gap</tr>{table}</table>
<pre>python pick_template.py --set {label}={rows[0]['pick']}</pre>
{''.join(sections)}
"""
    with open(out_path, "w") as fh:
        fh.write(html)
    print(f"\nlabel {label} vs {against}: {len(rows)} candidate windows")
    print(f"  {'pick':<14} {'self':>7} {'other':>7} {'gap':>8}")
    for r in rows[:8]:
        print(f"  {r['pick']:<14} {r['self']:>7.3f} {r['other']:>7.3f} "
              f"{r['gap']:>+8.3f}")
    print(f"\nopen {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", type=int)
    ap.add_argument("--against", type=int,
                    help="the confusable label; inferred from the last "
                         "experiment run when omitted")
    ap.add_argument("--phrases", type=int, default=8)
    ap.add_argument("--stride", type=float, default=0.05,
                    help="seconds between candidate windows")
    ap.add_argument("--set", nargs="+", metavar="LABEL=PICK",
                    help="record picks, e.g. --set 3=p41@0.32 26=p07@0.10")
    ap.add_argument("--clear", nargs="+", type=int, metavar="LABEL")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--split", choices=["date", "file"], default=None)
    ap.add_argument("--open", action="store_true")
    args = ap.parse_args()

    if args.set or args.clear:
        picks = load_picks()
        for item in args.set or []:
            label, _, pick = item.partition("=")
            parse_pick(pick)                      # fail early on a bad id
            picks[str(int(label))] = pick
        for label in args.clear or []:
            picks.pop(str(label), None)
        print(f"manual picks -> {save_picks(picks)}")
        print(json.dumps(picks, indent=2, sort_keys=True))
        print("\nRe-run the experiment to see the effect:")
        print("  python experiment.py --split date")
        return

    if args.show or not args.label:
        picks = load_picks()
        print(json.dumps(picks, indent=2, sort_keys=True) if picks
              else "No manual picks recorded.")
        return

    against = args.against
    if against is None:
        against = most_confused_with(args.label, args.split)
        if against is None:
            raise SystemExit(
                f"--against not given and no experiment results to infer it "
                f"from.\nRun the experiment first:\n"
                f"    python experiment.py --split {args.split or config.SPLIT_BY}\n"
                f"or name the confusable label yourself:\n"
                f"    python pick_template.py --label {args.label} --against N")
        print(f"[pick] --against not given; label {against} is the one most "
              f"often confused with {args.label} in the last run")
    path = browse(args.label, against, n_phrases=args.phrases,
                  stride_seconds=args.stride, split_by=args.split)
    if args.open:
        import subprocess
        subprocess.run(["open", path], check=False)


if __name__ == "__main__":
    main()
