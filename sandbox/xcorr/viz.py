"""Stage 4 — look at individual cases.

A confusion matrix says *how often*; these figures say *why*.  Each one shows
the phrase, every label's template, and every label's ZNCC surface, so a
disagreement can be adjudicated by eye — which is the only way to tell an
annotation error from a method error.

Run:  python viz.py [--cases 12] [--only-errors] [--split date]
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import scoring
from cache import Patches


def _spectrogram_extent(patch, hop, F):
    return [0, patch.shape[1] * hop, F[0] / 1000.0, F[-1] / 1000.0]


def figure_for_case(patches, record, specs, out_path, max_panels=6):
    """One figure: the phrase, each template, and each ZNCC surface."""
    i = record["index"]
    patch = patches[i]
    hop = patches.hop_seconds
    assigned, predicted = record["assigned"], record["predicted"]
    # With many labels a full row of panels is unreadable, and most of them
    # score near zero anyway.  Show the assigned and predicted labels plus the
    # next best competitors — the ones a human actually has to weigh.
    ranked = sorted(record["scores"], key=lambda l: -record["scores"][l]["peak"])
    keep = [assigned, predicted]
    for l in (int(x) for x in ranked):
        if len(keep) >= max_panels:
            break
        if l not in keep:
            keep.append(l)
    labels = [l for l in sorted(specs) if l in keep]

    n = len(labels)
    fig = plt.figure(figsize=(3.1 * n, 8.4))
    grid = fig.add_gridspec(3, n, height_ratios=[1.5, 1.0, 1.0], hspace=0.45,
                            wspace=0.28)

    # Row 0: the phrase itself, spanning the full width.
    ax = fig.add_subplot(grid[0, :])
    band = specs[assigned]["band"]
    shown, F_shown = scoring.band_limit(patch, patches.F, band[0], band[1])
    ax.imshow(shown, aspect="auto", origin="lower", cmap="gray_r",
              extent=_spectrogram_extent(shown, hop, F_shown))
    verdict = "AGREES" if assigned == predicted else "DISAGREES"
    ax.set_title(
        f"{record['file_name']}  {record['t_start']:.2f}-{record['t_end']:.2f}s"
        f"   |   annotated {assigned}, method says {predicted}  [{verdict}]"
        f"   margin {record['margin']:+.3f}",
        fontsize=11,
        color=("black" if assigned == predicted else "firebrick"))
    ax.set_xlabel("time (s)")
    ax.set_ylabel("kHz")

    for k, label in enumerate(labels):
        spec = specs[label]
        template = spec["templates"][0]

        ax = fig.add_subplot(grid[1, k])
        ax.imshow(template, aspect="auto", origin="lower", cmap="gray_r")
        ax.set_xticks([])
        ax.set_yticks([])
        colour = ("seagreen" if label == assigned else
                  "firebrick" if label == predicted else "black")
        ax.set_title(f"template {label}", fontsize=10, color=colour)

        # Surface: recomputed rather than stored, so the figure always matches
        # the current scoring code rather than a stale run.
        query, _ = scoring.band_limit(patch, patches.F, spec["band"][0],
                                      spec["band"][1])
        ax = fig.add_subplot(grid[2, k])
        if query.shape[1] < template.shape[1]:
            ax.text(0.5, 0.5, "phrase shorter\nthan template",
                    ha="center", va="center", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            continue
        surface = scoring.zncc_surface(query, template)
        peak, lag, psr = scoring.peak_psr(surface, template.shape[1])
        ax.plot(np.arange(surface.size) * hop, surface, lw=0.9, color=colour)
        ax.axvline(lag * hop, color=colour, ls="--", lw=0.8)
        ax.set_ylim(-0.6, 1.0)
        ax.set_xlabel("lag (s)", fontsize=8)
        if k == 0:
            ax.set_ylabel("ZNCC", fontsize=8)
        ax.set_title(f"peak {peak:.3f}   PSR {psr:.1f}", fontsize=9,
                     color=colour)
        ax.tick_params(labelsize=7)

    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def run(n_cases=12, only_errors=False, split_by=None):
    split_by = split_by or config.SPLIT_BY
    results_path = os.path.join(config.CACHE_DIR, f"results_{split_by}.json")
    if not os.path.exists(results_path):
        raise SystemExit(f"No results at {results_path} — run experiment.py first.")
    with open(results_path) as fh:
        results = json.load(fh)

    # Templates are rebuilt from the same split so the figures show exactly
    # what the experiment scored.
    import experiment as experiment_mod
    import templates as templates_mod
    from collections import defaultdict

    patches = Patches()
    template_idx, _, _ = experiment_mod.split_groups(
        patches, by=split_by, seed=config.RANDOM_SEED)
    pools = defaultdict(list)
    for i in template_idx:
        pools[int(patches.label[i])].append(int(i))
    specs = templates_mod.build(pools, patches=patches, verbose=False)

    records = results["records"]
    errors = [r for r in records if r["assigned"] != r["predicted"]]
    if only_errors:
        chosen = errors[:n_cases]
        if not chosen:
            print("No disagreements to show — the method agreed everywhere.")
            return
    else:
        # Disagreements first (they are what needs adjudicating), then the
        # thinnest agreements, which are where a threshold would sit.
        agree = sorted((r for r in records if r["assigned"] == r["predicted"]),
                       key=lambda r: r["margin"])
        chosen = (errors + agree)[:n_cases]

    os.makedirs(config.FIGURE_DIR, exist_ok=True)
    made = []
    for n, r in enumerate(chosen, 1):
        tag = "err" if r["assigned"] != r["predicted"] else "ok"
        name = (f"{tag}_{r['assigned']}as{r['predicted']}_"
                f"m{r['margin']:+.3f}_{r['index']}.png")
        figure_for_case(patches, r, specs,
                        os.path.join(config.FIGURE_DIR, name))
        made.append((name, r))
        print(f"  [{n}/{len(chosen)}] {name}")

    index = write_index(made, results, len(errors))
    print(f"\n{len(made)} figure(s); {len(errors)} disagreement(s) in the run")
    print(f"open {index}")
    return index


def write_index(made, results, n_errors):
    """One scrollable page over every figure, worst disagreement first.

    Adjudicating a disagreement means looking at it, and forty-eight separate
    PNGs is not a thing anyone looks at.
    """
    rows = []
    for name, r in made:
        disagrees = r["assigned"] != r["predicted"]
        ranked = sorted(r["scores"].items(),
                        key=lambda kv: -kv[1]["peak"])[:4]
        scores = " &nbsp; ".join(
            f"<b>{l}</b>&nbsp;{v['peak']:.3f}"
            f"<span class=psr>/{v['psr']:.1f}</span>" for l, v in ranked)
        rows.append(f"""
  <section class="{'err' if disagrees else 'ok'}">
    <h2>{'DISAGREES' if disagrees else 'agrees'} &mdash;
        annotated <b>{r['assigned']}</b>, method says
        <b>{r['predicted']}</b>, margin {r['margin']:+.3f}</h2>
    <p class=meta>{r['date']} &nbsp; {r['file_name']} &nbsp;
       {r['t_start']:.2f}&ndash;{r['t_end']:.2f}s</p>
    <p class=meta>top peaks (peak/PSR): {scores}</p>
    <img src="{name}" alt="{name}">
  </section>""")

    accuracy = results.get("accuracy", float("nan")) * 100
    html = f"""<!doctype html><meta charset=utf-8>
<title>xcorr sandbox &mdash; {n_errors} disagreements</title>
<style>
 body {{ font: 14px/1.5 -apple-system, system-ui, sans-serif;
        margin: 0 auto; padding: 24px; max-width: 1500px; color: #1a1a1a; }}
 h1 {{ font-size: 20px; margin: 0 0 4px; }}
 .lede {{ color: #555; margin: 0 0 24px; }}
 section {{ border-top: 1px solid #ddd; padding: 18px 0; }}
 section.err h2 {{ color: #a01b1b; }}
 h2 {{ font-size: 15px; font-weight: 600; margin: 0 0 4px; }}
 .meta {{ color: #666; margin: 2px 0; font-size: 13px;
          font-variant-numeric: tabular-nums; }}
 .psr {{ color: #999; }}
 img {{ width: 100%; height: auto; margin-top: 10px;
        border: 1px solid #e5e5e5; }}
</style>
<h1>Cross-correlation sandbox &mdash; case review</h1>
<p class=lede>
  {len(results.get('labels', []))} labels, split by {results['split']['by']},
  accuracy {accuracy:.1f}%, {n_errors} disagreement(s).
  Red headings are cases where the method disagrees with the annotator &mdash;
  each is either an annotation error or a method error, and only looking
  tells you which.
</p>
{''.join(rows)}
"""
    path = os.path.join(config.FIGURE_DIR, "index.html")
    with open(path, "w") as fh:
        fh.write(html)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", type=int, default=12)
    ap.add_argument("--only-errors", action="store_true")
    ap.add_argument("--split", choices=["date", "file"], default=None)
    ap.add_argument("--open", action="store_true", help="open the index")
    args = ap.parse_args()
    index = run(n_cases=args.cases, only_errors=args.only_errors,
                split_by=args.split)
    if args.open and index:
        import subprocess
        subprocess.run(["open", index], check=False)
