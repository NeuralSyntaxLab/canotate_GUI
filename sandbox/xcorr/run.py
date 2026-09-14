#!/usr/bin/env python
"""One command to run the whole sandbox and open the results.

    python run.py                 # the 7 requested labels (4 5 6 9 10 15 16)
    python run.py --all-labels    # every label with >=30 instances  <- the honest test
    python run.py --skip-cache    # reuse the cached patches (the slow part)

Stages: cache (reads the external drive once, ~4 min) -> experiment (~30 s)
-> figures + an HTML index, which is opened in a browser unless --no-open.

The 7-label run scores 100%, which is why --all-labels exists: a closed set of
seven stereotyped syllable types from one bird is an easy question.  See
README.md.
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = ("/private/tmp/claude-501/-Users-nitzankarby-Desktop-masters-"
                 "yarden-cohen-dev-cannotate/"
                 "8084d949-d156-4331-b7f9-c1fbcd12ad30/scratchpad/xcorr")


def stage(title, argv, env):
    print(f"\n{'=' * 72}\n== {title}\n{'=' * 72}", flush=True)
    started = time.time()
    result = subprocess.run([sys.executable] + argv, cwd=HERE, env=env)
    if result.returncode != 0:
        sys.exit(f"\n!! {title} failed (exit {result.returncode})")
    print(f"-- {title} took {time.time() - started:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all-labels", action="store_true",
                    help="every label with >=30 instances, not just the 7")
    ap.add_argument("--labels", default=None,
                    help="explicit comma list, e.g. 4,5,6")
    ap.add_argument("--skip-cache", action="store_true",
                    help="reuse existing patches.npz (skips the slow stage)")
    ap.add_argument("--max-per-label", type=int, default=None)
    ap.add_argument("--split", choices=["date", "file"], default="date")
    ap.add_argument("--n-templates", type=int, default=None)
    ap.add_argument("--tie-break-psr", action="store_true")
    ap.add_argument("--cases", type=int, default=40,
                    help="figures to render (disagreements come first)")
    ap.add_argument("--only-errors", action="store_true")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    env = dict(os.environ)
    if args.labels:
        env["XCORR_LABELS"] = args.labels
        suffix = "_" + args.labels.replace(",", "-")
    elif args.all_labels:
        env["XCORR_LABELS"] = "all"
        suffix = "_all"
    else:
        suffix = ""
    # Each label set gets its own cache: mixing them would silently score
    # against templates for labels the run does not contain.
    env["XCORR_CACHE_DIR"] = args.cache_dir or (DEFAULT_CACHE + suffix)
    print(f"labels    : {env.get('XCORR_LABELS', 'config default (7)')}")
    print(f"cache dir : {env['XCORR_CACHE_DIR']}")

    patches = os.path.join(env["XCORR_CACHE_DIR"], "patches.npz")
    if args.skip_cache and not os.path.exists(patches):
        sys.exit(f"!! --skip-cache but no cache at {patches}")
    if not args.skip_cache:
        argv = ["cache.py"]
        if args.max_per_label is not None:
            argv += ["--max-per-label", str(args.max_per_label)]
        elif args.all_labels:
            argv += ["--max-per-label", "120"]
        stage("1/3  cache  (reads the external drive)", argv, env)
    else:
        print(f"\n-- reusing {patches}")

    argv = ["experiment.py", "--split", args.split]
    if args.n_templates:
        argv += ["--n-templates", str(args.n_templates)]
    if args.tie_break_psr:
        argv += ["--tie-break-psr"]
    stage("2/3  experiment  (confusion matrix, margins, controls)", argv, env)

    argv = ["viz.py", "--split", args.split, "--cases", str(args.cases)]
    if args.only_errors:
        argv += ["--only-errors"]
    if not args.no_open:
        argv += ["--open"]
    stage("3/3  figures", argv, env)

    print("\nDone.  To dig further:")
    print(f"  cd {HERE}")
    print("  python experiment.py --shuffle-templates   # control: must give ~0%")
    print("  python experiment.py --split file          # session-effect check")
    print("  python experiment.py --n-templates 3       # bimodality probe")
    print("  python -m pytest test_scoring.py -q")


if __name__ == "__main__":
    main()
