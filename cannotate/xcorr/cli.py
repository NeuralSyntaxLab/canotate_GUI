"""Command line for the template bank.

    python -m cannotate.xcorr fit --annotations verified.json \\
                                  --audio-root /Volumes/Drive/BIRD \\
                                  --out templates.npz
    python -m cannotate.xcorr describe --templates templates.npz
    python -m cannotate.xcorr show --templates templates.npz

Fitting is a *separate* step from correcting on purpose.  The bank must come
from annotations you trust — fitting it from the same file you are about to
correct would let a systematic mislabel define the template that then blesses
it.  Run this once against verified data; the Error Correction window only
ever loads the result.
"""
import argparse
import json
import os
import sys


def wav_paths_for_root(audio_root):
    """Map ``(date, file_name)`` to an absolute path under *audio_root*.

    A date is a folder directly inside the root; its songs are what
    :func:`utils.song_wavs_under` finds beneath it.  That is the same rule the
    annotation GUI applies, so fitting and the GUI always agree on which files
    exist, whatever the nesting.  The date is the annotation file's key for
    the folder (:func:`utils.annotation_date_key`), so "YL10RB_2026-10-01"
    and "2026-10-01" both come out as "2026-10-01".
    """
    import utils

    paths = {}
    try:
        dates = sorted(d for d in os.listdir(audio_root)
                       if os.path.isdir(os.path.join(audio_root, d)))
    except OSError as exc:
        raise ValueError(f"Cannot read the audio root {audio_root}: {exc}")
    for date in dates:
        key = utils.annotation_date_key(date)
        for path in utils.song_wavs_under(os.path.join(audio_root, date)):
            paths.setdefault((key, os.path.basename(path)), path)
    return paths


def phrase_specs(db_data):
    """``(date, file_name, t_start, t_end, label)`` for every phrase.

    Phrases — runs of consecutive same-label segments — are the unit the
    templates are cut from and the unit Error Correction flags, so the two
    must be built the same way.  That grouping lives in
    :mod:`cannotate.error_correction`; importing it here costs a tkinter
    import in a one-shot command, which is cheaper than a second copy of the
    rule that could drift from the first.
    """
    from ..error_correction import build_all_phrases

    return [(p.date, p.file_name, p.start, p.end, p.label)
            for p in build_all_phrases(db_data)
            if p.label is not None]


def _parse_labels(text):
    """``"3,26"`` or ``"3 26"`` -> ``[3, 26]``; ``None`` stays ``None``."""
    if not text:
        return None
    try:
        return [int(x) for x in text.replace(",", " ").split()]
    except ValueError:
        raise ValueError(f"--labels wants whole numbers, not {text!r}.")


def _load_annotations(path):
    import utils

    try:
        with open(path) as fh:
            db_data = json.load(fh)
    except OSError as exc:
        raise ValueError(f"Cannot read the annotation file: {exc}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}")
    # Days filed under a folder name match the plain dates of the audio root.
    utils.normalize_annotation_dates(db_data)
    return db_data


def _fit(args):
    from . import bank as bank_mod
    from cannotate.cannotate_utils import read_wav

    verbose = not args.quiet
    db_data = _load_annotations(args.annotations)
    specs = phrase_specs(db_data)
    if not specs:
        raise ValueError(f"{args.annotations} holds no labelled phrases.")

    wav_paths = wav_paths_for_root(args.audio_root)
    if not wav_paths:
        raise ValueError(
            f"No wav files found under {args.audio_root}. Expected "
            f"<audio-root>/<date>/.../wav/*.wav.")

    known = [s for s in specs if (s[0], s[1]) in wav_paths]
    if not known:
        raise ValueError(
            f"None of the {len(specs)} annotated phrases match a wav under "
            f"{args.audio_root} — is this the right bird folder?")
    if verbose and len(known) < len(specs):
        print(f"[fit] {len(specs) - len(known)} phrase(s) have no wav and "
              f"are ignored")

    def load_song(date, file_name):
        return read_wav(wav_paths[(date, file_name)])

    labels = _parse_labels(args.labels)

    if verbose:
        print(f"[fit] {len(known)} phrase(s), "
              f"{len({(s[0], s[1]) for s in known})} file(s)")

    fitted = bank_mod.fit(known, load_song, labels=labels,
                          band=(args.band[0], args.band[1]),
                          n_candidates=args.n_candidates,
                          n_keep=args.n_keep, verbose=verbose)

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fitted.save(out)
    if verbose:
        print(bank_mod.describe(out))
    return 0


def _show(args):
    """Draw every template in a saved bank and put it on screen.

    Read-only, so it is the safe thing to run against a bank you are unsure
    of — and the numbers ``describe`` prints go to the terminal alongside the
    window, since "22 labels" and a picture of them answer different halves of
    "is this bank any good?".
    """
    from . import bank as bank_mod
    from . import view

    try:
        loaded = bank_mod.Bank.load(args.templates)
    except Exception as e:                                    # noqa: BLE001
        raise ValueError(f"{os.path.basename(args.templates)} is not a "
                         f"readable template bank: {e}")

    labels = _parse_labels(args.labels)
    # The figure is built before anything is printed, so a command that draws
    # nothing — a typo in --labels — does not also dump a summary implying it
    # worked.
    figure = view.figure(loaded, labels=labels)
    print(bank_mod.describe(args.templates))
    if labels is not None:
        # Without this the terminal reads "22 label(s)" beside a window
        # showing two, which looks like the bank, not the flag, is wrong.
        print(f"  showing {', '.join(str(x) for x in sorted(set(labels)))}")
    # Under a headless backend there is no window to title, and saying so is
    # not worth an error: the figure is still correct.
    manager = getattr(figure.canvas, "manager", None)
    if manager is not None:
        manager.set_window_title(os.path.basename(args.templates))
    view.plt.show()
    return 0


def _describe(args):
    from . import bank as bank_mod

    print(bank_mod.describe(args.templates))
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        prog="python -m cannotate.xcorr", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    fit = sub.add_parser("fit", help="build templates from verified annotations")
    fit.add_argument("--annotations", required=True)
    fit.add_argument("--audio-root", required=True,
                     help="the bird folder holding <date>/.../wav/*.wav")
    fit.add_argument("--out", required=True, help="where to write templates.npz")
    fit.add_argument("--labels", default=None,
                     help="comma list to restrict fitting (default: all)")
    fit.add_argument("--band", nargs=2, type=float, default=(1000.0, 13000.0),
                     metavar=("FMIN", "FMAX"))
    fit.add_argument("--n-candidates", type=int, default=30,
                     help="phrases read per label when ranking (default 30)")
    fit.add_argument("--n-keep", type=int, default=1,
                     help="templates kept per label (default 1)")
    fit.add_argument("--quiet", action="store_true")
    fit.set_defaults(func=_fit)

    show = sub.add_parser("show", help="draw a saved bank's templates")
    show.add_argument("--templates", required=True)
    show.add_argument("--labels", default=None,
                      help="comma list to restrict the grid (default: all)")
    show.set_defaults(func=_show)

    describe = sub.add_parser("describe", help="summarise a saved bank")
    describe.add_argument("--templates", required=True)
    describe.set_defaults(func=_describe)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
