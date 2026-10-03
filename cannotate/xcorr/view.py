"""Look at a fitted template bank.

:func:`describe` in :mod:`cannotate.xcorr.bank` says how many templates a bank
holds and how wide they are; only a picture says whether they are the
syllables you meant.  A template that looks like silence, or like a neighbour
label, is the difference between a scan worth trusting and one that quietly
flags the wrong phrases.

This module is deliberately separate from :mod:`cannotate.xcorr.bank`:
``bank`` is imported by every scan worker, and a matplotlib import in that
path would cost each worker's startup for something it never draws.

No backend is selected here.  The sandbox pins ``Agg`` because it writes
files; a window needs whatever backend the machine actually has, and the
tests pin ``Agg`` themselves.
"""
import math

import matplotlib.pyplot as plt


def _panels(bank, labels=None):
    """``(label, rank, template)`` for every template to draw, in label order.

    *rank* is 1-based and only meaningful when a label holds more than one
    template, which happens when the bank was fitted with ``--n-keep`` above
    one.
    """
    if labels is None:
        wanted = bank.labels
    else:
        missing = [label for label in labels if label not in bank.templates]
        if missing:
            raise ValueError(
                f"The bank has no template for "
                f"{', '.join(str(m) for m in missing)}. It holds: "
                f"{', '.join(str(label) for label in bank.labels)}.")
        wanted = sorted(set(labels))

    return [(label, rank, template)
            for label in wanted
            for rank, template in enumerate(bank.templates[label], 1)]


def _title(label, rank, total):
    return f"label {label}" if total == 1 else f"label {label} #{rank}"


def figure(bank, labels=None):
    """A grid of every template in *bank*, one panel each.

    Panels are drawn the way the Error Correction window draws the template
    beside a flagged phrase — ``gray_r``, origin lower, band-limited extent —
    so the same syllable looks the same in both places.

    *labels* restricts the grid; asking for a label the bank lacks raises
    :class:`ValueError` rather than silently drawing fewer panels.
    """
    panels = _panels(bank, labels)
    n_cols = max(1, math.ceil(math.sqrt(len(panels))))
    n_rows = math.ceil(len(panels) / n_cols)

    fig = plt.figure(figsize=(min(3.0 * n_cols, 16), min(2.4 * n_rows, 10)))
    for position, (label, rank, template) in enumerate(panels):
        ax = fig.add_subplot(n_rows, n_cols, position + 1)
        ax.imshow(template, aspect="auto", origin="lower",
                  extent=[0, template.shape[1] * bank.hop_seconds,
                          bank.band[0], bank.band[1]],
                  cmap="gray_r")
        ax.set_title(_title(label, rank, len(bank.templates[label])),
                     fontsize=9)
        ax.tick_params(labelsize=6)
        # Only the outer panels carry tick labels: a grid of twenty fully
        # labelled axes is unreadable, and every panel shares the same axes.
        if position % n_cols:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("Hz", fontsize=7)
        if position < len(panels) - n_cols:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("s", fontsize=7)

    fig.tight_layout()
    return fig
