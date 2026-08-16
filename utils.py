"""Annotation-file helpers used by the Cannotate GUI.

This is a trimmed copy of the ``utils.py`` from the birdsong-preprocessing-gui
project. That file also held format converters (MAT→JSON), dataset-pruning
helpers and JSON-merging tools, none of which the annotation tool imports.
Only the six names below are used, so only those were carried over — which also
drops the scipy and sqlalchemy imports the original file carried.
"""

from collections import defaultdict

# Name of the per-file flag recording that a human reviewed the annotation.
HUMAN_VERIFIED_KEY = "human_verified"


def is_human_verified(entry):
    """Return whether a file entry has been reviewed by a human.

    A missing field counts as not verified, so annotation files written by
    other tools (or by older versions of this one) keep loading.
    """
    return bool(entry.get(HUMAN_VERIFIED_KEY, False))


def ensure_human_verified_field(annotation, default=False):
    """Add a missing ``human_verified`` field to every file entry, in place.

    Entries that already carry the field are left untouched, so a file that has
    been migrated once is never rewritten again.

    The field is inserted right after ``file_name`` rather than appended, so it
    stays readable at the top of an entry instead of trailing a long
    ``segments`` list. Entry dicts are rewritten in place (same object), so
    callers holding references to them keep seeing the update.

    Parameters:
    annotation (dict): Cannotate-format annotation (mutated in place).
    default (bool): value given to entries that lack the field.

    Returns:
    int: number of entries that gained the field.
    """
    n_added = 0
    for entries in annotation.values():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict) or HUMAN_VERIFIED_KEY in entry:
                continue
            reordered = {}
            for key, value in entry.items():
                reordered[key] = value
                if key == "file_name":
                    reordered[HUMAN_VERIFIED_KEY] = bool(default)
            reordered.setdefault(HUMAN_VERIFIED_KEY, bool(default))
            entry.clear()
            entry.update(reordered)
            n_added += 1
    return n_added


def _normalize_label(label):
    """Return label as int when it looks like an integer, otherwise unchanged.

    Annotation segments store the label as the 3rd element ([start, end, label]).
    Labels are normally integers but may arrive as strings from the UI, so we
    coerce to int when possible to keep comparisons/merges consistent.
    """
    try:
        return int(label)
    except (TypeError, ValueError):
        return label


def count_label_classes(annotation):
    """Count how many segments use each label class across the whole annotation.

    Parameters:
    annotation (dict): Cannotate-format annotation
        {date: [{"file_name": str, "segments": [[start, end, label], ...]}, ...]}.

    Returns:
    dict: {label -> number of segments with that label}, labels normalized to int.
    """
    counts = defaultdict(int)
    for entries in annotation.values():
        for entry in entries:
            for seg in entry.get("segments", []):
                if len(seg) < 3:
                    continue
                counts[_normalize_label(seg[2])] += 1
    return dict(counts)


def merge_label_classes(annotation, source_labels, target_label):
    """Merge one or more label classes into a single target label, in place.

    Every segment whose label is in ``source_labels`` has its label rewritten to
    ``target_label`` across all dates and files. Useful for collapsing
    over-split syllable classes after annotation.

    Parameters:
    annotation (dict): Cannotate-format annotation (mutated in place).
    source_labels (int | iterable[int]): label(s) to be replaced.
    target_label (int): label to assign in place of each source label.

    Returns:
    tuple: (annotation, n_changed) where n_changed is the number of segments
        whose label was rewritten.
    """
    if isinstance(source_labels, (str, bytes, int)):
        source_labels = [source_labels]
    sources = {_normalize_label(lbl) for lbl in source_labels}
    target = _normalize_label(target_label)
    sources.discard(target)  # merging a label into itself is a no-op

    n_changed = 0
    for entries in annotation.values():
        for entry in entries:
            for seg in entry.get("segments", []):
                if len(seg) < 3:
                    continue
                if _normalize_label(seg[2]) in sources:
                    seg[2] = target
                    n_changed += 1
    return annotation, n_changed
