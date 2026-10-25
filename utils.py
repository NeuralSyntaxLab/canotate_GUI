"""
file that will contain utility function for technical function such as converting annotation files, reading formats etc.
"""

import os
import json
import re
import datetime
import scipy.io as sio
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


# ---------------------------------------------------------------------------
# Day folders and the date keys of the annotation file
#
# A bird's recordings are split into one folder per day, named either
# "<year>_<month>_<day>" or the same date behind a bird-name prefix
# ("YL10RB_2026-10-01", "lb 4444 2024_06_15"). The annotation file groups its
# entries by day, and the key it uses is always the plain "YYYY-MM-DD" date --
# never the folder name -- so a file written against one naming style still
# matches a dataset that uses the other, and so the batch pipeline (which
# reads the date out of the wav name) and the GUI agree.
# ---------------------------------------------------------------------------

# '-', '_' and spaces all separate. Bird names may end in a letter or a digit,
# so the year is taken to start after a separator, after a letter, or at the
# very start -- never by splitting a run of digits, which would let a numeric
# name like "RT10" bleed into the year.
DATE_FOLDER_RE = re.compile(
    r'(?:^|[-_\s]|(?<=[^\W\d_]))(\d{4})[-_\s](\d{2})[-_\s](\d{2})\s*$')


def folder_date(folder_name):
    """Return the ``YYYY-MM-DD`` a day-folder name encodes, or None if it is
    not a day folder.

    The date must be a real calendar date, so folders that merely end in three
    number groups are not mistaken for days.
    """
    m = DATE_FOLDER_RE.search(folder_name.strip())
    if not m:
        return None
    try:
        return datetime.date(*(int(g) for g in m.groups())).isoformat()
    except ValueError:
        return None


def annotation_date_key(name):
    """The key a day folder (or an existing annotation key) is stored under.

    Anything that does not encode a date, such as ``"unknown_date"``, is kept
    as it is.
    """
    return folder_date(name) or name


def _preferred_entry(a, b):
    """Of two entries for the same wav, the one worth keeping.

    A human's review beats an unreviewed entry, and an annotated one beats an
    empty one; otherwise the first one seen stays.
    """
    def rank(e):
        return (is_human_verified(e), bool(e.get("segments")))
    return b if rank(b) > rank(a) else a


def normalize_annotation_dates(annotation):
    """Re-key ``annotation`` in place so every day sits under its plain date.

    Entries filed under "YL10RB_2026-10-01" and under "2026-10-01" are merged
    into "2026-10-01". If both hold the same wav, the entry kept is chosen by
    :func:`_preferred_entry`.

    Returns the number of keys that were renamed or merged away, so a caller
    knows whether the file needs writing back.
    """
    merged = {}
    changed = 0
    for key, entries in annotation.items():
        new_key = annotation_date_key(key)
        if new_key != key:
            changed += 1
        day = merged.setdefault(new_key, [])
        for entry in entries:
            name = entry.get("file_name")
            idx = next((i for i, e in enumerate(day)
                        if e.get("file_name") == name), None)
            if idx is None:
                day.append(entry)
            else:
                day[idx] = _preferred_entry(day[idx], entry)
    annotation.clear()
    annotation.update(merged)
    return changed


def song_wavs_by_date(bird_folder):
    """Map each date key to ``{wav file name: path}`` for the song wavs under
    the bird folder's day folders.

    Day folders of the same date are pooled, whatever their naming style.
    """
    by_date = defaultdict(dict)
    for d in sorted(os.listdir(bird_folder)):
        day = os.path.join(bird_folder, d)
        if not (os.path.isdir(day) and folder_date(d)):
            continue
        for path in song_wavs_under(day):
            by_date[folder_date(d)].setdefault(os.path.basename(path), path)
    return dict(by_date)


def merge_json_files(json_file_1, json_file_2, output_file):
    """
    Merge two annotation JSON files into one.

    Structure of each file is {date: [{"file_name": str, "segments": [...]}, ...]}.

    Merge logic:
    - If a day exists in only one of the files, it is copied as-is.
    - If the same file (by name) exists in both files for a day, the whole entry
      that carries ``human_verified`` wins: a human review beats an unreviewed
      annotation. If neither entry is verified, or both are, the first file's
      entry is kept as the default.
    - Files that exist in only one of the two annotations for a shared day are
      copied as-is.

    Parameters:
    json_file_1 (str): Path to the first JSON file.
    json_file_2 (str): Path to the second JSON file.
    output_file (str): Path to the output merged JSON file.
    """
    with open(json_file_1, 'r') as f1, open(json_file_2, 'r') as f2:
        data1 = json.load(f1)
        data2 = json.load(f2)

    merged_data = {}
    all_dates = list(data1.keys()) + [d for d in data2 if d not in data1]
    data2 = _align_merged_annotations_to_master(data1, data2)

    for date in all_dates:
        # Day present in only one file -> copy as-is.
        if date not in data2:
            merged_data[date] = data1[date]
            continue
        if date not in data1:
            merged_data[date] = data2[date]
            continue

        # Day present in both files -> pick one entry per file_name.
        # Group file2 entries by name (lists, to handle repeated file names).
        entries_2 = defaultdict(list)
        for entry in data2[date]:
            entries_2[entry["file_name"]].append(entry)

        merged_entries = []
        for entry1 in data1[date]:
            name = entry1["file_name"]
            if entries_2[name]:
                entry2 = entries_2[name].pop(0)
                # The human-verified entry wins. When neither or both are
                # verified there is no reason to prefer one, so the first
                # file is the default.
                if is_human_verified(entry2) and not is_human_verified(entry1):
                    merged_entries.append(entry2)
                else:
                    merged_entries.append(entry1)
            else:
                # File only in the first annotation -> copy as-is.
                merged_entries.append(entry1)

        # Files only in the second annotation -> copy as-is (preserve order).
        for entry in data2[date]:
            if entry in entries_2[entry["file_name"]]:
                entries_2[entry["file_name"]].remove(entry)
                merged_entries.append(entry)

        merged_data[date] = merged_entries

    with open(output_file, 'w') as out_f:
        json.dump(merged_data, out_f, indent=4)
    new_annotated_files = get_unique_wav_paths(json_file_1, json_file_2)
    return new_annotated_files


def _segment_overlap(seg_a, seg_b):
    """Length of the time overlap between two [start, end, ...] segments (0 if disjoint)."""
    return max(0.0, min(seg_a[1], seg_b[1]) - max(seg_a[0], seg_b[0]))


def _align_merged_annotations_to_master(dataset_0, dataset_1):
    """
    some labels in some dataset are merged to another label in the master dataset.
    this method, finds diffs in label types, and aligns the labels in the second dataset to the master dataset.

    A label that appears in dataset_1 but nowhere in the master (dataset_0) is
    taken to have been merged away by the master annotator. Every dataset_1
    segment carrying that label is matched to the master segment it overlaps
    most in time (same date and file_name):
      - if there is such a segment, it takes the master's label for it, so a
        label the master split between several classes follows the master;
      - otherwise it takes the most frequent master label among all the
        matches of that label.
    Labels with no overlapping master segment anywhere are left unchanged.

    Parameters:
    dataset_0 (dict): master annotation (not modified).
    dataset_1 (dict): second annotation (not modified).

    Returns:
    dict: a copy of dataset_1 with the merged-away labels rewritten.
    """
    master_labels = set(count_label_classes(dataset_0))
    removed_labels = set(count_label_classes(dataset_1)) - master_labels
    if not removed_labels:
        return dataset_1

    master_segments = {
        (date, entry["file_name"]): entry.get("segments", [])
        for date, entries in dataset_0.items()
        for entry in entries
    }

    aligned = json.loads(json.dumps(dataset_1))  # deep copy
    votes = defaultdict(lambda: defaultdict(int))
    unmatched = []  # segments with a removed label and no overlapping master segment
    for date, entries in aligned.items():
        for entry in entries:
            master_segs = master_segments.get((date, entry["file_name"])) or []
            for seg in entry.get("segments", []):
                if len(seg) < 3 or _normalize_label(seg[2]) not in removed_labels:
                    continue
                best = max(master_segs, key=lambda m: _segment_overlap(seg, m), default=None)
                if best is not None and len(best) >= 3 and _segment_overlap(seg, best) > 0:
                    votes[_normalize_label(seg[2])][_normalize_label(best[2])] += 1
                    seg[2] = _normalize_label(best[2])
                else:
                    unmatched.append(seg)

    mapping = {}
    for label in sorted(removed_labels, key=str):
        if votes[label]:
            mapping[label] = max(votes[label], key=votes[label].get)
            print(f"Label {label} not in master -> overlaps follow master "
                  f"(votes: {dict(votes[label])}), elsewhere -> {mapping[label]}")
        else:
            print(f"Label {label} not in master and never overlaps it; left unchanged.")

    for seg in unmatched:
        label = _normalize_label(seg[2])
        if label in mapping:
            seg[2] = mapping[label]
    return aligned




def convert_mat_annotation_to_json_annotation(mat_file_path, json_file_path, bird_path):
    """
    Convert annotation from .mat format to .json format.

    Parameters:
    mat_file_path (str): Path to the input .mat file.
    json_file_path (str): Path to the output .json file.
    """
    # Load the .mat file
    mat_data = sio.loadmat(mat_file_path)
    wav_files = _get_wav_files(bird_path)
    # Assuming the .mat file constructed according to the Yarden's convention
    annotations_keys = mat_data.get('keys', [])[0]
    annotation_data = mat_data.get('elements', [])[0]
    json_output_data = {}
    for date, data in zip(annotations_keys, annotation_data):
        song_name = date[0]
        date = _extract_date_from_song_name(song_name)
        if not date in json_output_data.keys():
            json_output_data[date] = []
        annotation_segments = data['segFileStartTimes'][0][0].reshape(-1)
        annotation_segment_end = data['segFileEndTimes'][0][0].reshape(-1)
        annotation_labels = data['segType'][0][0].reshape(-1)
        json_output_data[date].append({"file_name": _get_recording_original_name(song_name, wav_files[date]),
                                       HUMAN_VERIFIED_KEY: False,
                                       "segments": []})
        for start, end, label in zip(annotation_segments, annotation_segment_end, annotation_labels):
            json_output_data[date][-1]["segments"].append([start, end, int(label)])

    # Save the JSON output
    with open(json_file_path, 'w') as json_file:
        json.dump(json_output_data, json_file, indent=4)

def add_human_verified_flag_to_json(json_file_path, default=True):
    """
    Add a 'human_verified' flag to each entry in the JSON annotation file.

    Parameters:
    json_file_path (str): Path to the input .json file.
    default (bool): Default value for the 'human_verified' flag.
    """
    with open(json_file_path, 'r') as f:
        annotation_data = json.load(f)

    n_added = ensure_human_verified_field(annotation_data, default=default)

    with open(json_file_path, 'w') as f:
        json.dump(annotation_data, f, indent=4)

    print(f"Added 'human_verified' flag to {n_added} entries in {json_file_path}.")

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


def merge_label_classes_in_file(json_file_path, source_labels, target_label,
                                out_path=None, backup=True):
    """Load an annotation JSON, merge label classes, and write the result.

    Parameters:
    json_file_path (str): path to the input annotation .json file.
    source_labels (int | iterable[int]): label(s) to be replaced.
    target_label (int): label to assign in place of each source label.
    out_path (str | None): where to write the result. Defaults to overwriting
        ``json_file_path`` in place.
    backup (bool): if writing in place, first copy the original to
        ``<json_file_path>.bak``.

    Returns:
    int: number of segments whose label was rewritten.
    """
    with open(json_file_path, "r") as f:
        annotation = json.load(f)

    out_path = out_path or json_file_path
    if backup and out_path == json_file_path:
        import shutil
        shutil.copyfile(json_file_path, json_file_path + ".bak")

    _, n_changed = merge_label_classes(annotation, source_labels, target_label)

    with open(out_path, "w") as f:
        json.dump(annotation, f, indent=2)
    return n_changed


def _extract_date_from_song_name(song_name):
    """
    Extract date from the song name.

    Parameters:
    song_name (str): The name of the song file given in format 'birdName_idx_year_month_day_hour_min_sec.wav'.

    Returns:
    str: Extracted date in 'YYYY-MM-DD' format.
    """
    # Assuming the date is in the format 'YYYYMMDD' and is part of the song name
    # Example: 'bird_idx_2024_21_12_06_58_11.wav' -> '2024-21-12'
    # Pattern to match: birdName_idx_year_month_day_hour_min_sec.wav
    # We want to capture year, month, day
    pattern = r".*_(\d{4})_(\d{2})_(\d{2})_\d{2}_\d{2}_\d{2}\.wav"
    match = re.search(pattern, song_name)
    
    if match:
        year, month, day = match.groups()
        return f"{year}-{month}-{day}"
    return "unknown_date"

def _get_recording_original_name(file_name, wav_files):
    pattern = r".*_(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})\.wav"
    match = re.search(pattern, file_name)
    if match:
        year, month, day, hour, minute, second = match.groups()
        time_stamp = f"{year}_{month}_{day}_{hour}_{minute}_{second}.wav"
        for wav_file in wav_files:
            if wav_file.endswith(time_stamp):
                return os.path.split(wav_file)[-1]
    return None

def _get_wav_files(bird_path):
    """
    Get all .wav files in the given bird path.

    Parameters:
    bird_path (str): Path to the bird directory.

    Returns:
    list: dict where keys are the dates and values are lists of .wav file paths of that date.
    """
    wav_files = defaultdict(list)
    for wav_path in song_wavs_under(bird_path):
        cur_date = _extract_date_from_song_name(os.path.basename(wav_path))
        wav_files[cur_date].append(wav_path)
    return wav_files

def integrate_annotation_correction_into_json(annotation_json, correction_data, output_data):
    """
    Integrate annotation corrections into the original annotation JSON.

    Parameters:
    annotation_json (dict): Original annotation data.
    correction_data (dict): Correction data to be integrated.
    output_data (str): Path to save the integrated annotation JSON.
    """
    pass

def remove_borring_wavs_from_auto_annotation(annotation_json, wav_dir, calls_labels:list=["17","23","25", "24"], minimal_labels_to_acount_as_song:int=2):
    """
    Remove boring wav files

    Parameters:
    annotation_json (dict): Original annotation data.
    wav_dir (str): Directory containing the wav files.
    """
    #load the annotation json
    with open(annotation_json, 'r') as f:
        annotation_data = json.load(f)
    wavs = song_wavs_by_date(wav_dir)
    for date, entries in annotation_data.items():
        for entry in entries:
            file_name = entry["file_name"]
            wav_path = wavs.get(annotation_date_key(date), {}).get(file_name)
            if wav_path is None:
                print(f"Warning: no wav for {date}/{file_name}.")
                continue
            # Check if the segments contain any of the calls_labels
            segments = entry.get("segments", [])
            songs_labels = [str(seg[2]) for seg in segments if str(seg[2]) not in calls_labels]
            if len(set(songs_labels)) <= minimal_labels_to_acount_as_song:
                print(f"Removing {file_name} from annotation as it contains only boring calls.")
                os.remove(wav_path)

def remove_annotation_if_no_wav(annotation_json, wav_dir):
    """
    Removes the file entry from an annotation json if the corresponding wav file does not exist in the specified directory.

    Parameters:
    annotation_json (str): Path to the annotation JSON, overwritten in place.
    wav_dir (str): Directory containing the wav files.
    """
    # Load the annotation json
    with open(annotation_json, 'r') as f:
        annotation_data = json.load(f)

    wavs = song_wavs_by_date(wav_dir)
    cleaned_data = {}
    removed_count = 0
    for date, entries in annotation_data.items():
        kept_entries = []
        for entry in entries:
            file_name = entry["file_name"]
            if file_name in wavs.get(annotation_date_key(date), {}):
                kept_entries.append(entry)
            else:
                print(f"Removing {file_name} from annotation as no {date} folder holds it.")
                removed_count += 1
        # Drop dates that no longer have any annotated file
        if kept_entries:
            cleaned_data[date] = kept_entries
        else:
            print(f"Removing date {date} from annotation as it has no remaining files.")

    print(f"Removed {removed_count} annotation entries without a matching wav file.")

    with open(annotation_json, 'w') as f:
        json.dump(cleaned_data, f, indent=4)



def remove_file_not_in_annotation(annotation_json, wav_dir):
    """
    Remove wav files that are not present in the annotation JSON.

    Parameters:
    annotation_json (dict): Original annotation data.
    wav_dir (str): Directory containing the wav files.
    """
    # Load the annotation json
    with open(annotation_json, 'r') as f:
        annotation_data = json.load(f)
    
    # Create a set of all file names in the annotation
    annotated_files = set()
    for entries in annotation_data.values():
        for entry in entries:
            annotated_files.add(entry["file_name"])
    
    # Only song wavs are pruned: a syllable clip or an export was never going
    # to appear in the annotation, so its absence there says nothing.
    for wav_path in song_wavs_under(wav_dir):
        if os.path.basename(wav_path) in annotated_files:
            continue
        print(f"Removing {wav_path} as it is not present in the annotation.")
        if os.path.exists(wav_path):
            os.remove(wav_path)
    


# ---------------------------------------------------------------------------
# Which wavs are songs
#
# A bird folder holds more audio than recordings: syllable clips the viewer
# writes to ``syllable_templates/``, ``no_song`` and ``noise`` siblings,
# exports, AppleDouble "._" side-cars.  The one thing that marks a file as a
# song is sitting directly inside a folder named ``wav``, usually at
#
#     <root>/<[bird_]date>/chop_data/wav/<file>.wav
#
# but the nesting varies between drives, so only the ``wav`` parent is
# required.  Every scan in the project goes through the two functions below,
# so the batch tools and the annotation GUI always see the same files -- and
# nothing else gets annotated, templated, or deleted.
# ---------------------------------------------------------------------------

SONG_WAV_FOLDER = 'wav'


def is_song_wav(path):
    """Return whether ``path`` is a song wav rather than some other audio.

    Only the tail of the path is inspected, so it works for relative and
    absolute paths alike, and regardless of where the dataset root sits.
    """
    path = os.path.normpath(str(path))
    base = os.path.basename(path)
    if not base.lower().endswith('.wav') or base.startswith('._'):
        return False
    return os.path.basename(os.path.dirname(path)).lower() == SONG_WAV_FOLDER


def song_wavs_under(root):
    """Every song wav anywhere beneath ``root``, sorted by full path."""
    found = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if os.path.basename(dirpath).lower() != SONG_WAV_FOLDER:
            continue
        found.extend(os.path.join(dirpath, f) for f in filenames
                     if f.lower().endswith('.wav') and not f.startswith('._'))
    return sorted(found)

def get_unique_wav_paths(json_1, json_2):
    """
    Get unique wav paths from two annotation JSON files.

    Parameters:
    json_1 (str): Path to the first JSON file.
    json_2 (str): Path to the second JSON file.

    Returns:
    set: Unique wav paths from both JSON files.
    """
    with open(json_1, 'r') as f1, open(json_2, 'r') as f2:
        data1 = json.load(f1)
        data2 = json.load(f2)

    wav_paths = []
    data = data1
    for entries in data.values():
        for entry in entries:
            wav_paths.append(entry["file_name"])
    new_files = set()
    for entries in data2.values():
        for entry in entries:
            if entry["file_name"] not in wav_paths:
                new_files.add(entry["file_name"])
    return new_files

if __name__=="__main__":
    # Example usage
    json_file_0 = '/Volumes/Extreme Pro/LY10RB/LY10RB_annotation_talgr_annotation_merged_1910.json'
    json_file_1 = '/Users/nitzankarby/Downloads/LY10RB_annotation_talgr_annotation_merged_2310.json'
    output = '/Volumes/Extreme Pro/LY10RB/LY10RB_annotation_talgr_annotation_merged_2310.json'
    # # merge_json_files(json_file_0, json_file_1, output)

    # print(len(get_unique_wav_paths('/Volumes/Extreme Pro/LY10RB/LY10RB_annotation_talgr_annotation_merged.json', "/Users/nitzankarby/Downloads/LY10RB_annotation_talgr_annotation_merged.json")))
    new_annotations = merge_json_files(json_file_0, json_file_1, output)
    print(f" {len(new_annotations)} Newly annotated files: {new_annotations}")