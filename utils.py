"""
file that will contain utility function for technical function such as converting annotation files, reading formats etc.
"""

import os
import json
import re
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
    for date, entries in annotation_data.items():
        for entry in entries:
            file_name = entry["file_name"]
            wav_path = os.path.join(wav_dir, date, "chop_data", "wav", file_name)
            if not os.path.exists(wav_path):
                print(f"Warning: {wav_path} does not exist.")
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

    cleaned_data = {}
    removed_count = 0
    for date, entries in annotation_data.items():
        kept_entries = []
        for entry in entries:
            file_name = entry["file_name"]
            wav_path = os.path.join(wav_dir, date, "chop_data", "wav", file_name)
            if os.path.exists(wav_path):
                kept_entries.append(entry)
            else:
                print(f"Removing {file_name} from annotation as {wav_path} does not exist.")
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


if __name__=="__main__":
    # Example usage
    # json_file_1 = '/Volumes/Extreme Pro/LBRK219RT/LBRK219_auto_merged_with_human_verified.json'
    # json_file_0 = '/Volumes/Extreme Pro/LBRK219RT/LBRK219_auto_merged_with_human_verified_jacob.json'
    # output = '/Volumes/Extreme Pro/LBRK219RT/LBRK219_auto_merged_with_human_verified.json'
    # # merge_json_files(json_file_0, json_file_1, output)
    # # add_human_verified_flag_to_json("/Volumes/Extreme Pro/LBRK219RT/LBRK219RT_annotation_merged_22_07_26.json", default=True)
    # remove_borring_wavs_from_auto_annotation('/Volumes/Extreme Pro/LBRK219RT/LBRK219_auto_merged_with_human_verified.json', "/Volumes/Extreme Pro/LBRK219RT", calls_labels=["17","23","25", "24"], minimal_labels_to_acount_as_song=2)
    # remove_annotation_if_no_wav('/Volumes/Extreme Pro/LBRK219RT/LBRK219_auto_merged_with_human_verified.json', "/Volumes/Extreme Pro/LBRK219RT")
    merge_json_files('/Volumes/Extreme Pro/LY10RB/LY10RB_annotation_talgr_annotation_merged.json', "/Users/nitzankarby/Downloads/LY10RB_annotation_talgr_annotation_pessia_0924.json", '/Volumes/Extreme Pro/LY10RB/LY10RB_annotation_talgr_annotation_merged.json')