import json




def _merge_close_segments(annotation_data, gap_threshold=2/100):
    """
    Merges consecutive segments whose gap is smaller than gap_threshold (seconds).
    Keeps the minimum label of the merged group.
    """
    if not annotation_data:
        return annotation_data
    merged = [list(annotation_data[0])]
    for seg in annotation_data[1:]:
        gap = seg[0] - merged[-1][1]
        if gap < gap_threshold:
            merged[-1][1] = seg[1]                        # extend end time
            merged[-1][2] = min(merged[-1][2], seg[2])   # keep min label
        else:
            merged.append(list(seg))
    return merged


def _compute_overlap_ratio(seg_a, seg_b):
    """Returns overlap as a fraction of the shorter segment's duration."""
    overlap = max(0.0, min(seg_a[1], seg_b[1]) - max(seg_a[0], seg_b[0]))
    if overlap == 0.0:
        return 0.0
    min_duration = min(seg_a[1] - seg_a[0], seg_b[1] - seg_b[0])
    return overlap / min_duration if min_duration > 0 else 0.0



def _merge_segment_lists(segs_a, segs_b):
    """
    Merges two lists of [time_start, time_end, label] segments:
      - >50% overlap  -> [max(start), min(end), min(label)]
      - 0–50% overlap -> keep the segment with the lower (more specific) label
      - no overlap    -> keep both as-is
    """
    merged = []
    used_b = set()

    for seg_a in segs_a:
        best_ratio, best_j = 0.0, -1
        for j, seg_b in enumerate(segs_b):
            if j in used_b:
                continue
            ratio = _compute_overlap_ratio(seg_a, seg_b)
            if ratio > best_ratio:
                best_ratio, best_j = ratio, j

        if best_j >= 0 and best_ratio > 0.5:
            seg_b = segs_b[best_j]
            merged.append([max(seg_a[0], seg_b[0]), min(seg_a[1], seg_b[1]), min(seg_a[2], seg_b[2])])
            used_b.add(best_j)
        elif best_j >= 0 and best_ratio > 0.0:
            seg_b = segs_b[best_j]
            merged.append(seg_a if seg_a[2] <= seg_b[2] else seg_b)
            used_b.add(best_j)
        else:
            merged.append(seg_a)

    # Add unmatched segments from b
    for j, seg_b in enumerate(segs_b):
        if j not in used_b:
            merged.append(seg_b)

    merged.sort(key=lambda s: s[0])
    return merged

def merge_annotations(annotation_a, annotation_b, output_file_path):
    """
    merges annotations from 2 sources into single annotation json
    Both files are structured as {date_key: [list of {file_name, segments}]}
    Files are matched by bird name + recording timestamp via _get_assential_info_from_filename.
    Segment merge prioritizes valid annotations (!= 999).
    return: None (saves to json_file)
    """
    with open(annotation_a,"r") as f_a:
        data_a = json.load(f_a)
    with open(annotation_b, "r") as f_b:
        data_b = json.load(f_b)

    # Collect all date keys from both sources
    all_dates = set(data_a.keys()) | set(data_b.keys())
    merged_result = {}

    for date_key in all_dates:
        print(date_key)
        files_a = data_a.get(date_key, [])
        files_b = data_b.get(date_key, [])

        # Build lookup from essential name -> file entry for source b
        b_by_essential = {}
        for entry_b in files_b:
            if entry_b["file_name"] is None:
                print(f"Warning: file_name is None in annotation_b for date {date_key}. Skipping entry.")
                continue
            essential_b = _get_assential_info_from_filename(entry_b["file_name"])
            b_by_essential[essential_b] = entry_b

        merged_files = []
        used_b_keys = set()

        for entry_a in files_a:
            if entry_a["file_name"] is None:
                print(f"Warning: file_name is None in annotation_a for date {date_key}. Skipping entry.")
                continue
            essential_a = _get_assential_info_from_filename(entry_a["file_name"])
            entry_b = b_by_essential.get(essential_a)
            if entry_b is not None:
                used_b_keys.add(essential_a)
                merged_segs = _merge_segment_lists(entry_a["segments"], entry_b["segments"])
                merged_files.append({"file_name": entry_a["file_name"], "segments": merged_segs})
            else:
                merged_files.append(entry_a)

        # Add files only present in b
        for essential_b, entry_b in b_by_essential.items():
            if essential_b not in used_b_keys:
                merged_files.append(entry_b)

        merged_result[date_key] = merged_files

    with open(output_file_path, "w") as f:
        json.dump(merged_result, f, indent=4)
    print(f"Merged annotations saved to {output_file_path}. Successfully merged {len(all_dates)} date(s).")


def _get_assential_info_from_filename(filename):
    birds_name = filename.split('_')[0]  # Assuming bird name is the first part of the filename
    time = filename.split('_')[-6:]  # Assuming time is the last 6 parts of the filename
    return birds_name + '_' + '_'.join(time)

def match_names(keys_a, keys_b):
    """
    names of wav files may differ in the data they hold, we will match them by birds name and date of recording (if exists in the name)
    """
    matched_keys = []
    for key_a in keys_a:
        for key_b in keys_b:
            assential_name_a = _get_assential_info_from_filename(key_a)
            assential_name_b = _get_assential_info_from_filename(key_b)
            if assential_name_a == assential_name_b:  # Match by bird name and time
                matched_keys.append(assential_name_a)
                break
    return matched_keys


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Merge two annotation JSON files.")
    parser.add_argument("-annotation_a", type=str, help="Path to the first annotation JSON file.")
    parser.add_argument("-annotation_b", type=str, help="Path to the second annotation JSON file.")
    parser.add_argument("-output_file_path", type=str, help="Path to save the merged annotation JSON file.")
    args = parser.parse_args()
    annotation_a = args.annotation_a
    annotation_b = args.annotation_b
    output_file_path = args.output_file_path
    merge_annotations(annotation_a, annotation_b, output_file_path)


if __name__ == "__main__":
    main()