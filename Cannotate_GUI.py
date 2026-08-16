"""
Cannotate_GUI.py  --  main entry-point for the bird-song annotation tool.

All heavy-lifting is delegated to extracted modules:
    cannotate_utils    - shared helpers, constants
    navigator          - FullEntropyNavigator
    spectrogram_viewer - IntegratedSpectrogramViewer
    dtw_suggestions    - DTW template logic
    template_browser   - template concatenation browser
"""
import os
import re
import json
import glob

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from tkinter import ttk
import numpy as np
import glob

# -- project modules -------------------------------------------------------
from cannotate import (
    clean_segments,
    binary_to_segments,
    segments_to_binary,
    FullEntropyNavigator,
    IntegratedSpectrogramViewer,
    open_template_browser,
    regenerate_disk_templates,
)
from cannotate import read_wav, read_wav_channels
from utils import (
    count_label_classes,
    merge_label_classes,
    ensure_human_verified_field,
    is_human_verified,
    HUMAN_VERIFIED_KEY,
)


# Prefixes shown in front of a WAV file name in the selection combobox so the
# annotator can see at a glance which songs still need a human review.
VERIFIED_PREFIX = "✓ "
UNVERIFIED_PREFIX = "○ "


def _base_from_display(text):
    """Strip the verification marker from a combobox entry.

    Tolerates an unprefixed string so a stale selection still resolves to a
    real file name.
    """
    for prefix in (VERIFIED_PREFIX, UNVERIFIED_PREFIX):
        if text.startswith(prefix):
            return text[len(prefix):]
    return text


# How many appearances of each syllable are loaded to build its DTW template.
# Templates are built from this many randomly-chosen instances per label instead
# of from every annotation, so startup cost stays flat as the dataset grows.
# Override with the CANNOTATE_TEMPLATE_INSTANCES environment variable.
try:
    TEMPLATE_INSTANCES_PER_SYLLABLE = int(
        os.environ.get("CANNOTATE_TEMPLATE_INSTANCES", "60"))
except ValueError:
    TEMPLATE_INSTANCES_PER_SYLLABLE = 60


# -------------------------------------
# Main Cannotate GUI
# -------------------------------------
def Cannotate():
    """
    Main GUI for selecting a date & WAV file in the bird folder.
    Adds global song numbering across all days (Song #1..N).
    Annotation counts per-date and overall are displayed.
    """
    root = tk.Tk()
    root.geometry("1000x500")
    root.withdraw()
    messagebox.showinfo("Directions", "Please navigate to your bird folder.")

    bird_folder = filedialog.askdirectory(title="Select Bird Folder")
    if not bird_folder:
        messagebox.showinfo("Cancelled", "No folder selected.")
        root.destroy()
        return

    bird_name = os.path.basename(bird_folder)
    pattern = os.path.join(bird_folder, f"{bird_name}_annotation*.json")
    annotation_files = glob.glob(pattern)

    if not annotation_files:
        user_name = simpledialog.askstring(
            "New Annotation File",
            "No annotation file found.\nEnter your user name to create a new annotation file:",
            parent=root
        )
        if not user_name:
            messagebox.showerror("Error", "User name required.")
            root.destroy()
            return
        annotation_file = os.path.join(bird_folder, f"{bird_name}_annotation_{user_name}.json")
        db_data = {}
        with open(annotation_file, "w") as f:
            json.dump(db_data, f, indent=2)
    else:
        choice = messagebox.askyesno(
            "Annotation File Choice",
            "Annotation file(s) found.\nDo you want to choose an existing annotation file?\n"
            "(Yes = choose existing, No = create new)",
            parent=root
        )
        if choice:
            annotation_file = filedialog.askopenfilename(
                initialdir=bird_folder,
                title="Select Annotation File",
                filetypes=[("JSON Files", "*.json")],
                parent=root
            )
            if not annotation_file:
                messagebox.showinfo("Error", "No annotation file selected. Exiting.")
                root.destroy()
                return
            with open(annotation_file, "r") as f:
                db_data = json.load(f)
            # One-time migration: give every file entry a human_verified flag.
            n_added = ensure_human_verified_field(db_data)
            if n_added:
                with open(annotation_file, "w") as f:
                    json.dump(db_data, f, indent=2)
                print(f"Added '{HUMAN_VERIFIED_KEY}' to {n_added} file entry/entries "
                      f"in {os.path.basename(annotation_file)}")
        else:
            user_name = simpledialog.askstring(
                "New Annotation File",
                "Enter your user name to create a new annotation file:",
                parent=root
            )
            if not user_name:
                messagebox.showerror("Error", "User name required.")
                root.destroy()
                return
            annotation_file = os.path.join(bird_folder, f"{bird_name}_annotation_{user_name}.json")
            db_data = {}
            with open(annotation_file, "w") as f:
                json.dump(db_data, f, indent=2)

    # --- find date subfolders ---
    date_pattern = re.compile(r'^\d{4}[-_]\d{2}[-_]\d{2}$')
    date_folders = [
        d for d in os.listdir(bird_folder)
        if os.path.isdir(os.path.join(bird_folder, d)) and date_pattern.match(d)
    ]
    if not date_folders:
        messagebox.showinfo("Error", "No date subfolders found.")
        root.destroy()
        return
    date_folders.sort()

    # --- map date -> wavs ---
    date_to_wavs = {}
    for d in date_folders:
        wav_folder = os.path.join(bird_folder, d, "chop_data", "wav")
        if os.path.isdir(wav_folder):
            wfs = [f for f in os.listdir(wav_folder)
                   if f.lower().endswith(".wav") and not f.startswith("._")]
            if wfs:
                full_paths = [os.path.join(wav_folder, wf) for wf in sorted(wfs)]
                date_to_wavs[d] = full_paths

    # --- build global flattened song list ---
    global_songs = []
    for d in sorted(date_to_wavs.keys()):
        for fp in date_to_wavs[d]:
            global_songs.append((d, fp, os.path.basename(fp)))
    base_names_by_date = {d: [os.path.basename(p) for p in fps] for d, fps in date_to_wavs.items()}

    # --- build global syllable templates ONCE at startup ---
    # Only TEMPLATE_INSTANCES_PER_SYLLABLE randomly-chosen appearances of each
    # syllable are loaded, and only the wav files holding them are read — the
    # dataset as a whole is never loaded into memory, however large it grows.
    # Backed by an on-disk cache next to the annotation file: on unchanged
    # annotations (and source wavs) the templates load instantly with zero wav
    # reads; otherwise they are rebuilt (streaming, one file at a time) and
    # re-cached.
    templates_cache_path = os.path.splitext(annotation_file)[0] + ".templates_cache.pkl"
    global_syllable_templates = IntegratedSpectrogramViewer.build_global_syllable_templates_cached(
        db_data, date_to_wavs, templates_cache_path,
        template_cols=IntegratedSpectrogramViewer.TEMPLATE_COLS,
        max_instances_per_label=TEMPLATE_INSTANCES_PER_SYLLABLE)
    n_tmpl = len(global_syllable_templates)
    if n_tmpl:
        print(f"Built {n_tmpl} global syllable template(s): {sorted(global_syllable_templates.keys())}")
    else:
        print("No annotated syllables found yet — templates will be empty until annotations are made.")

    def global_index_of(date_str, base_name):
        for i, (d, _, bn) in enumerate(global_songs):
            if d == date_str and bn == base_name:
                return i
        return -1

    # --- verification-aware display names -------------------------------
    def entry_for(date_str, base_name):
        """Return the annotation entry for a song, or None if never saved."""
        for entry in db_data.get(date_str, []):
            if entry.get("file_name") == base_name:
                return entry
        return None

    def display_name(date_str, base_name):
        """Prefix a file name with its human-verification marker."""
        entry = entry_for(date_str, base_name)
        verified = entry is not None and is_human_verified(entry)
        return (VERIFIED_PREFIX if verified else UNVERIFIED_PREFIX) + base_name

    def display_names_for(date_str):
        # Collect the verified names once instead of scanning the day's entries
        # per file, so refreshing stays cheap on dates with many recordings.
        verified = {e.get("file_name") for e in db_data.get(date_str, [])
                    if is_human_verified(e)}
        return [(VERIFIED_PREFIX if bn in verified else UNVERIFIED_PREFIX) + bn
                for bn in base_names_by_date.get(date_str, [])]

    def current_file_base():
        """The real WAV base name behind the current combobox selection."""
        return _base_from_display(selected_file.get())

    def refresh_file_list_display():
        """Recompute the combobox markers, keeping the current selection."""
        date_str = selected_date.get()
        base = current_file_base()
        updating_selection["on"] = True
        try:
            file_combo['values'] = display_names_for(date_str)
            if base:
                selected_file.set(display_name(date_str, base))
        finally:
            updating_selection["on"] = False

    def select_by_global_index(idx):
        if not global_songs:
            selected_date.set("")
            selected_file.set("")
            selected_index.set("")
            return
        idx = max(0, min(idx, len(global_songs)-1))
        d, _, bn = global_songs[idx]
        updating_selection["on"] = True
        try:
            selected_date.set(d)
            file_combo['values'] = display_names_for(d)
            selected_file.set(display_name(d, bn))
            selected_index.set(str(idx + 1))
        finally:
            updating_selection["on"] = False

    root.deiconify()
    root.title(f"Annotations for {bird_name}")

    tk.Label(root, text="Select Date:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
    selected_date = tk.StringVar(value=list(date_to_wavs.keys())[0] if date_to_wavs else "")
    date_combo = ttk.Combobox(root, textvariable=selected_date,
                              values=list(date_to_wavs.keys()), state="readonly", width=18)
    date_combo.grid(row=0, column=1, padx=5, pady=5, sticky="w")

    tk.Label(root, text="Select WAV File:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
    selected_file = tk.StringVar()
    file_combo = ttk.Combobox(root, textvariable=selected_file, state="readonly", width=100)
    file_combo.grid(row=1, column=1, padx=5, pady=5, sticky="we")

    tk.Label(root, text="Select Song #:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
    selected_index = tk.StringVar()
    index_combo = ttk.Combobox(root, textvariable=selected_index,
                               values=[str(i) for i in range(1, len(global_songs)+1)],
                               state="readonly", width=10)
    index_combo.grid(row=2, column=1, padx=5, pady=5, sticky="w")

    lbl_counts = tk.Label(
        root,
        text="Annotated: 0 / 0 | Verified: 0 / 0 | Total annotated: 0 / 0 | Total verified: 0 / 0")
    lbl_counts.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky="w")

    # small panel showing used syllable types (label and count)
    types_frame_main = tk.Frame(root)
    types_frame_main.grid(row=3, column=2, padx=6, pady=5, sticky="ne")
    tk.Label(types_frame_main, text="Used syllable types:").pack(anchor="nw")
    types_lb_main = tk.Listbox(types_frame_main, width=16, height=6)
    types_lb_main.pack(anchor="ne")

    root.grid_columnconfigure(1, weight=1)
    updating_selection = {"on": False}

    # --- counts ---
    def refresh_counts():
        # If the main window was destroyed, skip updating counts.
        try:
            if not root.winfo_exists():
                return
        except Exception:
            return
        date = selected_date.get()
        files = date_to_wavs.get(date, [])
        n_files = len(files)
        annotated_for_date = 0
        verified_for_date = 0
        if date in db_data:
            annotated_for_date = sum(1 for e in db_data[date] if e.get("segments", []))
            verified_for_date = sum(1 for e in db_data[date] if is_human_verified(e))
        total_files = sum(len(v) for v in date_to_wavs.values())
        total_annotated = sum(
            sum(1 for e in db_data.get(d, []) if e.get("segments", []))
            for d in date_to_wavs
        )
        total_verified = sum(
            sum(1 for e in db_data.get(d, []) if is_human_verified(e))
            for d in date_to_wavs
        )
        try:
            lbl_counts.config(
                text=f"Annotated: {annotated_for_date} / {n_files} | "
                     f"Verified: {verified_for_date} / {n_files} | "
                     f"Total annotated: {total_annotated} / {total_files} | "
                     f"Total verified: {total_verified} / {total_files}"
            )
        except Exception:
            # If label cannot be configured (e.g., widget destroyed), ignore silently.
            return
        # update main used-types list (if present)
        try:
            type_counts = {}
            for d, entries in db_data.items():
                for e in entries:
                    for s in e.get("segments", []):
                        try:
                            lab = s[2] if len(s) >= 3 else None
                        except Exception:
                            continue
                        try:
                            labv = int(lab)
                        except Exception:
                            labv = lab
                        type_counts[labv] = type_counts.get(labv, 0) + 1
            try:
                types_lb_main.delete(0, tk.END)
                for labv, cnt in sorted(type_counts.items(), key=lambda x: (str(x[0]))):
                    types_lb_main.insert(tk.END, f"{labv} ({cnt})")
            except Exception:
                pass
        except Exception:
            pass

    # --- sync logic ---
    def update_file_list(*args):
        date = selected_date.get()
        base_names = base_names_by_date.get(date, [])
        file_combo['values'] = display_names_for(date)

        default_base = base_names[-1] if base_names else ""
        if date in db_data and db_data[date]:
            last_file = db_data[date][-1]["file_name"]
            if last_file in base_names:
                default_base = last_file

        updating_selection["on"] = True
        try:
            if default_base:
                selected_file.set(display_name(date, default_base))
                gi = global_index_of(date, default_base)
                selected_index.set(str(gi + 1) if gi >= 0 else "")
            else:
                selected_file.set("")
                selected_index.set("")
        finally:
            updating_selection["on"] = False

        refresh_counts()

    def on_file_combo_change(event=None):
        if updating_selection["on"]:
            return
        date = selected_date.get()
        cur = current_file_base()
        gi = global_index_of(date, cur)
        if gi >= 0:
            updating_selection["on"] = True
            try:
                selected_index.set(str(gi + 1))
            finally:
                updating_selection["on"] = False

    def on_index_combo_change(event=None):
        if updating_selection["on"]:
            return
        try:
            idx1 = int(selected_index.get())
        except Exception:
            return
        select_by_global_index(idx1 - 1)

    date_combo.bind("<<ComboboxSelected>>", update_file_list)
    file_combo.bind("<<ComboboxSelected>>", on_file_combo_change)
    index_combo.bind("<<ComboboxSelected>>", on_index_combo_change)

    # initialize
    update_file_list()
    select_by_global_index(0)

    def open_annotation():
        nonlocal db_data
        date = selected_date.get()
        file_base = current_file_base()
        full_paths = date_to_wavs.get(date, [])
        full_path = next((fp for fp in full_paths if os.path.basename(fp) == file_base), None)
        if full_path is None:
            messagebox.showerror("Error", "Could not locate full path for the selected file.")
            return

        try:
            fs, y, noise_audio = read_wav_channels(full_path)
        except Exception as e:
            messagebox.showerror("Error", f"Error reading WAV file:\n{e}")
            return

        annotation_entry = None
        if date in db_data:
            for i, entry in enumerate(db_data[date]):
                if entry["file_name"] == file_base:
                    annotation_entry = (i, entry)
                    break

        total_samples = len(y)
        if annotation_entry is None:
            seg_vec, ann_vec = None, None
            existing_segments = []
        else:
            existing_segs = annotation_entry[1]["segments"]
            seg_vec, ann_vec = segments_to_binary(existing_segs, total_samples, fs)
            existing_segments = existing_segs[:]

        initial_verified = (annotation_entry is not None
                            and is_human_verified(annotation_entry[1]))

        def on_verified_change(value):
            """Persist the human-verification flag as soon as it is toggled."""
            entries = db_data.setdefault(date, [])
            entry = next((e for e in entries
                          if e.get("file_name") == file_base), None)
            if entry is None:
                # A song can be marked verified before it is ever annotated:
                # "a human looked at this and there is nothing here".
                entry = {"file_name": file_base,
                         HUMAN_VERIFIED_KEY: False,
                         "segments": []}
                entries.append(entry)
            entry[HUMAN_VERIFIED_KEY] = bool(value)
            try:
                with open(annotation_file, "w") as f:
                    json.dump(db_data, f, indent=2)
            except Exception as e:
                messagebox.showerror(
                    "Error", f"Could not save verification status:\n{e}")
                return
            refresh_counts()
            refresh_file_list_display()

        viewer_win = tk.Toplevel(root)
        viewer = IntegratedSpectrogramViewer(viewer_win, bird_folder, annotation_callback=None,
                                               db_data=db_data, date_to_wavs=date_to_wavs,
                                               global_syllable_templates=global_syllable_templates,
                                               verified_callback=on_verified_change)
        viewer.load_existing_data(y, fs, file_base, seg_vec, ann_vec,
                                  existing_segments=existing_segments,
                                  noise_audio=noise_audio)
        viewer.set_human_verified(initial_verified)
        palette = viewer.palette

        def on_navigator_select(t0, t1):
            viewer.current_window_start = t0
            viewer.window_duration = (t1 - t0)
            viewer.update_plots()

        navigator = FullEntropyNavigator(
            viewer_win, y, fs, on_select=on_navigator_select,
            initial_segments=existing_segments, palette=palette,
            nfft=viewer.nfft, percentile_low=viewer.percentile_low,
            percentile_high=viewer.percentile_high
        )
        viewer.annotation_callback = navigator.update_annotations
        navigator.update_annotations(existing_segments)
        viewer_win.wait_window()
        try:
            navigator.win.destroy()
        except Exception:
            pass

        res_file, _, _, res_seg_vec, res_ann_vec = viewer.get_results()
        if res_seg_vec is None or res_ann_vec is None:
            messagebox.showerror("Error", "No annotation was made.")
            return
        # Use viewer.segments directly so AI flags (4th element) are preserved.
        segs = viewer.segments if viewer.segments else binary_to_segments(res_seg_vec, res_ann_vec, fs)
        # Look the entry up again rather than reusing the index captured before
        # the viewer opened: toggling "human verified" may have created it.
        # The viewer holds the authoritative flag, since that is where the user
        # last set it.
        entries = db_data.setdefault(date, [])
        idx = next((i for i, e in enumerate(entries)
                    if e.get("file_name") == res_file), None)
        new_entry = {
            "file_name": res_file,
            HUMAN_VERIFIED_KEY: bool(viewer.human_verified),
            "segments": clean_segments(segs),
        }
        if idx is None:
            entries.append(new_entry)
        else:
            entries[idx] = new_entry
        with open(annotation_file, "w") as f:
            json.dump(db_data, f, indent=2)
        messagebox.showinfo("Saved", f"Annotations saved for {file_base}")
        refresh_counts()
        refresh_file_list_display()

    # ----- Merge label classes -----
    def _open_merge_labels():
        counts = count_label_classes(db_data)
        if not counts:
            messagebox.showinfo("Merge Labels", "No labels found in the annotation file yet.")
            return

        win = tk.Toplevel(root)
        win.title("Merge Label Classes")
        win.geometry("360x420")

        tk.Label(
            win,
            text="Select one or more SOURCE labels to merge,\nthen enter the TARGET label they become.",
            justify="left"
        ).pack(anchor="w", padx=8, pady=(8, 4))

        tk.Label(win, text="Source labels (multi-select):").pack(anchor="w", padx=8)
        list_frame = tk.Frame(win)
        list_frame.pack(fill="both", expand=True, padx=8)
        scrollbar = tk.Scrollbar(list_frame, orient="vertical")
        src_lb = tk.Listbox(list_frame, selectmode="extended", yscrollcommand=scrollbar.set)
        scrollbar.config(command=src_lb.yview)
        scrollbar.pack(side="right", fill="y")
        src_lb.pack(side="left", fill="both", expand=True)

        sorted_labels = sorted(counts.keys(), key=lambda x: str(x))
        for lab in sorted_labels:
            src_lb.insert(tk.END, f"{lab}  ({counts[lab]} segments)")

        target_frame = tk.Frame(win)
        target_frame.pack(fill="x", padx=8, pady=6)
        tk.Label(target_frame, text="Target label:").pack(side="left")
        target_var = tk.StringVar()
        tk.Entry(target_frame, textvariable=target_var, width=10).pack(side="left", padx=6)

        def apply_merge():
            sel = src_lb.curselection()
            if not sel:
                messagebox.showwarning("Merge Labels", "Select at least one source label.", parent=win)
                return
            target_raw = target_var.get().strip()
            if not target_raw:
                messagebox.showwarning("Merge Labels", "Enter a target label.", parent=win)
                return
            try:
                target = int(target_raw)
            except ValueError:
                target = target_raw
            source_labels = [sorted_labels[i] for i in sel]

            if not messagebox.askyesno(
                "Confirm Merge",
                f"Merge label(s) {source_labels} into {target!r}?\n"
                "This rewrites the annotation file and cannot be undone.",
                parent=win
            ):
                return

            _, n_changed = merge_label_classes(db_data, source_labels, target)
            with open(annotation_file, "w") as f:
                json.dump(db_data, f, indent=2)

            # Rebuild templates so the class (spectrogram) and audio templates
            # reflect the merged labels: the target's medoid is recomputed over
            # all merged segments, and orphaned source-label templates are
            # dropped. Mutate the shared dict in place so viewers keep working.
            new_templates = IntegratedSpectrogramViewer.build_global_syllable_templates_cached(
                db_data, date_to_wavs, templates_cache_path,
                template_cols=IntegratedSpectrogramViewer.TEMPLATE_COLS,
                max_instances_per_label=TEMPLATE_INSTANCES_PER_SYLLABLE)
            global_syllable_templates.clear()
            global_syllable_templates.update(new_templates)
            template_dir = os.path.join(bird_folder, "syllable_templates")
            regenerate_disk_templates(global_syllable_templates, template_dir)

            messagebox.showinfo(
                "Merge Labels",
                f"Rewrote {n_changed} segment(s) to label {target!r}.\n"
                f"Templates rebuilt: {sorted(global_syllable_templates.keys())}",
                parent=win
            )
            refresh_counts()
            win.destroy()

        tk.Button(win, text="Apply Merge", command=apply_merge, fg="black").pack(pady=8)

    # ----- Remove a file's entry from the annotation JSON -----
    def _remove_file_entry():
        """Drop the selected file's entry from the annotation JSON.

        Only the JSON entry is removed — the WAV file on disk is never touched,
        so the song stays in the file list and can be annotated again.
        """
        date = selected_date.get()
        file_base = current_file_base()
        if not file_base:
            messagebox.showinfo("Remove Entry", "No file is selected.")
            return

        entries = db_data.get(date, [])
        idx = next((i for i, e in enumerate(entries)
                    if e.get("file_name") == file_base), None)
        if idx is None:
            messagebox.showinfo(
                "Remove Entry",
                f"{file_base} has no entry in the annotation file — "
                "there is nothing to remove.")
            return

        entry = entries[idx]
        n_segments = len(entry.get("segments", []))
        verified_note = ("\nIts 'human verified' mark will be lost."
                         if is_human_verified(entry) else "")
        if not messagebox.askyesno(
            "Remove Entry",
            f"Remove the annotation entry for:\n{file_base}\n\n"
            f"This deletes {n_segments} segment(s) from "
            f"{os.path.basename(annotation_file)}.{verified_note}\n\n"
            "The WAV file itself is NOT deleted.\n"
            "This cannot be undone."
        ):
            return

        entries.pop(idx)
        try:
            with open(annotation_file, "w") as f:
                json.dump(db_data, f, indent=2)
        except Exception as e:
            # Put the entry back so memory and disk stay in agreement.
            entries.insert(idx, entry)
            messagebox.showerror("Error", f"Could not save annotations:\n{e}")
            return

        refresh_counts()
        refresh_file_list_display()
        messagebox.showinfo(
            "Remove Entry",
            f"Removed the entry for {file_base}.\n"
            "Syllable templates still include its segments until they are "
            "rebuilt (Recalc Templates, or the next launch).")

    # ----- Template browser (delegated to template_browser module) -----
    def _open_template_browser():
        open_template_browser(
            root, db_data, date_to_wavs, global_songs,
            annotation_file, global_index_of, refresh_counts)

    tk.Button(root, text="Templates (from annotations)", command=_open_template_browser, fg="black").grid(
        row=5, column=0, columnspan=2, pady=4
    )

    tk.Button(root, text="Merge Label Classes", command=_open_merge_labels, fg="black").grid(
        row=6, column=0, columnspan=2, pady=4
    )

    tk.Button(root, text="Remove Entry from Annotation File",
              command=_remove_file_entry, fg="#B00020").grid(
        row=7, column=0, columnspan=2, pady=4
    )

    tk.Button(root, text="Open Annotation", command=open_annotation, fg="black").grid(
        row=4, column=0, columnspan=2, pady=10
    )
    root.mainloop()

if __name__ == "__main__":
    Cannotate()