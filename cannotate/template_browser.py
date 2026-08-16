"""
Template browser window — browse syllable instances from annotation data
and build random / run-based concatenations.
"""
import json
import os
import re
import random

import tkinter as tk
from tkinter import messagebox, simpledialog
import numpy as np
from scipy.signal import spectrogram, windows
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from colors import distinguishable_colors
from .cannotate_utils import read_wav


# ── Helpers ──────────────────────────────────────────────────────────

def _parse_label(lab):
    """Robustly convert a label to int (or float, or leave as-is)."""
    if lab is None:
        return lab
    if isinstance(lab, int):
        return lab
    if isinstance(lab, float):
        if abs(lab - round(lab)) < 1e-8:
            return int(round(lab))
        return lab
    if isinstance(lab, str):
        s = lab.strip()
        try:
            return int(s)
        except Exception:
            try:
                fv = float(s)
                if abs(fv - round(fv)) < 1e-8:
                    return int(round(fv))
                return fv
            except Exception:
                m = re.search(r"[-+]?\d+(?:\.\d+)?", s)
                if m:
                    tok = m.group(0)
                    try:
                        fv = float(tok)
                        if abs(fv - round(fv)) < 1e-8:
                            return int(round(fv))
                        return fv
                    except Exception:
                        return s
                return s
    return lab


def _compute_spectrogram_norm(y, fs, nfft=512, percentile_low=5,
                              percentile_high=96, freq_cutoff=0):
    """Compute DPSS two-taper spectrogram, percentile clip / normalise."""
    noverlap = nfft - int(0.001 * fs)
    NW = 4
    E = windows.dpss(nfft, NW, Kmax=2)
    window1, window2 = E[0], E[1]
    F, T, S1 = spectrogram(y, fs, window=window1, noverlap=noverlap,
                           nfft=nfft, nperseg=nfft, mode='complex')
    _, _, S2  = spectrogram(y, fs, window=window2, noverlap=noverlap,
                            nfft=nfft, nperseg=nfft, mode='complex')
    S_power = np.abs(S1)**2 + np.abs(S2)**2
    S_raw = np.log1p(S_power)
    vmin, vmax = np.percentile(S_raw, [percentile_low, percentile_high])
    S_norm = np.clip(S_raw, vmin, vmax)
    if vmax > vmin:
        S_norm = (S_norm - vmin) / (vmax - vmin)
    else:
        S_norm = np.zeros_like(S_norm)
    if freq_cutoff > 0:
        mask = F >= freq_cutoff
        F = F[mask]
        S_norm = S_norm[mask, :]
    return S_norm, F, T


# ── Public entry-point ───────────────────────────────────────────────

def open_template_browser(root, db_data, date_to_wavs, global_songs,
                          annotation_file, global_index_of, refresh_counts):
    """Browse syllable instances from *db_data* and build concatenations.

    Parameters
    ----------
    root : tk.Tk or tk.Toplevel
        Parent window.
    db_data : dict
        ``{date_str: [{file_name, segments}, ...]}``.
    date_to_wavs : dict
        ``{date_str: [full_path, ...]}``.
    global_songs : list[tuple]
        Flattened ``[(date, full_path, base_name), ...]``.
    annotation_file : str
        Path to the annotation JSON (for saving label changes).
    global_index_of : callable
        ``global_index_of(date_str, base_name) -> int`` index into
        *global_songs*, or -1 if not found.
    refresh_counts : callable
        Called after label changes so the main window can update counts.
    """

    missing_song_map = {}
    missing_next = len(global_songs)

    def gather_instances(syll_num):
        nonlocal missing_next
        instances = []
        for d, entries in db_data.items():
            for entry in entries:
                fn = entry.get("file_name")
                for seg in entry.get("segments", []):
                    try:
                        s0, s1, lab = seg
                    except Exception:
                        continue
                    labv = _parse_label(lab)
                    if labv == syll_num:
                        gi = global_index_of(d, fn)
                        if gi < 0:
                            key = (d, fn)
                            if key in missing_song_map:
                                gi = missing_song_map[key]
                            else:
                                gi = missing_next
                                missing_song_map[key] = gi
                                missing_next += 1
                        instances.append({
                            "date": d, "file_name": fn,
                            "t0": float(s0), "t1": float(s1),
                            "song_index": gi,
                        })
        return instances

    def load_segment(inst):
        date = inst["date"]
        fn = inst["file_name"]
        paths = date_to_wavs.get(date, [])
        fullp = next((p for p in paths if os.path.basename(p) == fn), None)
        if fullp is None:
            return None, None
        try:
            fs, y = read_wav(fullp)
        except Exception:
            return None, None
        if y.ndim == 2:
            y = y.mean(axis=1)
        i0 = max(0, int(round(inst["t0"] * fs)))
        i1 = min(len(y), int(round(inst["t1"] * fs)))
        return y[i0:i1], fs

    # --- initial syllable type ---
    type_counts = {}
    for d, entries in db_data.items():
        for e in entries:
            for s in e.get("segments", []):
                try:
                    lab = s[2]
                except Exception:
                    continue
                labv = _parse_label(lab)
                type_counts[labv] = type_counts.get(labv, 0) + 1
    if not type_counts:
        messagebox.showinfo("No instances",
                            "No annotated syllable types found in the annotation file.")
        return

    try:
        syll_num = sorted(type_counts.keys(), key=lambda x: str(x))[0]
    except Exception:
        syll_num = list(type_counts.keys())[0]
    instances = gather_instances(syll_num)

    # Group by song index
    songs_map = {}
    for inst in instances:
        si = inst["song_index"]
        songs_map.setdefault(si, []).append(inst)
    song_indices = sorted(songs_map.keys())
    filtered_song_indices = song_indices.copy()

    top = tk.Toplevel(root)
    top.title(f"Syllable {syll_num} — Template Builder")
    top.geometry("900x600")

    try:
        local_palette = distinguishable_colors(1000)
    except Exception:
        local_palette = None

    # ── Left panel ─────────────────────────────────────────
    left = tk.Frame(top)
    left.pack(side=tk.LEFT, fill="y", padx=6, pady=6)
    tk.Label(left, text="Songs containing syllable (select one or multiple):").pack(anchor="w")

    # Used-types list
    types_frame = tk.Frame(left)
    types_frame.pack(fill="x", pady=(6, 4))
    tk.Label(types_frame, text="Used syllable types:").pack(anchor="w")
    types_lb = tk.Listbox(types_frame, selectmode="browse", width=20, height=6)
    types_lb.pack(fill="x", expand=False)
    def refresh_types_list():
        types_lb.delete(0, tk.END)
        tc2 = {}
        for d2, entries2 in db_data.items():
            for e2 in entries2:
                for s2 in e2.get("segments", []):
                    try:
                        _, _, lab2 = s2
                    except Exception:
                        continue
                    labv2 = _parse_label(lab2)
                    tc2[labv2] = tc2.get(labv2, 0) + 1
        for labv2, cnt in sorted(tc2.items(), key=lambda x: str(x[0])):
            types_lb.insert(tk.END, f"{labv2} ({cnt})")

    try:
        refresh_types_list()
    except Exception:
        pass

    lb = tk.Listbox(left, selectmode="extended", width=40, height=20)
    lb.pack(fill="y", expand=True)

    def on_type_select(event=None):
        sel = types_lb.curselection()
        if not sel:
            return
        txt = types_lb.get(sel[0])
        try:
            new_lab = int(txt.split()[0])
        except Exception:
            new_lab = txt.split()[0]
        change_syllable(new_lab)

    types_lb.bind('<<ListboxSelect>>', on_type_select)

    # ── Run detection ──────────────────────────────────────

    def find_runs_in_song(syll_num_local, song_idx_local, kreq=None):
        """Return list of runs for the given syllable in the song."""
        runs = []
        if song_idx_local not in songs_map:
            return runs
        insts = songs_map[song_idx_local]
        if not insts:
            return runs
        date = insts[0]["date"]
        fn = insts[0]["file_name"]
        ann_entry = None
        for e in db_data.get(date, []):
            if e.get("file_name") == fn:
                ann_entry = e
                break
        if ann_entry is None:
            return runs
        segs = ann_entry.get("segments", [])
        try:
            segs_sorted = sorted(
                [s for s in segs if s and len(s) >= 3],
                key=lambda x: float(x[0]),
            )
        except Exception:
            segs_sorted = [s for s in segs if s and len(s) >= 3]

        cur_run = []
        for s in segs_sorted:
            try:
                s0, s1, lab = s
            except Exception:
                continue
            labv = _parse_label(lab)
            if labv == syll_num_local:
                try:
                    cur_run.append({
                        "date": date, "file_name": fn,
                        "t0": float(s0), "t1": float(s1),
                        "song_index": song_idx_local,
                    })
                except Exception:
                    cur_run.append({
                        "date": date, "file_name": fn,
                        "t0": float(s0) if s0 is not None else 0.0,
                        "t1": float(s1) if s1 is not None else 0.0,
                        "song_index": song_idx_local,
                    })
            else:
                if cur_run:
                    runs.append(cur_run)
                    cur_run = []
        if cur_run:
            runs.append(cur_run)
        if kreq is not None:
            runs = [r for r in runs if len(r) >= kreq]
        return runs

    # ── Song list ──────────────────────────────────────────

    def update_song_list():
        nonlocal filtered_song_indices
        lb.delete(0, tk.END)
        filtered = []
        for si in song_indices:
            insts_si = songs_map.get(si, [])
            if not insts_si:
                continue
            runs = find_runs_in_song(syll_num, si)
            num_runs = len(runs)
            max_run = max((len(r) for r in runs), default=0)
            display_max = min(max_run, 20)
            d = insts_si[0]["date"]
            fn = insts_si[0]["file_name"]
            lb.insert(tk.END,
                      f"Song #{si+1}  {d} / {fn}   runs:{num_runs} max:{display_max}")
            filtered.append(si)
        filtered_song_indices = filtered
        return filtered

    # ── Options row ────────────────────────────────────────

    options = tk.Frame(top)
    options.pack(side=tk.TOP, fill="x", padx=6, pady=6)
    tk.Label(options, text="Concatenate how many syllables (1-20):").grid(row=0, column=0, sticky="w")
    spin_k = tk.Spinbox(options, from_=1, to=1000, width=5)
    spin_k.grid(row=0, column=1, sticky="w", padx=4)
    tk.Label(options, text="Random seed (optional):").grid(row=0, column=2, sticky="w", padx=8)
    seed_entry = tk.Entry(options, width=8)
    seed_entry.grid(row=0, column=3, sticky="w")

    btn_frame = tk.Frame(top)
    btn_frame.pack(side=tk.TOP, fill="x", padx=6, pady=6)
    btn_filter = tk.Button(btn_frame, text="Filter Songs by k", command=update_song_list)
    btn_filter.pack(side=tk.LEFT, padx=6)
    sel_status = tk.Label(btn_frame, text="", fg="gray30")
    sel_status.pack(side=tk.RIGHT, padx=6)

    canvas_frame = tk.Frame(top)
    canvas_frame.pack(side=tk.RIGHT, fill="both", expand=True, padx=6, pady=6)

    fig = plt.Figure(figsize=(6, 4), dpi=100)
    ax = fig.add_subplot(111)
    canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
    canvas.get_tk_widget().pack(fill="both", expand=True)

    concatenated_audio = {"audio": None, "fs": None}
    last_displayed_picks = []
    # Each displayed highlight span, with its pick and toggle state:
    # {"pick": inst, "b0": float, "b1": float, "patch": Polygon, "selected": bool}
    displayed_segments = []

    def _syll_color():
        try:
            return (local_palette[int(syll_num) - 1]
                    if local_palette is not None else 'yellow')
        except Exception:
            return 'yellow'

    def _update_sel_status():
        n_sel = sum(1 for d in displayed_segments if d["selected"])
        n_tot = len(displayed_segments)
        if n_tot == 0:
            sel_status.config(text="")
        elif n_sel == 0:
            sel_status.config(
                text=f"{n_tot} segments shown — click a segment to select it")
        else:
            sel_status.config(
                text=f"{n_sel}/{n_tot} segments selected for relabeling")

    def _render_display(full, fs0, seg_infos, title):
        """Draw spectrogram of *full* with one toggleable span per pick.

        seg_infos : list of (pick, t_start, t_end) in display coordinates.
        """
        nonlocal last_displayed_picks
        displayed_segments.clear()
        last_displayed_picks = [p for p, _, _ in seg_infos]
        try:
            S_norm, F, T = _compute_spectrogram_norm(full, fs0)
            ax.clear()
            ax.imshow(S_norm, aspect='auto', origin='lower',
                      extent=[0, len(full) / fs0, F[0], F[-1]],
                      cmap='gray_r', vmin=0, vmax=1)
            color = _syll_color()
            for p, b0, b1 in seg_infos:
                patch = ax.axvspan(b0, b1, color=color, alpha=0.25)
                displayed_segments.append({
                    "pick": p, "b0": b0, "b1": b1,
                    "patch": patch, "selected": False,
                })
            ax.set_title(title)
            ax.set_xlabel("Time (s)")
            canvas.draw()
        except Exception as e:
            ax.clear()
            ax.text(0.5, 0.5, f"Could not generate spectrogram: {e}", ha='center')
            canvas.draw()
        _update_sel_status()

    def _set_segment_selected(d, selected):
        d["selected"] = selected
        if selected:
            d["patch"].set_alpha(0.45)
            d["patch"].set_edgecolor('red')
            d["patch"].set_linewidth(2)
        else:
            d["patch"].set_alpha(0.25)
            d["patch"].set_edgecolor('none')
            d["patch"].set_linewidth(0)

    def _on_canvas_click(event):
        if event.inaxes is not ax or event.xdata is None:
            return
        x = event.xdata
        for d in displayed_segments:
            if d["b0"] <= x <= d["b1"]:
                _set_segment_selected(d, not d["selected"])
                canvas.draw_idle()
                _update_sel_status()
                return

    canvas.mpl_connect('button_press_event', _on_canvas_click)

    # ── Build & show ───────────────────────────────────────

    def build_and_show():
        try:
            k = int(spin_k.get())
        except Exception:
            k = 1
        sel = lb.curselection()
        pool = []
        if sel:
            try:
                selected_song_indices = [filtered_song_indices[i] for i in sel]
            except Exception:
                selected_song_indices = [song_indices[i] for i in sel]
            for si in selected_song_indices:
                pool.extend(songs_map.get(si, []))
        else:
            pool = instances[:]

        if not pool:
            messagebox.showerror("Error", "No instances available for selected songs.")
            return

        s = seed_entry.get()
        if s != "":
            try:
                random.seed(int(s))
            except Exception:
                pass

        # Auto-determine default k as max contiguous run in selected songs
        try:
            if sel:
                max_run = 0
                for si in selected_song_indices:
                    runs = find_runs_in_song(syll_num, si)
                    for r in runs:
                        if len(r) > max_run:
                            max_run = len(r)
                if max_run > 0:
                    try:
                        spin_k.delete(0, tk.END)
                        spin_k.insert(0, str(max_run))
                    except Exception:
                        pass
                    k = max_run
        except Exception:
            pass

        if not pool:
            messagebox.showerror("Error", "No instances available for selected songs.")
            return
        if len(pool) >= k:
            picks = random.sample(pool, k)
        else:
            picks = random.choices(pool, k=k)

        concat_list = []
        seg_infos = []
        fs0 = None
        gap_dur = 0.02
        cur_pos = 0.0
        for p in picks:
            seg, fs_l = load_segment(p)
            if seg is None:
                continue
            if fs0 is None:
                fs0 = fs_l
            if fs_l != fs0:
                continue
            seg_len = len(seg) / fs0
            seg_infos.append((p, cur_pos, cur_pos + seg_len))
            concat_list.append(seg)
            concat_list.append(np.zeros(int(round(gap_dur * fs0))))
            cur_pos += seg_len + gap_dur

        if not concat_list:
            messagebox.showerror("Error", "Could not load any segments for the chosen instances.")
            return

        full = np.concatenate(concat_list)
        concatenated_audio["audio"] = full
        concatenated_audio["fs"] = fs0

        uniq = sorted({p['song_index'] for p, _, _ in seg_infos})
        _render_display(full, fs0, seg_infos,
                        f"Syllable {syll_num} — concatenation ({k} items) — songs: "
                        + ", ".join(str(u+1) for u in uniq))

    # ── Change label ───────────────────────────────────────

    def change_label_of_displayed():
        if not last_displayed_picks:
            messagebox.showerror("No selection", "No displayed concatenation to relabel.")
            return
        selected_picks = [d["pick"] for d in displayed_segments if d["selected"]]
        if selected_picks:
            targets = selected_picks
            scope = f"the {len(targets)} selected segment(s)"
        else:
            targets = last_displayed_picks
            scope = f"all {len(targets)} displayed segment(s)"
        new_label = simpledialog.askinteger(
            "Change Label",
            f"Enter new label (integer) for {scope}:", parent=top)
        if new_label is None:
            return
        changed = 0
        for p in targets:
            d = p.get("date")
            fn = p.get("file_name")
            t0 = float(p.get("t0"))
            t1 = float(p.get("t1"))
            if d not in db_data:
                continue
            for entry in db_data[d]:
                if entry.get("file_name") != fn:
                    continue
                for seg in entry.get("segments", []):
                    try:
                        ss, se, lab_s = seg
                    except Exception:
                        continue
                    if abs(float(ss) - t0) < 1e-3 and abs(float(se) - t1) < 1e-3:
                        try:
                            seg[2] = int(new_label)
                        except Exception:
                            seg[2] = new_label
                        changed += 1
        try:
            with open(annotation_file, "w") as f:
                json.dump(db_data, f, indent=2)
        except Exception as e:
            messagebox.showerror("Error", f"Could not save annotations: {e}")
            return
        try:
            refresh_counts()
        except Exception:
            pass
        try:
            refresh_types_list()
        except Exception:
            pass
        try:
            change_syllable(syll_num)
        except Exception:
            pass
        messagebox.showinfo("Labels updated", f"Updated {changed} segments to label {new_label}.")

    def clear_segment_selection():
        for d in displayed_segments:
            _set_segment_selected(d, False)
        canvas.draw_idle()
        _update_sel_status()

    try:
        btn_change_label = tk.Button(btn_frame, text="Change Label of Selected/Displayed",
                                     command=change_label_of_displayed)
        btn_change_label.pack(side=tk.LEFT, padx=6)
        btn_clear_sel = tk.Button(btn_frame, text="Clear Selection",
                                  command=clear_segment_selection)
        btn_clear_sel.pack(side=tk.LEFT, padx=6)
    except Exception:
        pass

    # ── Contiguous section loader ─────────────────────────

    def load_contiguous_section(picks, pad=0.5):
        if not picks:
            return None, None
        date = picks[0]["date"]
        fn = picks[0]["file_name"]
        paths = date_to_wavs.get(date, [])
        fullp = next((p for p in paths if os.path.basename(p) == fn), None)
        if fullp is None:
            return None, None
        try:
            fs_c, y_c = read_wav(fullp)
        except Exception:
            return None, None
        if y_c.ndim == 2:
            y_c = y_c.mean(axis=1)
        t0 = picks[0]["t0"]
        t1 = picks[-1]["t1"]
        i0 = max(0, int(round((t0 - pad) * fs_c)))
        i1 = min(len(y_c), int(round((t1 + pad) * fs_c)))
        return y_c[i0:i1], fs_c

    # ── Build from picks ──────────────────────────────────

    def _build_from_picks(picks):
        nonlocal last_displayed_picks
        last_displayed_picks = picks[:]
        try:
            same_file = all(
                p["file_name"] == picks[0]["file_name"]
                and p["date"] == picks[0]["date"]
                for p in picks
            )
        except Exception:
            same_file = False

        if same_file:
            full, fs0 = load_contiguous_section(picks, pad=0.5)
            if full is None:
                messagebox.showerror("Error", "Could not load contiguous section from file.")
                return
            concatenated_audio["audio"] = full
            concatenated_audio["fs"] = fs0
            pad = 0.5
            window_start = picks[0]['t0'] - pad
            seg_infos = [(p, p['t0'] - window_start, p['t1'] - window_start)
                         for p in picks]
            uniq = sorted({p['song_index'] for p in picks})
            _render_display(full, fs0, seg_infos,
                            f"Syllable {syll_num} — natural concatenation — songs: "
                            + ", ".join(str(u+1) for u in uniq))
            return

        # Fallback: concatenate individual segments with small gaps
        concat_list = []
        seg_infos = []
        fs0 = None
        gap_dur = 0.02
        cur_pos = 0.0
        for p in picks:
            seg, fs_l = load_segment(p)
            if seg is None:
                continue
            if fs0 is None:
                fs0 = fs_l
            if fs_l != fs0:
                continue
            seg_len = len(seg) / fs0
            seg_infos.append((p, cur_pos, cur_pos + seg_len))
            concat_list.append(seg)
            concat_list.append(np.zeros(int(round(gap_dur * fs0))))
            cur_pos += seg_len + gap_dur
        if not concat_list:
            messagebox.showerror("Error", "Could not load segments for the chosen run.")
            return

        full = np.concatenate(concat_list)
        concatenated_audio["audio"] = full
        concatenated_audio["fs"] = fs0
        uniq = sorted({p['song_index'] for p, _, _ in seg_infos})
        _render_display(full, fs0, seg_infos,
                        f"Syllable {syll_num} — natural concatenation — songs: "
                        + ", ".join(str(u+1) for u in uniq))

    # ── Initial spin_k value ──────────────────────────────
    try:
        overall_max = 0
        for si in song_indices:
            runs_si = find_runs_in_song(syll_num, si, kreq=None)
            for r in runs_si:
                if len(r) > overall_max:
                    overall_max = len(r)
        init_k = min(20, overall_max) if overall_max > 0 else 20
        try:
            spin_k.delete(0, tk.END)
            spin_k.insert(0, str(init_k))
        except Exception:
            pass
    except Exception:
        try:
            spin_k.delete(0, tk.END)
            spin_k.insert(0, '20')
        except Exception:
            pass

    # ── Selection helpers ─────────────────────────────────

    def _select_and_get_runs(event=None, allow_nearest=True, kreq=None):
        sel = lb.curselection()
        if not sel:
            if event is not None and allow_nearest:
                try:
                    idx = lb.nearest(event.y)
                except Exception:
                    return None, [], [], 0
            else:
                return None, [], [], 0
        else:
            idx = sel[0]
        try:
            song_idx = (filtered_song_indices[idx]
                        if idx < len(filtered_song_indices) else song_indices[idx])
        except Exception:
            try:
                song_idx = song_indices[idx]
            except Exception:
                return None, [], [], 0
        runs_all = find_runs_in_song(syll_num, song_idx, kreq=None)
        runs_k = find_runs_in_song(syll_num, song_idx, kreq=kreq)
        max_run = max((len(r) for r in runs_all), default=0)
        return song_idx, runs_all, runs_k, max_run

    def debug_selected_runs(event=None):
        song_idx, runs_all, runs_k, max_run = _select_and_get_runs(
            event=event, allow_nearest=True, kreq=None)
        if song_idx is None:
            messagebox.showinfo('Debug runs', 'No song selected.')
            return
        if not runs_all:
            messagebox.showinfo('Debug runs',
                                f'No runs found for song index {song_idx+1}.')
            return
        out = []
        for i, r in enumerate(runs_all):
            out.append(f'Run {i+1}: {len(r)} segs, {r[0]["t0"]:.3f}-{r[-1]["t1"]:.3f}')
        txt = '\n'.join(out)
        print('DEBUG runs for song', song_idx + 1)
        print(txt)
        messagebox.showinfo('Debug runs', txt)

    def on_song_double_click(event=None):
        song_idx, runs_all, runs_k, max_run = _select_and_get_runs(
            event=event, allow_nearest=True, kreq=None)
        if song_idx is None:
            return
        try:
            k_req = int(spin_k.get())
        except Exception:
            k_req = None
        runs = runs_k if (k_req is not None and runs_k) else runs_all
        if not runs:
            messagebox.showinfo("No repetition",
                                f"No runs of consecutive syllable {syll_num} "
                                f"found in the selected song.")
            return
        if len(runs_all) > 1:
            choose_top = tk.Toplevel(top)
            choose_top.title("Choose run")
            choose_top.transient(top)
            choose_top.grab_set()
            choose_top.lift()
            tk.Label(choose_top,
                     text=f"Found {len(runs_all)} runs — choose which to use:").pack()
            run_lb = tk.Listbox(choose_top, width=80)
            for i, run in enumerate(runs_all):
                r0 = run[0]
                rN = run[-1]
                run_lb.insert(tk.END,
                              f"Run #{i+1}: {len(run)} segs — "
                              f"{r0['date']}/{r0['file_name']} "
                              f"{r0['t0']:.3f}-{rN['t1']:.3f}")
            run_lb.pack()
            if runs_all:
                run_lb.selection_set(0)
                run_lb.activate(0)

            def choose_run():
                sel2 = run_lb.curselection()
                if not sel2:
                    return
                chosen = runs_all[sel2[0]]
                choose_top.destroy()
                try:
                    ku = int(spin_k.get())
                except Exception:
                    ku = len(chosen)
                ku = min(ku, len(chosen))
                _build_from_picks([seg for seg in chosen[:ku]])
            tk.Button(choose_top, text="Use Selected Run", command=choose_run).pack()
            return
        # single run
        L0 = len(runs[0])
        try:
            ku = int(spin_k.get())
        except Exception:
            ku = L0
        ku = min(ku, L0)
        _build_from_picks(runs[0][:ku])

    lb.bind('<Double-Button-1>', on_song_double_click)

    def on_song_select(event=None):
        sel = lb.curselection()
        if not sel:
            return
        idx = sel[0]
        try:
            song_idx = (filtered_song_indices[idx]
                        if idx < len(filtered_song_indices) else song_indices[idx])
        except Exception:
            song_idx = song_indices[idx]
        runs = find_runs_in_song(syll_num, song_idx)
        max_run = max((len(r) for r in runs), default=0)
        if max_run > 0:
            try:
                spin_k.delete(0, tk.END)
                spin_k.insert(0, str(max_run))
            except Exception:
                pass

    lb.bind('<<ListboxSelect>>', on_song_select)

    # ── Change syllable ───────────────────────────────────

    def change_syllable(new_syll):
        nonlocal syll_num, instances, songs_map, song_indices, filtered_song_indices
        nonlocal last_displayed_picks
        syll_num = new_syll
        displayed_segments.clear()
        last_displayed_picks = []
        try:
            _update_sel_status()
        except Exception:
            pass
        instances = gather_instances(syll_num)
        songs_map = {}
        for inst in instances:
            si = inst["song_index"]
            songs_map.setdefault(si, []).append(inst)
        song_indices = sorted(songs_map.keys())
        filtered_song_indices = song_indices.copy()
        try:
            for i in range(types_lb.size()):
                txt = types_lb.get(i)
                lab_txt = txt.split()[0]
                try:
                    labv = int(lab_txt)
                except Exception:
                    labv = lab_txt
                if labv == syll_num:
                    types_lb.selection_clear(0, tk.END)
                    types_lb.selection_set(i)
                    types_lb.activate(i)
                    break
        except Exception:
            pass
        try:
            update_song_list()
        except Exception:
            pass
        try:
            ax.clear()
            canvas.draw()
        except Exception:
            pass

    # populate list
    try:
        update_song_list()
    except Exception:
        pass
