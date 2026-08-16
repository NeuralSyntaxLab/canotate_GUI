"""
Full‐song entropy navigator window.
"""
import tkinter as tk

import numpy as np
from scipy.signal import spectrogram, windows
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle

from colors import distinguishable_colors


class FullEntropyNavigator:
    """
    Pop‐up window that displays the entire track's DPSS energy curve.
    - initial_segments: list of (t_start, t_end, label) already annotated
    - palette: same list of colors used by the spectrogram‐viewer
    - on_select: callback(t0, t1) whenever the user drags & releases a new region
    """

    def __init__(self, parent, audio, fs, on_select,
                 initial_segments=None, palette=None,
                 nfft=512, percentile_low=5, percentile_high=97):
        self.parent = parent
        self.audio = audio
        self.fs = fs
        self.on_select = on_select
        self.nfft = nfft
        self.percentile_low = percentile_low
        self.percentile_high = percentile_high

        # (1) Compute full‐file DPSS spectrogram → full_ent / full_T
        noverlap = self.nfft - int(0.001 * fs)
        NW = 4
        E = windows.dpss(self.nfft, NW, Kmax=2)
        window1, window2 = E[0], E[1]

        F_d, T_d, S1_d = spectrogram(
            audio, fs,
            window=window1,
            noverlap=noverlap,
            nfft=self.nfft,
            nperseg=self.nfft,
            mode='complex'
        )
        _, _, S2_d = spectrogram(
            audio, fs,
            window=window2,
            noverlap=noverlap,
            nfft=self.nfft,
            nperseg=self.nfft,
            mode='complex'
        )

        S_power_d = np.abs(S1_d)**2 + np.abs(S2_d)**2
        S_raw_d = np.log1p(S_power_d)

        vmin_d, vmax_d = np.percentile(S_raw_d, [self.percentile_low, self.percentile_high])
        S_norm_d = np.clip(S_raw_d, vmin_d, vmax_d)
        if vmax_d > vmin_d:
            S_norm_d = (S_norm_d - vmin_d) / (vmax_d - vmin_d)
        else:
            S_norm_d = np.zeros_like(S_raw_d)

        freq_mask_d = F_d > 1000
        S_crop_high_d = S_norm_d[freq_mask_d, :]
        full_energy = np.sum(S_crop_high_d, axis=0)

        # Smooth and normalize to [0,1]
        gwin = windows.gaussian(M=3, std=20/6)
        full_energy_sm = np.convolve(full_energy, gwin, mode='same')
        full_energy_sm = full_energy_sm / np.sum(np.abs(full_energy_sm))
        full_energy_sm = (full_energy_sm - full_energy_sm.min()) / (
                            full_energy_sm.max() - full_energy_sm.min() + 1e-8
                         )

        self.full_T = T_d
        self.full_ent = full_energy_sm

        # (2) Build a Toplevel + Figure for the navigator
        self.win = tk.Toplevel(parent)
        self.win.title("Full‐Song Entropy Navigator")
        self.win.update_idletasks()
        parent.update_idletasks()

        try:
            parent_x = parent.winfo_rootx()
            parent_y = parent.winfo_rooty()
            parent_w = parent.winfo_width()
            parent_h = parent.winfo_height()
            screen_h = parent.winfo_screenheight()
            window_height = 300
            menu_bar_clearance = 30  # avoid macOS/Linux menu bar

            above_y = parent_y - window_height
            below_y = parent_y + parent_h + 5

            if above_y >= menu_bar_clearance:
                top_y = above_y
            elif below_y + window_height <= screen_h:
                top_y = below_y
            else:
                top_y = menu_bar_clearance

            self.win.geometry(f"{parent_w}x{window_height}+{parent_x}+{top_y}")
        except Exception as e:
            print("Warning: could not align entropy navigator:", e)

        # Create the Figure with a dark‐gray facecolor
        self.fig = plt.Figure(figsize=(8, 3), dpi=100, facecolor="#222222")
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("#222222")

        # Plot the entropy curve in white
        self.ax.plot(self.full_T, self.full_ent, color="white", linewidth=1.5)
        self.ax.set_xlabel("Time (s)", color="white")
        self.ax.set_ylabel("Normalized DPSS-Energy", color="white")
        self.ax.set_title("Drag to select a time window", color="white")

        for label in self.ax.get_xticklabels() + self.ax.get_yticklabels():
            label.set_color("white")

        # Add an initial "view rectangle"
        init_start = self.full_T[0]
        init_end = min(init_start + 1.0, self.full_T[-1])
        self.view_rect = Rectangle((init_start, 0.0),
                                   init_end - init_start,
                                   1.0,
                                   edgecolor="lime",
                                   facecolor="none",
                                   linewidth=2)
        self.ax.add_patch(self.view_rect)

        # (3) Prepare container for "annotation rectangles"
        self.annotation_patches = []
        self.palette = palette if palette is not None else distinguishable_colors(1000)

        # (4) Create and pack the FigureCanvasTkAgg
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.win)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Draw any existing annotations
        if initial_segments:
            self._draw_annotations(initial_segments)

        # Mouse-event connections
        self.canvas.mpl_connect("button_press_event", self.on_press)
        self.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.canvas.mpl_connect("button_release_event", self.on_release)

        self.press_x = None

    def _draw_annotations(self, segments):
        """Helper: draw colored rectangles for each (t0, t1, label)."""
        for p in self.annotation_patches:
            try:
                p.remove()
            except Exception:
                pass
        self.annotation_patches.clear()

        for seg in segments:
            t0, t1, label = seg[0], seg[1], seg[2]
            is_ai = len(seg) > 3 and seg[3]
            color = self.palette[int(label) - 1] if label is not None else "gray"
            rect = Rectangle((t0, 0.0),
                             t1 - t0,
                             1.0,
                             facecolor=color,
                             alpha=0.45 if is_ai else 0.8,
                             edgecolor="white" if is_ai else None,
                             linewidth=0.5 if is_ai else 0)
            self.ax.add_patch(rect)
            self.annotation_patches.append(rect)

        self.canvas.draw()

    def update_annotations(self, segments):
        """
        Call this whenever the set of annotated segments changes.
        `segments` is a list of (t_start, t_end, label).
        """
        self._draw_annotations(segments)

    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        self.press_x = event.xdata
        self.view_rect.set_x(self.press_x)
        self.view_rect.set_width(0)
        self.canvas.draw()

    def on_motion(self, event):
        if self.press_x is None or event.inaxes != self.ax:
            return
        x0 = self.press_x
        x1 = event.xdata
        if x1 is None:
            return
        start, end = sorted([x0, x1])
        self.view_rect.set_x(start)
        self.view_rect.set_width(end - start)
        self.canvas.draw()

    def on_release(self, event):
        if self.press_x is None or event.inaxes != self.ax:
            return
        x0 = self.press_x
        x1 = event.xdata
        if x1 is None:
            return
        t_start, t_end = sorted([x0, x1])
        t_start = max(self.full_T[0], min(t_start, self.full_T[-1]))
        t_end = max(self.full_T[0], min(t_end, self.full_T[-1]))
        if t_end <= t_start:
            t_end = t_start + 0.1
            if t_end > self.full_T[-1]:
                t_start = self.full_T[-1] - 0.1
                t_end = self.full_T[-1]
        self.view_rect.set_x(t_start)
        self.view_rect.set_width(t_end - t_start)
        self.canvas.draw()

        self.on_select(t_start, t_end)
        self.press_x = None
