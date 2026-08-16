"""
IntegratedSpectrogramViewer — the main spectrogram annotation window.

This module contains the full viewer class; DTW helpers are delegated to
``dtw_suggestions`` and utilities to ``cannotate_utils``.
"""
import os
import re
import tempfile

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from tkinter import ttk
import numpy as np
from scipy.io import wavfile
from scipy.signal import spectrogram, windows
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import sounddevice as sd

from colors import distinguishable_colors
from .cannotate_utils import (
    generate_pretty_sonogram,
    read_wav,
    NN_MODEL_ARCHITECTURES, NN_WEIGHTS_DIR,
    ONNX_DEFAULT_WEIGHTS,
)
from . import dtw_suggestions as dtw


# ── NN imports (optional at top-level) ───────────────────────────────
# Manual annotation never touches torch, so a machine without the deep-learning
# stack should still get a fully working GUI. When the import fails we keep the
# NN rows on screen but disabled, so it is obvious the feature exists and why it
# is unavailable, rather than the window silently missing a third of its buttons.
try:
    import torch
    from bird_song_detector.nn_segmentation import InferenceToAnnotationData
except ImportError as _torch_err:
    torch = None
    InferenceToAnnotationData = None
    TORCH_AVAILABLE = False
    TORCH_IMPORT_ERROR = str(_torch_err)
else:
    TORCH_AVAILABLE = True
    TORCH_IMPORT_ERROR = None


class OnnxSegmentationModel:
    """Wraps an ONNX InferenceSession to look like a PyTorch model.

    Compatible with ``InferenceToAnnotationData.run_inference()``, which calls
    ``model.eval()``, ``model.to(device)``, and ``model(tensor)`` and then
    applies ``torch.sigmoid`` on the result.

    The ONNX graph accepts ``[batch, 256]`` float32 and returns raw logits
    ``[batch, 1]`` — identical contract to the PyTorch ConvMLP.
    """

    def __init__(self, onnx_path: str):
        import onnxruntime as ort
        providers = ["CPUExecutionProvider"]
        try:
            available = ort.get_available_providers()
            for p in ("CUDAExecutionProvider", "CoreMLExecutionProvider"):
                if p in available:
                    providers.insert(0, p)
                    break
        except Exception:
            pass
        self._session = ort.InferenceSession(onnx_path, providers=providers)
        self._input_name = self._session.get_inputs()[0].name

    def eval(self):
        return self

    def to(self, device):
        return self  # onnxruntime selects its own execution provider

    # Annotations are quoted: torch may be None on a machine without the
    # deep-learning stack, and a bare torch.Tensor would blow up at class
    # definition time, taking the whole GUI down with it.
    def __call__(self, x: "torch.Tensor") -> "torch.Tensor":
        x_np = x.cpu().numpy()
        outputs = self._session.run(None, {self._input_name: x_np})
        return torch.from_numpy(outputs[0])

MAX_RATIO = 0.50

# Default location of the trained 1D-CNN syllable-classification model.
CLASSIFICATION_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir,
    "classification_model", "conv1d_phrase_model",
)


def _is_unlabeled_label(label):
    """True if a segment label is the 999 unlabeled placeholder."""
    try:
        return int(label) == 999
    except (TypeError, ValueError):
        return str(label) == "999"


def _write_png_template(annot_id, tmpl, template_dir, palette):
    """Write ``syllable_{annot_id}.png`` from a template's medoid spectrogram.

    Overwrites any existing file so the on-disk class template stays in sync
    with the (possibly rebuilt) ``tmpl`` dict.
    """
    spec = tmpl.get("spec")
    if spec is None:
        return
    try:
        color = palette[int(annot_id) - 1]
    except Exception:
        color = "black"
    png_path = os.path.join(template_dir, f"syllable_{annot_id}.png")
    try:
        fig_tmpl, ax_tmpl = plt.subplots(figsize=(4, 3), dpi=100)
        ax_tmpl.imshow(spec, aspect='auto', origin='lower', cmap='inferno')
        ax_tmpl.set_title(f"Syllable {annot_id} Template", color=color)
        ax_tmpl.set_xlabel("Time (s)")
        ax_tmpl.tick_params(axis='y', which='both', left=False, labelleft=False)
        for spine in ax_tmpl.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)
        fig_tmpl.savefig(png_path, format='png')
        plt.close(fig_tmpl)
    except Exception as e:
        print(f"Failed to regenerate spectrogram template for {annot_id}: {e}")


def _write_wav_template(annot_id, tmpl, template_dir, pad=0.05):
    """Write ``syllable_{annot_id}.wav`` from a template's medoid audio.

    Reads the medoid's source wav and re-extracts the annotated segment (plus
    padding). Overwrites any existing file.
    """
    src = tmpl.get("source_path")
    t0 = tmpl.get("source_start")
    t1 = tmpl.get("source_end")
    if not src or t0 is None or t1 is None:
        return
    wav_path = os.path.join(template_dir, f"syllable_{annot_id}.wav")
    try:
        fs, data = read_wav(src)
        if getattr(data, "ndim", 1) == 2:
            data = data.mean(axis=1)
        start = max(0, int((t0 - pad) * fs))
        end = min(len(data), int((t1 + pad) * fs))
        if end <= start:
            return
        wavfile.write(wav_path, int(fs), np.asarray(data[start:end]))
    except Exception as e:
        print(f"Failed to regenerate audio template for {annot_id}: {e}")


def regenerate_disk_templates(templates, template_dir, pad=0.05):
    """Sync the on-disk syllable templates with a freshly built ``templates`` dict.

    Removes orphaned ``syllable_*.png`` / ``syllable_*.wav`` files for labels
    that no longer exist, and (re)writes the spectrogram (PNG) and audio (WAV)
    template for every surviving label from its medoid instance.

    Call this after merging label classes so the class/audio templates on disk
    reflect the new set of labels. ``templates`` is the dict returned by
    :func:`build_global_syllable_templates`.
    """
    if not template_dir or not os.path.isdir(template_dir):
        return

    valid_ids = set()
    for annot_id in templates:
        try:
            valid_ids.add(int(annot_id))
        except (TypeError, ValueError):
            valid_ids.add(annot_id)

    # 1) Remove orphaned template files for labels that no longer exist.
    for fname in os.listdir(template_dir):
        m = re.match(r"^syllable_(.+)\.(png|wav)$", fname)
        if not m:
            continue
        label_str = m.group(1)
        try:
            label = int(label_str)
        except ValueError:
            label = label_str
        if label not in valid_ids:
            try:
                os.remove(os.path.join(template_dir, fname))
            except OSError as e:
                print(f"Could not remove stale template {fname}: {e}")

    # 2) (Re)generate PNG + WAV for every surviving template from its medoid.
    palette = distinguishable_colors(1000)
    for annot_id, tmpl in templates.items():
        if not isinstance(tmpl, dict):
            continue
        _write_png_template(annot_id, tmpl, template_dir, palette)
        _write_wav_template(annot_id, tmpl, template_dir, pad)


class IntegratedSpectrogramViewer:
    """Full-featured spectrogram viewer / annotator."""

    TEMPLATE_COLS = dtw.TEMPLATE_COLS
    # Cap on how many appearances of each syllable are loaded to build its
    # template; the rest of the corpus is never read. See dtw_suggestions.
    MAX_INSTANCES_PER_LABEL = dtw.MAX_INSTANCES_PER_LABEL

    def __init__(self, master, bird_folder=None, annotation_callback=None,
                 db_data=None, date_to_wavs=None,
                 global_syllable_templates=None, verified_callback=None):
        self.annotation_callback = annotation_callback
        # Called with the new boolean whenever the user toggles "human
        # verified".  The viewer itself never touches the annotation file; the
        # owner of the file decides how to persist the flag.
        self.verified_callback = verified_callback
        self.human_verified = False
        self.root = master
        self.root.title("Spectrogram Viewer")
        screen_w = self.root.winfo_screenwidth()
        self.root.geometry(f"{screen_w}x700")

        # Global annotation references
        self.db_data = db_data
        self.date_to_wavs = date_to_wavs

        # Pre-built global syllable templates
        if global_syllable_templates is not None:
            self.syllable_templates = global_syllable_templates
        else:
            self.syllable_templates = {}

        self.bird_folder = bird_folder
        if self.bird_folder:
            self.template_dir = os.path.join(self.bird_folder, "syllable_templates")
            os.makedirs(self.template_dir, exist_ok=True)
        else:
            self.template_dir = None

        # ── Matplotlib figure ────────────────────────────────
        self.frame = tk.Frame(self.root)
        self.frame.pack(fill='both', expand=True)
        self.fig = plt.Figure(figsize=(8, 6), dpi=100)
        gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 0.6])
        self.ax_spec = self.fig.add_subplot(gs[0])
        self.ax_entropy = self.fig.add_subplot(gs[1], sharex=self.ax_spec)
        self.ax_entropy.set_xlabel("Time (s)")
        self.ax_entropy.set_ylabel("Energy (norm.)")
        self.ax_entropy.set_ylim(0, 1)
        self.ax_noise = self.fig.add_subplot(gs[2], sharex=self.ax_spec)
        self.ax_noise.set_ylabel("Noise", fontsize=7)
        self.ax_noise.set_yticks([])
        self.noise_audio = None
        self.entropy_thresh = 0.5
        self.min_syl_length = 10

        self.spec_window_type = "dpss"
        self.freq_cutoff = 0

        self.thresh_line = self.ax_entropy.axhline(
            self.entropy_thresh, color='red', linestyle='--', linewidth=2, picker=5)
        self.temp_rect = None
        self.deleted_segments = []

        self.show_suggestions = False
        self.slowmo_speed = 0.2
        self.show_ai_var = tk.BooleanVar(value=True)  # toggled by UI checkbox in btn_frame5

        # Mode flags
        self.segment_mode = False
        self.delete_region_mode = False
        self.segment_region_mode = False
        self.move_boundaries_mode = False
        self.annotate_region_mode = False
        self.region_start = None
        self.drag_boundary_info = None
        self.current_annotation = None

        # Embed canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        self.canvas.mpl_connect("pick_event", self.on_pick)
        self.canvas.mpl_connect("button_press_event", self.on_click)
        self.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.canvas.mpl_connect("button_release_event", self.on_release)
        self.dragging_thresh = False

        # ── Button row 1 ─────────────────────────────────────
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(fill='both')
        tk.Button(btn_frame, text="Play Window",
                  command=self.play_current_window).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Segment",
                  command=self.activate_segment_mode).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Delete Region",
                  command=self.activate_delete_region_mode).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Segment Region",
                  command=self.activate_segment_region_mode).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Move Boundaries",
                  command=self.activate_move_boundaries).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Annotate Region",
                  command=self.activate_annotate_region_mode).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Directions",
                  command=self.show_directions).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="View Syllable Template",
                  command=self.view_syllable_template).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Zoom",
                  command=self.activate_zoom_mode).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Reset Zoom",
                  command=self.reset_zoom).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Min Syl Length",
                  command=self.adjust_min_syl_length).pack(side=tk.LEFT, padx=5)

        # ── Button row 2 ─────────────────────────────────────
        btn_frame2 = tk.Frame(self.root)
        btn_frame2.pack(fill='x')
        tk.Button(btn_frame2, text="Show Suggestions",
                  command=self.toggle_suggestions).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame2, text="Spectrogram Calc",
                  command=self.choose_spectrogram_calc).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame2, text="Frequency Cutoff",
                  command=self.adjust_frequency_cutoff).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame2, text="Slow Motion Play",
                  command=self.play_slowmo_window).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame2, text="Set Slowmo Speed",
                  command=self.set_slowmo_speed).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame2, text="Audio Device",
                  command=self.select_audio_device).pack(side=tk.LEFT, padx=5)

        # Threshold fine control
        self.btn_thresh_down = tk.Button(btn_frame2, text="←", width=3,
                                         command=self.decrease_threshold)
        self.btn_thresh_down.pack(side=tk.LEFT, padx=2)
        self.lbl_thresh_val = tk.Label(btn_frame2,
                                       text=f"{self.entropy_thresh:.3f}", width=6)
        self.lbl_thresh_val.pack(side=tk.LEFT, padx=2)
        self.btn_thresh_up = tk.Button(btn_frame2, text="→", width=3,
                                       command=self.increase_threshold)
        self.btn_thresh_up.pack(side=tk.LEFT, padx=2)

        tk.Button(btn_frame2, text="Syllable Suggest",
                  command=self.activate_suggest_mode,
                  bg="#2196F3", fg="white",
                  font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame2, text="Recalc Templates",
                  command=self.recalculate_syllable_templates,
                  bg="#FF9800", fg="white",
                  font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame2, text="Auto Annotate",
                  command=self.activate_auto_suggest_mode,
                  bg="#9C27B0", fg="white",
                  font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)

        # ── Key bindings ─────────────────────────────────────
        self.root.bind("<Left>", lambda e: self.move_window(-1))
        self.root.bind("<Right>", lambda e: self.move_window(1))
        self.root.bind_all("<BackSpace>", self.delete_last_segment)
        self.root.bind("r", lambda e: self.activate_segment_region_mode())
        self.root.bind("s", lambda e: self.activate_segment_mode())
        self.root.bind("d", lambda e: self.activate_delete_region_mode())
        self.root.bind("m", lambda e: self.activate_move_boundaries())
        self.root.bind("a", self.toggle_annotate_region_mode)
        self.root.bind("i", self.zoom_in)
        self.root.bind("o", self.zoom_out)
        self.root.bind("h", lambda e: self.toggle_suggestions())
        self.root.bind("j", lambda e: self.activate_join_mode())
        self.root.bind("c", self.activate_entropy_replace_mode)
        self.root.bind("w", self.replace_with_all_entropy_suggestions)
        self.root.bind("t", lambda e: self.activate_suggest_mode())
        self.root.bind("g", lambda e: self.activate_auto_suggest_mode())
        self.root.bind("p", lambda e: self.prune_mismatched_labels())
        self.root.bind("v", self.toggle_human_verified)
        self.root.bind("<Up>", lambda e: self.increase_threshold())
        self.root.bind("<Down>", lambda e: self.decrease_threshold())

        self.join_mode = False
        self.suggest_mode = False

        # Viewing parameters
        self.window_duration = 1.0
        self.current_window_start = 0.0
        self.S_norm = None
        self.S_raw = None
        self.T = None
        self.F = None
        self.nfft = 512
        self.percentile_low = 5
        self.percentile_high = 96
        self.segments = []
        self.segment_start = None
        self.palette = distinguishable_colors(1000)
        # Cache of loaded SegmentClassifier objects, keyed by model directory.
        self._segment_classifier_cache = {}

        tk.Button(btn_frame2, text="Set Norm Percentiles",
                  command=self.adjust_norm_percentiles).pack(side=tk.LEFT, pady=5)

        # Every NN-backed control is built either way, but greyed out when the
        # deep-learning stack is missing. See the import guard at the top.
        nn_state = tk.NORMAL if TORCH_AVAILABLE else tk.DISABLED

        # ── NN model segmentation row ────────────────────────
        btn_frame3 = tk.Frame(self.root)
        btn_frame3.pack(fill='x')

        if not TORCH_AVAILABLE:
            tk.Label(btn_frame3,
                     text="⚠ NN features disabled — PyTorch not installed "
                          f"({TORCH_IMPORT_ERROR})",
                     fg="#B00020").pack(side=tk.LEFT, padx=5)

        tk.Label(btn_frame3, text="NN Architecture:").pack(side=tk.LEFT, padx=(5, 2))
        _arch_names = list(NN_MODEL_ARCHITECTURES.keys())
        self.nn_arch_var = tk.StringVar(
            value=_arch_names[0] if _arch_names else "")
        ttk.Combobox(btn_frame3, textvariable=self.nn_arch_var,
                     values=_arch_names,
                     state="readonly" if TORCH_AVAILABLE else tk.DISABLED,
                     width=30).pack(side=tk.LEFT, padx=2)

        tk.Label(btn_frame3, text="Weights:").pack(side=tk.LEFT, padx=(10, 2))
        self.nn_weights_var = tk.StringVar()
        self.nn_weights_combo = ttk.Combobox(
            btn_frame3, textvariable=self.nn_weights_var,
            state="readonly", width=45)
        self.nn_weights_combo.pack(side=tk.LEFT, padx=2)
        self._refresh_nn_weights_list()

        tk.Button(btn_frame3, text="Browse…",
                  command=self._browse_nn_weights).pack(side=tk.LEFT, padx=2)

        tk.Label(btn_frame3, text="Confidence:").pack(side=tk.LEFT, padx=(10, 2))
        self.nn_confidence_var = tk.DoubleVar(value=0.5)
        tk.Spinbox(btn_frame3, from_=0.0, to=1.0, increment=0.05,
                   textvariable=self.nn_confidence_var, width=5,
                   format="%.2f").pack(side=tk.LEFT, padx=2)

        tk.Button(btn_frame3, text="▶ Run NN Segmentation",
                  command=self.run_nn_segmentation,
                  bg="#4CAF50", fg="white", state=nn_state,
                  font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=10)

        # ── Syllable classification row (fills in 999 segments) ──────
        btn_frame4 = tk.Frame(self.root)
        btn_frame4.pack(fill='x')

        tk.Label(btn_frame4, text="Classifier model:").pack(side=tk.LEFT, padx=(5, 2))
        self.clf_model_var = tk.StringVar(
            value=os.path.normpath(CLASSIFICATION_MODEL_DIR))
        tk.Entry(btn_frame4, textvariable=self.clf_model_var,
                 width=50).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame4, text="Browse…", fg="black",
                  command=self._browse_clf_model).pack(side=tk.LEFT, padx=2)

        tk.Label(btn_frame4, text="Confidence:").pack(side=tk.LEFT, padx=(10, 2))
        self.clf_confidence_var = tk.DoubleVar(value=0.5)
        tk.Spinbox(btn_frame4, from_=0.0, to=1.0, increment=0.05,
                   textvariable=self.clf_confidence_var, width=5,
                   format="%.2f").pack(side=tk.LEFT, padx=2)

        tk.Button(btn_frame4, text="Classify 999 Segments",
                  command=self.classify_unlabeled_segments,
                  bg="#009688", fg="white", state=nn_state,
                  font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=10)

        # ── Full pipeline row (segmentation + classification in one click) ───
        btn_frame5 = tk.Frame(self.root)
        btn_frame5.pack(fill='x')

        tk.Button(btn_frame5, text="⚡ Full Auto Annotate",
                  command=self.run_full_annotation,
                  bg="#FF6F00", fg="white", state=nn_state,
                  font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=10, pady=3)

        tk.Label(btn_frame5,
                 text="Runs NN segmentation then classification in one step.",
                 fg="gray").pack(side=tk.LEFT, padx=5)

        self.show_ai_var = tk.BooleanVar(value=True)
        tk.Checkbutton(btn_frame5, text="Show AI annotations",
                       variable=self.show_ai_var,
                       command=self.update_plots).pack(side=tk.RIGHT, padx=10)

        # Human-verification toggle (press "v" for the same action).
        self.btn_verified = tk.Button(btn_frame5,
                                      command=self.toggle_human_verified,
                                      font=("Arial", 10, "bold"))
        self.btn_verified.pack(side=tk.RIGHT, padx=10, pady=3)
        self._update_verified_button()

    # ================================================================
    # Human verification flag
    # ================================================================

    def _update_verified_button(self):
        """Repaint the toggle so it reflects ``self.human_verified``."""
        if self.human_verified:
            self.btn_verified.config(text="✓ Human Verified",
                                     bg="#4CAF50", fg="white")
        else:
            self.btn_verified.config(text="✗ Not Verified",
                                     bg="#9E9E9E", fg="white")

    def set_human_verified(self, value):
        """Set the flag without notifying the callback.

        Used to seed the initial state when the window opens, so that merely
        displaying a file never rewrites the annotation on disk.
        """
        self.human_verified = bool(value)
        self._update_verified_button()

    def toggle_human_verified(self, event=None):
        """Flip the flag and notify the owner so it is persisted immediately."""
        self.human_verified = not self.human_verified
        self._update_verified_button()
        if self.verified_callback is not None:
            self.verified_callback(self.human_verified)

    # ================================================================
    # NN weights helpers
    # ================================================================

    def _refresh_nn_weights_list(self):
        weights_dir = os.path.normpath(NN_WEIGHTS_DIR)
        weight_files = []

        # Include the ONNX default first if it exists (absolute path).
        if os.path.isfile(ONNX_DEFAULT_WEIGHTS):
            weight_files.append(ONNX_DEFAULT_WEIGHTS)

        # Then scan the local weights directory for .pth and .onnx files.
        if os.path.isdir(weights_dir):
            for root_dir, dirs, files in os.walk(weights_dir):
                for f in sorted(files):
                    if f.endswith('.pth') or f.endswith('.onnx'):
                        full = os.path.join(root_dir, f)
                        if full not in weight_files:
                            rel = os.path.relpath(full, weights_dir)
                            weight_files.append(rel)

        self.nn_weights_combo['values'] = weight_files
        self._nn_weights_base_dir = weights_dir
        if weight_files:
            self.nn_weights_var.set(weight_files[0])  # ONNX default is first

    def _browse_nn_weights(self):
        path = filedialog.askopenfilename(
            title="Select Model Weights (.onnx / .pth)",
            filetypes=[
                ("Model weights", "*.onnx *.pth *.pt"),
                ("ONNX models", "*.onnx"),
                ("PyTorch weights", "*.pth *.pt"),
                ("All files", "*.*"),
            ],
            initialdir=os.path.normpath(NN_WEIGHTS_DIR),
        )
        if path:
            current_vals = list(self.nn_weights_combo['values'])
            if path not in current_vals:
                current_vals.append(path)
                self.nn_weights_combo['values'] = current_vals
            self.nn_weights_var.set(path)
            self._nn_weights_base_dir = None

    # ================================================================
    # NN segmentation
    # ================================================================

    def run_nn_segmentation(self):
        if not hasattr(self, 'audio') or self.audio is None:
            messagebox.showwarning("No Audio", "Please load a WAV file first.")
            return

        arch_name = self.nn_arch_var.get()
        if arch_name not in NN_MODEL_ARCHITECTURES:
            messagebox.showerror("Error", f"Unknown architecture: {arch_name}")
            return
        model_cls, model_kwargs = NN_MODEL_ARCHITECTURES[arch_name]

        weights_rel = self.nn_weights_var.get()
        if not weights_rel:
            messagebox.showwarning("No Weights", "Please select model weights.")
            return
        if os.path.isabs(weights_rel):
            weights_path = weights_rel
        else:
            weights_path = os.path.join(self._nn_weights_base_dir, weights_rel)
        if not os.path.isfile(weights_path):
            messagebox.showerror("Error", f"Weights file not found:\n{weights_path}")
            return

        confidence = self.nn_confidence_var.get()

        try:
            device = (torch.device('mps')
                      if getattr(torch.backends, 'mps', None)
                         and torch.backends.mps.is_available()
                      else torch.device('cpu'))
            if torch.cuda.is_available():
                device = torch.device('cuda')

            if weights_path.endswith('.onnx'):
                # ONNX model: architecture from bird_song_detector, runtime via onnxruntime
                model = OnnxSegmentationModel(weights_path)
            else:
                model = model_cls(**model_kwargs)
                state_dict = torch.load(weights_path, map_location=device)
                model.load_state_dict(state_dict)
                model.to(device)
                model.eval()

            from scipy.io import wavfile as wf_io
            tmp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            tmp_wav_path = tmp_wav.name
            tmp_wav.close()
            wf_io.write(tmp_wav_path, self.fs, self.audio)

            inferencer = InferenceToAnnotationData(model, device)
            annotation_data = inferencer.annotatate_data(tmp_wav_path)

            try:
                os.unlink(tmp_wav_path)
            except Exception:
                pass

            if not annotation_data:
                messagebox.showinfo("NN Segmentation",
                                    "No segments detected by the model.")
                return

            n_added = 0
            for seg in annotation_data:
                t_start, t_end, label = seg[0], seg[1], seg[2]
                overlap = False
                for existing in self.segments:
                    ex_start, ex_end, _ = existing
                    ov_start = max(t_start, ex_start)
                    ov_end = min(t_end, ex_end)
                    if ov_end > ov_start:
                        ov_len = ov_end - ov_start
                        seg_len = t_end - t_start
                        if seg_len > 0 and ov_len / seg_len > 0.5:
                            overlap = True
                            break
                if not overlap:
                    self.segments.append((t_start, t_end, int(label)))
                    n_added += 1

            self.segments.sort(key=lambda s: s[0])
            self.update_plots()
            if self.annotation_callback:
                self.annotation_callback(self.segments)

            messagebox.showinfo(
                "NN Segmentation",
                f"Added {n_added} segments from NN model.\n"
                f"({len(annotation_data)} detected, "
                f"{len(annotation_data) - n_added} skipped as overlapping).",
            )
        except Exception as e:
            import traceback
            messagebox.showerror(
                "NN Segmentation Error",
                f"Error running NN segmentation:\n{e}\n\n{traceback.format_exc()}",
            )

    # ================================================================
    # Syllable classification of unlabeled (999) segments
    # ================================================================

    def _browse_clf_model(self):
        path = filedialog.askdirectory(
            title="Select classification model folder",
            initialdir=os.path.dirname(os.path.normpath(CLASSIFICATION_MODEL_DIR)),
        )
        if path:
            self.clf_model_var.set(path)

    def classify_unlabeled_segments(self):
        """Label every unlabeled (999) segment with the trained 1D-CNN.

        Segments that already carry a label are left untouched. Short segments
        are classified using a fixed-length window centred on them (the model
        sees minimal surrounding context), and the predicted label is applied
        to the original segment span. Predictions below the confidence
        threshold keep the 999 placeholder.
        """
        if not hasattr(self, 'audio') or self.audio is None:
            messagebox.showwarning("No Audio", "Please load a WAV file first.")
            return

        unlabeled_idx = [
            i for i, s in enumerate(self.segments)
            if s[2] is None or _is_unlabeled_label(s[2])
        ]
        if not unlabeled_idx:
            messagebox.showinfo(
                "Classify", "No unlabeled (999) segments to classify.")
            return

        model_dir = self.clf_model_var.get().strip()
        if not model_dir or not os.path.isdir(model_dir):
            messagebox.showwarning(
                "Classify", "Select a valid classification model folder.")
            return
        try:
            min_conf = float(self.clf_confidence_var.get())
        except (tk.TclError, ValueError):
            min_conf = 0.0

        # Lazy import so the viewer still works without the deep-learning stack;
        # only this action needs the 1D-CNN classifier.
        try:
            from classification_model.segment_classifier import (
                SegmentClassifier, UNLABELED, predictions_to_triplets,
                fill_label_gaps,
            )
        except Exception as e:
            messagebox.showerror(
                "Classify",
                f"Could not import the classification model:\n{e}\n\n"
                "Make sure PyTorch is installed in this environment.",
            )
            return

        try:
            clf = self._segment_classifier_cache.get(model_dir)
            if clf is None:
                clf = SegmentClassifier.load(model_dir, verbose=False)
                self._segment_classifier_cache[model_dir] = clf
        except Exception as e:
            messagebox.showerror("Classify", f"Failed to load model:\n{e}")
            return

        # Sort by onset so the gap-filling heuristic below sees segments in
        # temporal order (its left/right neighbour lookup assumes this).
        self.segments.sort(key=lambda s: s[0])

        # Treat None labels as the 999 placeholder so freshly drawn segments are
        # classified too; already-labeled segments pass through unchanged.
        segs_for_clf = [
            (s[0], s[1], UNLABELED if s[2] is None else s[2])
            for s in self.segments
        ]
        try:
            preds = clf.classify_segments(
                self.audio, self.fs, segs_for_clf,
                windows_per_segment=3,
                only_unlabeled=True,
                unlabeled_value=UNLABELED,
                min_confidence=min_conf,
            )
        except Exception as e:
            import traceback
            messagebox.showerror(
                "Classify",
                f"Classification failed:\n{e}\n\n{traceback.format_exc()}",
            )
            return

        # Heuristic: a 999 segment whose nearest labeled neighbours on both
        # sides share the same label inherits that label. This propagates a
        # confident label across a low-confidence gap inside a run of the same
        # syllable (e.g. ... x 999 x ... -> ... x x x ...).
        n_model_classified = sum(1 for p in preds if p.classified)
        preds = fill_label_gaps(preds, UNLABELED)
        n_total_classified = sum(1 for p in preds if p.classified)
        n_gap_filled = n_total_classified - n_model_classified

        # Re-build segments from predictions, preserving any AI flags on
        # segments that were passed through unchanged (not classified).
        new_segs = []
        for orig_seg, pred in zip(self.segments, preds):
            triplet = pred.as_triplet()
            # If a user action classified this segment (model or gap-fill), the
            # AI flag is removed (label confirmed). Untouched 999 segments keep
            # their original tuple (and AI flag).
            if pred.classified:
                new_segs.append(triplet)
            else:
                # preserve original segment (with 4th element if present)
                new_segs.append(orig_seg if len(orig_seg) > 3 else triplet)
        # The rebuild preserves the temporal order established above.
        self.segments = new_segs
        n_low_conf = len(unlabeled_idx) - n_total_classified

        self.update_plots()
        if self.annotation_callback:
            self.annotation_callback(self.segments)

        msg = f"Classified {n_model_classified} segment(s) with the model."
        if n_gap_filled:
            msg += (f"\nFilled {n_gap_filled} gap(s) flanked by the same label "
                    f"on both sides.")
        msg += f"\n{n_low_conf} stayed 999 (below confidence {min_conf:g})."
        messagebox.showinfo("Classify", msg)

    # ================================================================
    # Full annotation pipeline (segmentation + classification)
    # ================================================================

    def run_full_annotation(self):
        """Run the end-to-end pipeline: NN segmentation then classification.

        Reuses the architecture/weights already configured in the NN-segmentation
        row and the classifier model directory from the classification row.
        AI-generated segments are added with a 4th element (True) and rendered
        with dashed lines so they can be toggled or overridden manually.
        """
        if not hasattr(self, 'audio') or self.audio is None:
            messagebox.showwarning("No Audio", "Please load a WAV file first.")
            return

        arch_name = self.nn_arch_var.get()
        if arch_name not in NN_MODEL_ARCHITECTURES:
            messagebox.showerror("Error", f"Unknown architecture: {arch_name}")
            return
        model_cls, model_kwargs = NN_MODEL_ARCHITECTURES[arch_name]

        weights_rel = self.nn_weights_var.get()
        if not weights_rel:
            messagebox.showwarning("No Weights", "Please select segmentation model weights.")
            return
        weights_path = (
            weights_rel
            if os.path.isabs(weights_rel)
            else os.path.join(self._nn_weights_base_dir, weights_rel)
        )
        if not os.path.isfile(weights_path):
            messagebox.showerror("Error", f"Weights file not found:\n{weights_path}")
            return

        clf_model_dir = self.clf_model_var.get()
        if not os.path.isdir(clf_model_dir):
            messagebox.showerror("Error",
                                 f"Classification model directory not found:\n{clf_model_dir}")
            return

        seg_confidence = self.nn_confidence_var.get()
        clf_confidence = self.clf_confidence_var.get()

        try:
            import tempfile
            from scipy.io import wavfile as wf_io
            from full_annotation_pipeline import FullAnnotationPipeline

            device = (torch.device('mps')
                      if getattr(torch.backends, 'mps', None)
                         and torch.backends.mps.is_available()
                      else torch.device('cpu'))
            if torch.cuda.is_available():
                device = torch.device('cuda')

            if weights_path.endswith('.onnx'):
                pipeline = FullAnnotationPipeline.load_onnx(
                    onnx_path=weights_path,
                    clf_model_dir=clf_model_dir,
                    seg_confidence=seg_confidence,
                    clf_confidence=clf_confidence,
                    device=device,
                    verbose=False,
                )
            else:
                pipeline = FullAnnotationPipeline.load(
                    model_cls, model_kwargs,
                    seg_weights_path=weights_path,
                    clf_model_dir=clf_model_dir,
                    seg_confidence=seg_confidence,
                    clf_confidence=clf_confidence,
                    device=device,
                    verbose=False,
                )

            tmp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            tmp_wav_path = tmp_wav.name
            tmp_wav.close()
            wf_io.write(tmp_wav_path, self.fs, self.audio)

            ai_segments = pipeline.annotate_wav(tmp_wav_path)

            try:
                os.unlink(tmp_wav_path)
            except Exception:
                pass

        except Exception as e:
            import traceback
            messagebox.showerror(
                "Full Annotation Error",
                f"Pipeline failed:\n{e}\n\n{traceback.format_exc()}",
            )
            return

        if not ai_segments:
            messagebox.showinfo("Full Auto Annotate", "No segments detected.")
            return

        # Merge: keep manual segments, replace any prior AI segments
        manual = [s for s in self.segments if not (len(s) > 3 and s[3])]
        n_added = 0
        for ai_seg in ai_segments:
            t_start, t_end = ai_seg[0], ai_seg[1]
            overlap = False
            for existing in manual:
                ex_start, ex_end = existing[0], existing[1]
                ov_len = max(0, min(t_end, ex_end) - max(t_start, ex_start))
                seg_len = t_end - t_start
                if seg_len > 0 and ov_len / seg_len > 0.5:
                    overlap = True
                    break
            if not overlap:
                manual.append(ai_seg)
                n_added += 1

        self.segments = sorted(manual, key=lambda s: s[0])
        self.update_plots()
        if self.annotation_callback:
            self.annotation_callback(self.segments)

        messagebox.showinfo(
            "Full Auto Annotate",
            f"Added {n_added} AI-annotated segment(s).\n"
            f"({len(ai_segments)} detected, {len(ai_segments) - n_added} skipped as overlapping)\n"
            f"AI segments shown with dashed lines — toggle with 'Show AI annotations'.",
        )

    # ================================================================
    # Entropy-based helpers
    # ================================================================

    def replace_with_all_entropy_suggestions(self, event=None):
        window_start = self.current_window_start
        window_end   = window_start + self.window_duration

        suggestions = self.compute_suggestions()
        if not suggestions:
            return

        def overlaps(a_start, a_end, b_start, b_end, eps=1e-9):
            return (a_start < b_end - eps) and (b_start < a_end - eps)

        in_view = [(t0, t1) for (t0, t1) in suggestions
                   if (t0 >= window_start) and (t1 <= window_end)]
        if not in_view:
            return

        in_view.sort(key=lambda x: (x[0], x[1]))
        candidates = in_view[:-1]
        if not candidates:
            return

        new_segments = list(self.segments)
        existing_intervals = [(s[0], s[1]) for s in new_segments]

        accepted = []
        for (t0, t1) in candidates:
            if any(overlaps(t0, t1, e0, e1) for (e0, e1) in existing_intervals):
                continue
            if any(abs(t0 - e0) < 1e-6 and abs(t1 - e1) < 1e-6
                   for (e0, e1) in existing_intervals):
                continue
            new_segments.append((t0, t1, None))
            existing_intervals.append((t0, t1))
            accepted.append((t0, t1))

        if accepted:
            self.segments = new_segments
            self.update_plots()
            if self.annotation_callback:
                self.annotation_callback(self.segments)

    def activate_entropy_replace_mode(self, event=None):
        self.deactivate_all_modes()
        self.entropy_replace_mode = True
        self.entropy_replace_start = None
        self.entropy_rect = None
        self.entropy_cid_press = self.canvas.mpl_connect(
            "button_press_event", self.on_entropy_replace_press)
        self.entropy_cid_motion = self.canvas.mpl_connect(
            "motion_notify_event", self.on_entropy_replace_motion)
        self.entropy_cid_release = self.canvas.mpl_connect(
            "button_release_event", self.on_entropy_replace_release)

    def on_entropy_replace_press(self, event):
        if not self.entropy_replace_mode or event.inaxes != self.ax_spec:
            return
        self.entropy_replace_start = event.xdata
        self.entropy_rect = Rectangle(
            (self.entropy_replace_start, self.F[0]), 0,
            self.F[-1] - self.F[0],
            linewidth=1, edgecolor='purple', linestyle='--', facecolor='none')
        self.ax_spec.add_patch(self.entropy_rect)
        self.canvas.draw()

    def on_entropy_replace_motion(self, event):
        if (not self.entropy_replace_mode or self.entropy_rect is None
                or event.inaxes != self.ax_spec):
            return
        x_current = event.xdata
        if x_current is None:
            return
        new_x = min(self.entropy_replace_start, x_current)
        new_width = abs(x_current - self.entropy_replace_start)
        self.entropy_rect.set_x(new_x)
        self.entropy_rect.set_width(new_width)
        self.canvas.draw()

    def on_entropy_replace_release(self, event):
        if (not self.entropy_replace_mode or self.entropy_rect is None
                or event.inaxes != self.ax_spec):
            return
        x_end = event.xdata
        if x_end is None:
            return
        region_start, region_end = sorted([self.entropy_replace_start, x_end])

        region_segments = [
            seg for seg in self.segments
            if region_start <= (seg[0] + seg[1]) / 2 <= region_end
        ]
        preserved_annotations = []
        for s in region_segments:
            midpoint = (s[0] + s[1]) / 2
            preserved_annotations.append((midpoint, s[2]))

        new_segments = [
            seg for seg in self.segments
            if not (region_start <= (seg[0] + seg[1]) / 2 <= region_end)
        ]

        suggestions = self.compute_suggestions()
        for t_start, t_end in suggestions:
            mid = (t_start + t_end) / 2
            if region_start <= mid <= region_end:
                matched_annot = None
                for old_mid, old_annot in preserved_annotations:
                    if abs(old_mid - mid) <= 0.05:
                        matched_annot = old_annot
                        break
                new_segments.append((t_start, t_end, matched_annot))

        self.segments = new_segments
        self.entropy_rect.remove()
        self.entropy_rect = None
        self.entropy_replace_mode = False
        self.update_plots()

        if self.annotation_callback:
            self.annotation_callback(self.segments)

    # ================================================================
    # Threshold controls
    # ================================================================

    def update_thresh_display(self):
        self.lbl_thresh_val.config(text=f"{self.entropy_thresh:.3f}")
        self.update_plots()

    def increase_threshold(self):
        self.entropy_thresh += 0.005
        self.update_thresh_display()

    def decrease_threshold(self):
        self.entropy_thresh -= 0.01
        self.update_thresh_display()

    # ================================================================
    # Playback
    # ================================================================

    def play_slowmo_window(self):
        if not hasattr(self, 'audio'):
            return
        start_sample = int(self.current_window_start * self.fs)
        end_sample = int((self.current_window_start + self.window_duration) * self.fs)
        segment = self.audio[start_sample:end_sample]
        sd.stop()
        try:
            sd.play(segment, int(self.fs * self.slowmo_speed))
        except Exception as e:
            messagebox.showerror("Error", f"Playback failed: {e}")

    def set_slowmo_speed(self):
        new_speed = simpledialog.askfloat(
            "Set Slow Motion Speed",
            "Enter playback speed (e.g., 0.5 for half speed):",
            initialvalue=self.slowmo_speed, minvalue=0.1, maxvalue=2.0)
        if new_speed:
            self.slowmo_speed = new_speed
            messagebox.showinfo("Slow Motion Speed",
                                f"Set to {self.slowmo_speed}× speed.")

    def select_audio_device(self):
        try:
            devices = sd.query_devices()
            hostapis = sd.query_hostapis()
        except Exception as e:
            messagebox.showerror("Audio Error", f"Could not query devices: {e}")
            return
        current_out = sd.default.device[1]
        out_devs = []
        for i, d in enumerate(devices):
            if d['max_output_channels'] > 0:
                api = hostapis[d['hostapi']]['name']
                out_devs.append((i, d['name'], api))
        if not out_devs:
            messagebox.showerror("Audio Error", "No output devices found.")
            return

        top = tk.Toplevel(self.root)
        top.title("Select Audio Output")
        top.geometry("400x300")
        tk.Label(top, text="Select Output Device:").pack(pady=5)
        lb_dev = tk.Listbox(top, selectmode=tk.SINGLE)
        lb_dev.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        mapping = []
        for idx, (dev_id, name, api) in enumerate(out_devs):
            marker = "*" if dev_id == current_out else " "
            lb_dev.insert(tk.END, f"{marker} [{dev_id}] {name} ({api})")
            mapping.append(dev_id)
            if dev_id == current_out:
                lb_dev.selection_set(idx)
                lb_dev.activate(idx)

        def on_confirm():
            sel = lb_dev.curselection()
            if not sel:
                return
            chosen_idx = mapping[sel[0]]
            try:
                current_in = sd.default.device[0]
                sd.default.device = (current_in, chosen_idx)
                dev_name = devices[chosen_idx]['name']
                messagebox.showinfo("Audio Device",
                                    f"Output set to:\n{dev_name}")
                top.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to set device: {e}")

        tk.Button(top, text="OK", command=on_confirm).pack(pady=10)

    # ================================================================
    # Join mode
    # ================================================================

    def activate_join_mode(self):
        self.deactivate_all_modes()
        self.join_mode = True
        self.join_start = None
        self.join_cid_press = self.canvas.mpl_connect(
            "button_press_event", self.on_join_click)
        self.join_cid_motion = self.canvas.mpl_connect(
            "motion_notify_event", self.on_join_motion)
        self.join_cid_release = self.canvas.mpl_connect(
            "button_release_event", self.on_join_release)

    def on_join_click(self, event):
        if not self.join_mode or event.inaxes != self.ax_spec:
            return
        self.join_start = event.xdata
        self.temp_rect = Rectangle(
            (self.join_start, self.F[0]), 0, self.F[-1] - self.F[0],
            linewidth=1, edgecolor='magenta', linestyle='--', facecolor='none')
        self.ax_spec.add_patch(self.temp_rect)
        self.canvas.draw()

    def on_join_motion(self, event):
        if (not self.join_mode or self.temp_rect is None
                or event.inaxes != self.ax_spec):
            return
        x_current = event.xdata
        if x_current is None:
            return
        new_x = min(self.join_start, x_current)
        new_width = abs(x_current - self.join_start)
        self.temp_rect.set_x(new_x)
        self.temp_rect.set_width(new_width)
        self.canvas.draw()

    def on_join_release(self, event):
        if (not self.join_mode or self.temp_rect is None
                or event.inaxes != self.ax_spec):
            return
        x_end = event.xdata
        if x_end is None:
            return
        region_start, region_end = sorted([self.join_start, x_end])

        new_segments = []
        merged_start = None
        merged_end = None
        for seg in self.segments:
            t_start, t_end, annot = seg[0], seg[1], seg[2]
            midpoint = (t_start + t_end) / 2.0
            if region_start <= midpoint <= region_end:
                if merged_start is None or t_start < merged_start:
                    merged_start = t_start
                if merged_end is None or t_end > merged_end:
                    merged_end = t_end
            else:
                new_segments.append(seg)

        if merged_start is not None and merged_end is not None:
            new_segments.append((merged_start, merged_end, None))

        self.segments = new_segments
        self.temp_rect.remove()
        self.temp_rect = None
        self.canvas.draw()
        self.update_plots()
        if self.annotation_callback:
            self.annotation_callback(self.segments)

    # ================================================================
    # Spectrogram settings
    # ================================================================

    def adjust_frequency_cutoff(self):
        new_cutoff = simpledialog.askfloat(
            "Frequency Cutoff",
            "Enter minimum frequency to display (Hz):",
            initialvalue=self.freq_cutoff)
        if new_cutoff is not None:
            self.freq_cutoff = new_cutoff
            messagebox.showinfo("Frequency Cutoff",
                                f"Frequencies below {self.freq_cutoff} Hz will be ignored.")
            self.recalculate_spectrogram()

    def choose_spectrogram_calc(self):
        window_type = simpledialog.askstring(
            "Spectrogram Calculation",
            "Enter spectrogram window type ('dpss', 'hann' or 'pretty'):",
            initialvalue=self.spec_window_type)
        if window_type:
            window_type = window_type.lower()
            if window_type in ["dpss", "hann", "pretty"]:
                self.spec_window_type = window_type
                messagebox.showinfo("Spectrogram Calculation",
                                    f"Using '{self.spec_window_type}' window.")
                self.recalculate_spectrogram()
            else:
                messagebox.showerror("Invalid Input",
                                     "Please enter either 'hann' or 'dpss'.")

    def recalculate_spectrogram(self):
        if not hasattr(self, "audio"):
            return
        prev_window_start = self.current_window_start
        y = self.audio
        fs = self.fs

        if self.spec_window_type == "dpss":
            noverlap = self.nfft - int(0.001 * fs)
            NW = 4
            E = windows.dpss(self.nfft, NW, Kmax=2)
            window1, window2 = E[0], E[1]
            F, T, S1 = spectrogram(y, fs, window=window1, noverlap=noverlap,
                                   nfft=self.nfft, nperseg=self.nfft, mode='complex')
            _, _, S2 = spectrogram(y, fs, window=window2, noverlap=noverlap,
                                   nfft=self.nfft, nperseg=self.nfft, mode='complex')
            S_power = np.abs(S1)**2 + np.abs(S2)**2
            S_raw = np.log1p(S_power)
        elif self.spec_window_type == "hann":
            noverlap = self.nfft - int(0.001 * fs)
            window1 = window2 = 'hann'
            F, T, S1 = spectrogram(y, fs, window=window1, noverlap=noverlap,
                                   nfft=self.nfft, nperseg=self.nfft, mode='complex')
            _, _, S2 = spectrogram(y, fs, window=window2, noverlap=noverlap,
                                   nfft=self.nfft, nperseg=self.nfft, mode='complex')
            S_power = np.abs(S1)**2 + np.abs(S2)**2
            S_raw = np.log1p(S_power)
        elif self.spec_window_type == "pretty":
            img, F, T = generate_pretty_sonogram(y, fs, postproc='y', norm_amp=0)
            S_raw = img
        else:
            raise ValueError(f"Unknown spectrogram type: {self.spec_window_type!r}")

        vmin, vmax = np.percentile(S_raw, [self.percentile_low, self.percentile_high])
        S_norm = np.clip(S_raw, vmin, vmax)
        if vmax > vmin:
            S_norm = (S_norm - vmin) / (vmax - vmin)
        else:
            S_norm = np.zeros_like(S_norm)

        self.S_raw  = S_raw
        self.S_norm = S_norm
        self.F      = F
        self.T      = T

        if self.freq_cutoff > 0:
            mask = self.F >= self.freq_cutoff
            self.F      = self.F[mask]
            self.S_norm = self.S_norm[mask, :]

        self.current_window_start = prev_window_start
        self.update_plots()

    # ================================================================
    # Entropy suggestions
    # ================================================================

    def compute_suggestions(self):
        suggestions = []
        if not hasattr(self, 'current_ent') or self.current_ent is None:
            return suggestions

        ent = self.current_ent
        T_crop = self.current_T_crop
        binary = (ent >= self.entropy_thresh).astype(int)
        diff = np.diff(binary)
        rising = np.where(diff == 1)[0] + 1
        falling = np.where(diff == -1)[0] + 1

        if binary[0] == 1:
            rising = np.insert(rising, 0, 0)
        if binary[-1] == 1:
            falling = np.append(falling, len(binary) - 1)

        combined = []
        for r, f in zip(rising, falling):
            if (f - r) < self.min_syl_length:
                combined.extend([0, 0])
            else:
                combined.extend([r, f])

        for i in range(0, len(combined), 2):
            start_idx = combined[i]
            end_idx = combined[i + 1]
            if start_idx == 0 and end_idx == 0:
                continue
            seg_start = T_crop[start_idx]
            seg_end = T_crop[end_idx]
            suggestions.append((seg_start, seg_end))
        return suggestions

    def toggle_suggestions(self):
        self.show_suggestions = not self.show_suggestions
        self.update_plots()

    def adjust_min_syl_length(self):
        new_threshold = simpledialog.askinteger(
            "Minimum Syllable Length",
            "Enter minimum syllable length (in index units):",
            initialvalue=self.min_syl_length, minvalue=1)
        if new_threshold is not None:
            self.min_syl_length = new_threshold
            messagebox.showinfo("Updated",
                                f"Minimum syllable length set to {self.min_syl_length}")

    # ================================================================
    # Playback & plotting
    # ================================================================

    def play_current_window(self):
        start_sample = int(self.current_window_start * self.fs)
        end_sample = int((self.current_window_start + self.window_duration) * self.fs)
        segment = self.audio[start_sample:end_sample]
        sd.stop()
        sd.play(segment, self.fs)

    def update_plots(self):
        window_start = self.current_window_start
        window_end = window_start + self.window_duration
        idx = np.where((self.T >= window_start) & (self.T <= window_end))[0]
        if len(idx) == 0:
            return
        S_norm_crop = self.S_norm[:, idx]
        T_crop = self.T[idx]
        F_crop = self.F

        self.ax_spec.clear()
        self.ax_spec.imshow(S_norm_crop, aspect='auto', origin='lower',
                            extent=[window_start, window_end,
                                    self.F[0], self.F[-1]],
                            cmap='gray_r', vmin=0, vmax=1)
        self.ax_spec.set_ylabel("Frequency (Hz)")
        self.ax_spec.set_xticklabels([])

        for seg in self.segments:
            t_start, t_end, annot = seg[0], seg[1], seg[2]
            ai = len(seg) > 3 and seg[3]
            if ai and not self.show_ai_var.get():
                continue
            if t_end < window_start or t_start > window_end:
                continue
            t_draw_start = max(t_start, window_start)
            t_draw_end = min(t_end, window_end)
            lw = 1.5 if ai else 2
            ls = '--' if ai else '-'
            alpha = 0.6 if ai else 1.0
            if annot is not None:
                color = self.palette[int(annot) - 1]
                self.ax_spec.axvline(t_draw_start, color=color, linewidth=lw,
                                     linestyle=ls, alpha=alpha)
                self.ax_spec.axvline(t_draw_end, color=color, linewidth=lw,
                                     linestyle=ls, alpha=alpha)
                mid_t = (t_draw_start + t_draw_end) / 2.0
                ylim = self.ax_spec.get_ylim()
                text_y = ylim[1] - 0.05 * (ylim[1] - ylim[0])
                label_txt = f"[{annot}]" if ai else str(annot)
                self.ax_spec.text(mid_t, text_y, label_txt, fontsize=12,
                                  fontweight='bold', color=color,
                                  ha='center', va='bottom', alpha=alpha)
            else:
                self.ax_spec.axvline(t_draw_start, color='green', linewidth=lw,
                                     linestyle=ls, alpha=alpha)
                self.ax_spec.axvline(t_draw_end, color='red', linewidth=lw,
                                     linestyle=ls, alpha=alpha)

        # Fast energy envelope: sum spectral energy above 1 kHz for each
        # time column (fully vectorized), then normalize to [0, 1] across
        # the current window. One value per column -> matches T_crop length.
        freq_mask = F_crop > 1000
        S_crop_high = S_norm_crop[freq_mask, :]
        energy = S_crop_high.sum(axis=0)

        gwin = windows.gaussian(M=5, std=5 / 6)
        energy = np.convolve(energy, gwin / gwin.sum(), mode='same')

        emin, emax = energy.min(), energy.max()
        if emax > emin:
            energy = (energy - emin) / (emax - emin)
        else:
            energy = np.zeros_like(energy)

        ent_vals = energy
        T_ent = T_crop

        self.current_ent = ent_vals
        self.current_T_crop = T_ent

        if self.show_suggestions:
            suggestions = self.compute_suggestions()
            for seg in suggestions:
                t_start, t_end = seg
                if t_end < window_start or t_start > window_end:
                    continue
                t_draw_start = max(t_start, window_start)
                t_draw_end = min(t_end, window_end)
                self.ax_spec.axvline(t_draw_start, color='gray',
                                     linestyle='--', linewidth=1)
                self.ax_spec.axvline(t_draw_end, color='gray',
                                     linestyle='--', linewidth=1)

        self.ax_entropy.clear()
        self.ax_entropy.plot(T_ent, ent_vals, color='blue', linewidth=1.5)
        self.ax_entropy.set_xlabel("Time (s)")
        self.ax_entropy.set_ylabel("Energy (norm.)")
        self.ax_entropy.set_ylim(0, 1)
        self.ax_entropy.set_xlim(window_start, window_end)
        self.ax_entropy.axhline(self.entropy_thresh, color='red',
                                linestyle='--', linewidth=2, picker=5)

        self.ax_noise.clear()
        self.ax_noise.set_ylabel("Noise ch", fontsize=7)
        self.ax_noise.set_yticks([])
        self.ax_noise.set_xlim(window_start, window_end)
        if self.noise_audio is not None:
            n_start = max(0, int(window_start * self.fs))
            n_end = min(len(self.noise_audio), int(window_end * self.fs))
            noise_seg = self.noise_audio[n_start:n_end].astype(np.float64)
            t_noise = np.linspace(window_start, window_end, len(noise_seg))
            # downsample for display performance
            step = max(1, len(noise_seg) // 4000)
            self.ax_noise.plot(t_noise[::step], noise_seg[::step],
                               color='orange', linewidth=0.8)
        self.fig.tight_layout()
        self.canvas.draw()

    def move_window(self, direction):
        step = self.window_duration * 0.25
        new_start = self.current_window_start + direction * step
        new_start = max(self.T[0],
                        min(new_start, self.T[-1] - self.window_duration))
        self.current_window_start = new_start
        self.update_plots()

    # ================================================================
    # Mode management
    # ================================================================

    def deactivate_all_modes(self):
        self.segment_mode = False
        self.delete_region_mode = False
        self.segment_region_mode = False
        self.move_boundaries_mode = False
        self.annotate_region_mode = False
        self.segment_start = None
        self.region_start = None
        self.join_mode = False

        if self.temp_rect is not None:
            try:
                self.temp_rect.remove()
            except Exception:
                pass
            self.temp_rect = None

        for group in [
            ("seg_cid_press", "seg_cid_motion", "seg_cid_release"),
            ("del_cid_press", "del_cid_motion", "del_cid_release"),
            ("boundary_cid_press", "boundary_cid_motion", "boundary_cid_release"),
            ("join_cid_press", "join_cid_motion", "join_cid_release"),
            ("entropy_cid_press", "entropy_cid_motion", "entropy_cid_release"),
            ("suggest_cid_press", "suggest_cid_motion", "suggest_cid_release"),
            ("auto_suggest_cid_press", "auto_suggest_cid_motion",
             "auto_suggest_cid_release"),
        ]:
            for cid_name in group:
                try:
                    self.canvas.mpl_disconnect(getattr(self, cid_name))
                except Exception:
                    pass
        try:
            self.canvas.mpl_disconnect(getattr(self, "seg_cid"))
        except Exception:
            pass
        try:
            self.canvas.mpl_disconnect(getattr(self, "annot_cid"))
        except Exception:
            pass

        self.entropy_replace_mode = False
        self.suggest_mode = False
        self.auto_suggest_mode = False

    # ================================================================
    # Segment mode
    # ================================================================

    def activate_segment_mode(self):
        self.deactivate_all_modes()
        self.segment_mode = True
        self.seg_cid_press = self.canvas.mpl_connect(
            "button_press_event", self.on_spec_click)
        self.seg_cid_motion = self.canvas.mpl_connect(
            "motion_notify_event", self.on_spec_motion)
        self.seg_cid_release = self.canvas.mpl_connect(
            "button_release_event", self.on_spec_release)

    def on_spec_click(self, event):
        if not self.segment_mode or event.inaxes != self.ax_spec:
            return
        self.segment_start = event.xdata
        self.temp_rect = Rectangle(
            (self.segment_start, self.F[0]), 0, self.F[-1] - self.F[0],
            linewidth=1, edgecolor='cyan', linestyle='--', facecolor='none')
        self.ax_spec.add_patch(self.temp_rect)
        self.canvas.draw()

    def on_spec_motion(self, event):
        if (not self.segment_mode or self.temp_rect is None
                or event.inaxes != self.ax_spec):
            return
        x_current = event.xdata
        if x_current is None:
            return
        new_x = min(self.segment_start, x_current)
        new_width = abs(x_current - self.segment_start)
        self.temp_rect.set_x(new_x)
        self.temp_rect.set_width(new_width)
        self.canvas.draw()

    def on_spec_release(self, event):
        if (not self.segment_mode or self.temp_rect is None
                or event.inaxes != self.ax_spec):
            return
        x0 = self.segment_start
        x1 = event.xdata
        if x0 is None or x1 is None:
            return
        t_start, t_end = sorted([x0, x1])
        self.segments.append((t_start, t_end, None))
        self.temp_rect.remove()
        self.temp_rect = None
        self.canvas.draw()
        self.update_plots()
        if self.annotation_callback:
            self.annotation_callback(self.segments)

    # ================================================================
    # Delete region mode
    # ================================================================

    def activate_delete_region_mode(self):
        self.deactivate_all_modes()
        self.delete_region_mode = True
        self.del_cid_press = self.canvas.mpl_connect(
            "button_press_event", self.on_del_click)
        self.del_cid_motion = self.canvas.mpl_connect(
            "motion_notify_event", self.on_del_motion)
        self.del_cid_release = self.canvas.mpl_connect(
            "button_release_event", self.on_del_release)

    def on_del_click(self, event):
        if not self.delete_region_mode or event.inaxes != self.ax_spec:
            return
        self.del_start = event.xdata
        self.temp_rect = Rectangle(
            (self.del_start, self.F[0]), 0, self.F[-1] - self.F[0],
            linewidth=1, edgecolor='cyan', linestyle='--', facecolor='none')
        self.ax_spec.add_patch(self.temp_rect)
        self.canvas.draw()

    def on_del_motion(self, event):
        if (not self.delete_region_mode or self.temp_rect is None
                or event.inaxes != self.ax_spec):
            return
        x_current = event.xdata
        if x_current is None:
            return
        new_x = min(self.del_start, x_current)
        new_width = abs(x_current - self.del_start)
        self.temp_rect.set_x(new_x)
        self.temp_rect.set_width(new_width)
        self.canvas.draw()

    def on_del_release(self, event):
        if (not self.delete_region_mode or self.temp_rect is None
                or event.inaxes != self.ax_spec):
            return
        x0 = self.del_start
        x1 = event.xdata
        if x0 is None or x1 is None:
            return
        region_start, region_end = sorted([x0, x1])
        self.segments = [
            seg for seg in self.segments
            if seg[1] <= region_start or seg[0] >= region_end
        ]
        self.temp_rect.remove()
        self.temp_rect = None
        self.canvas.draw()
        self.update_plots()
        if self.annotation_callback:
            self.annotation_callback(self.segments)

    # ================================================================
    # Segment region mode
    # ================================================================

    def activate_segment_region_mode(self):
        self.deactivate_all_modes()
        self.segment_region_mode = True
        self.seg_cid = self.canvas.mpl_connect(
            "button_press_event", self.on_spec_region_click)
        self.canvas.mpl_connect("motion_notify_event", self.on_spec_region_motion)
        self.canvas.mpl_connect("button_release_event", self.on_spec_region_release)

    def on_spec_region_click(self, event):
        if not self.segment_region_mode or event.inaxes != self.ax_spec:
            return
        self.region_start = event.xdata
        self.temp_rect = Rectangle(
            (self.region_start, self.F[0]), 0, self.F[-1] - self.F[0],
            linewidth=1, edgecolor='cyan', linestyle='--', facecolor='none')
        self.ax_spec.add_patch(self.temp_rect)
        self.canvas.draw()

    def on_spec_region_motion(self, event):
        if (not self.segment_region_mode or self.temp_rect is None
                or event.inaxes != self.ax_spec):
            return
        x_current = event.xdata
        if x_current is None:
            return
        new_x = min(self.region_start, x_current)
        new_width = abs(x_current - self.region_start)
        self.temp_rect.set_x(new_x)
        self.temp_rect.set_width(new_width)
        self.canvas.draw()

    def on_spec_region_release(self, event):
        if (not self.segment_region_mode or self.temp_rect is None
                or event.inaxes != self.ax_spec):
            return
        x_end = event.xdata
        if x_end is None:
            return
        region_start, region_end = sorted([self.region_start, x_end])
        self.temp_rect.remove()
        self.temp_rect = None
        self.canvas.draw()
        T_region_full = self.current_T_crop
        ent_region_full = self.current_ent
        region_mask = ((T_region_full >= region_start)
                       & (T_region_full <= region_end))
        if not np.any(region_mask):
            return
        T_region = T_region_full[region_mask]
        ent_region = ent_region_full[region_mask]
        binary = (ent_region >= self.entropy_thresh).astype(int)
        diff = np.diff(binary)
        rising = np.where(diff == 1)[0] + 1
        falling = np.where(diff == -1)[0] + 1
        if binary[0] == 1:
            rising = np.insert(rising, 0, 0)
        if binary[-1] == 1:
            falling = np.append(falling, len(binary) - 1)
        new_segments = []
        combined = []
        for r, f in zip(rising, falling):
            if (f - r) < self.min_syl_length:
                combined.extend([0, 0])
            else:
                combined.extend([r, f])
        for i in range(0, len(combined), 2):
            start_idx = combined[i]
            end_idx = combined[i + 1]
            if start_idx == 0 and end_idx == 0:
                continue
            seg_start = T_region[start_idx]
            seg_end = T_region[end_idx]
            new_segments.append((seg_start, seg_end, None))
        self.segments.extend(new_segments)
        self.update_plots()
        if self.annotation_callback:
            self.annotation_callback(self.segments)

    # ================================================================
    # Move boundaries mode
    # ================================================================

    def activate_move_boundaries(self):
        self.deactivate_all_modes()
        self.move_boundaries_mode = True
        self.boundary_cid_press = self.canvas.mpl_connect(
            "button_press_event", self.on_boundary_press)
        self.boundary_cid_motion = self.canvas.mpl_connect(
            "motion_notify_event", self.on_boundary_motion)
        self.boundary_cid_release = self.canvas.mpl_connect(
            "button_release_event", self.on_boundary_release)

    def on_boundary_press(self, event):
        if not self.move_boundaries_mode or event.inaxes != self.ax_spec:
            return
        clicked_time = event.xdata
        tol = 0.05 * self.window_duration
        best_diff = np.inf
        best_info = None
        for i, seg in enumerate(self.segments):
            t_start, t_end, annot = seg[0], seg[1], seg[2]
            diff_start = abs(clicked_time - t_start)
            diff_end = abs(clicked_time - t_end)
            if diff_start < best_diff and diff_start < tol:
                best_diff = diff_start
                best_info = (i, 'start')
            if diff_end < best_diff and diff_end < tol:
                best_diff = diff_end
                best_info = (i, 'end')
        self.drag_boundary_info = best_info

    def on_boundary_motion(self, event):
        if (not self.move_boundaries_mode or event.inaxes != self.ax_spec
                or self.drag_boundary_info is None):
            return
        new_val = event.xdata
        if new_val is None:
            return
        window_start = self.current_window_start
        window_end = window_start + self.window_duration
        new_val = max(window_start, min(new_val, window_end))
        seg = list(self.segments[self.drag_boundary_info[0]])
        if self.drag_boundary_info[1] == 'start':
            new_val = min(new_val, seg[1] - 0.01)
            seg[0] = new_val
        else:
            new_val = max(new_val, seg[0] + 0.01)
            seg[1] = new_val
        self.segments[self.drag_boundary_info[0]] = tuple(seg)
        self.update_plots()

    def on_boundary_release(self, event):
        self.drag_boundary_info = None
        if self.annotation_callback:
            self.annotation_callback(self.segments)

    # ================================================================
    # Annotate region mode
    # ================================================================

    def activate_annotate_region_mode(self):
        self.deactivate_all_modes()
        self.annotate_region_mode = True
        color_number = simpledialog.askinteger(
            "Annotate Region", "Enter syllable number (1-1000):",
            minvalue=1, maxvalue=1000)
        if color_number is None:
            self.annotate_region_mode = False
            return
        self.current_annotation = color_number
        self.annot_cid = self.canvas.mpl_connect(
            "button_press_event", self.on_annotate_region_click)
        self.canvas.mpl_connect("motion_notify_event",
                                self.on_annotate_region_motion)
        self.canvas.mpl_connect("button_release_event",
                                self.on_annotate_region_release)

    def on_annotate_region_click(self, event):
        if not self.annotate_region_mode or event.inaxes != self.ax_spec:
            return
        self.region_start = event.xdata
        self.temp_rect = Rectangle(
            (self.region_start, self.F[0]), 0, self.F[-1] - self.F[0],
            linewidth=1, edgecolor='orange', linestyle='--', facecolor='none')
        self.ax_spec.add_patch(self.temp_rect)
        self.canvas.draw()

    def on_annotate_region_motion(self, event):
        if (not self.annotate_region_mode or self.temp_rect is None
                or event.inaxes != self.ax_spec):
            return
        x_current = event.xdata
        if x_current is None:
            return
        new_x = min(self.region_start, x_current)
        new_width = abs(x_current - self.region_start)
        self.temp_rect.set_x(new_x)
        self.temp_rect.set_width(new_width)
        self.canvas.draw()

    def on_annotate_region_release(self, event):
        if (not self.annotate_region_mode or self.temp_rect is None
                or event.inaxes != self.ax_spec):
            return
        x_end = event.xdata
        if x_end is None:
            return
        region_start, region_end = sorted([self.region_start, x_end])
        self.temp_rect.remove()
        self.temp_rect = None
        self.canvas.draw()
        new_segments = []
        for seg in self.segments:
            t_seg_start, t_seg_end, annot = seg[0], seg[1], seg[2]
            mid = (t_seg_start + t_seg_end) / 2.0
            if region_start <= mid <= region_end:
                # user-assigned label removes the AI flag
                new_segments.append((t_seg_start, t_seg_end,
                                     self.current_annotation))
            else:
                new_segments.append(seg)
        self.segments = new_segments
        self.update_plots()
        for seg in self.segments:
            t_start, t_end, annot = seg[0], seg[1], seg[2]
            if annot is not None:
                self.save_template_for_annotation(annot, t_start, t_end)
        if self.annotation_callback:
            self.annotation_callback(self.segments)

    # ================================================================
    # Auto-suggest annotation mode
    # ================================================================

    def activate_auto_suggest_mode(self):
        if not self.syllable_templates:
            messagebox.showinfo(
                "Auto Annotate",
                "No syllable templates available yet.\n"
                "Annotate some segments first, then press 'Recalc Templates'.")
            return
        self.deactivate_all_modes()
        self.auto_suggest_mode = True
        self.auto_suggest_start = None
        self.auto_suggest_cid_press = self.canvas.mpl_connect(
            "button_press_event", self.on_auto_suggest_press)
        self.auto_suggest_cid_motion = self.canvas.mpl_connect(
            "motion_notify_event", self.on_auto_suggest_motion)
        self.auto_suggest_cid_release = self.canvas.mpl_connect(
            "button_release_event", self.on_auto_suggest_release)
        self.root.title(
            "Spectrogram Viewer  [AUTO ANNOTATE – drag to select region]")

    def on_auto_suggest_press(self, event):
        if not self.auto_suggest_mode or event.inaxes != self.ax_spec:
            return
        self.auto_suggest_start = event.xdata
        self.temp_rect = Rectangle(
            (self.auto_suggest_start, self.F[0]), 0,
            self.F[-1] - self.F[0],
            linewidth=2, edgecolor='#9C27B0', linestyle='--', facecolor='none')
        self.ax_spec.add_patch(self.temp_rect)
        self.canvas.draw()

    def on_auto_suggest_motion(self, event):
        if (not self.auto_suggest_mode or self.temp_rect is None
                or event.inaxes != self.ax_spec):
            return
        x_current = event.xdata
        if x_current is None:
            return
        new_x = min(self.auto_suggest_start, x_current)
        new_width = abs(x_current - self.auto_suggest_start)
        self.temp_rect.set_x(new_x)
        self.temp_rect.set_width(new_width)
        self.canvas.draw()

    def on_auto_suggest_release(self, event):
        if (not self.auto_suggest_mode or self.temp_rect is None
                or event.inaxes != self.ax_spec):
            return
        x_end = event.xdata
        if x_end is None:
            return
        region_start, region_end = sorted([self.auto_suggest_start, x_end])
        self.temp_rect.remove()
        self.temp_rect = None
        self.canvas.draw()
        self.root.title("Spectrogram Viewer")

        if (region_end - region_start) < 1e-4:
            return

        new_segments = []
        changed = 0
        for seg in self.segments:
            t_seg_start, t_seg_end, annot = seg[0], seg[1], seg[2]
            mid = (t_seg_start + t_seg_end) / 2.0
            if region_start <= mid <= region_end:
                suggestions = self.syllable_suggestions(t_seg_start, t_seg_end)
                if suggestions:
                    best_id, best_dist = suggestions[0]
                    new_segments.append((t_seg_start, t_seg_end, best_id))
                    changed += 1
                else:
                    new_segments.append(seg)
            else:
                new_segments.append(seg)
        self.segments = new_segments
        self.update_plots()
        if self.annotation_callback:
            self.annotation_callback(self.segments)
        messagebox.showinfo("Auto Annotate",
                            f"Applied top DTW suggestion to {changed} segment(s).")

    # ================================================================
    # Misc event handlers
    # ================================================================

    def delete_last_segment(self, event=None):
        if self.segments:
            self.segments.pop()
            self.update_plots()

    def on_pick(self, event):
        if event.artist == self.thresh_line:
            self.dragging_thresh = True

    def on_click(self, event):
        if event.inaxes == self.ax_entropy:
            if abs(event.ydata - self.entropy_thresh) < 0.2:
                self.dragging_thresh = True

    def on_motion(self, event):
        if self.dragging_thresh and event.inaxes == self.ax_entropy:
            self.entropy_thresh = event.ydata
            self.update_plots()

    def on_release(self, event):
        self.dragging_thresh = False

    def show_directions(self):
        directions_text = (
            "Keyboard Shortcuts:\n"
            "a: Annotate Region\n"
            "r: Segment Region\n"
            "s: Segment Individual\n"
            "d: Delete Region\n"
            "Backspace: Delete Last Segment\n"
            "m: Move Segments"
            "o: Zoom Out\n"
            "i: Zoom In\n"
            "v: Toggle Human Verified\n"
        )
        messagebox.showinfo("Directions", directions_text)

    def toggle_annotate_region_mode(self, event=None):
        new_number = simpledialog.askinteger(
            "Annotate Region", "Enter syllable number (1-1000):",
            minvalue=1, maxvalue=1000)
        if new_number is None:
            return
        self.current_annotation = new_number
        if not self.annotate_region_mode:
            self.deactivate_all_modes()
            self.annotate_region_mode = True
            self.annot_cid = self.canvas.mpl_connect(
                "button_press_event", self.on_annotate_region_click)
            self.canvas.mpl_connect("motion_notify_event",
                                    self.on_annotate_region_motion)
            self.canvas.mpl_connect("button_release_event",
                                    self.on_annotate_region_release)

    # ================================================================
    # Template saving & viewing
    # ================================================================

    def save_template_for_annotation(self, annot, t_start, t_end):
        if not self.template_dir:
            print("Template directory not set.")
            return

        # ── Spectrogram template (PNG) ───────────────────────────────
        template_filename = f"syllable_{annot}.png"
        template_filepath = os.path.join(self.template_dir, template_filename)
        if os.path.exists(template_filepath):
            print(f"Template already exists: {template_filepath}")
        else:
            midpoint = (t_start + t_end) / 2.0
            window_width = 0.1
            new_t_start = midpoint - window_width
            new_t_end = midpoint + window_width

            indices = np.where((self.T >= new_t_start)
                               & (self.T <= new_t_end))[0]
            if indices.size == 0:
                print(f"No time indices found in range "
                      f"{new_t_start} to {new_t_end}.")
            else:
                cropped_S = self.S_norm[:, indices]
                cropped_T = self.T[indices]
                color = self.palette[int(annot) - 1]

                fig_tmpl, ax_tmpl = plt.subplots(figsize=(4, 3), dpi=100)
                extent = [cropped_T[0], cropped_T[-1], self.F[0], self.F[-1]]
                ax_tmpl.imshow(cropped_S, aspect='auto', origin='lower',
                               extent=extent, cmap='inferno')
                ax_tmpl.set_title(f"Syllable {annot} Template", color=color)
                ax_tmpl.set_xlabel("Time (s)")
                ax_tmpl.tick_params(axis='y', which='both',
                                    left=False, labelleft=False)
                for spine in ax_tmpl.spines.values():
                    spine.set_edgecolor(color)
                    spine.set_linewidth(2)
                fig_tmpl.savefig(template_filepath, format='png')
                print(f"Template saved as {template_filepath}")
                plt.close(fig_tmpl)

        # ── Audio template (WAV) ─────────────────────────────────────
        # Saved so the syllable can be *played back* during later
        # annotation sessions to tell similar syllables apart by ear.
        self.save_audio_template_for_annotation(annot, t_start, t_end)

    def save_audio_template_for_annotation(self, annot, t_start, t_end,
                                           pad=0.05):
        """Save the syllable's audio as ``syllable_{annot}.wav``.

        Captures the full annotated segment plus a little padding on each
        side so the syllable is heard in context. Written once per syllable
        (skipped if the WAV already exists).
        """
        if not self.template_dir:
            return
        if not hasattr(self, "audio") or self.audio is None or self.fs is None:
            print("No audio loaded; skipping audio template.")
            return

        wav_filename = f"syllable_{annot}.wav"
        wav_filepath = os.path.join(self.template_dir, wav_filename)
        if os.path.exists(wav_filepath):
            print(f"Audio template already exists: {wav_filepath}")
            return

        start_sample = max(0, int((t_start - pad) * self.fs))
        end_sample = min(len(self.audio), int((t_end + pad) * self.fs))
        if end_sample <= start_sample:
            print(f"Empty audio range for syllable {annot}; skipping.")
            return

        segment = np.asarray(self.audio[start_sample:end_sample])
        try:
            wavfile.write(wav_filepath, int(self.fs), segment)
            print(f"Audio template saved as {wav_filepath}")
        except Exception as e:
            print(f"Failed to save audio template: {e}")

    def view_syllable_template(self):
        syllable_number = simpledialog.askinteger(
            "View Syllable Template", "Enter syllable number:")
        if syllable_number is None:
            return

        tmpl_data = self.syllable_templates.get(syllable_number)
        top = tk.Toplevel(self.root)
        top.title(f"Syllable {syllable_number} Template")

        if not isinstance(tmpl_data, dict) or tmpl_data.get("spec") is None:
            cv = tk.Canvas(top, width=400, height=300)
            cv.pack()
            cv.create_text(
                200, 150, justify='center', font=("Arial", 14),
                text=("No template for this syllable yet.\n"
                      "Annotate some segments, then press 'Recalc Templates'."))
            return

        # ── Spectrogram of the medoid (most representative) instance ──
        # Same logic as the DTW template: the displayed/played syllable is the
        # real instance with the smallest total DTW distance to all others.
        spec = tmpl_data["spec"]
        try:
            color = self.palette[syllable_number - 1]
        except Exception:
            color = "black"
        fig = plt.Figure(figsize=(4, 3), dpi=100)
        ax = fig.add_subplot(111)
        ax.imshow(spec, aspect='auto', origin='lower', cmap='inferno')
        ax.set_title(f"Syllable {syllable_number} Template", color=color)
        ax.tick_params(axis='both', labelsize=7)
        fig.tight_layout()
        FigureCanvasTkAgg(fig, master=top).get_tk_widget().pack(padx=5, pady=5)

        # ── Audio playback of that same instance ──
        seg = self._load_medoid_audio(tmpl_data)
        if seg is not None:
            data, fs = seg
            play_frame = tk.Frame(top)
            play_frame.pack(pady=5)
            tk.Button(play_frame, text="▶ Play Template Audio",
                      command=lambda d=data, f=fs: self._play_audio_segment(d, f)
                      ).pack(side=tk.LEFT, padx=5)
            tk.Button(play_frame, text="▶ Play Slow Motion",
                      command=lambda d=data, f=fs: self._play_audio_segment(
                          d, f, speed=self.slowmo_speed)
                      ).pack(side=tk.LEFT, padx=5)
        else:
            tk.Label(top, text="(audio for this template is unavailable)",
                     fg="gray").pack(pady=5)

    def _load_medoid_audio(self, tmpl_data, pad=0.05):
        """Load the audio for a template's medoid instance.

        Returns ``(data, fs)`` for the syllable segment (plus a little padding),
        or ``None``.  Reads the source wav when its path is known; falls back to
        the currently-loaded audio for templates whose medoid came from the
        current file (``source_path`` is ``None``).
        """
        t0 = tmpl_data.get("source_start")
        t1 = tmpl_data.get("source_end")
        if t0 is None or t1 is None:
            return None
        src = tmpl_data.get("source_path")
        try:
            audio = getattr(self, "audio", None)
            fs_cur = getattr(self, "fs", None)
            if src:
                fs, data = read_wav(src)
                if getattr(data, "ndim", 1) == 2:
                    data = data.mean(axis=1)
            elif audio is not None and fs_cur is not None:
                fs, data = fs_cur, np.asarray(audio)
            else:
                return None
            start = max(0, int((t0 - pad) * fs))
            end = min(len(data), int((t1 + pad) * fs))
            if end <= start:
                return None
            return data[start:end], int(fs)
        except Exception as e:
            print(f"Could not load template audio: {e}")
            return None

    def _safe_play(self, data, samplerate):
        """Play audio, falling back to a working output device.

        On macOS, Bluetooth devices like AirPods intermittently fail to open
        an output stream (PortAudio internal error ``-9986``) when they are
        also acting as a microphone or are in low-power mode.  Retry on the
        first real output device (typically the built-in speakers) before
        giving up.
        """
        try:
            sd.stop()
            sd.play(data, samplerate)
            return
        except Exception as first_err:
            for idx, dev in enumerate(sd.query_devices()):
                if dev.get("max_output_channels", 0) < 1:
                    continue
                try:
                    sd.play(data, samplerate, device=idx)
                    print(f"Default output failed ({first_err}); "
                          f"played on '{dev['name']}' instead.")
                    return
                except Exception:
                    continue
            raise first_err

    def _play_audio_segment(self, data, fs, speed=1.0):
        """Play an in-memory audio segment.  ``speed`` < 1.0 lowers the output
        sample rate for slow-motion playback."""
        try:
            self._safe_play(data, int(fs * speed))
        except Exception as e:
            messagebox.showerror("Error",
                                 f"Could not play template audio: {e}")

    def play_template_audio(self, wav_filepath, speed=1.0):
        """Play a saved syllable audio template (.wav).

        ``speed`` < 1.0 plays the template back in slow motion by lowering
        the output sample rate (e.g. 0.5 for half speed).
        """
        try:
            fs, data = read_wav(wav_filepath)
            self._safe_play(data, int(fs * speed))
        except Exception as e:
            messagebox.showerror("Error",
                                 f"Could not play audio template: {e}")

    # ================================================================
    # Data loading & results
    # ================================================================

    def load_existing_data(self, y, fs, file, seg_vector, annot_vector,
                           existing_segments=None, noise_audio=None):
        b, a = butter(N=4, Wn=1000 / (fs / 2), btype='highpass')
        y_hp = filtfilt(b, a, y)  # noqa: F841  # computed but original used
        self.audio = y
        self.fs = fs
        self.wav_filename = file
        self.noise_audio = noise_audio

        if seg_vector is None or (hasattr(seg_vector, "size")
                                  and seg_vector.size == 0):
            seg_vector = np.zeros(len(y), dtype=int)
        if annot_vector is None or (hasattr(annot_vector, "size")
                                    and annot_vector.size == 0):
            annot_vector = np.zeros(len(y), dtype=int)

        seg_changes = np.diff(np.concatenate(([0], seg_vector, [0])))
        seg_starts = np.where(seg_changes == 1)[0]
        seg_ends = np.where(seg_changes == -1)[0]
        self.segments = []
        for start, end in zip(seg_starts, seg_ends):
            t_start = start / fs
            t_end = end / fs
            seg_ann = annot_vector[start:end]
            nonzero = np.nonzero(seg_ann)[0]
            if nonzero.size > 0:
                annot = int(seg_ann[nonzero[0]])
            else:
                annot = 999
            self.segments.append((t_start, t_end, annot))

        # Re-attach AI flags: the binary round-trip loses the 4th element.
        # Match by (onset, offset) with ±10 ms tolerance.
        if existing_segments:
            ai_pairs = [
                (s[0], s[1])
                for s in existing_segments
                if len(s) > 3 and s[3]
            ]
            if ai_pairs:
                tagged = []
                for seg in self.segments:
                    t_s, t_e, ann = seg
                    is_ai = any(
                        abs(t_s - p0) < 0.01 and abs(t_e - p1) < 0.01
                        for p0, p1 in ai_pairs
                    )
                    tagged.append((t_s, t_e, ann, True) if is_ai else seg)
                self.segments = tagged

        noverlap = self.nfft - int(0.001 * fs)
        NW = 4
        if self.spec_window_type == "dpss":
            E = windows.dpss(self.nfft, NW, Kmax=2)
            window1 = E[0]
            window2 = E[1]
        elif self.spec_window_type == "hann":
            window1 = 'hann'
            window2 = 'hann'

        F, T, S1 = spectrogram(y, fs, window=window1, noverlap=noverlap,
                                nfft=self.nfft, nperseg=self.nfft)
        _, T, S2 = spectrogram(y, fs, window=window2, noverlap=noverlap,
                                nfft=self.nfft, nperseg=self.nfft)

        S_power = (np.abs(S1) + np.abs(S2)) / 2
        S_raw = np.log10(1 + (S_power + 1e-12))

        cutoff = 0.0
        mask = F >= cutoff
        S_raw[~mask, :] = 0

        eps = 1e-8
        minval = S_raw.min()
        maxval = S_raw.max()
        S_raw = (S_raw - minval) / (maxval - minval + eps)
        vmin = np.percentile(S_raw, 5) + eps
        vmax = np.percentile(S_raw, 97) + eps
        S_norm = np.clip((S_raw - vmin) / (vmax - vmin), 0, 1)

        self.S_raw  = S_raw
        self.S_norm = S_norm
        self.T      = T
        self.F      = F
        self.current_window_start = T[0]
        self.update_plots()

    def adjust_norm_percentiles(self):
        new_low = simpledialog.askinteger(
            "Normalization", "Enter low percentile:",
            initialvalue=self.percentile_low, minvalue=0, maxvalue=100)
        new_high = simpledialog.askinteger(
            "Normalization", "Enter high percentile:",
            initialvalue=self.percentile_high, minvalue=0, maxvalue=100)
        if new_low is not None and new_high is not None:
            self.percentile_low = new_low
            self.percentile_high = new_high
            vmin, vmax = np.percentile(
                self.S_raw, [self.percentile_low, self.percentile_high])
            S_norm = np.clip(self.S_raw, vmin, vmax)
            S_norm = (S_norm - vmin) / (vmax - vmin)
            self.S_norm = S_norm
            self.update_plots()

    def get_results(self):
        if (not hasattr(self, "audio") or self.audio is None
                or self.fs is None):
            return None, None, None, None, None
        n_samples = len(self.audio)
        seg_vector = np.zeros(n_samples, dtype=int)
        annot_vector = np.zeros(n_samples, dtype=int)
        for seg in self.segments:
            t_start, t_end, annot = seg[0], seg[1], seg[2]
            i_start = int(t_start * self.fs)
            i_end = int(t_end * self.fs)
            seg_vector[i_start:i_end] = 1
            if annot is not None:
                annot_vector[i_start:i_end] = annot
        return self.wav_filename, self.audio, self.fs, seg_vector, annot_vector

    # ================================================================
    # DTW / syllable suggestions  (delegates to dtw_suggestions module)
    # ================================================================

    def _extract_syllable_spectrogram(self, t_start, t_end):
        return dtw.extract_syllable_spectrogram(
            self.S_norm, self.T, t_start, t_end)

    @staticmethod
    def _resample_spectrogram_time(spec, target_cols=TEMPLATE_COLS):
        return dtw.resample_spectrogram_time(spec, target_cols)

    @staticmethod
    def _dtw_distance(seq_a, seq_b):
        return dtw.dtw_distance(seq_a, seq_b)

    @staticmethod
    def _compute_spectrogram_for_audio(y, fs, nfft=512):
        return dtw.compute_spectrogram_for_audio(y, fs, nfft)

    @staticmethod
    def build_global_syllable_templates(db_data, date_to_wavs, nfft=512,
                                        template_cols=TEMPLATE_COLS,
                                        max_instances_per_label=MAX_INSTANCES_PER_LABEL,
                                        verbose=False):
        return dtw.build_global_syllable_templates(
            db_data, date_to_wavs, nfft=nfft, template_cols=template_cols,
            max_instances_per_label=max_instances_per_label, verbose=verbose)

    @staticmethod
    def build_global_syllable_templates_cached(db_data, date_to_wavs, cache_path,
                                               nfft=512, template_cols=TEMPLATE_COLS,
                                               max_instances_per_label=MAX_INSTANCES_PER_LABEL,
                                               verbose=True):
        return dtw.build_global_syllable_templates_cached(
            db_data, date_to_wavs, cache_path,
            nfft=nfft, template_cols=template_cols,
            max_instances_per_label=max_instances_per_label, verbose=verbose)

    def _build_syllable_templates(self):
        self.syllable_templates = dtw.build_syllable_templates(
            self.S_norm, self.T, self.segments,
            self.db_data, self.date_to_wavs,
            nfft=self.nfft,
            wav_filename=getattr(self, 'wav_filename', None),
            template_cols=self.TEMPLATE_COLS,
            max_instances_per_label=self.MAX_INSTANCES_PER_LABEL)
        return self.syllable_templates

    def recalculate_syllable_templates(self):
        self._build_syllable_templates()
        n = len(self.syllable_templates)
        ids = sorted(self.syllable_templates.keys())
        messagebox.showinfo(
            "Recalculate Templates",
            f"Rebuilt {n} syllable template(s).\nIDs: {ids}")

    def syllable_suggestions(self, t_start, t_end):
        return dtw.syllable_suggestions(
            self.S_norm, self.T, self.syllable_templates,
            t_start, t_end, template_cols=self.TEMPLATE_COLS)

    # ================================================================
    # Prune mismatched-duration labels
    # ================================================================

    def prune_mismatched_labels(self):
        """For each syllable label, remove annotations from segments whose
        duration deviates >MAX_RATIO % from the majority cluster.

        Uses single-linkage clustering: segments are sorted by duration and
        split whenever consecutive durations differ by more than MAX_RATIO %.
        The largest cluster is kept; all other segments of that label have
        their annotation cleared (set to ``None``).
        """
         # same threshold used by DTW filtering

        # Gather segments grouped by label
        label_indices = {}  # label -> [index into self.segments]
        for i, (t0, t1, lab) in enumerate(self.segments):
            if lab is None:
                continue
            label_indices.setdefault(lab, []).append(i)

        total_cleared = 0
        details = []  # per-label summary lines

        for lab in sorted(label_indices.keys()):
            idxs = label_indices[lab]
            if len(idxs) < 2:
                continue

            # Sort indices by segment duration
            durations = []
            for idx in idxs:
                t0, t1, _ = self.segments[idx]
                durations.append((t1 - t0, idx))
            durations.sort(key=lambda x: x[0])

            # Single-linkage split: break whenever consecutive durations
            # differ by more than MAX_RATIO relative to the longer one
            clusters = [[durations[0]]]
            for k in range(1, len(durations)):
                prev_dur = durations[k - 1][0]
                curr_dur = durations[k][0]
                longer = max(prev_dur, curr_dur)
                if longer > 0 and (curr_dur - prev_dur) / longer > MAX_RATIO:
                    clusters.append([])
                clusters[-1].append(durations[k])

            if len(clusters) <= 1:
                continue  # all durations are consistent

            # Keep the largest cluster, clear annotations for the rest
            clusters.sort(key=len, reverse=True)
            majority = clusters[0]
            majority_indices = {item[1] for item in majority}

            cleared_this = 0
            for idx in idxs:
                if idx not in majority_indices:
                    t0, t1, _ = self.segments[idx]
                    self.segments[idx] = (t0, t1, None)
                    cleared_this += 1

            if cleared_this:
                maj_durs = [d for d, _ in majority]
                min_d, max_d = min(maj_durs), max(maj_durs)
                details.append(
                    f"  Label {lab}: kept {len(majority)}, "
                    f"cleared {cleared_this}  "
                    f"(kept range {min_d:.3f}–{max_d:.3f} s)")
                total_cleared += cleared_this

        if total_cleared == 0:
            messagebox.showinfo(
                "Prune Labels",
                f"All labels are duration-consistent (within {MAX_RATIO * 100:.0f} %).\n"
                "Nothing to change.")
            return

        self.update_plots()
        if self.annotation_callback:
            self.annotation_callback(self.segments)

        summary = "\n".join(details)
        messagebox.showinfo(
            "Prune Labels",
            f"Cleared {total_cleared} outlier annotation(s):\n{summary}")

    # ================================================================
    # Suggest mode (interactive)
    # ================================================================

    def activate_suggest_mode(self):
        self.deactivate_all_modes()
        self.suggest_mode = True
        self.suggest_start = None
        self.suggest_cid_press = self.canvas.mpl_connect(
            "button_press_event", self.on_suggest_press)
        self.suggest_cid_motion = self.canvas.mpl_connect(
            "motion_notify_event", self.on_suggest_motion)
        self.suggest_cid_release = self.canvas.mpl_connect(
            "button_release_event", self.on_suggest_release)
        self.root.title(
            "Spectrogram Viewer  [SUGGEST MODE – drag to select region]")

    def on_suggest_press(self, event):
        if not self.suggest_mode or event.inaxes != self.ax_spec:
            return
        self.suggest_start = event.xdata
        self.temp_rect = Rectangle(
            (self.suggest_start, self.F[0]), 0, self.F[-1] - self.F[0],
            linewidth=2, edgecolor='#2196F3', linestyle='--', facecolor='none')
        self.ax_spec.add_patch(self.temp_rect)
        self.canvas.draw()

    def on_suggest_motion(self, event):
        if (not self.suggest_mode or self.temp_rect is None
                or event.inaxes != self.ax_spec):
            return
        x_current = event.xdata
        if x_current is None:
            return
        new_x = min(self.suggest_start, x_current)
        new_width = abs(x_current - self.suggest_start)
        self.temp_rect.set_x(new_x)
        self.temp_rect.set_width(new_width)
        self.canvas.draw()

    def on_suggest_release(self, event):
        if (not self.suggest_mode or self.temp_rect is None
                or event.inaxes != self.ax_spec):
            return
        x_end = event.xdata
        if x_end is None:
            return
        t_start, t_end = sorted([self.suggest_start, x_end])
        self.temp_rect.remove()
        self.temp_rect = None
        self.canvas.draw()
        self.root.title("Spectrogram Viewer")

        if (t_end - t_start) < 1e-4:
            return

        suggestions = self.syllable_suggestions(t_start, t_end)
        if not suggestions:
            messagebox.showinfo(
                "Syllable Suggest",
                "No annotated syllable templates found yet.\n"
                "Annotate some segments first (key 'a').")
            return
        self._show_suggestions_popup(t_start, t_end, suggestions)

    def _show_suggestions_popup(self, t_start, t_end, suggestions):
        top = tk.Toplevel(self.root)
        top.title("Syllable Suggestions")
        top.geometry("700x520")

        # Query preview
        query_spec = self._extract_syllable_spectrogram(t_start, t_end)
        fig_q = plt.Figure(figsize=(6, 1.5), dpi=100)
        ax_q = fig_q.add_subplot(111)
        if query_spec is not None:
            ax_q.imshow(query_spec, aspect='auto', origin='lower', cmap='inferno')
        ax_q.set_title(f"Query  [{t_start:.3f}s – {t_end:.3f}s]", fontsize=10)
        ax_q.tick_params(axis='both', labelsize=7)
        fig_q.tight_layout()
        FigureCanvasTkAgg(fig_q, master=top).get_tk_widget().pack(
            fill='x', padx=5, pady=(5, 2))

        tk.Label(top,
                 text="Double-click a row to assign that label (or press Assign).",
                 font=("Arial", 9)).pack(anchor='w', padx=8)

        cols = ("Rank", "Syllable ID", "DTW Distance")
        tree = ttk.Treeview(top, columns=cols, show='headings', height=8)
        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=100, anchor='center')
        tree.pack(fill='both', expand=True, padx=5, pady=5)

        for rank, (annot_id, dist) in enumerate(suggestions, start=1):
            tree.insert('', 'end',
                        values=(rank, annot_id, f"{dist:.4f}"))

        fig_t = plt.Figure(figsize=(6, 1.5), dpi=100)
        ax_t = fig_t.add_subplot(111)
        ax_t.set_title("Select a row to preview template", fontsize=10)
        fig_t.tight_layout()
        canvas_t = FigureCanvasTkAgg(fig_t, master=top)
        canvas_t.get_tk_widget().pack(fill='x', padx=5, pady=(0, 5))

        def on_tree_select(event=None):
            sel = tree.selection()
            if not sel:
                return
            vals = tree.item(sel[0], 'values')
            annot_id = int(vals[1])
            tmpl_data = self.syllable_templates.get(annot_id)
            ax_t.clear()
            if tmpl_data is not None:
                tmpl_spec = tmpl_data["spec"] if isinstance(tmpl_data, dict) else tmpl_data
                color = self.palette[annot_id - 1]
                ax_t.imshow(tmpl_spec, aspect='auto', origin='lower', cmap='inferno')
                ax_t.set_title(f"Template – Syllable {annot_id}",
                               fontsize=10, color=color)
            else:
                ax_t.set_title("Template not available", fontsize=10)
            fig_t.tight_layout()
            canvas_t.draw()

        tree.bind('<<TreeviewSelect>>', on_tree_select)

        def assign_selected():
            sel = tree.selection()
            if not sel:
                messagebox.showwarning("Syllable Suggest",
                                       "Select a suggestion first.")
                return
            vals = tree.item(sel[0], 'values')
            annot_id = int(vals[1])
            changed = False
            new_segments = []
            for seg in self.segments:
                ts, te, ann = seg[0], seg[1], seg[2]
                mid = (ts + te) / 2.0
                if t_start <= mid <= t_end:
                    new_segments.append((ts, te, annot_id))  # user choice removes AI flag
                    changed = True
                else:
                    new_segments.append(seg)
            if not changed:
                new_segments.append((t_start, t_end, annot_id))
            self.segments = new_segments
            self.update_plots()
            if self.annotation_callback:
                self.annotation_callback(self.segments)
            top.destroy()

        tree.bind('<Double-1>', lambda e: assign_selected())

        btn_frame_popup = tk.Frame(top)
        btn_frame_popup.pack(fill='x', padx=5, pady=5)
        tk.Button(btn_frame_popup, text="Assign Selected Label",
                  command=assign_selected,
                  bg="#4CAF50", fg="white",
                  font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame_popup, text="Cancel",
                  command=top.destroy).pack(side=tk.LEFT, padx=5)

    # ================================================================
    # Zoom
    # ================================================================

    def activate_zoom_mode(self):
        self.prev_window_start = self.current_window_start
        self.prev_window_duration = self.window_duration
        self.deactivate_all_modes()
        self.zoom_mode = True
        self.zoom_start = None
        self.zoom_rect = None
        self.zoom_cid_press = self.canvas.mpl_connect(
            "button_press_event", self.on_zoom_press)
        self.zoom_cid_motion = self.canvas.mpl_connect(
            "motion_notify_event", self.on_zoom_motion)
        self.zoom_cid_release = self.canvas.mpl_connect(
            "button_release_event", self.on_zoom_release)

    def on_zoom_press(self, event):
        if not self.zoom_mode or event.inaxes != self.ax_spec:
            return
        self.zoom_start = event.xdata
        self.zoom_rect = Rectangle(
            (self.zoom_start, self.F[0]), 0, self.F[-1] - self.F[0],
            linewidth=1, edgecolor='blue', linestyle='--', facecolor='none')
        self.ax_spec.add_patch(self.zoom_rect)
        self.canvas.draw()

    def on_zoom_motion(self, event):
        if (not self.zoom_mode or self.zoom_rect is None
                or event.inaxes != self.ax_spec):
            return
        x_current = event.xdata
        if x_current is None:
            return
        new_x = min(self.zoom_start, x_current)
        new_width = abs(x_current - self.zoom_start)
        self.zoom_rect.set_x(new_x)
        self.zoom_rect.set_width(new_width)
        self.canvas.draw()

    def on_zoom_release(self, event):
        if (not self.zoom_mode or self.zoom_rect is None
                or event.inaxes != self.ax_spec):
            return
        x_end = event.xdata
        if x_end is None:
            return
        self.zoom_rect.remove()
        self.zoom_rect = None
        try:
            self.canvas.mpl_disconnect(self.zoom_cid_press)
        except Exception:
            pass
        try:
            self.canvas.mpl_disconnect(self.zoom_cid_motion)
        except Exception:
            pass
        try:
            self.canvas.mpl_disconnect(self.zoom_cid_release)
        except Exception:
            pass
        self.zoom_mode = False
        new_start = min(self.zoom_start, x_end)
        new_duration = abs(x_end - self.zoom_start)
        if new_duration < 0.01:
            new_duration = 0.01
        self.current_window_start = new_start
        self.window_duration = new_duration
        self.update_plots()

    def reset_zoom(self):
        if (hasattr(self, 'prev_window_start')
                and self.prev_window_start is not None):
            self.current_window_start = self.prev_window_start
            self.window_duration = self.prev_window_duration
            self.update_plots()
            self.prev_window_start = None
            self.prev_window_duration = None
        else:
            messagebox.showinfo("Reset Zoom",
                                "No previous zoom position available.")

    def zoom_in(self, event=None):
        factor = 0.9
        center = self.current_window_start + self.window_duration / 2.0
        new_duration = self.window_duration * factor
        new_start = center - new_duration / 2.0
        if new_start < self.T[0]:
            new_start = self.T[0]
        if new_start + new_duration > self.T[-1]:
            new_duration = self.T[-1] - new_start
        self.current_window_start = new_start
        self.window_duration = new_duration
        self.update_plots()

    def zoom_out(self, event=None):
        factor = 1.1
        center = self.current_window_start + self.window_duration / 2.0
        new_duration = self.window_duration * factor
        new_start = center - new_duration / 2.0
        if new_start < self.T[0]:
            new_start = self.T[0]
        if new_start + new_duration > self.T[-1]:
            new_duration = self.T[-1] - new_start
        self.current_window_start = new_start
        self.window_duration = new_duration
        self.update_plots()
