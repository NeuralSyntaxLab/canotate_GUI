# Cannotate — Annotation Tool Guide

A start-to-finish introduction.

Cannotate is a desktop (Tkinter) application for **segmenting and labelling birdsong
recordings**: you draw time boundaries around syllables on a spectrogram, give each
syllable a number, and the result is written to a JSON annotation file for the bird.
Manual drawing, entropy-based auto-segmentation, and neural-network segmentation +
classification all live in the same window, so you can annotate from scratch or just
review and correct what a model produced.

Entry point: Cannotate.app or Cannotate_GUI.py via the command line
---

## 1. Before you start

### 1.1 Your data must be laid out like this

```
<BIRD_FOLDER>/                                  ← you point the tool at this folder
├── LBRK219RT_annotation_<yourname>.json        ← annotation file (created for you)
├── 2024-07-05/                                 ← one folder per day, YYYY-MM-DD or YYYY_MM_DD
│   └── chop_data/
│       └── wav/
│           ├── bird_0001_2024_07_05_06_58_44.wav
│           └── ...
├── 2024-07-08/
│   └── chop_data/wav/...
├── <bird>_templates_<YYYYMMDD>.npz             ← the syllable template bank (§8)
└── syllable_templates/                         ← created automatically (PNG + WAV per syllable)
```

Only date-named folders that actually contain `chop_data/wav/*.wav` show up in the tool.
If your WAVs are in a flat folder, run
[wav_folder_to_cannotate_form_folderizer.py](wav_folder_to_cannotate_form_folderizer.py)
first — it builds exactly this structure (note: it **moves** files).


### 1.2 Environment

The tool needs the usual scientific stack (`numpy`, `scipy`, `matplotlib`,
`sounddevice`, `soundfile`) 


---

## 2. The three windows

| Window | What it is for |
|---|---|
| **Main window** (`Annotations for <bird>`) | Pick which song to work on; whole-dataset actions (templates, label merging, progress counts). |
| **Spectrogram Viewer** | Where all annotating happens: one song, one time window at a time. |
| **Full-Song Entropy Navigator** | A dark strip showing the *whole* song's energy; drag on it to jump the viewer to any part of the song. Existing annotations appear as coloured bars. |

The viewer and navigator open together when you press **Open Annotation**, and the song
is saved automaticly when you **close the viewer window**.

---

## 3. First launch, step by step

1. A dialog says "Please navigate to your bird folder" → choose `<BIRD_FOLDER>`.
2. Annotation file:
   - **case of no `*_annotation*.json` in the parent folder** → you are asked for a user name, and
     `<bird>_annotation_<yourname>.json` is created.
   - **Files exist** → "Do you want to choose an existing annotation file?"
     *Yes* opens a file picker (this is the normal choice — pick up where you left off);
     *No* asks for a user name and starts a fresh file.
3. The main window appears immediately, and the syllable templates are checked in the
   background (see §8). Only labels that have no template yet are fitted, so the first
   launch on a new bird spends a few minutes reading wavs behind a progress window and
   every launch after that reads one small file.

### The main window

- **Select Date** — the day folders found.
- **Select WAV File** — songs in that day. Each name carries a marker:
  `✓` = a human has verified this file, `○` = not verified yet.
- **Select Song #** — a global index across *all* days (Song #1 … #N), so you can walk
  the whole dataset in order without thinking about dates.
- **Counts line** — `Annotated / Verified` for the current date, and totals for the bird.
  "Annotated" means the file has at least one segment; "Verified" means someone ticked
  the human-verified flag (§7).
- **Used syllable types** — every label in the annotation file with its segment count.
  Useful for spotting an over-split class before it spreads.
- **Open Annotation** — opens the selected song in the viewer. This is the main button.
- **Templates (from annotations)** — the template browser (§9).
- **Merge Label Classes** — collapse labels, e.g. everything labelled 7 becomes 3 (§9).
- **Remove Entry from Annotation File** — deletes this file's entry from the JSON. The
  WAV is never touched, so the song can be annotated again from scratch.

---

## 4. Reading the viewer

The viewer shows three stacked panels for the **current time window** (1 second at first):

1. **Spectrogram** (top, largest) — where you draw.
2. **Energy** (middle) — normalised energy above 1 kHz, with a **red dashed threshold
   line**. Everything above the line counts as sound for the automatic segmenters. Drag
   the line with the mouse, or use `↑` / `↓`, or the `←` `→` buttons next to the numeric
   readout.
3. **Noise channel** (bottom, orange) — the white-noise/trigger channel, if the WAV is
   stereo. Display only. *not relevant at this stage*

How segments are drawn on the spectrogram:

| Appearance | Meaning |
|---|---|
| Green line (start) + red line (end) | Segment with no label yet (saved as label `999`). |
| Two coloured lines + a number on top | Labelled syllable; colour is fixed per label. |
| Dashed, faded, label in `[brackets]` | AI-generated segment (from Full Auto Annotate). Hide/show them with the **Show AI annotations** checkbox. |
| Grey dashed lines | Entropy *suggestions*, not real segments — they only exist while suggestions are toggled on (`h`). |

Labelling or reassigning an AI segment by hand promotes it to a normal (non-dashed)
segment — that is how you "accept" model output.

---

## 5. The basic workflow

A typical pass over one song:

1. **Open Annotation** → viewer + navigator appear.
2. Drag on the **navigator** to jump to the part of the song you want, or step through
   with `←` / `→` (each press moves by ¼ of the visible window).
3. **Listen**: `Play Window`, or `Slow Motion Play` for fast syllables
   (`Set Slowmo Speed`, default 0.2× — 0.5 is half speed).
4. **Get boundaries**:
   - Adjust the red threshold until the energy curve cleanly separates syllables from
     silence, press `h` to preview the suggestions, then press `r` and drag over a region
     to turn the suggestions inside it into real segments; or
   - press `s` and drag out each syllable by hand.
5. **Fix boundaries**: `m`, then drag any segment edge. `d` + drag deletes every segment
   touching the dragged region. `Backspace` removes the most recently added segment.
   `j` + drag merges everything in the region into one segment.
6. **Label**: press `a`, type the syllable number (1–1000), then drag over the syllables
   that should get that number. Every segment whose *midpoint* falls inside the drag gets
   the label. Repeat per label. (A template image and audio clip are saved the first time
   a label is used.)
7. **Verify**: when the song looks right, press `v` (or the big button) to mark it
   **✓ Human Verified**.
8. **Close the viewer window** → the annotation is written to the JSON and a "Saved"
   confirmation appears. The main window's counts and `✓` markers update.

> **Saving:** segments are written **only when you close the viewer**. The
> human-verified flag is the exception — it is written to disk the moment you toggle it.
> There is no undo beyond `Backspace`, and no autosave.

Keyboard shortcuts act on the viewer window, so click on it once if a key seems to do
nothing (focus may still be on the navigator or the main window).

---

## 6. Full control reference

### Keyboard

| Key | Action |
|---|---|
| `←` / `→` | Move the window earlier / later by ¼ window. |
| `↑` / `↓` | Raise / lower the energy threshold in fine steps. |
| `s` | Segment mode — drag to create one segment. |
| `r` | Segment Region — drag; suggestions inside become segments. |
| `d` | Delete Region — drag; every segment overlapping it is removed. |
| `Backspace` | Delete the last segment added. |
| `m` | Move Boundaries — drag a segment edge (grab within 5 % of the window width). |
| `j` | Join — drag; all segments in the region merge into one (label cleared). |
| `a` | Annotate Region — asks for a syllable number, then drag to apply it. |
| `c` | Entropy Replace — drag; segments in the region are replaced by fresh suggestions, reusing the old labels where midpoints match within 50 ms. |
| `w` | Accept the entropy suggestions fully inside the current view (the last one is skipped, and anything overlapping an existing segment is ignored). |
| `h` | Show / hide entropy suggestions. |
| `t` | Syllable Suggest — drag a region, get the three best-matching labels (§8). |
| `g` | Auto Annotate — drag a region; every segment in it gets the top match. |
| `p` | Prune mismatched labels — clears labels whose duration is an outlier for that class. |
| `i` / `o` | Zoom in / out around the centre of the view. |
| `v` | Toggle **Human Verified** (saved immediately). |

### Buttons

**Row 1 — drawing and viewing**

`Play Window` · `Segment` (`s`) · `Delete Region` (`d`) · `Segment Region` (`r`) ·
`Move Boundaries` (`m`) · `Annotate Region` (`a`) · `Directions` (short shortcut
reminder) · `View Syllable Template` (shows a label's template spectrogram + a play
button) · `Zoom` (drag a region to zoom to it) · `Reset Zoom` (back to the pre-zoom view)
· `Min Syl Length` (suggestions shorter than this are dropped; units are spectrogram
columns ≈ milliseconds, default 10).

**Row 2 — suggestions, audio, display**

`Show Suggestions` (`h`) · `Spectrogram Calc` (`dpss` default / `hann` / `pretty`) ·
`Frequency Cutoff` (hide everything below N Hz) · `Slow Motion Play` ·
`Set Slowmo Speed` · `Audio Device` (pick an output device — use this if playback is
silent or fails on Bluetooth headphones) · `←  0.500  →` (threshold) ·
`Syllable Suggest` (`t`) · `Recalc Templates` · `Auto Annotate` (`g`) ·
`Set Norm Percentiles` (spectrogram contrast, default 5 / 96).

**Row 3 — NN segmentation.** Architecture, weights (`.onnx` / `.pth`; the lab's default
ONNX model is preselected), a confidence spinbox, and `▶ Run NN Segmentation`. It finds
syllable boundaries and adds them as plain segments, skipping anything that overlaps an
existing segment by more than 50 %. Note that the confidence value in this row is only
applied by **Full Auto Annotate**, not by this button.

**Row 4 — classification.** Model folder (defaults to
[classification_model/conv1d_phrase_model](classification_model/), `Browse…` to change),
a confidence spinbox, and `Classify 999 Segments`, which assigns labels to unlabelled
segments only — already-labelled segments are never overwritten. Predictions below the
confidence stay `999`. A gap-filling rule then relabels a lone `999` when the labelled
segments on both sides agree.

**Row 5 — pipeline and status.** `⚡ Full Auto Annotate` runs segmentation then
classification in one go, adding results as dashed **AI** segments;
`Show AI annotations` hides/shows them; the green/grey button on the right is the
**Human Verified** toggle.

---

## 7. The human-verified flag

Each file entry carries `human_verified: true/false`. It answers "has a person actually
looked at this song?", which is different from "does this song have segments" — a song
you inspected and found empty should still be marked verified.

- Toggle with `v` or the button; it is saved to the JSON immediately, even if you then
  close the viewer without changing anything else.
- The main window shows `✓` / `○` per file and a `Verified: n / N` count.
- Older annotation files without the field are migrated on load (everything starts
  unverified).

---

## 8. How the AI-assist features decide things

- **Entropy suggestions** (`h`, `r`, `c`, `w`) are pure signal processing: energy above
  1 kHz, smoothed, thresholded at the red line, with runs shorter than *Min Syl Length*
  discarded. No model, no training — if the suggestions look wrong, move the threshold.
- **Syllable templates** live in one file per bird at the top of the bird folder,
  `<bird>_templates_<YYYYMMDD>.npz`, and are the same templates Error Correction scores
  against (§9.1). A label earns a template once it has **15 annotated segments**; a
  200 ms window is cut from the centre of its most typical rendition. Templates are
  **never rebuilt on their own** — start-up only fits labels the file does not already
  hold, so the templates you checked yesterday are the templates you get today. Press
  `Recalc Templates` to refit every label from current annotations and drop the labels
  that no longer exist. Each refit writes the day's file; earlier days are kept.
- **Syllable Suggest** (`t`) cross-correlates the region you dragged against every
  template and lists the **three best labels** by match score (peak ZNCC: 1.000 is
  identical, 0 is unrelated), with a preview of the query and of the selected template.
  Double-click a row (or *Assign Selected Label*) to apply it.
  The **Covered** column is worth reading. It is how much of the template your selection
  was long enough to be compared against. At 100% the whole template was scored. Below
  that, your selection was shorter than the template and was matched against the best-
  fitting *part* of it — which is easy to do well against a long, featureless template,
  so a high score at 40% coverage is much weaker evidence than the same score at 100%.
- **Auto Annotate** (`g`) is the same comparison applied silently to every segment in the
  dragged region, taking the top match each time. Fast, but it never says "I'm not sure" —
  check its work.
- **Prune mismatched labels** (`p`) clusters each label's segments by duration and clears
  the label on segments more than 50 % away from the majority cluster. It is a way to
  find syllables that were mislabelled by `g`, not a general cleanup.
- **NN segmentation / classification** use trained models (see rows 3–5 above) and are the
  right starting point on a fresh bird once models exist for it.

---

## 9. Dataset-wide tools (main window)

**Templates (from annotations)** — the template browser. Pick a syllable type from *Used
syllable types*; the left list shows every song containing it, with how many consecutive
**runs** it forms. Set `k` and press *Filter Songs by k* / build to hear and see `k`
instances concatenated, or double-click a song to load a real contiguous run from that
recording. Click any highlighted span in the display to select it, then
*Change Label of Selected/Displayed* to relabel — this writes straight to the annotation
JSON. This is the fastest way to audit a class: if one instance in the concatenation
sounds or looks different, relabel it on the spot.

Each span can also be **reshaped and cut up in place**, and every edit is saved to the
JSON immediately:

- **Move a border** — drag either edge of a span. A drag stops at the neighbouring
  segment, and in a spliced concatenation it stays inside that instance's own audio; in
  a contiguous run loaded by double-clicking a song, borders may move anywhere inside the
  loaded window (including the half-second of padding on each side).
- **Split** — tick *Split mode* and click inside a span to cut it at that point, or select
  one span and press *Split Selected into N…* for equal parts. Both halves keep the
  original label, so the usual next step is to select one of them and relabel it.

Each span is captioned with its current label and coloured by it, so sub-segments you
relabel are immediately distinguishable.

**Merge Label Classes** — select one or more source labels, type the target, confirm.
Every matching segment in the whole file is rewritten and the JSON is saved. The
templates are then refitted — merging is exactly the case where an existing template must
be replaced rather than kept — and the on-disk `syllable_templates/` PNG/WAV files are
regenerated (orphans removed). Cannot be undone — copy the JSON first if you are unsure.

**Remove Entry from Annotation File** — removes the selected file's entry (segments +
verified flag) from the JSON only. Templates keep including its segments until the next
rebuild.

### 9.1 Error Correction — find labels that are probably wrong

The template browser asks *"show me everything labelled 6, does anything look out of
place?"*. **Error Correction** asks the sharper question: *"which phrases labelled 6 does
the evidence say are not 6, and what are they instead?"*

A **phrase** here is a run of consecutive segments sharing one label. Pick a detection
method, pick a label, press **Run Detection**, and the left-hand list fills with
suspicious phrases, worst first. Selecting one shows it on a spectrogram with its
neighbours for context, and you correct it without leaving the window — every edit is
written to the annotation JSON immediately.

Two methods ship with the tool:

| Method | What it flags | Needs |
|---|---|---|
| **Duration (z-score)** | Phrases far from the mean duration for their label. Fast, and catches gross mistakes, but duration alone tells apart very little — many syllable types have near-identical lengths. | nothing |
| **Cross correlation** | Phrases whose *spectrogram* matches some other label's template better than their own. This is the one that says what the phrase probably is. | a template bank — the bird's own, or one you point it at |

#### The bank it uses by default

**Templates…** starts pointed at the bird's own bank, `<bird>_templates_<date>.npz` at the
top of the bird folder — the same one Syllable Suggest ranks against, kept up to date by
the tool itself (§8). If that is the bank you want, there is nothing to build and nothing
to choose.

The button is still there for pointing it somewhere else: a bank you fitted from a
different annotation file, or an older day's bank you want to compare against.

#### Fitting a bank yourself, from the command line

A bank is a single `.npz` of about 1 MB. To build one outside the tool:

```bash
python -m cannotate.xcorr fit \
    --annotations /path/to/LBRK219_verified.json \
    --audio-root  "/Volumes/Extreme Pro/LBRK219RT" \
    --out         "/Volumes/Extreme Pro/LBRK219RT/templates.npz"
```

This reads only about 30 example phrases per label, so it takes a minute or two, not an
hour (71 s for this bird). `python -m cannotate.xcorr describe --templates <file>` prints
what a bank contains.

#### Looking at a bank

`describe` says how many templates and how wide. To see whether they are the syllables you
meant, open them:

```bash
python -m cannotate.xcorr show --templates "/Volumes/Extreme Pro/LBRK219RT/templates.npz"
```

A window with one panel per template, drawn exactly as Error Correction draws the template
beside a flagged phrase. Add `--labels 3,26` to compare just a few. It only reads the bank,
so it is the safe thing to run against one you are unsure of.

Worth doing once after every fit: a template that is silence, or that is really the
neighbouring syllable, makes every later scan flag the wrong phrases, and the numbers
`describe` prints cannot show you that.

**Keep the bank in the bird folder**, beside the `<date>/` directories, so it travels with
the audio it describes and the file picker opens near it. Note that a bank named
`<bird>_templates_<YYYYMMDD>.npz` there becomes the one the tool picks up by itself, newest
date first — so name a hand-fitted experimental bank something else unless you mean to
replace the bird's own.

**Fit it from annotations you trust.** The bank defines what each label *looks like*, so
if you fit it from a file full of the mistakes you are about to hunt for, the mistakes
help define the templates that then bless them. Use verified data.

Then in the window: press **Templates…**, choose the `.npz`, pick a label, press
**Run Detection**. A progress bar appears with a working **Cancel** — cancelling keeps
whatever was found so far, so it is a fine way to sample a large dataset.

#### Correcting, quickly

Each flagged phrase arrives with a **suggestion** — the label the evidence prefers —
shown after an arrow in the list (`… — label 6 — 2 seg — 0.412s  →  3`). The detail view
shows the phrase on the left and, on the right, **the template it was matched against**,
so you can see why before you agree.

Under the spectrogram sits a strip of up to three candidate labels with their match
scores (`3 · 0.842`). Clicking one *previews* its template in the right-hand panel — it
does not change anything — so you can compare two candidates before committing.

| Key | Does |
|---|---|
| `←` `→` | previous / next flagged phrase |
| `a` | **accept** the suggestion and move to the next phrase |
| `1` `2` `3` | preview candidate 1, 2 or 3 |
| `Enter` | apply the **previewed** candidate and move on |
| `d` | keep this phrase as it is, and move on |
| `r` | the full Relabel dialog — any label, and it respects a segment selection |
| `e` | set label 0 |

`a`, `Enter` and `d` all advance automatically, which is the point: a list of two hundred
phrases is triaged by pressing one key per phrase.

`a` and `Enter` always relabel the **whole phrase**, because the phrase as a whole is what
was scored. To change only part of it, click segments on the spectrogram to select them
and use `r` instead.

#### The margin, and when to widen the net

**margin ≤** is the flagging threshold. The **margin** is the phrase's own label's match
score minus the best competing label's, so it says how clearly the label you already have
won. It is a difference rather than an absolute score on purpose: recording gain, noise
floor and the bird's condition are the same for every label being compared, so they
cancel out and nothing needs calibrating per file.

- **negative** — another label matched better. A disagreement.
- **near zero** — a near-tie; the method has no real opinion.
- **large positive** — the current label won comfortably.

Measured on 200 files of this bird, the median margin is **+0.30**, and the distribution
is clearly bimodal with a valley at **+0.10 to +0.15** — which is where the threshold has
somewhere principled to sit:

| margin ≤ | phrases flagged | share of scored |
|---|---|---|
| `0.00` (default) | 109 | 4.2% — outright disagreements only |
| `0.15` | 443 | 17.2% — everything below the valley |
| `0.30` | 1295 | 50.2% — past the main mass; too many to be useful |

Start at `0.00`. Raise it to `0.15` on a class you particularly distrust: that adds the
near-ties, which are phrases the method probably got right but was not confident about.

#### When there is no suggestion

The list never hides a phrase just because the method cannot advise on it. Instead the
phrase appears with an explanation in place of a suggestion, and you correct it by hand:

- **no template for label N** — the bank was never fitted on this label. The candidate
  strip still shows the closest labels the bank *does* hold; pick one, or use `r`.
- **too short to score** — the phrase is shorter than every template in the bank.
- **too short to compare with label N** — the phrase is shorter than its *own* label's
  template, so it could never be tested against the label it already has. Other, shorter
  templates may still be listed, but a short phrase matching a short template is weak
  evidence — check it by eye rather than trusting the ranking.
- **audio unavailable** — the WAV could not be read, so nothing was scored.

#### How good is it?

Measured on this bird's human-verified annotations, with templates and test phrases drawn
from **different recording dates**: **95% agreement** across 22 labels. Shuffling which
template belongs to which label drops that to 0%, so it is genuinely reading the audio.

Treat the 5% as *disagreements to look at*, not as errors — plenty of them are the method
being wrong, which is exactly why you look at the spectrogram before pressing `a`. Two
confusions dominate and are worth recognising: **3 → 26** and **18 → 1**. Label 26 is a
sustained harmonic stack, which is a *part* of what a label-3 phrase contains, so it can
match a 3 very well; the score rewards the less specific template. If you see those two
pairs coming up repeatedly, that is the known limitation, not new information.

Scanning one label across 3000 songs takes roughly 3-4 minutes on four cores.

---

## 10. The annotation file

```json
{
  "2024-07-05": [
    {
      "file_name": "bird_0001_2024_07_05_06_58_44.wav",
      "human_verified": true,
      "segments": [
        [2.2462, 2.5483, 1],
        [3.3215, 3.6485, 1],
        [6.2215, 6.5205, 999, true]
      ]
    }
  ]
}
```

- Top level: date → list of file entries. One entry per WAV that has been opened.
- `segments`: `[onset_seconds, offset_seconds, label]`, optionally a 4th element `true`
  marking an AI-generated segment.
- **`999` means "segmented but not labelled"** — that is the placeholder the classifier
  fills in and what an unlabelled hand-drawn segment becomes on save.
- Times are seconds from the start of the WAV; labels are integers 1–1000.

Helper functions for these files (merging two annotators' files, MAT→JSON conversion,
counting/merging label classes, dropping entries whose WAV is gone) live in
[utils.py](utils.py).

---

## 11. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Tool exits with `ModuleNotFoundError: bird_song_detector` | Wrong environment — use the one with the lab package installed (`moove_env` here). |
| "No date subfolders found" | The bird folder has no `YYYY-MM-DD` (or `YYYY_MM_DD`) day folders, or they contain no `chop_data/wav/*.wav`. |
| A day or file is missing from the dropdown | Same cause; also, files starting with `._` are ignored on purpose. |
| Keyboard shortcuts do nothing | The viewer window does not have focus — click on it. |
| Nothing plays / playback error | Use `Audio Device` and pick the built-in output; Bluetooth devices intermittently fail to open a stream on macOS. |
| "No annotation was made" when closing | The viewer had no audio loaded; nothing was saved. Reopen the file. |
| Suggestions cover silence, or miss syllables | Move the red threshold (`↑`/`↓` or drag) and adjust `Min Syl Length`. |
| Suggested labels look outdated | Templates are never refitted on their own. Press `Recalc Templates` to rebuild them from your current annotations. |
| A label never appears in the suggestions | It has fewer than 15 annotated segments, so it has no template yet. Annotate more of it, then relaunch (or press `Recalc Templates`). |
| A long wait behind "Reading annotated songs…" on first launch | Templates are being fitted for a bird that had none. It happens once; later launches read the `.npz` alone. You can Cancel, and nothing is written. |
| A label you merged away is still suggested | Start-up only adds templates. Press `Recalc Templates`, which is the one path that drops them. |
| Closed the viewer by accident | Segments are saved on close, so your work is in the JSON — just reopen the song. |
