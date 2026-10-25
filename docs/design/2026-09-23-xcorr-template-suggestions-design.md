# Syllable suggestions from cross-correlation templates

Replaces the DTW suggestion path with the cross-correlation template matching
already validated for outlier detection (`cannotate/xcorr/`, and
`docs/design/2026-09-14-xcorr-validation-design.md`).  One template bank per
bird, persisted at the dataset root, feeds both the suggestion popup and the
error-correction browser.

## Why

The DTW path resamples every syllable to 50 time columns and compares them
with a banded DTW distance.  The resampling discards duration and smears the
time-frequency structure; the 15% duration gate then has to put some of it
back.  The xcorr path measures 95.6% agreement with human labels across 17
labels on a date-level split, and 0.0% with the template-to-label mapping
shuffled, so it reads real template content.  Keeping two template systems in
one program is the thing worth avoiding most: they disagree, and only one of
them is measured.

## The file

    <bird_folder>/<bird_name>_templates_YYYYMMDD.npz

The newest date wins on load; ties break on mtime, and names that do not match
the pattern are ignored.  A write never overwrites an *earlier day*: a new
dated file lands beside the old, which stays as history.  A second write on
the same day replaces that day's file — history is kept per day, not per
write, and a folder accumulating a 1 MB file per template addition would be
worse than no history at all.

`xcorr.bank.Bank` gains provenance parallel to its existing `label` / `cols`
arrays: `src_date`, `src_file`, `src_t0`, `src_t1` — the exact window a
template was cut from, not the whole phrase it came from.  `_centred_window`
currently discards its offset; it returns it instead, and `fit` carries it
through.  Provenance is what lets `syllable_templates/*.png` and `*.wav` be
regenerated and a template be played back.  A bank saved before this change
loads with empty provenance rather than failing.

## Eligibility and fitting

`template_store.fit_pool` turns `db_data` into the
`(date, file_name, t_start, t_end, label)` specs `xcorr.bank.fit` wants:

* 4-element segments (the AI-generated flag) are excluded, as the DTW path
  already did;
* labels `None`, `0` and `999` are excluded — `0` is the reject marker the
  error-correction "e" shortcut writes, and a template cut from it would be
  heterogeneous by construction and would win against anything unusual;
* human-verified entries are preferred.  A label with fewer than
  `MIN_SEGMENTS` verified segments falls back to its unverified ones, so a
  corpus with no verification yet still gets templates.

A label is eligible once it has **`MIN_SEGMENTS = 10` segments** in that pool.

`ensure(bird_folder, db_data, date_to_wavs, progress)` loads the newest bank,
finds the eligible labels it lacks, fits **only those**, merges, and writes a
new dated file.  When nothing is missing it writes nothing and returns the
bank it loaded: templates are never refit automatically, which is what keeps
startup free on every launch after the first.

`ensure(..., force=True)` refits every eligible label from current data and
drops templates whose label no longer exists.  That is the "Recalc Templates"
button, and the escape hatch for a bad early template.

## Suggestions

The viewer's `S_norm` cannot be scored against a bank: it is percentile-clipped
to [0, 1] at a 1 ms hop over the full band, while the bank lives on raw
`log1p` power at the bank's own hop, nfft and band.  The viewer therefore
computes a second spectrogram per song, with `xcorr.spectro.compute` at the
**bank's** parameters, once at load and cached.  Following the bank rather than
a constant means an older bank still scores correctly.

For a selection `[t0, t1]`, per label:

* **query >= template** — peak of `zncc_surface(query, template)`.  Every such
  template goes through one `TemplateBank.surfaces(query)` call, which
  transforms the query once and reuses it across labels.
* **query < template** — the roles reverse: `zncc_surface(template, query)`,
  peak over lags.  That is "find the best alignment inside the template and
  score against the trim at that lag", and it is the only way a selection
  shorter than a 200 ms template can be scored at all.

Ranked by peak ZNCC descending; the **top 3** are shown, with the score and a
coverage figure (`query ms / template ms`).

Coverage is displayed because the reversed path sharpens a failure mode the
validation already found: a long, generic template can host a short query at
high ZNCC anywhere inside it, exactly as label 26's sustained stack outscored
label 3's transient-plus-stack.  Peak ZNCC rewards the least specific
template.  Showing coverage makes a suspiciously partial match visible instead
of silent; it is a mitigation, not a fix.

No score floor anywhere.  The popup always shows its three rows — the user is
looking at them and can judge — and bulk auto-annotate keeps taking rank 1
unconditionally, as it does today.  The 15% duration gate is deleted: xcorr
does not need a duration prior, and applying one would discard the reversed
path's whole purpose.

## Wiring

* **Startup**: `ensure()` runs on a worker thread behind a small cancellable
  progress window.  The main window opens immediately; the suggestion keys
  report that templates are still building until it lands.  The bank lives in
  a mutable holder, so a viewer opened during the fit picks it up when it
  finishes.
* **Error correction**: the bank path initialises to the discovered file,
  and the "Templates…" picker stays as an override for experimental banks.
* **Disk templates**: `regenerate_disk_templates` reads bank provenance and
  re-reads the source wav, rather than a medoid dict.  The bank reaches every
  one of these consumers through `template_store.as_display_templates`, which
  presents it in the dict shape the DTW templates had — so the template
  viewer, the PNG/WAV regeneration and the popup preview needed no changes
  beyond the wording on screen.

## Removals

`dtw_distance`, `medoid_index`, `medoid_template`, `medoid_instance`,
`_subsample_for_medoid`, `resample_spectrogram_time`,
`collect_syllable_instances`, `build_global_syllable_templates`,
`build_global_syllable_templates_cached`, `build_syllable_templates`,
`syllable_suggestions`, and the entire `.templates_cache.pkl` fingerprint and
pickle layer, with `tests/test_templates_cache.py` and
`tests/test_templates_fingerprint.py`.

What survives is the spectrogram machinery the viewer's own display depends
on — `extract_syllable_spectrogram`, `compute_spectrogram_for_audio`,
`compute_spectrogram_for_ranges`, `frame_times`.  The annotation walk that
chose which instances to read (`_labelled_segments`, `_candidate_pool`,
`plan_instance_selection`) goes too: its only caller was the DTW template
build, and `template_store.fit_pool` is the replacement.  With DTW gone
the module name is a lie, so `cannotate/dtw_suggestions.py` becomes
`cannotate/spectrograms.py` and its importers are updated.  No shim: nothing
outside this repo imports it.

## Testing

Headless, no tkinter, in the style of the existing `tests/`:

* provenance round-trips through `save`/`load`, and a bank saved without it
  still loads;
* a fitted template's provenance re-cuts the same patch from its source;
* eligibility: the 10-segment threshold, `0`/`999`/`None` exclusion,
  AI-flagged exclusion, verified-preferred with fallback;
* `ensure()` writes a new dated file only when a label was added, the union is
  preserved, and the old file is untouched;
* `newest()` orders by date and ignores foreign names;
* ranking is descending and capped at three; a self-match scores ~1.0;
* a query shorter than every template still ranks, through the reversed path;
* `force=True` prunes templates whose label has vanished.

## Risks

* The reversed path is statistically permissive for very short selections.
  Coverage is shown; a specificity-aware statistic (PSR was measured and
  rejected as a primary) is out of scope here.
* A cold bird with many labels spends a few minutes reading wavs on first
  launch before suggestions work.  Subsequent launches read only the npz.
