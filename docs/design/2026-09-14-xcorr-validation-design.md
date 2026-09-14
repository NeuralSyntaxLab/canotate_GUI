# Cross-correlation validation sandbox — design

Status: implemented — see `sandbox/xcorr/` and the Results section
Date: 2026-09-14

## Goal

Decide, with evidence, whether **spectrogram cross-correlation against per-label
templates** can identify mislabeled phrases in the LBRK219 annotations.

The deliverable of this design is *an answer*, not a shipped feature. The sandbox
is disposable. If the answer is yes, the scoring core graduates into
`cannotate/` and fills in `CrossCorrelationOutlierDetector`
(`cannotate/error_correction.py:235`), which is currently a stub returning `[]`.

## Error class in scope

**Label confusion only** — a phrase annotated `5` that is really a `6`.

Explicitly *not* in scope for this sandbox:

- boundary errors (over/under-splitting),
- atypical-but-correctly-labeled renditions.

Those need different scores and different reductions. They are noted in
"Deferred" below.

## Ground truth

`/Users/nitzankarby/Desktop/masters/yarden cohen/dev/behavioral_experiment_syntax_manipulation/LBRK219_auto_phase_level_only_verified.json`

- 282 files across 7 dates, **all `human_verified: true`**
- phrase-level segments (a segment is a run of renditions, not one syllable)
- labels in scope: 4 (n=384), 5 (n=331), 6 (n=335), plus 9 (n=219),
  10 (n=359), 15 (n=300), 16 (n=302) — added because duration cannot separate
  them (9 vs 10 differ by 50 ms with ~100 ms spread; 6, 15 and 16 all sit near
  1.8 s).  A `XCORR_LABELS=all` mode widens this to every label with >= 30
  instances, which is the honest test.
- degenerate-short segments are negligible here: 3 / 0 / 0 under 100 ms

Audio root: `/Volumes/Extreme Pro/LBRK219RT`, one directory per date.

This file is the whole reason the experiment is meaningful. The earlier
candidate (`LBRK_auto_annotated_phrase_level_cleaned.json`) is 2386/2392
machine-labeled, so validating against it would only measure agreement with the
classifier that produced it — templates would inherit its systematic errors and
the confusion matrix would look good for the wrong reason.

## Decision: timing is not a feature

Phrase duration alone classifies 4/5/6 at **81.5%** accuracy (leave-one-out
Gaussian; 194 errors in 1050; 126 of them the 4↔5 pair).  Across the seven
labels in scope it falls to **52.6%**, and across all 17 labels to **23.7%**.

Duration is nevertheless **excluded from the score**, by decision: at correction
time the segment boundaries are themselves the suspect quantity, so duration is
contaminated by the very error being detected. A detector that leans on duration
would agree with the annotator precisely when the annotator drew the boundaries
wrong.

Consequence: the 81.5% figure is **context, not a target**. The sandbox reports
it so the reader knows what the easy signal is worth, but correlation is not
required to beat it. The metric that matters is the 4↔5 error rate using
spectral evidence alone.

## Non-goals

- No DTW. Removed by decision: DTW minimizes dissimilarity by warping, so it
  flattens scores for exactly the bad instances being hunted, and its freedom
  lets a 5 deform into a 6. Cross-correlation over lags handles translation.
- No GUI. The sandbox prints matrices and writes figures.
- No integration with `error_correction.py` until the numbers justify it.

## Architecture

`sandbox/xcorr/` — four modules, each independently runnable.

```
cache.py       annotation + audio  ->  patches.npz        (slow, run once)
templates.py   patches.npz         ->  templates.npz      (ZNCC-medoid)
scoring.py     pure functions, no I/O, no plotting        (the graduating core)
experiment.py  patches + templates ->  matrices, metrics
viz.py         a case               ->  a figure
```

The split exists so that the external drive is read exactly once. Every later
iteration reads `.npz` from the scratchpad. Nothing else in the repo imports
from `sandbox/`.

### cache.py

Walk the annotation metadata first and decide which instances are needed
*before* touching audio — the same discipline as
`dtw_suggestions.plan_instance_selection`, for the same reason: only files
holding a wanted instance are ever read.

Spectrogram for the scoring path is **raw**, not the viewer's normalized form:

```
S = log1p(|S1|^2 + |S2|^2)      # DPSS two-taper, NW=4, Kmax=2, nfft=512,
                                # hop 1 ms — same parameters as the viewer
```

No percentile clipping. The viewer's `_compute_spectrogram_norm` clips at the
5th/96th percentile **over the whole recording**, which is non-linear and
recording-dependent: a loud file's quiet syllables are crushed toward 0 before
any comparison happens, and ZNCC's z-scoring cannot undo a clip. ZNCC supplies
all the normalization this path needs.

Output `patches.npz`: concatenated float32 patches + per-patch offsets, plus a
metadata table (`date`, `file_name`, `t_start`, `t_end`, `label`, `n_cols`).

### templates.py

Template duration: `min(200 ms, mean_duration − std)` per label. On this data
that is 200 ms for every label in scope (the `mean − std` term only binds for
pathologically short labels such as 27, at 107 ms).

Selection, per label:

1. Draw 30 phrases at random.
2. Cut a centred window of the template duration from each — 30 candidates.
3. Score each candidate by the **mean peak-ZNCC of that candidate slid across
   the other 29 full phrases**. Scoring candidate-against-full-phrase rather
   than candidate-against-candidate measures the exact quantity used at test
   time, and being a maximum over lags it is indifferent to where in the
   rendition cycle each window happened to land.
4. Keep the winner. Also keep ranks 2 and 3 behind a `--n-templates` switch.

Ranks 2 and 3 exist to test for **bimodal renditions**, a known birdsong
pattern: if a label has two context-dependent forms, a single medoid lands in
one mode and flags the entire other mode as errors. The sandbox reports how
often an alternate wins, which settles empirically whether that risk is real
here.

Templates are drawn only from files in the template split (see below).

### scoring.py — the core that graduates

Pure functions, no I/O, no plotting, no Tk. This is the module that later moves
into `cannotate/`.

```python
band_limit(S, F, fmin, fmax) -> S'
zncc_surface(phrase, template) -> np.ndarray      # one ZNCC per time lag
peak_psr(surface, template_cols) -> (peak, peak_lag, psr)
score_phrase(phrase, templates_by_label) -> {label: (peak, psr)}
```

**`zncc_surface`** computes ZNCC at *every* lag, stride one frame. Sliding, not
tiling: a fixed non-overlapping tiling would make the score a function of where
the phrase onset fell modulo the template length — a rendition straddling a
chunk boundary would score badly in both chunks despite being perfect.

Exact and O(n log n), with `T` the template (`nf × m`) and `P` the phrase
(`nf × n`):

```
Tc   = T - mean(T);  normT = sqrt(sum(Tc**2))
num[k] = sum_f correlate(P[f], Tc[f], mode='valid')[k]
S1[k], S2[k] = windowed sums of P and P**2      # via cumsum along time
N      = nf * m
den[k] = sqrt(S2[k] - S1[k]**2 / N) * normT
ZNCC[k] = num[k] / den[k]
```

The window-mean term vanishes from the numerator because `sum(Tc) == 0`, which
is why only `P` needs running sums.

**`peak_psr`** returns the peak, its lag, and the peak-to-sidelobe ratio:

```
peak      = max(ZNCC)
sidelobe  = ZNCC outside +/- m lags of argmax
psr       = (peak - mean(sidelobe)) / std(sidelobe)
```

PSR measures **distinctiveness**, not similarity — a flat or noisy patch can hit
a respectable peak ZNCC by accident but produces no sharp peak. It requires
enough lags to estimate a sidelobe: phrases with `n < 3m` (under ~600 ms here)
yield `psr = nan` and are reported as **"insufficient support"**, a named
outcome rather than a silent skip.

**Frequency band** is derived per label from the template's own row-energy
percentiles (5th–95th) rather than one global `(min_freq, max_freq)`. A single
global band handicaps whichever label is furthest from the average — a low
harmonic stack and a high whistle do not share a useful band. A global override
and a sensitivity sweep are both exposed.

**Score vector and decision:**

```
score[L]  = max over that label's templates of peak ZNCC
predicted = argmax_L score[L]
margin    = score[assigned] - max_{L != assigned} score[L]
```

The margin is the production statistic. Scoring one patch against all candidate
templates makes every nuisance factor — recording gain, noise floor, bird
condition — common-mode across the three comparisons, so it cancels in the
difference. This is what makes confusion-hunting tractable without absolute
calibration.

### experiment.py

**Splits.** Template and test instances come from **disjoint files**, and the
primary split is **by date** (7 available), not by file: phrases within one
recording session share bird state and background, so a file-level split still
leaks. A file-level split is reported alongside as a robustness check — if the
two disagree substantially, session effects are carrying the score.

**Outputs:**

1. Confusion matrix, rows = verified label, columns = argmax.
2. Margin histogram. The shape is the finding:
   - **bimodal with a clean valley** — it works; the valley is the flagging
     threshold and the near-zero band becomes an honest "uncertain" class;
   - **unimodal straddling zero** — no discriminative power; any threshold flags
     an arbitrary slice.
3. The duration-only matrix, as context (see "timing is not a feature").
4. Per-pair error rates, with **4↔5 as the headline**. 4↔6 is nearly solved by
   any signal and is not informative.
5. Counts by outcome class: scored, insufficient support, no template.
6. How often template ranks 2/3 win (the bimodality probe).

### viz.py

One figure per case: the phrase spectrogram; the three templates; the three ZNCC
surfaces with peak and PSR marked; the score vector. Sampled across both
diagonal and off-diagonal cells, because the off-diagonal cases are the ones a
human must adjudicate and the figure is how they do it.

## Parameters

| parameter | default | note |
|---|---|---|
| `nfft` | 512 | matches the rest of the codebase |
| hop | 1 ms | `noverlap = nfft - 0.001*fs` |
| taper | DPSS NW=4, Kmax=2 | matches the viewer |
| spectrogram | `log1p(power)`, no clip | deliberate divergence from the viewer |
| template duration | `min(200ms, mean-std)` | 200 ms for 4/5/6 |
| candidates per label | 30 | |
| templates kept | 1 (switch: 3) | bimodality probe |
| frequency band | per-label, 5th–95th pct of template row energy | global override available |
| PSR exclusion | ±m lags | |
| minimum phrase | `n >= 3m` | else "insufficient support" |
| split | by date | by-file reported as robustness check |

## Results

Measured with templates and test phrases from disjoint recording dates, over
human-verified annotations.  Full detail in `sandbox/xcorr/README.md`.

| run | labels | accuracy |
|---|---|---|
| peak ZNCC | 7 | 100.0% (0 / 788) |
| peak ZNCC | **17** | **95.6%** (48 / 1102) |
| peak ZNCC, file split | 17 | 95.7% |
| peak ZNCC, 3 templates | 17 | 95.9% |
| peak ZNCC + PSR tie-break | 17 | 96.5% |
| duration only (context) | 17 | 23.7% |
| shuffled templates (control) | 17 | 0.0% |

Against the four questions below: (1) yes, by a wide margin, and without using
duration; (2) yes — the 17-label margin histogram is bimodal with a valley near
+0.15 and 4.4% negative margins; (3) yes, alternate templates win 656 of 1102
cases, so within-label variation is real, though exploiting it buys only +0.3
points; (4) the dominant error, 3→26, is a method error with an identifiable
cause, described next.

### Peak ZNCC rewards the least specific template

The one finding that contradicts the design as written.  A label-3 rendition is
an onset transient followed by a sustained harmonic stack; template 26 is only
a sustained stack — a strict **subset** of what a label-3 phrase contains — so
it wins on peak (0.958 vs 0.761) despite being the wrong syllable.

PSR detects this (1.3 against template 3's 2.9: the generic template matches at
many lags, so its peak does not stand out), which is a point in favour of
computing the surface rather than an aligned scalar.  But PSR cannot be
promoted to the primary statistic — ranking by it alone gives **0.4%**, since
its scale tracks template type more than match quality.  As a near-tie breaker
it trades 3→26 (26 cases down to 11) against 18→1 (14 up to 18) for a net +0.9
points that is unstable across the threshold, i.e. within noise.  It ships as
an off-by-default option; the real fix is likely a template *selected for
specificity* rather than a cleverer scalar.

## Success criteria

The sandbox has done its job when it can answer:

1. Is the 4↔5 confusion rate materially below what duration achieves, using
   spectral evidence alone?
2. Is the margin histogram bimodal with a usable valley?
3. Do alternate templates ever win — i.e. is any label in scope bimodal?
4. Eyeballing ~20 off-diagonal cases: are they annotation errors, method
   errors, or genuinely ambiguous?

A negative answer is a successful outcome. The point is to find out cheaply.

## Risks

| risk | status |
|---|---|
| Circularity: templates inherit annotation errors | **resolved** — the corpus is fully human-verified |
| Tiling makes score a function of onset phase | **resolved** — sliding at stride 1 |
| DTW flattens scores for bad instances | **resolved** — DTW removed |
| PSR undefined on an aligned pair | **resolved** — ZNCC computed over lags, so a surface exists |
| File-level percentile clipping biases cross-file comparison | **resolved** — raw `log1p`, no clip |
| Absolute thresholds do not transfer across labels | **resolved** — margin is relative, nuisance factors cancel |
| One global frequency band handicaps some labels | **resolved** — per-label band |
| Bimodal renditions break a single medoid | **mitigated** — top-3 switch measures it |
| Session effects inflate accuracy | **mitigated** — date-level split, file-level as check |
| 4/5/6 may be too easy to be informative | **confirmed** — a closed set of 7 gives 100%; only the 17-label number is meaningful |
| Session effects inflate accuracy | **measured** — date split 95.6%, file split 95.7%; no session effect |
| Bimodal renditions break a single medoid | **measured** — alternates win 656/1102, but worth only +0.3 points |
| Peak ZNCC favours non-specific templates | **new, open** — the dominant residual error; see Results |
| Cost on the full corpus | **partly measured** — 1117 phrases x 17 labels in ~30 s; caching 251 wavs takes ~4 min |

## Deferred

Not built now, but the design leaves room:

- **Boundary errors.** The ZNCC surface's peak *positions* locate renditions, so
  comparing peak count and spacing against the annotated span detects
  over/under-splitting. This is timing information, excluded from the *score* by
  the decision above, but valid as a separate detector where it is the signal
  rather than a confound.
- **Vote disagreement within a phrase.** Each rendition votes independently; a
  phrase voting `5,5,6,5` means something different from one voting `6,6,6,6`.
  Available free once per-peak scoring exists.
- **Atypical-rendition detection.** Needs within-label score calibration
  (z-score per label, as `DurationOutlierDetector` already does), not margins.
- **Short-segment class.** Negligible in this verified file (3/0/0), but the
  auto-annotated corpus has 583 sub-100 ms label-4 segments against a 908 ms
  median — a real population that must be surfaced, never silently skipped.
