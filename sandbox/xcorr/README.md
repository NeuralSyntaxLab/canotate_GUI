# Cross-correlation validation sandbox

Disposable experiment behind
`docs/design/2026-09-14-xcorr-validation-design.md`.  It answers one question:
**can spectrogram cross-correlation against per-label templates identify
mislabeled phrases?**

Nothing in the repo imports from here.  If the answer holds up, `scoring.py`
is the piece that graduates into `cannotate/` and fills in
`CrossCorrelationOutlierDetector`.

## Running it

```bash
python cache.py                  # ~4 min; reads the external drive once
python experiment.py             # ~30 s
python viz.py --only-errors      # figures for every disagreement
python -m pytest test_scoring.py -q
```

All seven requested labels (4, 5, 6, 9, 10, 15, 16) are the default.  To widen
to every label with at least 30 instances — which is the honest test, since
real error correction chooses among all of a bird's syllable types rather than
a hand-picked few:

```bash
export XCORR_LABELS=all XCORR_CACHE_DIR=/some/other/dir
python cache.py --max-per-label 120 && python experiment.py
```

Controls (each writes to its own results file, never the real one):

```bash
python experiment.py --shuffle-templates          # must collapse to ~0%
python experiment.py --global-band 1000 13000     # removes the per-label band
python experiment.py --split file                 # session-effect check
python experiment.py --n-templates 3              # bimodality probe
python experiment.py --tie-break-psr
```

## Results

Date-level split (templates and test phrases from disjoint recording dates),
human-verified annotations throughout.

| run | labels | accuracy |
|---|---|---|
| peak ZNCC, date split | 7 | 100.0% (0 / 788) |
| peak ZNCC, date split | 17 | **95.6%** (48 / 1102) |
| peak ZNCC, file split | 17 | 95.7% |
| peak ZNCC, 3 templates | 17 | 95.9% |
| peak ZNCC + PSR tie-break | 17 | 96.5% |
| duration only (context) | 17 | 23.7% |
| **shuffled templates (control)** | 17 | **0.0%** |

The seven-label result is 100% because a closed set of seven stereotyped
syllable types from one bird is an easy question.  **Quote the 17-label
number.**  The margin histogram at 17 labels is bimodal with a valley near
+0.15 and 4.4% negative margins, so a flagging threshold has somewhere
principled to sit.

### Why the 7-label 100% is believed

A result that good is usually a leak.  Three checks:

* templates and test phrases come from **disjoint dates** (2 template dates,
  5 test dates, up to 16 days apart);
* forcing a **single shared frequency band** leaves it at 100%, so the
  per-label band — derived from each label's own data and applied to the
  query — is not smuggling in the answer;
* **shuffling which label owns which template drops it to 0.0%**, so the
  pipeline genuinely reads template content.

`zncc_surface` is also checked against a brute-force implementation in
`test_scoring.py`.

### The failure mode worth knowing about

Nearly all remaining errors are three pairs: 3→26 (26 cases), 18→1 (14),
1→11 (5).  The figures explain 3→26, and it is a property of the *statistic*,
not of the data:

> A label-3 rendition is an onset transient followed by a sustained harmonic
> stack.  Template 26 is only a sustained stack — a strict **subset** of what
> a label-3 phrase contains.  So it matches at peak 0.958 against template 3's
> 0.761.

**Peak ZNCC rewards the least specific template.**  PSR sees through it: the
generic template matches at many lags, so its peak does not stand out (PSR 1.3
against template 3's 2.9).

PSR is nevertheless useless as a primary statistic — ranking by it alone gives
**0.4%**, because its scale depends on template type far more than on match
quality, and every multiplicative blend tested was worse than peak alone
(`peak*PSR` 68.5%, `peak*tanh(PSR/3)` 86.5%).  Consulted only among near-tied
peaks it cuts 3→26 from 26 cases to 11 — but raises 18→1 from 14 to 18, for a
net +0.9 points that is unstable across the threshold (96.1 / 96.1 / 96.5 /
96.0 / 96.3 / 95.5 / 94.2 as delta goes 0.02 → 0.20).  That is within noise on
1102 samples, so **it stays off by default**.  The clean fix is probably a
more specific template — one required to contain a transient — rather than a
cleverer scalar.

### Other observations

* With three templates per label, ranks 2 and 3 win 656 of 1102 cases.  That is
  strong within-label rendition variation, but it buys only +0.3 points, so a
  single medoid template is not the bottleneck.
* `insufficient_support` (phrase shorter than 3x the template, so PSR has too
  few lags for a sidelobe) covers ~12% of phrases.  They are still scored and
  still classified correctly; only their PSR is `nan`.

## Layout

| file | role |
|---|---|
| `config.py` | every tunable, and the data paths |
| `spectro.py` | raw `log1p` two-taper spectrogram (no percentile clip) |
| `cache.py` | annotations + audio -> `patches.npz`; reads the drive once |
| `templates.py` | 200 ms window per label, chosen by consistency |
| `scoring.py` | **the core that graduates** — ZNCC surface, PSR, decision |
| `experiment.py` | splits, confusion matrices, margin histogram, controls |
| `viz.py` | per-case figures: phrase, templates, ZNCC surfaces |
| `test_scoring.py` | ZNCC checked against a brute-force reference |

## Not answered here

The corpus used is human-verified throughout, so this measures whether the
method **agrees with correct labels** — which is the prerequisite.  It does not
yet show what the method flags on the *unverified* corpus, which is the actual
error-correction use case and the obvious next step.
