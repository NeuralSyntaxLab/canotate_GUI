"""
cannotate.xcorr — cross-correlation scoring of phrases against label templates.

Graduated from ``sandbox/xcorr/`` after the validation experiment (see
``docs/design/2026-09-14-xcorr-validation-design.md`` and the sandbox README):
95.6% agreement with human labels across 17 labels on a date-level split,
0.0% with the template-to-label mapping shuffled.

Modules
-------
scoring : ZNCC surfaces, ``TemplateBank``, peak/PSR, the decision rule
spectro : the two-taper log-power spectrogram this path scores on
bank    : fitting, saving and loading ``templates.npz``
cli     : ``python -m cannotate.xcorr fit --annotations ... --out t.npz``

Nothing here imports tkinter or matplotlib: a scan worker loads ``scoring``
and ``spectro`` only, and pays for numpy and scipy alone.
"""
