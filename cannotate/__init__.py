"""
cannotate — package housing the annotation GUI components.

Modules
-------
cannotate_utils    : shared helpers, constants, pretty-sonogram
navigator          : FullEntropyNavigator pop-up
spectrogram_viewer : IntegratedSpectrogramViewer class
spectrograms       : the viewer's display spectrogram
template_store     : the per-bird template bank on disk
xcorr_suggestions  : ranking a selection against that bank
template_browser   : template concatenation browser
error_correction   : outlier-phrase detection and correction browser
xcorr              : cross-correlation scoring of phrases against templates
"""

from .cannotate_utils import (          # noqa: F401
    generate_pretty_sonogram,
    clean_segments,
    binary_to_segments,
    segments_to_binary,
    read_wav,
    read_wav_channels,
    NN_MODEL_ARCHITECTURES,
    NN_WEIGHTS_DIR,
    ONNX_DEFAULT_WEIGHTS,
    NN_AVAILABLE,
    NN_IMPORT_ERROR,
)
from .navigator import FullEntropyNavigator          # noqa: F401
from .spectrogram_viewer import (                    # noqa: F401
    IntegratedSpectrogramViewer,
    regenerate_disk_templates,
)
from .template_browser import open_template_browser  # noqa: F401
from .error_correction import open_error_correction_browser  # noqa: F401
from . import spectrograms                           # noqa: F401
from . import template_store                         # noqa: F401
from . import xcorr_suggestions                      # noqa: F401
