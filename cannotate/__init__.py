"""
cannotate — package housing the annotation GUI components.

Modules
-------
cannotate_utils    : shared helpers, constants, pretty-sonogram
navigator          : FullEntropyNavigator pop-up
spectrogram_viewer : IntegratedSpectrogramViewer class
dtw_suggestions    : DTW template logic
template_browser   : template concatenation browser
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
from . import dtw_suggestions                        # noqa: F401
