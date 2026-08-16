"""nn_segmentation — Neural network based segmentation and annotation utilities."""

from bird_song_detector.nn_segmentation.segmentation_utils import (
    merge_annotations,
    _merge_close_segments,
    _merge_segment_lists,
    _compute_overlap_ratio,
    _get_assential_info_from_filename,
    match_names,
)
from bird_song_detector.nn_segmentation.inference_to_annotation_data import InferenceToAnnotationData

__all__ = [
    "InferenceToAnnotationData",
    "merge_annotations",
    "_merge_close_segments",
    "_merge_segment_lists",
    "_compute_overlap_ratio",
    "_get_assential_info_from_filename",
    "match_names",
]
