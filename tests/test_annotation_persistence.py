"""Writing one song's annotation into the bird's JSON.

This is the step the viewer's Save button and the viewer's close path both run.
It used to exist only as a block of code inside the close path, which is why
pressing Save was impossible without duplicating it -- and why nothing was on
disk until the window shut.

The entry is looked up by file name rather than by a remembered index: marking
a song verified can create its entry while the viewer is open, so an index
captured beforehand goes stale.

Runs under pytest or directly:  python tests/test_annotation_persistence.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from Cannotate_GUI import write_annotation_entry
from utils import HUMAN_VERIFIED_KEY

DATE = "2024-07-05"
WAV = "bird_0001_2024_07_05_06_58_44.wav"


@pytest.fixture
def annotation_file(tmp_path):
    return tmp_path / "bird_annotation_tester.json"


def test_a_song_with_no_entry_yet_gets_one(annotation_file):
    db = {}
    write_annotation_entry(db, DATE, WAV, [(0.1, 0.4, 3)], True, annotation_file)

    on_disk = json.loads(annotation_file.read_text())
    assert [e["file_name"] for e in on_disk[DATE]] == [WAV]
    assert on_disk[DATE][0]["segments"] == [[0.1, 0.4, 3]]


def test_saving_twice_replaces_the_entry_instead_of_appending(annotation_file):
    db = {}
    write_annotation_entry(db, DATE, WAV, [(0.1, 0.4, 3)], True, annotation_file)
    write_annotation_entry(db, DATE, WAV, [(0.2, 0.9, 7)], True, annotation_file)

    on_disk = json.loads(annotation_file.read_text())
    assert len(on_disk[DATE]) == 1, "a second save must not duplicate the song"
    assert on_disk[DATE][0]["segments"] == [[0.2, 0.9, 7]]


def test_the_verification_flag_is_written_with_the_segments(annotation_file):
    db = {}
    write_annotation_entry(db, DATE, WAV, [(0.1, 0.4, 3)], False, annotation_file)

    on_disk = json.loads(annotation_file.read_text())
    assert on_disk[DATE][0][HUMAN_VERIFIED_KEY] is False


def test_other_songs_on_the_same_date_are_left_alone(annotation_file):
    other = "bird_0002_2024_07_05_07_01_12.wav"
    db = {DATE: [{"file_name": other, HUMAN_VERIFIED_KEY: True,
                  "segments": [[5.0, 5.5, 1]]}]}
    write_annotation_entry(db, DATE, WAV, [(0.1, 0.4, 3)], True, annotation_file)

    on_disk = json.loads(annotation_file.read_text())
    by_name = {e["file_name"]: e for e in on_disk[DATE]}
    assert by_name[other]["segments"] == [[5.0, 5.5, 1]]
    assert by_name[WAV]["segments"] == [[0.1, 0.4, 3]]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
