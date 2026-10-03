"""Saving from inside the annotator window.

Until this button existed, the annotation JSON was only written when the viewer
window *closed*: ``Cannotate_GUI`` calls ``viewer_win.wait_window()`` and
persists afterwards. An annotator who worked for an hour had nothing on disk
until they shut the window, and a crash took the session with it.

The viewer still never touches the annotation file itself -- the owner of the
file decides how to persist, exactly as with ``verified_callback``. The button
just hands over what the viewer owns: the segments and the verification flag.

Needs a display (it builds real Tk widgets) and is skipped without one.

Runs under pytest or directly:  python tests/test_viewer_save_button.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import pytest
import tkinter as tk

from cannotate.spectrogram_viewer import IntegratedSpectrogramViewer


@pytest.fixture
def tk_root():
    try:
        root = tk.Tk()
    except tk.TclError as e:                                  # pragma: no cover
        pytest.skip(f"no display: {e}")
    root.withdraw()
    yield root
    root.destroy()


def _viewer(tk_root, **kwargs):
    window = tk.Toplevel(tk_root)
    window.withdraw()
    return IntegratedSpectrogramViewer(window, **kwargs)


def test_save_button_hands_the_callback_the_segments_and_the_verified_flag(tk_root):
    handed = []
    view = _viewer(tk_root, save_callback=lambda segs, verified: handed.append((segs, verified)))
    view.segments = [(0.0, 0.5, 3), (0.9, 1.4, 7)]
    view.set_human_verified(True)

    view.btn_save.invoke()

    assert handed == [([(0.0, 0.5, 3), (0.9, 1.4, 7)], True)]


def test_save_button_is_disabled_when_nobody_owns_the_annotation_file(tk_root):
    view = _viewer(tk_root)
    assert str(view.btn_save.cget("state")) == "disabled"


def test_save_button_is_enabled_when_a_callback_is_given(tk_root):
    view = _viewer(tk_root, save_callback=lambda segs, verified: None)
    assert str(view.btn_save.cget("state")) == "normal"


def test_a_successful_save_is_reported_in_the_window(tk_root):
    """A non-programmer needs to see that the click did something."""
    view = _viewer(tk_root, save_callback=lambda segs, verified: None)
    assert view.lbl_save_status.cget("text") == ""

    view.btn_save.invoke()

    assert "Saved" in view.lbl_save_status.cget("text")


def test_a_failed_save_does_not_claim_to_have_saved(tk_root):
    """Reporting success for a write that raised is worse than reporting nothing.

    ``save_now`` is called directly rather than through ``btn_save.invoke()``:
    Tk traps exceptions raised inside a widget command and routes them to
    ``report_callback_exception``, so invoking the button would hide the very
    propagation this test is about.
    """
    def exploding_save(segs, verified):
        raise OSError("disk full")

    view = _viewer(tk_root, save_callback=exploding_save)
    with pytest.raises(OSError):
        view.save_now()

    assert "Saved" not in view.lbl_save_status.cget("text")


@pytest.mark.parametrize("sequence", ["<Control-s>", "<Command-s>"])
def test_save_is_reachable_from_the_keyboard(tk_root, sequence):
    """A bare "s" is already segment mode, so Save takes the modifier form.

    Both are bound because the annotator is on macOS, where Cmd-S is the
    reflex, while the same code ships to Windows machines where it is Ctrl-S.
    """
    saved = []
    view = _viewer(tk_root, save_callback=lambda segs, verified: saved.append(segs))

    assert view.root.bind(sequence), f"{sequence} is not bound"

    # Tk only delivers a generated key event to a window that is actually
    # mapped, so the withdrawn test window has to be shown for this one check.
    view.root.deiconify()
    view.root.update()
    view.root.focus_force()
    try:
        view.root.event_generate(sequence, when="now")
    finally:
        view.root.withdraw()
    assert saved, f"{sequence} did not reach save_now"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
