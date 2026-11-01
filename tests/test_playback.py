"""Playback volume and the window playhead in the spectrogram viewer.

Some recordings were nearly inaudible on laptop speakers, so every sound now
passes through a volume / auto-level stage; and playing the window shows a
playhead that can be paused, resumed, sought and stopped.

The viewer tests need a display (they build real Tk widgets) and are skipped
without one. ``sounddevice`` is replaced by a recorder, and the clock by a
counter, so nothing is actually played and nothing waits.

Runs under pytest or directly:  python tests/test_playback.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest
import tkinter as tk

from cannotate import spectrogram_viewer as sv
from cannotate.spectrogram_viewer import (
    IntegratedSpectrogramViewer, playback_signal, AUTO_LEVEL_PEAK,
)


# ── playback_signal ──────────────────────────────────────────────────

def test_int16_is_scaled_to_unit_range():
    out = playback_signal(np.array([0, 16384, -32768], dtype=np.int16))
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, [0.0, 0.5, -1.0])


def test_float_holding_int16_values_is_scaled_not_clipped():
    # read_wav_channels hands stereo songs over as float64 int16 values.
    out = playback_signal(np.array([0.0, 3276.8, -16384.0]))
    np.testing.assert_allclose(out, [0.0, 0.1, -0.5], rtol=1e-6)


def test_unit_float_is_left_alone():
    np.testing.assert_allclose(playback_signal(np.array([0.25, -0.5])),
                               [0.25, -0.5])


def test_volume_multiplies_and_clips():
    x = np.array([0.1, -0.3], dtype=np.float32)
    np.testing.assert_allclose(playback_signal(x, volume=3.0), [0.3, -0.9],
                               rtol=1e-6)
    np.testing.assert_allclose(playback_signal(x, volume=10.0), [1.0, -1.0])


def test_auto_level_brings_a_quiet_signal_up_to_the_target_peak():
    quiet = np.array([0, 100, -200], dtype=np.int16)
    out = playback_signal(quiet, auto_level=True)
    assert np.max(np.abs(out)) == pytest.approx(AUTO_LEVEL_PEAK)


def test_auto_level_against_the_file_peak_keeps_quiet_parts_quiet():
    segment = np.array([0, 100], dtype=np.int16)
    out = playback_signal(segment, auto_level=True, ref_peak=1000)
    assert out[1] == pytest.approx(0.1 * AUTO_LEVEL_PEAK)


def test_silence_survives_auto_level():
    out = playback_signal(np.zeros(4, dtype=np.int16), auto_level=True)
    assert not np.any(out)


# ── The viewer ───────────────────────────────────────────────────────

class FakeSD:
    def __init__(self):
        self.plays = []
        self.stops = 0

    def play(self, data, samplerate, device=None):
        self.plays.append((np.array(data), samplerate))

    def stop(self):
        self.stops += 1


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


FS = 8000


@pytest.fixture
def env(monkeypatch, tmp_path):
    fake_sd, clock = FakeSD(), FakeClock()
    monkeypatch.setattr(sv, "sd", fake_sd)
    monkeypatch.setattr(sv.time, "monotonic", clock)
    monkeypatch.setattr(sv, "_SETTINGS_FILE", str(tmp_path / "settings.json"))
    try:
        root = tk.Tk()
    except tk.TclError as e:                                  # pragma: no cover
        pytest.skip(f"no display: {e}")
    root.withdraw()
    view = IntegratedSpectrogramViewer(tk.Toplevel(root), None)
    # 3 s of a quiet ramp; peak is 1000 of int16's 32768.
    y = (np.linspace(-1, 1, 3 * FS) * 1000).astype(np.int16)
    view.load_existing_data(y, FS, "song.wav")
    view.current_window_start = 0.0
    view.window_duration = 1.0
    view.update_plots()
    yield view, fake_sd, clock, tmp_path / "settings.json"
    root.destroy()


def test_play_starts_the_playhead_at_the_window_start(env):
    view, sd, clock, _ = env
    view.play_current_window()
    assert view._playing
    data, rate = sd.plays[-1]
    assert rate == FS and len(data) == FS
    assert view._play_pos == pytest.approx(0.0)
    assert view.btn_play_pause.cget("text").endswith("Pause")


def test_playhead_follows_the_clock_and_is_drawn(env):
    view, _, clock, _ = env
    view.play_current_window()
    clock.now += 0.25
    view._play_tick()
    assert view._play_pos == pytest.approx(0.25)
    assert view.seek_var.get() == pytest.approx(0.25)
    assert "0.250" in view.lbl_play_time.cget("text")
    assert all(line.get_visible() and line.get_xdata()[0] == pytest.approx(0.25)
               for line in view._playhead_lines)


def test_pause_keeps_the_position_and_play_resumes_from_it(env):
    view, sd, clock, _ = env
    view.play_current_window()
    clock.now += 0.4
    view.pause_playback()
    assert not view._playing
    assert view._play_pos == pytest.approx(0.4)

    clock.now += 5        # time passing while paused moves nothing
    view.start_playback()
    data, _ = sd.plays[-1]
    assert len(data) == pytest.approx(0.6 * FS, abs=1)
    assert view._play_pos == pytest.approx(0.4)


def test_stop_rewinds(env):
    view, _, clock, _ = env
    view.play_current_window()
    clock.now += 0.4
    view.stop_playback()
    assert view._play_pos is None and not view._playing
    assert not any(line.get_visible() for line in view._playhead_lines)


def test_reaching_the_end_of_the_window_rewinds(env):
    view, _, clock, _ = env
    view.play_current_window()
    clock.now += 1.5
    view._play_tick()
    assert view._play_pos is None and not view._playing


def test_slow_motion_moves_the_playhead_at_its_speed(env):
    view, sd, clock, _ = env
    view.slowmo_speed = 0.5
    view.play_slowmo_window()
    assert sd.plays[-1][1] == FS // 2
    clock.now += 1.0
    view._play_tick()
    assert view._play_pos == pytest.approx(0.5)


def test_seeking_sets_where_play_starts(env):
    view, sd, _, _ = env
    view._on_seek(0.75)
    view.start_playback()
    assert len(sd.plays[-1][0]) == pytest.approx(0.25 * FS, abs=1)


def test_seeking_while_playing_jumps_there(env):
    view, sd, _, _ = env
    view.play_current_window()
    view._on_seek(0.5)
    assert view._playing
    assert len(sd.plays[-1][0]) == pytest.approx(0.5 * FS, abs=1)


def test_moving_the_window_away_drops_the_playhead(env):
    view, sd, clock, _ = env
    view.play_current_window()
    clock.now += 0.1
    view._play_tick()
    view.current_window_start = 2.0
    view.update_plots()
    assert view._play_pos is None and not view._playing


def test_space_toggles_play_and_pause(env):
    view, _, _, _ = env
    view.toggle_play_pause()
    assert view._playing
    view.toggle_play_pause()
    assert not view._playing and view._play_pos is not None


def test_volume_is_applied_to_what_is_played(env):
    view, sd, _, _ = env
    view.auto_level_var.set(False)
    view.volume_var.set(200)
    view.play_current_window()
    data, _ = sd.plays[-1]
    assert np.max(np.abs(data)) == pytest.approx(2 * 1000 / 32768, rel=1e-3)


def test_auto_level_uses_the_whole_file_peak(env):
    view, sd, _, _ = env
    view.auto_level_var.set(True)
    view.volume_var.set(100)
    view.current_window_start = 1.4      # middle of the ramp: quiet part
    view.update_plots()
    view.play_current_window()
    data, _ = sd.plays[-1]
    assert np.max(np.abs(data)) < AUTO_LEVEL_PEAK * 0.8


def test_template_playback_stops_the_playhead(env):
    view, _, _, _ = env
    view.play_current_window()
    view._play_audio_segment(np.zeros(10, dtype=np.int16), FS)
    assert not view._playing


def test_volume_and_auto_level_are_remembered(env):
    view, _, _, settings = env
    view.volume_var.set(350)
    view._on_volume_changed()
    view._save_volume()
    view.auto_level_var.set(False)
    view._on_auto_level_toggled()
    saved = json.loads(settings.read_text())
    assert saved["playback_volume_percent"] == 350
    assert saved["playback_auto_level"] is False

    again = IntegratedSpectrogramViewer(tk.Toplevel(view.root.master), None)
    assert again.volume_var.get() == 350
    assert again.auto_level_var.get() is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
