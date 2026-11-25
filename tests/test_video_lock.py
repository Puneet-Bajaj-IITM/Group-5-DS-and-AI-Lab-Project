"""The time-based video lock (video.pooled_lock) on made-up frames."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from serpentguard import video  # noqa: E402

N_CLASSES, CONF, PEAK = 184, 0.29, 0.9
A, B = 1, 0


def sure(species, p=PEAK):
    probs = np.full(N_CLASSES, (1 - p) / (N_CLASSES - 1))
    probs[species] = p
    return probs


def run(frames, fps, **settings):
    times = np.arange(len(frames)) / fps
    return video.pooled_lock(frames, times, CONF, **settings), times


def first_lock(steps, times):
    return next((float(t) for s, t in zip(steps, times) if s["state"] == "locked"), None)


@pytest.mark.parametrize("fps", [10, 2.5, 1.25])
def test_window_is_in_seconds(fps):
    steps, times = run([sure(A)] * int(6 * fps), fps, window_seconds=1.0)
    assert 1.0 - 1e-9 <= first_lock(steps, times) <= 1.8


def test_species_holds_through_a_short_rival():
    steps, _ = run([sure(A)] * 15 + [sure(B, 0.95)] * 5 + [sure(A)] * 15, 10)
    assert {s["species"] for s in steps if s["species"] is not None} == {A}


def test_rival_for_switch_time_switches_once():
    steps, _ = run([sure(A)] * 20 + [sure(B)] * 40, 10)
    shown = [s["species"] for s in steps if s["species"] is not None]
    assert sum(a != b for a, b in zip(shown, shown[1:])) == 1
    assert steps[-1]["species"] == B
    assert [s["event"] for s in steps if s["event"]] == ["lock", "switch"]


def test_hold_expiry_drops_the_lock():
    gap = int(video.HOLD_SECONDS * 10) + 2
    steps, _ = run([sure(A)] * 20 + [None] * gap + [sure(B)] * 20, 10)
    assert steps[20]["species"] == A
    assert steps[20 + gap - 1]["state"] == "checking"
    assert steps[20 + gap]["state"] == "checking"
    assert steps[-1]["species"] == B and [s["event"] for s in steps if s["event"]] == ["lock", "lock"]


def test_gate_share_blocks_sparse_snakes():
    frames = [sure(A) if i % 2 == 0 else None for i in range(60)]
    assert all(s["state"] == "checking" for s in run(frames, 10, gate_share=0.8)[0])
    assert run(frames, 10)[0][-1]["state"] == "locked"


def test_no_snake_never_locks():
    assert all(s["state"] == "checking" for s in run([None] * 40, 10)[0])


def test_unknown_setting_is_refused():
    with pytest.raises(TypeError):
        video.pooled_lock([sure(A)], [0.0], CONF, window=5)


def test_slow_phone_still_pools_min_frames():
    steps, times = run([sure(A)] * 6, 1 / 0.8, window_seconds=0.5, min_frames=2)
    assert first_lock(steps, times) == pytest.approx(0.8)
    steps, _ = run([sure(A)] * 6, 1 / 0.8, window_seconds=0.5, min_frames=3)
    assert steps[1]["state"] == "checking" and steps[2]["state"] == "locked"
