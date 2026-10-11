import numpy as np

from forgesight.zones import (
    ZoneMonitor,
    foot_point,
    point_in_polygon,
    polygon_to_pixels,
    validate_polygon,
)

SQUARE = [[50, 50], [100, 50], [100, 100], [50, 100]]


def run(monitor, sequence, key=1, fps=10.0):
    """Feed a sequence of True/False/None (None = not detected) for one key."""
    events = []
    for i, state in enumerate(sequence):
        obs = {} if state is None else {key: state}
        events += monitor.update(i / fps, i, obs)
    return events


def test_polygon_scaling_and_point_test():
    poly = polygon_to_pixels(SQUARE, 200, 100)
    assert np.allclose(poly[0], [100, 50])
    assert point_in_polygon(poly, 150, 75)
    assert not point_in_polygon(poly, 50, 75)


def test_foot_point_is_bottom_centre():
    assert foot_point([10, 20, 30, 80]) == (20.0, 80.0)


def test_validate_polygon():
    assert validate_polygon(SQUARE)[0]
    assert not validate_polygon(SQUARE[:2])[0]
    assert not validate_polygon([[0, 0], [10, 10], [20, 20]])[0]  # zero area
    assert not validate_polygon([[0, 0], [100, 100], [100, 0], [0, 100]])[0]  # bow-tie
    assert not validate_polygon([[0, 0], [120, 0], [50, 50]])[0]  # out of frame


def test_single_noisy_frame_does_not_trigger():
    m = ZoneMonitor(entry_frames=2, exit_frames=3)
    assert run(m, [False, True, False, False, True, False]) == []


def test_staying_inside_creates_exactly_one_event():
    m = ZoneMonitor(entry_frames=2, exit_frames=3)
    events = run(m, [False] + [True] * 30)
    assert len(events) == 1
    assert events[0].entry_t == 0.1  # first inside frame, not the confirmation frame
    assert events[0].confirmed_t == 0.2


def test_brief_step_out_is_not_a_reentry():
    m = ZoneMonitor(entry_frames=2, exit_frames=3)
    events = run(m, [True] * 5 + [False] * 2 + [True] * 5)
    assert len(events) == 1


def test_confirmed_exit_then_reentry_creates_second_event():
    m = ZoneMonitor(entry_frames=2, exit_frames=3)
    events = run(m, [True] * 5 + [False] * 3 + [True] * 5)
    assert len(events) == 2
    assert m.visits[0].exit_t == 0.4
    assert m.visits[0].exit_reason == "left zone"


def test_lost_track_closes_visit():
    m = ZoneMonitor(entry_frames=2, exit_frames=3, lost_frames=4)
    run(m, [True] * 4 + [None] * 4)
    assert m.visits[0].exit_reason == "track lost"
    assert m.visits[0].dwell_s == 0.3


def test_short_occlusion_keeps_visit_open():
    m = ZoneMonitor(entry_frames=2, exit_frames=3, lost_frames=10)
    events = run(m, [True] * 4 + [None] * 3 + [True] * 4)
    assert len(events) == 1
    assert m.visits[0].exit_t is None


def test_independent_tracks():
    m = ZoneMonitor(entry_frames=2, exit_frames=3)
    new = []
    for i in range(5):
        new += m.update(i * 0.1, i, {1: True, 2: i >= 2})
    assert sorted(v.key for v in new) == [1, 2]


def test_finish_closes_open_visits():
    m = ZoneMonitor(entry_frames=1, exit_frames=3)
    run(m, [True] * 5)
    m.finish(0.4)
    assert m.visits[0].exit_reason == "video ended"
    assert m.visits[0].dwell_s == 0.4
