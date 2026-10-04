"""mk ref: statistics on synthetic landmark series with known answers (no network, no MediaPipe)."""
import argparse
import json

import cv2
import numpy as np
import pytest

from mkmmd.cli import ref as CLI
from mkmmd.core import perform as PF
from mkmmd.ref import body as BO
from mkmmd.ref import dsp
from mkmmd.ref import face as FA
from mkmmd.ref import measure as ME
from mkmmd.ref import pexels, store
from mkmmd.ref import recommend as RC
from mkmmd.ref import RefUsage, track as TR

FPS = 30.0
W, H = 1280, 720


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def pulse(t, t0, peak, rise, fall):
    """Asymmetric eyelid closure: smooth rise over `rise` s to the peak at t0, smooth return over `fall` s."""
    return peak * np.where(t <= t0, smoothstep((t - (t0 - rise)) / rise), 1 - smoothstep((t - t0) / fall))


def rot(yaw, pitch_up, roll):
    """Facial transformation matrix rotation for the conventions of face.head_angles (R = Ry(yaw) Rx(-pitch_up) Rz(roll))."""
    a, b, c = np.radians([yaw, -pitch_up, roll])
    ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    return ry @ rx @ rz


def lowpass_time(x, tau_s, fps=FPS):
    a = 1 - np.exp(-1 / (tau_s * fps))
    y, out = x[0], np.empty_like(x)
    for i, v in enumerate(x):
        y = y + a * (v - y)
        out[i] = y
    return out


# ============================================================================================ synthetic clip
def synth_track(seconds=60.0, blink_times=(), long_closure=None, look_down=None, nod=None, gaze=None, head_share=0.7, jaw=None,
                shoulders=None, hand=None, cam=None, cut_at=None, hop_at=None, seed=1):
    """A complete track dict (the arrays `track.py` writes) from known behaviour. nod = (deg, period s); gaze = list of
    (t, yaw deg, pitch deg) targets with 0.12 s transitions; jaw = (t0, t1, amplitude, Hz) talking burst; shoulders =
    (roll deg, period s); hand = (still-until s, moving-until s, amplitude px, period s); cam = (px, period s) pan."""
    rng = np.random.default_rng(seed)
    n = int(seconds * FPS)
    t = np.arange(n) / FPS
    # ---- eyes: lid closure
    closure = 0.06 + rng.normal(0, 0.01, n)
    for b in blink_times:
        closure = np.maximum(closure, 0.06 + pulse(t, b, 0.62, 0.07, 0.12))
    if long_closure:
        a, b = long_closure
        closure = np.maximum(closure, 0.6 * (smoothstep((t - a) / 0.2) * (1 - smoothstep((t - b) / 0.2))) + 0.06)
    if look_down:                                            # slow lid lowering of a downward glance, not a blink
        a, b = look_down
        closure = np.maximum(closure, 0.06 + 0.28 * smoothstep((t - a) / 1.0) * (1 - smoothstep((t - b) / 1.0)))
    # ---- gaze (head + eyes) and head pose
    G = np.zeros((n, 2))
    for (t0, yaw, pitch) in (gaze or []):
        w = smoothstep((t - t0) / 0.12)[:, None]
        G = G * (1 - w) + np.array([yaw, pitch]) * w
    head = np.column_stack([lowpass_time(np.r_[np.zeros(3), head_share * G[:-3, 0]], 0.12),
                            lowpass_time(np.r_[np.zeros(3), head_share * G[:-3, 1]], 0.12)])
    eyes = G - head
    pitch = head[:, 1] + 3.0
    if nod:
        pitch = pitch + nod[0] * np.sin(2 * np.pi * t / nod[1])
    yaw = head[:, 0]
    roll = 2.0 + rng.normal(0, 0.1, n)
    pitch, yaw = pitch + rng.normal(0, 0.1, n), yaw + rng.normal(0, 0.1, n)
    eyes = eyes + rng.normal(0, 0.3, eyes.shape)
    # ---- blendshapes
    bs = np.zeros((n, 52), np.float32)
    bs[:, FA.BS["eyeBlinkLeft"]] = bs[:, FA.BS["eyeBlinkRight"]] = closure
    ey, ep = eyes[:, 0] / FA.EYE_FULL_DEG, eyes[:, 1] / FA.EYE_FULL_DEG
    bs[:, FA.BS["eyeLookOutLeft"]], bs[:, FA.BS["eyeLookInRight"]] = np.clip(ey, 0, 1), np.clip(ey, 0, 1)
    bs[:, FA.BS["eyeLookInLeft"]], bs[:, FA.BS["eyeLookOutRight"]] = np.clip(-ey, 0, 1), np.clip(-ey, 0, 1)
    bs[:, FA.BS["eyeLookUpLeft"]], bs[:, FA.BS["eyeLookUpRight"]] = np.clip(ep, 0, 1), np.clip(ep, 0, 1)
    bs[:, FA.BS["eyeLookDownLeft"]], bs[:, FA.BS["eyeLookDownRight"]] = np.clip(-ep, 0, 1), np.clip(-ep, 0, 1)
    j = 0.01 + np.abs(rng.normal(0, 0.003, n))
    if jaw:
        t0, t1, amp, hz = jaw
        m = (t >= t0) & (t < t1)
        j = j + m * amp * (0.5 + 0.5 * np.sin(2 * np.pi * hz * t))
    bs[:, FA.BS["jawOpen"]] = j
    mat = np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))
    for i in range(n):
        mat[i, :3, :3] = rot(yaw[i], pitch[i], roll[i]) * 1.7              # scale must not matter
        mat[i, :3, 3] = (0.0, 0.0, -30.0)
    lm = np.zeros((n, 478, 3), np.float32)
    lm[:, :, 0], lm[:, :, 1] = 0.5, 0.4
    lm[:, 234, 0], lm[:, 454, 0] = 0.45, 0.55                               # cheek to cheek = 128 px at 1280
    if hop_at is not None:                                                  # the tracker switches to another person's face
        lm[int(hop_at * FPS):, :, 0] += 0.3
    # ---- pose: shoulders
    pose = np.full((n, 33, 4), np.nan, np.float32)
    roll_sh = np.zeros(n)
    if shoulders:
        roll_sh = shoulders[0] * np.sin(2 * np.pi * t / shoulders[1]) + rng.normal(0, 0.05, n)
    th = np.radians(roll_sh)
    for k, idx in ((1, 11), (-1, 12)):
        pose[:, idx, 0] = (640 + k * 150 * np.cos(th)) / W
        pose[:, idx, 1] = (400 + k * 150 * np.sin(th)) / H
        pose[:, idx, 2], pose[:, idx, 3] = 0.0, 1.0
    # ---- hand and camera
    cam_xy = np.zeros((n, 2))
    if cam:
        cam_xy[:, 0] = cam[0] * np.sin(2 * np.pi * t / cam[1])
    hand_lm = np.full((n, 2, 21, 3), np.nan, np.float32)
    if hand:
        still_until, moving_until, amp, period = hand
        x = np.where((t >= still_until) & (t < moving_until), amp * np.sin(2 * np.pi * (t - still_until) / period), 0.0)
        base = np.zeros((21, 2))
        base[0] = (0, 60)
        base[5], base[9], base[13], base[17] = (-50, -10), (-17, -15), (17, -15), (50, -10)       # knuckle breadth 100 px
        for i in range(n):
            pts = base + (800 + x[i], 400) - cam_xy[i] * 0     # in the world; the camera shifts everything below
            hand_lm[i, 0, :, 0] = (pts[:, 0] + cam_xy[i, 0]) / W
            hand_lm[i, 0, :, 1] = pts[:, 1] / H
            hand_lm[i, 0, :, 2] = 0.0
    fdiff = np.abs(rng.normal(0.8, 0.1, n)).astype(np.float32)
    if cut_at is not None:
        fdiff[int(cut_at * FPS)] = 40.0
    return {"fps": FPS, "W": W, "H": H, "n": n, "face_lm": lm, "face_bs": bs, "face_mat": mat, "pose_lm": pose, "hand_lm": hand_lm,
            "cam": cam_xy.astype(np.float32), "cam_inl": np.full(n, 0.95, np.float32), "fdiff": fdiff}


# ============================================================================================ head pose
def test_head_angles_recover_yaw_pitch_roll_from_a_scaled_matrix():
    cases = [(0, 0, 0), (20, 0, 0), (-35, 10, 5), (10, -25, -12), (40, 30, 20)]
    mat = np.tile(np.eye(4), (len(cases), 1, 1))
    for i, (y, p, r) in enumerate(cases):
        mat[i, :3, :3] = rot(y, p, r) * 2.3
        mat[i, :3, 3] = (1, 2, -40)
    ang = FA.head_angles(mat)
    for (y, p, r), (pp, yy, rr) in zip(cases, ang):
        assert (pp, yy, rr) == pytest.approx((p, y, r), abs=1e-6)
    assert np.isnan(FA.head_angles(np.full((1, 4, 4), np.nan))).all()


def test_oscillation_finds_period_and_sinusoid_equivalent_amplitude():
    rng = np.random.default_rng(3)
    t = np.arange(0, 20, 1 / FPS)
    x = 5 + 1.5 * np.sin(2 * np.pi * t / 2.5) + rng.normal(0, 0.15, len(t))
    o = dsp.oscillation(x, FPS, np.ones(len(t), bool), 0.2, 1.5, 3.0, noise_hz=4.0)
    assert o["peak_distinct"] and o["period_s"] == pytest.approx(2.5, rel=0.05)
    assert o["amp"] == pytest.approx(1.5, rel=0.1)
    assert o["noise_amp"] < 0.15                                           # the jitter floor is far below the signal


def test_oscillation_does_not_call_noise_or_drift_a_rhythm():
    rng = np.random.default_rng(4)
    t = np.arange(0, 30, 1 / FPS)
    noise = dsp.oscillation(rng.normal(0, 1, len(t)), FPS, np.ones(len(t), bool), 0.2, 1.5, 3.0)
    assert not noise["peak_distinct"]
    drift = dsp.oscillation(8 * np.sin(2 * np.pi * t / 25.0) + rng.normal(0, 0.3, len(t)), FPS, np.ones(len(t), bool), 0.15, 1.0, 6.0)
    assert not drift["peak_distinct"]                                      # a 25 s drift is below the band: no rhythm there
    assert drift["amp"] < 0.3                                              # and it hardly leaks into it


def test_oscillation_ignores_gaps_and_short_segments():
    t = np.arange(0, 30, 1 / FPS)
    x = np.sin(2 * np.pi * t / 2.0)
    valid = np.ones(len(t), bool)
    valid[int(10 * FPS):int(11 * FPS)] = False                              # one second without data splits the series
    o = dsp.oscillation(x, FPS, valid, 0.2, 1.5, 3.0)
    assert o["n_segments"] == 2 and o["seconds"] == pytest.approx(29.0, abs=0.1)
    assert dsp.oscillation(x, FPS, valid, 0.2, 1.5, 25.0) is None


# ============================================================================================ blinks
def test_blinks_are_counted_timed_and_measured_but_slow_lid_changes_and_long_closures_are_not_blinks():
    planted = [4.0, 9.5, 15.2, 21.0, 26.4, 33.3, 38.0, 44.1, 50.5, 55.0]
    d = synth_track(60, blink_times=planted, long_closure=(28.0, 29.2), look_down=(40.0, 43.0))
    closure = FA.closure_series(d["face_bs"])
    ev, rise = FA.detect_blinks(closure, FPS)
    assert len(ev) == len(planted)
    assert [e["t"] for e in ev] == pytest.approx(planted, abs=1.5 / FPS)
    fwhm = [e["fwhm_s"] for e in ev]
    assert np.median(fwhm) == pytest.approx(0.11, abs=0.04)                  # rise 0.07 s, fall 0.12 s, smoothstep: half height at 0.035 + 0.06
    assert all(e["depth"] > 0.5 for e in ev)
    assert rise.max() < 1.0


def test_blink_rate_and_durations_end_to_end():
    planted = [3.0 + 4.4 * k for k in range(13)]                          # 13 blinks in 60 s, every 4.4 s
    d = synth_track(60, blink_times=planted)
    res, aux = FA.analyze_face(d, FPS, W, H, np.ones(d["n"], bool))
    b = res["blinks"]
    assert b["count"]["value"] == 13
    assert b["rate_per_min"]["value"] == pytest.approx(13 / 60 * 60, rel=0.08)
    assert b["spontaneous"]["count"] == 13 and b["at_gaze_shift"]["count"] == 0
    assert b["interval"]["median"] == pytest.approx(4.4, abs=0.1)
    assert b["duration_fwhm"]["median"] == pytest.approx(0.11, abs=0.04)
    assert res["lids"]["rest_level"]["value"] == pytest.approx(0.06, abs=0.02)   # eyelid rest level away from blinks
    assert res["lids"]["eyes_shut_share"]["value"] == 0.0


def test_eyes_shut_for_a_long_time_is_reported_apart_from_blinks():
    d = synth_track(40, blink_times=[5.0, 12.0], long_closure=(20.0, 22.0))
    res, _ = FA.analyze_face(d, FPS, W, H, np.ones(d["n"], bool))
    assert res["blinks"]["count"]["value"] == 2
    assert res["lids"]["eyes_shut_share"]["value"] == pytest.approx(2.0 / 40, abs=0.015)


# ============================================================================================ gaze
PLANTED_GAZE = [(6.0, 14.0, 0.0), (12.5, -8.0, 6.0), (21.0, 0.0, 0.0), (30.0, 20.0, -5.0), (36.0, 20.0, -5.0 + 0.0), (44.0, -12.0, 0.0)]


def test_gaze_shifts_are_counted_with_holds_and_head_share():
    gaze = [g for g in PLANTED_GAZE if g[0] != 36.0]                      # five shifts: 14, 22 (yaw -8, pitch +6), 10, 28, 33
    d = synth_track(52, gaze=gaze, head_share=0.7)
    res, aux = FA.analyze_face(d, FPS, W, H, np.ones(d["n"], bool))
    sh = aux["shifts"]
    assert len(sh) == 5
    assert [s["t_on"] for s in sh] == pytest.approx([6.0, 12.5, 21.0, 30.0, 44.0], abs=0.35)
    amps = [s["amp"] for s in sh]
    assert amps == pytest.approx([14.0, 22.8, 10.0, 20.6, 32.4], abs=1.5)
    g = res["gaze"]
    assert g["shifts"]["count"] == 5
    assert g["shifts"]["rate_per_10s"]["value"] == pytest.approx(10 * 5 / 52, rel=0.1)
    holds = [s["dwell_s"] for s in sh if s["dwell_s"] is not None]
    assert holds == pytest.approx([6.5, 8.5, 9.0, 14.0], abs=0.7)           # time spent between two shifts
    assert g["shifts"]["head_share"]["median"] == pytest.approx(0.7, abs=0.12)
    assert g["shifts"]["rise_gaze"]["median"] < 0.5                          # eye-led shifts are quick


def test_wiggles_below_the_shift_size_and_slow_drift_are_not_shifts():
    rng = np.random.default_rng(8)
    n = 40 * 30
    t = np.arange(n) / FPS
    gaze = np.column_stack([2.0 * np.sin(2 * np.pi * t / 3.0) + 3.0 * np.sin(2 * np.pi * t / 0.7) * 0 + rng.normal(0, 0.4, n),
                            1.5 * np.sin(2 * np.pi * t / 5.0) + rng.normal(0, 0.4, n)])
    assert FA.gaze_shifts(gaze, gaze.copy(), FPS) == []
    step = gaze.copy()
    step[:, 0] += np.where(t > 20.0, 5.0, 0.0)                              # a 5 degree step is below the 6 degree minimum
    assert FA.gaze_shifts(step, step.copy(), FPS) == []
    step[:, 0] += np.where(t > 20.0, 3.0, 0.0)
    assert len(FA.gaze_shifts(step, step.copy(), FPS)) == 1                 # 8 degrees is a shift


def test_gaze_is_undefined_while_the_eyes_are_shut():
    d = synth_track(40, gaze=[(10.0, 15.0, 0.0)], long_closure=(18.0, 21.0))
    res, aux = FA.analyze_face(d, FPS, W, H, np.ones(d["n"], bool))
    shut = (np.arange(d["n"]) / FPS > 18.5) & (np.arange(d["n"]) / FPS < 20.5)
    assert np.isnan(aux["gaze"][shut]).all()
    assert res["gaze"]["shifts"]["count"] == 1                              # the closure does not fake a downward glance


def test_blinks_near_a_gaze_shift_are_kept_out_of_the_spontaneous_rate():
    d = synth_track(60, blink_times=[10.1, 25.0, 40.0, 50.0], gaze=[(10.0, 15.0, 0.0)])
    res, aux = FA.analyze_face(d, FPS, W, H, np.ones(d["n"], bool))
    b = res["blinks"]
    assert b["count"]["value"] == 4
    assert b["at_gaze_shift"]["count"] == 1 and b["spontaneous"]["count"] == 3


# ============================================================================================ head, mouth, body, hands
def test_head_nod_period_and_amplitude_end_to_end():
    d = synth_track(40, nod=(1.2, 2.5))
    res, _ = FA.analyze_face(d, FPS, W, H, np.ones(d["n"], bool))
    nod = res["head_motion"]["nod"]
    assert nod["period_s"] == pytest.approx(2.5, rel=0.08)
    assert nod["amp"] == pytest.approx(1.2, rel=0.12)
    assert nod["noise_amp"] < 0.3 * nod["amp"]
    assert res["head_pose"]["pitch_up"]["mean"] == pytest.approx(3.0, abs=0.3)
    assert res["head_pose"]["roll"]["mean"] == pytest.approx(2.0, abs=0.2)


def test_mouth_activity_finds_the_talking_stretch():
    d = synth_track(60, jaw=(20.0, 30.0, 0.3, 4.0))
    res, _ = FA.analyze_face(d, FPS, W, H, np.ones(d["n"], bool))
    m = res["mouth"]
    assert m["moving_share"]["value"] == pytest.approx(10.5 / 60, abs=0.03)
    assert m["peak_p90_when_moving"]["value"] == pytest.approx(0.29, abs=0.03)
    assert m["cycle_period_s"]["value"] == pytest.approx(0.25, abs=0.04)
    quiet = FA.analyze_face(synth_track(30), FPS, W, H, np.ones(900, bool))[0]["mouth"]
    assert quiet["moving_share"]["value"] == 0.0 and "peak_p90_when_moving" not in quiet


def test_shoulder_sway_amplitude_and_period():
    d = synth_track(40, shoulders=(2.0, 2.5))
    res = BO.analyze_body(d["pose_lm"], W, H, FPS, np.zeros((d["n"], 2)), np.ones(d["n"], bool))
    sw = res["shoulder_sway"]
    assert sw["amp"] == pytest.approx(2.0, rel=0.1)
    assert sw["period_s"] == pytest.approx(2.5, rel=0.08)
    assert sw["noise_amp"] < 0.2
    d["pose_lm"][:, 11:13, 3] = 0.2                                         # shoulders not visible: no body statistics
    assert BO.analyze_body(d["pose_lm"], W, H, FPS, np.zeros((d["n"], 2)), np.ones(d["n"], bool)) is None


def test_hand_stillness_and_speed_with_camera_stabilisation():
    d = synth_track(60, hand=(20.0, 40.0, 150.0, 4.0))
    ones = np.ones(d["n"], bool)
    res, raw = BO.analyze_hands(d["hand_lm"], W, H, FPS, np.zeros((d["n"], 2)), ones)
    assert res["tracked"] and res["still_share"]["value"] == pytest.approx(2 / 3, abs=0.08)
    speed_mm = res["speed_mm_s"]["p90"]
    assert speed_mm == pytest.approx(150 * 2 * np.pi / 4.0 / 100 * BO.HAND_WIDTH_MM, rel=0.25)     # peak palm speed in mm/s
    assert 8 <= res["gestures"]["count"] <= 11                              # half-cycles of the 4 s period over 20 s
    # a hand that does not move in the world, seen by a panning camera
    d = synth_track(60, hand=(0.0, 0.0, 0.0, 4.0), cam=(120.0, 10.0))
    still, _ = BO.analyze_hands(d["hand_lm"], W, H, FPS, d["cam"].astype(float), ones)
    drift, _ = BO.analyze_hands(d["hand_lm"], W, H, FPS, np.zeros((d["n"], 2)), ones)
    assert still["still_share"]["value"] > 0.95
    assert drift["still_share"]["value"] < 0.3                              # without the camera correction the pan reads as motion


def test_a_clip_without_hands_has_no_hand_statistics():
    d = synth_track(20)
    res, _ = BO.analyze_hands(d["hand_lm"], W, H, FPS, np.zeros((d["n"], 2)), np.ones(d["n"], bool))
    assert res["tracked"] is False


# ============================================================================================ camera drift
def test_camera_fit_rejects_foreground_outliers():
    rng = np.random.default_rng(5)
    bg0 = rng.uniform(0, 640, (160, 2))
    fg0 = rng.uniform(200, 440, (110, 2))                                   # 40 % of the points belong to a moving subject
    shift_bg, shift_fg = np.array([3.2, -1.5]), np.array([-9.0, 6.0])
    p0 = np.vstack([bg0, fg0])
    p1 = np.vstack([bg0 + shift_bg, fg0 + shift_fg]) + rng.normal(0, 0.15, p0.shape)
    M, ratio = TR.fit_similarity(p0, p1)
    assert M[:2, 2] == pytest.approx(shift_bg, abs=0.1)
    assert ratio == pytest.approx(160 / 270, abs=0.08)
    naive = (p1 - p0).mean(0)                                               # what a plain average would say
    assert np.hypot(*(naive - shift_bg)) > 2.0
    assert TR.fit_similarity(p0[:5], p1[:5]) == (None, pytest.approx(np.nan, nan_ok=True))


def _texture(h, w, seed):
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 2.2)
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def test_camera_steps_follow_the_background_not_a_big_moving_subject():
    bg = _texture(400, 560, 1)
    person = _texture(190, 200, 2)
    T = np.eye(3)
    prev = None
    dx, dy = 1.6, -0.9                                                      # background drifts this much per frame
    for k in range(25):
        M = np.float32([[1, 0, 40 - k * dx], [0, 1, 40 - k * dy]])           # camera view window moves over the background
        frame = cv2.warpAffine(bg, M, (480, 320), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        x0, y0 = 120 + int(6 * np.sin(k / 4.0)), 60 + 3 * k // 2             # the subject moves by itself, covering a quarter of the view
        frame[y0:y0 + 190, x0:x0 + 200] = person
        if prev is not None:
            T, ratio = TR.camera_step(prev, frame, T, scale=1.0)
        prev = frame
    # the background moves by (-dx, -dy) per frame in the image, 24 steps. The subject drifts down by 36 px meanwhile: following it
    # would put T[1, 2] near +36 instead of the background's +21.6
    assert T[0, 2] == pytest.approx(-24 * dx, abs=3.0)
    assert T[1, 2] == pytest.approx(-24 * dy, abs=3.0)


def test_subject_mask_keeps_features_on_the_background():
    pose = np.full((33, 4), np.nan, np.float32)
    pose[:, 3] = 0.9
    pose[:, 0], pose[:, 1] = np.linspace(0.4, 0.6, 33), np.linspace(0.2, 0.8, 33)
    m = TR.subject_mask(pose, (180, 320))
    assert m is not None and m[90, 160] == 0 and m[5, 5] == 255
    pose[:, 0], pose[:, 1] = np.linspace(0.02, 0.98, 33), np.linspace(0.02, 0.98, 33)
    assert TR.subject_mask(pose, (180, 320)) is None                        # the subject fills the frame: no mask


def test_cuts_are_isolated_spikes_not_pans_or_repeated_frames():
    rng = np.random.default_rng(6)
    f = np.abs(rng.normal(0.8, 0.1, 600)).astype(np.float32)
    f[100:110] = 6.0                                                        # a fast pan: many large differences in a row
    f[200::2] = 0.0                                                         # frame-rate conversion repeats frames
    f[400] = 45.0                                                           # a hard cut
    f[1] = np.nan
    assert ME.find_cuts(f, FPS) == [400]
    ok = ME.usable_mask(600, [400], FPS)
    assert not ok[400] and not ok[398] and ok[380] and ok[420]


def test_camera_track_does_not_count_a_cut_as_a_camera_move():
    d = {"cam": np.cumsum(np.tile([1.0, 0.0], (100, 1)), axis=0)}
    d["cam"][50:] += 500.0                                                  # the tracker's fit across a cut is garbage
    cam = ME.camera_track(d, [50])
    assert cam[-1, 0] == pytest.approx(99.0 - 0.0, abs=1.0)                 # 500 px of cut jump removed
    assert cam[10, 0] == pytest.approx(10.0)


def test_cut_frames_are_dropped_from_the_statistics_of_a_clip():
    d = synth_track(30, blink_times=[5.0, 20.0], cut_at=12.0)
    res, _ = ME.analyze_clip({"id": 1}, d)
    assert res["qc"]["cuts_s"] == [12.0]
    assert any("hard cut" in w for w in res["qc"]["warnings"])
    assert res["face"]["blinks"]["count"]["value"] == 2


# ============================================================================================ whole clip, pooling, recommendations
def test_clip_result_has_qc_and_all_sections():
    d = synth_track(40, blink_times=[5.0, 15.0, 30.0], nod=(1.0, 2.5), shoulders=(1.5, 2.5), hand=(0.0, 40.0, 100.0, 3.0),
                    jaw=(10.0, 20.0, 0.25, 3.0), gaze=[(8.0, 12.0, 0.0), (25.0, -10.0, 0.0)])
    res, aux = ME.analyze_clip({"id": 7, "url": "u", "author": "a"}, d)
    assert res["id"] == 7 and res["duration_s"] == 40.0 and res["size_px"] == [W, H]
    assert res["face"]["usable"] is True and res["face"]["face_width_px"]["value"] == pytest.approx(128, abs=2)   # synthetic face is 128 px wide
    assert not any("face statistics" in w for w in res["qc"]["warnings"])
    assert res["qc"]["camera_drift_p95"]["value"] == 0.0 and res["qc"]["duplicate_frames"]["value"] == 0.0
    assert res["body"]["shoulder_sway"]["amp"] == pytest.approx(1.5, rel=0.15)
    assert res["hands"]["tracked"]
    json.dumps(res)                                                         # JSON-able as it is


def _clip(rate_n, rate_s, shifts=0, gaze_s=60.0, holds=(), rises=(), head_share=()):
    """A minimal per-clip result for pooling tests."""
    return {"duration_s": rate_s, "hands": {"tracked": False}, "body": None,
            "face": {"usable": True, "tracked_seconds": {"value": rate_s},
                     "blinks": {"count": {"value": rate_n}, "observed_seconds": {"value": rate_s},
                                "rate_per_min": {"value": 60 * rate_n / rate_s},
                                "spontaneous": {"count": rate_n, "observed_seconds": rate_s, "rate_per_min": {"value": 60 * rate_n / rate_s}},
                                "at_gaze_shift": {"count": 0}, "duration_fwhm": dsp.ev([0.2] * rate_n, "s"),
                                "duration_total_20pct": dsp.ev([], "s"), "depth": dsp.ev([], "x"), "interval": dsp.ev([], "s")},
                     "lids": {"rest_level": {"value": 0.1}, "eyes_shut_share": {"value": 0.0}},
                     "gaze": {"observed_seconds": {"value": gaze_s},
                              "shifts": {"count": shifts, "rate_per_10s": {"value": 10 * shifts / gaze_s}, "amplitude": dsp.ev([15.0] * shifts, "deg"),
                                         "peak_speed": dsp.ev([], "x"), "rise_gaze": dsp.ev([0.2] * shifts, "s"),
                                         "head_rise": dsp.ev(list(rises), "s"), "head_share": dsp.ev(list(head_share), "x")},
                              "hold": dsp.ev(list(holds), "s"), "eye_in_head": {"p95_magnitude": {"value": 18.0}}},
                     "head_pose": {}, "head_motion": {}, "mouth": None}}


def test_pooled_rates_weigh_clips_by_observed_time():
    P = ME.pool({1: _clip(10, 60.0), 2: _clip(2, 30.0)})
    r = P["blinks"]["rate_per_min"]
    assert r["value"] == pytest.approx(60 * 12 / 90)                         # 8/min, not the mean of 10 and 4
    assert r["n"] == 12 and r["observed_seconds"] == 90.0 and r["n_clips"] == 2
    assert P["blinks"]["rate_by_clip"]["median"] == pytest.approx(7.0)
    assert P["blinks"]["duration_fwhm"]["n"] == 12                           # events pooled over clips


def test_describe_summarises_median_iqr_and_ignores_missing():
    d = dsp.describe([1, 2, 3, 4, 100, None, float("nan")], "s")
    assert d["n"] == 5 and d["median"] == 3 and d["iqr"] == [2, 4] and d["max"] == 100
    assert dsp.describe([], "s") == {"n": 0, "unit": "s"}


def test_recommendation_maps_pooled_numbers_onto_perform_parameters():
    holds, rises = [1.0, 1.2, 0.8, 1.4], [0.45, 0.5, 0.55, 0.6]
    results = {1: _clip(20, 60.0, shifts=12, gaze_s=60.0, holds=holds, rises=rises, head_share=[0.6, 0.7, 0.8]),
               2: _clip(10, 60.0, shifts=6, gaze_s=60.0)}
    P = ME.pool(results)
    rec, why, skipped = RC.recommend(P, results, fps=30.0)
    assert rec["blink"]["per_min"] == pytest.approx(15.0)                    # 30 blinks in 120 s
    assert rec["gaze"]["rate_per_10s"] == pytest.approx(0.75)                # 18 shifts / 120 s = 1.5 per 10 s, an event is out and back
    assert rec["head_share"] == pytest.approx(0.7)
    assert rec["eye_max"] == pytest.approx(18.0)
    assert rec["lids"] == pytest.approx(0.1)
    # the envelope rise, run through the performance follower, reproduces the measured head turn
    assert RC.turn_time(rec["gaze"]["rise_s"], 30.0, RC.HEAD_FOLLOW_FRAMES) == pytest.approx(0.525, abs=0.02)
    assert rec["gaze"]["hold_s"] == pytest.approx(np.median(holds) + rec["gaze"]["rise_s"], abs=0.01)
    assert why["blink.per_min"]["n"] == 30 and why["blink.per_min"]["confidence"] == "high"
    assert set(why) >= {"blink.per_min", "gaze.rate_per_10s", "gaze.rise_s", "gaze.hold_s", "head_share", "eye_max", "lids"}
    for k in ("sway", "breath", "bob", "sing.mouth", "nod"):
        assert k in skipped and isinstance(skipped[k], str)


def test_recommendation_with_no_usable_face_says_why():
    P = ME.pool({1: {"duration_s": 10.0, "face": None, "body": None, "hands": {"tracked": False}, "qc": {}}})
    rec, why, skipped = RC.recommend(P, {})
    assert rec == {} or "blink" not in rec
    assert "no clip has a usable face" in skipped["blink.per_min"]


def test_the_recommended_blink_table_holds_only_keys_the_performance_stage_reads():
    from mkmmd.core import tables as TB
    results = {1: _clip(20, 60.0, shifts=12, gaze_s=60.0), 2: _clip(10, 60.0, shifts=6, gaze_s=60.0)}
    rec, why, _skipped = RC.recommend(ME.pool(results), results, fps=30.0)
    assert rec["blink"] and set(rec["blink"]) <= set(TB.PERFORM["blink"])                  # per_min; no duration key is read
    assert "blink.duration_s" not in why


def test_the_rise_that_reproduces_a_measured_turn_runs_through_the_performance_follower():
    r = RC.rise_for_turn(0.5, 30.0, 3.0)
    assert RC.turn_time(r, 30.0, 3.0) == pytest.approx(0.5, abs=0.01)
    assert RC.rise_for_turn(0.1, 30.0, 3.0) == 0.1                          # the follower alone is slower than that


# ============================================================================================ Pexels renditions, store
def _file(w, h, fps=25.0, size=3_000_000, link="u"):
    return {"file_type": "video/mp4", "width": w, "height": h, "fps": fps, "size": size, "link": link, "quality": "hd"}


def test_pick_rendition_is_the_smallest_one_at_or_above_the_height_within_the_cap():
    files = [_file(640, 360), _file(1280, 720, 25, 5_000_000), _file(1280, 720, 50, 9_000_000), _file(1920, 1080, 25, 12_000_000),
             _file(3840, 2160, 25, 90_000_000), {"file_type": "video/webm", "width": 1280, "height": 720, "link": "x"}]
    assert pexels.pick_rendition(files, 720)["fps"] == 50.0                  # same size class: the higher frame rate
    assert pexels.pick_rendition(files, 1000)["width"] == 1920
    assert pexels.pick_rendition(files, 2000, max_bytes=50_000_000) is None
    portrait = [_file(360, 640), _file(720, 1280), _file(1080, 1920)]
    assert pexels.pick_rendition(portrait, 720)["width"] == 720             # '720p' is the short side
    assert pexels.pick_rendition([_file(1280, 720, size=None)], 720) is not None   # unknown size: the download enforces the cap


def test_search_results_keep_only_clips_with_a_usable_rendition():
    video = {"id": 5, "duration": 12, "width": 4096, "height": 2160, "url": "https://x/5", "user": {"name": "A", "url": "https://x/@a"},
             "video_files": [_file(426, 226), _file(1366, 720, 25, 3_000_000)]}
    s = pexels.summarize(video, 720)
    assert s["pick"] == "1366x720@25 3.0MB" and s["user"] == "A" and s["license"] == "Pexels License"
    assert s["renditions"] == ["1366x720@25 3.0MB"]                          # thumbnail-sized files are not listed
    assert pexels.summarize(video, 1080)["pick"] is None
    e = pexels.clip_entry(video)
    assert e["id"] == 5 and e["author"] == "A" and e["license"] == "Pexels License" and len(e["files"]) == 2


def test_reference_sets_live_in_the_cache_and_clean_up(tmp_path, monkeypatch):
    monkeypatch.setenv("MK_CACHE", str(tmp_path))
    scope = store.scope_dir(None)
    assert scope == tmp_path / "ref" / "default"
    rs = store.RefSet(scope, "verify")
    entry = {"id": 11, "url": "u", "author": "A", "files": []}
    assert rs.put(entry, "front-on") == "added"
    assert rs.put(dict(entry, author="B")) == "updated"
    c = rs.clips()
    assert len(c) == 1 and c[0]["author"] == "B" and c[0]["why"] == "front-on"        # the note survives a refresh
    rs.video_dir.mkdir(parents=True)
    rs.track_dir.mkdir(parents=True)
    rs.video_path(11).write_bytes(b"x" * 1000)
    rs.track_path(11).write_bytes(b"y" * 400)
    rs.sheet_file.write_bytes(b"z" * 100)
    freed = rs.clean(videos=True, tracks=False)
    assert freed["videos_bytes"] == 1000 and freed["sheet_bytes"] == 100 and rs.track_path(11).exists()
    assert rs.clips_file.exists()
    assert rs.clean()["tracks_bytes"] == 400 and rs.clips_file.exists()
    rs.clean(forget=True)
    assert not rs.dir.exists()
    with pytest.raises(RefUsage):
        store.RefSet(scope, "../evil")
    with pytest.raises(RefUsage):
        store.RefSet(scope, "x").require_clips()


def test_project_sets_use_the_project_name(tmp_path, monkeypatch):
    monkeypatch.setenv("MK_CACHE", str(tmp_path))

    class P:
        name = "my video/1"
    assert store.scope_dir(P()) == tmp_path / "ref" / "my_video_1"
    other = store.RefSet(store.scope_dir(None), "a")
    other.put({"id": 1, "files": []})
    other.dir.joinpath("clips").mkdir()
    other.video_path(1).write_bytes(b"1" * 10)
    store.models_dir().mkdir(parents=True)
    (store.models_dir() / "m.task").write_bytes(b"m" * 7)
    rep = store.clean_scope(store.scope_dir(None))
    assert store.total_freed(rep) == 17 and not store.models_dir().exists()


# ============================================================================================ robustness and the command
def test_tiny_and_empty_tracks_do_not_crash():
    d = synth_track(30, blink_times=[5.0], hand=(0.0, 30.0, 100.0, 3.0), shoulders=(1.0, 2.5))
    for a, b in ((0, 4), (0, 20), (100, 130), (0, 900)):
        cut = {k: (v[a:b] if isinstance(v, np.ndarray) and v.ndim and v.shape[0] == d["n"] else v) for k, v in d.items()}
        cut["n"] = len(cut["fdiff"])
        res, _ = ME.analyze_clip({"id": 1}, cut)
        json.dumps(res)
    empty = dict(d, face_lm=np.full_like(d["face_lm"], np.nan), face_bs=np.full_like(d["face_bs"], np.nan),
                 face_mat=np.full_like(d["face_mat"], np.nan), pose_lm=np.full_like(d["pose_lm"], np.nan),
                 hand_lm=np.full_like(d["hand_lm"], np.nan))
    res, _ = ME.analyze_clip({"id": 1}, empty)
    assert res["face"] is None and res["body"] is None and res["hands"]["tracked"] is False
    assert len(res["qc"]["warnings"]) >= 3


def _cli(argv):
    parser = argparse.ArgumentParser(prog="mk")
    CLI.add(parser.add_subparsers(dest="command"))
    args = parser.parse_args(["ref"] + argv)
    return args.func(args)


def _seed_set(scope, name="t"):
    rs = store.RefSet(scope, name)
    rs.put({"id": 1, "url": "https://x/1", "author": "A", "license": "Pexels License", "files": []}, "front-on")
    rs.track_dir.mkdir(parents=True, exist_ok=True)
    d = synth_track(45, blink_times=[3.0 + 4.0 * k for k in range(10)], nod=(1.0, 2.5), shoulders=(1.5, 2.5), jaw=(10.0, 20.0, 0.3, 4.0))
    np.savez_compressed(rs.track_path(1), **d)
    return rs


def test_measure_and_clean_work_offline_outside_a_project(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MK_CACHE", str(tmp_path / "cache"))
    monkeypatch.chdir(tmp_path)
    rs = _seed_set(store.scope_dir(None))
    assert _cli(["measure", "--set", "t"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["out"] == str(rs.measure_file) and rs.measure_file.exists()
    assert out["recommend"]["blink"]["per_min"] == pytest.approx(10 / 45 * 60, rel=0.1)
    assert out["recommend"]["sing"]["mouth"] == pytest.approx(0.28, abs=0.04)
    assert out["recommend"]["nod"]["period"] == pytest.approx(2.5, rel=0.1)
    assert out["why"]["blink.per_min"]["n"] == 10 and "skipped" in out and out["sources"][0]["author"] == "A"
    assert out["clips"][0]["face"] == "usable" and "pooled" in out and "meta" not in out      # the digest
    full = json.loads(rs.measure_file.read_text())
    assert {"meta", "clips", "pooled", "recommend", "why", "sources"} <= set(full) and full["sources"][0]["why"] == "front-on"
    assert _cli(["measure", "--set", "t", "--full"]) == 0
    assert "meta" in json.loads(capsys.readouterr().out)
    assert _cli(["clean", "--set", "t"]) == 0
    freed = json.loads(capsys.readouterr().out)
    assert freed["freed_bytes"] > 0 and not rs.track_path(1).exists() and rs.clips_file.exists()
    with pytest.raises(Exception) as e:                                      # nothing tracked any more
        _cli(["measure", "--set", "t"])
    assert "mk ref track" in str(e.value)


def test_measure_inside_a_project_writes_the_project_file(tmp_path, monkeypatch, capsys):
    proj = tmp_path / "demo"
    proj.mkdir()
    (proj / "mk.toml").write_text('[project]\nname = "demo"\nfps = 24\nframe0 = 1\nduration = 10\n')
    monkeypatch.setenv("MK_CACHE", str(tmp_path / "cache"))
    monkeypatch.chdir(proj)
    _seed_set(tmp_path / "cache" / "ref" / "demo")
    assert _cli(["measure", "--set", "t"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["out"] == str(proj / "ref" / "t.json") and out["project"] == "demo"
    assert (proj / "ref" / "t.json").exists()
    assert _cli(["clean", "--all"]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["scope"].endswith("ref/demo") and "t" in rep["sets"] and (tmp_path / "cache" / "ref" / "demo" / "t" / "clips.json").exists()


def test_a_tracker_hop_between_two_people_cuts_the_face_series():
    d = synth_track(40, blink_times=[5.0, 20.0, 30.0], hop_at=20.0)
    res, _ = FA.analyze_face(d, FPS, W, H, np.ones(d["n"], bool))
    assert res["identity_jumps"] == 1
    assert res["blinks"]["count"]["value"] == 2                              # the blink at the hop is not a blink
    assert res["coverage"]["value"] == pytest.approx(1 - 10 / d["n"], abs=0.003)
    clean = FA.analyze_face(synth_track(40, blink_times=[5.0, 20.0, 30.0]), FPS, W, H, np.ones(1200, bool))[0]
    assert clean["identity_jumps"] == 0 and clean["blinks"]["count"]["value"] == 3
