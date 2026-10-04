"""Tracking: one streaming pass per clip -> <set>/track/<id>.npz.

Per frame: pose (33 landmarks) and hands (2 x 21) on the full frame; the face landmarker (478 landmarks, 52
blendshapes, facial transformation matrix) on an enlarged square crop around the head, found with the pose model on
24 sampled frames (the face model finds small or downward-looking faces far better when they are enlarged);
the global camera translation (RANSAC similarity fit of sparse optical flow on the background); a thumbnail
difference per frame for cut detection. Not-tracked entries are NaN.

npz arrays (n = frames; coordinates normalised to the full frame unless noted)
  fps, W, H, n, roi (x0, y0, side in px, or -1)         face_lm (n, 478, 3)     face_bs (n, 52)    face_mat (n, 4, 4)
  hand_lm (n, 2, 21, 3)     pose_lm (n, 33, 4) (x, y, z, visibility)     cam (n, 2) cumulative camera translation of
  the frame centre in px     cam_inl (n,) inlier ratio of each frame-to-frame fit     fdiff (n,) mean abs thumbnail
  difference to the previous frame     schema, tracker (strings)"""
import contextlib
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")           # MediaPipe's C++ logging, before it is imported
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from . import RefError, RefUsage, models, pexels, store  # noqa: E402

SCHEMA = 2
FACE_CROP = 512                  # px; the head crop is resampled to this for the face model
MIN_CONF = 0.3
MAX_SECONDS = 90.0               # tracking stops here (the track says so) so one long clip cannot hog the machine
THUMB = (64, 36)


# ============================================================================================ MediaPipe glue
def _mp():
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    return mp, BaseOptions, vision


def _image(mp, rgb):
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))


def head_roi(path, W, H, pose_model, n_picks=24):
    """Square crop (x0, y0, side) around the head for the face model, from the pose model on sampled frames; None
    when no person is found."""
    mp, BaseOptions, vision = _mp()
    pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=pose_model), running_mode=vision.RunningMode.IMAGE, num_poses=1))
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    picks = set(np.linspace(0, max(n - 1, 0), n_picks).astype(int).tolist())
    centers, widths = [], []
    try:
        for i in range(n):
            if not cap.grab():
                break
            if i not in picks:
                continue
            ok, bgr = cap.retrieve()
            if not ok:
                continue
            r = pose.detect(_image(mp, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
            if not r.pose_landmarks:
                continue
            lm = np.array([[l.x * W, l.y * H, l.visibility or 0.0] for l in r.pose_landmarks[0]])
            if lm[:9, 2].mean() < 0.5:
                continue
            centers.append(lm[:9, :2].mean(0))
            if lm[11, 2] > 0.5 and lm[12, 2] > 0.5:
                widths.append(np.linalg.norm(lm[11, :2] - lm[12, :2]) / 2.6)      # head width ~ shoulder width / 2.6
            else:
                widths.append(np.linalg.norm(lm[3, :2] - lm[6, :2]) * 1.7)        # from the outer eye corners
    finally:
        cap.release()
        pose.close()
    if len(centers) < 6:
        return None
    centers = np.array(centers)
    hw = float(np.median(widths))
    lo, hi = np.percentile(centers, 5, axis=0), np.percentile(centers, 95, axis=0)
    mid = (lo + hi) / 2
    side = max(2.8 * hw, float((hi - lo).max()) + 2.0 * hw, 192.0)
    side = int(min(side, min(W, H)))
    x0 = int(np.clip(mid[0] - side / 2, 0, W - side))
    y0 = int(np.clip(mid[1] - side / 2, 0, H - side))
    return x0, y0, side


# ============================================================================================ camera motion
def fit_similarity(p0, p1, thresh=1.0):
    """Dominant (RANSAC) similarity transform p0 -> p1 of matched points (n, 2): background points vote together,
    foreground points moving differently are rejected as outliers. Returns (3x3 matrix, inlier ratio) or (None, nan)."""
    p0 = np.asarray(p0, np.float32).reshape(-1, 1, 2)
    p1 = np.asarray(p1, np.float32).reshape(-1, 1, 2)
    if len(p0) < 10:
        return None, np.nan
    M, inl = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC, ransacReprojThreshold=thresh)
    if M is None:
        return None, np.nan
    M3 = np.eye(3)
    M3[:2] = M
    return M3, float(inl.mean())


def camera_step(prev, cur, T, mask=None, scale=2.0):
    """Compose the dominant motion prev -> cur of two (half-scale) grey frames onto T (full-resolution px; `scale` is
    full / half). `mask` (uint8, 255 = usable) keeps the features on the background when the subject is known; with too
    few features outside it, the whole frame is used. Returns (T, inlier ratio); T is unchanged where no fit exists."""
    p0 = cv2.goodFeaturesToTrack(prev, 300, 0.01, 8, mask=mask)
    if mask is not None and (p0 is None or len(p0) < 12):
        p0 = cv2.goodFeaturesToTrack(prev, 300, 0.01, 8)
    if p0 is None or len(p0) < 12:
        return T, np.nan
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, p0, None, winSize=(21, 21), maxLevel=3)
    ok = st[:, 0] == 1
    if ok.sum() < 10:
        return T, np.nan
    M3, ratio = fit_similarity(p0[ok][:, 0], p1[ok][:, 0])
    if M3 is None:
        return T, np.nan
    M3[:2, 2] *= scale
    return M3 @ T, ratio


def subject_mask(pose_lm, shape, margin=0.2):
    """Background mask (255 outside, 0 over the subject) from one frame's pose landmarks (x, y, z, visibility), or None."""
    if pose_lm is None or not np.isfinite(pose_lm[:, 0]).any():
        return None
    seen = (pose_lm[:, 3] > 0.3) & np.isfinite(pose_lm[:, 0])
    if seen.sum() < 5:
        return None
    h, w = shape
    x, y = pose_lm[seen, 0] * w, pose_lm[seen, 1] * h
    mx, my = margin * (x.max() - x.min()) + 8, margin * (y.max() - y.min()) + 8
    x0, x1 = int(max(x.min() - mx, 0)), int(min(x.max() + mx, w))
    y0, y1 = int(max(y.min() - my, 0)), int(min(y.max() + my, h))
    if (x1 - x0) * (y1 - y0) > 0.85 * w * h:
        return None
    mask = np.full((h, w), 255, np.uint8)
    mask[y0:y1, x0:x1] = 0
    return mask


# ============================================================================================ the pass
@contextlib.contextmanager
def quiet_native_stderr(log_path):
    """MediaPipe's C++ layer writes warnings to file descriptor 2 (dozens of lines per clip). Keep them in a log file
    while tracking runs and give back a function that still prints our own progress to the real stderr."""
    sys.stderr.flush()
    saved = os.dup(2)
    log = open(log_path, "wb")
    real = os.fdopen(os.dup(saved), "w")
    os.dup2(log.fileno(), 2)

    def say(msg):
        real.write(msg + "\n")
        real.flush()

    try:
        yield say
    finally:
        sys.stderr.flush()
        os.dup2(saved, 2)
        os.close(saved)
        log.close()
        real.close()


def track_clip(job):
    """job: {id, video, out, models: {name: path}, max_seconds}. Returns a small summary dict. MediaPipe's native log
    goes to <out>.log (deleted when the clip tracked fine)."""
    out = Path(job["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    log = out.with_suffix(".log")
    with quiet_native_stderr(log) as say:
        res = _track(job, say)
    log.unlink(missing_ok=True)
    return res


def _track(job, say):
    mp, BaseOptions, vision = _mp()
    t0 = time.time()
    path, out, M = str(job["video"]), Path(job["out"]), job["models"]
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RefError(f"cannot open {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not (1.0 <= fps <= 240.0):
        say(f"  clip {job['id']}: implausible frame rate {fps}, assuming 30")
        fps = 30.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_est = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = int(job.get("max_seconds", MAX_SECONDS) * fps)
    roi = head_roi(path, W, H, M["pose_landmarker_full.task"])
    RM = vision.RunningMode.VIDEO
    face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=M["face_landmarker.task"]), running_mode=RM, num_faces=1,
        min_face_detection_confidence=MIN_CONF, min_face_presence_confidence=MIN_CONF, min_tracking_confidence=MIN_CONF,
        output_face_blendshapes=True, output_facial_transformation_matrixes=True))
    hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=M["hand_landmarker.task"]), running_mode=RM, num_hands=2,
        min_hand_detection_confidence=MIN_CONF, min_hand_presence_confidence=MIN_CONF, min_tracking_confidence=MIN_CONF))
    pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=M["pose_landmarker_full.task"]), running_mode=RM, num_poses=1,
        min_pose_detection_confidence=MIN_CONF, min_pose_presence_confidence=MIN_CONF, min_tracking_confidence=MIN_CONF))

    nan = np.nan
    face_lm, face_bs, face_mat, hand_lm, pose_lm = [], [], [], [], []
    cam, cam_inl, fdiff = [], [], []
    T_cam = np.eye(3)
    centre = np.array([W / 2, H / 2, 1.0])
    prev = prev_thumb = prev_pose = None
    i = 0
    try:
        while i < max_frames:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            ts = int(round(i * 1000.0 / fps))
            img = _image(mp, rgb)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            thumb = cv2.resize(gray, THUMB, interpolation=cv2.INTER_AREA).astype(np.float32)
            fdiff.append(float(np.abs(thumb - prev_thumb).mean()) if prev_thumb is not None else nan)
            prev_thumb = thumb
            # camera: background features only (the subject is masked out using the previous frame's pose)
            g = cv2.resize(gray, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
            if prev is None:
                cam_inl.append(nan)
            else:
                mask = subject_mask(prev_pose, (H // 2, W // 2)) if prev_pose is not None else None
                T_cam, ratio = camera_step(prev, g, T_cam, mask)
                cam_inl.append(ratio)
            cam.append((T_cam @ centre)[:2] - centre[:2])
            prev = g
            # pose
            r = pose.detect_for_video(img, ts)
            pl = np.full((33, 4), nan, np.float32)
            if r.pose_landmarks:
                pl = np.array([[l.x, l.y, l.z, l.visibility or 0.0] for l in r.pose_landmarks[0]], np.float32)
            pose_lm.append(pl)
            prev_pose = pl
            # hands
            r = hand.detect_for_video(img, ts)
            hl = np.full((2, 21, 3), nan, np.float32)
            for k, pts in enumerate(r.hand_landmarks[:2]):
                hl[k] = np.array([[l.x, l.y, l.z] for l in pts], np.float32)
            hand_lm.append(hl)
            # face (head crop enlarged to FACE_CROP px)
            if roi is not None:
                x0, y0, s = roi
                crop = cv2.resize(rgb[y0:y0 + s, x0:x0 + s], (FACE_CROP, FACE_CROP),
                                  interpolation=cv2.INTER_CUBIC if s < FACE_CROP else cv2.INTER_AREA)
                fimg = _image(mp, crop)
            else:
                fimg = img
            r = face.detect_for_video(fimg, ts)
            fl, fb, fm = np.full((478, 3), nan, np.float32), np.full(52, nan, np.float32), np.full((4, 4), nan, np.float32)
            if r.face_landmarks:
                lm = np.array([[l.x, l.y, l.z] for l in r.face_landmarks[0]], np.float32)
                if roi is not None:                    # crop-normalised -> full-frame normalised
                    lm[:, 0] = (x0 + lm[:, 0] * s) / W
                    lm[:, 1] = (y0 + lm[:, 1] * s) / H
                    lm[:, 2] = lm[:, 2] * s / W
                fl = lm
                if r.face_blendshapes:
                    for c in r.face_blendshapes[0]:
                        fb[c.index] = c.score
                if r.facial_transformation_matrixes:
                    fm = np.asarray(r.facial_transformation_matrixes[0], np.float32).reshape(4, 4)
            face_lm.append(fl)
            face_bs.append(fb)
            face_mat.append(fm)
            i += 1
            if i % 150 == 0:
                say(f"  clip {job['id']}: {i}/{n_est or '?'} frames, {time.time() - t0:.0f} s")
    finally:
        cap.release()
        for m in (face, hand, pose):
            m.close()
    if i < 2:
        raise RefError(f"clip {job['id']}: no frames could be decoded from {path}")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    with open(tmp, "wb") as fh:
        np.savez_compressed(
            fh, schema=SCHEMA, tracker=f"mediapipe {mp.__version__}", fps=fps, W=W, H=H, n=i,
            roi=np.array(roi if roi else [-1, -1, -1]), face_lm=np.stack(face_lm), face_bs=np.stack(face_bs),
            face_mat=np.stack(face_mat), hand_lm=np.stack(hand_lm), pose_lm=np.stack(pose_lm),
            cam=np.array(cam, np.float32), cam_inl=np.array(cam_inl, np.float32), fdiff=np.array(fdiff, np.float32))
    tmp.replace(out)
    return {"id": job["id"], "frames": i, "seconds": round(i / fps, 2), "truncated": bool(i >= max_frames and n_est > i),
            "fps": fps, "size": [W, H], "head_roi": list(roi) if roi else None, "track_seconds": round(time.time() - t0, 1),
            "track_bytes": out.stat().st_size}


def _job_run(job):
    """Pool entry point: failures come back as data so one bad clip does not stop the others."""
    try:
        return track_clip(job)
    except Exception as e:                      # noqa: BLE001 - reported per clip
        return {"id": job["id"], "error": f"{type(e).__name__}: {e}", "log": str(Path(job["out"]).with_suffix(".log"))}


# ============================================================================================ the set
def fetch_video(rs, entry, min_height=720, max_bytes=pexels.MAX_DOWNLOAD_BYTES):
    """Download a clip's smallest rendition >= min_height (capped) unless it is already there; returns its path and the
    rendition description."""
    dest = rs.video_path(entry["id"])
    pick = pexels.pick_rendition(entry.get("files") or [], min_height, max_bytes)
    if pick is None:
        raise RefUsage(f"clip {entry['id']}: no mp4 rendition with a short side >= {min_height} px and <= "
                       f"{max_bytes / 1e6:.0f} MB ({[pexels.describe_rendition(f) for f in entry.get('files') or []]})")
    if dest.exists():
        return dest, pexels.describe_rendition(pick)
    try:
        pexels.download(pick["link"], dest, max_bytes, pick.get("size"), label=dest.name)
    except RefError:
        # the stored link may have expired: ask the API for fresh ones once
        fresh = pexels.clip_entry(pexels.video(entry["id"]))
        rs.put(fresh)
        pick = pexels.pick_rendition(fresh["files"], min_height, max_bytes)
        if pick is None:
            raise
        pexels.download(pick["link"], dest, max_bytes, pick.get("size"), label=dest.name)
    return dest, pexels.describe_rendition(pick)


def track_set(rs, jobs=None, min_height=720, max_seconds=MAX_SECONDS, force=False):
    """Download (capped) and track every clip of the set that has no track yet. Returns a report dict."""
    import concurrent.futures as cf
    import multiprocessing as mpc
    clips = rs.require_clips()
    todo = [c for c in clips if force or not rs.track_path(c["id"]).exists()]
    report = {"set": rs.name, "tracked": [], "skipped": [c["id"] for c in clips if c not in todo], "failed": []}
    if not todo:
        return report
    try:
        import mediapipe  # noqa: F401
    except ImportError:
        raise RefError("mediapipe is not installed in this environment: `uv tool install --editable '.[ref]'` "
                       "(or pip install 'mk-mmd[ref]')")
    mdl = models.ensure()
    work = []
    for c in todo:
        try:
            video, rendition = fetch_video(rs, c, min_height)
        except (RefError, RefUsage) as e:
            report["failed"].append({"id": c["id"], "error": str(e)})
            continue
        work.append({"id": c["id"], "video": str(video), "out": str(rs.track_path(c["id"])), "models": mdl,
                     "max_seconds": max_seconds, "rendition": rendition})
    if jobs is None:
        jobs = max(1, min(3, (os.cpu_count() or 2) // 4))          # MediaPipe threads already use every core: more jobs hardly help
    jobs = max(1, min(jobs, len(work)))
    if not work:
        return report
    print(f"  tracking {len(work)} clip(s) with {jobs} job(s)", file=sys.stderr, flush=True)
    with cf.ProcessPoolExecutor(max_workers=jobs, mp_context=mpc.get_context("spawn")) as pool:
        futs = {pool.submit(_job_run, w): w for w in work}
        for fut in cf.as_completed(futs):
            job = futs[fut]
            try:
                res = fut.result()
            except Exception as e:              # noqa: BLE001 - a crashed worker (native code) takes its siblings down too
                res = {"id": job["id"], "error": f"tracker process died ({type(e).__name__}); native log: {Path(job['out']).with_suffix('.log')}"}
            if "error" in res:
                report["failed"].append(res)
                continue
            res["rendition"] = job["rendition"]
            report["tracked"].append(res)
            print(f"  tracked {res['id']}: {res['frames']} frames in {res['track_seconds']:.0f} s", file=sys.stderr, flush=True)
    report["tracked"].sort(key=lambda r: r["id"])
    return report
