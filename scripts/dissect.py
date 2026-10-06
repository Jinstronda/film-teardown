#!/usr/bin/env python3
"""Frame-exact teardown of a reference film.

usage: dissect.py VIDEO OUTDIR [--width 480] [--k 4.0] [--floor 6.0]

Writes OUTDIR/frames.csv (per-frame stats), shots.json (cut table, per-shot
camera move, palette), contact.jpg (one mid frame per shot, labelled) and
audio.json (tempo, beats, onsets, cut-to-onset offsets). Frames are 1-based,
so "cut at f32" means f32 is the first frame of the new shot.
"""
import argparse, json, os, subprocess
import cv2, numpy as np


def read_frames(path, width):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        h = int(f.shape[0] * width / f.shape[1])
        frames.append(cv2.resize(f, (width, h), interpolation=cv2.INTER_AREA))
    return frames, fps


def frame_stats(frames):
    rows, prev = [], None
    for i, f in enumerate(frames):
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
        hist = cv2.calcHist([f], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3).ravel()
        hist /= hist.sum() + 1e-9
        diff = float(np.abs(g - prev[0]).mean()) if prev is not None else 0.0
        hd = float(0.5 * np.abs(hist - prev[1]).sum()) if prev is not None else 0.0
        b, gr, r = f.reshape(-1, 3).mean(0)
        rows.append(dict(f=i + 1, diff=diff, hist=hd, luma=float(g.mean()), r=float(r), g=float(gr), b=float(b)))
        prev = (g, hist)
    return rows


def detect_cuts(rows, k, floor):
    # spike = diff well above its local median AND histogram moved; local
    # window so slow pans and grain don't trigger
    d = np.array([r["diff"] for r in rows])
    h = np.array([r["hist"] for r in rows])
    cuts = []
    for i in range(1, len(d)):
        lo, hi = max(1, i - 6), min(len(d), i + 7)
        neigh = np.delete(d[lo:hi], i - lo)
        med = np.median(neigh) if len(neigh) else 0
        if d[i] > max(floor, k * med + 1.0) and (h[i] > 0.12 or d[i] > 3 * floor):
            cuts.append(i + 1)
    return cuts


def shot_motion(a, b):
    # similarity transform from first to last frame of a shot
    orb = cv2.ORB_create(1500)
    ga, gb = (cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in (a, b))
    ka, da = orb.detectAndCompute(ga, None)
    kb, db = orb.detectAndCompute(gb, None)
    if da is None or db is None or len(ka) < 12 or len(kb) < 12:
        return None
    m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, db)
    if len(m) < 12:
        return None
    pa = np.float32([ka[x.queryIdx].pt for x in m])
    pb = np.float32([kb[x.trainIdx].pt for x in m])
    M, inl = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=2.0)
    if M is None or inl.sum() < 10:
        return None
    s = float(np.hypot(M[0, 0], M[1, 0]))
    return dict(scale=round(s, 4), rot_deg=round(float(np.degrees(np.arctan2(M[1, 0], M[0, 0]))), 2),
                dx=round(float(M[0, 2]), 1), dy=round(float(M[1, 2]), 1), inliers=int(inl.sum()))


def palette(img, n=4):
    z = img.reshape(-1, 3).astype(np.float32)
    _, lab, cen = cv2.kmeans(z, n, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0), 2, cv2.KMEANS_PP_CENTERS)
    share = np.bincount(lab.ravel(), minlength=n) / len(lab)
    order = np.argsort(-share)
    return [dict(hex="#%02x%02x%02x" % tuple(int(c) for c in cen[i][::-1]), share=round(float(share[i]), 3)) for i in order]


def contact_sheet(frames, shots, out, cols=6):
    tw = 320
    th = int(frames[0].shape[0] * tw / frames[0].shape[1])
    rows = (len(shots) + cols - 1) // cols
    sheet = np.full((rows * (th + 22), cols * tw, 3), 20, np.uint8)
    for n, s in enumerate(shots):
        mid = frames[(s["start"] + s["end"]) // 2 - 1]
        t = cv2.resize(mid, (tw, th))
        y, x = (n // cols) * (th + 22), (n % cols) * tw
        sheet[y:y + th, x:x + tw] = t
        cv2.putText(sheet, f"S{n+1} f{s['start']}-{s['end']} ({s['len']})", (x + 4, y + th + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.imwrite(out, sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])


def audio_report(video, cuts, fps, outdir):
    wav = os.path.join(outdir, "audio.wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", video, "-ac", "1", "-ar", "22050", wav], check=True)
    import librosa
    y, sr = librosa.load(wav, sr=22050)
    if not len(y) or np.abs(y).max() < 1e-4:
        return dict(silent=True)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time", backtrack=False)
    beat_t = librosa.frames_to_time(beats, sr=sr)
    rms = librosa.feature.rms(y=y, hop_length=int(sr / fps))[0]
    cut_t = [(c - 1) / fps for c in cuts]
    near = lambda t, arr: float(min(arr, key=lambda a: abs(a - t)) - t) if len(arr) else None
    return dict(tempo=float(np.atleast_1d(tempo)[0]), beats=[round(float(b), 3) for b in beat_t],
                onsets=[round(float(o), 3) for o in onsets],
                cut_to_onset_s=[round(near(t, onsets), 3) for t in cut_t],
                cut_to_beat_s=[round(near(t, beat_t), 3) for t in cut_t],
                rms_db_per_frame=[round(float(20 * np.log10(r + 1e-6)), 1) for r in rms])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video"); ap.add_argument("outdir")
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--k", type=float, default=4.0)
    ap.add_argument("--floor", type=float, default=6.0)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    frames, fps = read_frames(a.video, a.width)
    rows = frame_stats(frames)
    with open(os.path.join(a.outdir, "frames.csv"), "w") as fh:
        fh.write(",".join(rows[0]) + "\n")
        fh.writelines(",".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r.values()) + "\n" for r in rows)
    cuts = detect_cuts(rows, a.k, a.floor)
    bounds = [1] + cuts + [len(frames) + 1]
    shots = []
    for s, e in zip(bounds, bounds[1:]):
        mid = frames[(s + e - 1) // 2 - 1]
        shots.append(dict(start=s, end=e - 1, len=e - s, t0=round((s - 1) / fps, 3),
                          luma=round(float(np.mean([rows[i - 1]["luma"] for i in range(s, e)])), 1),
                          motion=shot_motion(frames[s - 1], frames[e - 2]) if e - s > 2 else None,
                          palette=palette(mid)))
    json.dump(dict(video=a.video, fps=fps, n_frames=len(frames), cuts=cuts, shots=shots),
              open(os.path.join(a.outdir, "shots.json"), "w"), indent=1)
    contact_sheet(frames, shots, os.path.join(a.outdir, "contact.jpg"))
    json.dump(audio_report(a.video, cuts, fps, a.outdir), open(os.path.join(a.outdir, "audio.json"), "w"))
    print(f"{len(frames)} frames @ {fps} fps, {len(cuts)} cuts, {len(shots)} shots")
    print("cuts:", cuts)


if __name__ == "__main__":
    main()
