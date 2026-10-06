#!/usr/bin/env python3
"""Per-panel timeline for split-screen / grid films.

usage: panels.py VIDEO OUTDIR [--win 3] [--cut 18]

Finds gutters (rows/cols dark in >=60% of sampled frames), splits the frame
into panels, then prints per window: which panels are lit, cuts inside
panels, loudness. Writes OUTDIR/panels.json.
"""
import argparse, json, os, subprocess
import cv2, numpy as np


def runs(idx):
    out = []
    for i in idx:
        if out and i == out[-1][1] + 1:
            out[-1][1] = i
        else:
            out.append([i, i])
    return out


def gutters(path, samples=60):
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cols = rows = None
    for k in range(samples):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * (k + 0.5) / samples))
        ok, f = cap.read()
        if not ok:
            continue
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        c = (np.percentile(g, 90, axis=0) < 30).astype(int)
        r = (np.percentile(g, 90, axis=1) < 30).astype(int)
        cols = c if cols is None else cols + c
        rows = r if rows is None else rows + r
    h, w = g.shape
    gc = runs([i for i, v in enumerate(cols) if v >= 0.6 * samples])
    gr = runs([i for i, v in enumerate(rows) if v >= 0.6 * samples])
    return w, h, gc, gr


def spans(gut, size):
    # inner spans between gutter runs
    edges = [-1] + [x for g in gut for x in g] + [size]
    pairs = list(zip(edges[::2], edges[1::2]))
    return [(a + 1, b) for a, b in pairs if b - a > 40]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video"); ap.add_argument("outdir")
    ap.add_argument("--win", type=float, default=3.0)
    ap.add_argument("--cut", type=float, default=18.0)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    w, h, gc, gr = gutters(a.video)
    xs, ys = spans(gc, w), spans(gr, h)
    panels = {f"r{j}c{i}": (x0, y0, x1, y1) for j, (y0, y1) in enumerate(ys) for i, (x0, x1) in enumerate(xs)}
    print("gutter cols", gc, "\ngutter rows", gr, "\npanels", panels)
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    lum = {k: [] for k in panels}; cuts = {k: [] for k in panels}; prev = {}; i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        i += 1
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)[::2, ::2]
        for k, (x0, y0, x1, y1) in panels.items():
            p = g[y0 // 2:y1 // 2, x0 // 2:x1 // 2]
            lum[k].append(float(p.mean()))
            if k in prev and float(np.abs(p - prev[k]).mean()) > a.cut:
                cuts[k].append(i)
            prev[k] = p
    wav = os.path.join(a.outdir, "panels_audio.wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", a.video, "-ac", "1", "-ar", "8000", wav], check=True)
    import wave
    raw = wave.open(wav).readframes(10 ** 9)
    y = np.frombuffer(raw, np.int16).astype(np.float32) / 32768
    hop = int(8000 / fps)
    rms = [20 * np.log10(np.sqrt(np.mean(y[j * hop:(j + 1) * hop] ** 2)) + 1e-6) for j in range(i)]
    step = int(a.win * fps)
    print(" sec | lit panels | panel cuts | dB")
    for lo in range(0, i, step):
        hi = min(lo + step, i)
        lit = " ".join(k if np.mean(lum[k][lo:hi]) > 12 else "." * len(k) for k in panels)
        nc = sum(sum(1 for c in cuts[k] if lo < c <= hi) for k in panels)
        print(f"{lo / fps:5.1f} | {lit} | {nc:4d} | {np.mean(rms[lo:hi]):6.1f}")
    json.dump(dict(fps=fps, gutter_cols=gc, gutter_rows=gr, panels=panels, lum=lum, cuts=cuts),
              open(os.path.join(a.outdir, "panels.json"), "w"))


if __name__ == "__main__":
    main()
