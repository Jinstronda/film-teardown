---
name: film-teardown
description: Reverse-engineer a reference film (brand film, launch video, X/YouTube video) frame by frame into a measured spec that can be rebuilt or adapted to another idea. Use when you have a reference video and want to study how it was made, get its essence, make one like it, or before writing any remake prompt. Produces the cut table, camera moves, palette, type behaviour, panel timeline, sound arc, and the invariants vs variables split.
---

# Film teardown

Measure the film before describing it. A description written from memory or from a 2-second
sample misses the rules that make the film work. The Claude Opus 5.5 film (2026-10-04) has 37
cuts, and `dissect.py` found all 37 on the exact frame given in the remake prompt.

## 1. Get the file

```bash
mkdir -p ~/film-study && cd ~/film-study
yt-dlp -f "bv*+ba/b" --merge-output-format mp4 -o "ref_NAME.%(ext)s" URL   # YouTube, X, Vimeo
yt-dlp --dump-json URL | python3 -c "import json,sys;d=json.load(sys.stdin);print(d['title'],d['duration']);print(d.get('description'))"
ffprobe -v error -show_entries stream=codec_type,width,height,r_frame_rate,nb_frames,duration -of compact ref_NAME.mp4
uv venv -q .venv && uv pip install -q --python .venv/bin/python opencv-python-headless numpy pillow scipy librosa
```

Record resolution, fps and frame count. Work in the reference's native frame numbers (1-based).

## 2. Measure

```bash
S=path/to/film-teardown/scripts
.venv/bin/python $S/dissect.py ref_NAME.mp4 out/      # cuts, per-shot camera move, palette, contact sheet, audio sync
.venv/bin/python $S/panels.py  ref_NAME.mp4 out/      # only for split-screen / grid films
whisper ref_NAME.mp4 --model small --language en --word_timestamps True --output_format json --output_dir out/
```

- `out/contact.jpg`: one mid frame per shot, labelled `S# fstart-end (len)`. Read it first.
- `out/shots.json`: per shot `motion` = similarity transform first→last frame (scale, dx, dy,
  rotation, px at 480 wide; multiply by native/480). `scale 1.0 dx 0` means a locked still.
  ORB fails on blur and returns odd rotations; confirm any move with two full-res frames.
- `out/audio.json`: tempo, beats, onsets, `cut_to_onset_s`, `rms_db_per_frame`. Report the
  median offset, not "cuts are on the beat".
- If cuts seem wrong, tune `--k` (spike vs local median) and `--floor`. Grid films need
  `panels.py`, because a full-frame scan mixes cuts from separate panels.
- Whisper invents text over instrumental audio ("In YouTube videos like this..." on the Opus
  film, "AI." over the Prime tail). Trust a line only where `rms` shows a voice-level section.

Then look at real pixels: extract full-res frames at every act boundary and 2x2 them with
`ffmpeg xstack`; crop the type region across 15 to 20 shots into one `tile` grid to see how the
type is treated.

## 3. Write the anatomy

Write each of these from numbers, with frame or second ranges:

1. Concept in one sentence: the single rule every shot obeys.
2. Acts: boundaries, what changes at each (word, density, sound).
3. Rhythm: shot lengths per act as a list. Name the curve (accelerating, breathing, plateau).
4. Motion law: which shots move, by how much, linear or eased. Everything else is a still.
5. Layout: fixed geometry (arc, grid, gutters) in px.
6. Type: face class, case, size progression in px, position, how it sits in the image.
7. Look: grain, blur, palette per act, source material classes.
8. Sound: loudness per second, the drop, the hits, where picture events land on sound.
9. Ending: hold lengths, logo swap frame, loop or hard out.

## 4. Split invariants from variables

Invariants are the rules that make the film recognizable; keep them when adapting. Variables are
content (words, objects, footage, brand, palette) that the new idea replaces. Write the adapted
brief as a parameterized prompt: invariants stated as measured numbers, variables as `<inputs>`
with defaults. A remake prompt written this way states every rule as a number a renderer can check.

## 5. Verify a remake

Run `dissect.py` on the render and diff its cut list against the spec; diff `panels.py` output
for grid films. Build a synced side-by-side with
`ffmpeg -i ref.mp4 -i remake.mp4 -filter_complex hstack=inputs=2`, scaling both to the same height first.

Films taken apart with this method so far: the Claude Opus 5.5 launch film (37 cuts, all found on the exact frame)
and the Prime Intellect Series A film (2x2 screen grid, 2660 frames).
