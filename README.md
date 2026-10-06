# film-teardown

I kept trying to remake launch films I liked, and I kept getting them wrong. I would watch a film a few times, write down what I thought it did, and hand that to a renderer. The result never felt like the original. The reason was simple: what I remembered was not what the film did. I remembered "fast cuts and a big drop". The film had 37 cuts on exact frames, a camera that moved on four shots and stood still on the rest, and a drop that landed 0.12 s after a cut.

So I stopped describing films and started measuring them. This repo is the method, written as a skill an AI agent can follow, plus the two scripts that do the measuring.

## What it does

You give it a video file. It gives you back a spec you can rebuild from:

- a cut table, with every cut on its exact frame;
- the camera move inside each shot (zoom, pan, rotation), so you know which shots are stills;
- the palette per shot and per act;
- for grid or split-screen films, which panel is lit when, and where each panel cuts;
- the sound: tempo, beats, onsets, loudness per frame, and how far each cut sits from the nearest beat;
- a contact sheet with one labelled frame per shot.

From those numbers you write the anatomy of the film: the one rule every shot obeys, the acts, the rhythm, the motion law, the layout, the type, the look, the sound and the ending. Then you split what makes the film recognizable (invariants) from what can change (words, footage, brand). The invariants become a remake prompt stated as numbers, so the remake can be checked against it.

## How to use it

```bash
pip install opencv-python-headless numpy pillow scipy librosa
python scripts/dissect.py reference.mp4 out/    # cuts, camera moves, palette, contact sheet, audio sync
python scripts/panels.py  reference.mp4 out/    # only for grid / split-screen films
```

Read `out/contact.jpg` first, then `out/shots.json` and `out/audio.json`. `SKILL.md` walks through the whole method: getting the file, measuring, writing the anatomy, splitting invariants from variables, and verifying a remake by running the same scripts on it and diffing the cut lists.

If you use an agent such as Claude Code, drop this folder into your skills directory and ask it to take a film apart. It will follow `SKILL.md`.

## What I learned doing it

- Measure before you describe. A two-second sample misses the rules that make a film work.
- A full-frame cut scan fails on grid films, because cuts in separate panels mix together. That is why `panels.py` exists.
- Speech-to-text invents words over music. Trust a transcribed line only where the loudness shows a voice.
- Check every number against real frames. Camera-move estimates fail on motion blur, so confirm any move with two full-resolution frames.
