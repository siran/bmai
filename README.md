# bmai — Jack Noise + Spectrum Locks (Presence & Modulation)

This repo explores a surprisingly sensitive “sensor” you can build with **no mic capsule** and **no explicit sensor**:

- Plug a cable into your computer’s **mic/line-in** jack.
- Leave the far end **floating** (acts like an antenna).
- Record the input as an audio stream.
- Analyze it as **noise + interference**, not “audio content”.

Even when nothing is “connected”, the analog front-end + floating cable can pick up a mix of:
- acoustic bleed / mechanical coupling,
- mains hum & harmonics (50/60 Hz + multiples),
- body proximity (capacitance/grounding changes),
- general electromagnetic interference in the environment.

The goal is to **measure and visualize** what changes, then test hypotheses with controls.

---

## Scripts

### 1) `live_noise_scope.py`
Quick live scope: RMS + bandpowers + strongest spectral peaks.

Run:
```bash
python live_noise_scope.py --list
python live_noise_scope.py --device 5
```

### 2) `bmai.py`
Full-screen spectrum view + sweep-lock + modulation estimate (breath/heart bands).

Run:
```bash
python bmai.py --list
python bmai.py --device 5
```

### 3) `presence_lock.py`
Practical “presence detector”:
- chooses several stable peaks (“locks”),
- tracks them with Goertzel,
- compares current lock-shape vector to a baseline → presence index,
- TUI + optional matplotlib plot,
- CSV logging with rich columns.

Run:
```bash
python presence_lock.py --list
python presence_lock.py --device 5 --plot --csv run.csv
```

Hotkeys:
- `b` baseline (5s delay: step away)
- `l` force relock
- `q` quit

### 4) `analyze_csv.py`
Offline plotting + simple stats from a recorded CSV.

Run:
```bash
python analyze_csv.py run.csv
```

---

## Hardware setup

- Plug any cable into the mic/line-in jack.
- Leave the other end floating (unconnected).
- Keep cable layout consistent when testing.

**Windows note:** Many drivers apply “Enhancements”, AGC, noise suppression.
Disable them for raw behavior if you can.

---

## What to do next (repeatable tests)

1) **Proximity test**
Baseline empty room → approach cable with hand → step away.

2) **Grounding test**
Touch a grounded object vs stand isolated.

3) **Cable geometry**
Coil vs straight vs near power brick.

4) **Controls**
Repeat with:
- cable unplugged,
- different audio device / host API,
- laptop on battery vs plugged in,
- Wi‑Fi off vs on.

---

## CSV columns (what gets logged)

The logger is designed for later science.

Core:
- `t` (unix seconds), `dt_iso` (UTC-ish iso), `frame_idx`
- `idx` (presence index), `here` (0/1), `here_thresh`
- `rms_dbfs`

Baseline + decomposition (so you can debug *why* idx changed):
- `baseline_set` (0/1 when baseline captured this frame)
- `dshape`, `dhum`, `dhf`
- `hum_rel_db`, `hf_floor_db`

Locks (for K locks, default K=6):
- `lock_hz_0..K-1`
- `lock_db_0..K-1`   (smoothed rel power, dB, Goertzel / total power)
- `track_k`, `track_hz`, `track_db` (the currently strongest lock)

Modulation summary:
- `breath_hz`, `heart_hz` (dominant peak estimates from `track_db` history)

---

## Requirements

```bash
pip install -r requirements.txt
```

---

## License

Pick one: MIT / CC-BY / etc.
