# presence_lock.py — Presence via EM‑coupled modulation structure

This tool detects **presence** by tracking **structure** (not raw power) in EM‑coupled noise that appears inside an audio interface bandwidth.

Truthful claims:
- Detects presence via **EM‑coupled modulation patterns**.
- Sees **low‑frequency structure** (0.5–30 Hz) in the **envelope** of high‑frequency spurs.
- Sensitive to proximity, posture, and movement.

Important limits:
- With `FS=48000`, the spectrum view is limited to **0–24 kHz (Nyquist)**.
- This is **not** an RF spectrum analyzer. It observes **demodulated / aliased** products that land in audio/ultrasonic.

---

## Requirements

Python packages:
- `numpy`
- `scipy`
- `sounddevice`
- `matplotlib` (optional, for plots)

Windows: `msvcrt` is used for hotkeys (built-in).

Install example:
```bash
pip install numpy scipy sounddevice matplotlib
```

---

## Quick start

### 1) One‑time device selection (writes cache)
Sweeps input devices, scores them, and saves the best to a cache JSON.

```bash
python presence_lock.py --setup --setup-plot
```

This creates a cache file next to the script:
- `presence_lock_device_cache.json`

### 2) Normal run (no device arg needed)
Auto-uses the cached device.

```bash
python presence_lock.py --plot-spectrum
```

---

## Hotkeys (while running)

- `b` = capture baseline (step away + stay quiet)
- `l` = force relock (re-pick HF carriers)
- `q` = quit

Baseline matters. Do it in a stable “no‑presence” state.

---

## What to look at

### Console lines
- `pres| idx: ... HERE/----`  
  Presence index (structure change) + threshold.

- `env | ... (envΔ=..., lockΔ=...)`  
  Envelope-structure distance (0.5–30 Hz) and lock-shape distance.

- `band| delta/theta/alpha/beta`  
  Dominant modulation peaks in each band (if enough history).

### Plots (`--plot-spectrum`)
- Presence index vs time + threshold
- Envelope structure distance vs time
- Envelope spectrum (0.5–30 Hz)
- Audio-band PSD (optional) with mains ladder markers

---

## Recommended settings

### For alpha/beta (~8–30 Hz) structure
Keep envelope sampling high enough:
- Default: `--env-hz 64`  → Nyquist 32 Hz (covers up to 30 Hz)

You can raise it (CPU cost rises slightly):
```bash
python presence_lock.py --plot-spectrum --env-hz 100
```

### Make index mostly envelope-driven
```bash
python presence_lock.py --plot-spectrum --w-shape 0 --w-hum 0 --w-hf 0 --w-env 6
```

---

## Common issues

### “NO AUDIO CALLBACKS”
That device/host API is not providing input callbacks (common with some WDM-KS/WASAPI modes).
Fix:
- Run `--setup` and choose a different device/host API.
- Avoid WDM-KS unless you know it works (or use `--allow-wdmks`).

### Plot shows only to ~20k
Matplotlib tick labels may stop at 20k even if axis is 24k. Force it:
```bash
python presence_lock.py --plot-spectrum --plot-fmax 24000
```

---

## Output / logging

Enable CSV:
```bash
python presence_lock.py --csv run.csv --plot-spectrum
```

The CSV includes:
- presence index, threshold, lock frequencies + levels
- envelope band dominants (delta/theta/alpha/beta)
- hum/hf/voice proxies

---

## Safety + interpretation

This tool **does not** read brain waves.  
It detects **presence-related coupling changes** that modulate ambient EM‑coupled carriers into audio/ultrasonic bands.

If you want true MHz–GHz RF analysis, you need an SDR or spectrum analyzer.
