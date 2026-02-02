\
import argparse, time, threading, sys, shutil, math
import numpy as np
import sounddevice as sd
from scipy.signal import welch, find_peaks

try:
    import msvcrt
except ImportError:
    msvcrt = None


"""
bmai.py

Full-screen spectrum visualization + sweep-lock + modulation estimate.

It treats the input as "noise + interference" from an audio front-end with a
floating cable (antenna) plugged into mic/line-in.

Displays:
- log-spaced band powers (as bars) from the PSD
- per-band peak frequency marker ("needle")
- sweep-lock: finds strongest narrowband peak and tracks it with Goertzel
- modulation: estimates dominant low-frequency modulation of the tracked lock

Hotkeys:
- b : re-baseline lock_rel (Δlock = 0)
- l : relock
"""

FS = 48000
CHANNELS = 1

PRINT_HZ = 10
FRAME_SECONDS = 0.8
HOP_SECONDS = 1.0 / PRINT_HZ
NFFT = 8192

SMOOTH_BANDS = 0.85
SMOOTH_LOCK  = 0.80

LOCK_FMIN = 40
LOCK_FMAX = 20000
LOCK_PROM_DB = 6
LOCK_REFRESH_SEC = 3.0

FMIN, FMAX = 20, 18000
GAMMA = 0.55
FILL = "█"
NEEDLE = "│"
TAIL = "▏"

MOD_SECONDS = 25
MOD_FS = PRINT_HZ
BREATH_BAND = (0.08, 0.50)
HEART_BAND  = (0.70, 3.00)

def cursor_to(r, c=1):
    sys.stdout.write(f"\x1b[{r};{c}H")

def clear_to_eol():
    sys.stdout.write("\x1b[K")

def hide_cursor():
    sys.stdout.write("\x1b[?25l")

def show_cursor():
    sys.stdout.write("\x1b[?25h")

def goertzel_power(x, fs, f0):
    w = 2.0 * np.pi * f0 / fs
    cw = np.cos(w)
    coeff = 2.0 * cw
    s0 = s1 = s2 = 0.0
    for v in x:
        s0 = v + coeff * s1 - s2
        s2 = s1
        s1 = s0
    return s1*s1 + s2*s2 - coeff*s1*s2

def make_log_edges(fmin, fmax, nbands):
    edges = np.logspace(np.log10(fmin), np.log10(fmax), nbands + 1)
    edges = np.round(edges).astype(int)
    out = [int(edges[0])]
    for e in edges[1:]:
        e = int(e)
        if e > out[-1]:
            out.append(e)
    out[-1] = fmax
    return np.array(out, dtype=int)

def bandstats_from_psd(f, pxx, edges):
    integ = getattr(np, "trapezoid", None) or np.trapz
    logp = 10.0 * np.log10(pxx + 1e-24)
    out_pow = []
    out_pk = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (f >= a) & (f < b)
        if not np.any(m):
            out_pow.append(0.0)
            out_pk.append((0.0, -999.0))
            continue
        out_pow.append(float(integ(pxx[m], f[m])))
        idx = int(np.argmax(logp[m]))
        out_pk.append((float(f[m][idx]), float(logp[m][idx])))
    return np.array(out_pow, dtype=np.float64), out_pk

def sweep_lock_freq(x):
    x = x.astype(np.float64)
    x -= np.mean(x)
    f, pxx = welch(x, fs=FS, nperseg=NFFT, noverlap=NFFT//2, scaling="density")
    logp = 10.0 * np.log10(pxx + 1e-24)
    m = (f >= LOCK_FMIN) & (f <= LOCK_FMAX)
    f2, logp2 = f[m], logp[m]
    if len(f2) < 10:
        return None
    peaks, props = find_peaks(logp2, prominence=LOCK_PROM_DB)
    if len(peaks) == 0:
        return None
    prom = props["prominences"]
    best = peaks[int(np.argmax(prom))]
    return float(f2[best])

def fmt_hz(hz):
    if hz is None:
        return "none"
    return f"{hz/1000:.2f}kHz" if hz >= 1000 else f"{hz:.1f}Hz"

def right_bar(bar_w, frac, needle_frac):
    frac = 0.0 if frac < 0 else (1.0 if frac > 1 else frac)
    needle_frac = 0.0 if needle_frac < 0 else (1.0 if needle_frac > 1 else needle_frac)
    n = int(frac * bar_w)
    n = max(0, min(bar_w, n))
    s = [" "] * bar_w
    for i in range(bar_w - n, bar_w):
        s[i] = FILL
    p = int(needle_frac * (bar_w - 1))
    p = max(0, min(bar_w - 1, p))
    s[p] = NEEDLE
    s[-1] = TAIL
    return "".join(s)

def sparkline(vals, width):
    if len(vals) == 0:
        return " " * width
    v = np.array(vals[-width:], dtype=np.float64)
    if len(v) < width:
        v = np.pad(v, (width - len(v), 0), mode="edge")
    lo, hi = float(np.min(v)), float(np.max(v))
    rng = (hi - lo) or 1.0
    chars = " ▁▂▃▄▅▆▇█"
    out = []
    for x in v:
        t = (x - lo) / rng
        k = int(round(t * 8))
        k = max(0, min(8, k))
        out.append(chars[k])
    return "".join(out)

def dominant_in_band(x, fs, band):
    if len(x) < int(fs * 6):
        return None, None
    x = np.asarray(x, dtype=np.float64)
    x = x - np.mean(x)
    nper = min(256, len(x))
    f, pxx = welch(x, fs=fs, nperseg=nper, noverlap=nper//2, scaling="density")
    m = (f >= band[0]) & (f <= band[1])
    if not np.any(m):
        return None, None
    idx = int(np.argmax(pxx[m]))
    ff = float(f[m][idx])
    pp = float(pxx[m][idx] + 1e-30)
    db = 10.0 * math.log10(pp)
    return ff, db

def list_devices():
    devs = sd.query_devices()
    hostapis = sd.query_hostapis()
    print("Input devices:")
    for i, d in enumerate(devs):
        if d.get("max_input_channels", 0) > 0:
            ha = hostapis[d["hostapi"]]["name"]
            print(f"  [{i}] {d['name']}  (in={d['max_input_channels']})  hostapi={ha}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--allow-wdmks", action="store_true")
    args = ap.parse_args()

    if args.list:
        list_devices()
        return

    device = args.device
    if device is None:
        print("Run: python bmai.py --list")
        print("Then: python bmai.py --device <INDEX>")
        sys.exit(1)

    devinfo = sd.query_devices(device)
    ha_name = sd.query_hostapis(devinfo["hostapi"])["name"]
    print(f"Using device {device}: {devinfo['name']} (hostapi={ha_name})")
    if (not args.allow-wdmks) and ("WDM-KS" in ha_name):
        print("\nWDM-KS often produces NO callbacks. Pick WASAPI/MME/DirectSound.\n")
        sys.exit(2)

    frame_n = int(FS * FRAME_SECONDS)
    hop_n = max(1, int(FS * HOP_SECONDS))

    ring = np.zeros(frame_n, dtype=np.float32)
    write_pos = 0
    filled = 0

    baseline_lock = None
    baseline_guard = threading.Lock()

    lock_freq = None
    lock_guard = threading.Lock()
    last_lock_time = 0.0

    sm_band = None
    sm_lock_db = None

    mod_len = int(MOD_SECONDS * MOD_FS)
    lock_hist = []

    last_lines = {}

    def keythread():
        nonlocal baseline_lock, lock_freq
        while True:
            if msvcrt and msvcrt.kbhit():
                ch = msvcrt.getch().lower()
                if ch == b"b":
                    with baseline_guard:
                        baseline_lock = None
                elif ch == b"l":
                    with lock_guard:
                        lock_freq = None
            time.sleep(0.01)

    if msvcrt:
        threading.Thread(target=keythread, daemon=True).start()

    def callback(indata, frames, time_info, status):
        nonlocal ring, write_pos, filled
        x = indata[:, 0].astype(np.float32)
        n = len(x)
        if n >= frame_n:
            ring[:] = x[-frame_n:]
            write_pos = 0
            filled = frame_n
            return
        end = write_pos + n
        if end <= frame_n:
            ring[write_pos:end] = x
        else:
            k = frame_n - write_pos
            ring[write_pos:] = x[:k]
            ring[:end - frame_n] = x[k:]
        write_pos = (write_pos + n) % frame_n
        filled = min(frame_n, filled + n)

    sys.stdout.write("\x1b[2J\x1b[H")
    hide_cursor()
    sys.stdout.flush()

    try:
        with sd.InputStream(
            samplerate=FS, channels=CHANNELS, dtype="float32",
            device=device, blocksize=hop_n, callback=callback
        ):
            while True:
                time.sleep(HOP_SECONDS)
                if filled < frame_n:
                    continue

                x = np.concatenate((ring[write_pos:], ring[:write_pos])).astype(np.float32)
                x64 = x.astype(np.float64)
                x64 -= np.mean(x64)
                rms = float(np.sqrt(np.mean(x64 * x64)) + 1e-12)

                cols, rows = shutil.get_terminal_size((150, 45))
                header_lines = 8
                nbands = max(8, rows - header_lines)
                edges = make_log_edges(FMIN, FMAX, nbands)

                f, pxx = welch(x64, fs=FS, nperseg=NFFT, noverlap=NFFT//2, scaling="density")
                band, band_pk = bandstats_from_psd(f, pxx, edges)

                if sm_band is None or len(sm_band) != len(band):
                    sm_band = band.copy()
                else:
                    sm_band = SMOOTH_BANDS * sm_band + (1 - SMOOTH_BANDS) * band

                # Sweep-lock: choose strongest narrow peak, then track it with Goertzel
                now = time.time()
                with lock_guard:
                    lf = lock_freq
                if lf is None or (now - last_lock_time) > LOCK_REFRESH_SEC:
                    cand = sweep_lock_freq(x64)
                    if cand is not None:
                        with lock_guard:
                            lock_freq = cand
                        lf = cand
                        last_lock_time = now

                if lf is not None:
                    p_lock = float(goertzel_power(x64, FS, lf) + 1e-24)
                    p_ref = float(np.mean(x64 * x64) + 1e-24)
                    lock_rel_db = float(10.0 * np.log10(p_lock / p_ref))
                    sm_lock_db = lock_rel_db if sm_lock_db is None else (SMOOTH_LOCK * sm_lock_db + (1 - SMOOTH_LOCK) * lock_rel_db)
                    lock_hist.append(sm_lock_db)
                    if len(lock_hist) > mod_len:
                        lock_hist = lock_hist[-mod_len:]
                else:
                    sm_lock_db = None

                with baseline_guard:
                    if baseline_lock is None and sm_lock_db is not None:
                        baseline_lock = sm_lock_db
                    dlock = (sm_lock_db - baseline_lock) if (sm_lock_db is not None and baseline_lock is not None) else 0.0

                fb, _ = dominant_in_band(lock_hist, MOD_FS, BREATH_BAND)
                fh, _ = dominant_in_band(lock_hist, MOD_FS, HEART_BAND)

                eps = 1e-30
                lg = np.log10(np.maximum(sm_band, eps))
                lo, hi = float(np.min(lg)), float(np.max(lg))
                rng = (hi - lo) or 1.0

                label_w = 12
                meta_w = 14
                bar_w = max(12, cols - (label_w + meta_w))

                lines = []
                lines.append("bmai — spectrum + sweep-lock + modulation (b=rebase, l=relock)")
                lines.append(f"input rms: {rms:.6f}")
                lines.append(f"lock: {fmt_hz(lf)}   lock_rel: {('%.2f dB' % sm_lock_db) if sm_lock_db is not None else 'none'}   Δlock: {dlock:+.2f} dB")
                lines.append(f"bands:{len(edges)-1}  {edges[0]}..{edges[-1]}Hz(log)  hz:{PRINT_HZ}  smooth:{SMOOTH_BANDS}/{SMOOTH_LOCK}  gamma:{GAMMA}")
                spark_w = max(20, min(cols - 20, 80))
                lines.append(f"lock mod: {sparkline(lock_hist, spark_w)}")
                if fb is None:
                    lines.append("breath: (need more time)     heart: (need more time)")
                else:
                    bpm_b = fb * 60.0
                    bpm_h = (fh * 60.0) if fh is not None else None
                    lines.append(
                        f"breath: {fb:5.2f} Hz ({bpm_b:5.1f}/min)    "
                        f"heart: {('%.2f Hz (%.1f/min)' % (fh, bpm_h)) if fh is not None else 'none'}"
                    )
                lines.append("")

                for i in range(len(edges) - 1):
                    a, b = int(edges[i]), int(edges[i+1])
                    t = (lg[i] - lo) / rng
                    t = 0.0 if t < 0 else (1.0 if t > 1 else t)
                    t = t ** GAMMA
                    pk_hz, pk_db = band_pk[i]
                    needle_frac = (pk_hz - a) / (b - a) if (pk_hz > 0 and b > a) else 0.0
                    bar = right_bar(bar_w, t, needle_frac)
                    label = f"{a:5d}-{b:<5d}"
                    meta = f"{pk_hz:5.0f} {pk_db:5.0f}"
                    lines.append(f"{label:>{label_w}}{bar}{meta:>{meta_w}}")

                max_lines = min(rows, len(lines))
                for r in range(1, max_lines + 1):
                    s = lines[r - 1]
                    if last_lines.get(r) != s:
                        cursor_to(r, 1)
                        sys.stdout.write(s[:cols])
                        clear_to_eol()
                        last_lines[r] = s

                sys.stdout.flush()

    except KeyboardInterrupt:
        pass
    finally:
        show_cursor()
        cursor_to(shutil.get_terminal_size((150, 45)).lines, 1)
        print("\nStopped.")

if __name__ == "__main__":
    main()
