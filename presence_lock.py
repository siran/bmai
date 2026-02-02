# presence_lock.py
# DROP-IN (Windows) — Presence via EM-coupled modulation structure
#
# What it does (truthful):
# - Finds stable high-frequency “carrier” lines (locks) in the audio-band spectrum
# - Tracks their amplitude over time (envelope)
# - Computes envelope spectrum in 0.5–30 Hz (delta/theta/alpha/beta structure)
# - Builds a presence index from (a) lock-shape change + (b) envelope-structure change
#
# Hotkeys:
#   b = capture baseline (step away/quiet)
#   l = force relock
#   q = quit
#
# Setup + cache:
#   python presence_lock.py --setup --setup-plot
#   then just:
#   python presence_lock.py --plot-spectrum
#
# Notes on “EM frequencies”:
# - With FS=48 kHz you only see up to 24 kHz (Nyquist). This is NOT RF spectrum.
# - You’re seeing demodulated/aliased envelope structure caused by EM pickup + nonlinearity.
#
import argparse, time, threading, sys, math, csv, json
from pathlib import Path
import numpy as np
import sounddevice as sd
from scipy.signal import welch, find_peaks
import msvcrt

# =========================
# Fixed audio parameters
# =========================
FS = 48000
CHANNELS = 1
NFFT = 8192

# Locks (HF carriers)
K_LOCKS = 6
LOCK_FMIN = 80
LOCK_FMAX = 18000
LOCK_PROM_DB = 8
LOCK_REFRESH_SEC = 3.0
LOCK_DROP_DB = 10
LOCK_SEP_HZ = 40.0

# Smoothing
SMOOTH_LOCK  = 0.80
SMOOTH_IDX   = 0.85
SMOOTH_HUM   = 0.85
SMOOTH_HF    = 0.85
SMOOTH_VOICE = 0.85

# Mains proxy
MAINS = 60.0
MAINS_HARMONICS = 6
HF_FLOOR = (2000, 18000)
VOICE_BAND = (200, 4000)

# Envelope (modulation) bands (Hz)
DELTA_BAND = (0.5, 4.0)
THETA_BAND = (4.0, 8.0)
ALPHA_BAND = (8.0, 12.5)
BETA_BAND  = (13.0, 30.0)

# =========================
# Helpers
# =========================
def trapz(y, x):
    integ = getattr(np, "trapezoid", None)
    if integ is None:
        return np.trapz(y, x)
    return integ(y, x)

def db10(x): return 10.0 * math.log10(max(1e-30, float(x)))
def db20(x): return 20.0 * math.log10(max(1e-30, float(x)))

def fmt_hz(hz):
    if hz is None: return "none"
    return f"{hz/1000:.2f}k" if hz >= 1000 else f"{hz:.1f}"

def cosine_distance(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na = np.linalg.norm(a) + 1e-12
    nb = np.linalg.norm(b) + 1e-12
    return 1.0 - float(np.dot(a, b) / (na * nb))

def spark(vals, width):
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

# =========================
# DSP primitives
# =========================
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

def welch_psd(x, fs=FS):
    return welch(x, fs=fs, nperseg=NFFT, noverlap=NFFT//2, scaling="density")

def band_db_from_psd(f, pxx, band):
    a, b = band
    m = (f >= a) & (f <= b)
    if not np.any(m):
        return -999.0
    p = float(trapz(pxx[m], f[m]) + 1e-30)
    return db10(p)

def hf_floor_db(f, pxx):
    logp = 10.0 * np.log10(pxx + 1e-24)
    m = (f >= HF_FLOOR[0]) & (f <= HF_FLOOR[1])
    return float(np.median(logp[m])) if np.any(m) else float(np.median(logp))

def hum_rel_db_goertzel(x, fs=FS, mains=MAINS, nh=MAINS_HARMONICS):
    x = x.astype(np.float64)
    x -= np.mean(x)
    pref = float(np.mean(x*x) + 1e-24)
    ph = 0.0
    for k in range(1, nh + 1):
        ph += float(goertzel_power(x, fs, mains*k) + 1e-24)
    return db10(ph / (pref + 1e-24))

def pick_candidates(f, pxx, fmin, fmax, prom_db):
    logp = 10.0 * np.log10(pxx + 1e-24)
    m = (f >= fmin) & (f <= fmax)
    f2, logp2 = f[m], logp[m]
    if len(f2) < 50:
        return []
    peaks, props = find_peaks(logp2, prominence=prom_db)
    if len(peaks) == 0:
        return []
    prom = props["prominences"]
    order = np.argsort(prom)[::-1]
    out = []
    for idx in order[: max(3*K_LOCKS, 12)]:
        p = peaks[idx]
        out.append((float(f2[p]), float(logp2[p]), float(prom[idx])))
    return out

# =========================
# Envelope PSD (0.5–30 Hz)
# =========================
def envelope_psd(ts, fs_env, fmax=30.0):
    ts = np.asarray(ts, dtype=np.float64)
    if len(ts) < int(fs_env * 6):
        return None, None
    ts = ts - np.mean(ts)
    nper = min(1024, len(ts))
    f, pxx = welch(ts, fs=fs_env, nperseg=nper, noverlap=nper//2, scaling="density")
    m = (f >= 0) & (f <= fmax)
    return f[m], pxx[m]

def env_shape_vector(f, pxx, band=(0.5, 30.0)):
    if f is None or pxx is None:
        return None
    a, b = band
    m = (f >= a) & (f <= b)
    if not np.any(m):
        return None
    v = np.maximum(pxx[m], 1e-30)
    v = v / (np.sum(v) + 1e-12)
    return v

def dominant_in_env_band(f, pxx, band):
    if f is None or pxx is None:
        return None, None
    a, b = band
    m = (f >= a) & (f <= b)
    if not np.any(m):
        return None, None
    idx = int(np.argmax(pxx[m]))
    return float(f[m][idx]), db10(float(pxx[m][idx] + 1e-30))

# =========================
# Cache
# =========================
def default_cache_path():
    return (Path(__file__).resolve().parent / "presence_lock_device_cache.json")

def load_cache(path: Path):
    try:
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def save_cache(path: Path, payload: dict):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return True
    except Exception:
        return False

# =========================
# Device setup
# =========================
def list_devices():
    devs = sd.query_devices()
    hostapis = sd.query_hostapis()
    print("Input devices:")
    for i, d in enumerate(devs):
        if d.get("max_input_channels", 0) > 0:
            ha = hostapis[d["hostapi"]]["name"]
            sr = d.get("default_samplerate", None)
            sr_s = f"{sr:.0f}" if sr else "?"
            print(f"  [{i}] {d['name']}  (in={d['max_input_channels']})  hostapi={ha}  defSR={sr_s}")

def record_device_quick(device_idx, seconds=1.2, fs=FS):
    try:
        x = sd.rec(int(seconds*fs), samplerate=fs, channels=1, dtype="float32", device=device_idx)
        sd.wait()
        x = x[:, 0].astype(np.float64)
        x -= np.mean(x)
        return x, None
    except Exception as e:
        return None, str(e)

def score_device_metrics(x, fs=FS):
    rms = float(np.sqrt(np.mean(x*x)) + 1e-12)
    rms_db = db20(rms)

    f, pxx = welch(x, fs=fs, nperseg=NFFT, noverlap=NFFT//2, scaling="density")
    logp = 10.0 * np.log10(pxx + 1e-24)

    hf = hf_floor_db(f, pxx)
    voice_db = band_db_from_psd(f, pxx, VOICE_BAND)

    mains_strength = 0.0
    for k in range(1, MAINS_HARMONICS + 1):
        fk = MAINS * k
        if fk >= f[-1]:
            break
        idx = int(np.argmin(np.abs(f - fk)))
        mains_strength += max(0.0, float(logp[idx] - np.median(logp[max(0, idx-12):min(len(logp), idx+13)])))

    cands = pick_candidates(f, pxx, 80, 18000, prom_db=6.0)
    n_peaks = len(cands)

    # Higher score is better
    score = 0.0
    score += 1.0 * (-hf)                              # prefer quieter HF floor
    score += -2.0 * max(0.0, mains_strength - 10.0)   # penalize mains ladder
    score += -0.3 * max(0.0, n_peaks - 60)            # penalize peak soup

    # Reject near-silent/broken streams
    if rms_db < -90.0:
        score += -500.0
    elif rms_db < -80.0:
        score += -150.0

    return {
        "score": float(score),
        "rms_dbfs": float(rms_db),
        "hf_floor_db": float(hf),
        "voice_db": float(voice_db),
        "mains_strength": float(mains_strength),
        "n_peaks": int(n_peaks),
        "f": f,
        "logp": logp
    }

def setup_sweep(args, cache_path: Path):
    devs = sd.query_devices()
    hostapis = sd.query_hostapis()

    cand = []
    for i, d in enumerate(devs):
        if d.get("max_input_channels", 0) <= 0:
            continue
        ha = hostapis[d["hostapi"]]["name"]
        if (not args.allow_wdmks) and ("WDM-KS" in ha):
            continue
        cand.append(i)

    if len(cand) == 0:
        print("No usable input devices found (or all filtered).")
        return 2

    plt = None
    if args.setup_plot:
        try:
            import matplotlib.pyplot as _plt
            plt = _plt
        except Exception:
            plt = None
            print("matplotlib not available; continuing without setup plots.")

    results = []
    print("\n[setup] Sweeping input devices (short capture each)...\n")
    for idx in cand:
        d = devs[idx]
        ha = hostapis[d["hostapi"]]["name"]
        name = d["name"]

        x, err = record_device_quick(idx, seconds=args.setup_sec, fs=FS)
        if err:
            print(f"[{idx}] FAIL: {name} ({ha})  err={err}")
            continue

        met = score_device_metrics(x, fs=FS)
        results.append((idx, name, ha, met))
        print(f"[{idx}] ok  score={met['score']:+8.2f}  rms={met['rms_dbfs']:7.1f}dBFS  "
              f"hf={met['hf_floor_db']:7.1f}dB  mainsS={met['mains_strength']:5.1f}  peaks={met['n_peaks']:3d}  {name} ({ha})")

        if plt is not None:
            f = met["f"]; logp = met["logp"]
            fmax_view = min(float(args.plot_fmax), float(f[-1]))
            m = (f >= 0) & (f <= fmax_view)
            plt.ion()
            fig, ax = plt.subplots(1, 1, figsize=(10, 4))
            ax.plot(f[m], logp[m])
            ax.set_title(f"Device [{idx}] PSD: {name} ({ha}) score={met['score']:+.2f}")
            ax.set_xlabel("Hz"); ax.set_ylabel("dB")
            ax.set_xlim(0, fmax_view)
            for k in range(1, MAINS_HARMONICS + 1):
                fk = MAINS*k
                if fk > fmax_view: break
                ax.axvline(fk, linestyle="--", linewidth=1.0)
            plt.pause(0.001)
            print("   [setup] close the plot window to continue...")
            plt.show(block=True)

    if len(results) == 0:
        print("\n[setup] All devices failed.")
        return 3

    results.sort(key=lambda r: r[3]["score"], reverse=True)
    best_idx, best_name, best_ha, best_met = results[0]

    print("\n[setup] Suggestions (top 5):")
    for j, (idx, name, ha, met) in enumerate(results[:5], start=1):
        print(f"  {j}. --device {idx}  score={met['score']:+8.2f}  rms={met['rms_dbfs']:7.1f}dBFS  "
              f"hf={met['hf_floor_db']:7.1f}dB  mainsS={met['mains_strength']:5.1f}  {name} ({ha})")

    payload = {
        "device": int(best_idx),
        "name": str(best_name),
        "hostapi": str(best_ha),
        "fs": int(FS),
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(time.time())),
        "metrics": {
            "score": float(best_met["score"]),
            "rms_dbfs": float(best_met["rms_dbfs"]),
            "hf_floor_db": float(best_met["hf_floor_db"]),
            "mains_strength": float(best_met["mains_strength"]),
            "n_peaks": int(best_met["n_peaks"]),
        }
    }
    if not args.no_cache:
        if save_cache(cache_path, payload):
            print(f"\n[setup] Cached device -> {cache_path}")
        else:
            print(f"\n[setup] Failed to write cache -> {cache_path}")

    print(f"\n[setup] Choosing device {best_idx}: {best_name} ({best_ha})")
    print(f"        characteristics: rms={best_met['rms_dbfs']:.1f}dBFS  hf={best_met['hf_floor_db']:.1f}dB  mainsS={best_met['mains_strength']:.1f}")
    print(f"\n[setup] Recommended: --device {best_idx}\n")
    return 0

# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--list", action="store_true")
    ap.add_argument("--setup", action="store_true")
    ap.add_argument("--setup-sec", type=float, default=1.2)
    ap.add_argument("--setup-plot", action="store_true")

    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--allow-wdmks", action="store_true")

    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--plot-spectrum", action="store_true")
    ap.add_argument("--plot-fmax", type=float, default=(FS/2), help="spectrum plot max Hz (default FS/2)")

    ap.add_argument("--csv", type=str, default=None)

    ap.add_argument("--cache-file", type=str, default=None)
    ap.add_argument("--no-cache", action="store_true")

    # Baseline / thresholding
    ap.add_argument("--baseline-sec", type=float, default=4.0)
    ap.add_argument("--thr-sigma", type=float, default=4.0)

    # Audio rejection
    ap.add_argument("--voice-reject-db", type=float, default=8.0)
    ap.add_argument("--hf-lock-min", type=float, default=8000.0)

    # Envelope sampling (this is crucial for ~10 Hz)
    ap.add_argument("--env-hz", type=float, default=64.0, help="envelope sampling rate (Nyquist ~ env_hz/2)")

    # Presence index weights
    ap.add_argument("--w-shape", type=float, default=4.0, help="weight for lock-shape distance")
    ap.add_argument("--w-env", type=float, default=4.0, help="weight for envelope-structure distance")
    ap.add_argument("--w-hum", type=float, default=0.15)
    ap.add_argument("--w-hf", type=float, default=0.05)

    args = ap.parse_args()

    cache_path = Path(args.cache_file).expanduser().resolve() if args.cache_file else default_cache_path()

    if args.list:
        list_devices()
        return

    if args.setup:
        rc = setup_sweep(args, cache_path)
        sys.exit(rc)

    # Device from cache if not specified
    if args.device is None and not args.no_cache:
        cache = load_cache(cache_path)
        if cache and isinstance(cache, dict) and "device" in cache:
            args.device = int(cache["device"])
            print(f"[cache] using device {args.device}: {cache.get('name','?')} ({cache.get('hostapi','?')})")

    if args.device is None:
        print("Run: python presence_lock.py --setup")
        print("Or : python presence_lock.py --list  (then pick --device INDEX)")
        sys.exit(1)

    devinfo = sd.query_devices(args.device)
    ha_name = sd.query_hostapis(devinfo["hostapi"])["name"]
    print(f"Using device {args.device}: {devinfo['name']} (hostapi={ha_name})")
    if (not args.allow_wdmks) and ("WDM-KS" in ha_name):
        print("\nWDM-KS often produces NO callbacks. Pick WASAPI/MME/DirectSound, or pass --allow-wdmks.\n")
        sys.exit(2)

    # Envelope hop
    PRINT_HZ = float(args.env_hz)
    HOP_SECONDS = 1.0 / PRINT_HZ

    # Frame for PSD / lock measurement (longer for stable peaks)
    FRAME_SECONDS = 0.8
    frame_n = int(FS * FRAME_SECONDS)
    hop_n = max(1, int(FS * HOP_SECONDS))

    plot = None
    if args.plot or args.plot_spectrum:
        import matplotlib.pyplot as plt
        plot = {"plt": plt, "ready": False}

    # CSV
    csv_f = None
    csv_w = None
    if args.csv:
        csv_f = open(args.csv, "w", newline="")
        csv_w = csv.writer(csv_f)
        csv_w.writerow([
            "t","dt_iso","frame_idx",
            "idx","here","thr",
            "rms_dbfs",
            "env_shape_dist","lock_shape_dist",
            "hum_rel_db","hf_floor_db",
            "voice_db","dvoice_db","audio_flag",
            "env_dom_delta_hz","env_dom_theta_hz","env_dom_alpha_hz","env_dom_beta_hz",
            *[f"lock_hz_{k}" for k in range(K_LOCKS)],
            *[f"lock_db_{k}" for k in range(K_LOCKS)],
        ])
        csv_f.flush()

    # Ring buffer
    ring = np.zeros(frame_n, dtype=np.float32)
    write_pos = 0
    filled = 0
    last_cb_time = time.time()

    # Locks
    lock_freqs = [None] * K_LOCKS
    lock_db_s = np.array([-200.0] * K_LOCKS, dtype=np.float64)
    lock_db_best = np.array([-200.0] * K_LOCKS, dtype=np.float64)
    last_refresh = 0.0

    # Baselines
    baseline_lock_shape = None
    baseline_hum = None
    baseline_hf = None
    baseline_voice = None
    baseline_env_shape = None
    here_thresh = None
    baseline_set = False

    # Baseline capture state
    baseline_armed = False
    baseline_until = None
    baseline_msg = "press b to baseline"
    baseline_collect = {
        "lock_shape": [],
        "hum": [],
        "hf": [],
        "voice": [],
        "env_shape": [],
        "idx_tmp": [],
    }

    # Smoothed metrics
    idx_s = 0.0
    hum_s = None
    hf_s = None
    voice_s = None

    idx_hist = []
    env_shape_hist = []

    # Envelope tracking: weighted sum of lock powers over time
    # Keep ~60s window for good 0.5–30 Hz structure
    ENV_SECONDS = 60.0
    env_hist = []

    def callback(indata, frames, time_info, status):
        nonlocal ring, write_pos, filled, last_cb_time
        last_cb_time = time.time()
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
            ring[:end-frame_n] = x[k:]
        write_pos = (write_pos + n) % frame_n
        filled = min(frame_n, filled + n)

    def keythread():
        nonlocal baseline_armed, baseline_until, baseline_msg, baseline_set
        nonlocal baseline_collect, lock_freqs, last_refresh
        while True:
            if msvcrt.kbhit():
                ch = msvcrt.getch().lower()
                if ch == b"b":
                    baseline_armed = True
                    baseline_set = False
                    baseline_until = time.time() + float(args.baseline_sec)
                    baseline_collect = {"lock_shape": [], "hum": [], "hf": [], "voice": [], "env_shape": [], "idx_tmp": []}
                    baseline_msg = f"[baseline] capturing {args.baseline_sec:.1f}s — step away/quiet"
                elif ch == b"l":
                    lock_freqs = [None] * K_LOCKS
                    last_refresh = 0.0
                    baseline_msg = "[locks] forced relock"
                elif ch == b"q":
                    raise SystemExit
            time.sleep(0.02)

    threading.Thread(target=keythread, daemon=True).start()

    print("\nHotkeys: b=baseline, l=relock, q=quit, Ctrl+C=stop\n")
    print("pres|\nlock|\nidx |\nenv |\naud |\nhum |\nhf  |\nband|")

    def move_up(n): sys.stdout.write(f"\x1b[{n}A")
    def clear_eol(): sys.stdout.write("\x1b[K")

    frame_idx = 0

    try:
        with sd.InputStream(
            samplerate=FS, channels=CHANNELS, dtype="float32",
            device=args.device, blocksize=hop_n, callback=callback
        ):
            while True:
                time.sleep(HOP_SECONDS)
                if time.time() - last_cb_time > 2.0:
                    print("NO AUDIO CALLBACKS. Try DirectSound/MME/WASAPI device.")
                    continue
                if filled < frame_n:
                    continue

                frame_idx += 1
                tnow = time.time()
                dt_iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(tnow))

                x = np.concatenate((ring[write_pos:], ring[:write_pos])).astype(np.float32)
                x64 = x.astype(np.float64)
                x64 -= np.mean(x64)

                rms = float(np.sqrt(np.mean(x64*x64)) + 1e-12)
                rms_db = db20(rms)

                f, pxx = welch_psd(x64)
                logp = 10.0 * np.log10(pxx + 1e-24)

                hf = hf_floor_db(f, pxx)
                hum = hum_rel_db_goertzel(x64)
                voice_db = band_db_from_psd(f, pxx, VOICE_BAND)

                if hum_s is None:
                    hum_s, hf_s, voice_s = hum, hf, voice_db
                else:
                    hum_s   = SMOOTH_HUM*hum_s + (1-SMOOTH_HUM)*hum
                    hf_s    = SMOOTH_HF*hf_s   + (1-SMOOTH_HF)*hf
                    voice_s = SMOOTH_VOICE*voice_s + (1-SMOOTH_VOICE)*voice_db

                # Audio rejection (speech spike)
                dvoice = float(voice_s - (baseline_voice if baseline_voice is not None else voice_s))
                audio_flag = (dvoice > float(args.voice_reject_db))

                # Refresh locks (HF-only by default)
                lock_fmin_eff = max(LOCK_FMIN, float(args.hf_lock_min))
                if (tnow - last_refresh) > LOCK_REFRESH_SEC:
                    last_refresh = tnow
                    cands = pick_candidates(f, pxx, lock_fmin_eff, LOCK_FMAX, LOCK_PROM_DB)
                    used = []
                    def ok(ff): return all(abs(ff-u) >= LOCK_SEP_HZ for u in used)
                    for k in range(K_LOCKS):
                        if lock_freqs[k] is not None:
                            used.append(lock_freqs[k])
                    for k in range(K_LOCKS):
                        if lock_freqs[k] is None:
                            for ff0, dbp, pr in cands:
                                if ok(ff0):
                                    lock_freqs[k] = ff0
                                    used.append(ff0)
                                    break

                # Lock levels (relative to frame power)
                pref = float(np.mean(x64*x64) + 1e-24)
                lock_db = np.array([-200.0]*K_LOCKS, dtype=np.float64)
                for k in range(K_LOCKS):
                    ff0 = lock_freqs[k]
                    if ff0 is None: continue
                    pl = float(goertzel_power(x64, FS, ff0) + 1e-24)
                    lock_db[k] = db10(pl / pref)

                lock_db_s[:] = SMOOTH_LOCK*lock_db_s + (1-SMOOTH_LOCK)*lock_db

                # Drop dead locks
                for k in range(K_LOCKS):
                    if lock_freqs[k] is None:
                        continue
                    if lock_db_s[k] > lock_db_best[k]:
                        lock_db_best[k] = lock_db_s[k]
                    if (lock_db_best[k] - lock_db_s[k]) > LOCK_DROP_DB:
                        lock_freqs[k] = None
                        lock_db_best[k] = -200.0
                        lock_db_s[k] = -200.0

                # Lock-shape vector (structure across carriers)
                v = np.maximum(lock_db_s, -180.0)
                v = v - np.max(v)
                w_shape = np.exp(v / 6.0)
                w_shape = w_shape / (np.sum(w_shape) + 1e-12)

                # Envelope value: coherent sum of lock powers (structure carrier)
                # Convert lock_db_s (dB ratio) -> linear ratio
                lin = np.power(10.0, lock_db_s / 10.0)
                env_val = float(np.dot(w_shape, lin))
                env_hist.append(env_val)
                max_env_len = int(ENV_SECONDS * PRINT_HZ)
                if len(env_hist) > max_env_len:
                    env_hist = env_hist[-max_env_len:]

                # Envelope PSD and shape vector (0.5–30 Hz)
                f_env, p_env = envelope_psd(env_hist, PRINT_HZ, fmax=min(30.0, PRINT_HZ/2.0))
                env_vec = env_shape_vector(f_env, p_env, band=(0.5, min(30.0, PRINT_HZ/2.0)))

                # Initialize baselines on first loop
                if baseline_lock_shape is None:
                    baseline_lock_shape = w_shape.copy()
                    baseline_hum = float(hum_s)
                    baseline_hf = float(hf_s)
                    baseline_voice = float(voice_s)
                    baseline_env_shape = env_vec.copy() if env_vec is not None else None
                    here_thresh = None
                    baseline_set = False

                # Baseline capture window
                if baseline_armed:
                    rem = baseline_until - tnow
                    if rem > 0:
                        baseline_msg = f"[baseline] {rem:4.1f}s left — stay away/quiet"
                        baseline_collect["lock_shape"].append(w_shape.copy())
                        baseline_collect["hum"].append(float(hum_s))
                        baseline_collect["hf"].append(float(hf_s))
                        baseline_collect["voice"].append(float(voice_s))
                        if env_vec is not None:
                            baseline_collect["env_shape"].append(env_vec.copy())
                    else:
                        # Finalize baseline
                        if len(baseline_collect["lock_shape"]) >= 5:
                            W = np.vstack(baseline_collect["lock_shape"])
                            lock_mean = np.mean(W, axis=0)
                            lock_mean = lock_mean / (np.sum(lock_mean) + 1e-12)
                            baseline_lock_shape = lock_mean

                            baseline_hum = float(np.mean(baseline_collect["hum"]))
                            baseline_hf  = float(np.mean(baseline_collect["hf"]))
                            baseline_voice = float(np.mean(baseline_collect["voice"]))

                            if len(baseline_collect["env_shape"]) >= 5:
                                E = np.vstack(baseline_collect["env_shape"])
                                env_mean = np.mean(E, axis=0)
                                env_mean = env_mean / (np.sum(env_mean) + 1e-12)
                                baseline_env_shape = env_mean

                            # Build idx distribution during baseline for threshold
                            idx_tmp = []
                            for i in range(len(baseline_collect["lock_shape"])):
                                ww = baseline_collect["lock_shape"][i]
                                dshape0 = cosine_distance(ww, baseline_lock_shape)
                                dhum0 = baseline_collect["hum"][i] - baseline_hum
                                dhf0  = baseline_collect["hf"][i]  - baseline_hf

                                # env component if we have enough env vectors
                                envd0 = 0.0
                                if baseline_env_shape is not None and i < len(baseline_collect["env_shape"]):
                                    ev = baseline_collect["env_shape"][i]
                                    # if dimension mismatch, skip
                                    if ev.shape == baseline_env_shape.shape:
                                        envd0 = cosine_distance(ev, baseline_env_shape)

                                idx0 = float(args.w_shape)*dshape0 + float(args.w_env)*envd0 + float(args.w_hum)*abs(dhum0) + float(args.w_hf)*abs(dhf0)
                                idx_tmp.append(idx0)

                            mu = float(np.mean(idx_tmp))
                            sdv = float(np.std(idx_tmp) + 1e-12)
                            here_thresh = mu + float(args.thr_sigma)*sdv
                            baseline_set = True
                            baseline_msg = f"[baseline] SET — now approach (thr={here_thresh:.2f})"
                        else:
                            baseline_msg = "[baseline] too few samples — press b again"
                        baseline_armed = False

                # Compute structure distances
                lock_shape_dist = cosine_distance(w_shape, baseline_lock_shape)

                env_shape_dist = 0.0
                if (baseline_env_shape is not None) and (env_vec is not None) and (env_vec.shape == baseline_env_shape.shape):
                    env_shape_dist = cosine_distance(env_vec, baseline_env_shape)

                dhum = float(hum_s - baseline_hum)
                dhf  = float(hf_s  - baseline_hf)

                # Presence index (structure-driven)
                idx = float(args.w_shape)*lock_shape_dist + float(args.w_env)*env_shape_dist + float(args.w_hum)*abs(dhum) + float(args.w_hf)*abs(dhf)

                # Freeze idx updates on speech-like frames (optional)
                if not audio_flag:
                    idx_s = SMOOTH_IDX*idx_s + (1-SMOOTH_IDX)*idx

                thr = here_thresh if (here_thresh is not None) else 2.5
                here = 1 if (idx_s > thr) else 0
                tag = "HERE" if here else "----"

                idx_hist.append(idx_s)
                if len(idx_hist) > int(PRINT_HZ * 40):
                    idx_hist = idx_hist[-int(PRINT_HZ * 40):]

                env_shape_hist.append(env_shape_dist)
                if len(env_shape_hist) > int(PRINT_HZ * 40):
                    env_shape_hist = env_shape_hist[-int(PRINT_HZ * 40):]

                # Dominant peaks per band (envelope spectrum)
                d_hz, _ = dominant_in_env_band(f_env, p_env, DELTA_BAND)
                t_hz, _ = dominant_in_env_band(f_env, p_env, THETA_BAND)
                a_hz, _ = dominant_in_env_band(f_env, p_env, ALPHA_BAND)
                b_hz, _ = dominant_in_env_band(f_env, p_env, BETA_BAND)

                # CSV
                if csv_w:
                    row = [
                        tnow, dt_iso, frame_idx,
                        float(idx_s), int(here), float(thr),
                        float(rms_db),
                        float(env_shape_dist), float(lock_shape_dist),
                        float(hum_s), float(hf_s),
                        float(voice_s), float(dvoice), int(audio_flag),
                        float(d_hz or 0.0), float(t_hz or 0.0), float(a_hz or 0.0), float(b_hz or 0.0),
                    ]
                    row += [float(ff0 or 0.0) for ff0 in lock_freqs]
                    row += [float(x) for x in lock_db_s.tolist()]
                    csv_w.writerow(row)
                    csv_f.flush()

                # =============================
                # Plots
                # =============================
                if plot and (not plot["ready"]):
                    plt = plot["plt"]
                    plt.ion()

                    nrows = 4 if args.plot_spectrum else 3
                    fig, ax = plt.subplots(nrows, 1, figsize=(12, 9))
                    plot["fig"], plot["ax"] = fig, ax

                    # idx
                    plot["l1"], = ax[0].plot([], [])
                    ax[0].set_title("Presence index (structure)")
                    plot["thr_line"] = ax[0].axhline(thr, linestyle="--")

                    # env structure distance
                    plot["l3"], = ax[1].plot([], [])
                    ax[1].set_title("Envelope structure distance (0.5–30 Hz)")

                    # envelope PSD
                    plot["lenv"], = ax[2].plot([], [])
                    ax[2].set_title("Envelope spectrum (0.5–30 Hz)")
                    ax[2].set_xlabel("Hz")
                    ax[2].set_ylabel("PSD (dB)")
                    plot["band_lines"] = []

                    if args.plot_spectrum:
                        plot["ax_spec"] = ax[3]
                        plot["lpsd"], = ax[3].plot([], [])
                        ax[3].set_title("Audio-band PSD + mains ladder")
                        ax[3].set_xlabel("Hz")
                        ax[3].set_ylabel("dB")
                        plot["mains_lines"] = []

                    plot["ready"] = True

                if plot and plot["ready"]:
                    plt = plot["plt"]
                    ax = plot["ax"]

                    # idx
                    t = np.arange(len(idx_hist)) / PRINT_HZ
                    plot["l1"].set_data(t, idx_hist)
                    ax[0].relim(); ax[0].autoscale_view()
                    plot["thr_line"].set_ydata([thr, thr])

                    # env distance
                    t3 = np.arange(len(env_shape_hist)) / PRINT_HZ
                    plot["l3"].set_data(t3, env_shape_hist)
                    ax[1].relim(); ax[1].autoscale_view()

                    # env PSD
                    if f_env is not None and p_env is not None:
                        p_db = 10.0*np.log10(p_env + 1e-30)
                        plot["lenv"].set_data(f_env, p_db)
                        ax[2].relim(); ax[2].autoscale_view()

                        # band markers (draw once)
                        if len(plot["band_lines"]) == 0:
                            for ff in [DELTA_BAND[0], DELTA_BAND[1], THETA_BAND[1], ALPHA_BAND[1], BETA_BAND[0], BETA_BAND[1]]:
                                plot["band_lines"].append(ax[2].axvline(ff, linestyle="--", linewidth=1.0))
                        ax[2].set_xlim(0.0, min(30.0, PRINT_HZ/2.0))

                    # audio-band spectrum
                    if args.plot_spectrum:
                        axS = plot["ax_spec"]
                        fmax_view = min(float(args.plot_fmax), float(f[-1]))
                        mview = (f >= 0) & (f <= fmax_view)
                        plot["lpsd"].set_data(f[mview], logp[mview])
                        axS.relim(); axS.autoscale_view()
                        axS.set_xlim(0, fmax_view)

                        # mains ladder (recreate if needed)
                        if len(plot["mains_lines"]) == 0:
                            for k in range(1, MAINS_HARMONICS + 1):
                                fk = MAINS * k
                                if fk > fmax_view:
                                    break
                                plot["mains_lines"].append(axS.axvline(fk, linestyle="--", linewidth=1.0))

                    plt.pause(0.001)

                # =============================
                # Console render (8 lines)
                # =============================
                move_up(8)

                sys.stdout.write(f"pres| idx:{idx_s:5.2f}  {tag}  thr:{thr:5.2f}  rms:{rms_db:6.1f} dBFS   {baseline_msg}")
                clear_eol(); sys.stdout.write("\n")

                locks_str = " ".join([fmt_hz(ff0) for ff0 in lock_freqs])
                sys.stdout.write(f"lock| {locks_str}")
                clear_eol(); sys.stdout.write("\n")

                sys.stdout.write(f"idx | {spark(idx_hist, width=60)}")
                clear_eol(); sys.stdout.write("\n")

                sys.stdout.write(f"env | {spark(env_shape_hist, width=60)}  (envΔ={env_shape_dist:4.2f}, lockΔ={lock_shape_dist:4.2f})")
                clear_eol(); sys.stdout.write("\n")

                sys.stdout.write(f"aud | voice:{voice_s:7.2f} dB  Δv:{dvoice:+6.2f} dB  flag:{'YES' if audio_flag else 'no '} (reject>{args.voice_reject_db:.1f}dB)")
                clear_eol(); sys.stdout.write("\n")

                sys.stdout.write(f"hum | rel:{hum_s:7.2f} dB   Δ:{(hum_s-baseline_hum):+6.2f} dB   (mains {int(MAINS)}Hz..)")
                clear_eol(); sys.stdout.write("\n")

                sys.stdout.write(f"hf  | floor:{hf_s:7.2f} dB   Δ:{(hf_s-baseline_hf):+6.2f} dB   (2k..18k median)")
                clear_eol(); sys.stdout.write("\n")

                sys.stdout.write(
                    "band| delta:"
                    f"{fmt_hz(d_hz) if d_hz else 'none':>6}  "
                    "theta:"
                    f"{fmt_hz(t_hz) if t_hz else 'none':>6}  "
                    "alpha:"
                    f"{fmt_hz(a_hz) if a_hz else 'none':>6}  "
                    "beta:"
                    f"{fmt_hz(b_hz) if b_hz else 'none':>6}  "
                    f"(envHz={PRINT_HZ:.0f})"
                )
                clear_eol(); sys.stdout.write("\n")

                sys.stdout.flush()

    except KeyboardInterrupt:
        pass
    finally:
        if csv_f:
            csv_f.close()
        print("\nStopped.")

if __name__ == "__main__":
    main()
