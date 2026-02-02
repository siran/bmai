# presence_lock.py
# Drop-in script: presence index + HF locks + harmonic-family highlighting + device setup sweep
#
# Windows hotkeys (console):
#   b = capture baseline (step away/quiet)
#   l = force relock
#   q = quit
#
# Plots:
#   --plot            => idx + lock track plots
#   --plot-spectrum   => adds live PSD plot with harmonic-family + mains-family highlighting
#
# Device setup:
#   --setup           => sweep input devices, score them, suggest best
#   --setup-plot      => during --setup, show a PSD plot for each device (step through)
#
import argparse, time, threading, sys, math, csv
import numpy as np
import sounddevice as sd
from scipy.signal import welch, find_peaks
import msvcrt

# =========================
# Core sampling + analysis
# =========================
FS = 48000
CHANNELS = 1

PRINT_HZ = 10
FRAME_SECONDS = 0.8
HOP_SECONDS = 1.0 / PRINT_HZ
NFFT = 8192

# Locks
K_LOCKS = 6
LOCK_FMIN = 80
LOCK_FMAX = 18000
LOCK_PROM_DB = 8
LOCK_REFRESH_SEC = 3.0
LOCK_DROP_DB = 10
LOCK_SEP_HZ = 40.0

# Smoothing
SMOOTH_LOCK = 0.80
SMOOTH_IDX  = 0.80
SMOOTH_HUM  = 0.85
SMOOTH_HF   = 0.85
SMOOTH_VOICE = 0.85

# Modulation (lock track)
MOD_SECONDS = 25
MOD_FS = PRINT_HZ
BREATH_BAND = (0.08, 0.50)
HEART_BAND  = (0.70, 3.00)

# Mains hum proxy (Goertzel)
MAINS = 60.0
HARMONICS = 6
HF_FLOOR = (2000, 18000)

# Audio rejection band (speech-ish)
VOICE_BAND = (200, 4000)

# ================
# Small utilities
# ================
def trapz(y, x):
    integ = getattr(np, "trapezoid", None)
    if integ is None:
        return np.trapz(y, x)
    return integ(y, x)

def db10(x): return 10.0 * math.log10(max(1e-30, float(x)))
def db20(x): return 20.0 * math.log10(max(1e-30, float(x)))

def fmt_hz(hz):
    if hz is None: return "none"
    return f"{hz/1000:.2f}k" if hz >= 1000 else f"{hz:.0f}"

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

def hum_rel_db_goertzel(x, fs=FS, mains=MAINS, nh=HARMONICS):
    x = x.astype(np.float64)
    x -= np.mean(x)
    pref = float(np.mean(x*x) + 1e-24)
    ph = 0.0
    for k in range(1, nh + 1):
        ph += float(goertzel_power(x, fs, mains*k) + 1e-24)
    return db10(ph / (pref + 1e-24))

def pick_candidates(f, pxx, lock_fmin, lock_fmax, prom_db):
    logp = 10.0 * np.log10(pxx + 1e-24)
    m = (f >= lock_fmin) & (f <= lock_fmax)
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
    return float(f[m][idx]), db10(float(pxx[m][idx] + 1e-30))

# =========================
# Harmonic-family helpers
# =========================
def nearest_bin(f, target_hz):
    return int(np.argmin(np.abs(f - target_hz)))

def local_floor_db(logp, idx, half_window=12):
    a = max(0, idx - half_window)
    b = min(len(logp), idx + half_window + 1)
    return float(np.median(logp[a:b]))

def harmonic_family_score(f, logp, f0, nh=10, tol_hz=8.0):
    per = []
    bins = []
    score = 0.0
    for k in range(1, nh + 1):
        fk = f0 * k
        if fk >= f[-1]:
            break
        idx = nearest_bin(f, fk)
        if abs(f[idx] - fk) > tol_hz:
            continue
        floor = local_floor_db(logp, idx)
        strength = float(logp[idx] - floor)
        if strength < 0:
            strength = 0.0
        per.append(strength)
        bins.append(idx)
        score += strength
    return score, per, bins

def pick_f0_from_candidates(f, pxx, cands, fmin=80.0, fmax=1200.0, nh=10):
    logp = 10.0 * np.log10(pxx + 1e-24)
    best = (None, -1e9, None, None)  # f0, score2, per, bins
    for ff, dbp, prom in cands:
        if not (fmin <= ff <= fmax):
            continue
        score, per, bins = harmonic_family_score(f, logp, ff, nh=nh)
        score2 = score - 0.002 * ff
        if score2 > best[1]:
            best = (ff, score2, per, bins)
    return best

def mains_family_strength_db(f, logp, mains=MAINS, nh=HARMONICS, tol_hz=8.0):
    s = 0.0
    for k in range(1, nh + 1):
        fk = mains * k
        if fk >= f[-1]:
            break
        idx = nearest_bin(f, fk)
        if abs(f[idx] - fk) > tol_hz:
            continue
        s += max(0.0, float(logp[idx] - local_floor_db(logp, idx)))
    return float(s)

# =========================
# Device tools
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
    """
    Goal for setup:
      - pick a device that is NOT near-silent/broken
      - prefer lower (more negative) HF noise floor
      - penalize strong mains ladder / EM pickup
    """
    rms = float(np.sqrt(np.mean(x*x)) + 1e-12)
    rms_db = db20(rms)

    f, pxx = welch(x, fs=fs, nperseg=NFFT, noverlap=NFFT//2, scaling="density")
    logp = 10.0 * np.log10(pxx + 1e-24)

    hf = hf_floor_db(f, pxx)
    hum = hum_rel_db_goertzel(x, fs=fs)
    voice_db = band_db_from_psd(f, pxx, VOICE_BAND)

    cands = pick_candidates(f, pxx, 80, 18000, prom_db=6.0)
    n_peaks = len(cands)

    f0, f0score2, per, bins = pick_f0_from_candidates(f, pxx, cands, fmin=80.0, fmax=1200.0, nh=10)
    harm_strength = float(0.0 if f0 is None else max(0.0, f0score2))

    mains_strength = mains_family_strength_db(f, logp)

    # Scoring (higher is better)
    score = 0.0

    # prefer quiet HF floor (hf is negative => -hf is positive)
    score += 1.0 * (-hf)

    # penalize strong mains ladder
    score += -2.0 * max(0.0, mains_strength - 10.0)

    # penalize “peak soup”
    score += -0.3 * max(0.0, n_peaks - 60)

    # small reward for clean harmonic family capture
    score += 0.1 * harm_strength

    # disqualify near-silence / broken stream
    if rms_db < -90.0:
        score += -500.0
    elif rms_db < -80.0:
        score += -150.0

    return {
        "rms_dbfs": rms_db,
        "hf_floor_db": hf,
        "hum_rel_db": hum,
        "voice_db": voice_db,
        "n_peaks": n_peaks,
        "harm_strength": harm_strength,
        "mains_strength": mains_strength,
        "score": float(score),
        "f": f,
        "logp": logp
    }

def setup_sweep(args):
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
            results.append((idx, name, ha, None, err))
            print(f"[{idx}] FAIL: {name} ({ha})  err={err}")
            continue

        met = score_device_metrics(x, fs=FS)
        results.append((idx, name, ha, met, None))
        print(f"[{idx}] ok  score={met['score']:+8.2f}  rms={met['rms_dbfs']:7.1f}dBFS  "
              f"hf={met['hf_floor_db']:7.1f}dB  mainsS={met['mains_strength']:5.1f}  "
              f"harmS={met['harm_strength']:5.1f}  peaks={met['n_peaks']:3d}  {name} ({ha})")

        if plt is not None:
            plt.ion()
            fig, ax = plt.subplots(1, 1, figsize=(10, 4))
            ax.plot(met["f"], met["logp"])
            ax.set_title(f"Device [{idx}] PSD: {name} ({ha})  score={met['score']:+.2f}")
            ax.set_xlabel("Hz"); ax.set_ylabel("dB")
            ax.set_xlim(0, 18000)
            for k in range(1, HARMONICS + 1):
                fk = MAINS*k
                if fk > 18000: break
                ax.axvline(fk, linestyle="--", linewidth=1.0)
            plt.pause(0.001)
            print("   [setup] close the plot window to continue...")
            plt.show(block=True)

    ok = [r for r in results if (r[3] is not None)]
    if len(ok) == 0:
        print("\n[setup] All devices failed. Try --allow-wdmks or different hostapi.")
        return 3

    ok_sorted = sorted(ok, key=lambda r: r[3]["score"], reverse=True)
    print("\n[setup] Suggestions (top 5):")
    for j, (idx, name, ha, met, _) in enumerate(ok_sorted[:5], start=1):
        print(f"  {j}. --device {idx}  score={met['score']:+8.2f}  rms={met['rms_dbfs']:7.1f}dBFS  "
              f"hf={met['hf_floor_db']:7.1f}dB  mainsS={met['mains_strength']:5.1f}  {name} ({ha})")

    best_idx = ok_sorted[0][0]
    print(f"\n[setup] Recommended: --device {best_idx}\n")
    return 0

# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--list", action="store_true")
    ap.add_argument("--setup", action="store_true", help="Sweep devices, score, suggest best")
    ap.add_argument("--setup-sec", type=float, default=1.2, help="Seconds per device during --setup")
    ap.add_argument("--setup-plot", action="store_true", help="Show PSD plot per device during --setup")

    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--allow-wdmks", action="store_true")

    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--plot-spectrum", action="store_true")
    ap.add_argument("--csv", type=str, default=None)

    # baseline / thresholding
    ap.add_argument("--baseline-sec", type=float, default=3.0, help="Baseline capture duration after pressing b")
    ap.add_argument("--thr-sigma", type=float, default=4.0, help="threshold = baseline_mean + thr_sigma*baseline_std")

    # audio rejection
    ap.add_argument("--voice-reject-db", type=float, default=8.0, help="If voice_db - baseline_voice_db > this, freeze idx update")
    ap.add_argument("--hf-lock-min", type=float, default=8000.0, help="Locks searched above this frequency (reduce speech dominance)")

    # harmonic family detection / plot
    ap.add_argument("--harmonics", type=int, default=10, help="How many harmonics to highlight")
    ap.add_argument("--f0-max", type=float, default=1200.0, help="Max f0 for harmonic family search")
    ap.add_argument("--f0-prom-db", type=float, default=6.0, help="Peak prominence for f0 candidate search")
    ap.add_argument("--f0-stable-hz", type=float, default=2.0, help="f0 stability threshold for 'audio-ish' label")
    ap.add_argument("--fam-audio-score", type=float, default=18.0, help="harmonic family score above this => audio-ish")
    ap.add_argument("--fam-mains-score", type=float, default=18.0, help="mains ladder score above this => mains-ish/EM-ish")

    args = ap.parse_args()

    if args.list:
        list_devices()
        return

    if args.setup:
        rc = setup_sweep(args)
        sys.exit(rc)

    if args.device is None:
        print("Run: python presence_lock.py --list")
        print("Or : python presence_lock.py --setup")
        print("Then: python presence_lock.py --device <INDEX> [--plot-spectrum]")
        sys.exit(1)

    devinfo = sd.query_devices(args.device)
    ha_name = sd.query_hostapis(devinfo["hostapi"])["name"]
    print(f"Using device {args.device}: {devinfo['name']} (hostapi={ha_name})")
    if (not args.allow_wdmks) and ("WDM-KS" in ha_name):
        print("\nWDM-KS often produces NO callbacks. Pick WASAPI/MME/DirectSound, or pass --allow-wdmks.\n")
        sys.exit(2)

    plot = None
    if args.plot or args.plot_spectrum:
        import matplotlib.pyplot as plt
        plot = {"plt": plt, "ready": False}

    csv_f = None
    csv_w = None
    if args.csv:
        csv_f = open(args.csv, "w", newline="")
        csv_w = csv.writer(csv_f)
        csv_w.writerow([
            "t","dt_iso","frame_idx",
            "idx","here","here_thresh",
            "rms_dbfs",
            "baseline_set",
            "dshape","dhum","dhf",
            "hum_rel_db","hf_floor_db",
            "voice_db","dvoice_db","audio_flag",
            "track_k","track_hz","track_db",
            "breath_hz","heart_hz",
            "f0_hz","harm_score","mains_score","noise_class",
            *[f"lock_hz_{k}" for k in range(K_LOCKS)],
            *[f"lock_db_{k}" for k in range(K_LOCKS)],
        ])
        csv_f.flush()

    frame_n = int(FS * FRAME_SECONDS)
    hop_n = max(1, int(FS * HOP_SECONDS))

    ring = np.zeros(frame_n, dtype=np.float32)
    write_pos = 0
    filled = 0
    last_cb_time = time.time()

    lock_freqs = [None] * K_LOCKS
    lock_db_s = np.array([-200.0] * K_LOCKS, dtype=np.float64)
    lock_db_best = np.array([-200.0] * K_LOCKS, dtype=np.float64)
    last_refresh = 0.0

    # baselines
    baseline_vec = None
    baseline_hum = None
    baseline_hf  = None
    baseline_voice = None

    # baseline capture state
    baseline_armed = False
    baseline_until = None
    baseline_msg = "press b to baseline"
    baseline_collect = {"w": [], "hum": [], "hf": [], "voice": []}
    here_thresh = None
    baseline_set = False

    idx_s = 0.0
    idx_hist = []
    lock_hist = []

    hum_s = None
    hf_s = None
    voice_s = None

    # harmonic tracking stability
    f0_s = None
    f0_hist = []

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
        nonlocal baseline_armed, baseline_until, baseline_msg
        nonlocal lock_freqs, last_refresh, baseline_collect, baseline_set
        while True:
            if msvcrt.kbhit():
                ch = msvcrt.getch().lower()
                if ch == b"b":
                    baseline_armed = True
                    baseline_set = False
                    baseline_until = time.time() + float(args.baseline_sec)
                    baseline_collect = {"w": [], "hum": [], "hf": [], "voice": []}
                    baseline_msg = f"[baseline] capturing {args.baseline_sec:.1f}s — step away now"
                elif ch == b"l":
                    lock_freqs = [None] * K_LOCKS
                    last_refresh = 0.0
                    baseline_msg = "[locks] forced relock"
                elif ch == b"q":
                    raise SystemExit
            time.sleep(0.02)

    threading.Thread(target=keythread, daemon=True).start()

    print("\nHotkeys: b=baseline, l=relock, q=quit, Ctrl+C=stop\n")
    print("pres|")
    print("lock|")
    print("idx |")
    print("mod |")
    print("bre |")
    print("aud |")
    print("hum |")
    print("hf  |")
    print("fam |")

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
                            for ff, dbp, pr in cands:
                                if ok(ff):
                                    lock_freqs[k] = ff
                                    used.append(ff)
                                    break

                # lock levels (relative)
                pref = float(np.mean(x64*x64) + 1e-24)
                lock_db = np.array([-200.0]*K_LOCKS, dtype=np.float64)
                for k in range(K_LOCKS):
                    ff = lock_freqs[k]
                    if ff is None: continue
                    pl = float(goertzel_power(x64, FS, ff) + 1e-24)
                    lock_db[k] = db10(pl / pref)

                lock_db_s[:] = SMOOTH_LOCK*lock_db_s + (1-SMOOTH_LOCK)*lock_db

                # drop dead locks
                for k in range(K_LOCKS):
                    if lock_freqs[k] is None:
                        continue
                    if lock_db_s[k] > lock_db_best[k]:
                        lock_db_best[k] = lock_db_s[k]
                    if (lock_db_best[k] - lock_db_s[k]) > LOCK_DROP_DB:
                        lock_freqs[k] = None
                        lock_db_best[k] = -200.0
                        lock_db_s[k] = -200.0

                # build shape vector from locks
                v = np.maximum(lock_db_s, -180.0)
                v = v - np.max(v)
                w = np.exp(v / 6.0)
                w = w / (np.sum(w) + 1e-12)

                # initialize baseline if never set
                if baseline_vec is None:
                    baseline_vec = w.copy()
                    baseline_hum = float(hum_s)
                    baseline_hf  = float(hf_s)
                    baseline_voice = float(voice_s)
                    here_thresh = None
                    baseline_set = False

                # baseline capture window
                if baseline_armed:
                    rem = baseline_until - tnow
                    if rem > 0:
                        baseline_msg = f"[baseline] {rem:4.1f}s left — stay away/quiet"
                        baseline_collect["w"].append(w.copy())
                        baseline_collect["hum"].append(float(hum_s))
                        baseline_collect["hf"].append(float(hf_s))
                        baseline_collect["voice"].append(float(voice_s))
                    else:
                        if len(baseline_collect["w"]) >= 3:
                            W = np.vstack(baseline_collect["w"])
                            w_mean = np.mean(W, axis=0)
                            w_mean = w_mean / (np.sum(w_mean) + 1e-12)

                            baseline_vec = w_mean
                            baseline_hum = float(np.mean(baseline_collect["hum"]))
                            baseline_hf  = float(np.mean(baseline_collect["hf"]))
                            baseline_voice = float(np.mean(baseline_collect["voice"]))

                            idx_tmp = []
                            for ww, hh, hff in zip(
                                baseline_collect["w"],
                                baseline_collect["hum"],
                                baseline_collect["hf"],
                            ):
                                dshape0 = cosine_distance(ww, baseline_vec)
                                dhum0 = hh - baseline_hum
                                dhf0  = hff - baseline_hf
                                idx0 = 6.0*dshape0 + 0.25*abs(dhum0) + 0.10*abs(dhf0)
                                idx_tmp.append(idx0)

                            mu = float(np.mean(idx_tmp))
                            sdv = float(np.std(idx_tmp) + 1e-12)
                            here_thresh = mu + float(args.thr_sigma)*sdv
                            baseline_set = True
                            baseline_msg = f"[baseline] SET — now approach (thr={here_thresh:.2f})"
                        else:
                            baseline_msg = "[baseline] too few samples — press b again"
                        baseline_armed = False

                # audio rejection
                dvoice = float(voice_s - baseline_voice)
                audio_flag = (dvoice > float(args.voice_reject_db))

                # presence idx
                dshape = cosine_distance(w, baseline_vec)
                dhum = float(hum_s - baseline_hum)
                dhf  = float(hf_s  - baseline_hf)
                idx = 6.0*dshape + 0.25*abs(dhum) + 0.10*abs(dhf)

                if not audio_flag:
                    idx_s = SMOOTH_IDX*idx_s + (1-SMOOTH_IDX)*idx

                thr = here_thresh if (here_thresh is not None) else 2.5
                here = 1 if (idx_s > thr) else 0
                tag = "HERE" if here else "----"

                idx_hist.append(idx_s)
                if len(idx_hist) > 600:
                    idx_hist = idx_hist[-600:]

                # lock track history
                kmax = int(np.argmax(lock_db_s))
                lock_track_db = float(lock_db_s[kmax])
                lock_track_hz = lock_freqs[kmax]
                lock_hist.append(lock_track_db)
                maxlen = int(MOD_SECONDS * MOD_FS)
                if len(lock_hist) > maxlen:
                    lock_hist = lock_hist[-maxlen:]

                fb, _ = dominant_in_band(lock_hist, MOD_FS, BREATH_BAND)
                fh, _ = dominant_in_band(lock_hist, MOD_FS, HEART_BAND)

                # harmonic family / noise class
                cands_f0 = pick_candidates(f, pxx, 80.0, float(args.f0_max), prom_db=float(args.f0_prom_db))
                f0, f0score2, per, bins = pick_f0_from_candidates(
                    f, pxx, cands_f0, fmin=80.0, fmax=float(args.f0_max), nh=int(args.harmonics)
                )
                harm_score = float(0.0 if f0 is None else max(0.0, f0score2))
                mains_score = float(mains_family_strength_db(f, logp))

                if f0 is not None:
                    if f0_s is None:
                        f0_s = float(f0)
                    else:
                        f0_s = 0.85*float(f0_s) + 0.15*float(f0)
                    f0_hist.append(float(f0_s))
                else:
                    f0_hist.append(float("nan"))
                if len(f0_hist) > 50:
                    f0_hist = f0_hist[-50:]

                recent = np.array([v for v in f0_hist[-20:] if np.isfinite(v)], dtype=np.float64)
                f0_std = float(np.std(recent)) if len(recent) >= 6 else 999.0

                noise_class = "mixed"
                if mains_score >= float(args.fam_mains_score) and (harm_score < 0.8*float(args.fam_audio_score)):
                    noise_class = "mains/EM"
                elif harm_score >= float(args.fam_audio_score) and (f0_std <= float(args.f0_stable_hz)):
                    noise_class = "audio-ish"
                elif (harm_score < 0.5*float(args.fam_audio_score)) and (mains_score < 0.5*float(args.fam_mains_score)):
                    noise_class = "broad/noise"

                # CSV
                if csv_w:
                    row = [
                        tnow, dt_iso, frame_idx,
                        float(idx_s), here, float(thr),
                        float(rms_db),
                        int(baseline_set),
                        float(dshape), float(dhum), float(dhf),
                        float(hum_s), float(hf_s),
                        float(voice_s), float(dvoice), int(audio_flag),
                        int(kmax), float(lock_track_hz or 0.0), float(lock_track_db),
                        float(fb or 0.0), float(fh or 0.0),
                        float(f0 or 0.0), float(harm_score), float(mains_score), noise_class
                    ]
                    row += [float(ff or 0.0) for ff in lock_freqs]
                    row += [float(x) for x in lock_db_s.tolist()]
                    csv_w.writerow(row)
                    csv_f.flush()

                # Plots
                if plot and (not plot["ready"]):
                    plt = plot["plt"]
                    plt.ion()

                    nrows = 3 if args.plot_spectrum else 2
                    fig, ax = plt.subplots(nrows, 1, figsize=(11, 8))
                    plot["fig"], plot["ax"] = fig, ax

                    plot["l1"], = ax[0].plot([], [])
                    ax[0].set_title("Presence index (idx)")
                    # keep a handle to threshold line (fixes your error)
                    plot["thr_line"] = ax[0].axhline(thr, linestyle="--")

                    plot["l2"], = ax[1].plot([], [])
                    ax[1].set_title("Track lock relative power (dB)")

                    if args.plot_spectrum:
                        plot["ax_spec"] = ax[2]
                        plot["lpsd"], = ax[2].plot([], [])
                        ax[2].set_title("PSD + harmonic family (thick) + mains family (dashed)")
                        ax[2].set_xlabel("Hz")
                        ax[2].set_ylabel("dB")
                        plot["harm_lines"] = []
                        plot["mains_lines"] = []
                        plot["txt"] = ax[2].text(0.02, 0.95, "", transform=ax[2].transAxes, va="top")

                    plot["ready"] = True

                if plot and plot["ready"]:
                    plt = plot["plt"]
                    ax = plot["ax"]

                    t = np.arange(len(idx_hist)) / PRINT_HZ
                    plot["l1"].set_data(t, idx_hist)
                    ax[0].relim(); ax[0].autoscale_view()

                    # update threshold line without touching ax.lines
                    plot["thr_line"].set_ydata([thr, thr])

                    t2 = np.arange(len(lock_hist)) / PRINT_HZ
                    plot["l2"].set_data(t2, lock_hist)
                    ax[1].relim(); ax[1].autoscale_view()

                    if args.plot_spectrum:
                        axS = plot["ax_spec"]
                        fmax_view = min(18000.0, float(f[-1]))
                        mview = (f >= 0) & (f <= fmax_view)

                        plot["lpsd"].set_data(f[mview], logp[mview])
                        axS.relim(); axS.autoscale_view()

                        for ln in plot["harm_lines"]:
                            ln.remove()
                        plot["harm_lines"].clear()
                        for ln in plot["mains_lines"]:
                            ln.remove()
                        plot["mains_lines"].clear()

                        for k in range(1, HARMONICS + 1):
                            fk = MAINS * k
                            if fk > fmax_view:
                                break
                            ln = axS.axvline(fk, linestyle="--", linewidth=1.0)
                            plot["mains_lines"].append(ln)

                        if f0 is not None and per is not None:
                            for i, strength in enumerate(per, start=1):
                                fk = float(f0) * i
                                if fk > fmax_view:
                                    break
                                aalpha = max(0.10, min(0.95, float(strength) / 18.0))
                                ln = axS.axvline(fk, linewidth=2.0, alpha=aalpha)
                                plot["harm_lines"].append(ln)
                            plot["txt"].set_text(
                                f"f0≈{float(f0):.1f}Hz  harmS≈{harm_score:.1f}  mainsS≈{mains_score:.1f}  class={noise_class}"
                            )
                        else:
                            plot["txt"].set_text(f"f0:none  mainsS≈{mains_score:.1f}  class={noise_class}")

                        axS.set_xlim(0, fmax_view)

                    plt.pause(0.001)

                # Console render
                move_up(9)

                sys.stdout.write(
                    f"pres| idx:{idx_s:5.2f}  {tag}  thr:{thr:5.2f}  rms:{rms_db:6.1f} dBFS   {baseline_msg}"
                )
                clear_eol(); sys.stdout.write("\n")

                locks_str = " ".join([fmt_hz(ff) for ff in lock_freqs])
                sys.stdout.write(f"lock| {locks_str}")
                clear_eol(); sys.stdout.write("\n")

                sys.stdout.write(f"idx | {spark(idx_hist, width=60)}")
                clear_eol(); sys.stdout.write("\n")

                sys.stdout.write(f"mod | {spark(lock_hist, width=60)}  (track:{fmt_hz(lock_track_hz)})")
                clear_eol(); sys.stdout.write("\n")

                if fb is None:
                    sys.stdout.write("bre | (need more time)   heart: (need more time)")
                else:
                    bre = fb*60.0
                    heart = (fh*60.0) if fh is not None else None
                    sys.stdout.write(
                        f"bre | {fb:4.2f}Hz ({bre:4.1f}/min)   heart: " +
                        (f"{fh:4.2f}Hz ({heart:4.1f}/min)" if fh is not None else "none")
                    )
                clear_eol(); sys.stdout.write("\n")

                sys.stdout.write(
                    f"aud | voice:{voice_s:7.2f} dB  Δv:{dvoice:+6.2f} dB  flag:{'YES' if audio_flag else 'no '} "
                    f"(reject>{args.voice_reject_db:.1f}dB)"
                )
                clear_eol(); sys.stdout.write("\n")

                sys.stdout.write(f"hum | rel:{hum_s:7.2f} dB   Δ:{(hum_s-baseline_hum):+6.2f} dB   (goertzel {int(MAINS)}Hz..)")
                clear_eol(); sys.stdout.write("\n")

                sys.stdout.write(f"hf  | floor:{hf_s:7.2f} dB   Δ:{(hf_s-baseline_hf):+6.2f} dB   (2k..18k median)")
                clear_eol(); sys.stdout.write("\n")

                f0_disp = (f"{float(f0):6.1f}Hz" if f0 is not None else "   none")
                sys.stdout.write(
                    f"fam | f0:{f0_disp}  harmS:{harm_score:6.1f}  mainsS:{mains_score:6.1f}  f0σ:{f0_std:5.2f}  class:{noise_class}"
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
