\
import argparse, queue, sys, time
import numpy as np
import sounddevice as sd
from scipy.signal import welch, find_peaks


"""
live_noise_scope.py

Quick live scope for a floating-cable audio-jack sensor.

Shows:
- RMS level (dBFS)
- bandpowers (LF/MID/HF + 60Hz hum band)
- strongest peaks in the current PSD

Use it to confirm you're receiving a nontrivial signal, and to see what parts of
the spectrum are active / stable on your setup.
"""

FS = 48000
CHANNELS = 1
BLOCK_SEC = 0.50
PSD_SEC = 2.0
NFFT = 8192

BANDS = [
    ("hum60", 55, 65),
    ("LF", 1, 40),
    ("MID", 200, 2000),
    ("HF", 2000, 10000),
]

def integrator_trap(y, x):
    integ = getattr(np, "trapezoid", None) or np.trapz
    return float(integ(y, x))

def bandpower(f, pxx, f1, f2):
    m = (f >= f1) & (f <= f2)
    if not np.any(m):
        return 0.0
    return integrator_trap(pxx[m], f[m])

def dbfs_rms(x):
    rms = np.sqrt(np.mean(x*x) + 1e-20)
    return 20*np.log10(rms + 1e-20)

def ascii_bar(val_db, lo=-120, hi=-20, width=28):
    v = (val_db - lo) / (hi - lo)
    v = max(0.0, min(1.0, v))
    n = int(v * width)
    return "[" + "#"*n + "-"*(width-n) + "]"

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
    ap.add_argument("--allow-wdmks", action="store_true", help="Allow Windows WDM-KS hostapi (often buggy)")
    args = ap.parse_args()

    if args.list:
        list_devices()
        return

    if args.device is None:
        print("Usage: python live_noise_scope.py --device <index>")
        print("Run:   python live_noise_scope.py --list")
        sys.exit(1)

    devinfo = sd.query_devices(args.device)
    ha_name = sd.query_hostapis(devinfo["hostapi"])["name"]
    print(f"Using device {args.device}: {devinfo['name']}  (hostapi={ha_name})")

    if (not args.allow_wdmks) and ("WDM-KS" in ha_name):
        print("\nThis device is WDM-KS. On many systems it fails via PortAudio.")
        print("Pick WASAPI/MME/DirectSound from --list.")
        sys.exit(2)

    block_n = int(FS * BLOCK_SEC)
    psd_n = int(FS * PSD_SEC)

    ring = np.zeros(psd_n, dtype=np.float32)
    write = 0
    q = queue.Queue(maxsize=50)

    def callback(indata, frames, time_info, status):
        x = indata[:, 0].copy()
        try:
            q.put_nowait((x, status))
        except queue.Full:
            pass

    stream = sd.InputStream(
        samplerate=FS,
        channels=CHANNELS,
        dtype="float32",
        device=args.device,
        callback=callback,
        blocksize=block_n
    )

    print("Recording... Ctrl+C to stop.")
    print("Tip: plug cable into mic/line-in, leave far end floating.\n")

    with stream:
        try:
            while True:
                try:
                    x, status = q.get(timeout=2.0)
                except queue.Empty:
                    print("NO AUDIO (no callbacks). Try a different device.")
                    continue

                if (not np.isfinite(x).all()) or np.max(np.abs(x)) > 2.0:
                    print("BAD BLOCK -> skipped")
                    continue

                # update ring buffer
                n = len(x)
                end = write + n
                if end <= psd_n:
                    ring[write:end] = x
                else:
                    k = psd_n - write
                    ring[write:] = x[:k]
                    ring[:end-psd_n] = x[k:]
                write = end % psd_n

                buf = np.concatenate([ring[write:], ring[:write]])
                buf = buf - np.mean(buf)

                level = dbfs_rms(x)
                bar = ascii_bar(level)

                f, pxx = welch(buf, fs=FS, nperseg=NFFT, noverlap=NFFT//2, scaling="density")
                if not np.isfinite(pxx).all() or np.max(pxx) <= 0:
                    print("BAD PSD -> skipped")
                    continue

                feats = {name: bandpower(f, pxx, a, b) for name, a, b in BANDS}

                logp = 10*np.log10(pxx + 1e-24)
                logp = logp - np.median(logp)
                peaks, props = find_peaks(logp, prominence=10)

                top = []
                if len(peaks) > 0:
                    prom = props["prominences"]
                    idx = np.argsort(prom)[::-1][:3]
                    for j in idx:
                        p = peaks[j]
                        top.append((float(f[p]), float(logp[p])))

                line = (
                    f"lvl {level:7.1f} dBFS {bar}  "
                    f"hum60 {feats['hum60']:.2e}  LF {feats['LF']:.2e}  MID {feats['MID']:.2e}  HF {feats['HF']:.2e}"
                )
                if top:
                    line += "  peaks: " + ", ".join([f"{hz:6.1f}Hz(+{db:3.0f}dB)" for hz, db in top])
                if status:
                    line += f"  [{status}]"
                print(line)

        except KeyboardInterrupt:
            print("\nStopped.")

if __name__ == "__main__":
    main()
