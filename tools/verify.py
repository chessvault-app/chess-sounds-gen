#!/usr/bin/env python3
"""Compare reference vs synthesized sounds: per-band envelope trajectories.

Both signals are peak-normalized and aligned at onset; for each frequency
band the RMS-envelope (2 ms hops) is compared in dB wherever the reference
band envelope is within 45 dB of that band's peak. Reported: mean |diff|.
"""
import array, math, os, wave

SR = 44100
HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)

BANDS = [(80, 200), (200, 400), (400, 800), (800, 1600),
         (1600, 3200), (3200, 6400), (6400, 11000)]

PAIRS = [
    ("Move.wav", "move-self"),
    ("Capture.wav", "capture"),
    ("Error.wav", "illegal"),
    ("GenericNotify.wav", "notify"),
    ("LowTime.wav", "low-time"),
    ("Select.wav", "select"),
]


def read_wav(path):
    with wave.open(path, "rb") as w:
        sr, n, ch = w.getframerate(), w.getnframes(), w.getnchannels()
        raw = w.readframes(n)
    a = array.array("h"); a.frombytes(raw)
    if ch > 1:
        a = array.array("h", [sum(a[i:i+ch])//ch for i in range(0, len(a), ch)])
    pk = max(abs(v) for v in a) or 1
    return [v/pk for v in a]


def bandpass(sig, f_lo, f_hi, passes=2):
    fc = math.sqrt(f_lo*f_hi)
    bw = math.log2(f_hi/f_lo)
    w0 = 2*math.pi*fc/SR
    alpha = math.sin(w0)*math.sinh(math.log(2)/2*bw*w0/math.sin(w0))
    a0 = 1+alpha
    b0, b2 = alpha/a0, -alpha/a0
    a1, a2 = -2*math.cos(w0)/a0, (1-alpha)/a0
    y = sig
    for _ in range(passes):
        x1 = x2 = y1 = y2 = 0.0
        out = [0.0]*len(y)
        for i, x in enumerate(y):
            v = b0*x + b2*x2 - a1*y1 - a2*y2
            x1, x2 = x, x1
            y1, y2 = v, y1
            out[i] = v
        y = out
    return y


def env_db(sig, hop=None, win=None):
    hop = hop or int(0.002*SR)
    win = win or int(0.005*SR)
    out = []
    for i in range(0, max(1, len(sig)-win), hop):
        seg = sig[i:i+win]
        r = math.sqrt(sum(v*v for v in seg)/len(seg))
        out.append(20*math.log10(r+1e-9))
    return out


def onset(sig):
    pk = max(abs(v) for v in sig)
    thr = pk*0.06
    for i, v in enumerate(sig):
        if abs(v) > thr:
            return max(0, i-int(0.003*SR))
    return 0


def median(xs):
    s = sorted(xs)
    return s[len(s)//2] if s else 0.0


def band_diffs(er, em, shift, gate):
    n = min(len(er), len(em)-shift if shift >= 0 else len(em),
            int(0.5/0.002))
    out = []
    for i in range(max(0, -shift), n):
        if er[i] > gate:
            out.append(abs(er[i] - em[i+shift]))
    return out


def main():
    hdr = "sound        " + "".join(f"{lo}-{hi}".rjust(10) for lo, hi in BANDS) + "   overall"
    print(hdr)
    for ref_name, mine_name in PAIRS:
        ref = read_wav(os.path.join(PROJ, "ref", ref_name))
        mine = read_wav(os.path.join(PROJ, "out", "wav", mine_name + ".wav"))
        ref = ref[onset(ref):]
        mine = mine[onset(mine):]
        erb, emb = env_db(ref), env_db(mine)
        gate = max(erb) - 42  # only frames with perceptually relevant energy
        # align: pick the envelope-frame shift that best matches broadband
        best_shift = min(range(-3, 4),
                         key=lambda s: median(band_diffs(erb, emb, s, gate)))
        row, alld = [], []
        for lo, hi in BANDS:
            er = env_db(bandpass(ref, lo, hi))
            em = env_db(bandpass(mine, lo, hi))
            ds = band_diffs(er, em, best_shift, gate)
            if len(ds) < 3:
                row.append("       -  ")
                continue
            m = median(ds)
            row.append(f"{m:7.1f}dB".rjust(10))
            alld += ds
        total = median(alld)
        print(f"{mine_name:<13}" + "".join(row) + f"{total:8.1f}dB")


if __name__ == "__main__":
    main()
