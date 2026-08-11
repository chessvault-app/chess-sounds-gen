#!/usr/bin/env python3
"""Measure vocoder-style resynthesis parameters from short reference sounds
and emit fitted_params.py.

Per sound, per frequency band (9 log-spaced bands, 80 Hz - 11 kHz):
  * the band's RMS envelope curve (2 ms grid), tail-floor subtracted in
    power so recording noise floor / hum is not reproduced
  * per hit (double hits like capture are segmented): the band's tonality
    (peak-to-median spectrum ratio -> noise power fraction) and up to 4
    spectral peak frequencies with relative amplitudes (the carrier)

The synthesizer rebuilds each band as [tones + noise] * envelope, which
reproduces the reference's time-frequency energy envelope by construction
while preserving tonality where the reference is tonal.

Only measured numbers are emitted — frequencies, ratios, envelope samples.
No audio data is copied.
"""
import array, cmath, math, os, wave

SR = 44100
HERE = os.path.dirname(os.path.abspath(__file__))

BANDS = [(80, 200), (200, 400), (400, 650), (650, 1000), (1000, 1600),
         (1600, 2600), (2600, 4200), (4200, 7000), (7000, 11000),
         (11000, 16000)]

HOP = 0.001        # broadband envelope hop (s)
WIN = 0.003        # broadband envelope RMS window (s)


def env_params(lo):
    """Per-band envelope resolution: fine grids in the mids/highs keep the
    attack transient sharp; low bands need longer RMS windows."""
    if lo >= 400:
        return 0.0005, 0.0015, 0.001   # hop, win, stored curve grid
    return 0.001, 0.004, 0.002


def read_wav(path):
    with wave.open(path, "rb") as w:
        sr, n, ch = w.getframerate(), w.getnframes(), w.getnchannels()
        raw = w.readframes(n)
    a = array.array("h"); a.frombytes(raw)
    if ch > 1:
        a = array.array("h", [sum(a[i:i+ch])//ch for i in range(0, len(a), ch)])
    assert sr == SR, f"{path}: {sr}"
    return [v/32768.0 for v in a]


def env_rms(sig, hop_s=HOP, win_s=WIN):
    hop, win = int(hop_s*SR), int(win_s*SR)
    out = []
    for i in range(0, max(1, len(sig)-win), hop):
        seg = sig[i:i+win]
        out.append(math.sqrt(sum(v*v for v in seg)/len(seg)))
    return out


def fft(x):
    n = len(x); x = x[:]
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit: j ^= bit; bit >>= 1
        j |= bit
        if i < j: x[i], x[j] = x[j], x[i]
    m = 2
    while m <= n:
        wm = cmath.exp(-2j*math.pi/m)
        for k in range(0, n, m):
            w = 1+0j
            for l in range(m//2):
                t = w*x[k+l+m//2]; u = x[k+l]
                x[k+l] = u+t; x[k+l+m//2] = u-t
                w *= wm
        m <<= 1
    return x


def spectrum(sig, start, length_s, nfft=16384):
    seg = sig[start:start+int(length_s*SR)]
    L = len(seg)
    if L < 32: return [0.0]*(nfft//2), SR/nfft
    seg = [seg[i]*(0.5 + 0.5*math.cos(math.pi*i/(L-1))) for i in range(L)]
    seg = seg + [0.0]*(nfft-L) if L < nfft else seg[:nfft]
    X = fft([complex(v, 0.0) for v in seg])
    return [abs(X[i]) for i in range(nfft//2)], SR/nfft


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


def onsets(sig, force_single):
    env = env_rms(sig)
    hop = int(HOP*SR)
    pk = max(env); pi = env.index(pk)
    thr = pk*0.06
    o1 = 0
    for i, e in enumerate(env):
        if e > thr:
            o1 = max(0, i*hop - int(0.002*SR)); break
    o2 = None
    if not force_single:
        lo = pi + int(0.025/HOP); hi = min(len(env), pi + int(0.130/HOP))
        if lo < hi:
            seg = env[lo:hi]
            m = max(seg)
            if m > pk*10**(-18/20) and min(env[pi:lo+seg.index(m)]) < m*0.5:
                o2 = (lo+seg.index(m))*hop - int(0.004*SR)
    return o1, o2, env, hop, pk


def band_hit_spec(mags, binw, lo, hi):
    """Tonality + carrier peaks of one band in one hit's spectrum."""
    i0, i1 = max(1, int(lo/binw)), int(hi/binw)
    bins = mags[i0:i1]
    if not bins: return {"noise": 1.0, "tones": []}
    mx = max(bins)
    med = sorted(bins)[len(bins)//2]
    pm = 20*math.log10((mx+1e-12)/(med+1e-12))
    pn = max(0.05, min(1.0, (28.0-pm)/16.0))  # peak/median dB -> noise power
    tones = []
    for j in range(1, len(bins)-1):
        if bins[j] >= bins[j-1] and bins[j] >= bins[j+1] and bins[j] > mx*10**(-18/20):
            tones.append((bins[j], (i0+j)*binw))
    tones.sort(reverse=True)
    sel = []
    for m, f in tones:
        if all(abs(f-f2) >= 25 for _, f2 in sel):
            sel.append((m, f))
        if len(sel) >= 4: break
    mmax = sel[0][0] if sel else 1.0
    return {"noise": round(pn, 3),
            "tones": [(round(f, 1), round(m/mmax, 4)) for m, f in sel]}


def process(name, path, force_single=False):
    sig = read_wav(path)
    o1, o2, env, hop, pk = onsets(sig, force_single)
    peak_db = 20*math.log10(max(abs(v) for v in sig)+1e-12)

    thr = pk*10**(-52/20)
    last = o1//hop
    for i in range(o1//hop, len(env)):
        if env[i] > thr: last = i
    dur = max(0.06, (last*hop - o1)/SR + 0.015)
    gap = ((o2 or o1) - o1)/SR

    # analysis span extends well past the broadband-audible end so slow
    # low-frequency tails are captured and the noise floor is estimated
    # from genuinely quiet frames
    seg = sig[o1:o1+int((dur+0.12)*SR)]

    # per-hit spectra
    specs = []
    starts = [o1] + ([o2] if o2 else [])
    for i, st in enumerate(starts):
        end_s = (starts[i+1]-st)/SR if i+1 < len(starts) else dur - (st-o1)/SR
        specs.append(spectrum(sig, st, min(0.35, max(0.03, end_s))))

    bands = []
    targets = {}
    for lo, hi in BANDS:
        hop_s, win_s, curve_hop = env_params(lo)
        be = env_rms(bandpass(seg, lo, hi), hop_s, win_s)
        tail = sorted(be[-max(3, len(be)//7):])
        floor = tail[len(tail)//4]   # lower quartile of the quiet end
        be = [math.sqrt(max(0.0, v*v - floor*floor)) for v in be]
        if max(be) < pk*10**(-50/20):
            continue
        targets[(lo, hi)] = sum(v*v for v in be)   # band energy
        step = int(curve_hop/hop_s)
        curve = [round(be[i], 6) for i in range(0, len(be), step)]
        trim = pk*10**(-55/20)
        while curve and curve[-1] < trim:
            curve.pop()
        if not curve:
            continue
        hits = [band_hit_spec(mags, binw, lo, hi) for mags, binw in specs]
        bands.append({"lo": lo, "hi": hi, "hop": curve_hop,
                      "curve": curve, "hits": hits})

    # let slow band tails (e.g. a capture's ~100 Hz thud) define the length
    tail_end = max((len(b["curve"])*b["hop"] for b in bands), default=dur)
    dur = max(dur, tail_end + 0.01)

    entry = {"peak_db": round(peak_db, 1), "dur": round(dur, 3),
             "gap": round(gap, 4), "bands": bands}
    # note: no closed-loop calibration against a re-render — narrowband
    # carriers physically cannot reproduce a broadband click's in-band
    # peak (it is masked by the louder mid bands at that instant), and
    # correcting for it just pumps the tails. Direct measurement + steep
    # carrier filters is both simpler and closer.

    print(f"{name}: dur {dur:.2f}s, peak {peak_db:.1f}dB"
          + (f", 2nd hit +{gap*1000:.0f}ms" if o2 else ""))
    for b in bands:
        h0 = b["hits"][0]
        tf = ",".join(f"{f:.0f}" for f, _ in h0["tones"][:3]) or "-"
        print(f"   {b['lo']}-{b['hi']}: env {len(b['curve'])*b['hop']*1000:.0f}ms "
              f"pk{20*math.log10(max(b['curve'])+1e-9):5.0f}dB  "
              f"noise {h0['noise']:.2f}  tones {tf}")
    return entry


def main():
    ref = os.path.join(os.path.dirname(HERE), "ref")
    jobs = [
        ("move", "Move.wav", False),
        ("capture", "Capture.wav", False),
        ("error", "Error.wav", True),
        ("notify", "GenericNotify.wav", True),
        ("lowtime", "LowTime.wav", True),
        ("select", "Select.wav", True),
    ]
    out = {}
    for name, fn, single in jobs:
        out[name] = process(name, os.path.join(ref, fn), force_single=single)
    dst = os.path.join(os.path.dirname(HERE), "fitted_params.py")
    with open(dst, "w") as f:
        f.write('"""Auto-generated by modal_fit.py: per-band envelope curves,\n'
                'tonality fractions and spectral-peak carriers measured from\n'
                'reference sounds (measured facts; contains no audio data)."""\n\n')
        import pprint
        f.write("PARAMS = ")
        f.write(pprint.pformat(out, width=100))
        f.write("\n")
    print("wrote", dst)


if __name__ == "__main__":
    main()
