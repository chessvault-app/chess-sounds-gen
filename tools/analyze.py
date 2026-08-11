#!/usr/bin/env python3
"""Numeric fingerprint of short UI sounds: envelope decay + spectral peaks."""
import array, cmath, math, sys, wave


def read_wav(path):
    with wave.open(path, "rb") as w:
        sr, n, ch = w.getframerate(), w.getnframes(), w.getnchannels()
        raw = w.readframes(n)
    a = array.array("h")
    a.frombytes(raw)
    if ch > 1:
        a = array.array("h", [sum(a[i:i + ch]) // ch for i in range(0, len(a), ch)])
    return sr, [v / 32768.0 for v in a]


def fft(x):
    n = len(x)
    x = x[:]
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j |= bit
        if i < j:
            x[i], x[j] = x[j], x[i]
    m = 2
    while m <= n:
        wm = cmath.exp(-2j * math.pi / m)
        for k in range(0, n, m):
            w = 1 + 0j
            for l in range(m // 2):
                t = w * x[k + l + m // 2]
                u = x[k + l]
                x[k + l] = u + t
                x[k + l + m // 2] = u - t
                w *= wm
        m <<= 1
    return x


def envelope(sig, sr, hop_ms=2.0, win_ms=5.0):
    hop, win = int(sr * hop_ms / 1000), int(sr * win_ms / 1000)
    env = []
    for i in range(0, max(1, len(sig) - win), hop):
        seg = sig[i:i + win]
        env.append(math.sqrt(sum(v * v for v in seg) / len(seg)))
    return env, hop


def spectral_peaks(sig, sr, start, size=4096, floor_db=-35.0, min_sep=60.0, top=8):
    seg = sig[start:start + size]
    seg = seg + [0.0] * (size - len(seg))
    w = [seg[i] * (0.5 - 0.5 * math.cos(2 * math.pi * i / (size - 1))) for i in range(size)]
    X = fft([complex(v, 0.0) for v in w])
    mags = [abs(X[i]) for i in range(size // 2)]
    mx = max(mags) or 1.0
    db = [20 * math.log10(m / mx + 1e-12) for m in mags]
    binw = sr / size
    cand = []
    for i in range(2, size // 2 - 1):
        f = i * binw
        if 40 <= f <= 10000 and db[i] > floor_db and db[i] >= db[i - 1] and db[i] >= db[i + 1]:
            cand.append((db[i], f))
    cand.sort(reverse=True)
    picked = []
    for d, f in cand:
        if all(abs(f - pf) >= min_sep for _, pf in picked):
            picked.append((d, f))
        if len(picked) >= top:
            break
    return picked


def analyze(path):
    sr, sig = read_wav(path)
    if not sig:
        print(f"{path}: empty")
        return
    peak = max(abs(v) for v in sig)
    rms = math.sqrt(sum(v * v for v in sig) / len(sig))
    env, hop = envelope(sig, sr)
    epk = max(env)
    pi = env.index(epk)
    t20 = t40 = None
    for i in range(pi, len(env)):
        if t20 is None and env[i] < epk * 0.1:
            t20 = (i - pi) * hop / sr * 1000
        if t40 is None and env[i] < epk * 0.01:
            t40 = (i - pi) * hop / sr * 1000
            break
    onset = 0
    for i, e in enumerate(env):
        if e > epk * 0.1:
            onset = max(0, i * hop - 64)
            break
    name = path.split("/")[-1]
    fmt = lambda v: f"{v:6.0f}ms" if v is not None else "   n/a "
    print(f"{name:<22} {len(sig)/sr:5.2f}s  peak {20*math.log10(peak+1e-12):6.1f}dB  "
          f"rms {20*math.log10(rms+1e-12):6.1f}dB  t-20 {fmt(t20)}  t-40 {fmt(t40)}")
    peaks = spectral_peaks(sig, sr, onset)
    print("    onset peaks: " + "  ".join(f"{f:6.0f}Hz({d:5.1f})" for d, f in peaks))
    later = onset + int(0.08 * sr)
    if later + 1024 < len(sig):
        peaks2 = spectral_peaks(sig, sr, later, size=2048)
        print("    +80ms peaks: " + "  ".join(f"{f:6.0f}Hz({d:5.1f})" for d, f in peaks2))


if __name__ == "__main__":
    for p in sys.argv[1:]:
        analyze(p)
