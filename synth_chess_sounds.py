#!/usr/bin/env python3
"""Synthesize a free chess-app sound set (no samples, stdlib only).

Every sound is generated from scratch:
  * board sounds  — modal synthesis: a low "thump" with a fast downward
    pitch glide, a few inharmonic damped-sine wood modes, and a short
    band-passed noise attack
  * melodic cues  — small bell tones (inharmonic partials, soft attack)

Usage:    python3 synth_chess_sounds.py
Output:   out/wav/*.wav (44.1 kHz / 16-bit / mono)
          out/ogg/*.ogg, out/mp3/*.mp3   (if ffmpeg is on PATH)
          preview.html                    (click-to-audition page)

Tuning knobs: knock(pitch, weight, bright), the s_*() builders below,
and per-sound volume in the SOUNDS table.
"""

import array
import base64
import math
import os
import random
import shutil
import subprocess
import wave

SR = 44100
HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "out")
rnd = random.Random(20260811)  # fixed seed: regenerating gives identical files

# ---------------------------------------------------------------- primitives

def canvas(dur):
    return [0.0] * int(round(dur * SR))


def place(cv, sig, at=0.0):
    """Mix `sig` into `cv` starting at time `at` (seconds)."""
    i = int(round(at * SR))
    n = min(len(sig), len(cv) - i)
    for k in range(max(0, n)):
        cv[i + k] += sig[k]


def damped_sine(freq, tau, dur, amp=1.0, detune_cents=0.0, phase=None):
    n = int(round(dur * SR))
    f = freq * 2 ** (detune_cents / 1200.0)
    w = 2 * math.pi * f / SR
    ph = rnd.uniform(0, 2 * math.pi) if phase is None else phase
    decay = math.exp(-1.0 / (tau * SR))
    out = [0.0] * n
    e = 1.0
    for i in range(n):
        out[i] = amp * math.sin(w * i + ph) * e
        e *= decay
    return out


def bandpass(sig, f_lo, f_hi, passes=2):
    """RBJ constant-peak band-pass defined by band edges, run `passes` times."""
    fc = math.sqrt(f_lo * f_hi)
    bw_oct = math.log2(f_hi / f_lo)
    w0 = 2 * math.pi * fc / SR
    alpha = math.sin(w0) * math.sinh(math.log(2) / 2 * bw_oct * w0 / math.sin(w0))
    a0 = 1 + alpha
    b0, b2 = alpha / a0, -alpha / a0
    a1, a2 = -2 * math.cos(w0) / a0, (1 - alpha) / a0
    y = sig
    for _ in range(passes):
        x1 = x2 = y1 = y2 = 0.0
        out = [0.0] * len(y)
        for i, x in enumerate(y):
            v = b0 * x + b2 * x2 - a1 * y1 - a2 * y2
            x1, x2 = x, x1
            y1, y2 = v, y1
            out[i] = v
        y = out
    return y


def noise_burst(f_lo, f_hi, tau, dur, amp=1.0):
    n = int(round(dur * SR))
    decay = math.exp(-1.0 / (tau * SR))
    e = 1.0
    x = [0.0] * n
    for i in range(n):
        x[i] = rnd.gauss(0, 1) * e
        e *= decay
    y = bandpass(x, f_lo, f_hi)
    peak = max(abs(v) for v in y) or 1.0
    return [amp * v / peak for v in y]


BELL = ((1.0, 1.0), (2.0, 0.30), (2.76, 0.18), (4.07, 0.08))  # glockenspiel-ish
SOFT = ((1.0, 1.0), (2.0, 0.15))                              # mellow, near-sine


def bell(freq, tau, dur, amp=1.0, partials=BELL, attack=0.004):
    n = int(round(dur * SR))
    out = [0.0] * n
    for ratio, a in partials:
        ptau = tau / (1 + 0.8 * (ratio - 1))  # higher partials die faster
        comp = damped_sine(freq * ratio, ptau, dur, amp=a,
                           detune_cents=rnd.uniform(-1.5, 1.5))
        for i in range(n):
            out[i] += comp[i]
    e, ka = 1.0, math.exp(-1.0 / (attack * SR))  # soft attack
    for i in range(n):
        out[i] *= amp * (1 - e)
        e *= ka
    return out


# --------------------------------------------------- measured resynthesis
# fitted_params.py holds modal/noise parameters measured from the lichess
# standard sounds (frequencies, decay times, band levels — facts, no audio).

from fitted_params import PARAMS


def resynth(entry, pitch=1.0, amp=1.0, pad=0.05, bright=0.0, stretch=1.0,
            rng=None):
    """Vocoder-style resynthesis from measured parameters: per band, a
    carrier of [measured tone peaks + noise, mixed by measured tonality]
    is shaped by the reference's measured band envelope curve.

    Variation knobs: `pitch` shifts carriers/bands, `bright` tilts the
    spectrum (+-0.15 is subtle), `stretch` time-stretches the envelope
    (including a double hit's rhythm), `rng` picks another noise/phase
    realization ("another take of the same piece")."""
    rr = rng or rnd
    dur = entry["dur"] * stretch + pad
    c = canvas(dur)
    n = len(c)
    gap_i = int(round(entry["gap"] * stretch * SR))
    xf = max(1, int(0.002 * SR))  # carrier crossfade into the second hit
    for b in entry["bands"]:
        segs = []
        for k, h in enumerate(b["hits"]):
            start = gap_i * k
            m = n - start
            if m <= 0:
                continue
            car = [0.0] * m
            pn = h["noise"]                 # noise power fraction
            pt = max(0.0, 1.0 - pn)
            if h["tones"] and pt > 0:
                norm = math.sqrt(sum(a * a for _, a in h["tones"]) / 2.0)
                for f, a in h["tones"]:
                    w = 2 * math.pi * f * pitch / SR
                    ph = rr.uniform(0, 2 * math.pi)
                    g = a / norm * math.sqrt(pt)
                    for i in range(m):
                        car[i] += g * math.sin(w * i + ph)
            if pn > 0:
                nz = bandpass([rr.gauss(0.0, 1.0) for _ in range(m)],
                              b["lo"] * pitch, b["hi"] * pitch, passes=3)
                r = math.sqrt(sum(v * v for v in nz) / m) or 1.0
                g = math.sqrt(pn) / r
                for i in range(m):
                    car[i] += g * nz[i]
            segs.append((start, car))
        if not segs:
            continue
        full = list(segs[0][1])
        for start, car in segs[1:]:
            for i in range(start, n):
                w2 = min(1.0, (i - start) / xf)
                full[i] = full[i] * (1 - w2) + car[i - start] * w2
        tilt = (math.sqrt(b["lo"] * b["hi"]) / 700.0) ** bright
        hop_n = b["hop"] * SR * stretch
        curve = b["curve"]
        for i in range(n):
            j = i / hop_n
            j0 = int(j)
            if j0 >= len(curve):
                break
            g1 = curve[j0 + 1] if j0 + 1 < len(curve) else 0.0
            g = curve[j0] + (g1 - curve[j0]) * (j - j0)
            c[i] += full[i] * g * tilt
    return [amp * v for v in c]


def ref_gain(name):
    """Per-sound volume chosen so output peaks match the reference peaks."""
    return min(1.12, 10 ** (PARAMS[name]["peak_db"] / 20) / 0.891)

# ------------------------------------------------------------- sound builders
# note names used below: G5 784, A5 880, C6 1046.5, E6 1318.5, G6 1568, C7 2093

def s_move_self():
    return resynth(PARAMS["move"])


def s_move_opponent():
    return resynth(PARAMS["move"], pitch=0.96)


def s_capture():
    return resynth(PARAMS["capture"])


def s_castle():
    c = canvas(PARAMS["move"]["dur"] + 0.11 + 0.10)
    place(c, resynth(PARAMS["move"], pitch=1.03, amp=0.85), 0.0)
    place(c, resynth(PARAMS["move"], pitch=0.97), 0.110)
    return c


def s_promote():
    c = canvas(0.80)
    notes = ((1046.5, 0.12, 0.46), (1318.5, 0.12, 0.42),
             (1568.0, 0.12, 0.40), (2093.0, 0.20, 0.50))
    for i, (f, tau, a) in enumerate(notes):
        place(c, bell(f, tau, 0.60, amp=a), i * 0.055)
    return c


def s_premove():
    return resynth(PARAMS["select"], pitch=0.90)  # same tick family, lower


def s_select():
    return resynth(PARAMS["select"])


def s_illegal():
    return resynth(PARAMS["error"])


def s_notify():
    return resynth(PARAMS["notify"])


def s_game_start():
    c = canvas(0.90)
    place(c, bell(784.0, 0.16, 0.60, amp=0.50, partials=SOFT), 0.0)
    place(c, bell(1046.5, 0.22, 0.75, amp=0.55, partials=SOFT), 0.14)
    return c


def s_game_end():
    c = canvas(1.15)
    place(c, bell(1046.5, 0.12, 0.50, amp=0.50, partials=SOFT), 0.0)
    place(c, bell(784.0, 0.12, 0.50, amp=0.45, partials=SOFT), 0.13)
    place(c, bell(1046.5, 0.28, 0.80, amp=0.55, partials=SOFT), 0.26)
    return c


def s_game_win():
    c = canvas(1.20)
    notes = ((1046.5, 0.14, 0.48), (1318.5, 0.14, 0.46),
             (1568.0, 0.16, 0.46), (2093.0, 0.30, 0.55))
    for i, (f, tau, a) in enumerate(notes):
        place(c, bell(f, tau, 0.80, amp=a), i * 0.11)
    place(c, bell(1046.5, 0.25, 0.70, amp=0.22, partials=SOFT), 0.33)
    place(c, bell(1568.0, 0.25, 0.70, amp=0.20, partials=SOFT), 0.33)
    return c


def s_game_lose():
    c = canvas(1.10)
    notes = ((880.0, 0.16, 0.46), (698.46, 0.16, 0.42), (587.33, 0.26, 0.46))
    for i, (f, tau, a) in enumerate(notes):
        place(c, bell(f, tau, 0.80, amp=a, partials=SOFT), i * 0.15)
    return c


def s_game_draw():
    c = canvas(0.90)
    place(c, bell(784.0, 0.14, 0.60, amp=0.50, partials=SOFT), 0.0)
    place(c, bell(784.0, 0.20, 0.70, amp=0.50, partials=SOFT), 0.16)
    return c


def s_low_time():
    return resynth(PARAMS["lowtime"], pad=0.03)


# name, builder, volume (relative to -1 dBFS peak), description
# board/UI sounds use the measured reference peak levels via ref_gain()
SOUNDS = [
    ("move-self",     s_move_self,     ref_gain("move"),    "You move — soft wooden knock"),
    ("move-opponent", s_move_opponent, ref_gain("move"),    "Opponent moves — same knock, a shade lower"),
    ("capture",       s_capture,       ref_gain("capture"), "Capture — double clack-knock"),
    ("castle",        s_castle,        1.00,                "Castle — two knocks (king, then rook)"),
    ("promote",       s_promote,       0.85,                "Promotion — quick ascending sparkle"),
    ("premove",       s_premove,       0.30,                "Premove set — tiny tick"),
    ("select",        s_select,        ref_gain("select"),  "Piece selected — bright micro-tick"),
    ("illegal",       s_illegal,       ref_gain("error"),   "Illegal move — dissonant buzz ('uh-oh')"),
    ("notify",        s_notify,        ref_gain("notify"),  "Notification (draw offer etc.) — mellow ding"),
    ("game-start",    s_game_start,    0.85,                "Game start — two rising chimes"),
    ("game-end",      s_game_end,      0.85,                "Game over (neutral) — small resolving motif"),
    ("game-win",      s_game_win,      0.90,                "Win — ascending arpeggio into a chord"),
    ("game-lose",     s_game_lose,     0.80,                "Loss — gentle descending line"),
    ("game-draw",     s_game_draw,     0.80,                "Draw — two equal chimes"),
    ("low-time",      s_low_time,      ref_gain("lowtime"), "Clock low — soft warm tone"),
]

RELEASE = {"notify": 0.06, "low-time": 0.06}  # longer end fade for tonal tails


# Round-robin variations of the board sounds: same measured character,
# slightly different pitch / brightness / timing and a different "take"
# (noise/phase realization), so repeated moves don't sound like one sample.

def make_variant(src, seed, **kw):
    def build():
        return resynth(PARAMS[src], rng=random.Random(seed), **kw)
    return build


VARIANTS = [
    ("move-self-1",     "move",    "baseline take",    dict(seed=11)),
    ("move-self-2",     "move",    "crisper, higher",  dict(seed=12, pitch=1.035, bright=0.12, stretch=0.94)),
    ("move-self-3",     "move",    "duller, heavier",  dict(seed=13, pitch=0.970, bright=-0.10, stretch=1.06)),
    ("move-self-4",     "move",    "quick",            dict(seed=14, pitch=1.010, bright=-0.04, stretch=0.90)),
    ("move-opponent-1", "move",    "baseline take",    dict(seed=21, pitch=0.960)),
    ("move-opponent-2", "move",    "crisper",          dict(seed=22, pitch=0.990, bright=0.10, stretch=0.95)),
    ("move-opponent-3", "move",    "duller, heavier",  dict(seed=23, pitch=0.930, bright=-0.12, stretch=1.07)),
    ("move-opponent-4", "move",    "quick",            dict(seed=24, pitch=0.955, bright=-0.05, stretch=0.91)),
    ("capture-1",       "capture", "baseline take",    dict(seed=31)),
    ("capture-2",       "capture", "crisper, tighter", dict(seed=32, pitch=1.040, bright=0.10, stretch=0.92)),
    ("capture-3",       "capture", "duller, heavier",  dict(seed=33, pitch=0.950, bright=-0.10, stretch=1.08)),
    ("capture-4",       "capture", "quick",            dict(seed=34, pitch=1.005, bright=0.04, stretch=0.90)),
    ("capture-5",       "capture", "high and snappy",  dict(seed=35, pitch=1.070, bright=0.20, stretch=0.86)),
    ("capture-6",       "capture", "bright, light",    dict(seed=36, pitch=1.050, bright=0.15, stretch=0.95)),
    ("capture-7",       "capture", "max snap",         dict(seed=37, pitch=1.000, bright=0.24, stretch=0.85)),
    ("capture-8",       "capture", "highest pitch",    dict(seed=38, pitch=1.100, bright=0.12, stretch=0.90)),
]

for _name, _src, _desc, _kw in VARIANTS:
    SOUNDS.append((_name, make_variant(_src, **_kw), ref_gain(_src),
                   "round-robin take — " + _desc))

# -------------------------------------------------------------- finishing

def trim_tail(x, thresh_db=-60.0, pad=0.02):
    peak = max(abs(v) for v in x)
    if peak <= 0:
        return x
    floor = peak * 10 ** (thresh_db / 20)
    last = 0
    for i, v in enumerate(x):
        if abs(v) > floor:
            last = i
    return x[: min(len(x), last + int(pad * SR))]


def finalize(x, gain, release=0.012):
    x = x[:]
    n = len(x)
    a = min(n, int(0.0004 * SR))  # 0.4 ms fade-in kills onset clicks
    for i in range(a):
        x[i] *= i / a
    r = min(n, int(release * SR))  # fade-out
    for i in range(r):
        x[n - r + i] *= 0.5 * (1 + math.cos(math.pi * i / r))
    peak = max(abs(v) for v in x)
    if peak > 0:
        k = 0.891 * gain / peak  # peak at -1 dBFS, scaled by per-sound volume
        x = [v * k for v in x]
    return x


def write_wav(path, x):
    pcm = array.array("h", (int(max(-1.0, min(1.0, v)) * 32767) for v in x))
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


# ------------------------------------------------------------ preview page

PREVIEW_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Chess sound set — preview</title>
<style>
:root{color-scheme:light dark;--bg:#f6f3ee;--card:#fff;--ink:#26211b;--sub:#7a7061;--edge:#e2dbd0;--hi:#7a5c37}
@media(prefers-color-scheme:dark){:root{--bg:#191512;--card:#221d18;--ink:#ece5da;--sub:#9a8f7f;--edge:#372f27;--hi:#c9a06a}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.45 system-ui,-apple-system,sans-serif;padding:40px 20px}
main{max-width:780px;margin:0 auto}
h1{font-size:22px;margin:0 0 4px}
p.sub{color:var(--sub);margin:0 0 20px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:10px}
button.s{display:flex;flex-direction:column;gap:4px;text-align:left;padding:12px 14px;border:1px solid var(--edge);border-radius:10px;background:var(--card);color:var(--ink);cursor:pointer;font:inherit}
button.s:hover{border-color:var(--hi)}
button.s:active{transform:translateY(1px)}
button.s.on{border-color:var(--hi);box-shadow:0 0 0 1px var(--hi)}
.n{font-weight:600}.d{color:var(--sub);font-size:13px}.t{color:var(--sub);font-size:12px}
#all{margin:0 0 18px;padding:10px 16px;border-radius:10px;border:1px solid var(--hi);background:transparent;color:var(--ink);font:inherit;cursor:pointer}
</style></head><body><main>
<h1>&#9822; Chess sound set — preview</h1>
<p class="sub">Synthesized from scratch &middot; 44.1 kHz mono &middot; click a card to play</p>
<button id="all">&#9654; Play all</button>
<div class="grid">__ROWS__</div>
</main><script>
const S={__MAP__};
const play=n=>{const a=new Audio(S[n]);a.play();return a};
document.querySelectorAll("button.s").forEach(b=>b.addEventListener("click",()=>play(b.dataset.s)));
document.getElementById("all").addEventListener("click",async()=>{
  for(const b of document.querySelectorAll("button.s")){
    b.classList.add("on");
    await new Promise(r=>{const a=play(b.dataset.s);a.addEventListener("ended",r);setTimeout(r,2000)});
    await new Promise(r=>setTimeout(r,250));
    b.classList.remove("on");
  }
});
</script></body></html>
"""


def make_preview(entries, path):
    rows, jsmap = [], []
    for name, desc, dur, wav_path in entries:
        with open(wav_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        jsmap.append('"%s":"data:audio/wav;base64,%s"' % (name, b64))
        rows.append(
            '<button class="s" data-s="%s"><span class="n">%s</span>'
            '<span class="d">%s</span><span class="t">%.2f s</span></button>'
            % (name, name, desc, dur))
    html = (PREVIEW_TEMPLATE
            .replace("__ROWS__", "\n".join(rows))
            .replace("__MAP__", ",".join(jsmap)))
    with open(path, "w") as f:
        f.write(html)

# --------------------------------------------------------------------- main

def main():
    wav_dir = os.path.join(OUTDIR, "wav")
    os.makedirs(wav_dir, exist_ok=True)
    entries = []
    print("%-14s %7s %9s %9s" % ("name", "dur", "peak", "rms"))
    for name, builder, gain, desc in SOUNDS:
        x = finalize(trim_tail(builder()), gain, RELEASE.get(name, 0.012))
        path = os.path.join(wav_dir, name + ".wav")
        write_wav(path, x)
        peak = 20 * math.log10(max(abs(v) for v in x) + 1e-12)
        rms = 20 * math.log10(math.sqrt(sum(v * v for v in x) / len(x)) + 1e-12)
        dur = len(x) / SR
        entries.append((name, desc, dur, path))
        print("%-14s %6.2fs %7.1fdB %7.1fdB" % (name, dur, peak, rms))

    if shutil.which("ffmpeg"):
        for fmt, args in (("ogg", ["-c:a", "libvorbis", "-qscale:a", "5"]),
                          ("mp3", ["-c:a", "libmp3lame", "-qscale:a", "2"])):
            d = os.path.join(OUTDIR, fmt)
            os.makedirs(d, exist_ok=True)
            for name, _, _, wav_path in entries:
                subprocess.run(
                    ["ffmpeg", "-y", "-loglevel", "error", "-i", wav_path,
                     *args, os.path.join(d, "%s.%s" % (name, fmt))],
                    check=True)
        print("converted to ogg + mp3")
    else:
        print("ffmpeg not found — wrote WAV only")

    make_preview(entries, os.path.join(HERE, "preview.html"))
    print("wrote preview.html")


if __name__ == "__main__":
    main()
