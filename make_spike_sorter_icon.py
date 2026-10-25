"""Generate the Spike Sorter app icon as an SVG.

Usage:
    python make_spike_sorter_icon.py [output.svg]

Requires: numpy
Optional PNG export:  pip install cairosvg
    python -c "import cairosvg; cairosvg.svg2png(url='spike-sorter-icon.svg', write_to='icon.png', output_width=1024)"

What it draws (1024 x 1024, rounded-square base):
  * a 32-site silicon probe (two staggered columns of 16)
  * four active channels, each one a single continuous trace: a thin noisy
    baseline that turns into a bold spike and then continues
  * a different waveform shape per unit; one channel carries two units
    (its electrode marker is split into two colors)
  * a PCA-style scatter of the five sorted clusters behind everything
  * everything is centered on the icon automatically
"""
import re
import sys

import numpy as np

OUT_PATH = sys.argv[1] if len(sys.argv) > 1 else "spike-sorter-icon.svg"
SEED = 11  # changes the random scatter points and baseline noise
rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------- palette
COL = {
    "sky": "#4cc9ff",
    "coral": "#ff6b57",
    "gold": "#ffd23f",
    "mint": "#4ff0a0",
    "orchid": "#e08cff",
}
BG_TOP, BG_BOTTOM = "#3a1d63", "#0e0620"  # background gradient
SHANK_LIGHT, SHANK_DARK = "#f3ecff", "#a798c9"  # probe body gradient
SHANK_EDGE = "#fbf8ff"
SITE_IDLE = "#1a0f33"  # inactive electrode sites
GUIDE = "#b9a9e0"  # PCA crosshair and noise dots

# ---------------------------------------------------- PCA background clusters
# (unit, center x, center y, std x, std y, rotation in degrees, n points)
CLUSTERS = [
    ("sky", 625, 326, 62, 22, -26, 150),
    ("coral", 805, 336, 48, 24, 38, 140),
    ("gold", 835, 480, 42, 22, -48, 130),
    ("mint", 610, 634, 60, 24, 18, 150),
    ("orchid", 790, 800, 54, 22, -14, 140),
]
# scatter is clipped to this box (x0, x1, y0, y1), in pre-centering coordinates
X0, X1, Y0, Y1 = 395, 940, 125, 885


def cluster_points(cx, cy, sx, sy, ang, n):
    a = np.deg2rad(ang)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    z = rng.normal(size=(n, 2)) * np.array([sx, sy])
    p = z @ rot.T + np.array([cx, cy])
    keep = (p[:, 0] > X0) & (p[:, 0] < X1) & (p[:, 1] > Y0) & (p[:, 1] < Y1)
    return p[keep]


def circles(pts):
    out = []
    for x, y in pts:
        r = 3.5 if rng.random() < 0.55 else 4.5
        out.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="{r}"/>')
    return "".join(out)


pca = []
for name, cx, cy, sx, sy, ang, n in CLUSTERS:  # soft halos first
    pca.append(
        f'<ellipse cx="{cx}" cy="{cy}" rx="{sx * 2.6:.0f}" ry="{sy * 3.4:.0f}" '
        f'transform="rotate({ang} {cx} {cy})" fill="url(#halo-{name})"/>'
    )
cl_pts = []
for name, cx, cy, sx, sy, ang, n in CLUSTERS:  # then the dots
    pts = cluster_points(cx, cy, sx, sy, ang, n)
    cl_pts.append(pts)
    pca.append(f'<g fill="{COL[name]}" opacity="0.85">{circles(pts)}</g>')

# ------------------------------------------------------------- layout / centering
END_X = 926  # right end of the traces
# content bounds before centering: probe left edge, probe top, probe tip, right edge
CONTENT_L, CONTENT_T, CONTENT_B = 235, 99, 949
XR = max(max(q[:, 0].max() for q in cl_pts) + 4.5, END_X + 4)
DX = 512 - (CONTENT_L + XR) / 2
DY = 512 - (CONTENT_T + CONTENT_B) / 2

noise = np.column_stack([rng.uniform(X0, XR - 5, 70), rng.uniform(Y0, Y1, 70)])
pca.append(f'<g fill="{GUIDE}" opacity="0.35">{circles(noise)}</g>')

# ------------------------------------------------ probe: 32 sites, 2 columns x 16
LX, RX = 276, 324
left = [(LX, 160 + 44 * i) for i in range(16)]
right = [(RX, 182 + 44 * i) for i in range(16)]
# active sites: (column, row index) -> unit color of that channel
ACTIVE = {("L", 2): "sky", ("R", 5): "coral", ("L", 9): "mint", ("R", 12): "orchid"}
idle = []
for i, (x, y) in enumerate(left):
    if ("L", i) not in ACTIVE:
        idle.append(f'<circle cx="{x}" cy="{y}" r="11"/>')
for i, (x, y) in enumerate(right):
    if ("R", i) not in ACTIVE:
        idle.append(f'<circle cx="{x}" cy="{y}" r="11"/>')

# ------------------------------------------- waveforms (origin = baseline, trough down)
WF = {
    # fast, narrow, tall biphasic
    "sky": "M0 0 L8 0 Q12 -7 17 -3 L30 100 L46 -34 Q56 -16 68 0 L74 0",
    # broad trough, slow rebound
    "coral": "M0 0 C8 0 12 -5 18 3 C28 32 34 62 46 62 C60 62 64 14 76 -12 C86 -26 100 -18 116 0 L122 0",
    # positive-first triphasic
    "gold": "M0 0 C6 -2 10 -46 20 -46 C30 -46 32 62 48 64 C60 66 66 -22 78 -20 C86 -18 90 0 100 0",
    # shallow monophasic with slow tail
    "mint": "M0 0 L6 0 C16 0 20 50 34 50 C48 50 52 22 66 8 C76 -2 88 -6 104 -3 L112 0",
    # small trough, tall positive peak
    "orchid": "M0 0 L8 0 L20 46 L34 -86 Q44 -52 52 -14 Q60 6 72 0",
}


def shifted(d, dx, dy):
    """Translate an absolute SVG path (M L Q C only); return (path, end_x)."""
    out, last_x, k = [], dx, 0
    for tok in re.findall(r"[MLQC]|-?\d+\.?\d*", d):
        if tok in "MLQC":
            out.append(tok)
            k = 0
        else:
            v = float(tok) + (dx if k % 2 == 0 else dy)
            if k % 2 == 0:
                last_x = v
            out.append(f"{v:g}")
            k += 1
    return " ".join(out), last_x


def noisy(xa, xb, y):
    """Baseline from xa to xb with small jitter, pinned exactly to y at both ends."""
    pts = [(xa, y)]
    x = xa + 5
    while x < xb - 3:
        pts.append((x, y + rng.normal(0, 1.5)))
        x += 5
    pts.append((xb, y))
    return "M" + " L".join(f"{px:.0f} {py:.1f}" for px, py in pts)


BASE_STROKE = 4  # thin baseline
SPIKE_STROKE = 10  # bold spikes

# (baseline y, channel color, [(unit waveform, spike start x), ...], start x)
TRACES = [
    (248, "sky", [("sky", 470), ("sky", 640), ("sky", 780)], LX),
    (402, "coral", [("coral", 440), ("gold", 590), ("coral", 720)], RX),  # two units
    (556, "mint", [("mint", 440), ("mint", 610), ("mint", 780)], LX),
    (710, "orchid", [("orchid", 480), ("orchid", 620), ("orchid", 810)], RX),
]

tr = []
cap = 'fill="none" stroke-linecap="round" stroke-linejoin="round"'
for y, c, spikes, x_start in TRACES:
    cur = x_start
    base, spk = [], []
    for unit, x in spikes:
        base.append(noisy(cur, x, y))  # baseline stops where the spike starts
        d, end_x = shifted(WF[unit], x, y)
        spk.append((unit, d))
        cur = end_x  # and resumes where the spike ends
    base.append(noisy(cur, END_X, y))
    for d in base:
        tr.append(f'<path d="{d}" {cap} stroke-width="{BASE_STROKE}" stroke="{COL[c]}"/>')
    for unit, d in spk:
        tr.append(
            f'<path d="{d}" {cap} stroke-width="{SPIKE_STROKE}" stroke="{COL[unit]}" '
            f'filter="url(#glow)"/>'
        )

# ---------------------------------------------------------- active site markers
sites = []
for (side, i), c in ACTIVE.items():
    x, y = (left if side == "L" else right)[i]
    if c == "coral":
        # channel with two units: marker split into coral and gold halves
        sites.append(f'<circle cx="{x}" cy="{y}" r="20" fill="{COL["coral"]}" opacity="0.28"/>')
        sites.append(f'<path d="M{x} {y - 12} A12 12 0 0 0 {x} {y + 12} Z" fill="{COL["coral"]}"/>')
        sites.append(f'<path d="M{x} {y - 12} A12 12 0 0 1 {x} {y + 12} Z" fill="{COL["gold"]}"/>')
    else:
        sites.append(f'<circle cx="{x}" cy="{y}" r="20" fill="{COL[c]}" opacity="0.30"/>')
        sites.append(f'<circle cx="{x}" cy="{y}" r="12" fill="{COL[c]}"/>')

halo_defs = "".join(
    f'<radialGradient id="halo-{n}"><stop offset="0" stop-color="{c}" stop-opacity="0.32"/>'
    f'<stop offset="1" stop-color="{c}" stop-opacity="0"/></radialGradient>'
    for n, c in COL.items()
)

# ------------------------------------------------------------------ assemble SVG
svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1024 1024" width="1024" height="1024">
  <title>Spike Sorter</title>
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="{BG_TOP}"/>
      <stop offset="1" stop-color="{BG_BOTTOM}"/>
    </linearGradient>
    <linearGradient id="shank" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0" stop-color="{SHANK_LIGHT}"/>
      <stop offset="1" stop-color="{SHANK_DARK}"/>
    </linearGradient>
    <radialGradient id="shine" cx="0.25" cy="0.12" r="0.9">
      <stop offset="0" stop-color="#ffffff" stop-opacity="0.14"/>
      <stop offset="1" stop-color="#ffffff" stop-opacity="0"/>
    </radialGradient>
    {halo_defs}
    <filter id="glow" x="-30%" y="-30%" width="160%" height="160%">
      <feGaussianBlur stdDeviation="8" result="b"/>
      <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
  </defs>

  <!-- icon base -->
  <rect width="1024" height="1024" rx="230" fill="url(#bg)"/>
  <rect width="1024" height="1024" rx="230" fill="url(#shine)"/>

  <!-- all content, centered on the icon -->
  <g transform="translate({DX:.1f} {DY:.1f})">
  <!-- PCA scatter of sorted clusters (background) -->
  <g>
    <line x1="{X0}" y1="505" x2="{XR:.0f}" y2="505" stroke="{GUIDE}" stroke-opacity="0.12" stroke-width="2" stroke-dasharray="4 10"/>
    <line x1="690" y1="{Y0}" x2="690" y2="{Y1}" stroke="{GUIDE}" stroke-opacity="0.12" stroke-width="2" stroke-dasharray="4 10"/>
    {"".join(pca)}
  </g>

  <!-- silicon probe shank, 32 sites -->
  <path d="M240 104 H360 V872 L300 944 L240 872 Z" fill="url(#shank)" stroke="{SHANK_EDGE}" stroke-width="10" stroke-linejoin="round"/>
  <rect x="262" y="104" width="76" height="20" rx="6" fill="#8576ad" opacity="0.55"/>
  <g fill="{SITE_IDLE}">{"".join(idle)}</g>

  <!-- traces and spikes -->
  {"".join(tr)}

  <!-- active electrode sites -->
  {"".join(sites)}
  </g>
</svg>
'''

with open(OUT_PATH, "w") as f:
    f.write(svg)
print(f"wrote {OUT_PATH} ({len(svg)} bytes, {len(left) + len(right)} sites)")
