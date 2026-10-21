#!/usr/bin/env python3
"""[2026-10-21] 变更：世界模型扩为 500m x 500m（10x10 tile），中央十字放 10/20/30 度长坡。
Generate a 500 m x 500 m complex-terrain set for the ancnw scene.

The world is a 10x10 grid of 50 m x 50 m themed heightfield tiles
(centers at i*50 - 225, j*50 - 225 for i,j in 0..9 -> world spans [-250, 250]).

Themes (per tile):
    ramps  : long 10 / 20 / 30 deg up/down ramps (two consecutive humps per tile),
             via a central "+" band so the robot meets them first from the origin.
             Each ramp tile gets one of the three slopes (see ramp_angle_deg) and
             its heightfield z_top is set to that hump's height so the slope stays
             exact after MuJoCo's min-max normalization.
    rubble : rough high-frequency rubble terrain.
    grass / sand / steps : the original rolling / dune / staircase themes.

Each tile is a SEPARATE MuJoCo heightfield asset (different friction/material
per theme). Every tile forces its border ring to elevation 0, so neighbouring
tiles meet flat at their shared edge -> the robot spawn (ancnw.xml: base_link
pos "0 0 0.537") sits at world (0,0) which is the corner of the 4 central
tiles, all of them ramps -> level spawn, seamless, and immediate ramp training.

Output format: MuJoCo custom hfield binary (npy is NOT supported by
<hfield file="...">): int32 nrow, int32 ncol, then nrow*ncol float32 elevation
values in [0, 1], row-major. MuJoCo min-max normalizes the data on load, so we
emit values spanning [0, 1] (border = 0, an interior peak = 1). Surface height
for normalized value h is  z_local = h * z_top  (see scene.xml size 3rd value),
geom placed at z = 0 so the border sits at world z = 0 and interiors rise.

This script ALSO emits two XML fragments (terrain_assets.frag / terrain_bodies.frag)
into the assets directory, listing all 100 <hfield> and <geom> entries. They are
injected into scene.xml (between the TILE_HFIELDS / TILE_GEOMS markers) by the
caller.

Usage:
    python3 policy/ancnw/tools_gen_terrain.py
"""
import math
import os
import struct
import numpy as np

TILE_M = 50.0    # each tile is TILE_M x TILE_M meters
NX = 10
NY = 10
HALF = 25.0      # tile half-extent (hfield size xhalf/yhalf)
RES = 256        # grid points per tile -> 50/256 ~= 0.195 m per cell
SEED = 7

# vertical scale (hfield size z) per theme; must match scene.xml and the
# per-theme scaling inside make_tile so that slopes/heights come out exact.
THEME_Z = {
    "grass": 0.50,
    "sand": 0.50,
    "steps": 0.90,
    "rubble": 0.30,
    "ramps": 4.90,   # fallback; ramp tiles use a per-slope z_top via tile_z()
}

THEME_FRICTION = {
    "grass": 0.9,
    "sand": 0.4,
    "steps": 0.7,
    "rubble": 0.7,
    "ramps": 0.7,
}

ASSETS = os.path.join(
    os.path.dirname(__file__), "..", "..", "src", "rl_sar_zoo",
    "ancnw_description", "mjcf", "assets",
)


# Ramp slope angles (deg) available for the central cross band, and the horizontal
# run of one ramp (m). Hump height H = tan(angle) * RAMP_D_RISE -> per-tile z_top.
RAMP_ANGLES = [10, 20, 30]
RAMP_D_RISE = 7.0


def theme_for(ix, iy):
    """Central '+' band -> long 10/20/30 deg ramps (met first from the origin)."""
    if ix in (4, 5) or iy in (4, 5):
        return "ramps"
    return ["grass", "sand", "steps", "rubble"][(ix + iy) % 4]


def ramp_angle_deg(ix, iy):
    """Pick one of RAMP_ANGLES for a ramp tile (deterministic, mixes all three)."""
    return RAMP_ANGLES[(ix * 7 + iy * 13) % len(RAMP_ANGLES)]


def tile_z(ix, iy):
    """Per-tile hfield z_top. Ramp tiles get H = tan(angle)*run so the slope is exact."""
    theme = theme_for(ix, iy)
    if theme == "ramps":
        ang = ramp_angle_deg(ix, iy)
        return math.tan(math.radians(ang)) * RAMP_D_RISE
    return THEME_Z[theme]


def _feather(x, lo, hi, f=3.0):
    return np.clip((x - lo) / f, 0.0, 1.0) * np.clip((hi - x) / f, 0.0, 1.0)


def make_tile(theme, res, rng, Z):
    """Return a (res, res) normalized elevation array in [0, 1] with border = 0."""
    c = np.linspace(0.0, 1.0, res)
    U, V = np.meshgrid(c, c)            # U along columns (x), V along rows (y)
    e = np.zeros((res, res))

    if theme == "grass":
        e += 0.30 * np.sin(5.0 * U * np.pi) * np.cos(4.0 * V * np.pi)
        e += 0.18 * np.sin(8.0 * (U + V) * np.pi)
        e += 0.10 * np.sin(11.0 * U * np.pi + 0.7) * np.cos(9.0 * V * np.pi)
        e += 0.12 * (rng.random((res, res)) - 0.5) * 2.0   # rough turf

    elif theme == "sand":
        e += 0.42 * np.sin(2.5 * U * np.pi) * np.cos(2.5 * V * np.pi)   # broad dunes
        e += 0.18 * np.sin(4.0 * (U + V) * np.pi)
        e += 0.05 * (rng.random((res, res)) - 0.5) * 2.0

    elif theme == "steps":
        e += 0.10 * np.sin(3.0 * U * np.pi) * np.cos(3.0 * V * np.pi)
        # staircase (5 rising steps)
        sx0, sx1, sy0, sy1 = 0.20, 0.55, 0.25, 0.70
        m = (U >= sx0) & (U <= sx1) & (V >= sy0) & (V <= sy1)
        t = np.clip((U - sx0) / (sx1 - sx0), 0.0, 1.0)
        e[m] = 0.25 + 0.55 * (np.floor(t[m] * 5.0) / 5.0)
        # ramp
        rx0, rx1, ry0, ry1 = 0.62, 0.92, 0.30, 0.65
        mr = (U >= rx0) & (U <= rx1) & (V >= ry0) & (V <= ry1)
        e[mr] = 0.20 + 0.65 * np.clip((U[mr] - rx0) / (rx1 - rx0), 0.0, 1.0)
        # ravine / ditch: a narrow valley (bottom at 0) flanked by raised ground
        dy = np.abs(V - 0.12)
        e[(dy < 0.05) & (U > 0.1) & (U < 0.9)] = 0.0
        near = (dy < 0.18) & (dy >= 0.05)
        e[near] = np.maximum(e[near], 0.55)

    elif theme == "rubble":
        # rough high-frequency rubble terrain filling most of the tile
        X = U * 50.0
        Y = V * 50.0
        m = _feather(X, 2, 48) * _feather(Y, 2, 48)
        rough = (0.30 * np.sin(26.0 * U * np.pi) * np.sin(26.0 * V * np.pi)
                 + 0.25 * np.sin(44.0 * (U + V) * np.pi)
                 + 0.20 * (rng.random((res, res)) - 0.5) * 2.0)
        rl = rough[m > 0]
        if rl.size:
            nrm = (rough - rl.min()) / (rl.max() - rl.min() + 1e-6)
            e += (0.25 * nrm) * m     # raw 0..0.25 -> scaled by hfield z_top

    elif theme == "ramps":
        # Long 10/20/30 deg ramps: up slope (run d_rise) -> 1 m flat -> down slope,
        # repeated. Two consecutive humps fit exactly in the band [8,39] m; the same
        # hump repeats tile-to-tile along the central '+' band, so the robot meets a
        # long sequence of consecutive ramps. The slope is set per tile (tile_z /
        # ramp_angle_deg) and the per-tile z_top is H = tan(angle)*d_rise, so here
        # Z == H and slope = H/d_rise = tan(angle) stays exact after normalization.
        X = U * 50.0
        Y = V * 50.0
        d_rise = RAMP_D_RISE             # horizontal run of one ramp (m)
        H = Z                            # hump height (m) == per-tile z_top
        s = H / d_rise                   # == tan(angle) -> exact slope (10/20/30)
        plat = 1.0                       # 1 m flat platform at the crest
        valley = 1.0                     # 1 m flat bottom between humps
        period = 2.0 * d_rise + plat + valley   # 17 m -> two humps per tile
        lo_ramp, hi_ramp = 8.0, 39.0     # band ends on a valley (h~0) -> seamless
        m = _feather(X, 5.0, 45.0)             # feather completes at X=8 and X=42
        in_band = (X >= lo_ramp) & (X <= hi_ramp)
        p = (X - lo_ramp) % period
        h = np.select(
            [p < d_rise,
             p < d_rise + plat,
             p < 2.0 * d_rise + plat],
            [s * p, H, H - s * (p - d_rise - plat)],
            default=0.0)
        h = np.where(in_band, h, 0.0)
        # (h / H) anchors the normalization so the final peak == Z (== H) and the
        # slope stays exactly tan(angle) = s.
        e += (h / H) * m

    # normalize to [0, 1]
    e = (e - e.min()) / (e.max() - e.min())
    # force a border ring to 0 -> flat shared edge (seamless between tiles)
    bw = max(2, int(res * 0.05))
    e[:bw, :] = 0.0
    e[-bw:, :] = 0.0
    e[:, :bw] = 0.0
    e[:, -bw:] = 0.0
    e = e / e.max()                      # keep border 0, rescale interior to [0, 1]
    return e.astype("<f4")


def main():
    rng = np.random.default_rng(SEED)
    os.makedirs(ASSETS, exist_ok=True)
    assets_lines = []
    bodies_lines = []
    for iy in range(NY):
        for ix in range(NX):
            cx = ix * TILE_M - (NX * TILE_M / 2.0 - HALF)   # = ix*50 - 225
            cy = iy * TILE_M - (NY * TILE_M / 2.0 - HALF)
            theme = theme_for(ix, iy)
            Z = tile_z(ix, iy)
            h = make_tile(theme, RES, rng, Z)
            name = f"terrain_{ix}_{iy}"
            out = os.path.join(ASSETS, name + ".bin")
            with open(out, "wb") as f:
                f.write(struct.pack("<ii", h.shape[0], h.shape[1]))
                f.write(h.tobytes(order="C"))
            print(f"wrote {out}  [{theme:7s}] grid {h.shape[0]}x{h.shape[1]} "
                  f"h[min={h.min():.3f}, max={h.max():.3f}] center=({cx:.0f},{cy:.0f})")

            assets_lines.append(
                f'    <hfield name="{name}" file="{name}.bin" '
                f'size="25 25 {Z:.2f} 1.0"/>')
            fr = THEME_FRICTION[theme]
            bodies_lines.append(
                f'    <geom name="tile_{ix}_{iy}" type="hfield" hfield="{name}" '
                f'pos="{cx:.0f} {cy:.0f} 0" friction="{fr} 0.05 0.02" '
                f'material="mat_{theme}"/>')

    with open(os.path.join(ASSETS, "terrain_assets.frag"), "w") as f:
        f.write("\n".join(assets_lines) + "\n")
    with open(os.path.join(ASSETS, "terrain_bodies.frag"), "w") as f:
        f.write("\n".join(bodies_lines) + "\n")
    print(f"done. {NX*NY} tiles. Fragments written to {ASSETS}")
    print("Then inject terrain_assets.frag / terrain_bodies.frag into scene.xml "
          "(between the TILE_HFIELDS / TILE_GEOMS markers).")


if __name__ == "__main__":
    main()
