#!/usr/bin/env python3
"""Generate the working-scale plan menu: model extent = round(render * scale).

AMD's rule (upstream-host/windows/src/core/nr_renodx.cpp:1108) has no alignment, so every entry
here is generated with --no-pad: the model's canvas *is* the model's extent. The backend picks an
entry by matching that exact extent, which is why this only has to enumerate the sizes a given
render size can ask for, not every (render, scale) pair in the abstract.

    python tools/gen_ws_plans.py [--check]
    python tools/gen_ws_plans.py --render 1706x960 --scales 0.5,0.67,0.8 --max 1728x960 \
        --src local/scratch/src-1440p.txt --ladder local/scratch/ladder-1440p

With no options this reproduces the shipped menu (1080p/1200p tiers at scales 0.50-0.90).
`--check` reports which (render, scale) pairs the ladder cannot serve.

`--render` makes the tool usable for any resolution the game actually hands us: pass the render
extent from the upscaler, the scale list you want to expose, and `--max` = the canvas the package's
buffers can hold (grow them with tools/resize_plan_budget.py, then point `--src` at the grown plan).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LADDER = ROOT / "local/small/manifest/ladder"
GEN = ROOT / "tools/gen_plan_small.py"
INSTALLED = ROOT / "local/small/manifest/port_network_runtime.txt"

# Render sizes the game actually hands us (DLSS/FSR/XeSS tiers at 1080p and 1200p).
RENDERS = [(640, 360), (640, 400), (960, 540), (960, 600), (1129, 635), (1129, 705),
           (1130, 636), (1130, 706), (1280, 720)]
# Only scales below 1.0: at 1.0 the working extent equals the render extent, which the *installed*
# padded plan already serves, and two plans with the same frame would make the lookup ambiguous.
FULL_MENU = {(960, 600), (1280, 720)}
SCALES = [0.50, 0.55, 0.60, 0.67, 0.70, 0.75, 0.80, 0.90]
# The installed plan's canvas is the buffer budget: a model larger than this cannot be allocated.
MAX_W, MAX_H = 1344, 768


def name(scale: float, w: int, h: int) -> str:
    return "plan-s%.2f-%dx%d.txt" % (scale, w, h)


def wanted(renders, scales, full_menu, max_w, max_h) -> dict[tuple[int, int], float]:
    """(width, height) -> the scale label to write into the table."""
    out: dict[tuple[int, int], float] = {}
    for rw, rh in renders:
        for s in (scales if (rw, rh) in full_menu else []):
            w, h = int(round(rw * s)), int(round(rh * s))
            if w > max_w or h > max_h or w < 64 or h < 64:
                continue
            # A size reachable from two (render, scale) pairs is generated once: the table is
            # searched by extent, so the label is documentation. First pairing wins, so the name
            # is deterministic across runs.
            out.setdefault((w, h), s)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="report missing extents only")
    ap.add_argument("--render", action="append", default=[],
                    help="render extent the game hands us, WxH; repeatable (default: the shipped menu)")
    ap.add_argument("--scales", help="comma-separated scales (default: %s)"
                                   % ",".join("%.2f" % s for s in SCALES))
    ap.add_argument("--max", help="canvas budget WxH (default: the installed plan's canvas %dx%d)"
                                 % (MAX_W, MAX_H))
    ap.add_argument("--ladder", default=str(LADDER), help="output directory")
    ap.add_argument("--force", action="store_true",
                    help="rewrite plans that already exist (needed when the buffer budget changed)")
    ap.add_argument("--src", default=None,
                    help="source plan whose BUFFER block becomes the budget (default: installed plan)")
    a = ap.parse_args()

    if a.render:
        renders = []
        for spec in a.render:
            w, h = (int(x) for x in spec.lower().split("x"))
            renders.append((w, h))
        # An explicit render list means "this resolution gets the whole menu".
        full_menu = set(renders)
    else:
        renders, full_menu = RENDERS, FULL_MENU
    scales = ([float(x) for x in a.scales.split(",")] if a.scales else SCALES)
    max_w, max_h = ((int(x) for x in a.max.lower().split("x")) if a.max else (MAX_W, MAX_H))

    ladder = Path(a.ladder)
    ladder.mkdir(parents=True, exist_ok=True)
    have = {p.name for p in ladder.glob("plan-*.txt")}
    todo = []
    for (w, h), s in sorted(wanted(renders, scales, full_menu, max_w, max_h).items()):
        n = name(s, w, h)
        if a.force or n not in have:
            todo.append((n, w, h))
    if a.check:
        print("ladder has %d plan(s); %d extent(s) missing" % (len(have), len(todo)))
        for n, w, h in todo:
            print("  missing %-28s (%dx%d)" % (n, w, h))
        return 0 if not todo else 1
    print("generating %d plan(s) into %s" % (len(todo), ladder))
    for n, w, h in todo:
        cmd = [sys.executable, str(GEN), "--size", "%dx%d" % (w, h), "--no-pad",
               "--out", str(ladder / n), "--canvas-max", "%dx%d" % (max_w, max_h)]
        if a.src:
            cmd += ["--src", a.src]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print("  FAIL %-28s %s" % (n, (r.stdout + r.stderr).strip()[-160:]))
            return 1
        print("  %-28s %dx%d" % (n, w, h))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
