#!/usr/bin/env python3
"""Print the geometry a PORT_NETWORK_1 plan declares.

The backend derives frame/canvas from the first pass's push block (`geometry_of()`:
push[16],push[17] = canvas, and the ioport frame words carry the frame). This reads the
same words straight out of the plan, so a plan can be checked without a build or a device.

    python tools/plan_geom.py local/vk102/manifest/ladder/*.txt
"""
from __future__ import annotations

import sys
from pathlib import Path


def read_plan(path: Path):
    tokens: list[str] = []
    text = path.read_text(encoding="utf-8", errors="replace")
    # Quoted paths may contain backslashes; only the numbers matter here, so split on
    # whitespace after dropping the quoted spans.
    import re

    text = re.sub(r'"[^"]*"', ' Q ', text)
    tokens = text.split()
    i = 0
    nbuf = int(tokens[3])
    npss = int(tokens[4])
    i = 5
    while tokens[i] != "PASS":
        i += 1
    # PASS row Q Q sg shared nbind npush gx gy gz
    i += 1
    row = int(tokens[i]); i += 3
    sg, shared, nbind, npush, gx, gy, gz = (int(tokens[i + k]) for k in range(7))
    i += 7
    for _ in range(nbind):
        i += 5  # BIND b r off bytes  -> BIND then 4 numbers
    assert tokens[i] == "PUSH", tokens[i]
    i += 1
    push = [int(tokens[i + k]) for k in range(npush)]
    return dict(row=row, nbuf=nbuf, npss=npss, sg=sg, shared=shared,
                nbind=nbind, npush=npush, grid=(gx, gy, gz), push=push)


def main(argv: list[str]) -> int:
    for arg in argv:
        p = Path(arg)
        try:
            g = read_plan(p)
        except Exception as exc:  # noqa: BLE001 - a diagnostic tool reports, it does not raise
            print(f"{p.name:34} ERROR {exc}")
            continue
        push = g["push"]
        canvas = (push[16], push[17]) if len(push) > 17 else ("?", "?")
        # The ioport frame words: block70_2 carries the frame at w2/w3 and w20/w21.
        frame_words = {k: push[k] for k in (2, 3, 20, 21) if k < len(push)}
        print(f"{p.name:34} canvas={canvas[0]}x{canvas[1]} grid0={g['grid'][0]}x{g['grid'][1]} "
              f"push[2,3,20,21]={list(frame_words.values())} passes={g['npss']}")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
