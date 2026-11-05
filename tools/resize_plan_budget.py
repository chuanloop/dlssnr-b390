#!/usr/bin/env python3
"""Re-size the `-` (runtime-only) buffers of a plan, leaving cold-file buffers untouched.

The plan's BUFFER block is a *budget*: every ladder plan in the small family declares the same
one, sized for the largest canvas the package supports, and the backend allocates from it.  A
larger working extent therefore needs a larger budget, and nothing else -- buffer capacity does
not enter any arithmetic (verified: a plan whose buffers are 2x oversized produces a bit-identical
frame_out).  The push fields that *do* encode geometry (strides, plane offsets) are rewritten by
`gen_plan_small.py`, not here.

    python tools/resize_plan_budget.py --plan P --factor 1.6 --out Q
    python tools/resize_plan_budget.py --plan P --factor 1.6 --out Q --round 4096

Buffers whose path is a cold `.bin` (weights, constants) are never touched: their size is the
file's size, and growing the allocation would only break the hash-pinned contract.
"""
from __future__ import annotations

import argparse
import io
import sys

ALIGN = 4096


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--factor", type=float, required=True, help="multiply every runtime buffer by this")
    ap.add_argument("--align", type=int, default=ALIGN, help="round the new size up to this many bytes")
    a = ap.parse_args()
    if a.factor <= 0.0:
        print("factor must be positive", file=sys.stderr)
        return 2
    grown = 0
    before = after = 0
    out: list[str] = []
    for line in io.open(a.plan, encoding="utf-8"):
        f = line.rstrip("\n").split(" ")
        if line.startswith("BUFFER ") and len(f) >= 4 and f[3] == '"-"':
            old = int(f[2])
            new = ((int(old * a.factor) + a.align - 1) // a.align) * a.align
            if new < old:                       # never shrink: an existing plan stays valid
                new = old
            if new != old:
                f[2] = str(new)
                line = " ".join(f) + "\n"
                grown += 1
            before += old
            after += new
        out.append(line)
    io.open(a.out, "w", encoding="utf-8", newline="").write("".join(out))
    print("resized %d buffer(s), runtime footprint %.1f -> %.1f MB  (x%.3f)"
          % (grown, before / 1e6, after / 1e6, after / before if before else 0.0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
