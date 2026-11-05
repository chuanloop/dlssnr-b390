#!/usr/bin/env python3
"""Make every plan in a ladder share the installed plan's BUFFER block.

`gen_plan_small.py` copies the BUFFER lines out of its `--src` verbatim ("buffers are not
resized"), so a ladder assembled over several batches ends up with several blocks: the entries
generated after a budget change carry one, the older ones another.  The backend reuses the
installed plan's pool aliases when it switches scale, so the family is supposed to be one block --
this rewrites the outliers to match the reference instead of regenerating them (regenerating would
need each entry's padding convention, and getting that wrong breaks the grid silently).

    python tools/normalize_budget.py --model local/small [--dry-run]
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def blocks(path: Path) -> tuple[list[str], list[str]]:
    """(indices, BUFFER lines) of a plan."""
    lines = io.open(path, encoding="utf-8").read().splitlines()
    idx = [i for i, l in enumerate(lines) if l.startswith("BUFFER ")]
    return idx, [lines[i] for i in idx]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="local/small")
    ap.add_argument("--ref", help="plan whose BUFFER block wins (default: the installed plan)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    model = (ROOT / a.model).resolve()
    ref = Path(a.ref) if a.ref else model / "manifest/port_network_runtime.txt"
    idx, want = blocks(ref)
    names = [l.split(" ")[1] for l in want]
    print("reference %s: %d buffer(s)" % (Path(ref).name, len(want)))

    changed = 0
    for plan in sorted((model / "manifest/ladder").glob("plan-s*.txt")):
        lines = io.open(plan, encoding="utf-8").read().splitlines()
        pidx, have = blocks(plan)
        if len(have) != len(want) or [l.split(" ")[1] for l in have] != names:
            print("  SKIP %-30s buffer list differs in shape" % plan.name)
            continue
        if have == want:
            print("  ok   %-30s" % plan.name)
            continue
        for i, l in zip(pidx, want):
            lines[i] = l
        if not a.dry_run:
            io.open(plan, "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
        changed += 1
        print("  %s %-30s" % ("would fix" if a.dry_run else "fixed    ", plan.name))
    print("%d plan(s) %s" % (changed, "need the reference block" if a.dry_run else "rewritten"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
