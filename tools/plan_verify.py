#!/usr/bin/env python3
"""Run gen_plan's own invariants over a plan file without touching any model directory.

freeze_plan.py calls `gen_plan.verify()` before pinning; this is the same check, callable on a
plan that is not (yet) the model's runtime plan. Reports the frame and canvas the plan declares
so a rewrite can be judged before it is installed.

    python tools/plan_verify.py plan.txt
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def main(argv: list[str]) -> int:
    gp = load("gen_plan_mod", "gen_plan.py")
    fp = load("freeze_plan_mod", "freeze_plan.py")
    rc = 0
    targets: list[Path] = []
    for arg in argv:
        p = Path(arg)
        targets.extend(sorted(p.parent.glob(p.name)) if any(c in arg for c in "*?[") else [p])
    for path in targets:
        text = path.read_text(encoding="utf-8")
        # Same two words freeze_plan reads: PUSH w17/w18 = canvas, and port_frame() = frame.
        frame = fp.port_frame(text)
        canvas = None
        for line in text.splitlines():
            if line.startswith("PUSH "):
                w = line.split()
                if len(w) > 18:
                    canvas = (int(w[17]), int(w[18]))
                break
        problems = gp.verify(frame[0], frame[1], text)
        status = "OK" if not problems else f"{len(problems)} problem(s)"
        print(f"{path.name:34} frame={frame[0]}x{frame[1]} canvas={canvas} verify={status}")
        for p in problems[:8]:
            print(f"    {p}")
        if problems:
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
