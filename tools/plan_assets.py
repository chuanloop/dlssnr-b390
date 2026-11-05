#!/usr/bin/env python3
"""Check that every asset a PORT_NETWORK_1 plan names actually exists under an assets dir.

A plan is bound to a specific kernel build: its BUFFER paths and PASS SPIR-V names are content
hashes. Installing a plan whose hashes are missing in the asset tree fails at device init, so
this reports the miss list before anything is deployed.

    python tools/plan_assets.py plan-1024x576.txt local/vk102/assets
"""
from __future__ import annotations

import sys
from pathlib import Path


def paths_in_plan(plan: Path) -> list[str]:
    out: list[str] = []
    base = plan.parent
    for line in plan.read_text(encoding="utf-8", errors="replace").splitlines():
        head = line.split(" ", 1)[0]
        if head not in ("BUFFER", "PASS"):
            continue
        quoted = [p for p in line.split('"')[1::2]]
        # BUFFER "name" bytes "path"; PASS row "name" "spv"
        for token in quoted[1:] if head == "BUFFER" else quoted[1:]:
            if token in ("-", "") or token.endswith(".spv") or "/" in token or "\\" in token:
                out.append(token)
    return out


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    plan = Path(argv[0])
    assets = Path(argv[1])
    refs = paths_in_plan(plan)
    missing = []
    for ref in refs:
        if ref == "-":
            continue
        if (plan.parent / ref).exists():
            continue
        # BUFFER paths are relative to the plan; SPIR-V names live under assets/<kind>/
        cand = [assets / ref, assets / "spv" / Path(ref).name, assets / "cold" / Path(ref).name]
        if any(c.exists() for c in cand):
            continue
        missing.append(ref)
    print(f"{plan.name}: {len(refs)} refs, {len(missing)} missing under {assets}")
    for m in missing[:12]:
        print(f"  MISSING {m}")
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
