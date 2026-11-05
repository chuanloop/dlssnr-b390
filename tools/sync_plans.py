"""Keep every plan pointed at the same shader modules as the production plan.

The profile carries two plans that must agree on *modules* while differing on
geometry: `port_network_runtime.txt` is the shipped 1024x576 operating point,
`port_network_runtime_1280x720.txt` is the 720p copy the measurement harness runs.

They can drift apart. `recompile_shader.py --also-plan` updates both, but a plan can
also be edited on its own, and the 720p plan sat on an f08 module (`92cd6aad`) that
no longer existed in `recipes.json` -- so every measurement until 2026-11-01 ran a
shader production never had.

Alignment is by PASS index, not by tag: both plans carry the same 232 passes in the
same order, and one tag can hold several modules (f12 has a KIND=2 and a KIND=3
variant), so the Nth pass of the aux plan must use the Nth pass's module from the
production plan. Geometry, push constants and bindings are untouched.

    python tools/sync_plans.py --check     # report only, exit 1 on drift
    python tools/sync_plans.py             # rewrite the aux plans
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "local/vk102"
PROD = MODEL / "manifest/port_network_runtime.txt"
AUX = [MODEL / "manifest/port_network_runtime_1280x720.txt"]

REF = re.compile(r'"\.\./assets/spv/([0-9a-f]{64})__([a-z0-9]+-[0-9a-f]+)\.spv"')


def prod_pass_lines() -> list[str]:
    return [l for l in PROD.read_text(encoding="utf-8").splitlines() if l.startswith("PASS ")]


def module_of(line: str) -> str | None:
    m = REF.search(line)
    return m.group(2) if m else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report drift and exit 1 without writing")
    a = ap.parse_args()

    prod = prod_pass_lines()
    prod_mods = [module_of(l) for l in prod]
    drift = 0

    for aux in AUX:
        if not aux.is_file():
            print(f"{aux.name}: absent, skipped")
            continue
        lines = aux.read_text(encoding="utf-8").splitlines(keepends=True)
        idx = -1
        changed = []
        for i, line in enumerate(lines):
            if not line.startswith("PASS "):
                continue
            idx += 1
            if idx >= len(prod) or prod_mods[idx] is None:
                continue
            have = module_of(line)
            if have == prod_mods[idx]:
                continue
            drift += 1
            changed.append((idx, have, prod_mods[idx]))
            if a.check:
                continue
            lines[i] = line.replace(REF.search(line).group(0), REF.search(prod[idx]).group(0))
        if changed and not a.check:
            aux.write_text("".join(lines), encoding="utf-8", newline="")
        print(f"{aux.name}: {'drift' if changed else 'in sync'}")
        for i, have, want in changed:
            print(f"   pass {i:<3} {have} -> {want}")

    if a.check and drift:
        print(f"\n{drift} pass(es) diverge; run without --check to align")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
