#!/usr/bin/env python3
"""Compare each plan pass's declared workgroup-memory size with the SPIR-V it points at.

The plan carries a `shared` word per pass and the runner passes it through untouched, so a
stale word is invisible at runtime: the real size comes from the module. That silence is the
problem - if the plan and the module disagree, at least one of them was edited without the
other, and the tree can no longer reproduce the shipped binary.

`f20-215aca679aaa` is the founding case: its source comment says 25088 B, its three shared
arrays add up to 24576 B, and the plan says 29184 B.

Usage:  python tools/audit_plan_shared.py <plan.txt> [--spirv-dis PATH] [--json OUT]
Exit code 0 when every size matches, 1 otherwise.
"""

import json
import os
import re
import subprocess
import sys
from collections import OrderedDict

DEFAULT_DIS = r"C:\VulkanSDK\1.4.363.0\Bin\spirv-dis.exe"

# One Workgroup variable: %name = OpVariable %_ptr_Workgroup__arr_half_uint_2048 Workgroup
# The element type is encoded in the pointer type name, which every glslang build emits.
WORKGROUP_VAR = re.compile(
    r"OpVariable\s+%_ptr_Workgroup__arr_(half|float|uint)_uint_(\d+)\s+Workgroup"
)
SCALAR_VAR = re.compile(
    r"OpVariable\s+%_ptr_Workgroup_(half|float|uint)\s+Workgroup"
)
ELEM_BYTES = {"half": 2, "float": 4, "uint": 4}


def shared_bytes(spv_path, dis):
    text = subprocess.run(
        [dis, spv_path, "--no-color"],
        capture_output=True, text=True, check=False,
    ).stdout
    if not text:
        return None
    total = 0
    for kind, count in WORKGROUP_VAR.findall(text):
        total += ELEM_BYTES[kind] * int(count)
    for kind in SCALAR_VAR.findall(text):
        total += ELEM_BYTES[kind]
    return total


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    plan_path = argv[1]
    dis = DEFAULT_DIS
    if "--spirv-dis" in argv:
        dis = argv[argv.index("--spirv-dis") + 1]
    out_json = None
    if "--json" in argv:
        out_json = argv[argv.index("--json") + 1]

    plan_dir = os.path.dirname(os.path.abspath(plan_path))
    passes = []  # (row, name, spv, declared)
    for line in open(plan_path, encoding="utf-8"):
        if not line.startswith("PASS "):
            continue
        words = line.split()
        # PASS <row> "<name>" "<spv>" <subgroup> <shared> <rows> <cols> <gx> <gy> <gz>
        row, name, spv = words[1], words[2].strip('"'), words[3].strip('"')
        declared = int(words[5])
        passes.append((int(row), name, spv, declared))

    cache = {}
    mismatches = []
    # Keyed by "<row>|<name>", not by row: the 152-row plan expands to 232 dispatches, so a
    # row alone is ambiguous and a row-keyed map silently keeps whichever pass came last.
    by_pass = {}
    for row, name, spv, declared in passes:
        path = os.path.normpath(os.path.join(plan_dir, spv))
        if path not in cache:
            cache[path] = shared_bytes(path, dis) if os.path.exists(path) else None
        actual = cache[path]
        by_pass[f"{row}|{name}"] = actual
        if actual is None:
            mismatches.append((row, name, declared, None, spv))
        elif actual != declared:
            mismatches.append((row, name, declared, actual, spv))

    if out_json:
        with open(out_json, "w", encoding="utf-8") as fh:
            json.dump({
                "plan": plan_path,
                "passes": by_pass,
                "rows": {str(k): v for k, v in sorted(by_pass.items())},
                "mismatches": [
                    {"row": r, "name": n, "declared": d, "module": a, "spv": s}
                    for r, n, d, a, s in mismatches
                ],
            }, fh, indent=2)
            fh.write("\n")

    print(f"plan: {plan_path}")
    print(f"passes: {len(passes)}  distinct modules: {len(cache)}")
    if not mismatches:
        print("every declared workgroup size matches its module")
        return 0
    print(f"mismatches: {len(mismatches)}")
    print(f"{'row':>5}  {'declared':>9}{'module':>9}{'delta':>8}  name")
    for row, name, declared, actual, spv in mismatches:
        delta = "" if actual is None else f"{actual - declared:+d}"
        shown = "missing" if actual is None else str(actual)
        print(f"{row:>5}  {declared:>9}{shown:>9}{delta:>8}  {name}")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
