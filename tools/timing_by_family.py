#!/usr/bin/env python3
"""Aggregate `PASSMS` lines from port_network_model_runner into families.

The runner emits one line per pass:

    PASSMS <name> <mean_ms> <gx> <gy> <shared_bytes> <plan_row>

Aggregating by pass name alone is misleading: `block1_swin_body_1` (0.49 ms) and
`block5_swin_body_1` (0.14 ms) are both "swin_body", and the first four blocks run a
single dispatch where the rest run three. This tool groups by *shape family* and keeps
the grid next to the time so an outlier can be explained by its workgroup count instead
of guessed at.

Usage:  python tools/timing_by_family.py <passms.txt> [--json OUT] [--shared-map AUDIT.json]

`--shared-map` takes the output of `tools/audit_plan_shared.py --json`. Without it the
`shared` column is the plan's word, which is inherited from the upstream 152-row plan and
disagrees with the shipped module on 18 of 232 rows; the occupancy column is only
trustworthy with the module's own number.
"""

import json
import re
import sys
import statistics as st
from collections import OrderedDict

# Order matters: first match wins. Every rule is anchored, so a new pass name that
# matches nothing lands in its own family instead of silently joining an existing one.
RULES = [
    (re.compile(r"^block\d+_swin_body_\d+$"), "swin_body"),
    (re.compile(r"^split16_b\d+_l\d+$"), "split16"),
    (re.compile(r"^vit_block\d+_ffn_expand$"), "vit_ffn_expand"),
    (re.compile(r"^vit_block\d+_ffn_contract$"), "vit_ffn_contract"),
    (re.compile(r"^vit_block\d+_qkv$"), "vit_qkv"),
    (re.compile(r"^vit_block\d+_attention$"), "vit_attention"),
    (re.compile(r"^vit_block\d+_projection$"), "vit_projection"),
    (re.compile(r"^c3x_pre_c\d+$"), "c3x_pre"),
    (re.compile(r"^c3x_body_c\d+$"), "c3x_body"),
    (re.compile(r"^f3c\d+_body_ds$"), "f3c_ds"),
    (re.compile(r"^ioport_block\d+$"), "ioport"),
]


def family_of(name):
    for pattern, family in RULES:
        if pattern.match(name):
            return family
    return name


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    path = argv[1]
    out_json = None
    if "--json" in argv:
        out_json = argv[argv.index("--json") + 1]
    shared_map = {}
    if "--shared-map" in argv:
        audit = json.load(open(argv[argv.index("--shared-map") + 1], encoding="utf-8"))
        shared_map = {k: v for k, v in audit["passes"].items() if v is not None}

    families = OrderedDict()
    total = None
    count = 0
    for line in open(path, encoding="utf-8"):
        words = line.split()
        if not words or words[0] != "PASSMS":
            continue
        if words[1] == "PASSMS_TOTAL":
            total = float(words[2])
            count = int(words[3])
            continue
        name, ms = words[1], float(words[2])
        gx = int(words[3]) if len(words) > 3 else 0
        gy = int(words[4]) if len(words) > 4 else 0
        shared = int(words[5]) if len(words) > 5 else 0
        row = int(words[6]) if len(words) > 6 else -1
        if f"{row}|{name}" in shared_map:
            shared = shared_map[f"{row}|{name}"]
        entry = families.setdefault(family_of(name), {
            "passes": 0, "ms": 0.0, "grids": [], "shared": [], "names": [],
        })
        entry["passes"] += 1
        entry["ms"] += ms
        entry["grids"].append(gx * gy)
        entry["shared"].append(shared)
        entry["names"].append((ms, name))

    if total is None:
        total = sum(e["ms"] for e in families.values())
    shown = total

    rows = sorted(families.items(), key=lambda kv: -kv[1]["ms"])
    header = f"{'family':<22}{'passes':>7}{'ms':>9}{'share':>8}{'per pass':>10}{'wg (med)':>10}{'shared':>8}"
    print(header)
    print("-" * len(header))
    for name, e in rows:
        print(f"{name:<22}{e['passes']:>7}{e['ms']:>9.3f}{e['ms'] / shown * 100:>7.1f}%"
              f"{e['ms'] / e['passes']:>10.3f}"
              f"{int(st.median(e['grids'])):>10}{max(e['shared']):>8}")
    print("-" * len(header))
    print(f"{'TOTAL':<22}{count:>7}{total:>9.3f}")

    print("\nworst single pass per family")
    for name, e in rows:
        worst = max(e["names"])
        print(f"  {name:<22}{worst[0]:>8.3f}  {worst[1]}")

    if out_json:
        payload = {
            "total_ms": total,
            "passes": count,
            "families": {
                name: {
                    "passes": e["passes"],
                    "ms": round(e["ms"], 4),
                    "share": round(e["ms"] / shown, 6),
                    "per_pass_ms": round(e["ms"] / e["passes"], 4),
                    "median_workgroups": int(st.median(e["grids"])),
                    "max_shared_bytes": max(e["shared"]),
                }
                for name, e in rows
            },
        }
        with open(out_json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
            fh.write("\n")
        print(f"\nwrote {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
