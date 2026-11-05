#!/usr/bin/env python3
"""Quality budget: what a candidate costs in dB, and what it buys in ms.

`nr_gate.py` answers "is this change bit-exact" with a yes/no. That is the right gate for
a refactor, but it cannot price a change that is *allowed* to move the picture (Winograd,
a cheaper activation, half-rate reuse). This tool produces that price list: for each of
the six gate inputs it reports max|d|, mean|d| and PSNR against v1, plus the per-family
GPU time of the same run, so "0.4 ms for 6 dB" becomes a decision instead of a guess.

Reference: the frozen v1 snapshot (`local/small/versions/v1-baseline`), whose manifest
points at `../spv` next to it, so a copy is written with that prefix rewritten.

Usage
  python tools/nr_budget.py --plan local/small/manifest/port_network_runtime.txt
  python tools/nr_budget.py --plan <variant.txt> --pairs 2 --json out.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNNER = os.path.join(ROOT, "build", "vulkan-small", "port_network_model_runner.exe")
GATE_IN = os.path.join(ROOT, "local", "small", "gate", "in")
V1 = os.path.join(ROOT, "local", "small", "versions", "v1-baseline")
INPUTS = ("base", "shift", "flip", "hdr", "dark", "noise")
W, H = 1280, 720


def v1_plan() -> str:
    """v1's manifest, repointed: spv to the snapshot beside it, cold assets to the live tree.

    The snapshot keeps only the shaders it froze; the weight/aux blobs are unchanged by any
    shader-level candidate, so they are read from local/small/assets/cold by absolute path.
    """
    src = os.path.join(V1, "manifest", "port_network_runtime.txt")
    dst = os.path.join(V1, "manifest", "tmp-budget-ref.txt")
    assets = os.path.join(ROOT, "local", "small", "assets").replace("\\", "/")
    text = open(src, encoding="utf-8").read()
    text = text.replace('"../assets/spv/', '"../spv/')
    text = text.replace('"../assets/cold/', '"%s/cold/' % assets)
    open(dst, "w", encoding="utf-8").write(text)
    return dst


def run(plan: str, in_path: str):
    out = os.path.join(tempfile.mkdtemp(prefix="nrb-"), "run")
    try:
        p = subprocess.run([RUNNER, "--plan", plan, "--out", out, "--input", f"{W}x{H}:{in_path}",
                            "--warm", "1", "--reps", "1", "--per-pass-timing"],
                           capture_output=True, text=True, cwd=ROOT)
        if p.returncode != 0:
            raise SystemExit("runner failed on %s\n%s" % (os.path.basename(in_path), p.stderr[-1500:]))
        frame = os.path.join(out, "frame_out.bin")
        if not os.path.exists(frame):
            raise SystemExit("no frame_out.bin for " + os.path.basename(in_path))
        # `PASSMS_TOTAL <sum> <count>` also starts with PASSMS; taking split()[1] as a name
        # made it a "pass" called 24.49 worth 108 ms and inflated every family sum.
        per = {}
        for line in p.stdout.splitlines():
            w = line.split()
            if len(w) == 7 and w[0] == "PASSMS":
                per[w[1]] = float(w[2])
        gpu = 0.0
        for l in p.stdout.splitlines():
            if l.startswith("network sample"):
                gpu = float(l.split("gpu_ms=")[1])
        return np.fromfile(frame, dtype=np.float32), per, gpu
    finally:
        shutil.rmtree(os.path.dirname(out), ignore_errors=True)


def score(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    d = a.astype(np.float64) - b.astype(np.float64)
    ad = np.abs(d)
    mse = float(np.mean(d * d))
    psnr = float("inf") if mse == 0 else 10.0 * np.log10(1.0 / mse)
    return float(ad.max()), float(ad.mean()), psnr


def families(per: dict[str, float]) -> dict[str, float]:
    out = {"viT": 0.0, "full-res": 0.0, "other": 0.0}
    for name, ms in per.items():
        if "vit_" in name:
            out["viT"] += ms
        elif name.startswith(("r00_", "r73_", "r74_", "r75_s37")):
            out["full-res"] += ms
        else:
            out["other"] += ms
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True, help="variant plan to price")
    ap.add_argument("--pairs", type=int, default=1, help="repeat and keep the fastest run per input")
    ap.add_argument("--json", default="", help="also write the table as JSON")
    a = ap.parse_args()

    ref_plan = v1_plan()
    rows = []
    try:
        for name in INPUTS:
            inp = os.path.join(GATE_IN, name + ".f32")
            if not os.path.exists(inp):
                raise SystemExit("missing gate input " + inp + " (run nr_gate.py record first)")
            best_ref = best_var = None
            per_ref, per_var = [], []
            for _ in range(max(1, a.pairs)):
                fr, pr, gr = run(ref_plan, inp)
                fv, pv, gv = run(a.plan, inp)
                per_ref.append(pr); per_var.append(pv)
                if best_ref is None or gr < best_ref[1]:
                    best_ref = (fr, gr)
                if best_var is None or gv < best_var[1]:
                    best_var = (fv, gv)
            # PASSMS occasionally reports a spike around a pre-empted dispatch (a single pass
            # claiming tens of ms while the frame stays at 23), so price the families from the
            # per-pass minimum across pairs, not from one run.
            def permin(ps):
                out = {}
                for d in ps:
                    for k, v in d.items():
                        out[k] = min(out.get(k, 1e9), v)
                return out
            mr, mv = permin(per_ref), permin(per_var)
            mx, mn, psnr = score(best_var[0], best_ref[0])
            rows.append({"input": name, "max": mx, "mean": mn, "psnr": psnr,
                         "ms_ref": best_ref[1], "ms_var": best_var[1],
                         "fam_ref": families(mr), "fam_var": families(mv)})
            print("%-6s max|d|=%-10.3g mean|d|=%-10.3g psnr=%s   %.2f -> %.2f ms"
                  % (name, mx, mn, "inf" if psnr == float("inf") else "%.1f dB" % psnr,
                     best_ref[1], best_var[1]))
    finally:
        p = os.path.join(V1, "manifest", "tmp-budget-ref.txt")
        if os.path.exists(p):
            os.remove(p)

    worst = min(r["psnr"] for r in rows)
    print("\nworst-case psnr: %s   (max|d| %.3g)"
          % ("inf" if worst == float("inf") else "%.1f dB" % worst, max(r["max"] for r in rows)))
    f0 = {k: sum(r["fam_ref"][k] for r in rows) / len(rows) for k in rows[0]["fam_ref"]}
    f1 = {k: sum(r["fam_var"][k] for r in rows) / len(rows) for k in rows[0]["fam_var"]}
    for k in f0:
        print("%-10s %7.2f -> %7.2f ms  (%+.2f)" % (k, f0[k], f1[k], f1[k] - f0[k]))
    if a.json:
        json.dump({"plan": a.plan, "rows": rows, "worst_psnr": worst,
                   "families_ref": f0, "families_var": f1},
                  open(a.json, "w", encoding="utf-8"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
