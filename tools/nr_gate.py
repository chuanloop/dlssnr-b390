#!/usr/bin/env python3
"""Numerical equivalence gate for the Small network.

Why this exists
---------------
On 2026-11-02 a shader swap produced byte-identical output on the one frame we
tested and still looked blurred and block-flickering in game. One frame is not a
proof: a single 1280x720 still happens to sit in the range where two different
code paths agree. The gate therefore runs a fixed set of derived inputs that
stress the parts a single still does not reach:

  base    the tone-mapped still we always used
  shift   circular shift by (13, 7) px -- moves every tile boundary
  flip    horizontal mirror -- reverses token order inside every block
  hdr     x16 -- drives values into the e4m3 clamp region
  dark    x0.05 -- drives values into the denormal / small-magnitude region
  noise   base + high-frequency detail -- worst case for any fusion change

Usage
-----
    # once per version you want to protect
    python tools/nr_gate.py record --plan local/small/manifest/port_network_runtime.txt \
        --ref local/small/versions/v1-baseline/gate

    # after any shader / plan change
    python tools/nr_gate.py check  --plan local/small/manifest/port_network_runtime.txt \
        --ref local/small/versions/v1-baseline/gate

`check` exits 1 on any mismatch. On mismatch pass `--diff` to get max|d| per
input; it re-runs the reference plan into a temp directory and compares the raw
frames, which is slower but tells you *how* wrong the change is, not just that
it is wrong.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNNER = os.path.join(ROOT, "build", "vulkan-small", "port_network_model_runner.exe")
SRC_FRAME = os.path.join(ROOT, "local", "quality", "qual_in.f32")
GATE_IN = os.path.join(ROOT, "local", "small", "gate", "in")

W, H, C = 1280, 720, 4


def _load() -> np.ndarray:
    a = np.fromfile(SRC_FRAME, dtype=np.float32)
    if a.size != W * H * C:
        raise SystemExit(f"{SRC_FRAME}: {a.size} floats, expected {W*H*C}")
    return a.reshape(H, W, C)


def build_inputs() -> dict[str, str]:
    os.makedirs(GATE_IN, exist_ok=True)
    base = _load()
    rng = np.random.default_rng(20261102)
    variants: dict[str, np.ndarray] = {
        "base": base,
        "shift": np.roll(np.roll(base, 13, axis=1), 7, axis=0),
        "flip": base[:, ::-1, :],
        "hdr": base * 16.0,
        "dark": base * 0.05,
        "noise": base + rng.standard_normal(base.shape, dtype=np.float32) * 0.15,
    }
    out = {}
    for name, arr in variants.items():
        path = os.path.join(GATE_IN, f"{name}.f32")
        arr.astype(np.float32).tofile(path)
        out[name] = path
    return out


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(plan: str, in_path: str) -> tuple[str, float]:
    # the runner refuses an existing output directory, so keep the fresh path
    # one level below the temp dir it owns
    out = os.path.join(tempfile.mkdtemp(prefix="nrg-"), "run")
    try:
        cmd = [RUNNER, "--plan", plan, "--out", out,
               "--input", f"{W}x{H}:{in_path}", "--warm", "1", "--reps", "1"]
        p = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
        if p.returncode != 0:
            raise SystemExit(f"runner failed on {os.path.basename(in_path)}\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}")
        frame = os.path.join(out, "frame_out.bin")
        if not os.path.exists(frame):
            raise SystemExit(f"no frame_out.bin for {os.path.basename(in_path)}")
        ms = 0.0
        for line in p.stdout.splitlines():
            if line.startswith("TOTAL_MS") or "total_ms" in line:
                try:
                    ms = float(line.split()[-1])
                except ValueError:
                    pass
        return sha256(frame), ms
    finally:
        shutil.rmtree(os.path.dirname(out), ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["record", "check"])
    ap.add_argument("--plan", required=True)
    ap.add_argument("--ref", required=True, help="directory holding gate.json")
    ap.add_argument("--diff", action="store_true", help="on mismatch, report max|d| (slow)")
    args = ap.parse_args()

    inputs = build_inputs()
    os.makedirs(args.ref, exist_ok=True)
    ref_file = os.path.join(args.ref, "gate.json")

    if args.mode == "record":
        data = {}
        for name, path in inputs.items():
            digest, ms = run(args.plan, path)
            data[name] = {"sha256": digest, "ms": ms}
            print(f"  record {name:6} {digest[:16]}")
        with open(ref_file, "w", encoding="utf-8") as f:
            json.dump({"plan": args.plan, "inputs": data}, f, indent=1)
        print(f"\nrecorded {len(data)} inputs -> {ref_file}")
        return 0

    if not os.path.exists(ref_file):
        raise SystemExit(f"no reference at {ref_file}; run `record` first")
    ref = json.load(open(ref_file, encoding="utf-8"))["inputs"]
    bad = []
    for name, path in inputs.items():
        digest, ms = run(args.plan, path)
        want = ref.get(name, {}).get("sha256")
        ok = digest == want
        print(f"  {'PASS' if ok else 'FAIL'} {name:6} {digest[:16]}"
              + ("" if ok else f"  != {str(want)[:16]}"))
        if not ok:
            bad.append(name)
    if not bad:
        print(f"\nGATE PASS  ({len(inputs)} inputs byte-identical to reference)")
        return 0
    print(f"\nGATE FAIL  {len(bad)}/{len(inputs)} inputs differ: {', '.join(bad)}")
    if args.diff:
        ref_plan = json.load(open(ref_file, encoding="utf-8"))["plan"]
        for name in bad:
            a = _frame_of(ref_plan, inputs[name])
            b = _frame_of(args.plan, inputs[name])
            if a is None or b is None:
                continue
            d = np.abs(b - a)
            print(f"    {name:6} max|d|={d.max():.3e}  mean|d|={d.mean():.3e}  "
                  f"frac>1e-3={(d > 1e-3).mean()*100:.2f}%")
    return 1


def _frame_of(plan: str, in_path: str):
    out = os.path.join(tempfile.mkdtemp(prefix="nrgd-"), "run")
    try:
        subprocess.run([RUNNER, "--plan", plan, "--out", out,
                        "--input", f"{W}x{H}:{in_path}", "--warm", "1", "--reps", "1"],
                       capture_output=True, cwd=ROOT)
        frame = os.path.join(out, "frame_out.bin")
        if not os.path.exists(frame):
            return None
        return np.fromfile(frame, dtype=np.float32)
    finally:
        shutil.rmtree(os.path.dirname(out), ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
