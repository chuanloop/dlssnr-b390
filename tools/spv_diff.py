"""Compare one of our kernels against AMD's shipped SPIR-V at the instruction level.

The two disassemblers disagree about layout, and both hide the opcode in different columns:

    spirv-dis (AMD, shipped.spvasm)      %42 = OpFMul %half %a %b
    glslang -H (ours)                     42:   24(float16_t)  FMul 40 41

so a naive `Op\\w+` search finds everything in AMD's file and nothing in ours, and a naive
"first word" search finds the *type* in ours. This parses both.

What it is for: the things that separate us from AMD are arithmetic choices, and the only
place those are certain is the shipped binary. Counting `FConvert` (f32<->f16), `ExtInst
Fma`, and the f16/f32 split of `FMul`/`FAdd` says where a kernel narrows and where it does
not -- the question every "we are 3 dB off" theory eventually comes down to.

Caveat when comparing totals: AMD ships one fused kernel per stage (`fswin` does the whole
swin block, 18k instructions), while our block is split across f04/f05/f06. Compare ratios
and per-matmul counts, not raw totals.

Usage
-----
    python tools/spv_diff.py --kernel source/network-fp16-20261027/f04-8aa2595a969b \
        --amd ../../20-AMD/dlssnr-src/shipped.spvasm [--top 20]
"""

from __future__ import annotations

import argparse
import collections
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GLSLANG = ROOT.parent.parent / "40-ARTIFACTS/_deps/glslang1650/bin/glslang.exe"

AMD_OP = re.compile(r"\b(Op[A-Z]\w+)")
# "  42:   24(float16_t)  FMul 40 41"  |  "  42:  Label"  |  "      Store 1 2"
OURS_OP = re.compile(r"^\s*(?:\d+:)?\s*(?:\d+(?:\([^)]*\))?\s+)?([A-Z]\w+)\b")


def hist_amd(path: Path) -> collections.Counter:
    c: collections.Counter = collections.Counter()
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = AMD_OP.search(line)
        if m:
            c[m.group(1)] += 1
    return c


def hist_ours(path: Path) -> collections.Counter:
    c: collections.Counter = collections.Counter()
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = OURS_OP.match(line)
        if m:
            c["Op" + m.group(1)] += 1
    return c


def disassemble(kernel_dir: Path) -> Path:
    src = kernel_dir / "kernel.comp"
    out = ROOT / "50-SCRATCH" / f"ours-{kernel_dir.name}.spvasm"
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(GLSLANG), "-V", "--target-env", "vulkan1.3", f"-I{src.parent}", "-H", str(src)]
    text = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    if not text.stdout:
        raise SystemExit(f"glslang produced no disassembly for {src}:\n{text.stderr[-800:]}")
    out.write_text(text.stdout, encoding="utf-8", newline="\n")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", help="a recipe directory (…/f04-8aa2595a969b)")
    ap.add_argument("--ours-spvasm", default="", help="an already produced disassembly")
    ap.add_argument("--amd", default=str(ROOT.parent.parent / "20-AMD/dlssnr-src/shipped.spvasm"))
    ap.add_argument("--top", type=int, default=16)
    a = ap.parse_args()

    if a.ours_spvasm:
        ours_path = Path(a.ours_spvasm)
    elif a.kernel:
        ours_path = disassemble(Path(a.kernel).resolve())
        print(f"disassembled -> {ours_path}")
    else:
        raise SystemExit("give --kernel or --ours-spvasm")

    ours = hist_ours(ours_path)
    amd = hist_amd(Path(a.amd)) if Path(a.amd).is_file() else collections.Counter()

    total = lambda c: sum(c.values())
    print(f"\nours {total(ours)} instructions, AMD {total(amd)}\n")
    print(f"{'op':<30}{'ours':>8}{'AMD':>8}")
    for k in sorted(ours, key=lambda k: -ours[k])[: a.top]:
        print(f"{k:<30}{ours[k]:>8}{amd[k]:>8}")

    print("\nprecision-relevant:")
    for k in ("OpFConvert", "OpExtInst", "OpFMul", "OpFAdd", "OpFSub", "OpFDiv",
              "OpCooperativeMatrixMulAddKHR", "OpGroupNonUniformShuffleXor"):
        print(f"  {k:<30} ours {ours[k]:>5}   AMD {amd[k]:>6}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
