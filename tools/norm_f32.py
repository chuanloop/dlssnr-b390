"""Sum the 32 head-dimension squares the way the reference does: in float32, one f16 step.

``swin._norm_scale`` is explicit about it: the squares are accumulated by fused multiply-add in
float32, in four interleaved chains over each of the two halves of the dimension set; the halves are
narrowed to f16 and added in f16; a sum that overflows f16 gives a scale of zero; and the reciprocal
root itself is *not* narrowed.

The port does every step in f16: `sq[c] = float16_t(row[c]) * float16_t(row[c])` (products rounded to
11 bits), a f16 tree, and a different association order ((0+2)+(1+3) instead of (0+1)+(2+3)). Each of
those is a ~5e-4-class difference, and each one sits upstream of an e4m3 point where a 12.5% step\r\nwaits, so the effect is larger than it looks.

This rewrites the body to mirror the reference exactly. The squares themselves are exact in f32 (an
f16 value squared needs 22 bits), so multiplication order inside a square does not matter; what does
matter is that every *sum* is f32 and that only the two half sums are narrowed.

Usage
-----
    python tools/norm_f32.py --dry-run          # every kernel under the fp16 profile
    python tools/norm_f32.py
    python tools/norm_f32.py --revert           # restore the .norm-bak copies
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "source/network-fp16-20261027"
FUNC = re.compile(r"float (\w+_norm_sum32)\(float\[32\] row\) \{.*?\n\}", re.S)

BODY = """float %(name)s(float[32] row) {
    // mirrors swin._norm_scale: every square is exact in f32 (an f16 value squared needs 22
    // bits), the four chains accumulate in f32 in the reference's order, and only the two half
    // sums are narrowed to f16 and added in f16.  The port used to do all of it in f16.
    precise float t[8];
    for (uint c = 0u; c < 8u; ++c) {
        precise float s0 = row[c] * row[c];
        precise float s1 = row[c + 16u] * row[c + 16u];
        precise float s2 = row[c + 8u] * row[c + 8u];
        precise float s3 = row[c + 24u] * row[c + 24u];
        t[c] = s3 + (s2 + (s1 + s0));
    }
    precise float16_t h0 = float16_t((t[0] + t[2]) + (t[4] + t[6]));
    precise float16_t h1 = float16_t((t[1] + t[3]) + (t[5] + t[7]));
    precise float16_t total = h0 + h1;
    return float(total);
}"""


def targets() -> list[Path]:
    out = []
    for p in sorted(PROFILE.rglob("*")):
        if p.suffix in (".glsl", ".comp") and "_norm_sum32" in p.read_text(encoding="utf-8", errors="ignore"):
            out.append(p)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--revert", action="store_true")
    a = ap.parse_args()

    files = targets()
    if not files:
        raise SystemExit("no kernel defines *_norm_sum32")

    if a.revert:
        n = 0
        for p in files:
            bak = p.with_suffix(p.suffix + ".norm-bak")
            if bak.is_file():
                if not a.dry_run:
                    p.write_text(bak.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
                    bak.unlink()
                n += 1
        print(f"reverted {n} file(s) from .norm-bak")
        return 0

    total = 0
    for p in files:
        text = p.read_text(encoding="utf-8")
        bak = p.with_suffix(p.suffix + ".norm-bak")
        new, count = FUNC.subn(lambda m: BODY % {"name": m.group(1)}, text)
        if count == 0:
            print(f"  {p.relative_to(PROFILE)}: no match")
            continue
        if not a.dry_run:
            if not bak.is_file():
                bak.write_text(text, encoding="utf-8", newline="\n")
            p.write_text(new, encoding="utf-8", newline="\n")
        total += count
        print(f"  {p.relative_to(PROFILE)}: {count} function(s)")
    print(f"{total} function(s) rewritten across {len(files)} file(s)"
          + (" (dry run)" if a.dry_run else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
