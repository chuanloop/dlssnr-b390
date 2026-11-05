"""Crop a planar/interleaved f32 input to a smaller WxH (timing probes only).

Usage: python tools/crop_input.py <src.f32> <WxH> [<WxH> ...]

Keeps the first W*H*4 floats of the source, so the pixel values are a prefix of the
original image rather than a resample. This is only meant to feed the runner a file of
the right shape when measuring time, never for numerics.
"""

import os
import sys

import numpy as np


def main() -> int:
    src = sys.argv[1]
    a = np.fromfile(src, dtype=np.float32)
    print(f"src {src}: {a.size} floats")
    for spec in sys.argv[2:]:
        w, h = (int(v) for v in spec.lower().split("x"))
        n = w * h * 4
        if n > a.size:
            print(f"  {spec}: needs {n} floats, source has {a.size} -> skipped")
            continue
        dst = os.path.join(os.path.dirname(src), f"in-{w}x{h}.f32")
        np.ascontiguousarray(a[:n]).tofile(dst)
        print(f"  {spec} -> {dst} ({os.path.getsize(dst)} B)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
