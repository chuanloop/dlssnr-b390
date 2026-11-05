#!/usr/bin/env python3
"""Rewrite canvas-size constants baked into a SPIR-V binary.

Some kernels do not take their extent from the push block: `s37_fold_body_r1a_v1` guards with
`px < 1280 && py < 720` and addresses its output as `(py * 1280 + px) * 4`, all as OpConstant.
At 1024x576 that writes 1024 valid texels per 1280-word row and leaves a 256-wide hole, which is
exactly the "one row in five" banding seen on screen (1280 / 256 tiles_x = 5).

A SPIR-V constant is four words (`opcode|wordcount, result_type, result_id, value`) and its value
sits at a fixed offset, so substituting it is length-preserving and needs no toolchain. Only exact
value matches are rewritten, and the caller is expected to have checked that the constant has one
meaning in that kernel.

    python tools/patch_spv_size.py in.spv out.spv 1280=1024 720=576
"""
import struct
import sys

MAGIC = 0x07230203
OP_CONSTANT = 43


def patch(src_bytes, repl):
    data = bytearray(src_bytes)
    magic = struct.unpack_from("<I", data, 0)[0]
    if magic != MAGIC:
        raise SystemExit(f"not a SPIR-V binary (magic {magic:#x})")
    off, hits = 20, {}
    while off + 4 <= len(data):
        word = struct.unpack_from("<I", data, off)[0]
        opcode, wcount = word & 0xFFFF, word >> 16
        if wcount == 0:
            raise SystemExit(f"zero word count at {off:#x}")
        if opcode == OP_CONSTANT and wcount == 4:
            value = struct.unpack_from("<I", data, off + 12)[0]
            if value in repl:
                struct.pack_into("<I", data, off + 12, repl[value])
                hits[value] = hits.get(value, 0) + 1
        off += wcount * 4
    return bytes(data), hits


def main():
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    repl = {}
    for spec in sys.argv[3:]:
        old, new = spec.split("=")
        repl[int(old)] = int(new)
    with open(sys.argv[1], "rb") as f:
        data = f.read()
    out, hits = patch(data, repl)
    with open(sys.argv[2], "wb") as f:
        f.write(out)
    print(f"{sys.argv[1]} -> {sys.argv[2]}: {hits or 'no constant matched'}")
    missing = [k for k in repl if k not in hits]
    if missing:
        print(f"  warning: no OpConstant for {missing}")


if __name__ == "__main__":
    main()
