"""Pack and transform Small or full VK102 model blocks into the FP16/FP32 cold-buffer layout bound to the Intel 8x16x16 fp16 cooperative-matrix recipe (B580 verified; B390 Xe3 iGPU adapted, same recipe)."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import struct
import sys
from pathlib import Path
import numpy as np

MODEL_TOOLS = Path(__file__).resolve().parents[1] / "third_party/dlssnr_amd_model_tools"
EXTRACTED: Path
COLD_OUT: Path
PROFILE: dict

# Every finite E4M3 value is exactly representable in binary16.
def e4m3_half_bits(code: int) -> int:
    e, m = (code >> 3) & 15, code & 7
    if e == 15 and m == 7:
        return 0x7E00
    value = m / 512.0 if e == 0 else (1.0 + m / 8.0) * 2.0 ** (e - 7)
    if code & 128:
        value = -value
    return struct.unpack("<H", struct.pack("<e", value))[0]

LUT_U16 = np.asarray([e4m3_half_bits(i) for i in range(256)], dtype="<u2")

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tile_transform(buf: bytes, N: int, K: int) -> bytes:
    a = np.frombuffer(buf, dtype="<u2").reshape(N, K)
    return a.reshape(N // 16, 16, K // 16, 16).transpose(0, 2, 1, 3).copy().tobytes()


def apply_tile_ranges(blob: bytearray, ranges: list[tuple[int, int, int, int]]) -> bytes:
    out = bytearray(blob)
    for start, end, N, K in ranges:
        assert end - start == N * K * 2, (start, end, N, K)
        out[start:end] = tile_transform(bytes(out[start:end]), N, K)
    return bytes(out)


PRE_REGIONS = [
    ("mlp_expand", 0, 4096, 128, 32, 8, "matrix"),
    ("mlp_contract", 4096, 8192, 32, 128, 2, "matrix"),
    ("input_lift", 8208, 9232, 32, 16, None, "fp16"),
    ("residual_scale", 9232, 9312, None, None, None, "fp16"),
    ("qkv", 9312, 12384, 96, 32, 6, "matrix"),
    ("attn_pos_bias", 12384, 20576, None, None, None, "fp16"),
    ("scalars_b", 20576, 20592, None, None, None, "f32"),
    ("attn_out_proj", 20592, 21616, 32, 32, 2, "matrix"),
    ("attn_residual_scale", 21616, 21696, None, None, None, "fp16"),
]

POST_REGIONS = [
    ("mlp_expand", 0, 4096, 128, 32, 8, "matrix"),
    ("mlp_contract", 4096, 8192, 32, 128, 2, "matrix"),
    ("residual_scale", 8208, 8272, None, None, None, "fp16"),
    ("main_gain", 8272, 8336, None, None, None, "fp16"),
    ("skip_gain", 8336, 8400, None, None, None, "fp16"),
    ("qkv", 8400, 11472, 96, 32, 6, "matrix"),
    ("attn_pos_bias", 11472, 19664, None, None, None, "fp16"),
    ("scalars_b", 19664, 19680, None, None, None, "f32"),
    ("attn_out_proj", 19680, 20704, 32, 32, 2, "matrix"),
    ("attn_residual_scale", 20704, 20784, None, None, None, "fp16"),
    ("out_project", 20784, 21808, 16, 32, None, "fp16"),
]


def fragment_indices(N: int, K: int, inner: int) -> np.ndarray:
    n, k = np.indices((N, K))
    return (
        (n // 16 % inner) * 512
        + (k // 32) * 512 * inner
        + (n // 16 // inner) * 512 * inner * (K // 32)
        + (4 * (n % 8) + (k % 8) // 2) * 16
        + (n % 16 // 8) * 8
        + 2 * (k % 32 // 8)
        + k % 2
    )


def pack_ioport(block: int, C: int, family_mod) -> tuple[bytes, bytes]:
    shape_block = int(PROFILE["ioport_shapes"][str(block)])
    name = f"block{block}.layer0.layer"
    payload = (EXTRACTED / "inventory" / "weights" / f"{name}.bin").read_bytes()
    if shape_block in (0, 30):
        regions = PRE_REGIONS if shape_block == 0 else POST_REGIONS
        directory = "unpacked-preblock" if shape_block == 0 else "unpacked-postblock"
    elif shape_block == 19:
        regions = [
            ("weight", 0, 524288, 512, 1024, 32, "matrix"),
            ("skip_weight", 524288, 525312, None, None, None, "fp16"),
        ]
        directory = "unpacked-splitswin"
    else:
        ci, co = (C, 2 * C) if shape_block in (2, 4, 6, 8) else (2 * C, C)
        regions = family_mod.regions(C, C // 32, ci, co)
        directory = "unpacked"

    blob = bytearray()
    raw_parts = {}

    def put(role: str, data: bytes):
        off = (len(blob) + 255) // 256 * 256
        blob.extend(bytes(off - len(blob)))
        blob.extend(data)

    for role, start, end, N, K, inner, kind in regions:
        if role == "tail":
            continue
        ext_path = EXTRACTED / directory / f"{name}.{role}.bin"
        ext = ext_path.read_bytes()
        region = payload[start:end]
        if kind == "matrix":
            si = fragment_indices(N, K, inner)
            recovered = np.frombuffer(region, np.uint8)[si].tobytes()
            assert recovered == ext, (block, role, "packed/extracted matrix mismatch")
            data = LUT_U16[np.frombuffer(ext, np.uint8)].tobytes()
        elif role == "out_project":
            n, k = np.indices((16, 32))
            si = (k // 16) * 256 + 32 * (n % 8) + 8 * ((k % 8) // 2) + 4 * (n // 8) + k % 2 + 2 * (k % 16 // 8)
            data = np.frombuffer(region, "<u2")[si].tobytes()
            assert data == ext, "post out_project extracted mismatch"
        elif role == "input_lift":
            assert region == ext, "pre input_lift extracted mismatch"
            # The source tensor is in the lane/byte scatter layout; the author
            # reference reconstructs the logical 32x16 matrix with this inverse.
            # The shader loads the logical matrix at p7.x+g*128, so store the
            # scattered (logical) matrix, not the raw file order.
            n, k = np.indices((32, 16))
            si = (n // 16) * 256 + 32 * (n % 8) + 8 * ((k % 8) // 2) + 4 * (n // 8 % 2) + k % 2 + 2 * (k // 8)
            data = np.frombuffer(ext, "<u2")[si].tobytes()
        else:
            assert region == ext, (block, role, "raw extracted mismatch")
            data = ext
        raw_parts[role] = data
        put(role, data)

    aux = bytearray()
    if shape_block != 19:
        rs = np.frombuffer(raw_parts["residual_scale"], "<u2")
        rbase = 8 if np.all(rs[:8] == 0) else 0
        if shape_block in (0, 30):
            normalized = bytes(16) + rs[rbase : rbase + C].tobytes() + bytes(16)
            put("body_residual_scale", normalized)
        if shape_block in (22, 24, 26, 28):
            gain = np.frombuffer(raw_parts["upsample_gain"], "<u2")
            if C > 32:
                gain = np.concatenate((rs[C : C + 16], gain))
            assert len(gain) == C
            put("blend_gain", gain.tobytes())
        if shape_block == 30:
            put("blend_gain", raw_parts["skip_gain"] + raw_parts["main_gain"])
        bias = np.frombuffer(raw_parts["attn_pos_bias"], "<u2")
        i, j = np.indices((64, 64))
        si = (
            (4 * (i // 16) + j // 16) * 256
            + (4 * (i % 8) + (j % 8) // 2) * 8
            + (j % 2)
            + (i % 16 // 8) * 2
            + (j % 16 // 8) * 4
        )
        canonical = np.stack([bias[h * 4096 + si] for h in range(C // 32)])
        aux.extend(canonical.view("<f2").astype("<f4").tobytes())
        aux.extend(raw_parts["scalars_b"])

    # Apply S1 / R4 16x16 B-tile transformations.
    # GPU A/B (runs 005 vs 007) rejects the "canonical pre/post pack" hypothesis:
    # canonical made b0 worse (mean_abs 0.094 -> 0.394), so keep TILE16.
    if shape_block == 0:
        ranges = [(0, 8192, 128, 32), (8192, 16384, 32, 128), (17664, 23808, 96, 32), (32256, 34304, 32, 32)]
    elif shape_block == 30:
        ranges = [(0, 8192, 128, 32), (8192, 16384, 32, 128), (17152, 23296, 96, 32), (31744, 33792, 32, 32)]
    elif shape_block == 19:
        ranges = [(0, 1048576, 512, 1024)]
    elif shape_block == 2:
        ranges = [(33536, 37632, 64, 32)]
    elif shape_block == 4:
        ranges = [(107264, 123648, 128, 64)]
    elif shape_block == 6:
        ranges = [(361472, 427008, 256, 128)]
    elif shape_block == 8:
        ranges = [(1312256, 1574400, 512, 256)]
    elif shape_block == 22:
        ranges = [(720896, 983040, 256, 512)]
    elif shape_block == 24:
        ranges = [(196608, 262144, 128, 256)]
    elif shape_block == 26:
        ranges = [(57344, 73728, 64, 128)]
    elif shape_block == 28:
        ranges = [(16384, 20480, 32, 64)]
    else:
        raise ValueError(block)

    transformed = apply_tile_ranges(blob, ranges)
    return transformed, bytes(aux or bytes(4))


SWIN_ROLES = (
    "attn_out_proj",
    "attn_pos_bias",
    "attn_residual_scale",
    "mlp_contract",
    "mlp_expand",
    "mlp_mid",
    "qkv",
    "residual_scale",
    "scalars_b",
)


def pack_swin(block: int, c: int, unpacked_manifest: dict) -> tuple[bytes, bytes]:
    record = f"block{block}.layer0.layer"
    layer = next(x for x in unpacked_manifest["layers"] if x["record"] == record)
    assert layer["C"] == c and layer["heads"] == c // 32
    raw = (EXTRACTED / "inventory" / "weights" / f"{record}.bin").read_bytes()
    tensors = {x["name"]: x for x in layer["tensors"]}
    unpacked = EXTRACTED / "unpacked"
    pack = bytearray()
    ext_bytes = {}
    role_off = {}
    for role in (r for r in SWIN_ROLES if c != 32 or r != "mlp_mid"):
        t = tensors[role]
        ext = (unpacked / f"{record}.{role}.bin").read_bytes()
        start, end = t["src_range"]
        if t["kind"] == "matrix":
            nrows, kcols = t["N"], t["K"]
            inner = 8 if role == "mlp_expand" else 2 if role == "mlp_mid" else nrows // 16
            si = fragment_indices(nrows, kcols, inner)
            recovered = np.frombuffer(raw[start:end], np.uint8)[si].tobytes()
            assert recovered == ext, (block, role)
            data = LUT_U16[np.frombuffer(ext, np.uint8)].tobytes()
        else:
            assert raw[start:end] == ext, (block, role)
            data = ext
        pack.extend(bytes((-len(pack)) % 256))
        role_off[role] = len(pack)
        pack.extend(data)
        ext_bytes[role] = ext

    bias = np.frombuffer(ext_bytes["attn_pos_bias"], "<u2")
    i, j = np.indices((64, 64))
    si = (
        (4 * (i // 16) + j // 16) * 256
        + (4 * (i % 8) + (j % 8) // 2) * 8
        + (j % 2)
        + (i % 16 // 8) * 2
        + (j % 16 // 8) * 4
    )
    canonical = np.stack([bias[h * 4096 + si] for h in range(c // 32)])
    aux = bytearray(canonical.view("<f2").astype("<f4").tobytes())
    aux.extend(ext_bytes["scalars_b"])

    if c == 32:
        ranges = [(0, 2048, 32, 32), (10496, 18688, 32, 128), (18688, 26880, 128, 32), (26880, 33024, 96, 32)]
    elif c == 64:
        ranges = [(0, 8192, 64, 64), (24832, 33024, 64, 64), (33024, 65792, 256, 64), (82176, 106752, 192, 64)]
    elif c == 128 and block == PROFILE["downsample_blocks"][2]:
        ranges = [(0, 32768, 128, 128), (65792, 98560, 128, 128), (98560, 229632, 512, 128), (262400, 360704, 384, 128)]
    elif c == 128:
        ranges = [(0, 32768, 128, 128), (66048, 98816, 128, 128), (98816, 229888, 512, 128), (262656, 360960, 384, 128)]
    elif c == 256 and block == PROFILE["downsample_blocks"][3]:
        ranges = [(197120, 328192, 256, 256), (328192, 852480, 1024, 256)]
    elif c == 256:
        ranges = [(197376, 328448, 256, 256), (328448, 852736, 1024, 256)]
    else:
        raise ValueError((block, c))

    # C256 layouts, measured against the frozen Small assets:
    # * attn_out_proj is loaded as MAT_B 16x16 tiles (body_4:
    #   coopMatLoad(b, pack, wo + n0*256 + k0, 256, LCOL)).
    # * qkv stays plain rows: c256_common.glsl walks it as
    #   wq + (h*3+2)*T*256 + t*256, i.e. contiguous 256-wide rows, so tiling it
    #   made Q/K uncorrelated (measured -0.002 / +0.052 against the reference).
    c256 = c == 256 and PROFILE["backend_key"] == "small"
    if c256 and block in PROFILE.get("c256_finaltile_blocks", ()):
        off = role_off["attn_out_proj"]
        ranges.append((off, off + 256 * 256 * 2, 256, 256))

    transformed = apply_tile_ranges(pack, ranges)
    if c256:
        # residual_scale: the runtime `rs` points at role_off, but the tensor
        # carries an 8-half zero lead-in, so every channel read the scale of
        # c-8 (the first eight read 0).  Rotate the lead-in behind the 256-entry
        # table, which is exactly what patch_small_c256.py used to do on the
        # installed assets.  Blocks whose scale has no lead-in are left alone.
        words = np.frombuffer(transformed, dtype="<u2").copy()
        off = role_off["residual_scale"] // 2
        if not np.any(words[off:off + 8]):
            words[off:off + 256] = words[off + 8:off + 8 + 256].copy()
            words[off + 256:off + 272] = 0
        transformed = words.tobytes()
    return transformed, bytes(aux)


def bits_field(x: np.ndarray, positions: list[int]) -> np.ndarray:
    v = np.zeros_like(x, dtype=np.int64)
    for b, dest in enumerate(positions):
        v |= ((x >> b) & 1) << dest
    return v


def tile16_indices(N: int, K: int) -> np.ndarray:
    n = np.arange(N, dtype=np.int64)[:, None]
    k = np.arange(K, dtype=np.int64)[None, :]
    return ((n // 16) * (K // 16) * 256 + (k // 16) * 256
            + (n % 16) * 16 + k % 16)


def ffwd_maps():
    # S20's L0 payload is not a plain row-major 512x512 matrix: the upstream
    # unpacker's three ffwd maps interleave two payload matrices.  The exact
    # consumer binding is still unproven (analysis/weights/proof.json), so this
    # pack keeps the previously verified value-multiset maps rather than an
    # unverified TILE16 rewrite.
    g = np.arange(8)[:, None, None]
    r = np.arange(64)[None, :, None]
    k = np.arange(512)[None, None, :]
    a = bits_field(g, [11, 12, 13]) | bits_field(r, [3, 6, 7, 8, 9, 10]) | bits_field(k, [0, 4, 5, 1, 2, 14, 15, 16, 17])
    j = np.arange(256)[None, :, None]
    r2 = np.arange(64)[None, None, :]
    q0 = 262144 + g * 16384 + (bits_field(r2, [1, 0, 4, 5, 2, 13]) | bits_field(j, [6, 3, 9, 7, 8, 10, 11, 12]))
    n = np.arange(64)[None, :, None]
    j2 = np.arange(256)[None, None, :]
    q2 = 393216 + g * 16384 + (bits_field(j2, [0, 1, 2, 4, 5, 11, 12, 13]) | bits_field(n, [6, 7, 8, 3, 9, 10]))
    return a, q0, q2


def matrix_map(N: int, K: int) -> np.ndarray:
    n = np.arange(N)[:, None]
    k = np.arange(K)[None, :]
    lane = 4 * (n % 8) + (k % 8) // 2
    byte = 2 * ((k % 32) // 8) + k % 2
    return ((k // 32) * (N // 16) + n // 16) * 512 + lane * 16 + (n % 16 // 8) * 8 + byte


def slot_indices(groups: int, outs: int, ins: int) -> np.ndarray:
    """Word index of (group, out, in) inside one kernel FFN matrix.

    Same rule as ``pack_split16_ffn.py`` (verified there with single-word
    pulses on block 9): out-major 16x16 tiles, input-block stride 256 words,
    out-block stride ``(ins/16)*256``, group stride ``outs*ins``.
    """
    g = np.arange(groups, dtype=np.int64)[:, None, None]
    o = np.arange(outs, dtype=np.int64)[None, :, None]
    k = np.arange(ins, dtype=np.int64)[None, None, :]
    return (g * outs * ins
            + (o // 16) * ((ins // 16) * 256)
            + (k // 16) * 256
            + 16 * (o % 16) + (k % 16))


def split16_l0(block: int) -> bytes:
    """The 1 MiB L0 zone exactly as the Small SplitFfwd consumer reads it.

    ``ffwd_maps`` is the raw -> logical map (expand_in [in, g*64+out],
    expand_hidden [g, in, out], contract [g, in, out]); ``slot_indices`` then
    writes the logical tensors in the consumer's grouped out-major tiling.
    Reproduces the frozen ``split16_b9/b10/b20/b21`` blobs byte for byte.
    """
    raw = np.frombuffer((EXTRACTED / "inventory/weights" / f"block{block}.layer0.layer.bin").read_bytes(), np.uint8)
    amap, bmap0, bmap2 = ffwd_maps()
    words = np.zeros(524288, dtype="<u2")
    words[slot_indices(8, 64, 512).reshape(-1)] = LUT_U16[raw[amap].reshape(-1)]
    words[slot_indices(8, 256, 64).reshape(-1) + 262144] = LUT_U16[raw[bmap0].reshape(-1)]
    words[slot_indices(8, 64, 256).reshape(-1) + 393216] = LUT_U16[raw[bmap2].reshape(-1)]
    return words.tobytes()


def pack_split16(block: int) -> tuple[bytes, bytes]:
    packed = bytearray()

    def add(data: bytes):
        off = (len(packed) + 255) & ~255
        packed.extend(bytes(off - len(packed)))
        packed.extend(data)

    sw = EXTRACTED / "unpacked-splitswin"
    # Small consumes the L0 zone in the grouped out-major tiling written by
    # split16_l0 (no extra TILE16 pass); VK102 S20 consumes the original
    # per-head ffwd maps before TILE16.
    if PROFILE["backend_key"] == "102":
        raw = np.frombuffer((EXTRACTED / "inventory/weights" / f"block{block}.layer0.layer.bin").read_bytes(), np.uint8)
        amap, bmap0, bmap2 = ffwd_maps()
        add(LUT_U16[raw[amap]].tobytes())
        add(LUT_U16[raw[bmap0]].tobytes())
        add(LUT_U16[raw[bmap2]].tobytes())
    elif PROFILE["backend_key"] == "small":
        add(split16_l0(block))
    else:
        a = LUT_U16[np.frombuffer((sw / f"block{block}.layer0.layer.ffwd_a.bin").read_bytes(), np.uint8)].tobytes()
        b = LUT_U16[np.frombuffer((sw / f"block{block}.layer0.layer.ffwd_b.bin").read_bytes(), np.uint8)].tobytes()
        assert len(a) == 524288 and len(b) == 524288, (block, len(a), len(b))
        add(a)
        add(b[:262144])
        add(b[262144:])

    auxiliary = b""
    for layer, role, N in ((1, "ff_proj", 512), (2, "qkv", 1536), (3, "out_proj", 512)):
        name = f"block{block}.layer{layer}.layer"
        raw = (EXTRACTED / "inventory" / "weights" / f"{name}.bin").read_bytes()
        canonical = np.frombuffer(raw, np.uint8)[matrix_map(N, 512)]
        ext = EXTRACTED / "unpacked-splitswin" / (name + (".qkv.bin" if layer == 2 else ".weight.bin"))
        assert canonical.tobytes() == ext.read_bytes(), str(ext)
        add(LUT_U16[canonical].tobytes())
        if layer != 2:
            ext_skip = EXTRACTED / "unpacked-splitswin" / f"{name}.skip_weight.bin"
            assert raw[N * 512 :] == ext_skip.read_bytes() and len(raw[N * 512 :]) == 1024
            add(ext_skip.read_bytes())
        else:
            bias_path = EXTRACTED / "unpacked-splitswin" / f"{name}.attn_pos_bias.bin"
            scale_path = EXTRACTED / "unpacked-splitswin" / f"{name}.tail.bin"
            assert raw[786432:917504] == bias_path.read_bytes()
            assert raw[917504:917568] == scale_path.read_bytes()
            raw_bias = np.frombuffer(bias_path.read_bytes(), "<f2")
            h, i, j = np.indices((16, 64, 64), dtype=np.int64)
            lane = 4 * (i % 8) + (j % 8) // 2
            sl = (j % 2) | ((i % 16 // 8) << 1) | ((j % 16 // 8) << 2)
            index = h * 4096 + (4 * (i // 16) + j // 16) * 256 + lane * 8 + sl
            bias = raw_bias[index].astype("<f4")
            scales = np.frombuffer(scale_path.read_bytes(), "<f4")
            assert scales.size == 16 and np.all(np.isfinite(scales))
            auxiliary = bias.tobytes() + scales.tobytes()

    if block == PROFILE["split_last"]:
        name = f"block{block}.layer4.layer"
        raw4 = (EXTRACTED / "inventory" / "weights" / f"{name}.bin").read_bytes()
        ext4 = EXTRACTED / "unpacked-splitswin" / f"{name}.weight.bin"
        canonical4 = np.frombuffer(raw4, np.uint8)[matrix_map(1024, 512)]
        assert canonical4.tobytes() == ext4.read_bytes()
        # Keep the prior canonical pack: the final-head consumer binding is
        # still unproven (analysis/weights/proof.json).
        add(LUT_U16[canonical4].tobytes())

    ranges = [
        (1573888, 3146752, 1536, 512),
        (3146752, 3671040, 512, 512),
    ]
    if PROFILE["backend_key"] != "small":
        ranges = [
            (0, 524288, 512, 512),
            (524288, 786432, 2048, 64),
            (786432, 1048576, 512, 256),
        ] + ranges
    if block == PROFILE["split_last"] and PROFILE["backend_key"] in ("102", "small"):
        # Frozen VK102 S24/S25 and Small b10 keep canonical (untiled) last-head
        # projection weights; their SplitFinalHead consumer reads plain rows.
        ranges = [r for r in ranges if r[0] != 3146752]
    # Small's final-head L4.weight stays plain row-major too: blob[o*in + k].
    # Tiling it was the last byte difference against frozen split16_b10.
    transformed = apply_tile_ranges(packed, ranges)
    return transformed, auxiliary


def vit_canonical(raw: bytes, n: int, k: int, qkv: bool = False) -> np.ndarray:
    r = np.arange(n, dtype=np.int32)[:, None]
    c = np.arange(k, dtype=np.int32)[None, :]
    if qkv:
        which, r = r // 1024, r % 1024
        b = (
            (c & 1)
            | ((c >> 1 & 1) << 4)
            | ((c >> 2 & 1) << 5)
            | ((c >> 3 & 1) << 1)
            | ((c >> 4 & 1) << 2)
            | ((r & 1) << 6)
            | ((r >> 1 & 1) << 7)
            | ((r >> 2 & 1) << 8)
            | ((r >> 3 & 1) << 3)
            | ((r >> 4 & 1) << 9)
        )
        addr = 128 + (3 * ((r >> 5) | ((c >> 5) << 5)) + which) * 1024 + b
    else:
        addr = (
            (c // 32) * (n // 16) * 512
            + (r // 16) * 512
            + (4 * (r % 8) + (c % 8) // 2) * 16
            + ((r % 16) // 8) * 8
            + 2 * ((c % 32) // 8)
            + c % 2
        )
    return np.frombuffer(raw, dtype=np.uint8)[addr]


def pack_vit(block: int) -> tuple[bytes, bytes]:
    rec = {}
    for layer in range(5):
        name = f"block{block}.layer{layer}.layer"
        rec[layer] = (EXTRACTED / "inventory" / "weights" / f"{name}.bin").read_bytes()
    pack = bytearray()
    for role, layer, n, k in [("WE", 0, 4096, 1024), ("WC", 1, 1024, 4096), ("WQ", 2, 3072, 1024), ("WO", 4, 1024, 1024)]:
        codes = vit_canonical(rec[layer], n, k, role == "WQ")
        parts = ["q", "k", "v"] if role == "WQ" else ["weight"]
        ext_parts = [(EXTRACTED / "unpacked-vit" / f"block{block}.layer{layer}.layer.{p}.bin").read_bytes() for p in parts]
        assert codes.tobytes() == b"".join(ext_parts), (block, role)
        bits = LUT_U16[codes]
        physical = bits.reshape(n // 32, 32, k // 16, 16).transpose(0, 2, 1, 3).copy()
        pack.extend(physical.tobytes())
    for role, layer, matrix_bytes in [("RS", 1, 1024 * 4096), ("ARS", 4, 1024 * 1024)]:
        raw = rec[layer][matrix_bytes:]
        ext = (EXTRACTED / "unpacked-vit" / f"block{block}.layer{layer}.layer.skip_weight.bin").read_bytes()
        assert len(raw) == 2048 and ext == raw, (block, role)
        pack.extend(raw)
    scalars = rec[2][:128]
    assert np.isfinite(np.frombuffer(scalars, dtype="<f4")).all()
    return bytes(pack), scalars


def save_cold(logical_name: str, filename_suffix: str, data: bytes, index: dict) -> Path:
    digest = sha256_bytes(data)
    out_name = f"{digest}__{filename_suffix}"
    out_path = COLD_OUT / out_name
    out_path.write_bytes(data)
    index[logical_name] = {
        "logical_name": logical_name,
        "path": str(out_path),
        "bytes": len(data),
        "sha256": digest,
    }
    return out_path


def main():
    global EXTRACTED, COLD_OUT, PROFILE
    parser = argparse.ArgumentParser(description="Prepare B580 FP16 cold resources from verified model tensors")
    parser.add_argument("--profile", choices=["small", "vk102"], default="small")
    parser.add_argument("--extracted", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--full-model", type=Path, help="Built Full model directory; supplies the accepted Small blend scalar")
    parser.add_argument("--blend-scalar", type=Path,
                        help="4-byte resource used instead of --full-model: the accepted Small blend scalar")
    args = parser.parse_args()
    PROFILE = json.loads((Path(__file__).with_name("model_profiles.json")).read_text(encoding="utf-8"))[args.profile]
    EXTRACTED = args.extracted.resolve()
    COLD_OUT = args.out.resolve()
    if COLD_OUT.exists() and any(COLD_OUT.iterdir()):
        raise SystemExit("Output must be a new or empty directory")
    COLD_OUT.mkdir(parents=True, exist_ok=True)

    spec = importlib.util.spec_from_file_location("unpack_swin_family", MODEL_TOOLS / "unpack_swin_family.py")
    family_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(family_mod)

    unpacked_manifest = json.loads((EXTRACTED / "unpacked" / "manifest.json").read_text(encoding="utf-8"))
    index = {}

    shape_width = {0:32, 2:32, 4:64, 6:128, 8:256, 19:512, 22:256, 24:128, 26:64, 28:32, 30:32}
    for block, shape in PROFILE["ioport_shapes"].items():
        b, c = int(block), shape_width[shape]
        w, aux = pack_ioport(b, c, family_mod)
        save_cold(f"ioport_{b}_ioport_pack", f"ioport_{b}_ioport_pack.fp16.bin", w, index)
        if b in (0, PROFILE["post_block"]):
            save_cold(f"ioport_{b}_ioport_body_aux", "ioport_aux.f32.bin", aux, index)
        print(f"Packed ioport_{b}: weights={len(w)}B aux={len(aux)}B")
    finaltile = set(PROFILE.get("c256_finaltile_blocks", ()))
    for block, c in PROFILE["swin_blocks"].items():
        b = int(block)
        w, aux = pack_swin(b, c, unpacked_manifest)
        # The frozen assets name the repaired C256 packs *_weights_finaltile;
        # pack_swin now writes that layout directly, so the name is the only
        # thing the old weight-transforms recipes still contributed.
        suffix = "_finaltile" if b in finaltile else ""
        save_cold(f"swin_{b}_weights{suffix}", f"swin_{b}_weights{suffix}.fp16.bin", w, index)
        save_cold(f"swin_{b}_aux", f"c{c}_aux.bin", aux, index)
    for b in PROFILE["split_blocks"]:
        w, aux = pack_split16(b)
        save_cold(f"split16_b{b}_weights", f"split16_b{b}_weights.fp16.bin", w, index)
        save_cold(f"split16_b{b}_aux", "split16_aux.bin", aux, index)
    for b in PROFILE["vit_blocks"]:
        w, scales = pack_vit(b)
        prefix = f"vit_block{b}" if args.profile == "vk102" else f"vit_b{b}"
        save_cold(prefix + "_weights", "vit_weights.fp16.bin", w, index)
        save_cold(prefix + "_scales", "vit_scales.fp32.bin", scales, index)
    post = PROFILE["post_block"]
    blend = (EXTRACTED / "inventory" / "weights" / f"block{post}.layer0.blend_scale.bin").read_bytes()
    assert len(blend) == 2
    save_cold(f"ioport_{post}_ioport_blend_scale", "ioport_blend_scale.fp16.bin", blend + bytes(2), index)

    # Accepted FP16 layouts: permute half words, never requantize model values.
    network = Path(__file__).resolve().parents[1] / PROFILE['network_dir']
    transforms = json.loads((network / 'weight-transforms.json').read_text(encoding='utf-8'))
    originals = dict(index)
    for recipe in transforms:
        source = originals[recipe['input']]
        if source['sha256'] != recipe['input_sha256']:
            raise ValueError('Weight transform input identity: ' + recipe['name'])
        data = Path(source['path']).read_bytes()
        values = np.frombuffer(data, dtype='<u2')
        result = values.copy()
        for offset, count, kind in recipe['regions']:
            part = values[offset:offset + count]
            if kind == 'transpose16':
                mapped = part.reshape(-1, 16, 16).transpose(0, 2, 1).reshape(-1)
            elif kind == 'tile256':
                mapped = part.reshape(16, 16, 16, 16).transpose(0, 2, 1, 3).reshape(-1)
            else:
                raise ValueError('Unknown weight permutation: ' + kind)
            result[offset:offset + count] = mapped
        packed = result.tobytes()
        if hashlib.sha256(packed).hexdigest() != recipe['output_sha256']:
            raise ValueError('Weight transform output identity: ' + recipe['name'])
        save_cold(recipe['name'], recipe['name'] + '.fp16.bin', packed, index)
    expected = json.loads((network / 'cold-hashes.json').read_text(encoding='utf-8'))
    if args.profile == 'small':
        if args.full_model is not None:
            full = json.loads((args.full_model / 'manifest/cold_resources_index.json').read_text(encoding='utf-8'))
            resource = full['ioport_70_ioport_blend_scale']
            data = Path(resource['path']).read_bytes()
        elif args.blend_scalar is not None:
            # Published releases ship this 4-byte resource, so a Small-only install never
            # needs the optimized vk102 build just to reproduce one blend scalar.
            data = args.blend_scalar.read_bytes()
            if len(data) != 4:
                raise ValueError('--blend-scalar must be the 4-byte ioport_blend_scale.fp16.bin resource')
        else:
            raise ValueError('Optimized Small requires --full-model or --blend-scalar')
        save_cold('ioport_30_ioport_blend_scale', 'ioport_blend_scale.fp16.bin', data, index)
    index = {name: index[name] for name in expected}
    for name, record in index.items():
        if (record['sha256'], record['bytes']) != (expected[name]['sha256'], expected[name]['bytes']):
            raise ValueError('Accepted cold resource mismatch: ' + name)

    manifest_dir = args.manifest.resolve()
    manifest_dir.mkdir(parents=True, exist_ok=True)
    index_path = manifest_dir / "cold_resources_index.json"
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"ALL {len(index)} COLD RESOURCES PACKED AND INDEXED AT {index_path}")


if __name__ == "__main__":
    main()
