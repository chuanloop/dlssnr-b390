#!/usr/bin/env python3
"""Generate (or check) the Small profile's PORT_NETWORK plan at an arbitrary frame size.

Why this exists
---------------
`port_network_runtime.txt` is not resolution independent.  Every PASS carries its dispatch
grid and a 32-word push block, both written for one canvas: 1280x720 in, 1344x768 canvas.  Feeding a 1024x576 frame to that plan
does not work -- the grid still covers 1344x768 and the push still spells those numbers out.

The layer geometry is resolution independent and already available: `tools/nr_geometry.py`
holds the Small network (`NETWORKS["small"]`) and reproduces the shipped plan's 30 dispatch
grids exactly at 1280x720.  Generating a plan is therefore mechanical:

    for each PASS:  grid <- geometry,  push[size fields] <- geometry,  rest untouched.

Two things had to be reverse engineered and are pinned down here:

1. **Push layout is per shader, not per pass family.**  `Swin1H` keeps (tiles_x, tiles_y) in
   words 0 and 14; `Swin2H/4H/8H` keep them in 0 and 1 and add the tile grid in 4.  That is
   why fields_of() keys on the pass family *and* the layer kind.

2. **Every kernel grids by `ceil(items / items_per_workgroup)`.**  The ViT stages consume 32 /
   128 / 128 / 64 / 128 tokens per workgroup; split16 l0/l1 consume 16 tokens per workgroup,
   l3 consumes 32 (first split block) or 16 (second, different shader), and l4 grids over the
   ViT token count rather than the layer's.  `w*h` is the *unpadded* extent: at 1280x720 the
   split16 layer is 24x44 = 1056 and ceil(1056/16) = 66, exactly the shipped grid -- the padded
   1152 would have given 72 and appears nowhere.

`--check` recomputes every grid and every declared push field at 1280x720 and demands it equal
the shipped value.  All 108 passes must pass; a wrong expression fails loudly instead of
silently mis-sizing a dispatch.  What --check cannot catch is a *missing* field, so any
generated plan still has to be validated numerically (nr_gate / nr_budget) before deployment.

Usage
-----
    python tools/gen_plan_small.py --check
    python tools/gen_plan_small.py --size 1024x576 --out local/small/manifest/plan-1024x576.txt

Note: buffers are *not* resized.  A smaller canvas fits the shipped 1344x768 allocations, so
generated plans run as-is (wasting memory); going above 1344x768 would need new sizes and is
rejected.
"""
from __future__ import annotations

import argparse
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nr_geometry as G  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PLAN = os.path.join(ROOT, "local/small/manifest/port_network_runtime.txt")
SRC_SIZE = (1280, 720)
SRC_CANVAS = G.working_extent(*SRC_SIZE, "small")

# ViT: tokens one workgroup consumes, output channels, and how grid_y relates to channels/32.
VIT_TOKENS_PER_WG = {"ffn_expand": 32, "ffn_contract": 128, "qkv": 128, "attention": 64,
                     "projection": 128}
VIT_CHANNELS = {"ffn_expand": 1024, "ffn_contract": 1024, "qkv": 3072, "attention": 1024,
                "projection": 1024}
# The ViT GEMM template emits 32 output columns per workgroup, so gy = channels/32
# for every stage.  The old factor {"projection": 2} was wrong: the projection
# writes a 1024-wide row (`o[t * 1024 + c]`, see vit_nb04_mpair.glsl) and its
# weight block is exactly 1024x1024 halves (WO = 12582912..12583912 in the push
# constant).  gy=64 read 1024 halves past that block and wrapped columns
# 1024..2047 onto row t+1 of the output tensor - twice the work, plus an
# overwrite of the next token's slot.
VIT_GRID_Y_FACTOR = {}


def load_plan(path: str) -> list[dict]:
    passes: list[dict] = []
    cur: dict | None = None
    for line in open(path, encoding="utf-8").read().splitlines():
        if line.startswith("PASS"):
            w = line.split()
            cur = {"name": w[2].strip('"'), "gx": int(w[8]), "gy": int(w[9]),
                   "spv": os.path.basename(w[3].strip('"')), "push": []}
            passes.append(cur)
        elif line.startswith("PUSH") and cur is not None and not cur["push"]:
            cur["push"] = [int(x) for x in line.split()[1:]]
    return passes


def layer_of(name: str, specs: dict[int, G.LayerSpec]) -> G.LayerSpec | None:
    """Which layer this pass runs on (the layer's own extent, not its output extent)."""
    if re.match(r"r\d+_f1c\d+_image_pre", name):
        return specs[(0, 0)]
    if "s37_fold" in name:
        return specs[(30, 0)]
    if "split16" in name:
        # l0..l3 all run on the 24x44 split extent (1152 tokens) of the *first* split block; only
        # l4 drops to the pooled extent, and b10/b21 reuse the same extents with another shader.
        blk = int(re.search(r"split16_b(\d+)", name).group(1))
        return specs[(9 if blk in (9, 20) else 10, 2)]
    if "c3x_" in name:
        chan = int(re.search(r"c3x_(?:pre|body)_c(\d+)", name).group(1))
        return {64: specs[(26, 0)], 128: specs[(24, 0)], 256: specs[(22, 0)]}[chan]
    if "f3c" in name and "body_ds" in name:
        chan = int(re.search(r"f3c(\d+)_body_ds", name).group(1))
        return {64: specs[(4, 0)], 128: specs[(6, 0)], 256: specs[(8, 0)]}[chan]
    if "t3_" in name and "project_blend" in name:
        return specs[(10, 0)]
    m = re.search(r"block(\d+)", name)
    if m and (int(m.group(1)), 0) in specs:
        return specs[(int(m.group(1)), 0)]
    if "ioport" in name:
        return specs[(int(re.search(r"ioport_block(\d+)", name).group(1)), 0)]
    return None


def vit_tokens(specs: dict) -> int:
    return next(s.tokens for s in specs.values() if s.kind.startswith("Vit"))


def resolve(expr: str, layer: G.LayerSpec | None, specs: dict,
            canvas: tuple[int, int], out: tuple[int, int]) -> int:
    if expr == "CANV_W":
        return canvas[0]
    if expr == "CANV_H":
        return canvas[1]
    if expr == "OUT_W":
        return out[0]
    if expr == "OUT_H":
        return out[1]
    if expr == "OUT_PIX":
        return out[0] * out[1] * 4           # f1c32 writes four f32 planes per pixel
    m = re.fullmatch(r"WH\*(\d+)(?:\+(\d+))?", expr)
    if m:                                    # frame-pixel planes: history / motion / depth bases
        return out[0] * out[1] * int(m.group(1)) + int(m.group(2) or 0)
    if expr == "VT":
        return vit_tokens(specs)
    if expr == "FH_TOK":
        return specs[(10, 4)].tokens
    if expr == "FH_H":
        return specs[(10, 4)].h
    if expr == "FH_W":
        return specs[(10, 4)].w
    if layer is None:
        raise KeyError(f"{expr} needs a layer")
    return {"w": layer.w, "h": layer.h, "ow": layer.out_w, "oh": layer.out_h,
            "gx": layer.grid_x, "gy": layer.grid_y, "tok": layer.tokens}[expr]


def grid_of(name: str, layer: G.LayerSpec | None, specs: dict,
            out: tuple[int, int] = SRC_SIZE) -> tuple[int, int] | None:
    if layer is None:
        return None
    if "vit" in name:
        stage = re.search(r"vit_block\d+_(\w+)$", name).group(1)
        gy = VIT_CHANNELS[stage] // 32 * VIT_GRID_Y_FACTOR.get(stage, 1)
        return math.ceil(vit_tokens(specs) / VIT_TOKENS_PER_WG[stage]), gy
    if "split16" in name:
        m = re.search(r"split16_b(\d+)_l(\d+)", name)
        blk, lvl = int(m.group(1)), int(m.group(2))
        wh = layer.w * layer.h                                  # unpadded extent
        if lvl == 0:
            return math.ceil(wh / 16), 2
        if lvl == 1:
            return math.ceil(wh / 16), 4
        if lvl == 2:
            attn = specs[(9 if blk in (9, 20) else 10, 2)]       # this block's attention tiling
            return attn.grid_x, attn.grid_y
        if lvl == 3:
            # first split block contracts 32 tokens per wg, the second 16 (different shader)
            return (math.ceil(wh / 16), 4) if blk == 10 else (math.ceil(wh / 32), 8)
        return math.ceil(vit_tokens(specs) / 16), 8             # l4 pools over the ViT tokens
    if "ioport" in name:
        if "block30" in name:
            return out[0] * out[1] // 32, 1                     # frame copy, 32 px per wg
        return specs[(9, 0)].tokens * 7, 2                       # 1152 tokens x seven planes
    if "t3_" in name:
        return math.ceil(vit_tokens(specs) / 8), 16
    return layer.grid_x, layer.grid_y


def fields_of(name: str, layer: G.LayerSpec | None) -> dict[int, tuple[str, int]]:
    """Size-carrying push words, grouped the way the shaders group them, not by task."""
    if re.match(r"r\d+_f1c\d+_image_pre$", name):
        # p4 = IW, IH, image float base, cold-noise float base (the last one is 11 frame-pixels in)
        return {0: ("h", 4), 14: ("w", 4), 16: ("CANV_W", 1), 17: ("CANV_H", 1),
                19: ("WH*11+20", 1),
                20: ("OUT_W", 1), 21: ("OUT_H", 1), 24: ("WH*4", 1), 25: ("WH*4+20", 1),
                26: ("WH*8+20", 1), 27: ("WH*10+20", 1)}
    if "split16" in name:
        return {0: ("h", 1), 1: ("w", 1), 2: ("tok", 1), 4: ("gx", 1), 5: ("gy", 1),
                28: ("w", 1), 29: ("w", 2), 30: ("FH_TOK", 1)}
    if "c3x_" in name:
        return {0: ("oh", 4), 1: ("ow", 4), 4: ("gx", 1), 19: ("h", 4),
                24: ("oh", 1), 25: ("ow", 1), 26: ("h", 1), 27: ("w", 1)}
    if "f3c" in name and "body_ds" in name:
        return {0: ("h", 4), 1: ("w", 4), 4: ("gx", 1), 21: ("oh", 1), 22: ("ow", 1),
                28: ("h", 2), 29: ("w", 2)}
    if "f2c" in name and "fused" in name:
        return {0: ("h", 2), 14: ("w", 2), 19: ("h", 4),
                24: ("oh", 1), 25: ("ow", 1), 26: ("h", 1), 27: ("w", 1)}
    if "s37_fold" in name:
        # 20/21 are the *frame* extent: the fold's full-resolution store computes its SSBO address
        # as (py*W+px)*4 and the shader reads W/H from p5.xy (f20-215aca679aaa). The deployed
        # version spelled 1280x720 there, so every row but the ones that happen to coincide landed
        # at another row's offset - the vertical banding seen at any extent but 720p.
        return {0: ("h", 2), 14: ("w", 2), 28: ("h", 1), 29: ("w", 1),
                20: ("OUT_W", 1), 21: ("OUT_H", 1)}
    if "vit" in name:
        return {0: ("tok", 1)}
    if "ioport" in name:
        if "block30" in name:
            return {0: ("oh", 1), 1: ("ow", 1), 2: ("OUT_W", 1), 3: ("OUT_H", 1),
                    20: ("OUT_W", 1), 21: ("OUT_H", 1), 24: ("OUT_PIX", 1),
                    25: ("WH*4+20", 1), 26: ("WH*8+20", 1), 27: ("WH*10+20", 1),
                    29: ("OUT_PIX", 1)}
        return {0: ("h", 1), 1: ("w", 1), 2: ("oh", 1), 3: ("ow", 1), 14: ("oh", 1), 15: ("ow", 1)}
    if "t3_" in name:
        return {0: ("FH_H", 1), 1: ("FH_W", 1), 2: ("FH_H", 1), 3: ("FH_W", 1),
                11: ("h", 1), 12: ("w", 1), 20: ("h", 1), 21: ("w", 1)}
    if re.search(r"swin_body_[234]$", name):
        # encoder downsample stages read at the source extent; decoder stages (block >= 22) read at
        # the upsampled one, so the same shader family needs two different scales
        if layer.block <= 8:
            return {0: ("h", 4), 1: ("w", 4), 4: ("gx", 1)}
        return {0: ("oh", 4), 1: ("ow", 4), 4: ("gx", 1)}
    if "swin_body" in name:
        kind = layer.kind if layer else ""
        if kind == "Swin1H":
            return {0: ("h", 4), 14: ("w", 4)}
        return {0: ("h", 4), 1: ("w", 4), 4: ("gx", 1)}
    return {}


def check(passes: list[dict]) -> int:
    specs = {(s.block, s.layer): s for s in G.build_plan(*SRC_SIZE, "small")}
    ok = bad = 0
    rows = []
    for p in passes:
        layer = layer_of(p["name"], specs)
        prob = []
        grid = grid_of(p["name"], layer, specs, SRC_SIZE)
        if grid is None:
            prob.append("no grid rule")
        elif (p["gx"], p["gy"]) != grid:
            prob.append("grid %dx%d != %dx%d" % (p["gx"], p["gy"], grid[0], grid[1]))
        for idx, (expr, div) in fields_of(p["name"], layer).items():
            want = resolve(expr, layer, specs, SRC_CANVAS, SRC_SIZE) // div
            if idx >= len(p["push"]):
                prob.append("push[%d] missing" % idx)
            elif p["push"][idx] != want:
                prob.append("push[%d] %d != %d (%s/%d)" % (idx, p["push"][idx], want, expr, div))
        if prob:
            bad += 1
            rows.append((p["name"], prob[:3]))
        else:
            ok += 1
    print("check at %dx%d (canvas %dx%d): %d pass(es) verified, %d wrong"
          % (*SRC_SIZE, *SRC_CANVAS, ok, bad))
    for nm, prob in rows[:14]:
        print("  %-38s %s" % (nm, "; ".join(prob)))
    return 0 if bad == 0 else 1


class _Unpadded:
    """Stand-in for nr_geometry.Padding that reports the frame as its own canvas."""
    def __init__(self, w: int, h: int) -> None:
        self.width, self.height = w, h
        self.pad_x = self.pad_y = 0


def generate(size: tuple[int, int], src: str, dst: str, no_pad: bool = False,
             canvas_max: tuple[int, int] | None = None) -> int:
    # AMD does not pad: `working_resolution()` is round(render * scale) and the model runs at
    # exactly that extent (windows/src/core/nr_renodx.cpp:1108). pad64 is this port's own
    # conservative habit, so `--no-pad` asks the geometry model for the frame itself.
    if no_pad:
        G.padding = lambda w, h, network="ht": _Unpadded(w, h)
    canvas = size if no_pad else G.working_extent(*size, "small")
    # The ceiling is the BUFFER block copied out of `src`, not a constant: a plan whose runtime
    # buffers were grown by tools/resize_plan_budget.py can hold a larger canvas, and capacity
    # alone does not enter any arithmetic (a 2x oversized plan renders a bit-identical frame).
    budget = canvas_max or SRC_CANVAS
    if canvas[0] > budget[0] or canvas[1] > budget[1]:
        print("refusing %dx%d: canvas %dx%d exceeds the buffer allocation %dx%d "
              "(grow the source with tools/resize_plan_budget.py, then pass --canvas-max)"
              % (*size, *canvas, *budget))
        return 2
    specs = {(s.block, s.layer): s for s in G.build_plan(*size, "small")}
    text = open(src, encoding="utf-8").read().splitlines()
    # How far the source plan's temporal planes already sit: its own HISTORY line declares
    # sourceOffset = 16*src_w*src_h, so read that instead of assuming the shipped 1280x720.
    # Assuming it shifted the offsets twice whenever `src` was itself a generated plan.
    src_hist = 16 * SRC_SIZE[0] * SRC_SIZE[1]
    for line in text:
        if line.startswith("HISTORY") and len(line.split()) > 3:
            src_hist = int(line.split()[3])
            break
    out_lines: list[str] = []
    n_grid = n_push = n_hist = 0
    cur_name: str | None = None      # BIND lines sit between PASS and PUSH, so track it explicitly
    for line in text:
        if line.startswith("PASS"):
            w = line.split()
            cur_name = w[2].strip('"')
            layer = layer_of(cur_name, specs)
            grid = grid_of(cur_name, layer, specs, size)
            if grid is not None:
                w[8], w[9] = str(grid[0]), str(grid[1])
                n_grid += 1
            out_lines.append(" ".join(w))
        elif line.startswith("PUSH") and cur_name is not None:
            name = cur_name
            layer = layer_of(name, specs)
            words = [int(x) for x in line.split()[1:]]
            for idx, (expr, div) in fields_of(name, layer).items():
                if idx < len(words):
                    words[idx] = resolve(expr, layer, specs, canvas, size) // div
                    n_push += 1
            out_lines.append("PUSH " + " ".join(str(x) for x in words))
        elif line.startswith("HISTORY"):
            # The temporal planes are addressed by *byte* offset into frame_out / ioport_images:
            #   sourceOffset      = 16*W*H   (one RGBA f32 image = frame_out's second half)
            #   destinationOffset = 16*W*H + 80   (80 = the 20 control floats that precede it)
            #   bytes             = 16*W*H
            #   controlsOffset    = 16*W*H
            # Leaving them at the installed plan's value was the whole reason a switched frame
            # denoised nothing: the copy read 14.7 MB into a 4.6 MB frame, so every frame's
            # history was stale bytes from the previous extent (or zeros) and the network fell
            # back to its single-frame path.
            w = line.split()
            old_s = src_hist
            new_s = 16 * size[0] * size[1]
            for i in (3, 4, 5, 6):
                v = int(w[i])
                if v >= old_s:
                    w[i] = str(v - old_s + new_s)
            out_lines.append(" ".join(w))
            n_hist += 1
        else:
            out_lines.append(line)
    open(dst, "w", encoding="utf-8", newline="").write("\n".join(out_lines) + "\n")
    print("wrote %s\n  frame %dx%d  canvas %dx%d  grid updated on %d pass(es), %d push words, "
          "%d history offset(s)" % (dst, *size, *canvas, n_grid, n_push, n_hist))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="verify against the shipped 1280x720 plan")
    ap.add_argument("--size", help="target frame size, e.g. 1024x576")
    ap.add_argument("--out", help="output plan path")
    ap.add_argument("--src", default=SRC_PLAN)
    ap.add_argument("--no-pad", action="store_true", help="canvas == frame (no 64-alignment)")
    ap.add_argument("--canvas-max", help="buffer allocation ceiling as WxH, e.g. 1728x960; default is "
                                        "the installed plan's canvas (grow it with tools/resize_plan_budget.py)")
    a = ap.parse_args()
    if a.check:
        return check(load_plan(a.src))
    if not a.size or not a.out:
        ap.error("give --size and --out, or --check")
    w, h = (int(x) for x in a.size.lower().split("x"))
    cm = None
    if a.canvas_max:
        cw, ch = (int(x) for x in a.canvas_max.lower().split("x"))
        cm = (cw, ch)
    return generate((w, h), a.src, a.out, a.no_pad, cm)


if __name__ == "__main__":
    raise SystemExit(main())
