"""Regenerate the Intel port's per-pass dispatch grids for any frame size.

The frozen plan (`PORT_NETWORK_1`) bakes `gx/gy` per pass. This module derives them
from the network's geometry model (`tools/nr_geometry.py`, vendored from the AMD
PyTorch reference), which reproduces the shipped 1280x720 plan exactly for the
Swin/ViT backbone and for four Intel-specific pass families via the rules below.

Self-test: `python tools/gen_grids.py` prints the match rate against the shipped plan.
"""
import importlib.util
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "source/network-fp16-20261027/runtime.template.txt"

spec = importlib.util.spec_from_file_location("nr_geometry", ROOT / "tools/nr_geometry.py")
geom = importlib.util.module_from_spec(spec)
sys.modules["nr_geometry"] = geom
spec.loader.exec_module(geom)

# The Intel port re-tiled four pass families instead of using the layer's window
# grid. Each rule was fitted on the 1280x720 plan and must keep holding for other
# sizes: gx covers the layer's pixel count at a fixed items-per-group, gy is a
# constant split that comes from the channel layout, not from the frame.
FAMILIES = (
    # split blocks: the four layers of one block each tile the same extent
    # differently - Ffwd 16 items/group with a 2-way split, FfwdProj the same
    # grid with 4, Proj 32 items/group with 8. QKVAttn (l2) uses the layer grid.
    (re.compile(r"^split\d+_b\d+_l0$"), 16, 2),
    (re.compile(r"^split\d+_b\d+_l1$"), 16, 4),
    (re.compile(r"^split\d+_b\d+_l3$"), 32, 8),
    # The bottleneck's ViT layers tile the token axis at a fixed items-per-group
    # and keep a channel-derived second dimension. Fitted on 288 tokens (12x24):
    # 288/32=9, 288/96=3, ceil(288/64)=5 - all four reproduce their plan grid.
    (re.compile(r"^vit_block\d+_ffn_expand$"), 32, 32),
    (re.compile(r"^vit_block\d+_ffn_contract$"), 96, 32),
    (re.compile(r"^vit_block\d+_qkv$"), 96, 96),
    (re.compile(r"^vit_block\d+_attention$"), 64, 32),
    # gy=32: projection is a 1024-wide output (weight block 1024x1024 halves),
    # not "both output halves".  64 read past the weights and wrapped onto the
    # next token's row.  See gen_plan_small.py / gen_plan.py.
    (re.compile(r"^vit_block\d+_projection$"), 96, 32),
    (re.compile(r"^ioport_block\d+_"), 32, 2),
)


def grids_for(width, height, passes):
    """Expected (gx, gy) for every (row, name) pair at `width` x `height`."""
    layers = geom.build_plan(width, height)
    out = []
    for row, name in passes:
        # The final ioport pass walks the frame's own pixels, not the padded
        # canvas: 1280*720/32 = 28800 groups. It was the one pass that made the
        # padded/unpadded distinction visible.
        if name == "ioport_block70_2":
            out.append((max(1, width * height // 32), 1))
            continue
        layer = layers[row]
        for pattern, per_group, gy in FAMILIES:
            if pattern.match(name):
                gx = max(1, (layer.w * layer.h + per_group - 1) // per_group)
                out.append((gx, gy))
                break
        else:
            out.append((layer.grid_x, layer.grid_y))
    return out


def read_plan(path=PLAN):
    passes = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.startswith("PASS "):
            continue
        parts = line.split()
        passes.append((int(parts[1]), parts[2].strip('"'), int(parts[8]), int(parts[9])))
    return passes


def main():
    passes = read_plan()
    exact = 0
    misses = []
    for (row, name, gx, gy), want in zip(passes, grids_for(1280, 720, [(r, n) for r, n, _, _ in passes])):
        if (gx, gy) == want:
            exact += 1
        else:
            misses.append((name, row, (gx, gy), want))
    print(f"1280x720: {exact}/{len(passes)} grids reproduced")
    for name, row, got, want in misses:
        print(f"  MISS {name:<28} row{row:>3}  plan {got}  derived {want}")

    for width, height in ((960, 540), (960, 600), (1129, 706), (1280, 800)):
        padded = geom.padding(width, height)
        grids = grids_for(width, height, [(r, n) for r, n, _, _ in passes])
        total = sum(gx * gy for gx, gy in grids)
        print(f"{width}x{height} -> padded {padded.width}x{padded.height}, "
              f"dispatch groups total {total}")


if __name__ == "__main__":
    main()
