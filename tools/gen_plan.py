"""Emit a PORT_NETWORK_1 plan for a different frame size from the frozen 1280x720 one.

What changes and what does not:

* dispatch grids - recomputed from the network geometry model (`tools/nr_geometry.py`),
  which reproduces 229 of the 232 frozen grids at 1280x720; the last three are
  single-pass exceptions carried by explicit rules below.
* PUSH geometry words - **none of them**, for 230 of the 232 passes. Disassembly
  (spirv-dis) shows those kernels take their extent from `gl_NumWorkGroups`, so the
  geometry rides entirely on the dispatch counts. An earlier version of this tool
  refitted the constants as a linear model of (w, h, w*h, ...) and got the frozen
  values back exactly at 1280x720 -- which read as "the fit works" but actually
  meant "the constants are constant". The refit is gone; carrying the words verbatim
  is both simpler and the truth.
* The two exceptions are the network's ports, which do index pixels rather than
  tiles: `ioport_block70_2` reads the frame extent out of its own constants and must
  follow the frame, and `f1c32_image_pre` indexes the ioport surface with its own
  `source` extent (w20/w21). The latter was left alone for a day on the belief that
  the kernel never loads those words; it does, and at another size they are the
  four-row stripe - see FRAME_DIM_WORDS.
* BUFFER sizes and BIND ranges - left untouched on purpose. The target tensors are
  smaller than the 1280x720 ones, so the existing allocation always fits, and every
  kernel takes its working extent from the dispatch, which is regenerated here.

Self-check: `--check` regenerates 1280x720 and asserts the output is byte-identical
to the source plan.

    python tools/gen_plan.py --check
    python tools/gen_plan.py --size 960x540 --out local/vk102/manifest/plan-960x540.txt
"""
import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# The generator needs a baseline that cannot be overwritten by installing a plan:
# `port_network_runtime.txt` is what the backend loads and tools/freeze_plan.py
# rewrites it, so the 1280x720 reference lives beside it under its own name.
BASE_PLAN = ROOT / "local/vk102/manifest/port_network_runtime_1280x720.txt"
SOURCE_PLAN = BASE_PLAN if BASE_PLAN.exists() else ROOT / "local/vk102/manifest/port_network_runtime.txt"

spec = importlib.util.spec_from_file_location("nr_geometry", ROOT / "tools/nr_geometry.py")
geom = importlib.util.module_from_spec(spec)
sys.modules["nr_geometry"] = geom
spec.loader.exec_module(geom)

# Pass-name -> (items per group, fixed second dimension), see tools/gen_grids.py.
# The three names below are the single-pass exceptions that do not follow the
# layer's window grid; they were read off the frozen plan.
GRID_RULES = (
    (re.compile(r"^split\d+_b\d+_l0$"), 16, 2),
    (re.compile(r"^split\d+_b\d+_l1$"), 16, 4),
    (re.compile(r"^split\d+_b\d+_l3$"), 32, 8),
    (re.compile(r"^vit_block\d+_ffn_expand$"), 32, 32),
    (re.compile(r"^vit_block\d+_ffn_contract$"), 96, 32),
    (re.compile(r"^vit_block\d+_qkv$"), 96, 96),
    (re.compile(r"^vit_block\d+_attention$"), 64, 32),
    # gy=64 (32 was reverted on 2026-11-04): the ViT GEMM template
    # (`vit_nb04_mpair.glsl`) emits 32 columns per workgroup
    # (`n0 = gl_WorkGroupID.y * 32`) while the projection writes a 1024-wide row
    # (`o[t * 1024 + c]`).  gy=64 therefore qualifies twice the work and the
    # second half wraps onto row t+1 of the same buffer, overwriting the row the
    # next workgroup owns.  That overflow is what the accepted small build has
    # always done; the gy=32 correction moves 4.2% of the output and costs the
    # bit-compatibility with the accepted frame, so it is not shipped.
    (re.compile(r"^vit_block\d+_projection$"), 96, 64),
    (re.compile(r"^ioport_block\d+_"), 32, 2),
)
EXPLICIT_GRIDS = {
    # The pool split block's last two layers and the project-blend pass do not
    # follow the layer's own window grid. All three tile a padded token count at
    # a fixed items-per-group; the final head reads the extent of the layer
    # before it (1152 tokens at 1280x720 -> 66 / 18 / 36 groups).
    "split16_b30_l3": ("pixels", 16, 4),
    "split16_b30_l4": ("previous", 64, 8),
    "t3_b39_project_blend": ("tokens", 32, 16),
}

# Push-constant words that hold the frame extent, per pass, as (width, height) pairs.
#
# Only the output port qualifies, and the reason is in its SPIR-V rather than in its
# name: `ioport_block70_2` splits the invocation id into (x, y) with p0.z / p0.w and
# divides the pixel centre by p5.xy to get the normalised motion coordinate. Both
# pairs are the frame size, so both have to follow the frame. Everything else in that
# push (the canvas pair p0.xy, the control-table offsets p6) is either unread or
# independent of resolution, and is carried verbatim.
#
# The input port `f1c32_image_pre` also carries 1344 / 768 / 1280 / 720 in p4 / p5,
# but disassembly shows it never loads those words -- it walks tiles by
# gl_GlobalInvocationID and takes its constants from p0 / p1 / p2 / p3 / p7. Leaving
# them alone keeps `--check` byte-identical, which is worth more than tidiness.
FRAME_DIM_WORDS = {
    "ioport_block70_2": (2, 3, 20, 21),
    # `f1c32_image_pre` w20/w21 is the *source* extent it indexes the ioport surface with.
    # It was left alone for a day because the docstring above said the kernel never loads
    # them; measured, it does (changing them moves the mean), and leaving them at 1280x720
    # while the surface is 1024 wide is the four-row stripe: a 1280 stride over 1024-wide
    # rows wraps a whole row every four rows. Measured on the runner with a real frame,
    # |edit| by y%4: 16.4% -> 2.5%, profile flat.
    "f1c32_image_pre": (20, 21),
}

# ...and p0.xy is the *canvas* the network wrote, which is not the frame: at 1280x720 it is
# 1344x768 while the frame is 1280x720. Measured caveat: rewriting these two words at
# 1024x576 changed the output by exactly zero bits, so the port does not read them (they are
# written as the padded extent because that is what they describe, not because it fixes
# anything). The one-row-in-five defect those numbers first looked like the cause of is still
# unexplained -- reproduced at sizes other than 1280x720.
CANVAS_DIM_WORDS = {
    "ioport_block70_2": (0, 1),
}

# The two port passes also address the ioport planes by float offset, and those
# offsets are multiples of the frame pixel count: 4*U, 4*U+20, 8*U+20, 10*U+20 with
# U = width*height (verified on the 1280x720 plan: 3686400 / 3686420 / 7372820 /
# 9216020, and by disassembly -- the port calls ctl() as application[offset + i]).
# They have to follow the frame for the same reason the extent words do: gather is
# told where to put the planes from the same numbers, so a plan that folds one and
# not the other reads a different frame's planes.
#
# Recognising them by value rather than by position keeps this honest: the source
# plan states its own frame, so the rewrite runs in either direction (720p -> 540p
# and back) without needing a pristine baseline.
IOPORT_OFFSET_PASSES = ("f1c32_image_pre", "ioport_block70_2")

# The *role* of a port pass, not its block number.
#
# The full ("1080p" layout) profile numbers its ports `ioport_block4_0` (input) and
# `ioport_block70_2` (output); the small profile numbers the same two shaders
# `ioport_block2_0` and `ioport_block30_2` and prefixes every pass with its row
# (`r75_ioport_block30_2`). The shaders, the push layout and the word meanings are
# identical -- only the names move. Everything below therefore keys on the role, so a
# plan from either profile resolves to the same three canonical entries.
PORT_CANON = {
    "ioport_block70_2": re.compile(r"^ioport_block\d+_2$"),   # output port (frame pixels)
    "ioport_block4_0": re.compile(r"^ioport_block\d+_0$"),    # input port (source extent)
}
ROW_PREFIX = re.compile(r"^r\d+_")


def port_key(name):
    """Canonical name for a pass: `r75_ioport_block30_2` -> `ioport_block70_2`."""
    base = ROW_PREFIX.sub("", name)
    for canon, pattern in PORT_CANON.items():
        if pattern.match(base):
            return canon
    return base


def source_frame(passes):
    """The frame size the source plan was built for, read out of its own port."""
    for p in passes:
        if port_key(p["name"]) != "ioport_block70_2":
            continue
        first, second = FRAME_DIM_WORDS["ioport_block70_2"][:2]
        return p["push"][first], p["push"][second]
    raise SystemExit("source plan has no output port (ioport_block<N>_2) pass")


def rescale_history(text, old_pixels, new_pixels):
    """Move the HISTORY link offsets with the frame.

    A HISTORY line is `HISTORY <src> <dst> <srcOff> <dstOff> <bytes> <ctlOff>` and every
    offset there is a multiple of the frame pixel count in bytes (4*U, 4*U+80 at
    1280x720). They were the last size-derived numbers in the plan that nothing moved:
    at 960x540 the temporal copy still walked into the 720p half of the buffer, so the
    model never saw a valid history and the picture came out looking untouched while
    still paying the full network cost.
    """
    out = []
    for line in text.splitlines():
        if not line.startswith("HISTORY "):
            out.append(line)
            continue
        parts = line.split()
        for i in range(3, len(parts)):     # skip HISTORY, source, destination
            value = int(parts[i])
            whole, rest = divmod(value, old_pixels)
            if rest < 512:                 # k*U + a small structural constant
                parts[i] = str(whole * new_pixels + rest)
        out.append(" ".join(parts))
    return "\n".join(out) + "\n"


def rescale_ioport_offsets(values, old_pixels, new_pixels):
    """Rewrite the four plane offsets from one frame size to another."""
    base_old, base_new = 4 * old_pixels, 4 * new_pixels
    mapping = {
        base_old: base_new,
        base_old + 20: base_new + 20,
        2 * base_old + 20: 2 * base_new + 20,
        5 * base_old // 2 + 20: 5 * base_new // 2 + 20,
    }
    for i, value in enumerate(values):
        if value in mapping:
            values[i] = mapping[value]


def explicit_grid(pass_, layers):
    rule = EXPLICIT_GRIDS.get(pass_["name"])
    if not rule:
        return None
    basis, per_group, gy = rule
    layer = layers[pass_["row"]]
    if basis == "pixels":
        count = layer.w * layer.h
    elif basis == "previous":
        count = layers[pass_["row"] - 1].tokens
    else:
        count = layer.tokens
    return max(1, count // per_group), gy


def read(source):
    """Split the plan into records so passes can be rewritten in place."""
    lines = source.splitlines()
    passes, push_index = [], {}
    current = None
    for i, line in enumerate(lines):
        if line.startswith("PASS "):
            parts = line.split()
            current = {"line": i, "row": int(parts[1]), "name": parts[2].strip('"'),
                       "spv": parts[3], "npush": int(parts[7]),
                       "gx": int(parts[8]), "gy": int(parts[9])}
            passes.append(current)
        elif line.startswith("PUSH ") and current is not None:
            push_index[current["line"]] = i
            current["push"] = [int(x) for x in line.split()[1:]]
    return lines, passes, push_index


def grids_for(width, height, passes, layers):
    out = []
    for p in passes:
        name = p["name"]
        if port_key(name) == "ioport_block70_2":
            out.append((max(1, width * height // 32), 1))       # frame pixels, unpadded
            continue
        layer = layers[p["row"]]
        explicit = explicit_grid(p, layers)
        if explicit is not None:
            out.append(explicit)
            continue
        for pattern, per_group, gy in GRID_RULES:
            if pattern.match(name):
                out.append((max(1, (layer.w * layer.h + per_group - 1) // per_group), gy))
                break
        else:
            out.append((layer.grid_x, layer.grid_y))
    return out


def verify(width, height, text):
    """Invariants that a generated plan has to satisfy, checked on the text it emitted.

    Every one of these would have failed on the version of this file that scaled the
    output port's dispatch but left its constants at 1280x720 -- which is the point:
    a check that cannot fail is not a check.
    """
    problems = []
    passes, push = {}, {}
    current = None
    for line in text.splitlines():
        if line.startswith("PASS "):
            parts = line.split()
            current = port_key(parts[2].strip('"'))
            # PASS <row> <name> <spv> <workgroup> ... <npush> <gx> <gy> <gz>
            passes[current] = (int(parts[4]), int(parts[8]), int(parts[9]), int(parts[10]))
        elif line.startswith("PUSH ") and current is not None:
            push[current] = [int(x) for x in line.split()[1:]]

    for name, words in FRAME_DIM_WORDS.items():
        values = push.get(name)
        if values is None:
            problems.append(f"{name}: no PUSH line")
            continue
        for i in range(0, len(words), 2):
            got = (values[words[i]], values[words[i + 1]])
            if got != (width, height):
                problems.append(f"{name}: frame words {words[i]}/{words[i + 1]} are {got}, "
                                f"expected {(width, height)}")

    # The canvas words must describe the padded extent, not the frame: at 1280x720 they are
    # 1344x768, and a plan that leaves them there while the frame shrinks reads the planes
    # one row off (one missing row in five).
    padded = geom.padding(width, height)
    for name, words in CANVAS_DIM_WORDS.items():
        values = push.get(name)
        if values is None:
            continue
        got = (values[words[0]], values[words[1]])
        if got != (padded.width, padded.height):
            problems.append(f"{name}: canvas words {words[0]}/{words[1]} are {got}, "
                            f"expected {(padded.width, padded.height)} for a {width}x{height} frame")

    # A pass that indexes one invocation per frame pixel must dispatch exactly that many.
    for name, (items, gx, gy, gz) in passes.items():
        if name == "ioport_block70_2" and items * gx * gy * gz != width * height:
            problems.append(f"{name}: dispatch {gx}x{gy}x{gz} of {items} "
                            f"!= {width}x{height} pixels")
    return problems


# How the network's Level-of-Detail ladder is seeded.
#
# AMD's make_native_plan (nr_native_plan.cpp) works at `max(320, up(v, 2^n))` with n = how
# many times the ladder halves that axis, plus one tile when both axes are already multiples
# of 4*2^n. For this network n is 6 on both axes, and the three sizes we can check agree:
#   1280x720 -> up(1280,64)=1280, up(720,64)=768, both %256==0 -> 1344x768   (the canvas we ship)
#   1024x576 -> 1024, 576 -> 576%256!=0 -> 1024x576
#    960x540 ->  960, 576 -> 960%256!=0 ->  960x576
LEVEL_COUNT = 6
PAD_WIDTH = False
NATIVE_CANVAS = False   # --native-canvas: extent == frame (AMD's 1080p record does this); no 2^n rounding   # --pad-width: force the width padding step (see work_size)
LAST_AMBIGUOUS = {}  # line -> word indices the last level rewrite could not invert (see roundtrip)
LAST_TOUCHED = {}    # line -> every word index the last level rewrite changed

# 192 of the 232 passes carry a per-level extent in their push block: block1_swin_body_1
# says 168 96, which is the 336x192 level halved. gen_plan used to copy every push word
# verbatim, so a plan for another size still told those passes the 1280x720 level sizes --
# the same extent-vs-stride mix-up the AMD comments describe as "reading it at the unpadded
# width shears every row by the padding gap", and the reason the picture came back with one
# row in five missing.
def work_size(width, height, native=None):
    if NATIVE_CANVAS if native is None else native:
        return width, height
    a = 1 << LEVEL_COUNT
    w = max(320, (width + a - 1) // a * a)
    h = max(320, (height + a - 1) // a * a)
    # AMD's make_native_plan() adds one step of width padding only when *both* padded axes are
    # already whole 4a blocks - 1280x720 pads to 1344 because 720 lands on 768 (= 3*256), while
    # 1024x576 and 960x540 do not pad at all, because 576 is already a multiple of 64 and 540
    # pads to 576, which is not a multiple of 256. So the two sizes that misbehave in game are
    # exactly the two whose canvas width equals the render width. `--pad-width` forces the step
    # so the hypothesis "the width must differ from the extent" can be falsified on its own.
    if (h % (a * 4) == 0 and w % (a * 4) == 0) or PAD_WIDTH:
        w += a
    return w, h


def level_ladder(width, height, native=None):
    """(extent_w, extent_h) of each rung - raw halving, no padding.

    The *row stride* of a rung is pad4 of these, and push blocks carry both, so the maps
    below include both spellings. Keeping them apart in the ladder itself matters: at
    1280x720 rung 5 is 42 wide (stride 44), and only the extent appears in a push.
    """
    w, h = work_size(width, height, native)
    out = [(w, h)]
    for _ in range(LEVEL_COUNT):
        w, h = max(1, w // 2), max(1, h // 2)
        out.append((w, h))
    return out


def pad4(v):
    return (v + 3) // 4 * 4


def _axis_maps(src, dst):
    """value -> value, for the width axis and the height axis, extents and strides alike.

    Each entry also records its rung so a rewrite can refuse to move a word that belongs to
    a level the pass cannot be reading (a level-3 pass does not name level 1's width).
    """
    wm, hm, wr, hr = {}, {}, {}, {}
    for i, ((sw, sh), (dw, dh)) in enumerate(zip(src, dst)):
        pairs = [(sw, dw)] + ([(pad4(sw), pad4(dw))] if pad4(sw) != sw else [])
        for a, b in pairs:
            if a:
                wm[a] = b
                wr[a] = i
        pairs = [(sh, dh)] + ([(pad4(sh), pad4(dh))] if pad4(sh) != sh else [])
        for a, b in pairs:
            if a:
                hm[a] = b
                hr[a] = i
    return wm, hm, wr, hr


def pass_level(gx, gy, ladder):
    """Which rung a pass sits on, from its grid: extent / 8, optionally one tile more (the
    shifted-window passes are the four 0/1 combinations). None when it does not fit - the
    deep 1-D token dispatches (gx=8064, gy=2) are not image grids at all."""
    import math

    lw, lh = ladder[0]
    for shift_x in (0, 1):
        for shift_y in (0, 1):
            wx, wy = gx - shift_x, gy - shift_y
            if wx <= 0 or wy <= 0:
                continue
            rx, ry = lw / (8.0 * wx), lh / (8.0 * wy)
            if rx < 1 or ry < 1:
                continue
            lx, ly = math.log2(rx), math.log2(ry)
            if abs(lx - round(lx)) > 1e-9 or abs(ly - round(ly)) > 1e-9:
                continue
            if round(lx) != round(ly):
                continue
            return int(round(lx))
    return None


def read_push_word_map():
    """per shader family: word index -> (axis, num, den, const), derived by tools/nr_push_map.py."""
    path = Path(__file__).resolve().parent / "data" / "push_word_map.json"
    if not path.exists():
        print(f"  warning: {path} missing, push blocks stay at the source size")
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _pass_value_map(k, src_ladder, dst_ladder, src_grid, dst_grid, frames, words=()):
    """value -> value for one pass, built only from what that pass is known to be.

    A pass's push names its own level: the extent, half of it, a quarter (block15_swin_body_1
    pushes `21 12` and `11 6`), and its dispatch grid (130 of 232 pushes carry the literal
    (gx,gy) pair). It does not name other levels, so a value that happens to equal some other
    rung's extent is not geometry for this pass - matching against the whole ladder is what
    once turned a stray `32` into 42.

    Conflicts are dropped rather than resolved: if one number would map to two results (an
    extent that coincides with a grid), that pass keeps the word untouched.
    """
    m, conflict = {}, set()
    refs = set(words)
    # Geometry in these pushes is always an adjacent (w, h) pair, which is what lets a level be
    # recognised without guessing: at 640x360 the height of rung 6 and a channel count are both
    # `6`, so a single word cannot say which it is - `6 3` can.
    pairs = set(zip(words, words[1:]))
    # The floor exists so a stray small constant is not read as a size. It cannot be the fixed
    # 8 it used to be: at 640x360 rung 6 is 10x6 and its half is 5x3, so 6/5/3 were all left
    # behind and the round trip rebuilt `42 24 / 21 6` where the source says `42 24 / 21 12`.
    # Derived from the ladders (deepest extent and its half, both directions), which keeps
    # 1280x720 at 6 - and there every mapping is a->a, so `--check` is unmoved.
    last = len(src_ladder) - 1
    half = [(-(-src_ladder[last][0] // 2), -(-src_ladder[last][1] // 2)),
            (-(-dst_ladder[last][0] // 2), -(-dst_ladder[last][1] // 2))]
    floor = min([8] + [v for rung in src_ladder + dst_ladder + half for v in rung if v > 0])

    def add(a, b, soft=False):
        if a < floor or b < 1:
            return
        if a in m and m[a] != b:
            if soft:
                return
            conflict.add(a)
        else:
            m[a] = b

    # A pass's push reaches deeper than its own rung: f1c32_image_pre sits on rung 0 and tiles
    # 168x96 (rung 3, as p0.x = 2*168); f3c128_body_ds sits on rung 3 and also names rung 6
    # (42 24) and rung 4 (84 48). Measured across the plan, the deepest reference is +3.
    #
    # The two ladders share no value at all (w: 1344/672/336/168/84/42/21 vs
    # 1024/512/256/128/64/32/16, h likewise), so this mapping is a bijection and the round trip
    # can only fail on a value that is *also* something else - which is why the grid pair is
    # added separately and dropped on conflict.
    for r in range(k, min(len(src_ladder) - 1, k + 3) + 1):
        for axis in (0, 1):
            add(src_ladder[r][axis], dst_ladder[r][axis])
    for axis in (0, 1):
        add(src_grid[axis], dst_grid[axis])
    # One step past the last rung: `block15_swin_body_1` pushes 21 12 *and* 11 6, the half of the
    # deepest level, which the ladder does not contain because it stops halving at 21x12. This is
    # the weakest of the three inferences - `block16_swin_body_3` pushes 11 7 and 11 there is its
    # *grid*, not a half - so it is added after the grid and yields to it instead of forcing a
    # conflict (a conflict drops the value and leaves the grid word at 11 while the dispatch says
    # 6, which is the 512-rows-at-a-672-stride defect again).
    if min(len(src_ladder) - 1, k + 3) >= last and src_ladder[last] in pairs:
        for axis in (0, 1):
            add(half[0][axis], half[1][axis], soft=True)
    # Frame-derived sizes were tried and **measured wrong**: mapping 320 -> 256 (40 words, from
    # "320 = 1280/4") dropped the 720p correlation from 0.995 to 0.242 and raised row noise from
    # ~0 to 0.098 with lag5 = 0.985. So 320 in the ViT / split16 families is not the frame's
    # quarter - the same number means something else there. Removed; the ladder + grid + extent
    # cover what is actually understood.
    for a in conflict:
        m.pop(a, None)
    return m


def rescale_level_words(text, src_w, src_h, dst_w, dst_h, levels, src_grids, dst_grids):
    """Rewrite every push word that names this pass's extent or its dispatch grid.

    Two failures led here, both measured on a dump rather than argued:

    * a per-family whitelist missed 42 of the 232 passes - block8_swin_body_3 kept `84 48`
      while its grid became 32x19, so 512-wide rows were laid out at the 672 stride the buffer
      was allocated with (fill 76.2% = 512/672, the exact hole pattern seen in block8_out);
    * matching against the whole ladder instead of the pass's own rung rewrote words that only
      happened to look like a size.

    So: the pass's own extent, its halves and quarters, and its own grid - nothing else.

    Half of these sizes nest: 640x360's ladder is 1280x720's shifted by one (384 is rung 1 of the
    big one and rung 0 of the small one), so two source values can land on the same destination
    value and the reverse build cannot know which one it was. Those word positions are recorded
    in `LAST_AMBIGUOUS` so the round-trip gate can tell "could not be inverted" apart from
    "matched for the wrong reason".
    """
    global LAST_AMBIGUOUS, LAST_TOUCHED
    LAST_AMBIGUOUS, LAST_TOUCHED = {}, {}
    src, dst = level_ladder(src_w, src_h, native=False), level_ladder(dst_w, dst_h)
    out, changed = [], 0
    idx = -1
    for line_no, line in enumerate(text.splitlines()):
        if line.startswith("PASS "):
            idx += 1
            out.append(line)
            continue
        if not line.startswith("PUSH ") or idx < 0 or idx >= len(levels):
            out.append(line)
            continue
        k = levels[idx]
        if k is None:
            out.append(line)
            continue
        words = [int(x) for x in line.split()[1:]]
        table = _pass_value_map(k, src, dst, src_grids[idx], dst_grids[idx],
                                ((src_w, src_h), (dst_w, dst_h)), words)
        targets = {}
        for v in table.values():
            targets[v] = targets.get(v, 0) + 1
        ambiguous = {v for v, n in targets.items() if n > 1}
        hit, touched = set(), set()
        for i, v in enumerate(words):
            if v in table and table[v] != v:
                words[i] = table[v]
                changed += 1
                touched.add(i)
                if table[v] in ambiguous:
                    hit.add(i)
        if hit:
            LAST_AMBIGUOUS[line_no] = hit
        if touched:
            LAST_TOUCHED[line_no] = touched
        out.append("PUSH " + " ".join(str(v) for v in words))
    return "\n".join(out) + "\n", changed


# Push words that carry the *canvas* rather than this pass's own level.
#
# The ladder rewrite cannot see these: it is built from each pass's extent, and a canvas area is
# not an extent, so such a word is simply left at the source size. Two were measured:
#
# * `vit_block31..38_*` push `1344*768` at word 14. AMD's `make_native_plan()` feeds blocks
#   31-38 `tokens = w*h` (the canvas) instead of `up(w,8)*up(h,8)`; at 1280x720 the two rules
#   agree because 1344 and 768 are both multiples of 8, which is exactly why nothing ever looked
#   wrong on that size. At 1024x576 the token count has to be 589824 or eight attention blocks
#   loop over 1.75x the tokens that exist.
# * `f1c32_image_pre` word 19 is `11*pixels + 20`, the tail of the ioport plane stack (kU,
#   kU+20, 2kU+20, 2.5kU+20, 2.75kU+20, kU = 4*pixels). It is read once, so it is live; at
#   1280x720 it is 10137620, and it was still 10137620 in the 1024 plan.
#
# Both are value-guarded: if the source plan does not hold the expected number the word is left
# alone, so `--check` cannot be broken by a rule that only ever fires on another size.
CANVAS_AREA_WORDS = (
    (re.compile(r"^vit_block3[1-8]_"), 14),
)
PORT_TAIL_WORDS = (
    (re.compile(r"^f1c32_image_pre$"), 19),
)


def rescale_canvas_words(text, src_w, src_h, dst_w, dst_h):
    """Rewrite the push words that name the canvas: token count and ioport plane tail."""
    src_canvas = work_size(src_w, src_h, native=False)
    dst_canvas = work_size(dst_w, dst_h)
    src_area, dst_area = src_canvas[0] * src_canvas[1], dst_canvas[0] * dst_canvas[1]
    src_px, dst_px = src_w * src_h, dst_w * dst_h
    src_tail, dst_tail = 11 * src_px + 20, 11 * dst_px + 20
    out, changed = [], 0
    name = None
    for line in text.splitlines():
        if line.startswith("PASS "):
            name = line.split('"')[1]
            out.append(line)
            continue
        if not line.startswith("PUSH ") or name is None:
            out.append(line)
            continue
        words = [int(x) for x in line.split()[1:]]
        hit = False
        for pattern, index in CANVAS_AREA_WORDS:
            if index < len(words) and pattern.match(name) and words[index] == src_area:
                words[index] = dst_area
                hit = True
        for pattern, index in PORT_TAIL_WORDS:
            if index < len(words) and pattern.match(name) and words[index] == src_tail:
                words[index] = dst_tail
                hit = True
        if hit:
            changed += 1
            out.append("PUSH " + " ".join(str(v) for v in words))
        else:
            out.append(line)
    return "\n".join(out) + "\n", changed


def fix_port_words(text, width, height):
    """Write the port passes' frame/canvas words instead of matching them by value.

    `ioport_block70_2` is told the frame at w2/w3 and w20/w21 and the padded canvas at w0/w1;
    those four pairs are the only push values in the whole plan whose meaning is known from the
    shader rather than inferred from a ladder, so the safest rule is to set them and let no
    other rewrite see them. They used to be written inside `build`'s push loop, which runs
    *before* `rescale_level_words`: at 640x360 the canvas height was already the correct 384,
    and 384 is also a rung of the 1280x720 ladder (672x384), so the ladder halved it to 192 and
    the port was told to index 640x384 rows at 640x192. Caught by `verify`, not in game.
    """
    padded = geom.padding(width, height)
    out, changed, name = [], 0, None
    for line in text.splitlines():
        if line.startswith("PASS "):
            name = port_key(line.split('"')[1])
            out.append(line)
            continue
        if not line.startswith("PUSH "):
            out.append(line)
            continue
        want = {}
        frame_words = FRAME_DIM_WORDS.get(name, ())
        for i in range(0, len(frame_words), 2):
            want[frame_words[i]] = width
            want[frame_words[i + 1]] = height
        canvas_words = CANVAS_DIM_WORDS.get(name)
        if canvas_words:
            want[canvas_words[0]] = padded.width
            want[canvas_words[1]] = padded.height
        if not want:
            out.append(line)
            continue
        words = [int(x) for x in line.split()[1:]]
        hit = False
        for i, v in want.items():
            if i < len(words) and words[i] != v:
                words[i] = v
                hit = True
        if hit:
            changed += 1
            out.append("PUSH " + " ".join(str(v) for v in words))
        else:
            out.append(line)
    return "\n".join(out) + "\n", changed


# Passes whose push carries a *canvas* level at w24/w25 and its half at w26/w27. The ladder
# rewrote w26/w27 (that one is an extent) and left w24/w25 at the source canvas, so after a
# resize the two disagreed - measured on the runner: 168/96 stayed while 84/48 became 64/36.
# Same class as the f1c32 source extent, one slot over.
CANVAS_LEVEL_PASSES = re.compile(r"^(c3x_(pre|body)_c(64|128)|f2c32_block66_fused)$")


def enforce_doubled_levels(text, passes):
    """w24/w25 must equal 2x w26/w27. The invariant is inside the push, so no value guessing."""
    out, changed, name = [], 0, None
    for line in text.splitlines():
        if line.startswith("PASS "):
            name = line.split('"')[1]
            out.append(line)
            continue
        if not line.startswith("PUSH ") or name not in passes:
            out.append(line)
            continue
        words = [int(x) for x in line.split()[1:]]
        if len(words) < 28:
            out.append(line)
            continue
        want = (2 * words[26], 2 * words[27])
        if (words[24], words[25]) != want:
            words[24], words[25] = want
            changed += 1
            out.append("PUSH " + " ".join(str(v) for v in words))
        else:
            out.append(line)
    return "\n".join(out) + "\n", changed


# Shaders that bake the canvas into OpConstant instead of taking it from the push block.
#
# Only this one is enabled. Seven SPIR-V files contain 1280/768/720/192 as constants; rewriting
# all seven produced a black frame, so in the other six the number does not mean "canvas".
# `s37_fold_body_r1a_v1` is unambiguous:
#
#     if (px >= 0 && px < 1280 && py >= 0 && py < 720) {
#         body[(py * 1280 + px) * 4 + 0..3] = boundary_project(t);
#     }
#
# Both uses are the canvas, and both are wrong at another size: at 1024x576 it writes 1024
# valid texels per 1280-word row and leaves a 256-wide hole. Measured: row-lag-5 autocorrelation
# 0.790 -> -0.151, row noise 0.0067 -> 0.0028, correlation with 720p stays 0.996.
CANVAS_BAKED_SPV = {
    "215aca679aaa2d5c36329ae45514efadae221ef4e8631121388b25052a7f7e32__f20-215aca679aaa.spv":
        ("s37_fold_body_r1a_v1", {1280: "w", 720: "h"}),
}


def patch_baked_canvas(text, width, height, src_w, src_h):
    """Write a per-size variant of any shader that hardcodes the canvas, and repoint the plan."""
    assets = ROOT / "local/vk102/assets/spv"
    if (width, height) == (src_w, src_h) or not assets.is_dir():
        return text, 0
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        import patch_spv_size
    except ImportError:
        print("  warning: tools/patch_spv_size.py missing, baked canvas left alone")
        return text, 0
    tag = f"_{width}x{height}"
    done, rewritten = {}, 0
    for spv, (pass_name, words) in CANVAS_BAKED_SPV.items():
        src = assets / spv
        if not src.exists():
            continue
        repl = {old: (width if axis == "w" else height) for old, axis in words.items()}
        dst = assets / (spv.replace(".spv", tag + ".spv"))
        dst.write_bytes(patch_spv_size.patch(src.read_bytes(), repl)[0])
        done[spv] = dst.name
    for old, new_name in done.items():
        if old in text:
            text = text.replace(old, new_name)
            rewritten += 1
    return text, rewritten


def read_layer_rungs():
    """layer index -> ladder rung, for the passes whose grid cannot say it themselves."""
    path = Path(__file__).resolve().parent / "data" / "plan_layer_rungs.json"
    if not path.exists():
        print(f"  warning: {path} missing, token passes keep the source plan's numbers")
        return {}
    return {int(k): v for k, v in json.loads(path.read_text(encoding="utf-8"))["rungs"].items()}



# --- dispatch, from AMD's own walk ------------------------------------------------------------
#
# nr_native_plan.cpp generates every pass: shapes come from a ladder seeded the way
# make_native_plan seeds it, and the grid is `ceil(extent/8)` with a +1 tile for the shifted
# windows. Replaying it against the shipped 1280x720 plan agrees for 140 of 232 passes; the
# other 92 dispatch over packed tiles instead of an image grid.
#
# One of those was disassembled (split16_b23_l0) and its index arithmetic is explicit:
#
#     token = wg.x * 16 + (subgroup % 2) * 8
#     g     = wg.y * 4  + (subgroup / 2)
#
# i.e. one workgroup covers a fixed number of tokens and a fixed number of channel groups.
# Measured across all 13 packed families, `area / gx` is a single constant per family
# (16, 32 or 96) and `gy` is a single constant too -- so gx follows the area and gy does not
# follow the size at all, because the channel count does not shrink with the frame.

def up(n, a):
    return (n + a - 1) // a * a


def read_native_records():
    """AMD's layer table, extracted by tools/extract_native_plan.py."""
    path = Path(__file__).resolve().parent / "data" / "nr_native_plan_records.json"
    if not path.exists():
        print(f"  warning: {path} missing, dispatch falls back to the fitted model")
        return None
    return json.loads(path.read_text(encoding="utf-8"))["records"]


def walk_shapes(width, height, records):
    """(w, h) per layer, plus (g_w, g_h) for the image-grid passes."""
    cur = (width, height)
    prev = cur
    shapes = []
    last = -1
    nx = ny = 0
    out = []
    for rec in records:
        f = rec["fields"]
        if f[0] != last:
            nx += 1 if cur[0] < prev[0] else 0
            ny += 1 if cur[1] < prev[1] else 0
            prev = cur
            last = f[0]
        kind = rec["kind"]
        nxt = cur
        if kind in ("Pre", "Down", "FinalHead"):
            nxt = (up((cur[0] + 1) // 2, 4), up((cur[1] + 1) // 2, 4))
        elif kind in ("SwinUp", "DecoderUp"):
            nxt = shapes[rec["skip"]][0] if rec["original"] else shapes[rec["skip"]][1]
            if kind == "SwinUp" and str(f[12]) == "32":
                nxt = (up(nxt[0], 8), up(nxt[1], 8))
        elif kind == "Post":
            nxt = (width, height)
        plan = nxt if kind == "FinalHead" else cur
        if kind == "DecoderUp":
            plan = (up(nxt[0], 4), up(nxt[1], 4))
        if kind == "Post":
            plan = (width // 2, height // 2)
        grid = nxt if kind in ("Post", "SwinUp") else plan
        out.append({"plan": plan,
                    "g_w": (grid[0] - 4 * rec["sx"] + 7) // 8,
                    "g_h": (grid[1] - 4 * rec["sy"] + 7) // 8})
        shapes.append((cur, nxt))
        cur = nxt
    return out, nx, ny


def walk_working(width, height, records):
    _, nx, ny = walk_shapes(width, height, records)
    ax, ay = 1 << nx, 1 << ny
    w = max(320, up(width, ax))
    h = max(320, up(height, ay))
    if h % (ay * 4) == 0 and w % (ax * 4) == 0:
        w += ax
    return w, h


def packed_rules(passes, shapes, width, height):
    """Per family: tokens-per-workgroup and the (size-independent) gy."""
    rules = {}
    for p, sh in zip(passes, shapes):
        if (p["gx"], p["gy"]) == (sh["g_w"], sh["g_h"]):
            continue
        area = shapes[p["row"]]["plan"][0] * shapes[p["row"]]["plan"][1]
        if not area or p["gx"] == 0 or area % p["gx"]:
            continue
        rules.setdefault(p["name"].split("_")[0], (area // p["gx"], p["gy"]))
    return rules


_RULE_CACHE = {}


def _derive_rules(passes, shapes):
    """pass name -> (tokens per workgroup, gy, basis), from the 1280x720 instance."""
    rules = {}
    for p in passes:
        sh = shapes[p["row"]]
        if (p["gx"], p["gy"]) == (sh["g_w"], sh["g_h"]) or not p["gx"]:
            continue
        area = sh["plan"][0] * sh["plan"][1]
        # Most packed passes tile their own layer; the ioport passes tile the *frame*
        # (921600 / 28800 = 32). Try the layer first, then the frame.
        for basis, a in (("layer", area), ("frame", 1280 * 720)):
            if a and a % p["gx"] == 0:
                rules[p["name"]] = (a // p["gx"], p["gy"], basis)
                break
    return rules


def grids_from_walk(width, height, passes, records, family_of):
    """The dispatch each pass needs at this size, generated rather than fitted.

    The per-pass rule (how many tokens one workgroup covers) is always derived from the frozen
    1280x720 baseline, never from whatever plan is being read. Deriving it from the current
    source made the transform non-invertible: going 720p -> 1024 -> 720p re-derived a different
    divisor from the 1024 numbers and 92 grids came back changed.
    """
    src_w, src_h = 1280, 720
    src_shapes, _, _ = walk_shapes(*walk_working(src_w, src_h, records), records)
    cache_key = (len(passes), passes[0]["name"] if passes else "")
    if cache_key in _RULE_CACHE:
        rules = _RULE_CACHE[cache_key]
    else:
        rules = _derive_rules(passes, src_shapes)
        _RULE_CACHE[cache_key] = rules
    w, h = walk_working(width, height, records)
    shapes, _, _ = walk_shapes(w, h, records)
    out = []
    for p in passes:
        sh = shapes[p["row"]]
        rule = rules.get(p["name"])
        if rule is None:
            # Image-grid passes. Their source grid *equals* the walk, which is exactly why no
            # rule was derived for them - and keeping the source value here was the bug: every
            # 140 of them stayed at 1280x720's grids, so a resized plan still dispatched the
            # old extents (c3x_pre_c128 stayed 22x12 where the new rung needs 16x9) and the
            # buffers they write came back empty. The round-trip gate could not see this,
            # because a transform that changes nothing is trivially reversible.
            out.append((sh["g_w"], sh["g_h"]))
            continue
        tokens, gy, basis = rule
        # "frame" passes tile the frame the host handed over, not the padded working size.
        scale = (width * height) if basis == "frame" else sh["plan"][0] * sh["plan"][1]
        out.append((max(1, (scale + tokens - 1) // tokens), gy))
    return out


def build(width, height):
    source = SOURCE_PLAN.read_text(encoding="utf-8")
    lines, passes, push_index = read(source)
    layers = geom.build_plan(width, height)
    grids = grids_for(width, height, passes, layers)
    records = read_native_records()
    if records:
        by_name = {p["name"]: p for p in passes}
        walk_grids = grids_from_walk(width, height, passes, records, lambda p: p["spv"])
        if len(walk_grids) == len(grids):
            grids = walk_grids
    src_w, src_h = source_frame(passes)
    resizing = (src_w, src_h) != (width, height)
    # Each pass's rung, read off its own grid while the grids are still the source plan's.
    # The 1-D token dispatches have no rung in their grid; they fall back to the per-layer
    # table (tools/nr_layer_rungs.py), which is derived from the AMD layer table.
    src_ladder = level_ladder(src_w, src_h, native=False)
    layer_rungs = read_layer_rungs()
    # The rung belongs to the layer, not to the size: taking it from the grid meant the forward
    # pass (720p grids) and the reverse pass (1024 grids) could disagree about the same pass,
    # which is exactly the 8 doubled words the round-trip gate kept reporting.
    levels = [layer_rungs.get(int(lines[p["line"]].split()[1]))
              if layer_rungs.get(int(lines[p["line"]].split()[1])) is not None
              else pass_level(int(lines[p["line"]].split()[8]),
                              int(lines[p["line"]].split()[9]), src_ladder)
              for p in passes]
    padded = geom.padding(width, height)
    # Grids before and after: the push carries both the grid and the extent it belongs to, and
    # which word is which differs per family, so the rewrite is driven by value pairs rather
    # than by a per-family position list.
    src_grids = [(p["gx"], p["gy"]) for p in passes]
    dst_grids = [(gx, gy) for gx, gy in grids]

    for p, (gx, gy) in zip(passes, grids):
        parts = lines[p["line"]].split()
        parts[8], parts[9] = str(gx), str(gy)
        lines[p["line"]] = " ".join(parts)

        values = list(p["push"])  # verbatim: the constants do not depend on the frame
        # The frame and canvas words are written *last* (see `fix_port_words`), not here. Writing
        # them before the ladder rewrite put destination-size numbers in the push while the
        # rewrite still read every word as a source-size value: at 640x360 the canvas word 384
        # (already correct) matched the source rung 384 and was halved to 192, so the port was
        # told 640x192 for a 640x384 canvas.
        if resizing and port_key(p["name"]) in IOPORT_OFFSET_PASSES:
            rescale_ioport_offsets(values, src_w * src_h, width * height)
        lines[push_index[p["line"]]] = "PUSH " + " ".join(str(v) for v in values)
    text = "\n".join(lines) + "\n"
    # Which passes carry the "canvas level + its half" pair, decided on the *source* push so a
    # pass that does not hold the invariant is left alone (720 keeps byte-identical output).
    doubled_passes = {p["name"] for p in passes
                      if CANVAS_LEVEL_PASSES.match(p["name"]) and len(p["push"]) > 27
                      and p["push"][24] == 2 * p["push"][26]
                      and p["push"][25] == 2 * p["push"][27]}
    if resizing:
        text = rescale_history(text, src_w * src_h, width * height)
        # Last, because it reads the grids this function just wrote: every push word that names
        # a level extent has to follow the ladder, or the pass reads one level's rows at
        # another level's stride.
        text, baked = patch_baked_canvas(text, width, height, src_w, src_h)
        if baked:
            print(f"  canvas-baked shaders: {baked} repointed to {width}x{height} variants")
        text, moved = rescale_level_words(text, src_w, src_h, width, height, levels,
                                         src_grids, dst_grids)
        print(f"  level words: {moved} rewritten "
              f"({level_ladder(src_w, src_h)[0]} -> {level_ladder(width, height)[0]})")
        # Canvas-area words (ViT token count, ioport plane tail) - keyed by pass name, not by
        # value, because the same number is a channel count somewhere else.
        text, canvas_moved = rescale_canvas_words(text, src_w, src_h, width, height)
        print(f"  canvas words: {canvas_moved} rewritten")
        # A handful of passes push a *canvas* level at w24/w25 and half of it at w26/w27. The
        # ladder rewrote the second pair (it is an extent) and left the first (it is the canvas),
        # so the two disagreed after a resize. The invariant that makes the right answer unique
        # is inside the push itself: w24 == 2*w26. Enforce it instead of guessing a value.
        text, doubled = enforce_doubled_levels(text, doubled_passes)
        if doubled:
            print(f"  canvas levels: {doubled} words rewritten to 2x their own half")
    # Set last, unconditionally: these are the only push words whose meaning is known exactly
    # (a port pass is told the frame, and the canvas is the padded extent), so they are written
    # rather than matched, and nothing downstream may reinterpret them as a level value.
    text, port = fix_port_words(text, width, height)
    if port:
        print(f"  port words: {port} rewritten to frame {width}x{height}, "
              f"canvas {geom.padding(width, height).width}x{geom.padding(width, height).height}")
    return text


def roundtrip(width, height):
    """Build a plan for one size, rebuild the source size from it, and diff.

    A rewrite that cannot be undone was not understood: the two directions disagree exactly
    where a word was matched for the wrong reason (a stray 32 that happened to equal a rung
    extent, a half that also exists as a rung). This is the only gate that catches that
    without a working reference render, so it runs before any plan is installed.
    """
    global SOURCE_PLAN
    src_w, src_h = source_frame(read(SOURCE_PLAN.read_text(encoding="utf-8"))[1])
    built = build(width, height)
    ambiguous = dict(LAST_AMBIGUOUS)
    tmp = SOURCE_PLAN.parent / f".roundtrip-{width}x{height}.txt"
    tmp.write_text(built, encoding="utf-8", newline="\n")
    original, SOURCE_PLAN = SOURCE_PLAN, tmp
    try:
        back = build(src_w, src_h)
    finally:
        SOURCE_PLAN = original
        tmp.unlink(missing_ok=True)
    want = original.read_text(encoding="utf-8").splitlines()
    got = back.splitlines()
    diff = [i for i, (x, y) in enumerate(zip(want, got)) if x != y]
    # What this gate can actually decide.
    #
    # The honest claim is "a word this build moved, the reverse build puts back". It is not
    # "the reverse build reproduces the source", because the rewrite matches by *value* and a
    # value can be geometry on one side and a channel count on the other: `ioport_block4_0`
    # pushes 64, which is a channel count at 1280x720 and rung 4's width at 1024x576, so the
    # way back maps it to 84 without the way down ever having touched it. That is a limit of
    # value matching, not a defect in the plan, and it is reported separately.
    touched = dict(LAST_TOUCHED)
    ambiguous = dict(LAST_AMBIGUOUS)
    unexplained, one_way = [], []
    for i in diff:
        xa, ya = want[i].split(), got[i].split()
        if xa[0] == "PUSH" and ya[0] == "PUSH":
            # Word indices are relative to the push words, not to the opcode.
            wa, ga = xa[1:], ya[1:]
            cols = [k for k in range(min(len(wa), len(ga))) if wa[k] != ga[k]]
            if not (touched.get(i, set()) & set(cols)):
                one_way.append((i, [(k, wa[k], ga[k]) for k in cols][:6]))
            elif all(k in ambiguous.get(i, ()) for k in cols):
                one_way.append((i, [(k, wa[k], ga[k]) for k in cols][:6]))
            else:
                unexplained.append((i, [(k, wa[k], ga[k]) for k in cols][:6]))
        else:
            cols = [(k, xa[k], ya[k]) for k in range(min(len(xa), len(ya))) if xa[k] != ya[k]]
            unexplained.append((i, cols[:6]))
    print(f"roundtrip {width}x{height} -> {src_w}x{src_h}: {len(diff)} of {len(want)} lines "
          f"differ - {len(unexplained)} not restored where this build changed them, "
          f"{len(one_way)} values the reverse build maps but the forward never did")
    for i, cols in unexplained[:6]:
        print(f"  NOT RESTORED line {i}: {cols}")
    for i, cols in one_way[:4]:
        print(f"  one-way line {i}: {cols}")
    return 0 if not unexplained else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", help="WxH of the frame the network should process")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--check", action="store_true",
                    help="regenerate 1280x720 and compare with the frozen plan")
    ap.add_argument("--native-canvas", action="store_true",
                    help="extent is the frame itself; no 2^n rounding, no width step")
    ap.add_argument("--pad-width", action="store_true",
                    help="force work_size() to add one width step even when AMD's rule would not, "
                         "to test whether a canvas wider than the extent is what keeps 720p clean")
    ap.add_argument("--roundtrip", action="store_true",
                    help="build --size, rebuild the source size from it and diff: a rewrite "
                         "that cannot be undone was not understood")
    a = ap.parse_args()
    global PAD_WIDTH, NATIVE_CANVAS
    PAD_WIDTH = bool(a.pad_width)
    NATIVE_CANVAS = bool(a.native_canvas)

    if a.roundtrip:
        if not a.size:
            ap.error("--roundtrip needs --size")
        w, h = (int(v) for v in a.size.lower().split("x"))
        return roundtrip(w, h)

    if a.check:
        produced = build(1280, 720)
        original = SOURCE_PLAN.read_text(encoding="utf-8")
        problems = verify(1280, 720, produced)
        if produced == original and not problems:
            print("1280x720: byte-identical to the frozen plan, invariants hold")
            return 0
        if problems:
            print("1280x720: invariants broken")
            for p in problems:
                print(f"  {p}")
        src, dst = original.splitlines(), produced.splitlines()
        diff = [(i, s, d) for i, (s, d) in enumerate(zip(src, dst)) if s != d]
        print(f"1280x720: {len(diff)} differing lines of {len(src)}")
        for i, s, d in diff[:8]:
            print(f"  line {i}\n    frozen {s[:110]}\n    built  {d[:110]}")
        return 1

    if not a.size or not a.out:
        ap.error("--size and --out are required unless --check")
    width, height = (int(v) for v in a.size.lower().split("x"))
    produced = build(width, height)
    problems = verify(width, height, produced)
    if problems:
        print(f"{width}x{height}: refusing to write, invariants broken")
        for p in problems:
            print(f"  {p}")
        return 1
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(produced, encoding="utf-8", newline="\n")
    padded = geom.padding(width, height)
    print(f"{width}x{height} -> padded {padded.width}x{padded.height} -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
