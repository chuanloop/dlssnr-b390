// e4m3 (OCP FP8 E4M3FN) rounding, for every write into the activation arena.
//
// The network NVIDIA ships holds activations as e4m3 at each operator boundary:
// 1 sign, 4 exponent (bias 7), 3 mantissa, max 448, no infinities. Its hardware
// converts with round-to-nearest-even and saturates. Running the network in plain
// FP16 instead is worth about 10 dB against NVIDIA's own output (DLSSNR-RDNA3,
// README-RDNA3.md: 49.0 -> 38.5 dB with the rounding removed), so this file is not
// an optimisation - it is part of the model.
//
// The method is AMD's (`linux/shaders/rdna3/include/e4m3_emul.glsl`,
// `nr_e4m3_round` / `nr_e4m3_round_pair`), not the bit-mask path this file used to
// carry. Adding M = 1.5 * 2^(E+20) to x, where 2^E <= |x| < 2^(E+1), leaves a sum
// whose ulp is 2^(E-3): exactly three mantissa bits survive, rounded to nearest
// even by the adder, and subtracting M gives that value back. M is positive for
// either sign of x, so no sign has to be copied into it. The exponent is floored at
// e4m3's smallest normal (2^-6) because below that the grid is the fixed subnormal
// step 2^-9, and |x| is clamped to 448 first, which is the saturation.
//
// Why this form: three instructions, no branches. The previous version had three
// branches plus a roundEven, and `tools/verify/e4m3.py` had to prove each of them.
// The claim is now carried by the constants instead, and
// `tools/verify/e4m3_magic.py` checks them the same way: all 65536 binary16
// patterns, all 251 grid midpoints (where ties-to-even decides), 600k random
// values, and every exponent with four mantissas, against the analytic oracle and
// against the independent 256-encoding brute force. Both entry points, both
// precisions. Zero mismatches. If you touch a constant, rerun that file.
//
// `precise` is load-bearing: without it the compiler is free to reassociate
// `(c + M) - M` to `c` and the rounding disappears silently.
//
// NaN: not defined, and unreachable - no activation of this network can be one.
// AMD's version does not define it either. Infinity saturates to +-448.
//
// Cost on Xe3: the pair form is two packed f16 additions plus the constant build,
// so a store costs about one extra instruction over the value it writes.

#ifndef NR_E4M3_GLSL
#define NR_E4M3_GLSL

#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require

// e4m3's smallest normal, as a float32/f16 exponent field. Everything below it
// quantises on the fixed 2^-9 subnormal grid, which the same constant handles.
const uint NR_E4M3_F32_EXP_MASK = 0x7F800000u;
const uint NR_E4M3_F32_MIN_NORMAL = 0x3C800000u;  // 2^-6
const uint NR_E4M3_F32_MAGIC_ADD = 0x0A400000u;   // exponent +20, mantissa bit 22

const uint16_t NR_E4M3_F16_EXP_MASK = 0x7C00us;
const uint16_t NR_E4M3_F16_MIN_NORMAL = 0x2400us;  // 2^-6
const uint16_t NR_E4M3_F16_MAGIC_ADD = 0x1E00us;   // exponent +7, mantissa bit 9

// float32 in, float32 out. The result is always exactly representable in
// float16, so a kernel may store it either way.
float nr_e4m3_round(float x) {
    precise float c = clamp(x, -448.0, 448.0);
    const uint u = floatBitsToUint(c);
    const uint magic = max(u & NR_E4M3_F32_EXP_MASK, NR_E4M3_F32_MIN_NORMAL)
                     + NR_E4M3_F32_MAGIC_ADD;
    precise float sum = c + uintBitsToFloat(magic);
    precise float r = sum - uintBitsToFloat(magic);
    return r;
}

// Two at a time, in packed 16-bit arithmetic. This is the form the kernels use:
// activations already live in float16_t, and an e4m3 value is exactly a float16
// value, so only the narrowing onto the grid is missing. The two lanes never
// interact, so this is two scalar roundings, not an approximation.
f16vec2 nr_e4m3_round_pair(f16vec2 x) {
    precise f16vec2 c = clamp(x, f16vec2(-448.0hf), f16vec2(448.0hf));
    const u16vec2 u = float16BitsToUint16(c);
    const f16vec2 magic = uint16BitsToFloat16(
        max(u & u16vec2(NR_E4M3_F16_EXP_MASK), u16vec2(NR_E4M3_F16_MIN_NORMAL))
        + u16vec2(NR_E4M3_F16_MAGIC_ADD));
    precise f16vec2 sum = c + magic;
    precise f16vec2 r = sum - magic;
    return r;
}

// --- entry points -------------------------------------------------------------
// Named explicitly rather than only overloaded: a kernel that stores float16_t
// should reach the packed path, and overloading float/float16_t leaves that to
// implicit conversion rules.

// Narrow a float16 onto the e4m3 grid. The hot path.
float16_t nr_e4m3_h(float16_t v) { return nr_e4m3_round_pair(f16vec2(v)).x; }

// Narrow a pair of float16s. Prefer this over two nr_e4m3_h calls in a packed store.
f16vec2 nr_e4m3_pair(f16vec2 v) { return nr_e4m3_round_pair(v); }
f16vec4 nr_e4m3_quad(f16vec4 v) {
    return f16vec4(nr_e4m3_round_pair(v.xy), nr_e4m3_round_pair(v.zw));
}

// A float32 input is rounded once, straight onto the grid.
float nr_e4m3_f32(float v) { return nr_e4m3_round(v); }
float16_t nr_e4m3_h(float v) { return float16_t(nr_e4m3_round(v)); }

// Overloads, so existing call sites keep working.
float nr_e4m3(float v) { return nr_e4m3_round(v); }
float16_t nr_e4m3(float16_t v) { return nr_e4m3_h(v); }
vec2 nr_e4m3(vec2 v) { return vec2(nr_e4m3_round(v.x), nr_e4m3_round(v.y)); }
vec4 nr_e4m3(vec4 v) { return vec4(nr_e4m3_round(v.x), nr_e4m3_round(v.y),
                                   nr_e4m3_round(v.z), nr_e4m3_round(v.w)); }

#endif  // NR_E4M3_GLSL
