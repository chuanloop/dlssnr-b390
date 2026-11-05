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

// --- 1-byte storage ------------------------------------------------------------
// Everything above keeps the value in a float16 whose low 7 mantissa bits are zero,
// i.e. the value is *already* on the e4m3 grid. That means it can be stored as one
// byte and read back **exactly** -- this is a packing, not a second rounding.
//
// `tools`-side proof: 50-SCRATCH/e4m3_codec.py round-trips all 252 grid values with
// zero error over all 256 byte patterns, so if a kernel writes only values that went
// through nr_e4m3_h, the frame must stay bit-identical after this change.
//
// The reference value of a byte: sign(1) exp(4, bias 7) mantissa(3); 0x7F is the only
// NaN, so the largest finite magnitude is 0x7E = 448.
uint NR_E4M3_BYTE_NAN = 0x7Fu;

// Decode one byte to float16. Exact for every byte (all e4m3 values are float16s).
//
// Pure bit manipulation rather than float arithmetic: e4m3 and float16 share the
// sign+exponent+mantissa layout, so a normal value is just two field shifts
// (e4m3 has 1 more exponent bit and 3 fewer mantissa bits: bias 7 -> 15 is +8,
// mantissa bit 3 -> bit 10 is +7). Only the subnormal range needs work, and it is
// three fixed cases because a 3-bit mantissa can have its leading one in bit 2, 1
// or 0. 50-SCRATCH/e4m3_codec.py checks this against the arithmetic definition for
// all 256 byte patterns.
float16_t nr_e4m3_decode(uint b) {
    const uint s = (b & 0x80u) << 8u;      // sign -> bit 15
    const uint e = (b >> 3u) & 0xFu;
    const uint m = b & 0x7u;
    if (e != 0u) {
        if (e == 15u && m == 7u) return uint16BitsToFloat16(uint16_t(0x7E00u));  // e4m3fn's only NaN
        return uint16BitsToFloat16(uint16_t(s | ((e + 8u) << 10u) | (m << 7u)));
    }
    if (m == 0u) return uint16BitsToFloat16(uint16_t(s));            // +-0
    // subnormal, value = m * 2^-9
    if ((m & 0x4u) != 0u) return uint16BitsToFloat16(uint16_t(s | 0x2000u | ((m - 4u) << 8u)));
    if ((m & 0x2u) != 0u) return uint16BitsToFloat16(uint16_t(s | 0x1C00u | ((m - 2u) << 9u)));
    return uint16BitsToFloat16(uint16_t(s | 0x1800u));
}

// Encode a float16 that is already on the e4m3 grid. The low 7 mantissa bits are
// dropped, not rounded -- callers must have gone through nr_e4m3_h first (that is what
// makes this lossless). Values off the grid still encode to the nearest grid point
// because dropping those bits IS round-toward-zero of the grid step; do not rely on it.
uint nr_e4m3_encode(float16_t v) {
    const uint16_t h = float16BitsToUint16(v);
    const uint s = uint(h >> 15u);
    const uint e16 = uint(h >> 10u) & 0x1Fu;
    const uint m16 = uint(h) & 0x3FFu;
    const uint sign = s << 7u;
    if (e16 == 0u) {
        // float16 zero or subnormal: |v| < 2^-14, far below e4m3's smallest step 2^-9
        return sign;
    }
    const int e = int(e16) - 8;                 // rebias 15 -> 7
    // e4m3's exponent field runs 1..15, so e == 15 is legal (256..448); only above that
    // is out of range. 0x7F is the sole NaN, hence saturation lands on 0x7E.
    if (e > 15) return sign | 0x7Eu;
    if (e <= 0) {                               // e4m3 subnormal band: step 2^-9
        if (e < -3) return sign;
        const uint m = (0x400u | m16) >> uint(1 - e);
        return sign | (m >> 7u);
    }
    const uint m3 = m16 >> 7u;
    if (e == 15 && m3 > 6u) return sign | 0x7Eu;   // > 448 cannot happen after nr_e4m3_h
    return sign | (uint(e) << 3u) | m3;
}

#endif  // NR_E4M3_GLSL
