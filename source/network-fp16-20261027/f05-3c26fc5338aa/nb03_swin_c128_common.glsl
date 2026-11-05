// c128_common.glsl — shared push-constant ABI, coopmat typedefs and math
// helpers for the four C128 pass shaders (guide §5/§7/§9).
// Shared ARRAYS are declared per-shader (guide §6: only the arrays a pass
// uses), never here.

#ifndef C128_COMMON_GLSL
#define C128_COMMON_GLSL

layout(set = 0, binding = 0) buffer PackBuf { float16_t pack[]; };
layout(set = 0, binding = 1) buffer AuxBuf  { float auxf[]; };    // bias + scalars
layout(set = 0, binding = 2) buffer ImageIn { float16_t imgin[]; };
layout(set = 0, binding = 3) buffer ImageOut { float16_t imgout[]; };
layout(set = 0, binding = 4) buffer DbgBuf  { float16_t dbg[]; }; // half byte-view
// Same buffer, F32 raw-L bridge; original half-word debug codec follows.
layout(set = 0, binding = 4) buffer DbgFloat { float dbgf[]; };

layout(push_constant) uniform PC {
    ivec4 p0;   // tiles_x=42, tiles_y=24, shift_x, shift_y (signed)
    ivec4 p1;   // grid_x=22, grid_y=13, single_mode, debug
    ivec4 p2;   // single_wx, single_wy, reserved, reserved
    ivec4 p3;   // we_half, wm_half, wc_half, wq_half
    ivec4 p4;   // wo_half, rs_half, ars_half, bias_float
    ivec4 p5;   // scalar_float, M_half_base, Y_half_base, T_half_base
    ivec4 p6;   // input_half_base, output_half_base, debug_half_base, profile_id
    ivec4 p7;   // reserved
} pc;

#define MAT_A coopmat<float16_t, gl_ScopeSubgroup, 8, 16, gl_MatrixUseA>
#define MAT_B coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB>
#define MAT_C coopmat<float, gl_ScopeSubgroup, 8, 16, gl_MatrixUseAccumulator>
#define LROW  gl_CooperativeMatrixLayoutRowMajor
#define LCOL  gl_CooperativeMatrixLayoutColumnMajor

// window/token geometry (guide §2): q=t/16 (x-first tile), s=t%16 (y-major
// slot); tx = 2wx+shift_x+(q&1), ty = 2wy+shift_y+(q>>1); addresses computed
// per element on the fly (no private coordinate tables — C32 lesson).
uint c128_img_half_index(uint t, uint c, uint wx, uint wy, out bool oob) {
    uint q = t >> 4u, s = t & 15u;
    int tx = 2 * int(wx) + pc.p0.z + int(q & 1u);
    int ty = 2 * int(wy) + pc.p0.w + int(q >> 1u);
    oob = (tx < 0 || ty < 0 || tx >= pc.p0.x || ty >= pc.p0.y);
    uint stx = uint(clamp(tx, 0, pc.p0.x - 1));
    uint sty = uint(clamp(ty, 0, pc.p0.y - 1));
    return (((sty * uint(pc.p0.x) + stx) * 8u + c / 16u) * 256u
            + s * 16u + c % 16u);
}

// half MpCubicSilu (guide §4.1, same as sealed C32)
float16_t c128_act_h(float16_t x) {
    float16_t t = clamp(x, float16_t(-4.0), float16_t(4.0));
    float16_t u = fma(float16_t(-0.055908203125), abs(t), float16_t(0.447265625));
    return x * fma(t, u, float16_t(0.89453125));
}

// half swin exp bit construction (guide §4.6, same as sealed C32)
float16_t c128_exp_h(float16_t x) {
    float16_t y = clamp(fma(x, float16_t(0.044921875), float16_t(1.30078125)),
                        float16_t(1.03125), float16_t(1.5693359375));
    uint h = packHalf2x16(vec2(float(y), 0.0)) & 0xffffu;
    return float16_t(unpackHalf2x16((h & 0x03ffu) << 5u).x);
}

// 32-dim half square + S4 tree (guide §4.5); row is float(half values);
// overflow propagates to inf (caller turns it into n = 0).
float c128_norm_sum32(float[32] row) {
    float16_t sq[32];
    for (uint c = 0u; c < 32u; ++c)
        sq[c] = float16_t(row[c]) * float16_t(row[c]);
    float16_t s[8];
    for (uint c = 0u; c < 8u; ++c)
        s[c] = (sq[c] + sq[c + 16u]) + (sq[c + 8u] + sq[c + 24u]);
    float16_t t4[4], t2[2];
    for (uint c = 0u; c < 4u; ++c) t4[c] = s[c] + s[c + 4u];
    for (uint c = 0u; c < 2u; ++c) t2[c] = t4[c] + t4[c + 2u];
    return float(t2[0] + t2[1]);
}

#endif
