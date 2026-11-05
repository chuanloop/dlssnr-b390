// ViT FP16 port. Scalar addresses, not AMD fragment lane mappings.
#ifndef VIT_COMMON_GLSL
#define VIT_COMMON_GLSL
layout(set=0,binding=0,std430) readonly buffer Weights { float16_t w[]; };
layout(set=0,binding=1,std430) readonly buffer Scales { float scales[]; };
layout(set=0,binding=2,std430) readonly buffer Input { float16_t x[]; };
layout(set=0,binding=3,std430) buffer Output { float16_t o[]; };
layout(set=0,binding=4,std430) buffer Hidden { float16_t aHidden[]; };
layout(set=0,binding=5,std430) buffer Residual { float16_t y[]; };
layout(set=0,binding=6,std430) buffer QKV { float16_t qkv[]; };
layout(set=0,binding=7,std430) buffer Context { float16_t context[]; };
layout(set=0,binding=8,std430) buffer Debug { float16_t dbg[]; };
layout(set=0,binding=8,std430) buffer DebugFloat { float dbgf[]; };
layout(push_constant) uniform Push {
    uvec4 p0; // live tokens, padded keys (64), stage, debug
    uvec4 p1; // WE, WC, WQKV, WO half offsets
    uvec4 p2; // FF skip, out skip half offsets, scale float offset, block
    uvec4 p3; // debug E half, QKV0 half, logits float, exponentials half
    uvec4 p4; // denominator half, numerator half, inverse half, reserved
    uvec4 p5; // input, output, hidden, Y half bases
    uvec4 p6; // QKV, context, debug half bases, reserved
    uvec4 p7; // debug bytes, profile=1, reserved, reserved
} pc;
#define MA coopmat<float16_t,gl_ScopeSubgroup,8,16,gl_MatrixUseA>
#define MB coopmat<float16_t,gl_ScopeSubgroup,16,16,gl_MatrixUseB>
#define MC coopmat<float,gl_ScopeSubgroup,8,16,gl_MatrixUseAccumulator>
#define RM gl_CooperativeMatrixLayoutRowMajor
#define CM gl_CooperativeMatrixLayoutColumnMajor

// W[N/32][K/16][32][16], offsets/strides in half elements.
uint vit_wbase(uint base, uint n, uint k, uint K) {
    return base + (n/32u)*(K/16u)*512u + (k/16u)*512u + (n%32u)*16u;
}
float16_t vit_cubic(float16_t z) {
    precise float16_t t=clamp(z,float16_t(-4.0),float16_t(4.0));
    precise float16_t u=fma(float16_t(-0.055908203125),abs(t),float16_t(0.447265625));
    precise float16_t v=fma(t,u,float16_t(0.89453125));
    return z*v;
}
float16_t vit_exp(float16_t z) {
    precise float16_t v=clamp(fma(z,float16_t(0.08953857421875),float16_t(1.708984375)),
                              float16_t(1.439453125),float16_t(1.9775390625));
    uint h=packHalf2x16(vec2(float(v),0.0))&0xffffu;
    return float16_t(unpackHalf2x16((h&0x3ffu)<<4u).x);
}
// M05 paired form of vit_exp() (merged from c5-alusupply/vit-attention-m05-expair-v1;
// same FMA/clamp constants and truncation shift as the scalar form, verbatim):
// mantissa is bits 0..9 of each 16-bit lane, so (h & 0x03ff03ff) << 4 sets only
// bits 4..13 (low lane) and bits 20..29 (high lane) — no carry can cross the
// 16-bit boundary (13 < 16), so both lanes are bit-identical to the scalar path
// applied elementwise. The padding subtraction keeps its scalar vit_exp(0.0).
f16vec2 vit_exp2(f16vec2 xy) {
    precise f16vec2 v=clamp(fma(xy,f16vec2(float16_t(0.08953857421875)),f16vec2(float16_t(1.708984375))),
                            f16vec2(float16_t(1.439453125)),f16vec2(float16_t(1.9775390625)));
    uint h=packFloat2x16(v);
    return unpackFloat2x16((h&0x3ff03ffu)<<4u);
}
#endif
