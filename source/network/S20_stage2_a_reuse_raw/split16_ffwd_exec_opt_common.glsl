#ifndef SPLIT16_COMMON_GLSL
#define SPLIT16_COMMON_GLSL
// ABI: offsets in half elements, except bias/scales/debug in float elements.
layout(set=0,binding=0) readonly buffer Weights { float16_t weights[]; };
layout(set=0,binding=1) readonly buffer Auxiliary { float aux[]; };
layout(set=0,binding=2) readonly buffer Input { float16_t gx[]; };
layout(set=0,binding=3) buffer Ffwd { float16_t gf[]; };
layout(set=0,binding=4) buffer Residual { float16_t gy[]; };
layout(set=0,binding=5) buffer Context { float16_t gt[]; };
layout(set=0,binding=6) buffer Output { float16_t go[]; };
layout(set=0,binding=7) buffer Pool { float16_t gp[]; };
layout(set=0,binding=8) buffer FinalHead { float16_t gh[]; };
layout(set=0,binding=9) buffer Debug { float dbg[]; };
layout(push_constant) uniform Push {
    ivec4 p0; // width, height, padded tokens, flags (inpview=1,pool=2,outview=4)
    ivec4 p1; // grid_x, grid_y, shift_x, shift_y
    ivec4 p2; // A, Q0, Q2, ffwd projection half offsets
    ivec4 p3; // ffwd residual scale, QKV, output projection, output scale
    ivec4 p4; // bias float offset, Q scale float offset, final-head half offset, 0
    ivec4 p5; // debug enabled, R float base, E float base, A float base
    ivec4 p6; // Q0 debug base, 0, 0, 0
    ivec4 p7; // pooled width, pooled height, final padded tokens, 0
} pc;
#define MA coopmat<float16_t,gl_ScopeSubgroup,8,16,gl_MatrixUseA>
#define MB coopmat<float16_t,gl_ScopeSubgroup,16,16,gl_MatrixUseB>
#define MC coopmat<float,gl_ScopeSubgroup,8,16,gl_MatrixUseAccumulator>
#define ROW gl_CooperativeMatrixLayoutRowMajor
#define COL gl_CooperativeMatrixLayoutColumnMajor
// C2 common layout0 (tile4x4_channel16, io_at layout_id==0 semantics):
// token is the 4x4-block-major token index; tiled() is the unchanged
// 16-token x 16-channel supertile address over that block-major token.
uint tiled(uint token,uint c,uint C) {
    return ((token/16u)*(C/16u)+c/16u)*256u+(token%16u)*16u+c%16u;
}
// view() is retained only for the retired inpview contract; the flag is 0
// for every L2 call (81..113, 156..187), so this branch is dead in L2.
uint view(uint token,uint c) {
    uint s=c%16u;
    uint perm=(s&1u)|((s>>3u&1u)<<1u)|((s>>1u&1u)<<2u)|((s>>2u&1u)<<3u);
    return (c/16u)*uint(pc.p0.x*pc.p0.y)*16u+token*16u+perm;
}
float16_t input_value(uint token,uint c) {
    return gx[(pc.p0.w&1)!=0 ? view(token,c) : tiled(token,c,512u)];
}
// layout0 block-major token of the raster pixel (px,py); W=pc.p0.x (44), a
// multiple of 4. Identical formula to ioport_common.glsl io_at(...,layout 0).
uint blk_token(uint px,uint py) {
    return ((py/4u)*uint(pc.p0.x/4)+px/4u)*16u+4u*(py%4u)+px%4u;
}
// Under layout0 the block-major token of a 4x4 body tile is tile*16+slot.
uint spatial_token(uint tile,uint slot) {
    return tile*16u+slot;
}
uint window_token(uint wx,uint wy,uint t,out bool valid) {
    int tx=2*int(wx)+pc.p1.z+int(t/16u&1u);
    int ty=2*int(wy)+pc.p1.w+int(t/32u);
    valid=tx>=0 && ty>=0 && tx<pc.p0.x/4 && ty<pc.p0.y/4;
    return blk_token(uint(tx)*4u+uint(t%4u),uint(ty)*4u+uint(t%16u/4u));
}
float16_t activation(float16_t x) {
    precise float16_t t=clamp(x,float16_t(-4.0),float16_t(4.0));
    precise float16_t u=fma(float16_t(-0.055908203125),abs(t),float16_t(0.447265625));
    precise float16_t v=fma(t,u,float16_t(0.89453125));
    return x*v;
}
float16_t exponential(float16_t x) {
    precise float16_t z=clamp(fma(x,float16_t(0.044921875),float16_t(1.30078125)),
                            float16_t(1.03125),float16_t(1.5693359375));
    uint b=packHalf2x16(vec2(float(z),0.0))&0xffffu;
    return float16_t(unpackHalf2x16((b&0x3ffu)<<5u).x);
}
// Sealed P1/P2 half math tree; explicit rounding after every add/multiply.
float norm32(float16_t row[32]) {
    float16_t z[32],s[8],u[4],v[2];
    for(uint c=0u;c<32u;c++) z[c]=row[c]*row[c];
    for(uint c=0u;c<8u;c++) {
        precise float16_t a=z[c]+z[c+16u],b=z[c+8u]+z[c+24u];
        s[c]=a+b;
    }
    for(uint c=0u;c<4u;c++) u[c]=s[c]+s[c+4u];
    for(uint c=0u;c<2u;c++) v[c]=u[c]+u[c+2u];
    return float(v[0]+v[1]);
}
uint qdbg(uint kind,uint w,uint h,uint e) {
    uint stride=uint(pc.p1.x*pc.p1.y)*16u*2048u;
    return uint(pc.p6.x)+kind*stride+(w*16u+h)*2048u+e;
}
uint ldbg(uint kind,uint w,uint h,uint e) {
    uint nwin=uint(pc.p1.x*pc.p1.y);
    return uint(pc.p6.x)+5u*nwin*16u*2048u+kind*nwin*16u*4096u+(w*16u+h)*4096u+e;
}
#endif
