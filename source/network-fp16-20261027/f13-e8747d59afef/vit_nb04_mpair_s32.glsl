// NB04: N32 retained; each B0/B1 serves two separate eight-row accumulators.
// n11-vit-ffn candidate (2026-10-27): parent = .../dsh-fp16-same-family-20261023/
// s32-k11-v1/source/vit_nb04_mpair_s32.glsl (sha256 d13e773b..b68a). Only the K-loop
// live guards changed; see SOURCE-DIFF.patch in this card directory.
layout(local_size_x=256) in;
#include "vit_common.glsl"
// e4m3 narrowing of the stored activation (VitFfnExpand).
// `layers/vit.py` VitFfnExpand.forward ends with `to_map(q8(f16(hidden)))`, and every
// ViT layer stores through the same rounding. `result[c]` after vit_cubic is already
// float16_t, so nr_e4m3_h is the f16 -> e4m3 half of that double round.
// NR_QUANT_E4M3=0 restores the unrounded behaviour for A/B measurement.
#ifndef NR_QUANT_E4M3
#define NR_QUANT_E4M3 1
#endif
#include "e4m3.glsl"
#define MH coopmat<float16_t,gl_ScopeSubgroup,8,16,gl_MatrixUseAccumulator>
#if VIT_STAGE != 0
shared float scratch[1024];
#endif
void main() {
    uint sg=gl_SubgroupID,lane=gl_SubgroupInvocationID;
    // S32: two adjacent M8 per SG, N32 selected by sg%4; grid = ceil(M/32) x (N/128).
    uint mA=gl_WorkGroupID.x*32u+16u*(sg/4u),mB=mA+8u,n0=gl_WorkGroupID.y*128u+32u*(sg%4u);
#if VIT_STAGE == 0
    const uint K=1024u; uint wb=pc.p1.x;
#elif VIT_STAGE == 1
    const uint K=4096u; uint wb=pc.p1.y;
#else
    const uint K=1024u; uint wb=pc.p1.w;
#endif
    MC accA0=MC(0.0),accA1=MC(0.0),accB0=MC(0.0),accB1=MC(0.0);
    // n11-vit-ffn: the per-iteration live guards are hoisted to ONE subgroup-uniform
    // branch and the row indices are clamped into the live range, so the K loop body
    // carries no branch and no join for the four accumulators. Per-output K order
    // (0,16,...,K-16), the B/A load count and the epilogue guard are unchanged; a dead
    // row may read the clamped live row but its accumulators are never stored.
    uint live=pc.p0.x,last=((live>0u?live-1u:0u)/8u)*8u;
    uint rA=min(mA,last),rB=min(mB,last);
    if(mA<live) {
        for(uint k0=0u;k0<K;k0+=16u) {
            MB b0,b1;
            coopMatLoad(b0,w,vit_wbase(wb,n0,k0,K),16u,CM);
            coopMatLoad(b1,w,vit_wbase(wb,n0+16u,k0,K),16u,CM);
            MA af;
#if VIT_STAGE == 0
            coopMatLoad(af,x,pc.p5.x+rA*K+k0,K,RM);
#elif VIT_STAGE == 1
            coopMatLoad(af,aHidden,pc.p5.z+rA*K+k0,K,RM);
#else
            coopMatLoad(af,context,pc.p6.y+rA*K+k0,K,RM);
#endif
            accA0=coopMatMulAdd(af,b0,accA0);
            accA1=coopMatMulAdd(af,b1,accA1);
#if VIT_STAGE == 0
            coopMatLoad(af,x,pc.p5.x+rB*K+k0,K,RM);
#elif VIT_STAGE == 1
            coopMatLoad(af,aHidden,pc.p5.z+rB*K+k0,K,RM);
#else
            coopMatLoad(af,context,pc.p6.y+rB*K+k0,K,RM);
#endif
            accB0=coopMatMulAdd(af,b0,accB0);
            accB1=coopMatMulAdd(af,b1,accB1);
        }
    }
    for(uint group=0u;group<2u;++group) {
        uint m=group==0u?mA:mB;
        if(m>=pc.p0.x)continue; // uniform within this SG; no WG barriers here.
        for(uint side=0u;side<2u;++side) {
#if VIT_STAGE == 0
            MH result;
            if(group==0u) {
                if(side==0u)result=MH(accA0);else result=MH(accA1);
            } else {
                if(side==0u)result=MH(accB0);else result=MH(accB1);
            }
            if(pc.p0.w!=0u)
                coopMatStore(result,dbg,pc.p6.z+pc.p3.x+m*4096u+n0+side*16u,4096u,RM);
#if NR_QUANT_E4M3
            for(uint c=0u;c<result.length();++c)result[c]=nr_e4m3_h(vit_cubic(result[c]));
#else
            for(uint c=0u;c<result.length();++c)result[c]=vit_cubic(result[c]);
#endif
            coopMatStore(result,aHidden,pc.p5.z+m*4096u+n0+side*16u,4096u,RM);
#else
            if(group==0u) {
                if(side==0u)coopMatStore(accA0,scratch,sg*128u,16u,RM);
                else coopMatStore(accA1,scratch,sg*128u,16u,RM);
            } else {
                if(side==0u)coopMatStore(accB0,scratch,sg*128u,16u,RM);
                else coopMatStore(accB1,scratch,sg*128u,16u,RM);
            }
            subgroupMemoryBarrierShared();subgroupBarrier();
            for(uint e=lane;e<128u;e+=32u) {
                uint t=m+e/16u,c=n0+side*16u+e%16u;
                float raw=scratch[sg*128u+e];
                if(t<pc.p0.x) {
#if VIT_STAGE == 1
                    y[pc.p5.w+t*1024u+c]=float16_t(fma(float(w[pc.p2.x+c]),
                                                         float(x[pc.p5.x+t*1024u+c]),raw));
#else
                    o[pc.p5.y+t*1024u+c]=float16_t(fma(float(w[pc.p2.y+c]),
                                                         float(y[pc.p5.w+t*1024u+c]),raw));
#endif
                }
            }
            subgroupMemoryBarrierShared();subgroupBarrier();
#endif
        }
    }
}
