// NB04: N32 retained; each B0/B1 serves two separate eight-row accumulators.
//
// T2 (2026-10-18) direct-epilogue variant: the F32 shared bridge of stages 1 and
// 3 (ffn_contract, projection) is deleted.  The parent stored each FP32
// accumulator to `shared float scratch[1024]`, synchronised the whole workgroup,
// read the value back at its (row,col) linear index, applied the residual FMA and
// re-narrowed to half.  The elementwise consumer read the *same* coordinate it
// owned, so no exchange was ever necessary: with layout RowMajor and stride 16 the
// fragment's linear index is `lane + 32*component`, i.e. component `ci` of lane
// `l` is (row = l/16 + 2*ci, col = l%16) - exactly the `e = lane + 32*ci`
// coordinate the parent recomputed.  The FMA, its operand order, its operand
// sources and the single half narrowing point are unchanged; only the transport
// (128 shared stores + 128 shared loads + 2 workgroup barriers per (group,side))
// is removed, and `scratch` (4096 B) disappears.
//
// Write domain is unchanged: `m` is a multiple of 8 and the live token count is a
// multiple of 8 (288), so every block with m < live covers rows m..m+7 < live and
// the parent's per-row `t < pc.p0.x` guard never fired.  Blocks with m >= live are
// still skipped by the existing `if (m>=pc.p0.x) continue;`.
layout(local_size_x=256) in;
#include "vit_common.glsl"
#define MH coopmat<float16_t,gl_ScopeSubgroup,8,16,gl_MatrixUseAccumulator>

// Stages 1 (ffn_contract) and 3/4 (projection) only: direct FP32 fragment ->
// single half narrowing -> fragment store.  No shared array is declared, so the
// kernel's workgroup shared footprint is 0.
#if VIT_STAGE == 1
void vit_direct_epilogue(MC acc,uint m,uint side,uint n0,uint lane) {
    MH res;
    uint col=n0+side*16u+(lane&15u);
    uint row0=m+(lane>>4u);
    for(uint ci=0u;ci<acc.length();++ci) {
        uint t=row0+2u*ci;
        res[ci]=float16_t(fma(float(w[pc.p2.x+col]),
                              float(x[pc.p5.x+t*1024u+col]),acc[ci]));
    }
    coopMatStore(res,y,pc.p5.w+m*1024u+n0+side*16u,1024u,RM);
}
#else
void vit_direct_epilogue(MC acc,uint m,uint side,uint n0,uint lane) {
    MH res;
    uint col=n0+side*16u+(lane&15u);
    uint row0=m+(lane>>4u);
    for(uint ci=0u;ci<acc.length();++ci) {
        uint t=row0+2u*ci;
        res[ci]=float16_t(fma(float(w[pc.p2.y+col]),
                              float(y[pc.p5.w+t*1024u+col]),acc[ci]));
    }
    coopMatStore(res,o,pc.p5.y+m*1024u+n0+side*16u,1024u,RM);
}
#endif

void main() {
    // v3 (2026-11-03) 64 columns per workgroup: eight independent accumulator chains.
    //
    // The measured 10.4 TFLOPS on this part is 67% of the *vector* FP16 peak and 17% of
    // the 61 TFLOPS XMX peak.  v2 (the 64-token split) doubled the workgroups and changed
    // nothing -- and it also halved the chains per subgroup, from four to two, which is
    // the wrong direction.  The signature is a dependency chain: four accumulators, each
    // waiting on a DPAS that takes ~4x its issue interval, gives exactly the ~17% we see.
    //
    // So go the other way: four N blocks (n0+0/16/32/48) x two M blocks = eight
    // accumulators per subgroup, same K order per output element, so the result stays
    // bit-identical.  A is loaded once per (K step, M block) and feeds four MMAs instead
    // of two, which also halves the A traffic (96 -> 50 MB per pass).
    uint sg=gl_SubgroupID,lane=gl_SubgroupInvocationID;
    uint mA=gl_WorkGroupID.x*128u+sg*8u,mB=mA+64u,n0=gl_WorkGroupID.y*64u;
#if VIT_STAGE == 0
    const uint K=1024u; uint wb=pc.p1.x;
#elif VIT_STAGE == 1
    const uint K=4096u; uint wb=pc.p1.y;
#else
    const uint K=1024u; uint wb=pc.p1.w;
#endif
    MC accA0=MC(0.0),accA1=MC(0.0),accA2=MC(0.0),accA3=MC(0.0);
    MC accB0=MC(0.0),accB1=MC(0.0),accB2=MC(0.0),accB3=MC(0.0);
    if(mA<pc.p0.x&&mB<pc.p0.x) {
        MB b0,b1,b2,b3,c0,c1,c2,c3;
        coopMatLoad(b0,w,vit_wbase(wb,n0,0u,K),16u,CM);
        coopMatLoad(b1,w,vit_wbase(wb,n0+16u,0u,K),16u,CM);
        coopMatLoad(b2,w,vit_wbase(wb,n0+32u,0u,K),16u,CM);
        coopMatLoad(b3,w,vit_wbase(wb,n0+48u,0u,K),16u,CM);
        coopMatLoad(c0,w,vit_wbase(wb,n0,16u,K),16u,CM);
        coopMatLoad(c1,w,vit_wbase(wb,n0+16u,16u,K),16u,CM);
        coopMatLoad(c2,w,vit_wbase(wb,n0+32u,16u,K),16u,CM);
        coopMatLoad(c3,w,vit_wbase(wb,n0+48u,16u,K),16u,CM);
        for(uint k0=0u;k0+32u<K;k0+=32u) {
            MA af;
            coopMatLoad(af,aHidden,pc.p5.z+mA*K+k0,K,RM);
            accA0=coopMatMulAdd(af,b0,accA0);
            accA1=coopMatMulAdd(af,b1,accA1);
            accA2=coopMatMulAdd(af,b2,accA2);
            accA3=coopMatMulAdd(af,b3,accA3);
            coopMatLoad(af,aHidden,pc.p5.z+mB*K+k0,K,RM);
            accB0=coopMatMulAdd(af,b0,accB0);
            accB1=coopMatMulAdd(af,b1,accB1);
            accB2=coopMatMulAdd(af,b2,accB2);
            accB3=coopMatMulAdd(af,b3,accB3);
            coopMatLoad(b0,w,vit_wbase(wb,n0,k0+32u,K),16u,CM);
            coopMatLoad(b1,w,vit_wbase(wb,n0+16u,k0+32u,K),16u,CM);
            coopMatLoad(b2,w,vit_wbase(wb,n0+32u,k0+32u,K),16u,CM);
            coopMatLoad(b3,w,vit_wbase(wb,n0+48u,k0+32u,K),16u,CM);
            coopMatLoad(af,aHidden,pc.p5.z+mA*K+k0+16u,K,RM);
            accA0=coopMatMulAdd(af,c0,accA0);
            accA1=coopMatMulAdd(af,c1,accA1);
            accA2=coopMatMulAdd(af,c2,accA2);
            accA3=coopMatMulAdd(af,c3,accA3);
            coopMatLoad(af,aHidden,pc.p5.z+mB*K+k0+16u,K,RM);
            accB0=coopMatMulAdd(af,c0,accB0);
            accB1=coopMatMulAdd(af,c1,accB1);
            accB2=coopMatMulAdd(af,c2,accB2);
            accB3=coopMatMulAdd(af,c3,accB3);
            coopMatLoad(c0,w,vit_wbase(wb,n0,k0+48u,K),16u,CM);
            coopMatLoad(c1,w,vit_wbase(wb,n0+16u,k0+48u,K),16u,CM);
            coopMatLoad(c2,w,vit_wbase(wb,n0+32u,k0+48u,K),16u,CM);
            coopMatLoad(c3,w,vit_wbase(wb,n0+48u,k0+48u,K),16u,CM);
        }
        // Last two K steps use already loaded slots; no tail prefetch.
        {
            MA af;
            coopMatLoad(af,aHidden,pc.p5.z+mA*K+K-32u,K,RM);
            accA0=coopMatMulAdd(af,b0,accA0);
            accA1=coopMatMulAdd(af,b1,accA1);
            accA2=coopMatMulAdd(af,b2,accA2);
            accA3=coopMatMulAdd(af,b3,accA3);
            coopMatLoad(af,aHidden,pc.p5.z+mB*K+K-32u,K,RM);
            accB0=coopMatMulAdd(af,b0,accB0);
            accB1=coopMatMulAdd(af,b1,accB1);
            accB2=coopMatMulAdd(af,b2,accB2);
            accB3=coopMatMulAdd(af,b3,accB3);
            coopMatLoad(af,aHidden,pc.p5.z+mA*K+K-16u,K,RM);
            accA0=coopMatMulAdd(af,c0,accA0);
            accA1=coopMatMulAdd(af,c1,accA1);
            accA2=coopMatMulAdd(af,c2,accA2);
            accA3=coopMatMulAdd(af,c3,accA3);
            coopMatLoad(af,aHidden,pc.p5.z+mB*K+K-16u,K,RM);
            accB0=coopMatMulAdd(af,c0,accB0);
            accB1=coopMatMulAdd(af,c1,accB1);
            accB2=coopMatMulAdd(af,c2,accB2);
            accB3=coopMatMulAdd(af,c3,accB3);
        }
    }
    for(uint group=0u;group<2u;++group) {
        uint m=group==0u?mA:mB;
        if(m>=pc.p0.x)continue; // uniform within this SG; no WG barriers here.
        for(uint side=0u;side<4u;++side) {
#if VIT_STAGE == 0
            MH result;
            if(group==0u) {
                if(side==0u)result=MH(accA0);
                else if(side==1u)result=MH(accA1);
                else if(side==2u)result=MH(accA2);
                else result=MH(accA3);
            } else {
                if(side==0u)result=MH(accB0);
                else if(side==1u)result=MH(accB1);
                else if(side==2u)result=MH(accB2);
                else result=MH(accB3);
            }
            if(pc.p0.w!=0u)
                coopMatStore(result,dbg,pc.p6.z+pc.p3.x+m*4096u+n0+side*16u,4096u,RM);
            for(uint c=0u;c<result.length();++c)result[c]=vit_cubic(result[c]);
            coopMatStore(result,aHidden,pc.p5.z+m*4096u+n0+side*16u,4096u,RM);
#else
            if(group==0u) {
                if(side==0u)vit_direct_epilogue(accA0,m,side,n0,lane);
                else if(side==1u)vit_direct_epilogue(accA1,m,side,n0,lane);
                else if(side==2u)vit_direct_epilogue(accA2,m,side,n0,lane);
                else vit_direct_epilogue(accA3,m,side,n0,lane);
            } else {
                if(side==0u)vit_direct_epilogue(accB0,m,side,n0,lane);
                else if(side==1u)vit_direct_epilogue(accB1,m,side,n0,lane);
                else if(side==2u)vit_direct_epilogue(accB2,m,side,n0,lane);
                else vit_direct_epilogue(accB3,m,side,n0,lane);
            }
#endif
        }
    }
}
