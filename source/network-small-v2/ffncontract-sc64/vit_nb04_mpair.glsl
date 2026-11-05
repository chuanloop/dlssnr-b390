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
    // v2 (2026-11-03) 64-token workgroups.  v1 gave each workgroup two eight-row
    // accumulator blocks per subgroup (128 tokens), so a 384-token frame needed only
    // 3 x-groups and the pass ran 96 workgroups for 2.68 ms -- the lowest occupancy
    // in the network (see local/small/perf/passms-1836.txt).  Measured on this part,
    // same work split into more groups goes 16.0 -> 24.7 TFLOPS at 192 groups,
    // because a single-wave kernel has no second workgroup to cover DPAS latency.
    //
    // The split is over M only.  Every output element still walks K in the same
    // order with the same operands, so the result is bit-identical; only the number
    // of accumulators per subgroup drops from four to two.  B is re-read by twice as
    // many x-groups (8 MB per pass fits in the 16 MB L2), A traffic is unchanged.
    uint sg=gl_SubgroupID,lane=gl_SubgroupInvocationID;
    uint mA=gl_WorkGroupID.x*64u+sg*8u,n0=gl_WorkGroupID.y*32u;
#if VIT_STAGE == 0
    const uint K=1024u; uint wb=pc.p1.x;
#elif VIT_STAGE == 1
    const uint K=4096u; uint wb=pc.p1.y;
#else
    const uint K=1024u; uint wb=pc.p1.w;
#endif
    // One block per subgroup, so `m >= live` is the only skip condition and the
    // parent's separate partial-tile path is unreachable: the live token count is a
    // multiple of 8 and every block with m < live covers rows m..m+7 < live.
    if(mA>=pc.p0.x) return;
    MC acc0=MC(0.0),acc1=MC(0.0);
    {
        MB bA0,bA1,bB0,bB1;
        coopMatLoad(bA0,w,vit_wbase(wb,n0,0u,K),16u,CM);
        coopMatLoad(bA1,w,vit_wbase(wb,n0+16u,0u,K),16u,CM);
        coopMatLoad(bB0,w,vit_wbase(wb,n0,16u,K),16u,CM);
        coopMatLoad(bB1,w,vit_wbase(wb,n0+16u,16u,K),16u,CM);
        for(uint k0=0u;k0+32u<K;k0+=32u) {
            MA af;
            coopMatLoad(af,aHidden,pc.p5.z+mA*K+k0,K,RM);
            acc0=coopMatMulAdd(af,bA0,acc0);
            acc1=coopMatMulAdd(af,bA1,acc1);
            coopMatLoad(bA0,w,vit_wbase(wb,n0,k0+32u,K),16u,CM);
            coopMatLoad(bA1,w,vit_wbase(wb,n0+16u,k0+32u,K),16u,CM);
            coopMatLoad(af,aHidden,pc.p5.z+mA*K+k0+16u,K,RM);
            acc0=coopMatMulAdd(af,bB0,acc0);
            acc1=coopMatMulAdd(af,bB1,acc1);
            coopMatLoad(bB0,w,vit_wbase(wb,n0,k0+48u,K),16u,CM);
            coopMatLoad(bB1,w,vit_wbase(wb,n0+16u,k0+48u,K),16u,CM);
        }
        // Last two K steps use already loaded slots; no tail prefetch.
        {
            MA af;
            coopMatLoad(af,aHidden,pc.p5.z+mA*K+K-32u,K,RM);
            acc0=coopMatMulAdd(af,bA0,acc0);
            acc1=coopMatMulAdd(af,bA1,acc1);
            coopMatLoad(af,aHidden,pc.p5.z+mA*K+K-16u,K,RM);
            acc0=coopMatMulAdd(af,bB0,acc0);
            acc1=coopMatMulAdd(af,bB1,acc1);
        }
    }
    for(uint side=0u;side<2u;++side) {
#if VIT_STAGE == 0
        MH result;
        if(side==0u)result=MH(acc0);else result=MH(acc1);
        if(pc.p0.w!=0u)
            coopMatStore(result,dbg,pc.p6.z+pc.p3.x+mA*4096u+n0+side*16u,4096u,RM);
        for(uint c=0u;c<result.length();++c)result[c]=vit_cubic(result[c]);
        coopMatStore(result,aHidden,pc.p5.z+mA*4096u+n0+side*16u,4096u,RM);
#else
        if(side==0u)vit_direct_epilogue(acc0,mA,side,n0,lane);
        else vit_direct_epilogue(acc1,mA,side,n0,lane);
#endif
    }
}
