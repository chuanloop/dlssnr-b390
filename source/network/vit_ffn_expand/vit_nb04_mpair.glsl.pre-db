// NB04: N32 retained; each B0/B1 serves two separate eight-row accumulators.
layout(local_size_x=256) in;
#include "vit_common.glsl"
#define MH coopmat<float16_t,gl_ScopeSubgroup,8,16,gl_MatrixUseAccumulator>
#if VIT_STAGE != 0
shared float scratch[1024];
#endif
void main() {
    uint sg=gl_SubgroupID,lane=gl_SubgroupInvocationID;
    uint mA=gl_WorkGroupID.x*128u+sg*8u,mB=mA+64u,n0=gl_WorkGroupID.y*32u;
#if VIT_STAGE == 0
    const uint K=1024u; uint wb=pc.p1.x;
#elif VIT_STAGE == 1
    const uint K=4096u; uint wb=pc.p1.y;
#else
    const uint K=1024u; uint wb=pc.p1.w;
#endif
    MC accA0=MC(0.0),accA1=MC(0.0),accB0=MC(0.0),accB1=MC(0.0);
    for(uint k0=0u;k0<K;k0+=16u) {
        MB b0,b1;
        coopMatLoad(b0,w,vit_wbase(wb,n0,k0,K),16u,CM);
        coopMatLoad(b1,w,vit_wbase(wb,n0+16u,k0,K),16u,CM);
        MA af;
        if(mA<pc.p0.x) {
#if VIT_STAGE == 0
            coopMatLoad(af,x,pc.p5.x+mA*K+k0,K,RM);
#elif VIT_STAGE == 1
            coopMatLoad(af,aHidden,pc.p5.z+mA*K+k0,K,RM);
#else
            coopMatLoad(af,context,pc.p6.y+mA*K+k0,K,RM);
#endif
            accA0=coopMatMulAdd(af,b0,accA0);
            accA1=coopMatMulAdd(af,b1,accA1);
        }
        if(mB<pc.p0.x) {
#if VIT_STAGE == 0
            coopMatLoad(af,x,pc.p5.x+mB*K+k0,K,RM);
#elif VIT_STAGE == 1
            coopMatLoad(af,aHidden,pc.p5.z+mB*K+k0,K,RM);
#else
            coopMatLoad(af,context,pc.p6.y+mB*K+k0,K,RM);
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
            for(uint c=0u;c<result.length();++c)result[c]=vit_cubic(result[c]);
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
