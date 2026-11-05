// Intel B580 sealed CH (8x16 half accumulator) lane map, used by this kernel
// only to run the parent's own Q/K norm tree inside the projecting subgroup.
//
// Provenance: B5 already ships this map in
// candidates/zcode-fp16-four-hotspots-20261020/wide-c64-norm-owner-v3/source/
// register_handoff.glsl (sha256 dd8307b1c59c76e41d705442207fafd1c87e2a87fe57d04f343af5c27caa5f26;
// nrh_coord_CH and nr_norm_component), where (row = 2*component + lane/16,
// col = lane%16) was used to execute the same 32-square/31-add Q/K norm tree on
// two half fragments with a col-0 owner and a lane&16 broadcast, and passed the
// full numeric gate. The deployed S24 KIND2 pool in the same B5 plan documents
// and uses the same map. This file is a physical copy of the needed relations.
#ifndef VIT_QKV_NORM_OWNER_HANDOFF_GLSL
#define VIT_QKV_NORM_OWNER_HANDOFF_GLSL
// component ci of a lane holds (row, col) = (2*ci + lane/16, lane%16).
uint vq_row(uint component,uint lane){return 2u*component+(lane>>4u);}
uint vq_col(uint lane){return lane&15u;}
// One row of the parent norm: s[d] = z[d]*z[d] + z[d+16]*z[d+16], with the two
// half multiplications and their half add rounded exactly as the parent's
// "precise float16_t a=..*..; precise float16_t b=..*..; s[d]=a+b;".
// Packing two components per f16vec2 packs two independent token rows; it does
// not join two rows of one reduction.
f16vec2 vq_pair_sum(f16vec2 lo,f16vec2 hi){
    precise f16vec2 a=lo*lo;
    precise f16vec2 b=hi*hi;
    precise f16vec2 s=a+b;
    return s;
}
// Parent tree over d=0..15: for step=8,4,2,1  s[d]=s[d]+s[d+step] (d<step).
// XOR shuffles inside the owning subgroup: lane d holds s[d] after every step,
// and the d<step predicates keep the parent's exact node order. Only the owner
// columns (lane&15==0) complete the whole tree; the other columns hold the
// same intermediate nodes the parent's scalar loop holds, so the reciprocal is
// evaluated in the owner and broadcast (v3 form) instead of being read from a
// partial node.
f16vec2 vq_tree_xor(f16vec2 s,uint lane){
    uint w=packFloat2x16(s);
    uint o=subgroupShuffleXor(w,8u);
    if((lane&15u)<8u){s=s+unpackFloat2x16(o);}
    w=packFloat2x16(s);
    o=subgroupShuffleXor(w,4u);
    if((lane&15u)<4u){s=s+unpackFloat2x16(o);}
    w=packFloat2x16(s);
    o=subgroupShuffleXor(w,2u);
    if((lane&15u)<2u){s=s+unpackFloat2x16(o);}
    w=packFloat2x16(s);
    o=subgroupShuffleXor(w,1u);
    if((lane&15u)==0u){s=s+unpackFloat2x16(o);}
    return s;
}
// Parent: inv = float16_t(inversesqrt(float(max(s[0], float16_t(0.000062))))).
// Evaluated once per row in the column-0 owner (lane 0 or lane 16); all sixteen
// column consumers receive that value from lane&16.
f16vec2 vq_norm_rcp(f16vec2 s,uint lane){
    f16vec2 n=f16vec2(0.0);
    if((lane&15u)==0u)
        n=f16vec2(float16_t(inversesqrt(float(max(s.x,float16_t(0.000062))))),
                  float16_t(inversesqrt(float(max(s.y,float16_t(0.000062))))));
    return unpackFloat2x16(subgroupShuffle(packFloat2x16(n),lane&16u));
}
// Parent Q: z=z*5.65625; z=z*inv; z=z*scale (three half multiplies, in order).
float16_t vq_scale_q(float16_t z,float16_t inv,float16_t scale){
    precise float16_t a=z*float16_t(5.65625);
    precise float16_t b=a*inv;
    precise float16_t c=b*scale;
    return c;
}
// Parent K: z=z*inv.
float16_t vq_scale_k(float16_t z,float16_t inv){
    precise float16_t b=z*inv;
    return b;
}
#endif
