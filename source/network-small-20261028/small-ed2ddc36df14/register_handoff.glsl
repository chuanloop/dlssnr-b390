// Intel B580 actual lane map; bound to accompanying compiler/driver/SPV identity.
// All subgroup operations require uniform participation. No shared/global/private transport.
#ifndef NRH_REGISTER_HANDOFF_GLSL
#define NRH_REGISTER_HANDOFF_GLSL
#extension GL_KHR_shader_subgroup_shuffle : require
#define NRH_A coopmat<float16_t,gl_ScopeSubgroup,8,16,gl_MatrixUseA>
#define NRH_B coopmat<float16_t,gl_ScopeSubgroup,16,16,gl_MatrixUseB>
#define NRH_C coopmat<float,gl_ScopeSubgroup,8,16,gl_MatrixUseAccumulator>
#define NRH_CH coopmat<float16_t,gl_ScopeSubgroup,8,16,gl_MatrixUseAccumulator>
#define NRH_A_COMPONENTS 4
#define NRH_B_COMPONENTS 8
#define NRH_C_COMPONENTS 4
#define NRH_CH_COMPONENTS 4
uint nrh_pack2(float16_t lo,float16_t hi){return packFloat2x16(f16vec2(lo,hi));}
// Every caller, including scalar unpack, requires all 32 lanes to participate.
f16vec2 nrh_unpack2(uint p){return unpackFloat2x16(p);}
uvec2 nrh_coord_A(uint component){
uint x=(gl_SubgroupInvocationID<<2u)|component;
uint z=(((x&3u)<<5u)^((x&124u)>>2u));return uvec2(z/16u,z%16u);
}
uint nrh_source_A(uint coordinate){return (((coordinate&31u)<<2u)^((coordinate&96u)>>5u));}
uvec2 nrh_coord_B(uint component){
uint x=(gl_SubgroupInvocationID<<3u)|component;
uint z=(((x&7u)<<5u)^((x&8u)<<1u)^((x&240u)>>4u));return uvec2(z/16u,z%16u);
}
uint nrh_source_B(uint coordinate){return (((coordinate&15u)<<4u)^((coordinate&16u)>>1u)^((coordinate&224u)>>5u));}
uvec2 nrh_coord_C(uint component){
uint x=(gl_SubgroupInvocationID<<2u)|component;
uint z=(((x&3u)<<5u)^((x&124u)>>2u));return uvec2(z/16u,z%16u);
}
uint nrh_source_C(uint coordinate){return (((coordinate&31u)<<2u)^((coordinate&96u)>>5u));}
bool nrh_owner_C(uint component){uint x=(gl_SubgroupInvocationID<<2u)|component;return 1u!=0u;}
uvec2 nrh_coord_CH(uint component){
uint x=(gl_SubgroupInvocationID<<2u)|component;
uint z=(((x&3u)<<5u)^((x&124u)>>2u));return uvec2(z/16u,z%16u);
}
uint nrh_source_CH(uint coordinate){return (((coordinate&31u)<<2u)^((coordinate&96u)>>5u));}
bool nrh_owner_CH(uint component){uint x=(gl_SubgroupInvocationID<<2u)|component;return 1u!=0u;}
// Measured same-query parity relation. partner returns a subgroup lane ID.
uint nrh_feature_parity(){return (((gl_SubgroupInvocationID&16u)>>4u));}
uint nrh_partner_feature_parity(){return (16u^(gl_SubgroupInvocationID&31u));}
float16_t nrh_get_CH(NRH_CH x,uint row,uint col){
uint source=nrh_source_CH(row*16u+col);uint sl=source/4u;uint sc=source%4u;
uint p0=subgroupShuffle(nrh_pack2(x[0],x[1]),sl);
uint p1=subgroupShuffle(nrh_pack2(x[2],x[3]),sl);
f16vec2 u=nrh_unpack2((sc<2u?p0:p1));return (sc&1u)==0u?u.x:u.y;
}
float nrh_get_C(NRH_C x,uint row,uint col){
uint source=nrh_source_C(row*16u+col);uint sl=source/4u;uint sc=source%4u;
uint p0=subgroupShuffle(floatBitsToUint(x[0]),sl);
uint p1=subgroupShuffle(floatBitsToUint(x[1]),sl);
uint p2=subgroupShuffle(floatBitsToUint(x[2]),sl);
uint p3=subgroupShuffle(floatBitsToUint(x[3]),sl);
return uintBitsToFloat((sc==0u?p0:(sc==1u?p1:(sc==2u?p2:p3))));
}
// row/col/value must be subgroup uniform for canonical set.
void nrh_set_CH_element(inout NRH_CH x,uint row,uint col,float16_t value){
{uvec2 rc=nrh_coord_CH(0u);if(rc.x==row&&rc.y==col)x[0]=value;}
{uvec2 rc=nrh_coord_CH(1u);if(rc.x==row&&rc.y==col)x[1]=value;}
{uvec2 rc=nrh_coord_CH(2u);if(rc.x==row&&rc.y==col)x[2]=value;}
{uvec2 rc=nrh_coord_CH(3u);if(rc.x==row&&rc.y==col)x[3]=value;}
}
NRH_CH nrh_materialize(NRH_CH h){return h;}
NRH_CH nrh_narrow(NRH_C x){return NRH_CH(x);}
NRH_A nrh_to_A(NRH_CH x){NRH_A o=NRH_A(float16_t(0.0));
uint f0p0=nrh_pack2(x[0],x[1]);
uint f0p1=nrh_pack2(x[2],x[3]);
{
uint sl=((gl_SubgroupInvocationID&31u));
uint v0=f0p0;
f16vec2 u=nrh_unpack2(v0);
o[0]=u.x;
}
{
uint sl=((gl_SubgroupInvocationID&31u));
uint v0=f0p0;
f16vec2 u=nrh_unpack2(v0);
o[1]=u.y;
}
{
uint sl=((gl_SubgroupInvocationID&31u));
uint v1=f0p1;
f16vec2 u=nrh_unpack2(v1);
o[2]=u.x;
}
{
uint sl=((gl_SubgroupInvocationID&31u));
uint v1=f0p1;
f16vec2 u=nrh_unpack2(v1);
o[3]=u.y;
}
return o;}
NRH_B nrh_join_rows_to_B(NRH_CH lo,NRH_CH hi){NRH_B o=NRH_B(float16_t(0.0));
uint f0p0=nrh_pack2(lo[0],lo[1]);
uint f0p1=nrh_pack2(lo[2],lo[3]);
uint f1p0=nrh_pack2(hi[0],hi[1]);
uint f1p1=nrh_pack2(hi[2],hi[3]);
uint sl=(((gl_SubgroupInvocationID&1u)<<4u)^((gl_SubgroupInvocationID&30u)>>1u));
{
uint v0=subgroupShuffle(f0p0,sl);
f16vec2 u=nrh_unpack2(v0);
o[0]=u.x;
o[1]=u.y;
}
{
uint v1=subgroupShuffle(f0p1,sl);
f16vec2 u=nrh_unpack2(v1);
o[2]=u.x;
o[3]=u.y;
}
{
uint v2=subgroupShuffle(f1p0,sl);
f16vec2 u=nrh_unpack2(v2);
o[4]=u.x;
o[5]=u.y;
}
{
uint v3=subgroupShuffle(f1p1,sl);
f16vec2 u=nrh_unpack2(v3);
o[6]=u.x;
o[7]=u.y;
}
return o;}
NRH_B nrh_transpose_to_B(NRH_CH lo,NRH_CH hi){NRH_B o=NRH_B(float16_t(0.0));
uint f0p0=nrh_pack2(lo[0],lo[1]);
uint f0p1=nrh_pack2(lo[2],lo[3]);
uint f1p0=nrh_pack2(hi[0],hi[1]);
uint f1p1=nrh_pack2(hi[2],hi[3]);
{
uint sl=(((gl_SubgroupInvocationID&2u)<<3u)^(gl_SubgroupInvocationID&1u));
uint v0=subgroupShuffle(f0p0,sl);
uint v1=subgroupShuffle(f0p1,sl);
uint v2=subgroupShuffle(f1p0,sl);
uint v3=subgroupShuffle(f1p1,sl);
uint key=(((gl_SubgroupInvocationID&24u)>>3u));
f16vec2 u=nrh_unpack2((key==0u?v0:(key==1u?v1:(key==2u?v2:v3))));
o[0]=((((gl_SubgroupInvocationID&4u)>>2u))==0u?u.x:u.y);
}
{
uint sl=(2u^((gl_SubgroupInvocationID&2u)<<3u)^(gl_SubgroupInvocationID&1u));
uint v0=subgroupShuffle(f0p0,sl);
uint v1=subgroupShuffle(f0p1,sl);
uint v2=subgroupShuffle(f1p0,sl);
uint v3=subgroupShuffle(f1p1,sl);
uint key=(((gl_SubgroupInvocationID&24u)>>3u));
f16vec2 u=nrh_unpack2((key==0u?v0:(key==1u?v1:(key==2u?v2:v3))));
o[1]=((((gl_SubgroupInvocationID&4u)>>2u))==0u?u.x:u.y);
}
{
uint sl=(4u^((gl_SubgroupInvocationID&2u)<<3u)^(gl_SubgroupInvocationID&1u));
uint v0=subgroupShuffle(f0p0,sl);
uint v1=subgroupShuffle(f0p1,sl);
uint v2=subgroupShuffle(f1p0,sl);
uint v3=subgroupShuffle(f1p1,sl);
uint key=(((gl_SubgroupInvocationID&24u)>>3u));
f16vec2 u=nrh_unpack2((key==0u?v0:(key==1u?v1:(key==2u?v2:v3))));
o[2]=((((gl_SubgroupInvocationID&4u)>>2u))==0u?u.x:u.y);
}
{
uint sl=(6u^((gl_SubgroupInvocationID&2u)<<3u)^(gl_SubgroupInvocationID&1u));
uint v0=subgroupShuffle(f0p0,sl);
uint v1=subgroupShuffle(f0p1,sl);
uint v2=subgroupShuffle(f1p0,sl);
uint v3=subgroupShuffle(f1p1,sl);
uint key=(((gl_SubgroupInvocationID&24u)>>3u));
f16vec2 u=nrh_unpack2((key==0u?v0:(key==1u?v1:(key==2u?v2:v3))));
o[3]=((((gl_SubgroupInvocationID&4u)>>2u))==0u?u.x:u.y);
}
{
uint sl=(8u^((gl_SubgroupInvocationID&2u)<<3u)^(gl_SubgroupInvocationID&1u));
uint v0=subgroupShuffle(f0p0,sl);
uint v1=subgroupShuffle(f0p1,sl);
uint v2=subgroupShuffle(f1p0,sl);
uint v3=subgroupShuffle(f1p1,sl);
uint key=(((gl_SubgroupInvocationID&24u)>>3u));
f16vec2 u=nrh_unpack2((key==0u?v0:(key==1u?v1:(key==2u?v2:v3))));
o[4]=((((gl_SubgroupInvocationID&4u)>>2u))==0u?u.x:u.y);
}
{
uint sl=(10u^((gl_SubgroupInvocationID&2u)<<3u)^(gl_SubgroupInvocationID&1u));
uint v0=subgroupShuffle(f0p0,sl);
uint v1=subgroupShuffle(f0p1,sl);
uint v2=subgroupShuffle(f1p0,sl);
uint v3=subgroupShuffle(f1p1,sl);
uint key=(((gl_SubgroupInvocationID&24u)>>3u));
f16vec2 u=nrh_unpack2((key==0u?v0:(key==1u?v1:(key==2u?v2:v3))));
o[5]=((((gl_SubgroupInvocationID&4u)>>2u))==0u?u.x:u.y);
}
{
uint sl=(12u^((gl_SubgroupInvocationID&2u)<<3u)^(gl_SubgroupInvocationID&1u));
uint v0=subgroupShuffle(f0p0,sl);
uint v1=subgroupShuffle(f0p1,sl);
uint v2=subgroupShuffle(f1p0,sl);
uint v3=subgroupShuffle(f1p1,sl);
uint key=(((gl_SubgroupInvocationID&24u)>>3u));
f16vec2 u=nrh_unpack2((key==0u?v0:(key==1u?v1:(key==2u?v2:v3))));
o[6]=((((gl_SubgroupInvocationID&4u)>>2u))==0u?u.x:u.y);
}
{
uint sl=(14u^((gl_SubgroupInvocationID&2u)<<3u)^(gl_SubgroupInvocationID&1u));
uint v0=subgroupShuffle(f0p0,sl);
uint v1=subgroupShuffle(f0p1,sl);
uint v2=subgroupShuffle(f1p0,sl);
uint v3=subgroupShuffle(f1p1,sl);
uint key=(((gl_SubgroupInvocationID&24u)>>3u));
f16vec2 u=nrh_unpack2((key==0u?v0:(key==1u?v1:(key==2u?v2:v3))));
o[7]=((((gl_SubgroupInvocationID&4u)>>2u))==0u?u.x:u.y);
}
return o;}
NRH_A nrh_transpose_to_A_lo(NRH_CH lo,NRH_CH hi){NRH_A o=NRH_A(float16_t(0.0));
uint f0p0=nrh_pack2(lo[0],lo[1]);
uint f0p1=nrh_pack2(lo[2],lo[3]);
uint f1p0=nrh_pack2(hi[0],hi[1]);
uint f1p1=nrh_pack2(hi[2],hi[3]);
{
uint sl=(((gl_SubgroupInvocationID&1u)<<4u)^((gl_SubgroupInvocationID&16u)>>4u));
uint v0=subgroupShuffle(f0p0,sl);
uint v1=subgroupShuffle(f0p1,sl);
uint v2=subgroupShuffle(f1p0,sl);
uint v3=subgroupShuffle(f1p1,sl);
uint key=(((gl_SubgroupInvocationID&12u)>>2u));
f16vec2 u=nrh_unpack2((key==0u?v0:(key==1u?v1:(key==2u?v2:v3))));
o[0]=((((gl_SubgroupInvocationID&2u)>>1u))==0u?u.x:u.y);
}
{
uint sl=(2u^((gl_SubgroupInvocationID&1u)<<4u)^((gl_SubgroupInvocationID&16u)>>4u));
uint v0=subgroupShuffle(f0p0,sl);
uint v1=subgroupShuffle(f0p1,sl);
uint v2=subgroupShuffle(f1p0,sl);
uint v3=subgroupShuffle(f1p1,sl);
uint key=(((gl_SubgroupInvocationID&12u)>>2u));
f16vec2 u=nrh_unpack2((key==0u?v0:(key==1u?v1:(key==2u?v2:v3))));
o[1]=((((gl_SubgroupInvocationID&2u)>>1u))==0u?u.x:u.y);
}
{
uint sl=(4u^((gl_SubgroupInvocationID&1u)<<4u)^((gl_SubgroupInvocationID&16u)>>4u));
uint v0=subgroupShuffle(f0p0,sl);
uint v1=subgroupShuffle(f0p1,sl);
uint v2=subgroupShuffle(f1p0,sl);
uint v3=subgroupShuffle(f1p1,sl);
uint key=(((gl_SubgroupInvocationID&12u)>>2u));
f16vec2 u=nrh_unpack2((key==0u?v0:(key==1u?v1:(key==2u?v2:v3))));
o[2]=((((gl_SubgroupInvocationID&2u)>>1u))==0u?u.x:u.y);
}
{
uint sl=(6u^((gl_SubgroupInvocationID&1u)<<4u)^((gl_SubgroupInvocationID&16u)>>4u));
uint v0=subgroupShuffle(f0p0,sl);
uint v1=subgroupShuffle(f0p1,sl);
uint v2=subgroupShuffle(f1p0,sl);
uint v3=subgroupShuffle(f1p1,sl);
uint key=(((gl_SubgroupInvocationID&12u)>>2u));
f16vec2 u=nrh_unpack2((key==0u?v0:(key==1u?v1:(key==2u?v2:v3))));
o[3]=((((gl_SubgroupInvocationID&2u)>>1u))==0u?u.x:u.y);
}
return o;}
NRH_A nrh_transpose_to_A_hi(NRH_CH lo,NRH_CH hi){NRH_A o=NRH_A(float16_t(0.0));
uint f0p0=nrh_pack2(lo[0],lo[1]);
uint f0p1=nrh_pack2(lo[2],lo[3]);
uint f1p0=nrh_pack2(hi[0],hi[1]);
uint f1p1=nrh_pack2(hi[2],hi[3]);
{
uint sl=(8u^((gl_SubgroupInvocationID&1u)<<4u)^((gl_SubgroupInvocationID&16u)>>4u));
uint v0=subgroupShuffle(f0p0,sl);
uint v1=subgroupShuffle(f0p1,sl);
uint v2=subgroupShuffle(f1p0,sl);
uint v3=subgroupShuffle(f1p1,sl);
uint key=(((gl_SubgroupInvocationID&12u)>>2u));
f16vec2 u=nrh_unpack2((key==0u?v0:(key==1u?v1:(key==2u?v2:v3))));
o[0]=((((gl_SubgroupInvocationID&2u)>>1u))==0u?u.x:u.y);
}
{
uint sl=(10u^((gl_SubgroupInvocationID&1u)<<4u)^((gl_SubgroupInvocationID&16u)>>4u));
uint v0=subgroupShuffle(f0p0,sl);
uint v1=subgroupShuffle(f0p1,sl);
uint v2=subgroupShuffle(f1p0,sl);
uint v3=subgroupShuffle(f1p1,sl);
uint key=(((gl_SubgroupInvocationID&12u)>>2u));
f16vec2 u=nrh_unpack2((key==0u?v0:(key==1u?v1:(key==2u?v2:v3))));
o[1]=((((gl_SubgroupInvocationID&2u)>>1u))==0u?u.x:u.y);
}
{
uint sl=(12u^((gl_SubgroupInvocationID&1u)<<4u)^((gl_SubgroupInvocationID&16u)>>4u));
uint v0=subgroupShuffle(f0p0,sl);
uint v1=subgroupShuffle(f0p1,sl);
uint v2=subgroupShuffle(f1p0,sl);
uint v3=subgroupShuffle(f1p1,sl);
uint key=(((gl_SubgroupInvocationID&12u)>>2u));
f16vec2 u=nrh_unpack2((key==0u?v0:(key==1u?v1:(key==2u?v2:v3))));
o[2]=((((gl_SubgroupInvocationID&2u)>>1u))==0u?u.x:u.y);
}
{
uint sl=(14u^((gl_SubgroupInvocationID&1u)<<4u)^((gl_SubgroupInvocationID&16u)>>4u));
uint v0=subgroupShuffle(f0p0,sl);
uint v1=subgroupShuffle(f0p1,sl);
uint v2=subgroupShuffle(f1p0,sl);
uint v3=subgroupShuffle(f1p1,sl);
uint key=(((gl_SubgroupInvocationID&12u)>>2u));
f16vec2 u=nrh_unpack2((key==0u?v0:(key==1u?v1:(key==2u?v2:v3))));
o[3]=((((gl_SubgroupInvocationID&2u)>>1u))==0u?u.x:u.y);
}
return o;}
NRH_C nrh_widen_exact(NRH_CH x){NRH_C o=NRH_C(0.0);
o[0]=float(x[0]);
o[1]=float(x[1]);
o[2]=float(x[2]);
o[3]=float(x[3]);
return o;}
NRH_A nrh_to_A(NRH_C x){return nrh_to_A(nrh_narrow(x));}
#endif
