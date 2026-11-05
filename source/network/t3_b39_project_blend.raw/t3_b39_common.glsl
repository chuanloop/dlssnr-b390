// T3 block39 fused projection+decups: private copy of
// candidates/amd-map-20261015/s1-project-split-complete/project-k1024/source/
// ioport_project_opt_common.glsl with one change and one addition:
//
//   * `io_write` (projection output, layout 2 canonical NHWC) is gone: the fused
//     kernel no longer materialises that tensor.
//   * `io_write_blend` is added: the decups output, layout 0 over out_W x out_H,
//     which is the parent blend's own `io_write` with its own geometry taken from
//     p0.z/p0.w -> now p5.x/p5.y so the projection's OW/OH stay in p0.z/p0.w.
//
// Everything else - io_at, io_read, io_skip, the binding order and the push field
// meaning for p0..p4 - is the parent's, so the same push words keep the same
// meaning and the io_skip record (skip_W p2.w, skip_H p3.x, layout p2.z) is
// untouched.
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
// All offsets below are typed element offsets, never weights_ht byte offsets.
layout(set=0,binding=0,std430) readonly buffer Weights { float16_t w[]; };
layout(set=0,binding=1,std430) readonly buffer Floats { float f[]; };
layout(set=0,binding=2,std430) readonly buffer Input { float16_t x[]; };
layout(set=0,binding=3,std430) buffer Output { float16_t o[]; };
layout(set=0,binding=4,std430) buffer Debug { float16_t d[]; };
layout(set=0,binding=5,std430) buffer Temporary { float16_t tmp[]; };
layout(set=0,binding=6,std430) readonly buffer Skip { float16_t sk[]; };
layout(set=0,binding=7,std430) buffer Surface { float surface[]; };
layout(push_constant) uniform Push {
    uvec4 p0; // IW,IH,OW,OH                      (projection geometry, unchanged)
    uvec4 p1; // K,N,weight_half,gain_half         (gain_half now the decups record)
    uvec4 p2; // src_layout,dst_layout,skip_layout,skip_W
    uvec4 p3; // skip_H,tmp_layout,valid_W,valid_H
    uvec4 p4; // input_half,output_half,skip_half,tmp_half
    uvec4 p5; // out_W,out_H,upsample_x,upsample_y (decups output geometry)
    uvec4 p6; // controls_float,history_float,motion_float,depth_float
    uvec4 p7; // aux0..aux3; p7.x = decups main gain flag
} pc;
// 0: spatial 4x4 tiles, channels16; 1: raster tokens, channels16;
// 2: canonical NHWC; 3: AMD level-boundary 16-channel planes.
uint io_at(uint px,uint py,uint c,uint W,uint H,uint C,uint layout_id) {
    if(layout_id==2u)return (py*W+px)*C+c;
    if(layout_id==3u) {
        uint sub=(c&1u)|((c>>3u&1u)<<1u)|((c>>1u&1u)<<2u)|((c>>2u&1u)<<3u);
        return (c/16u)*(W*H*16u)+(py*W+px)*16u+sub;
    }
    uint t=layout_id==1u?py*W+px:((py/4u)*(W/4u)+px/4u)*16u+4u*(py%4u)+px%4u;
    return ((t/16u)*(C/16u)+c/16u)*256u+(t%16u)*16u+c%16u;
}
float16_t io_read(uint px,uint py,uint c) {
    return x[pc.p4.x+io_at(px,py,c,pc.p0.x,pc.p0.y,pc.p1.x,pc.p2.x)];
}
float16_t io_skip(uint px,uint py,uint c) {
    return sk[pc.p4.z+io_at(px,py,c,pc.p2.w,pc.p3.x,pc.p1.y,pc.p2.z)];
}
void io_write_blend(uint px,uint py,uint c,float16_t v) {
    o[pc.p4.y+io_at(px,py,c,pc.p5.x,pc.p5.y,pc.p1.y,pc.p2.y)]=v;
}
