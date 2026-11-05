// SSBO image interface: RGBA float32 in raster order. Source/motion/depth
// use texel-centre nearest sampling; history uses the recovered five linear taps.
float ctl(uint i) { return f[pc.p6.x+i]; }
int mirror(int i,int size) {
    int period=2*size;
    int j=i%period;
    if(j<0)j+=period;
    return j<size?j:period-1-j;
}
vec4 rgba(uint base,uint W,uint H,ivec2 xy) {
    uint a=base+uint(mirror(xy.y,int(H)))*W*4u+uint(mirror(xy.x,int(W)))*4u;
    return vec4(f[a],f[a+1u],f[a+2u],f[a+3u]);
}
vec3 hist_linear(vec2 pos) {
    uint W=uint(ctl(9)),H=uint(ctl(10));
    vec2 q=pos-0.5,fr=fract(q);
    ivec2 ij=ivec2(floor(q));
    ivec2 hi=ivec2(int(W)-1,int(H)-1);
    vec3 a=rgba(pc.p6.y,W,H,clamp(ij,ivec2(0),hi)).rgb;
    vec3 b=rgba(pc.p6.y,W,H,clamp(ij+ivec2(1,0),ivec2(0),hi)).rgb;
    vec3 c=rgba(pc.p6.y,W,H,clamp(ij+ivec2(0,1),ivec2(0),hi)).rgb;
    vec3 e=rgba(pc.p6.y,W,H,clamp(ij+ivec2(1),ivec2(0),hi)).rgb;
    return mix(mix(a,b,fr.x),mix(c,e,fr.x),fr.y);
}
vec3 history5(vec2 pos) {
    vec2 lo=vec2(0.5),hi=vec2(ctl(9),ctl(10))-0.5;
    vec2 c=floor(pos-0.5)+0.5,t=clamp(pos-c,vec2(0),vec2(1));
    vec2 t2=t*t,t3=t2*t;
    vec2 w0=(t+t3)*(-0.5)+t2;
    vec2 w1=1.5*t3-2.5*t2+1.0,w3=0.5*(t3-t2),w2=1.0-w0-w1-w3,wc=w1+w2;
    vec2 p0=clamp(c-1.0,lo,hi),pm=clamp(c+w2/wc,lo,hi),p2=clamp(c+2.0,lo,hi);
    float kl=w0.x*wc.y,ku=w0.y*wc.x,kc=wc.x*wc.y,kd=w3.y*wc.x,kr=w3.x*wc.y;
    return (kl*hist_linear(vec2(p0.x,pm.y))+ku*hist_linear(vec2(pm.x,p0.y))
        +kc*hist_linear(pm)+kd*hist_linear(vec2(pm.x,p2.y))+kr*hist_linear(vec2(p2.x,pm.y)))
        /(kl+ku+kc+kd+kr);
}
float depth_at(ivec2 xy) {
    uint W=pc.p5.x,H=pc.p5.y;
    return f[pc.p6.w+uint(mirror(xy.y,int(H)))*W+uint(mirror(xy.x,int(W)))];
}
vec2 motion_uv(vec2 uv) {
    uint W=pc.p5.x,H=pc.p5.y;
    vec2 q=(uv*vec2(ctl(15),ctl(16))+vec2(ctl(17),ctl(18)))*vec2(W,H);
    ivec2 xy=ivec2(floor(q));
    uint a=pc.p6.z+(uint(mirror(xy.y,int(H)))*W+uint(mirror(xy.x,int(W))))*2u;
    return vec2(f[a]*ctl(13),f[a+1u]*ctl(14));
}
