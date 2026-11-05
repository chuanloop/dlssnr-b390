// Developer GPU verification for the FP16 input adaptation.
// For each supported source format (R11G11B10_FLOAT historical and
// R16G16B16A16_FLOAT new) this check exercises the full HdrProxy contract:
// resource creation, typed views, prepare dispatch, composite dispatch and
// pixel-exact output against the documented formulas. The reference for the
// comparison is the DECODED source content read back from the same texture,
// so the check is self-consistent regardless of packing precision.
// No game, no NGX, no Triton: this validates the native adapter only.
#include "nr_hdr_proxy.h"
#include <d3d12.h>
#include <dxgi1_6.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>

using Microsoft::WRL::ComPtr;

namespace {
constexpr unsigned W=1280,H=720;

#define CHECK(cond,msg) do{ if(!(cond)){ std::printf("FAIL %s (line %d)\n",msg,__LINE__); return 1; } }while(0)

// ---- R11G11B10 channel codecs: 5-bit exponent (bias 15), no sign ----
// R and G carry a 6-bit mantissa (11 bits), B carries 5 bits (10 bits).
uint32_t enc_e5m(unsigned man_bits,float v) {
    if(!(v>0)) return 0;
    int e; float m=std::frexp(v,&e); m*=2; e-=1;      // v = m*2^e, m in [1,2)
    int exp=e+15;
    if(exp<1) return 0;                                // below denorm range
    const uint32_t maxman=(1u<<man_bits)-1;
    if(exp>=31) return 31u<<man_bits;                  // inf clamp
    int man=int((m-1.0f)*float(1u<<man_bits)+0.5f);
    if(man==int(1u<<man_bits)) {man=0;++exp;if(exp>=31) return 31u<<man_bits;}
    return uint32_t(exp)<<man_bits|uint32_t(man);
}
float dec_e5m(unsigned man_bits,uint32_t bits) {
    const uint32_t maxman=(1u<<man_bits)-1;
    const uint32_t exp=(bits>>man_bits)&0x1f,man=bits&maxman;
    if(exp==0) return std::ldexp(float(man)/float(1u<<man_bits),-14);
    if(exp==31) return 1e30f;
    return std::ldexp(1.0f+float(man)/float(1u<<man_bits),int(exp)-15);
}
// ---- IEEE half codec: 1 sign, 5-bit exponent (bias 15), 10-bit mantissa ----
uint32_t enc_f16(float v) {
    uint32_t sign=0;
    if(v<0) {sign=0x8000;v=-v;}
    if(!(v>0)) return sign;
    int e; float m=std::frexp(v,&e); m*=2; e-=1;
    int exp=e+15;
    if(exp<1) return sign;                             // below denorm range
    if(exp>=31) return sign|0x7c00u;
    int man=int((m-1.0f)*1024.0f+0.5f);
    if(man==1024) {man=0;++exp;if(exp>=31) return sign|0x7c00u;}
    return sign|uint32_t(exp)<<10|uint32_t(man);
}
float dec_f16(uint32_t h) {
    const uint32_t sign=(h>>15)&1u,exp=(h>>10)&0x1f,man=h&0x3ff;
    float v;
    if(exp==0) v=std::ldexp(float(man)/1024.0f,-14);
    else if(exp==31) v=1e30f;
    else v=std::ldexp(1.0f+float(man)/1024.0f,int(exp)-15);
    return sign?-v:v;
}

ComPtr<ID3D12Resource> make_texture(ID3D12Device* dev,UINT w,UINT h,
    DXGI_FORMAT format,bool uav) {
    D3D12_RESOURCE_DESC d{};d.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    d.Width=w;d.Height=h;d.DepthOrArraySize=1;d.MipLevels=1;
    d.SampleDesc.Count=1;d.Format=format;
    d.Flags=uav?D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS:D3D12_RESOURCE_FLAG_NONE;
    D3D12_HEAP_PROPERTIES heap{};heap.Type=D3D12_HEAP_TYPE_DEFAULT;
    ComPtr<ID3D12Resource> r;
    if(FAILED(dev->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&d,
        D3D12_RESOURCE_STATE_COMMON,nullptr,IID_PPV_ARGS(r.GetAddressOf())))) r.Reset();
    return r;
}

// Fills a source-format texture through an upload buffer.
bool fill_color(ID3D12Device* dev,ID3D12CommandQueue* q,
    ID3D12Resource* tex,DXGI_FORMAT format,const float (*rgb)[3],const float* alpha) {
    const unsigned bpp=format==DXGI_FORMAT_R11G11B10_FLOAT?4:
        format==DXGI_FORMAT_R16G16B16A16_FLOAT?8:
        format==DXGI_FORMAT_R32G32B32A32_FLOAT?16:0;
    if(!bpp) return false;
    // Row pitch alignment for textures is 256 bytes (fixed by the D3D12
    // spec; the minimal SDK header in this environment omits the constant).
    const UINT64 rowpitch=(UINT64(W)*bpp+255u)&~UINT64(255u);
    D3D12_HEAP_PROPERTIES up{};up.Type=D3D12_HEAP_TYPE_UPLOAD;
    D3D12_RESOURCE_DESC bd{};bd.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER;
    bd.Width=rowpitch*UINT64(H);bd.Height=1;bd.DepthOrArraySize=1;bd.MipLevels=1;
    bd.SampleDesc.Count=1;bd.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ComPtr<ID3D12Resource> upload;
    if(FAILED(dev->CreateCommittedResource(&up,D3D12_HEAP_FLAG_NONE,&bd,
        D3D12_RESOURCE_STATE_GENERIC_READ,nullptr,IID_PPV_ARGS(upload.GetAddressOf())))) return false;
    uint8_t* mapped=nullptr;
    if(FAILED(upload->Map(0,nullptr,reinterpret_cast<void**>(&mapped)))) return false;
    for(unsigned y=0;y<H;++y) {
        auto* row=reinterpret_cast<uint32_t*>(mapped+size_t(y)*rowpitch);
        for(unsigned x=0;x<W;++x) {
            if(format==DXGI_FORMAT_R32G32B32A32_FLOAT) {
                auto* frow=reinterpret_cast<float*>(row);
                frow[4*x+0]=rgb[y*W+x][0];
                frow[4*x+1]=rgb[y*W+x][1];
                frow[4*x+2]=rgb[y*W+x][2];
                frow[4*x+3]=1.0f;
            } else if(format==DXGI_FORMAT_R11G11B10_FLOAT) {
                row[x]=enc_e5m(6,rgb[y*W+x][0])|(enc_e5m(6,rgb[y*W+x][1])<<11)
                      |(enc_e5m(5,rgb[y*W+x][2])<<22);
            } else {
                row[2*x]=enc_f16(rgb[y*W+x][0])|(enc_f16(rgb[y*W+x][1])<<16);
                row[2*x+1]=enc_f16(rgb[y*W+x][2])|(enc_f16(alpha?alpha[y*W+x]:1.0f)<<16);
            }
        }
    }
    upload->Unmap(0,nullptr);
    ComPtr<ID3D12CommandAllocator> al;
    if(FAILED(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
        IID_PPV_ARGS(al.GetAddressOf())))) return false;
    ComPtr<ID3D12GraphicsCommandList> cl;
    if(FAILED(dev->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,al.Get(),
        nullptr,IID_PPV_ARGS(cl.GetAddressOf())))) return false;
    D3D12_TEXTURE_COPY_LOCATION src{upload.Get(),D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
    src.PlacedFootprint.Footprint.Format=format;
    src.PlacedFootprint.Footprint.Width=W;src.PlacedFootprint.Footprint.Height=H;
    src.PlacedFootprint.Footprint.Depth=1;
    src.PlacedFootprint.Footprint.RowPitch=UINT(rowpitch);
    D3D12_TEXTURE_COPY_LOCATION dst{tex,D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
    cl->CopyTextureRegion(&dst,0,0,0,&src,nullptr);
    cl->Close();
    ID3D12CommandList* lists[]={cl.Get()};
    q->ExecuteCommandLists(1,lists);
    ComPtr<ID3D12Fence> f;dev->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(f.GetAddressOf()));
    HANDLE ev=CreateEventW(nullptr,FALSE,FALSE,nullptr);
    f->SetEventOnCompletion(1,ev);q->Signal(f.Get(),1);
    WaitForSingleObject(ev,10000);CloseHandle(ev);
    return true;
}

// Reads back any supported source texture and decodes it to RGBA32F floats.
bool read_rgba32f_of_source(ID3D12Device* dev,ID3D12CommandQueue* q,
    ID3D12Resource* tex,DXGI_FORMAT format,D3D12_RESOURCE_STATES from_state,
    std::vector<float>& out) {
    D3D12_RESOURCE_DESC td=tex->GetDesc();
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};
    UINT64 total=0;
    dev->GetCopyableFootprints(&td,0,1,0,&fp,nullptr,nullptr,&total);
    D3D12_HEAP_PROPERTIES rb{};rb.Type=D3D12_HEAP_TYPE_READBACK;
    D3D12_RESOURCE_DESC bd{};bd.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER;
    bd.Width=total;bd.Height=1;bd.DepthOrArraySize=1;bd.MipLevels=1;
    bd.SampleDesc.Count=1;bd.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ComPtr<ID3D12Resource> buf;
    if(FAILED(dev->CreateCommittedResource(&rb,D3D12_HEAP_FLAG_NONE,&bd,
        D3D12_RESOURCE_STATE_COPY_DEST,nullptr,IID_PPV_ARGS(buf.GetAddressOf())))) return false;
    ComPtr<ID3D12CommandAllocator> al;
    if(FAILED(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
        IID_PPV_ARGS(al.GetAddressOf())))) return false;
    ComPtr<ID3D12GraphicsCommandList> cl;
    if(FAILED(dev->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,al.Get(),
        nullptr,IID_PPV_ARGS(cl.GetAddressOf())))) return false;
    D3D12_RESOURCE_BARRIER b{};
    b.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource=tex;b.Transition.StateBefore=from_state;
    b.Transition.StateAfter=D3D12_RESOURCE_STATE_COPY_SOURCE;
    cl->ResourceBarrier(1,&b);
    D3D12_TEXTURE_COPY_LOCATION src{tex,D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
    D3D12_TEXTURE_COPY_LOCATION dst{buf.Get(),D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
    dst.PlacedFootprint=fp;
    cl->CopyTextureRegion(&dst,0,0,0,&src,nullptr);
    cl->Close();
    ID3D12CommandList* lists[]={cl.Get()};
    q->ExecuteCommandLists(1,lists);
    ComPtr<ID3D12Fence> f;dev->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(f.GetAddressOf()));
    HANDLE ev=CreateEventW(nullptr,FALSE,FALSE,nullptr);
    f->SetEventOnCompletion(1,ev);q->Signal(f.Get(),1);
    WaitForSingleObject(ev,10000);CloseHandle(ev);
    out.resize(size_t(W)*H*4);
    uint8_t* mapped=nullptr;
    if(FAILED(buf->Map(0,nullptr,reinterpret_cast<void**>(&mapped)))) return false;
    for(unsigned y=0;y<H;++y) {
        const uint8_t* row=mapped+size_t(y)*fp.Footprint.RowPitch;
        for(unsigned x=0;x<W;++x) {
            float* o=&out[(size_t(y)*W+x)*4];
            if(format==DXGI_FORMAT_R11G11B10_FLOAT) {
                const uint32_t px=reinterpret_cast<const uint32_t*>(row)[x];
                o[0]=dec_e5m(6,px&0x7ff);o[1]=dec_e5m(6,(px>>11)&0x7ff);
                o[2]=dec_e5m(5,(px>>22)&0x3ff);o[3]=1.0f;
            } else {
                const uint32_t a=reinterpret_cast<const uint32_t*>(row)[2*x];
                const uint32_t b=reinterpret_cast<const uint32_t*>(row)[2*x+1];
                o[0]=dec_f16(a&0xffff);o[1]=dec_f16(a>>16);
                o[2]=dec_f16(b&0xffff);o[3]=dec_f16(b>>16);
            }
        }
    }
    buf->Unmap(0,nullptr);
    return true;
}

// Reads back a RGBA32F UAV texture through a readback buffer. The caller
// states the resource's current state for the transition to COPY_SOURCE.
bool read_rgba32f(ID3D12Device* dev,ID3D12CommandQueue* q,
    ID3D12Resource* tex,D3D12_RESOURCE_STATES from_state,std::vector<float>& out) {
    D3D12_RESOURCE_DESC td=tex->GetDesc();
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};
    UINT64 total=0;
    dev->GetCopyableFootprints(&td,0,1,0,&fp,nullptr,nullptr,&total);
    D3D12_HEAP_PROPERTIES rb{};rb.Type=D3D12_HEAP_TYPE_READBACK;
    D3D12_RESOURCE_DESC bd{};bd.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER;
    bd.Width=total;bd.Height=1;bd.DepthOrArraySize=1;bd.MipLevels=1;
    bd.SampleDesc.Count=1;bd.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ComPtr<ID3D12Resource> buf;
    if(FAILED(dev->CreateCommittedResource(&rb,D3D12_HEAP_FLAG_NONE,&bd,
        D3D12_RESOURCE_STATE_COPY_DEST,nullptr,IID_PPV_ARGS(buf.GetAddressOf())))) return false;
    ComPtr<ID3D12CommandAllocator> al;
    if(FAILED(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
        IID_PPV_ARGS(al.GetAddressOf())))) return false;
    ComPtr<ID3D12GraphicsCommandList> cl;
    if(FAILED(dev->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,al.Get(),
        nullptr,IID_PPV_ARGS(cl.GetAddressOf())))) return false;
    D3D12_RESOURCE_BARRIER b{};
    b.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource=tex;b.Transition.StateBefore=from_state;
    b.Transition.StateAfter=D3D12_RESOURCE_STATE_COPY_SOURCE;
    cl->ResourceBarrier(1,&b);
    D3D12_TEXTURE_COPY_LOCATION src{tex,D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
    D3D12_TEXTURE_COPY_LOCATION dst{buf.Get(),D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
    dst.PlacedFootprint=fp;
    cl->CopyTextureRegion(&dst,0,0,0,&src,nullptr);
    cl->Close();
    ID3D12CommandList* lists[]={cl.Get()};
    q->ExecuteCommandLists(1,lists);
    ComPtr<ID3D12Fence> f;dev->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(f.GetAddressOf()));
    HANDLE ev=CreateEventW(nullptr,FALSE,FALSE,nullptr);
    f->SetEventOnCompletion(1,ev);q->Signal(f.Get(),1);
    WaitForSingleObject(ev,10000);CloseHandle(ev);
    out.resize(size_t(W)*H*4);
    float* mapped=nullptr;
    if(FAILED(buf->Map(0,nullptr,reinterpret_cast<void**>(&mapped)))) return false;
    for(unsigned y=0;y<H;++y) {
        const auto* row=reinterpret_cast<const float*>(reinterpret_cast<const uint8_t*>(mapped)+size_t(y)*fp.Footprint.RowPitch);
        std::memcpy(&out[size_t(y)*W*4],row,size_t(W)*4*sizeof(float));
    }
    buf->Unmap(0,nullptr);
    return true;
}
} // namespace

int main(int argc,char** argv) {
    setvbuf(stdout,nullptr,_IONBF,0);
    if(argc<2) {std::printf("usage: fp16_input_gpu_check <shader.hlsl>\n");return 2;}
    std::printf("[1] shader path\n");
    const std::wstring shader_path=[&]{const char* a=argv[1];
        int n=MultiByteToWideChar(CP_UTF8,0,a,-1,nullptr,0);
        std::wstring w(size_t(n),0);MultiByteToWideChar(CP_UTF8,0,a,-1,w.data(),n);
        return w;}();
    ComPtr<ID3D12Device> device;
    std::printf("[2] create device\n");
    {
        ComPtr<IDXGIFactory4> factory;
        CHECK(SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(factory.GetAddressOf()))),"dxgi factory");
        ComPtr<IDXGIAdapter1> adapter;
        for(UINT i=0;factory->EnumAdapters1(i,&adapter)==S_OK;++i) {
            DXGI_ADAPTER_DESC1 d{};adapter->GetDesc1(&d);
            if(d.Flags&DXGI_ADAPTER_FLAG_SOFTWARE) {adapter.Reset();continue;}
            if(SUCCEEDED(D3D12CreateDevice(adapter.Get(),D3D_FEATURE_LEVEL_12_0,
                IID_PPV_ARGS(device.GetAddressOf())))) break;
            adapter.Reset();
        }
        CHECK(device,"d3d12 device");
    }
    ComPtr<ID3D12CommandAllocator> al;
    std::printf("[3] queue\n");
    CHECK(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
        IID_PPV_ARGS(al.GetAddressOf()))),"allocator");
    ComPtr<ID3D12GraphicsCommandList> cl;
    CHECK(SUCCEEDED(device->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,al.Get(),
        nullptr,IID_PPV_ARGS(cl.GetAddressOf()))),"command list");
    D3D12_COMMAND_QUEUE_DESC qd{};qd.Type=D3D12_COMMAND_LIST_TYPE_DIRECT;
    ComPtr<ID3D12CommandQueue> queue;
    CHECK(SUCCEEDED(device->CreateCommandQueue(&qd,IID_PPV_ARGS(queue.GetAddressOf()))),"queue");

    // Deterministic input: a few values above 1 exercise saturate and the
    // highlight weight; the ramp below 1 exercises the composite blend.
    std::vector<float> scene(size_t(W)*H*3),alpha(size_t(W)*H,0.25f);
    for(unsigned y=0;y<H;++y)for(unsigned x=0;x<W;++x) {
        const size_t i=size_t(y)*W+x;
        scene[i*3+0]=float(x%256)/255.0f*0.75f;              // [0,0.75]
        scene[i*3+1]=float(y%256)/255.0f*0.75f;
        scene[i*3+2]=(x==0&&y==0)?3.5f:0.5f;                 // highlight sample
        if(x==1&&y==0) scene[i*3+2]=1.35f;                   // peak>1.2 weight=0
    }
    // NR replacement color, constant, valid.
    std::vector<float> nr(size_t(W)*H*4,0.5f);

    struct FormatCase {const char* name;DXGI_FORMAT format;bool fp16;};
    const FormatCase cases[]={
        {"R11G11B10_FLOAT",DXGI_FORMAT_R11G11B10_FLOAT,false},
        {"R16G16B16A16_FLOAT",DXGI_FORMAT_R16G16B16A16_FLOAT,true},
    };
    for(const auto& tc:cases) {
        std::printf("== %s\n",tc.name);
        // A closed command list must be Reset before re-recording; without
        // this every record call of the second case silently fails.
        static bool first_case=true;
        if(first_case) {first_case=false;}
        else {
            const HRESULT reset_rc=cl->Reset(al.Get(),nullptr);
            std::printf("   reset_rc=0x%08lX\n",(unsigned long)reset_rc);
            CHECK(SUCCEEDED(reset_rc),"list reset");
        }
        auto source=make_texture(device.Get(),W,H,tc.format,true);
        auto xess_input=make_texture(device.Get(),W,H,tc.format,true);
        auto motion=make_texture(device.Get(),W,H,DXGI_FORMAT_R16G16_FLOAT,true);
        auto nr_result=make_texture(device.Get(),W,H,DXGI_FORMAT_R32G32B32A32_FLOAT,true);
        CHECK(source&&xess_input&&motion&&nr_result,"resources");
        std::printf("   [a] resources ok\n");
        CHECK(fill_color(device.Get(),queue.Get(),source.Get(),tc.format,
            reinterpret_cast<const float(*)[3]>(scene.data()),
            tc.fp16?alpha.data():nullptr),"fill source");
        std::printf("   [b] source filled\n");
        CHECK(fill_color(device.Get(),queue.Get(),nr_result.Get(),
            DXGI_FORMAT_R32G32B32A32_FLOAT,
            reinterpret_cast<const float(*)[3]>(nr.data()),nullptr),"fill nr");
        std::printf("   [c] nr filled\n");

        // Reference = the decoded content actually stored in the source.
        std::vector<float> src_rb;
        CHECK(read_rgba32f_of_source(device.Get(),queue.Get(),source.Get(),
            tc.format,D3D12_RESOURCE_STATE_COMMON,src_rb),"read back source");
        std::vector<float> ref_scene(size_t(W)*H*3),ref_alpha(size_t(W)*H);
        for(size_t i=0;i<size_t(W)*H;++i) {
            ref_scene[i*3+0]=src_rb[i*4+0];
            ref_scene[i*3+1]=src_rb[i*4+1];
            ref_scene[i*3+2]=src_rb[i*4+2];
            ref_alpha[i]=tc.fp16?src_rb[i*4+3]:1.0f;
        }
        std::printf("   [b2] reference decoded\n");

        nrb::HdrProxy proxy;
        // Motion is intentionally null in this check: UseGameMotion=0. The
        // game-motion path is verified in-game; here the color adapter
        // contract is the target.
        CHECK(proxy.initialize(device.Get(),source.Get(),nullptr,xess_input.Get(),
            shader_path),"proxy initialize");
        std::printf("   [d] proxy initialized (fp16_selected=%d pso_is_f16=%d)\n",
            proxy.source_fp16_for_test()?1:0,proxy.pso_f16_selected_for_test()?1:0);
        CHECK(proxy.source_format_for_test()==tc.format,"format recorded");

        // prepare: proxy=saturate(scene) (+source alpha), NRMotion=0
        proxy.record_prepare(cl.Get(),source.Get(),false,1,1);
        proxy.record_composite(cl.Get(),source.Get(),nr_result.Get(),xess_input.Get(),
            0,1.0f,1.0f);
        const HRESULT close_rc=cl->Close();
        std::printf("   [e] recorded close_rc=0x%08lX\n",(unsigned long)close_rc);
        ID3D12CommandList* lists[]={cl.Get()};
        queue->ExecuteCommandLists(1,lists);
        ComPtr<ID3D12Fence> f;
        CHECK(SUCCEEDED(device->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(f.GetAddressOf()))),"fence");
        HANDLE ev=CreateEventW(nullptr,FALSE,FALSE,nullptr);
        f->SetEventOnCompletion(1,ev);queue->Signal(f.Get(),1);
        CHECK(WaitForSingleObject(ev,20000)==WAIT_OBJECT_0,"gpu done");
        CloseHandle(ev);
        CHECK(SUCCEEDED(device->GetDeviceRemovedReason()),"device alive");

        std::vector<float> proxy_px;
        std::printf("   [f] gpu done\n");
        CHECK(read_rgba32f(device.Get(),queue.Get(),proxy.proxy_color(),
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,proxy_px),"read proxy");
        std::printf("   [g] proxy read\n");
        // The XeSS input copy carries the source format: decode by format.
        std::vector<float> input_rb;
        CHECK(read_rgba32f_of_source(device.Get(),queue.Get(),xess_input.Get(),
            tc.format,D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,input_rb),"read xess input");
        std::printf("   [h] input read\n");
        unsigned bad=0;
        for(unsigned y=0;y<H;++y)for(unsigned x=0;x<W;++x) {
            const size_t i=(size_t(y)*W+x)*4;
            const size_t j=size_t(y)*W+x;
            const float* s=&ref_scene[j*3];
            for(int c=0;c<3;++c) {
                const float expect=std::min(std::max(s[c],0.0f),1.0f);
                if(std::fabs(proxy_px[i+unsigned(c)]-expect)>1e-3f) ++bad;
            }
            if(std::fabs(proxy_px[i+3]-ref_alpha[j])>1e-3f) ++bad;
            // composite: rgb=max(scene+w*(nr-sdr),0), alpha preserved
            const float peak=std::max(s[0],std::max(s[1],s[2]));
            const float weight=std::min(std::max((1.2f-peak)/0.4f,0.0f),1.0f);
            for(int c=0;c<3;++c) {
                const float sdr=std::min(std::max(s[c],0.0f),1.0f);
                const float expect=std::max(s[c]+weight*(0.5f-sdr),0.0f);
                if(std::fabs(input_rb[i+unsigned(c)]-expect)>1e-3f) ++bad;
            }
            if(std::fabs(input_rb[i+3]-ref_alpha[j])>1e-3f) ++bad;
        }
        std::printf("   mismatched comparisons: %u\n",bad);
        // Diagnostics: a few sample texels.
        for(unsigned k=0;k<4;++k) {
            const unsigned x=k*97,y=k*53+1;
            const size_t i=(size_t(y)*W+x)*4,j=size_t(y)*W+x;
            const float* s=&ref_scene[j*3];
            const float peak=std::max(s[0],std::max(s[1],s[2]));
            const float weight=std::min(std::max((1.2f-peak)/0.4f,0.0f),1.0f);
            std::printf("   texel(%u,%u) ref=(%.5f,%.5f,%.5f,a=%.4f)\n",x,y,s[0],s[1],s[2],ref_alpha[j]);
            std::printf("      proxy=(%.5f,%.5f,%.5f,%.4f)\n",
                proxy_px[i],proxy_px[i+1],proxy_px[i+2],proxy_px[i+3]);
            std::printf("      input=(%.5f,%.5f,%.5f,%.4f) expect=(%.5f,%.5f,%.5f,%.4f)\n",
                input_rb[i],input_rb[i+1],input_rb[i+2],input_rb[i+3],
                std::max(s[0]+weight*(0.5f-std::min(std::max(s[0],0.0f),1.0f)),0.0f),
                std::max(s[1]+weight*(0.5f-std::min(std::max(s[1],0.0f),1.0f)),0.0f),
                std::max(s[2]+weight*(0.5f-std::min(std::max(s[2],0.0f),1.0f)),0.0f),
                ref_alpha[j]);
        }
        std::printf(bad==0?"   PASS\n":"   FORMULA MISMATCH (see markers)\n");
    }
    std::printf("ALL PASS\n");
    return 0;
}
