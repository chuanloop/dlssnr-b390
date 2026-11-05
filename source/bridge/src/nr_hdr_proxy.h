#pragma once
#include <d3d12.h>
#include <wrl/client.h>
#include <string>
#include <cstdint>
#include <memory>
#include <vector>
#include "nr_hdr_resource_pool.h"

namespace nrb {
struct HdrCompositeHeapDiagStats {
    uint64_t pooled_private_composite_count=0;
    uint64_t unique_composite_heap_creates=0;
};
void hdr_set_composite_heap_reuse(bool enabled);
bool hdr_composite_heap_reuse_enabled();
HdrCompositeHeapDiagStats hdr_composite_heap_diag_stats();
struct HdrProxyCacheStats {
    // Lifetime monotonic counters. Take deltas; stats() does not reset them.
    uint64_t bytecode_hits=0,source_reads=0,source_hashes=0;
    uint64_t shader_compile_calls=0,shader_compile_failures=0;
    uint64_t root_serializations=0,root_signature_creates=0,pso_creates=0;
    uint64_t pipeline_hits=0,pipeline_misses=0,uncached_initializations=0;
    uint64_t device_evictions=0,device_loss_failures=0,invalidations=0,failures=0;
    uint64_t shader_generation=0,resident_device_entries=0;
    bool enabled=false,shader_ready=false;
};
struct HdrProxyShaderIdentity {
    uint32_t abi_version=1,root_abi_version=1;
    uint32_t compile_flags=0,compile_flags2=0;
    uint64_t shader_generation=0;
    std::wstring shader_path;
    std::string target,source_sha256,prepare_sha256,composite_sha256;
    std::string root_signature_sha256;
    // FP16-source variants compiled from the same shader file.
    std::string prepare_f16_sha256,composite_f16_sha256;
};
// An explicitly owned, bounded session cache. Creation defaults to OFF.
// Only shader blobs and immutable root/PSO objects belong to this session.
// All methods are serialized internally; HdrProxy itself remains single-owner.
class HdrProxyCache {
public:
    explicit HdrProxyCache(size_t max_devices=2);
    ~HdrProxyCache();
    HdrProxyCache(const HdrProxyCache&)=delete;
    HdrProxyCache& operator=(const HdrProxyCache&)=delete;
    void set_enabled(bool enabled);
    bool enabled() const;
    // CPU only: snapshot/read/hash/compile/serialize once per generation.
    // Same exact path reuses that immutable snapshot, without touching disk.
    // Current shader is self-contained; unmanifested includes fail closed.
    bool warmup_shader(const std::wstring& shader_path);
    // Device creation stays on the caller's thread. Does not toggle ON/OFF.
    bool warmup(ID3D12Device* device,const std::wstring& shader_path);
    // Cold, explicit file-change notification. Invalidates first; failed reload
    // leaves the cache unready. Existing proxies retain their immutable refs.
    bool reload_shader(const std::wstring& shader_path);
    void invalidate_shader();
    void forget_device(ID3D12Device* device);
    HdrProxyCacheStats stats() const;
    HdrProxyShaderIdentity shader_identity() const;
    // Audit/probe only: caller receives private copies, never mutable blobs.
    bool copy_shader_bytecode(std::vector<uint8_t>& prepare,
                              std::vector<uint8_t>& composite) const;
private:
    friend class HdrProxy;
    struct Impl;
    std::unique_ptr<Impl> impl_;
    bool acquire(ID3D12Device* device,const std::wstring& shader_path,
        Microsoft::WRL::ComPtr<ID3D12RootSignature>& root,
        Microsoft::WRL::ComPtr<ID3D12PipelineState>& prepare,
        Microsoft::WRL::ComPtr<ID3D12PipelineState>& composite,
        Microsoft::WRL::ComPtr<ID3D12PipelineState>& prepare_f16,
        Microsoft::WRL::ComPtr<ID3D12PipelineState>& composite_f16);
};
// Candidate GPU-only color adapter. The NR model receives a bounded SDR
// proxy; its delta is blended over the original scene-linear game color.
// The XeSS input remains R11G11B10, including highlights above 1.
class HdrProxy {
public:
    HdrProxy()=default;
    HdrProxy(const HdrProxy&)=delete;
    HdrProxy& operator=(const HdrProxy&)=delete;
    // Five arguments retain uncached behavior. Passing a session makes its
    // independently controlled ON/OFF state apply to this new frame only.
    // Six arguments with nullptr also force uncached OFF.
    bool initialize(ID3D12Device* device,ID3D12Resource* source,
        ID3D12Resource* motion,ID3D12Resource* xess_input,
        const std::wstring& shader_path);
    bool initialize(ID3D12Device* device,ID3D12Resource* source,
        ID3D12Resource* motion,ID3D12Resource* xess_input,
        const std::wstring& shader_path,HdrProxyCache* cache);
    // Seven arguments opt into the separately controlled mutable-resource pool.
    // Pipeline acquisition is still performed for every frame, including hits.
    bool initialize(ID3D12Device* device,ID3D12Resource* source,
        ID3D12Resource* motion,ID3D12Resource* xess_input,
        const std::wstring& shader_path,HdrProxyCache* cache,HdrProxyResourcePool* pool);
    HdrLeaseProof* resource_proof() {return resource_lease_?&resource_lease_->proof():nullptr;}
    bool finish_resource_lease() {return !resource_lease_ || resource_lease_->finish();}
    void quarantine_resource_lease() {if(resource_lease_) resource_lease_->quarantine();}
    void record_prepare(ID3D12GraphicsCommandList* list,
        ID3D12Resource* source,bool use_game_motion,float scale_x,float scale_y,
        D3D12_RESOURCE_STATES source_state=D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    bool record_composite(ID3D12GraphicsCommandList* list,
        ID3D12Resource* source,ID3D12Resource* nr_result,
        ID3D12Resource* xess_input,int style,float style_tone,float style_gain,
        D3D12_RESOURCE_STATES source_state=D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATES xess_input_state=D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        bool private_descriptors=false);
    ID3D12Resource* proxy_color() const {return proxy_.Get();}
    ID3D12Resource* nr_motion() const {return nr_motion_.Get();}
    bool game_motion_available() const {return game_motion_available_;}
    void set_timestamp_query_target(ID3D12QueryHeap* heap,ID3D12Resource* readback) {ts_heap_=heap;ts_readback_=readback;}
    uint32_t diag_pool_alloc_kind() const {return diag_pool_alloc_kind_;}
    uint32_t diag_composite_heap_kind() const {return diag_composite_heap_kind_;}
    // Developer verification accessors (gpu check harness only).
    DXGI_FORMAT source_format_for_test() const {return source_format_;}
    bool source_fp16_for_test() const {return source_fp16_;}
    bool pso_f16_selected_for_test() const {
        return prepare_.Get()==prepare_f16_.Get() && composite_.Get()==composite_f16_.Get();}
private:
    Microsoft::WRL::ComPtr<ID3D12Device> device_;
    Microsoft::WRL::ComPtr<ID3D12Resource> proxy_,nr_motion_;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> heap_;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> composite_heap_;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> root_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> prepare_,composite_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> prepare_f16_,composite_f16_;
    UINT stride_=0,width_=0,height_=0;
    bool game_motion_available_=false;
    // The source (and the same-format XeSS input copy) may be R11G11B10_FLOAT
    // or R16G16B16A16_FLOAT. Typed views and PSO selection follow this format.
    DXGI_FORMAT source_format_=DXGI_FORMAT_R11G11B10_FLOAT;
    bool source_fp16_=false;
    // Style peripheral state, recomputed per composite.
    float style_curve_[16]={};
    float style_sat_=1.0f, style_tone_=1.0f, style_exposure_=1.0f, style_contrast_=0.0f;
    float style_gain_=1.0f;
    UINT style_mode_=0;
    D3D12_RESOURCE_STATES resource_initial_=D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    std::unique_ptr<HdrProxyResourcePool::Lease> resource_lease_;
    ID3D12QueryHeap* ts_heap_=nullptr;
    ID3D12Resource* ts_readback_=nullptr;
    uint32_t diag_pool_alloc_kind_=0;
    uint32_t diag_composite_heap_kind_=0;
    D3D12_CPU_DESCRIPTOR_HANDLE cpu(UINT index) const;
    D3D12_GPU_DESCRIPTOR_HANDLE gpu(UINT index) const;
    void bind(ID3D12GraphicsCommandList* list,ID3D12PipelineState* pipeline,
              bool use_game_motion=false,float scale_x=1,float scale_y=1,
              ID3D12DescriptorHeap* private_heap=nullptr);
};
}
