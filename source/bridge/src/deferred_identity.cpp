#include "deferred_identity.h"
#include <cstdio>
#include <unordered_map>
#include <wrl/client.h>
#include <atomic>
#include <mutex>
#include <vector>
#include <string>
#include <array>
#include <algorithm>
#include <memory>
#include <chrono>
#include <cmath>
#include "periodic_flash_native_diag.h"
#include "nr_record_timing.h"
#include "nr_hdr_resource_pool_api.h"
#include "nr_hdr_resource_pool.h"
#include "nr_gpu_handoff.h"
#include "nr_gpu_handoff_reuse.h"
#ifdef NRB_LIVE_NR_TEST
#include "nr_hdr_proxy.h"
#endif

namespace nrb {
using DeferredLog = void (*)(const char*);
static DeferredLog g_deferred_log = nullptr;
void deferred_identity_set_log(DeferredLog log) { g_deferred_log = log; }
namespace {
using Microsoft::WRL::ComPtr;

// Bounded per-frame command resources. The old path created three allocators
// and three command lists every frame; at 20-30 fps that alone dominated the
// 2.2 ms record_resources bucket. A Pending holds one lease until its consumer
// fence completes, then the slot returns to this pool. Never reused in flight.
struct FrameCommands {
    ComPtr<ID3D12Device> device;   // owner of the allocators below
    ComPtr<ID3D12CommandAllocator> prep_allocator, deferred_allocator, composite_allocator;
    ComPtr<ID3D12GraphicsCommandList> prep_list, deferred_list, composite_list;
    ComPtr<ID3D12Resource> color_copy;
    ComPtr<ID3D12QueryHeap> ts_heap;
    ComPtr<ID3D12Resource> ts_readback;
    D3D12_RESOURCE_DESC color_desc{};
    D3D12_RESOURCE_STATES color_state = D3D12_RESOURCE_STATE_COPY_DEST;
    bool color_state_known = true;
    bool in_use = false;
    bool fresh = false;   // newly created objects: already recording, skip Reset
};
std::mutex frame_commands_mutex;
std::vector<std::unique_ptr<FrameCommands>> frame_commands_pool;

// [probe] Live bisect switch for the prep list: records only the first N
// commands (0..4). Read from nr_prep_level.txt next to the bridge once a
// second, so a launch is never needed to move one step.
// The published path always records the complete preparation list.
int prep_level() { return 4; }

FrameCommands* acquire_frame_commands(ID3D12Device* device,
                                      const D3D12_RESOURCE_DESC& copy_desc) {
    if (!device) return nullptr;
    std::lock_guard lock(frame_commands_mutex);
    FrameCommands* fc = nullptr;
    for (auto& candidate : frame_commands_pool)
        if (!candidate->in_use) { fc = candidate.get(); break; }
    if (!fc) {
        if (frame_commands_pool.size() >= 8) return nullptr;
        auto created = std::make_unique<FrameCommands>();
        fc = created.get();
        frame_commands_pool.push_back(std::move(created));
    }
    fc->in_use = true;
    // Allocators and lists belong to the device that created them. An engine
    // that recreates its D3D12 device (UE5 probes, then switches) would leave
    // stale objects whose Reset() always fails, so the slot is rebuilt.
    if (tail_generic_profile()) {
        // Intel/Wukong: Reset() on a recycled (collector, list) pair returns
        // E_INVALIDARG even though both belong to this device. Fresh objects
        // are already recording, so the record path skips Reset for them.
        fc->prep_list.Reset(); fc->prep_allocator.Reset();
        fc->deferred_list.Reset(); fc->deferred_allocator.Reset();
        fc->composite_list.Reset(); fc->composite_allocator.Reset();
        fc->fresh = true;
    }
    if (fc->device.Get() != device) {
        fc->prep_list.Reset(); fc->prep_allocator.Reset();
        fc->deferred_list.Reset(); fc->deferred_allocator.Reset();
        fc->composite_list.Reset(); fc->composite_allocator.Reset();
        fc->color_copy.Reset();
        fc->ts_heap.Reset();
        fc->ts_readback.Reset();
        fc->device = device;
    }
    const auto ensure = [&](ComPtr<ID3D12CommandAllocator>& allocator,
                            ComPtr<ID3D12GraphicsCommandList>& list) {
        if (!allocator && FAILED(device->CreateCommandAllocator(
                D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(allocator.GetAddressOf()))))
            return false;
        if (!list && FAILED(device->CreateCommandList(0,
                D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr,
                IID_PPV_ARGS(list.GetAddressOf()))))
            return false;
        return true;
    };
    if (!ensure(fc->prep_allocator, fc->prep_list) ||
        !ensure(fc->deferred_allocator, fc->deferred_list) ||
        !ensure(fc->composite_allocator, fc->composite_list)) {
        fc->in_use = false;
        return nullptr;
    }
    // A freshly created command list is open; a list from a failed frame may
    // still be open too. Close it so the record path can always Reset it.
    // Closing an already-closed list returns a failure but has no state effect.
    // Freshly created lists stay open: they are already recording
    // and the record path skips Reset for them.
    if (!fc->fresh) {
        (void)fc->prep_list->Close();
        (void)fc->deferred_list->Close();
        (void)fc->composite_list->Close();
    }
    const bool same_copy = fc->color_copy &&
        fc->color_desc.Format == copy_desc.Format &&
        fc->color_desc.Width == copy_desc.Width &&
        fc->color_desc.Height == copy_desc.Height &&
        fc->color_desc.DepthOrArraySize == copy_desc.DepthOrArraySize &&
        fc->color_desc.MipLevels == copy_desc.MipLevels &&
        fc->color_desc.SampleDesc.Count == copy_desc.SampleDesc.Count;
    if (!same_copy || !fc->color_state_known) {
        fc->color_copy.Reset();
        D3D12_HEAP_PROPERTIES heap{};
        heap.Type = D3D12_HEAP_TYPE_DEFAULT;
        if (FAILED(device->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,
                &copy_desc,D3D12_RESOURCE_STATE_COPY_DEST,nullptr,
                IID_PPV_ARGS(fc->color_copy.GetAddressOf())))) {
            fc->in_use = false;
            return nullptr;
        }
        fc->color_desc = copy_desc;
        fc->color_state = D3D12_RESOURCE_STATE_COPY_DEST;
        fc->color_state_known = true;
    }
    if (!fc->ts_heap || !fc->ts_readback) {
        fc->ts_heap.Reset();
        fc->ts_readback.Reset();
        D3D12_QUERY_HEAP_DESC qhd{};
        qhd.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
        qhd.Count = 8;
        qhd.NodeMask = 0;
        if (SUCCEEDED(device->CreateQueryHeap(&qhd, IID_PPV_ARGS(fc->ts_heap.GetAddressOf())))) {
            D3D12_HEAP_PROPERTIES hp_rb{};
            hp_rb.Type = D3D12_HEAP_TYPE_READBACK;
            D3D12_RESOURCE_DESC rd{};
            rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
            rd.Width = 8 * sizeof(uint64_t);
            rd.Height = 1;
            rd.DepthOrArraySize = 1;
            rd.MipLevels = 1;
            rd.SampleDesc.Count = 1;
            rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            if (FAILED(device->CreateCommittedResource(&hp_rb, D3D12_HEAP_FLAG_NONE,
                    &rd, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                    IID_PPV_ARGS(fc->ts_readback.GetAddressOf())))) {
                fc->ts_heap.Reset();
            }
        }
    }
    return fc;
}
void release_frame_commands(FrameCommands* fc) {
    if (!fc) return;
    std::lock_guard lock(frame_commands_mutex);
    if (!fc->color_state_known) {
        fc->color_copy.Reset();
        fc->color_state_known = true;
        fc->color_state = D3D12_RESOURCE_STATE_COPY_DEST;
    }
    fc->in_use = false;
}

struct Pending {
    ComPtr<ID3D12GraphicsCommandList> game_list;
    uint64_t game_list_generation=0;
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandAllocator> prep_allocator;
    ComPtr<ID3D12GraphicsCommandList> prep_list;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> deferred_list;
    ComPtr<ID3D12Resource> color_copy;
    ComPtr<ID3D12QueryHeap> ts_heap;
    ComPtr<ID3D12Resource> ts_readback;
    ComPtr<ID3D12Resource> original_color,output,motion,depth,exposure;
    ComPtr<ID3D12Fence> completion;
    FrameCommands* frame_commands=nullptr;
    NRB_Controls frame_controls{};
    NRB_Route route=NRB_ROUTE_UNKNOWN;
    NRB_RouteEvidence evidence=NRB_EVIDENCE_NONE;
    float motion_scale_x=1,motion_scale_y=1;
    bool game_motion=false,game_reset=false;
    flashdiag::Context diagnostic{};
    flashdiag::Reason diagnostic_reason=flashdiag::Reason::none;
#ifdef NRB_LIVE_NR_TEST
    std::unique_ptr<HdrProxy> hdr;
    ComPtr<ID3D12Fence> prepared;
    ComPtr<ID3D12CommandAllocator> composite_allocator;
    ComPtr<ID3D12GraphicsCommandList> composite_list;
    ComPtr<ID3D12Resource> nr_output;
    ComPtr<ID3D12Fence> nr_ready;
    ComPtr<ID3D12Resource> motion_probe_readback;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT motion_probe_layout{};
    unsigned motion_probe_index=0;
    NRB_Processor nr_processor{};
    bool nr_submitted=false;
    bool nr_processor_called=false, nr_helper_retired=false, nr_queue_exposed=false;
    bool nr_prepared_bypassed=false;
    ComPtr<ID3D12Fence> nr_reuse_completion;
    bool nr_timed=false;
    bool record_timed=false;
    uint64_t record_timing_epoch=0;
    double record_resources_ms=0, record_hdr_ms=0, record_xess_ms=0;
    std::chrono::steady_clock::time_point record_started{},record_finished{};
    uint64_t nr_timing_epoch=0;
    std::chrono::steady_clock::time_point nr_started{};
    std::chrono::steady_clock::time_point nr_prepared{};
    std::chrono::steady_clock::time_point nr_modeled{};
    std::chrono::steady_clock::time_point nr_pre_sr{};
    std::chrono::steady_clock::time_point submit_entry_tp{};
    std::chrono::steady_clock::time_point submit_return_tp{};
    uint64_t pending_seq=0;
    uint32_t diag_record_mode=0;
    uint32_t diag_window_id=0;
    uint32_t diag_arm_reuse=0;
    uint32_t diag_late_latch=1;
    uint64_t retire_prev_start_qpc=0;
    uint64_t retire_prev_end_qpc=0;
    uint32_t had_prev_proc=0;
    uint32_t had_net_fence=0;
    uint64_t nr_started_qpc=0;
    uint64_t nr_prepared_qpc=0;
    uint64_t vk_wait_start_qpc=0;
    uint64_t vk_wait_end_qpc=0;
    uint64_t nr_modeled_qpc=0;
    uint64_t nr_pre_sr_qpc=0;
    uint64_t deferred_exec_end_qpc=0;
    uint64_t tail_submit_end_qpc=0;
    uint64_t submit_return_qpc=0;
    uint64_t tail_wait_start_qpc=0;
    uint64_t tail_wait_end_qpc=0;
    uint64_t ended_qpc=0;
#endif
    bool submitted=false;
    bool retired_done=false;
    void release_retired_resources() {
#ifdef NRB_LIVE_NR_TEST
        if(hdr && hdr->resource_proof()) {
            // Remove every private record before a pristine/terminal lease can
            // become free. No private list is exported outside this Pending.
            const auto& proof=*hdr->resource_proof();
            if(proof.pristine_unsubmitted() || proof.terminal_safe()) {
                prep_list.Reset();deferred_list.Reset();composite_list.Reset();
                hdr->finish_resource_lease();
            } else hdr->quarantine_resource_lease();
            hdr.reset();
        }
#endif
        if(frame_commands) {
            release_frame_commands(frame_commands);
            frame_commands=nullptr;
        }
    }
    ~Pending() {
        release_retired_resources();
    }
};
std::recursive_mutex pending_mutex;
std::vector<std::unique_ptr<Pending>> pending;
// Retained (not freed) when the owning component is replaced: the
// textures and borrowed original may still be referenced, but no
// component can ever signal this record's fences again.
std::vector<std::unique_ptr<Pending>> abandoned;
std::atomic<bool> swap_suspended{false};
#ifdef NRB_LIVE_NR_TEST
struct PendingExitQuarantine {
    ~PendingExitQuarantine() {
        // Declared after pending, so this runs before its vector destructor.
        // Retain the complete Pending, not just the pool's textures, when DLL
        // teardown has no proof for GPU/XPU commands or borrowed originals.
        for(auto& item:pending) if(item) {
            if(item->hdr && item->hdr->resource_proof()) {
                const auto& proof=*item->hdr->resource_proof();
                if(!proof.pristine_unsubmitted() && !proof.terminal_safe()) (void)item.release();
            } else if(item->nr_queue_exposed && (!item->submitted ||
                      item->completion->GetCompletedValue()<1 ||
                      item->completion->GetCompletedValue()==UINT64_MAX ||
                      (item->nr_processor_called && !item->nr_helper_retired))) (void)item.release();
        }
    }
} pending_exit_quarantine;
#endif
std::atomic<unsigned> attempts{0}, recorded{0}, retired{0};
std::atomic<unsigned> tail_samples[4]{},tail_valid_samples[4]{};
std::atomic<unsigned> signal_failures{0},generation_mismatches{0},invalidated{0};
#ifdef NRB_LIVE_NR_TEST
HdrProxyCache& hdr_cache_session() {
    static auto cache=[] {
        auto result=std::make_unique<HdrProxyCache>(2);
        result->set_enabled(true);return result;
    }();
    return *cache;
}
HdrProxyResourcePool& hdr_resource_pool_session() {
    // Process lifetime. Enabled by default now that the deferred path owns the
    // HdrProxy lease for the whole Pending; the pool reuses the two textures
    // and the five-slot descriptor heap instead of rebuilding them every frame.
    // Busy/incompatible acquisition still falls back to a unique allocation.
    static auto* pool=[] {
        auto* result=new HdrProxyResourcePool;
        result->set_enabled(true);
        return result;
    }();
    return *pool;
}
std::atomic<uint64_t> nr_completed{0};
std::atomic<double> nr_last_ms{0};
std::atomic<double> nr_average_ms{0};
std::atomic<unsigned> nr_sample_count{0};
std::atomic<double> prep_last_ms{0}, prep_average_ms{0};
std::atomic<double> host_last_ms{0}, host_average_ms{0};
std::atomic<double> composite_submit_last_ms{0}, composite_submit_average_ms{0};
std::atomic<double> tail_last_ms{0}, tail_average_ms{0};
// The five channels share one 32-frame window, so their averages add up to
// the average total from the same completed frames.
std::array<std::array<double,32>,5> nr_timing_samples{};
std::array<double,5> nr_timing_sums{};
unsigned nr_timing_count=0;
std::mutex nr_timing_mutex;
RecordTimingWindow record_timing_window;
std::atomic<bool> nr_timing_enabled{true};
std::atomic<uint64_t> nr_timing_epoch{0};
std::atomic<bool> nr_failed{false};
std::atomic<uint64_t> frame_serial{0};
std::atomic<unsigned> hdr_ready{0};
std::atomic<unsigned> motion_probe_claimed{0};

struct NrvkDiagFrameRecordMirror {
    uint64_t frame_id;
    uint64_t returned_seq;
    uint32_t window_id;
    uint32_t arm_reuse;
    uint32_t slot_index;
    uint32_t copy_slot;
    uint32_t had_network_fence_pending;
    uint32_t d3d_valid;
    uint32_t vk_valid;
    uint32_t vk_valid_bits;
    uint64_t wait_fences_start_qpc;
    uint64_t wait_fences_end_qpc;
    uint64_t d3d_freq_hz;
    double   vk_timestamp_period_ns;
    uint64_t d4_tick;
    uint64_t d5_tick;
    uint64_t v_ticks[7];
};

struct BridgeDiagFrameRecord {
    uint64_t frame_id = 0;
    uint64_t pending_seq = 0;
    uint64_t completed_index = 0;
    uint32_t window_id = 0;
    uint32_t diag_mode = 0;
    uint32_t arm_reuse = 0;
    uint32_t late_latch = 1;
    uint32_t prepared_bypassed = 0;
    uint32_t private_descriptors = 0;
    uint32_t pool_alloc_kind = 0;
    uint32_t composite_heap_kind = 0;
    uint32_t had_previous_processor = 0;
    uint32_t had_network_fence_pending = 0;
    uint32_t d3d_valid = 0;
    uint64_t qpc_freq_hz = 0;
    uint64_t d3d_freq_hz = 0;
    double prep_ms = 0.0;
    double host_ms = 0.0;
    double composite_submit_ms = 0.0;
    double tail_ms = 0.0;
    double total_ms = 0.0;
    double record_total_ms = 0.0;
    double record_resources_ms = 0.0;
    double record_hdr_ms = 0.0;
    double record_xess_ms = 0.0;
    double submit_gap_ms = 0.0;
    double end_to_end_ms = 0.0;
    double retire_prev_cpu_ms = 0.0;
    double prep_wait_cpu_ms = 0.0;
    double vk_wait_fences_cpu_ms = 0.0;
    double vk_record_submit_cpu_ms = 0.0;
    double composite_submit_cpu_ms = 0.0;
    double tail_sr_exec_cpu_ms = 0.0;
    double tail_post_sr_submit_cpu_ms = 0.0;
    double tail_submit_cpu_ms = 0.0;
    double tail_wait_cpu_ms = 0.0;
    double tail_retire_cpu_ms = 0.0;
    uint64_t d_ticks[8] {};
};

std::atomic<bool> g_diag_active{false};
std::atomic<uint32_t> g_diag_mode{0};
std::atomic<uint32_t> g_diag_window_id{0};
std::atomic<uint32_t> g_diag_arm_reuse{1};
std::atomic<uint32_t> g_diag_late_latch{1};
std::atomic<uint64_t> g_diag_pending_seq{0};
uint32_t g_last_flush_seq = 0;
uint32_t g_last_reset_seq = 0;
uint64_t g_qpc_freq_hz = 0;
uint64_t g_d3d_freq_hz = 0;
uint64_t g_bridge_d3d_unavail = 0;
uint64_t g_bridge_diag_overflow = 0;
std::wstring g_diag_out_dir;

std::vector<BridgeDiagFrameRecord>& bridge_diag_records() {
    static auto* vec = [] {
        auto* v = new std::vector<BridgeDiagFrameRecord>();
        v->reserve(32768);
        return v;
    }();
    return *vec;
}

uint64_t qpc_now() {
    LARGE_INTEGER li{};
    QueryPerformanceCounter(&li);
    return (uint64_t)li.QuadPart;
}

uint64_t ensure_qpc_freq() {
    if (!g_qpc_freq_hz) {
        LARGE_INTEGER lf{};
        if (QueryPerformanceFrequency(&lf) && lf.QuadPart > 0)
            g_qpc_freq_hz = (uint64_t)lf.QuadPart;
    }
    return g_qpc_freq_hz;
}

double qpc_diff_ms(uint64_t t0, uint64_t t1, uint64_t freq) {
    if (!freq || !t0 || t1 < t0) return 0.0;
    return (double)(t1 - t0) * 1000.0 / (double)freq;
}

void nrvk_set_diag_state(uint32_t mode, uint32_t window_id, uint32_t arm_reuse) {
    HMODULE mod = GetModuleHandleW(L"nrvk_bridge.dll");
    if (!mod) return;
    using Fn = void (*)(uint32_t, uint32_t, uint32_t);
    static Fn fn = nullptr;
    static HMODULE cached_mod = nullptr;
    if (cached_mod != mod || !fn) {
        cached_mod = mod;
        fn = reinterpret_cast<Fn>(GetProcAddress(mod, "NRVK_DiagSetState"));
    }
    if (fn) fn(mode, window_id, arm_reuse);
}

void nrvk_get_last_cpu_substage(uint32_t* had_fence, uint64_t* wait_start_qpc, uint64_t* wait_end_qpc) {
    if (had_fence) *had_fence = 0;
    if (wait_start_qpc) *wait_start_qpc = 0;
    if (wait_end_qpc) *wait_end_qpc = 0;
    HMODULE mod = GetModuleHandleW(L"nrvk_bridge.dll");
    if (!mod) return;
    using Fn = void (*)(uint64_t*, uint32_t*, uint64_t*, uint64_t*);
    static Fn fn = nullptr;
    static HMODULE cached_mod = nullptr;
    if (cached_mod != mod || !fn) {
        cached_mod = mod;
        fn = reinterpret_cast<Fn>(GetProcAddress(mod, "NRVK_DiagGetLastCpuSubstage"));
    }
    uint64_t fid = 0;
    if (fn) fn(&fid, had_fence, wait_start_qpc, wait_end_qpc);
}

void nrvk_harvest_retired() {
    HMODULE mod = GetModuleHandleW(L"nrvk_bridge.dll");
    if (!mod) return;
    using Fn = void (*)(void);
    static Fn fn = nullptr;
    static HMODULE cached_mod = nullptr;
    if (cached_mod != mod || !fn) {
        cached_mod = mod;
        fn = reinterpret_cast<Fn>(GetProcAddress(mod, "NRVK_DiagHarvestRetired"));
    }
    if (fn) fn();
}

void nrvk_get_records(const NrvkDiagFrameRecordMirror** out_data, uint32_t* out_count,
                      uint64_t* out_d3d_unavail, uint64_t* out_vk_unavail, uint64_t* out_overflow) {
    if (out_data) *out_data = nullptr;
    if (out_count) *out_count = 0;
    if (out_d3d_unavail) *out_d3d_unavail = 0;
    if (out_vk_unavail) *out_vk_unavail = 0;
    if (out_overflow) *out_overflow = 0;
    HMODULE mod = GetModuleHandleW(L"nrvk_bridge.dll");
    if (!mod) return;
    using Fn = void (*)(const void**, uint32_t*, uint64_t*, uint64_t*, uint64_t*);
    static Fn fn = nullptr;
    static HMODULE cached_mod = nullptr;
    if (cached_mod != mod || !fn) {
        cached_mod = mod;
        fn = reinterpret_cast<Fn>(GetProcAddress(mod, "NRVK_DiagGetRecords"));
    }
    const void* raw = nullptr;
    if (fn) fn(&raw, out_count, out_d3d_unavail, out_vk_unavail, out_overflow);
    if (out_data) *out_data = static_cast<const NrvkDiagFrameRecordMirror*>(raw);
}

void nrvk_get_device_info(uint64_t* out_d3d_freq, double* out_vk_period, uint32_t* out_vk_bits) {
    if (out_d3d_freq) *out_d3d_freq = 0;
    if (out_vk_period) *out_vk_period = 0.0;
    if (out_vk_bits) *out_vk_bits = 0;
    HMODULE mod = GetModuleHandleW(L"nrvk_bridge.dll");
    if (!mod) return;
    using Fn = void (*)(uint64_t*, double*, uint32_t*);
    static Fn fn = nullptr;
    static HMODULE cached_mod = nullptr;
    if (cached_mod != mod || !fn) {
        cached_mod = mod;
        fn = reinterpret_cast<Fn>(GetProcAddress(mod, "NRVK_DiagGetDeviceInfo"));
    }
    if (fn) fn(out_d3d_freq, out_vk_period, out_vk_bits);
}
#endif
std::once_flag trigger_path_once;
std::wstring trigger_path;

bool user_armed_diagnostic() {
    try {std::call_once(trigger_path_once,[] {
        HMODULE module=nullptr;
        if(!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
            GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCWSTR>(&record_deferred_identity),&module)) return;
        wchar_t path[MAX_PATH]{};
        if(!GetModuleFileNameW(module,path,MAX_PATH)) return;
        trigger_path=path;
        const auto slash=trigger_path.find_last_of(L"\\/");
        if(slash==std::wstring::npos) {trigger_path.clear();return;}
        trigger_path.resize(slash+1);
        trigger_path+=L"NRB-deferred-identity.enable";
    });} catch(...) {return false;}
    if(trigger_path.empty()) return false;
    const DWORD attr=GetFileAttributesW(trigger_path.c_str());
    return attr!=INVALID_FILE_ATTRIBUTES &&
           !(attr&FILE_ATTRIBUTE_DIRECTORY);
}

D3D12_RESOURCE_BARRIER transition(ID3D12Resource* resource,
    D3D12_RESOURCE_STATES before,D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER b{};
    b.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource=resource;
    b.Transition.Subresource=D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore=before;
    b.Transition.StateAfter=after;
    return b;
}
#ifdef NRB_LIVE_NR_TEST
bool pool_terminal_proof(Pending& item,uint64_t done) {
    auto* proof=item.hdr?item.hdr->resource_proof():nullptr;
    if(!proof) return false;
    if(proof->poisoned() || FAILED(item.device->GetDeviceRemovedReason())) {
        proof->poison();item.hdr->quarantine_resource_lease();return false;
    }
    // Exact Pending completion, signaled after all inserted lists and the
    // remaining original batch. No prepare fence or unrelated fence suffices.
    proof->consumer_completed(done,true);
    if(proof->processor_called() && !proof->processor_retired() && !proof->poisoned()) {
        bool retired_ok=false;
        try {
            retired_ok=proof->processor_safe() && (item.nr_helper_retired ||
                (item.nr_processor.retire &&
                 item.nr_processor.retire(item.nr_processor.context,item.completion.Get(),1)==0));
        } catch(...) {}
        proof->processor_retirement(retired_ok);
        if(retired_ok) {item.nr_submitted=false;item.nr_helper_retired=true;}
    }
    if(!proof->terminal_safe()) {item.hdr->quarantine_resource_lease();return false;}
    return true;
}
// The helper owns one output. A second callback in the SAME intercepted batch
// must observe the preceding output's actual consumer completion and retire it.
// This extra fence follows that output's composite + deferred SR. It NEVER
// replaces Pending::completion (the final original-batch lifetime/pool proof).
bool retire_previous_for_reuse(Pending& previous,ID3D12CommandQueue* queue) {
    if(previous.nr_helper_retired) return true;
    bool ok=false;
    try {
        if(previous.nr_submitted && previous.nr_processor.retire &&
           SUCCEEDED(previous.device->GetDeviceRemovedReason()) &&
           (previous.nr_reuse_completion || SUCCEEDED(previous.device->CreateFence(0,
                D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(previous.nr_reuse_completion.GetAddressOf()))))) {
            const auto* exact=previous.nr_reuse_completion.Get();
            ok=gpuhandoff::retire_previous_consumer(exact,1,
                [&] {return SUCCEEDED(queue->Signal(previous.nr_reuse_completion.Get(),1));},
                [&] {
                    HANDLE event=CreateEventW(nullptr,FALSE,FALSE,nullptr);
                    const auto registered=event?previous.nr_reuse_completion->SetEventOnCompletion(1,event):E_FAIL;
                    const auto waited=event && SUCCEEDED(registered)?WaitForSingleObject(event,10000):WAIT_FAILED;
                    const auto done=previous.nr_reuse_completion->GetCompletedValue();
                    if(event) CloseHandle(event);
                    return gpuhandoff::ReuseObservation{exact,done,SUCCEEDED(registered) && waited==WAIT_OBJECT_0,
                        SUCCEEDED(previous.device->GetDeviceRemovedReason())};
                },
                [&] {return previous.nr_processor.retire(previous.nr_processor.context,
                    previous.nr_reuse_completion.Get(),1)==0;});
        }
    } catch(...) {ok=false;}
    gpuhandoff::previous_consumer_wait(ok);
    if(ok) previous.nr_helper_retired=true;
    else {nr_failed.store(true);if(previous.hdr) previous.hdr->quarantine_resource_lease();}
    return ok;
}
#endif
bool retire_submitted_pending(Pending& item, ID3D12CommandQueue* queue, bool wait_if_incomplete, void (*log)(const char*)) {
    if (item.retired_done) return true;
    if (!item.submitted) return false;
#ifdef NRB_LIVE_NR_TEST
    if (!g_d3d_freq_hz && queue) queue->GetTimestampFrequency(&g_d3d_freq_hz);
    if (!item.nr_submitted) {
        const uint64_t done = item.completion ? item.completion->GetCompletedValue() : 0;
        if (done >= 1 && done != UINT64_MAX) {
            if (item.hdr && item.hdr->resource_proof()) {
                if (!pool_terminal_proof(item, done)) return false;
            } else if (item.nr_processor_called && !item.nr_helper_retired) {
                bool ok = false;
                try {
                    ok = item.nr_processor.retire &&
                         SUCCEEDED(item.device->GetDeviceRemovedReason()) &&
                         item.nr_processor.retire(item.nr_processor.context, item.completion.Get(), 1) == 0;
                } catch (...) {}
                if (!ok) { nr_failed.store(true); gpuhandoff::process_failed(); return false; }
                item.nr_helper_retired = true;
            }
            item.release_retired_resources();
            item.retired_done = true;
            flashdiag::recorder.finish(item.diagnostic, item.diagnostic_reason);
            return true;
        }
        return false;
    }

    if (!wait_if_incomplete) {
        const uint64_t pre_done = item.completion ? item.completion->GetCompletedValue() : 0;
        if (pre_done < 1 || pre_done == UINT64_MAX) return false;
    }

    const uint32_t tail_diag_mode = g_diag_mode.load(std::memory_order_relaxed);
    const auto wait_started_tp = item.nr_timed ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    if (tail_diag_mode >= 1) item.tail_wait_start_qpc = qpc_now();
    uint64_t completed_value = item.completion ? item.completion->GetCompletedValue() : 0;
    HANDLE event = nullptr;
    HRESULT registered = S_OK;
    DWORD waited = WAIT_OBJECT_0;
    if (completed_value < 1) {
        event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        registered = event ? item.completion->SetEventOnCompletion(1, event) : E_FAIL;
        waited = (event && SUCCEEDED(registered)) ? WaitForSingleObject(event, 10000) : WAIT_FAILED;
        completed_value = item.completion->GetCompletedValue();
        if (event) CloseHandle(event);
    }
    const bool done = (completed_value >= 1 && completed_value != UINT64_MAX) &&
        (!event || (SUCCEEDED(registered) && waited == WAIT_OBJECT_0)) &&
        SUCCEEDED(item.device->GetDeviceRemovedReason());
    if (tail_diag_mode >= 1) item.tail_wait_end_qpc = qpc_now();
    bool retired_ok = false;
    if (item.hdr && item.hdr->resource_proof()) {
        retired_ok = done && pool_terminal_proof(item, completed_value);
        if (!retired_ok) item.hdr->quarantine_resource_lease();
    } else {
        try {
            retired_ok = done && (item.nr_helper_retired ||
                item.nr_processor.retire(item.nr_processor.context, item.completion.Get(), 1) == 0);
        } catch (...) { retired_ok = false; }
    }
    if (retired_ok) {
        item.nr_submitted = false;
        item.nr_helper_retired = true;
        item.diagnostic.stage_bits |= flashdiag::bit(flashdiag::Stage::retired);
        flashdiag::recorder.count(flashdiag::Counter::nr_retired);
        const auto ended = item.nr_timed ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        if (tail_diag_mode >= 1) item.ended_qpc = qpc_now();
        const uint64_t completed = nr_completed.fetch_add(1, std::memory_order_relaxed) + 1;
        if (item.nr_timed && nr_timing_enabled.load(std::memory_order_acquire) &&
            item.nr_timing_epoch == nr_timing_epoch.load(std::memory_order_acquire)) {
            std::lock_guard timing_guard(nr_timing_mutex);
            if (item.nr_timed && nr_timing_enabled.load(std::memory_order_relaxed) &&
                item.nr_timing_epoch == nr_timing_epoch.load(std::memory_order_relaxed)) {
                const auto millis = [](auto begin, auto end) {
                    return std::chrono::duration<double, std::milli>(end - begin).count();
                };
                const double prep = millis(item.nr_started, item.nr_prepared);
                const double host = millis(item.nr_prepared, item.nr_modeled);
                const double composite = millis(item.nr_modeled, item.nr_pre_sr);
                const bool has_split_tail = (item.submit_return_qpc != 0);
                const double tail = has_split_tail
                    ? std::max(0.0, millis(item.nr_pre_sr, item.submit_return_tp)) +
                      std::max(0.0, millis(wait_started_tp, ended))
                    : millis(item.nr_pre_sr, ended);
                const double ms = has_split_tail
                    ? (prep + host + composite + tail)
                    : millis(item.nr_started, ended);
                const auto gap_end_tp = (item.submit_entry_tp != std::chrono::steady_clock::time_point{})
                    ? item.submit_entry_tp
                    : item.nr_started;
                const double rec_tot_ms = millis(item.record_started, item.record_finished);
                const double sub_gap_ms = std::max(0.0, millis(item.record_finished, gap_end_tp));
                const double e2e_ms = has_split_tail
                    ? (rec_tot_ms + sub_gap_ms + ms)
                    : millis(item.record_started, ended);
                if (item.record_timed && item.record_timing_epoch == item.nr_timing_epoch) {
                    record_timing_window.append(item.diagnostic.nr_frame_id, {
                        rec_tot_ms,
                        item.record_resources_ms, item.record_hdr_ms, item.record_xess_ms,
                        sub_gap_ms,
                        e2e_ms});
                }
                const unsigned count = ++nr_timing_count;
                const unsigned samples = std::min(32u, count);
                const size_t slot = (count - 1) % nr_timing_samples[0].size();
                const auto update_stage = [&](size_t channel, std::atomic<double>& last,
                    std::atomic<double>& average, double value) {
                    if (count > nr_timing_samples[channel].size())
                        nr_timing_sums[channel] -= nr_timing_samples[channel][slot];
                    nr_timing_samples[channel][slot] = value;
                    nr_timing_sums[channel] += value;
                    last.store(value, std::memory_order_relaxed);
                    average.store(nr_timing_sums[channel] / samples,
                        std::memory_order_relaxed);
                };
                update_stage(0, nr_last_ms, nr_average_ms, ms);
                update_stage(1, prep_last_ms, prep_average_ms, prep);
                update_stage(2, host_last_ms, host_average_ms, host);
                update_stage(3, composite_submit_last_ms, composite_submit_average_ms, composite);
                update_stage(4, tail_last_ms, tail_average_ms, tail);
                nr_sample_count.store(count, std::memory_order_release);
                if (g_diag_active.load(std::memory_order_relaxed)) {
                    BridgeDiagFrameRecord drec{};
                    drec.frame_id = item.diagnostic.nr_frame_id;
                    drec.pending_seq = item.pending_seq;
                    drec.completed_index = completed;
                    drec.window_id = item.diag_window_id;
                    drec.diag_mode = tail_diag_mode;
                    drec.arm_reuse = item.diag_arm_reuse;
                    drec.late_latch = item.diag_late_latch;
                    drec.prepared_bypassed = item.nr_prepared_bypassed ? 1u : 0u;
                    drec.private_descriptors = item.nr_prepared_bypassed ? 1u : 0u;
                    drec.pool_alloc_kind = item.hdr ? item.hdr->diag_pool_alloc_kind() : 0u;
                    drec.composite_heap_kind = item.hdr ? item.hdr->diag_composite_heap_kind() : 0u;
                    drec.had_previous_processor = item.had_prev_proc;
                    drec.had_network_fence_pending = item.had_net_fence;
                    drec.prep_ms = prep;
                    drec.host_ms = host;
                    drec.composite_submit_ms = composite;
                    drec.tail_ms = tail;
                    drec.total_ms = ms;
                    if (item.record_timed && item.record_timing_epoch == item.nr_timing_epoch) {
                        drec.record_total_ms = rec_tot_ms;
                        drec.record_resources_ms = item.record_resources_ms;
                        drec.record_hdr_ms = item.record_hdr_ms;
                        drec.record_xess_ms = item.record_xess_ms;
                        drec.submit_gap_ms = sub_gap_ms;
                        drec.end_to_end_ms = e2e_ms;
                    }
                    if (tail_diag_mode >= 1) {
                        const uint64_t qf = ensure_qpc_freq();
                        drec.qpc_freq_hz = qf;
                        drec.retire_prev_cpu_ms = qpc_diff_ms(item.retire_prev_start_qpc, item.retire_prev_end_qpc, qf);
                        drec.prep_wait_cpu_ms = qpc_diff_ms(item.nr_started_qpc, item.nr_prepared_qpc, qf);
                        drec.vk_wait_fences_cpu_ms = qpc_diff_ms(item.vk_wait_start_qpc, item.vk_wait_end_qpc, qf);
                        const double host_qpc_ms = qpc_diff_ms(item.nr_prepared_qpc, item.nr_modeled_qpc, qf);
                        drec.vk_record_submit_cpu_ms = std::max(0.0, host_qpc_ms - drec.vk_wait_fences_cpu_ms);
                        drec.composite_submit_cpu_ms = qpc_diff_ms(item.nr_modeled_qpc, item.nr_pre_sr_qpc, qf);
                        drec.tail_sr_exec_cpu_ms = qpc_diff_ms(item.nr_pre_sr_qpc, item.deferred_exec_end_qpc, qf);
                        drec.tail_post_sr_submit_cpu_ms = qpc_diff_ms(item.deferred_exec_end_qpc, item.tail_submit_end_qpc, qf);
                        drec.tail_submit_cpu_ms = qpc_diff_ms(item.nr_pre_sr_qpc, item.tail_submit_end_qpc, qf);
                        drec.tail_wait_cpu_ms = qpc_diff_ms(item.tail_wait_start_qpc, item.tail_wait_end_qpc, qf);
                        drec.tail_retire_cpu_ms = qpc_diff_ms(item.tail_wait_end_qpc, item.ended_qpc, qf);
                    }
                    if (tail_diag_mode == 2 && item.diag_record_mode == 2 &&
                        item.frame_commands && item.frame_commands->ts_readback) {
                        drec.d3d_freq_hz = g_d3d_freq_hz;
                        D3D12_RANGE rr{0, 8 * sizeof(uint64_t)};
                        void* mapped = nullptr;
                        if (SUCCEEDED(item.frame_commands->ts_readback->Map(0, &rr, &mapped)) && mapped) {
                            const auto* ticks = static_cast<const uint64_t*>(mapped);
                            for (int qi = 0; qi < 8; ++qi) drec.d_ticks[qi] = ticks[qi];
                            D3D12_RANGE wr{0, 0};
                            item.frame_commands->ts_readback->Unmap(0, &wr);
                            if (drec.d_ticks[0] != 0 && drec.d_ticks[7] >= drec.d_ticks[0] && g_d3d_freq_hz != 0)
                                drec.d3d_valid = 1u;
                        }
                        if (!drec.d3d_valid) ++g_bridge_d3d_unavail;
                        nrvk_harvest_retired();
                    }
                    auto& vec = bridge_diag_records();
                    if (vec.size() < 32768) vec.push_back(drec);
                    else ++g_bridge_diag_overflow;
                }
                if (log && (completed <= 4 || completed % 120 == 0)) {
                    char line[280];
                    sprintf_s(line, "nr_stage_timing frame=%llu input=%u history=%u prep_ms=%.3f host_ms=%.3f composite_submit_ms=%.3f tail_ms=%.3f total_ms=%.3f\r\n",
                        static_cast<unsigned long long>(completed), item.frame_controls.input_height,
                        unsigned(item.frame_controls.history), prep, host, composite, tail, ms);
                    log(line);
                }
            }
        }
    } else {
        nr_failed.store(true);
        gpuhandoff::process_failed();
        flashdiag::recorder.count(flashdiag::Counter::failure);
        item.diagnostic_reason = !event ? flashdiag::Reason::completion_event :
            FAILED(registered) ? flashdiag::Reason::completion_registration :
            !done ? flashdiag::Reason::completion_timeout : flashdiag::Reason::retire_callback;
        flashdiag::recorder.event(flashdiag::EventKind::failure,
            item.diagnostic, item.diagnostic_reason, uint64_t(uint32_t(registered)), waited);
    }
    item.release_retired_resources();
    item.retired_done = true;
    flashdiag::recorder.finish(item.diagnostic, item.diagnostic_reason);
    return retired_ok;
#else
    (void)queue; (void)wait_if_incomplete; (void)log;
    const uint64_t done = item.completion ? item.completion->GetCompletedValue() : 0;
    if (done >= 1 && done != UINT64_MAX) {
        item.release_retired_resources();
        item.retired_done = true;
        flashdiag::recorder.finish(item.diagnostic, item.diagnostic_reason);
        return true;
    }
    return false;
#endif
}
void reap_locked() {
    for(size_t i=0;i<pending.size();) {
        if(pending[i]->submitted &&
           (pending[i]->retired_done ||
            retire_submitted_pending(*pending[i],nullptr,false,g_deferred_log))) {
            pending.erase(pending.begin()+i);
            retired.fetch_add(1,std::memory_order_relaxed);
        } else ++i;
    }
}
bool matching_device(ID3D12Device* device,ID3D12Resource* resource) {
    ComPtr<ID3D12Device> owner;
    return SUCCEEDED(resource->GetDevice(IID_PPV_ARGS(owner.GetAddressOf()))) &&
           owner.Get()==device;
}
}

bool record_deferred_identity(ID3D12GraphicsCommandList* game_list,
    const NVSDK_NGX_Handle* handle,NVSDK_NGX_Parameter* params,
    PFN_NVSDK_NGX_ProgressCallback callback,NgxEvaluate original,
    uint64_t game_list_generation,const NRB_Controls& frame_controls,
    NRB_Route route,NRB_RouteEvidence evidence,
    NVSDK_NGX_Result& result,flashdiag::Context* diagnostic) {
    // While a swap is draining no new record may enter: the component it
    // would point at is about to be torn down.
    if(swap_suspended.load(std::memory_order_relaxed)) return false;
#ifdef NRB_LIVE_NR_TEST
    const bool record_timed=nr_timing_enabled.load(std::memory_order_acquire);
    const uint64_t record_epoch=nr_timing_epoch.load(std::memory_order_acquire);
    const auto record_started=record_timed?std::chrono::steady_clock::now():
        std::chrono::steady_clock::time_point{};
#endif
    flashdiag::Context diag=diagnostic?*diagnostic:flashdiag::Context{};
    FrameCommands* frame_commands=nullptr;
    const auto reject=[&](flashdiag::Reason reason,uint64_t detail=0,
                          uint64_t detail2=0) {
        if(frame_commands) frame_commands->color_state_known=false;
        if(diagnostic) diagnostic->record_reason=uint32_t(reason);
        flashdiag::recorder.event(flashdiag::EventKind::record_reject,
            diag,reason,detail,detail2);
        if(g_deferred_log) {
            static std::atomic<uint32_t> reject_logs{0};
            if(reject_logs.fetch_add(1,std::memory_order_relaxed) < 32) {
                char line[192];
                sprintf_s(line,
                    "deferred_record_reject reason=%d detail=%llu detail2=%llu\r\n",
                    int(reason),(unsigned long long)detail,
                    (unsigned long long)detail2);
                g_deferred_log(line);
            }
        }
        return false;
    };
    if(!deferred_identity_tail_verified(route) && !tail_generic_profile())
        return reject(flashdiag::Reason::tail_unverified);
    if(!user_armed_diagnostic()) return reject(flashdiag::Reason::disarmed);
    if(!game_list || !handle || !params || !original || !game_list_generation)
        return reject(flashdiag::Reason::arguments);
    if(attempts.fetch_add(1,std::memory_order_relaxed)>=
#ifdef NRB_LIVE_NR_TEST
            100000
#else
            8
#endif
       ) return reject(flashdiag::Reason::attempt_budget);
    ID3D12Resource *color=nullptr,*output=nullptr;
    if(params->Get(NVSDK_NGX_Parameter_Color,&color)!=NVSDK_NGX_Result_Success ||
       params->Get(NVSDK_NGX_Parameter_Output,&output)!=NVSDK_NGX_Result_Success ||
        !color || !output) return reject(flashdiag::Reason::textures);
    const auto source_desc=color->GetDesc();
    const auto output_desc=output->GetDesc();
    // The game may deliver the SR color as R11G11B10_FLOAT (historical) or
    // R16G16B16A16_FLOAT. The copy, proxy and composite are format-agnostic
    // (copy_desc=source_desc); only the extent contract is fixed here.
    // Adaptive input: any 2D render-size source (>=64x64) is accepted; the
    // HdrProxy surfaces follow the source size and the Vulkan backend rescales
    // the frame onto the frozen 1280x720 network grid. Only the SR output
    // floor (>=1280x720) remains fixed by the model domain.
    if(source_desc.Dimension!=D3D12_RESOURCE_DIMENSION_TEXTURE2D ||
       (source_desc.Format!=DXGI_FORMAT_R11G11B10_FLOAT &&
        source_desc.Format!=DXGI_FORMAT_R16G16B16A16_FLOAT) ||
       source_desc.Width<64 || source_desc.Height<64 ||
       source_desc.DepthOrArraySize!=1 || source_desc.MipLevels!=1 ||
       source_desc.SampleDesc.Count!=1 ||
       output_desc.Dimension!=D3D12_RESOURCE_DIMENSION_TEXTURE2D ||
        // Any 2D SR output at least as large as the render target is allowed;
        // only the render extent is fixed by the model domain.
        output_desc.Width<1280 || output_desc.Height<720 ||
       output_desc.DepthOrArraySize!=1 || output_desc.MipLevels!=1 ||
        output_desc.SampleDesc.Count!=1)
        return reject(flashdiag::Reason::textures,
            (source_desc.Width<<32)|source_desc.Height,
            (output_desc.Width<<32)|output_desc.Height);

    {
        std::lock_guard guard(pending_mutex);
        reap_locked();
        if(pending.size()>=8)
            return reject(flashdiag::Reason::pending_full,pending.size());
        try {pending.reserve(8);}
        catch(...) {return reject(flashdiag::Reason::allocation);}
    }

    std::unique_ptr<Pending> owned;
    try {owned=std::make_unique<Pending>();}
    catch(...) {return reject(flashdiag::Reason::allocation);}
    Pending& entry=*owned;
    entry.diagnostic=diag;
#ifdef NRB_LIVE_NR_TEST
    entry.record_timed=record_timed;entry.record_timing_epoch=record_epoch;
    entry.record_started=record_started;
#endif
    entry.game_list=game_list;
    entry.original_color=color;
    entry.output=output;
    entry.frame_controls=frame_controls;
    entry.route=route;
    entry.evidence=evidence;
    ID3D12Resource* optional=nullptr;
    if(params->Get(NVSDK_NGX_Parameter_MotionVectors,&optional)==NVSDK_NGX_Result_Success && optional)
        entry.motion=optional;
    int reset=0;
    params->Get(NVSDK_NGX_Parameter_Reset,&reset);
    entry.game_reset=reset!=0;
    params->Get(NVSDK_NGX_Parameter_MV_Scale_X,&entry.motion_scale_x);
    params->Get(NVSDK_NGX_Parameter_MV_Scale_Y,&entry.motion_scale_y);
    entry.game_motion=frame_controls.history==NRB_HISTORY_REFERENCE ||
                      frame_controls.history==NRB_HISTORY_FUSED;
    unsigned int motion_base_x=0,motion_base_y=0;
    params->Get(NVSDK_NGX_Parameter_DLSS_Input_MV_SubrectBase_X,&motion_base_x);
    params->Get(NVSDK_NGX_Parameter_DLSS_Input_MV_SubrectBase_Y,&motion_base_y);
    int feature_flags=0;
    params->Get(NVSDK_NGX_Parameter_DLSS_Feature_Create_Flags,&feature_flags);
    if(entry.game_motion && (!entry.motion ||
       motion_base_x || motion_base_y ||
       (feature_flags & NVSDK_NGX_DLSS_Feature_Flags_MVJittered) ||
       !std::isfinite(entry.motion_scale_x) || !std::isfinite(entry.motion_scale_y) ||
       entry.motion_scale_x==0 || entry.motion_scale_y==0 ||
       std::fabs(entry.motion_scale_x)>4096 || std::fabs(entry.motion_scale_y)>4096))
        return reject(flashdiag::Reason::motion_contract,
            (uint64_t(motion_base_x)<<32)|motion_base_y,
            unsigned(feature_flags));
    optional=nullptr;
    if(params->Get(NVSDK_NGX_Parameter_Depth,&optional)==NVSDK_NGX_Result_Success && optional)
        entry.depth=optional;
    optional=nullptr;
    if(params->Get(NVSDK_NGX_Parameter_ExposureTexture,&optional)==NVSDK_NGX_Result_Success && optional)
        entry.exposure=optional;
    entry.game_list_generation=game_list_generation;
    if(FAILED(game_list->GetDevice(IID_PPV_ARGS(entry.device.GetAddressOf()))) ||
       !matching_device(entry.device.Get(),color) ||
        !matching_device(entry.device.Get(),output)) return reject(flashdiag::Reason::device);
#ifdef NRB_LIVE_NR_TEST
    const auto resource_started=record_timed?std::chrono::steady_clock::now():
        std::chrono::steady_clock::time_point{};
#endif
    auto copy_desc=source_desc;
    copy_desc.Flags=
#ifdef NRB_LIVE_NR_TEST
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
#else
        D3D12_RESOURCE_FLAG_NONE;
#endif
    frame_commands=acquire_frame_commands(entry.device.Get(),copy_desc);
    if(!frame_commands) return reject(flashdiag::Reason::record_resources,1);
    entry.frame_commands=frame_commands;
    entry.prep_allocator=frame_commands->prep_allocator;
    entry.prep_list=frame_commands->prep_list;
    entry.allocator=frame_commands->deferred_allocator;
    entry.deferred_list=frame_commands->deferred_list;
#ifdef NRB_LIVE_NR_TEST
    entry.composite_allocator=frame_commands->composite_allocator;
    entry.composite_list=frame_commands->composite_list;
#endif
    int reset_fail=0;
    const bool slot_is_fresh = entry.frame_commands && entry.frame_commands->fresh;
    HRESULT reset_hr=S_OK;
    // A list left open by an abandoned frame refuses Reset on some drivers;
    // closing first is harmless when it is already closed.
    (void)entry.prep_list->Close();
    (void)entry.deferred_list->Close();
    if(FAILED(reset_hr=entry.prep_allocator->Reset())) reset_fail=1;
    else if(FAILED(reset_hr=entry.prep_list->Reset(entry.prep_allocator.Get(),nullptr))) reset_fail=2;
    else if(FAILED(reset_hr=entry.allocator->Reset())) reset_fail=3;
    else if(FAILED(reset_hr=entry.deferred_list->Reset(entry.allocator.Get(),nullptr))) reset_fail=4;
    if (slot_is_fresh) { entry.frame_commands->fresh = false; reset_fail = 0; }
    bool commands_ok=(reset_fail==0);
#ifdef NRB_LIVE_NR_TEST
    if (!slot_is_fresh) {
        commands_ok=commands_ok &&
            SUCCEEDED(entry.composite_allocator->Reset()) &&
            SUCCEEDED(entry.composite_list->Reset(entry.composite_allocator.Get(),nullptr));
    }
#endif
    if(!commands_ok) {
        static std::atomic<unsigned> reset_diag{0};
        if(reset_diag.fetch_add(1,std::memory_order_relaxed)<3 && g_deferred_log) {
            char rb[256];
            sprintf_s(rb,"reset_fail=%d hr=%08x dev=%p alloc=%p list=%p gen=%llu\r\n",
                reset_fail,unsigned(reset_hr),(void*)entry.device.Get(),
                (void*)entry.prep_allocator.Get(),(void*)entry.prep_list.Get(),
                (unsigned long long)game_list_generation);
            g_deferred_log(rb);
        }
    if(!commands_ok) return reject(flashdiag::Reason::record_resources,20+reset_fail,uint64_t(uint32_t(reset_hr)));
    }
    if(FAILED(entry.device->CreateFence(0,D3D12_FENCE_FLAG_NONE,
        IID_PPV_ARGS(entry.completion.GetAddressOf()))))
        return reject(flashdiag::Reason::record_resources,3);
    entry.color_copy=frame_commands->color_copy;

#ifdef NRB_LIVE_NR_TEST
    if(record_timed) entry.record_resources_ms=std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now()-resource_started).count();
#endif
    auto* prep=entry.prep_list.Get();
    // The game's post-Evaluate tail leaves the DLSS color in UAV, but the
    // native XeSS route leaves it in a combined read state. The private list
    // executes only after that entire game list finishes.
    const auto color_tail_state=route==NRB_ROUTE_XESS?
        static_cast<D3D12_RESOURCE_STATES>(D3D12_RESOURCE_STATE_COPY_SOURCE |
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE):
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    const int lvl = prep_level();
#ifdef NRB_LIVE_NR_TEST
    entry.pending_seq = g_diag_pending_seq.fetch_add(1, std::memory_order_relaxed) + 1;
    entry.diag_record_mode = g_diag_mode.load(std::memory_order_relaxed);
    entry.diag_window_id = g_diag_window_id.load(std::memory_order_relaxed);
    entry.diag_arm_reuse = g_diag_arm_reuse.load(std::memory_order_relaxed);
    entry.diag_late_latch = g_diag_late_latch.load(std::memory_order_relaxed);
    const bool record_gpu_ts = (entry.diag_record_mode == 2 && frame_commands->ts_heap && frame_commands->ts_readback);
    if (record_gpu_ts) prep->EndQuery(frame_commands->ts_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
#endif
    if(lvl>=1 && frame_commands->color_state!=D3D12_RESOURCE_STATE_COPY_DEST) {
        auto to_copy_dest=transition(entry.color_copy.Get(),
            frame_commands->color_state,D3D12_RESOURCE_STATE_COPY_DEST);
        prep->ResourceBarrier(1,&to_copy_dest);
    }
    if(lvl>=2) {
    auto color_to_copy=transition(color,color_tail_state,
        D3D12_RESOURCE_STATE_COPY_SOURCE);
    prep->ResourceBarrier(1,&color_to_copy);
    }
    if(lvl>=3) prep->CopyResource(entry.color_copy.Get(),color);
    if(lvl>=2) {
    D3D12_RESOURCE_BARRIER ready[2]={
        transition(color,D3D12_RESOURCE_STATE_COPY_SOURCE,color_tail_state),
        transition(entry.color_copy.Get(),D3D12_RESOURCE_STATE_COPY_DEST,
            route==NRB_ROUTE_XESS?
                static_cast<D3D12_RESOURCE_STATES>(D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE |
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE):
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE)};
    prep->ResourceBarrier(2,ready);
    }
    frame_commands->color_state=route==NRB_ROUTE_XESS?
        static_cast<D3D12_RESOURCE_STATES>(D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE |
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE):
        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
#ifdef NRB_LIVE_NR_TEST
    if (record_gpu_ts) prep->EndQuery(frame_commands->ts_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 1);
    const auto hdr_started=record_timed?std::chrono::steady_clock::now():
        std::chrono::steady_clock::time_point{};
    wchar_t module_path[MAX_PATH]{};
    HMODULE module=nullptr;
    if(GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
        GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        reinterpret_cast<LPCWSTR>(&record_deferred_identity),&module) &&
       GetModuleFileNameW(module,module_path,MAX_PATH)) {
        std::wstring shader=module_path;
        shader.resize(shader.find_last_of(L"\\/")+1);
        shader+=L"nr_hdr_proxy.hlsl";
        entry.hdr=std::make_unique<HdrProxy>();
        if(!entry.hdr->initialize(entry.device.Get(),color,entry.motion.Get(),
            entry.color_copy.Get(),shader,&hdr_cache_session(),&hdr_resource_pool_session())) entry.hdr.reset();
    }
    if(record_timed) entry.record_hdr_ms=std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now()-hdr_started).count();
    if(entry.game_motion && !entry.hdr) return reject(flashdiag::Reason::hdr_missing);
    if(entry.hdr) {
        if(entry.game_motion && !entry.hdr->game_motion_available()) return reject(flashdiag::Reason::motion_proxy);
        hdr_ready.fetch_add(1,std::memory_order_relaxed);
        if (record_gpu_ts) entry.hdr->set_timestamp_query_target(frame_commands->ts_heap.Get(), frame_commands->ts_readback.Get());
        if(!tail_generic_profile()) entry.hdr->record_prepare(prep,color,entry.game_motion,
            entry.motion_scale_x,entry.motion_scale_y,color_tail_state);
        // Sample a few later frames too: the first frames after switching
        // modes can have a stationary camera and therefore zero motion.
        const unsigned motion_index=entry.game_motion?
            motion_probe_claimed.fetch_add(1,std::memory_order_relaxed):0;
        if(entry.game_motion) entry.diagnostic.motion_probe_index=motion_index;
        if(entry.game_motion && (motion_index<4 ||
           (motion_index<=480 && motion_index%60==0))) {
            entry.motion_probe_index=motion_index;
            UINT64 bytes=0;
            auto motion_desc=entry.hdr->nr_motion()->GetDesc();
            entry.device->GetCopyableFootprints(&motion_desc,0,1,0,
                &entry.motion_probe_layout,nullptr,nullptr,&bytes);
            D3D12_HEAP_PROPERTIES readback_heap{};
            readback_heap.Type=D3D12_HEAP_TYPE_READBACK;
            D3D12_RESOURCE_DESC buffer{};
            buffer.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER;
            buffer.Width=bytes;buffer.Height=1;buffer.DepthOrArraySize=1;
            buffer.MipLevels=1;buffer.SampleDesc.Count=1;
            buffer.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            if(bytes && SUCCEEDED(entry.device->CreateCommittedResource(
                &readback_heap,D3D12_HEAP_FLAG_NONE,&buffer,
                D3D12_RESOURCE_STATE_COPY_DEST,nullptr,
                IID_PPV_ARGS(entry.motion_probe_readback.GetAddressOf())))) {
                entry.diagnostic.stage_bits|=flashdiag::bit(flashdiag::Stage::motion_probe_recorded);
                auto to_copy=transition(entry.hdr->nr_motion(),
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                    D3D12_RESOURCE_STATE_COPY_SOURCE);
                prep->ResourceBarrier(1,&to_copy);
                D3D12_TEXTURE_COPY_LOCATION src{},dst{};
                src.pResource=entry.hdr->nr_motion();
                src.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
                src.SubresourceIndex=0;
                dst.pResource=entry.motion_probe_readback.Get();
                dst.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
                dst.PlacedFootprint=entry.motion_probe_layout;
                prep->CopyTextureRegion(&dst,0,0,0,&src,nullptr);
                auto restored=transition(entry.hdr->nr_motion(),
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                prep->ResourceBarrier(1,&restored);
            }
        }
        if(FAILED(entry.device->CreateFence(0,D3D12_FENCE_FLAG_NONE,
            IID_PPV_ARGS(entry.prepared.GetAddressOf()))))
            return reject(flashdiag::Reason::record_resources);
    }
#endif
    if(FAILED(prep->Close())) {
        static std::atomic<unsigned> close_diag{0};
        if(close_diag.fetch_add(1,std::memory_order_relaxed)<3 && g_deferred_log) {
            const auto cd = color ? color->GetDesc() : D3D12_RESOURCE_DESC{};
            const auto kd = entry.color_copy ? entry.color_copy->GetDesc() : D3D12_RESOURCE_DESC{};
            char cb[320];
            sprintf_s(cb,"close_fail lvl=%d color{fmt=%u %llux%u} copy{fmt=%u %llux%u} hdr=%d route=%u cached_state=%u\r\n",
                lvl,unsigned(cd.Format),(unsigned long long)cd.Width,cd.Height,
                unsigned(kd.Format),(unsigned long long)kd.Width,kd.Height,
                entry.hdr?1:0,unsigned(route),unsigned(frame_commands->color_state));
            g_deferred_log(cb);
        }
        return reject(flashdiag::Reason::record_close);
    }
    auto* list=entry.deferred_list.Get();
    auto output_write=transition(output,static_cast<D3D12_RESOURCE_STATES>(
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE |
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
#ifdef NRB_LIVE_NR_TEST
    if (record_gpu_ts) list->EndQuery(frame_commands->ts_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 6);
#endif
    list->ResourceBarrier(1,&output_write);
    params->Set(NVSDK_NGX_Parameter_Color,entry.color_copy.Get());
#ifdef NRB_LIVE_NR_TEST
    const auto xess_started=record_timed?std::chrono::steady_clock::now():
        std::chrono::steady_clock::time_point{};
#endif
    result=original(list,handle,params,callback);
#ifdef NRB_LIVE_NR_TEST
    if(record_timed) entry.record_xess_ms=std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now()-xess_started).count();
#endif
    params->Set(NVSDK_NGX_Parameter_Color,color);
    if(result!=NVSDK_NGX_Result_Success)
        return reject(flashdiag::Reason::original_evaluate,unsigned(result));
    auto output_read=transition(output,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        static_cast<D3D12_RESOURCE_STATES>(
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE |
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE));
    list->ResourceBarrier(1,&output_read);
#ifdef NRB_LIVE_NR_TEST
    if (record_gpu_ts) {
        list->EndQuery(frame_commands->ts_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 7);
        list->ResolveQueryData(frame_commands->ts_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 6, 2, frame_commands->ts_readback.Get(), 6 * sizeof(uint64_t));
    }
#endif
    if(FAILED(list->Close())) return reject(flashdiag::Reason::record_close);
#ifdef NRB_LIVE_NR_TEST
    if(entry.hdr && entry.hdr->resource_proof()) entry.hdr->resource_proof()->private_records_closed();
#endif
    {
        std::lock_guard guard(pending_mutex);
        reap_locked();
        entry.diagnostic.stage_bits|=flashdiag::bit(flashdiag::Stage::recorded);
        flashdiag::recorder.progress(entry.diagnostic);
#ifdef NRB_LIVE_NR_TEST
        if(record_timed) entry.record_finished=std::chrono::steady_clock::now();
#endif
        pending.push_back(std::move(owned));
    }
    recorded.fetch_add(1,std::memory_order_relaxed);
    flashdiag::recorder.count(flashdiag::Counter::delegated);
    return true;
}

unsigned submit_deferred_identity(ID3D12CommandQueue* queue,UINT count,
    ID3D12CommandList* const* lists,const SubmitObservation* observations,
    void (STDMETHODCALLTYPE *original_execute)(ID3D12CommandQueue*,UINT,
        ID3D12CommandList* const*),const NRB_Processor* processor,
    const NRB_Controls* controls,void (*log)(const char*)) {
    if(!queue || !count || !lists || !original_execute) return 0;
    std::lock_guard guard(pending_mutex);
#ifdef NRB_LIVE_NR_TEST
    if (!g_d3d_freq_hz) queue->GetTimestampFrequency(&g_d3d_freq_hz);
#endif
    reap_locked();
    if(pending.empty()) return 0;
    struct Match {UINT batch_index;Pending* item;};
    std::array<Match,8> matches{};
    unsigned matched=0;
    for(size_t p=0;p<pending.size();p++) {
        const auto& item=*pending[p];
        if(item.submitted) continue;
        for(UINT i=0;i<count;i++) if(item.game_list.Get()==lists[i]) {
            if(observations && (!observations[i].correlated ||
               item.game_list_generation!=observations[i].generation)) {
                generation_mismatches.fetch_add(1,std::memory_order_relaxed);
                flashdiag::recorder.count(flashdiag::Counter::generation_mismatch);
                flashdiag::recorder.event(flashdiag::EventKind::generation,
                    item.diagnostic,flashdiag::Reason::generation,
                    observations[i].generation,unsigned(observations[i].correlated));
            }
            if(matched==matches.size()) return 0;
            pending[p]->diagnostic.submit_batch=observations?observations[i].batch:0;
            matches[matched++]={i,pending[p].get()};break;
        }
    }
    if(!matched) return 0;
    std::sort(matches.begin(),matches.begin()+matched,
        [](const Match& a,const Match& b) {return a.batch_index<b.batch_index;});
    // A Direct list can only be executed on a Direct queue of its own device.
    // Allocate all fallible D3D12 objects while recording, before withholding
    // XeSS from the game list. Once that list appears here, always insert the
    // private list; a failed retirement Signal must not drop the frame.
    UINT begin=0;
#ifdef NRB_LIVE_NR_TEST
    Pending* previous_processor=nullptr;
#endif
    for(unsigned m=0;m<matched;m++) {
        const UINT index=matches[m].batch_index;
        if(index>=begin) {
            original_execute(queue,index+1-begin,lists+begin);
            begin=index+1;
        }
        flashdiag::recorder.count(flashdiag::Counter::submitted);
        matches[m].item->diagnostic.stage_bits|=flashdiag::bit(flashdiag::Stage::submitted);
        ID3D12CommandList* prep=matches[m].item->prep_list.Get();
#ifdef NRB_LIVE_NR_TEST
        if(matches[m].item->hdr && matches[m].item->hdr->resource_proof())
            matches[m].item->hdr->resource_proof()->prepare_submitted();
#endif
        original_execute(queue,1,&prep);
#ifdef NRB_LIVE_NR_TEST
        auto& item=*matches[m].item;
        item.nr_queue_exposed=true;
        item.submit_entry_tp = nr_timing_enabled.load(std::memory_order_acquire)
            ? std::chrono::steady_clock::now()
            : std::chrono::steady_clock::time_point{};
        // Keep the single helper's previous output alive through real completion.
        // Failure latches NR; never ignore busy or reset the helper's live frame.
        const uint32_t sub_diag_mode = g_diag_mode.load(std::memory_order_relaxed);
        item.diag_late_latch = g_diag_late_latch.load(std::memory_order_relaxed);
        item.had_prev_proc = previous_processor ? 1u : 0u;
        for (auto& prev_ptr : pending) {
            if (prev_ptr && prev_ptr.get() != &item && prev_ptr->submitted && !prev_ptr->retired_done) {
                item.had_prev_proc = 1u;
                retire_submitted_pending(*prev_ptr, queue, true, log);
            }
        }
        if (sub_diag_mode >= 1) item.retire_prev_start_qpc = qpc_now();
        if(previous_processor && !previous_processor->retired_done && !retire_previous_for_reuse(*previous_processor,queue)) {
            if(log) log("deferred_nr_retire_previous_failed\r\n");
            gpuhandoff::process_failed();
        }
        if (sub_diag_mode >= 1) item.retire_prev_end_qpc = qpc_now();
        previous_processor=nullptr;
        auto fallback_reason=flashdiag::Reason::prepared_wait;
        if(!item.hdr) fallback_reason=flashdiag::Reason::hdr_missing;
        else if(!processor || !processor->process || !processor->retire)
            fallback_reason=flashdiag::Reason::processor_missing;
        else if(!controls || !controls->enabled) fallback_reason=flashdiag::Reason::disabled;
        else if(nr_failed.load()) fallback_reason=flashdiag::Reason::latched_failure;
        if(item.hdr && processor && processor->process && processor->retire &&
           controls && controls->enabled && !nr_failed.load()) {
            item.nr_timing_epoch=nr_timing_epoch.load(std::memory_order_acquire);
            item.nr_timed=nr_timing_enabled.load(std::memory_order_acquire);
            if (sub_diag_mode >= 1) item.nr_started_qpc = qpc_now();
            const auto started=item.nr_timed?std::chrono::steady_clock::now():
                std::chrono::steady_clock::time_point{};
            bool valid=false;
            NRB_Result result{};
            const HRESULT prepare_signal=queue->Signal(item.prepared.Get(),1);
            if(SUCCEEDED(prepare_signal)) {
                // Signal the ACTUAL prepared point first. The first enable frame
                // has no confirmed capability and follows the CPU wait below.
                const bool bypass=gpuhandoff::prepared_bypass(*processor,item.device.Get(),queue,
                    item.motion_probe_readback.Get()!=nullptr);
                item.nr_prepared_bypassed=bypass;
                HANDLE event=bypass?nullptr:CreateEventW(nullptr,FALSE,FALSE,nullptr);
                if(bypass || event) {
                    const HRESULT registered=bypass?S_OK:item.prepared->SetEventOnCompletion(1,event);
                    const DWORD waited=!bypass && SUCCEEDED(registered)?WaitForSingleObject(event,10000):WAIT_FAILED;
                    const auto prepared_done=item.hdr->resource_proof()?item.prepared->GetCompletedValue():0;
                    const bool pooled_prepare_ok=!item.hdr->resource_proof() ||
                        (prepared_done>=1 && prepared_done!=UINT64_MAX &&
                         SUCCEEDED(item.device->GetDeviceRemovedReason()));
                    if(bypass || (SUCCEEDED(registered) && waited==WAIT_OBJECT_0 && pooled_prepare_ok)) {
                        if(auto* proof=item.hdr->resource_proof()) {
                            if(bypass) proof->prepare_gpu_ordered(true);
                            else proof->prepare_completed(true);
                        }
                        if(item.motion_probe_readback && log) {
                            void* mapped=nullptr;
                            if(SUCCEEDED(item.motion_probe_readback->Map(0,nullptr,
                                &mapped)) && mapped) {
                                // The motion surface follows the source size; never
                                // sample past its actual dimensions.
                                const auto md=item.hdr->nr_motion()->GetDesc();
                                const unsigned pw=unsigned(md.Width),ph=unsigned(md.Height);
                                unsigned tested=0,nonzero=0,max_half=0;
                                const auto* data=static_cast<const uint8_t*>(mapped);
                                for(unsigned y=0;y<ph;y+=16) {
                                    const auto* row=reinterpret_cast<const uint16_t*>(
                                        data+y*item.motion_probe_layout.Footprint.RowPitch);
                                    for(unsigned x=0;x<pw;x+=16) {
                                        const unsigned u=row[2*x]&0x7fff;
                                        const unsigned v=row[2*x+1]&0x7fff;
                                        ++tested;
                                        nonzero+=(u|v)!=0;
                                        max_half=std::max({max_half,u,v});
                                    }
                                }
                                item.motion_probe_readback->Unmap(0,nullptr);
                                char line[240];
                                sprintf_s(line,"game_motion_probe sample=%u pixels=%u nonzero=%u max_half_bits=%u scale=[%.3f,%.3f] reset=%u\r\n",
                                    item.motion_probe_index,tested,nonzero,max_half,item.motion_scale_x,
                                    item.motion_scale_y,unsigned(item.game_reset));
                                log(line);
                            }
                        }
                        if(item.nr_timed) item.nr_prepared=std::chrono::steady_clock::now();
                        if (sub_diag_mode >= 1) item.nr_prepared_qpc = qpc_now();
                        NRB_Frame frame{};
                        frame.abi_size=sizeof(frame);frame.route=item.route;
                        frame.route_evidence=item.evidence;
                        frame.device=item.device.Get();frame.queue=queue;
                        frame.command_list=item.prep_list.Get();
                        frame.color=item.hdr->proxy_color();
                        frame.motion=item.hdr->nr_motion();
                        frame.output=item.output.Get();
                        // Adaptive: the render size follows the actual source;
                        // the Vulkan backend rescales onto the 1280x720 grid.
                        const auto src_desc=item.original_color->GetDesc();
                        frame.render_width=uint32_t(src_desc.Width);
                        frame.render_height=uint32_t(src_desc.Height);
    {const auto od=item.output?item.output->GetDesc():D3D12_RESOURCE_DESC{};
     frame.output_width=uint32_t(od.Width);frame.output_height=od.Height;}
                        frame.motion_scale_x=1;frame.motion_scale_y=1;
                        frame.motion_scale_origin=item.route==NRB_ROUTE_FSR?
                            NRB_MV_FSR_TO_NGX:item.route==NRB_ROUTE_XESS?
                            NRB_MV_XESS_TO_NGX:NRB_MV_DLSS_NGX;
                        frame.pre_exposure=1;frame.exposure_scale=1;
                        frame.low_resolution_motion=1;
                        frame.frame_id=frame_serial.fetch_add(1)+1;
                        frame.reset_history=frame.frame_id==1 || item.game_reset;
                        frame.controls=item.frame_controls;
                        result.abi_size=sizeof(result);
                        fallback_reason=flashdiag::Reason::processor_result;
                        item.diagnostic.nr_frame_id=frame.frame_id;
                        item.diagnostic.nr_reset=frame.reset_history!=0;
                        item.diagnostic.stage_bits|=flashdiag::bit(flashdiag::Stage::processor_called);
                        flashdiag::recorder.count(flashdiag::Counter::nr_started);
                        flashdiag::recorder.progress(item.diagnostic);
                        int processor_result=-1;
                        item.nr_processor=*processor;item.nr_processor_called=true;
                        if(auto* proof=item.hdr->resource_proof()) proof->processor_begin();
                        try {
                            gpuhandoff::CallbackScope producer_scope(*processor,frame,item.prepared.Get(),1,!bypass);
                            flashdiag::ActiveContext context_scope(item.diagnostic);
                            processor_result=processor->process(processor->context,&frame,&result);
                            if(processor_result==0) {
                                item.diagnostic.stage_bits|=flashdiag::bit(flashdiag::Stage::processor_return_ok);
                                flashdiag::recorder.count(flashdiag::Counter::nr_return_ok);
                            }
                            valid=processor_result==0 && result.color && result.ready_fence && result.ready_value &&
                                result.color->GetDesc().Format==DXGI_FORMAT_R32G32B32A32_FLOAT &&
                                result.color->GetDesc().Width==frame.render_width &&
                                result.color->GetDesc().Height==frame.render_height;
                        } catch(...) {valid=false;fallback_reason=flashdiag::Reason::cpp_exception;}
                        // Returned references, even malformed ones, stay with
                        // Pending when processor use is unproven (pool ON/OFF).
                        item.nr_output.Attach(result.color);item.nr_ready.Attach(result.ready_fence);
                        if(auto* proof=item.hdr->resource_proof()) if(!valid) proof->poison();
                        if(valid) {
                            item.diagnostic.stage_bits|=flashdiag::bit(flashdiag::Stage::result_valid);
                            flashdiag::recorder.count(flashdiag::Counter::nr_processed);
                        } else flashdiag::recorder.event(flashdiag::EventKind::failure,item.diagnostic,
                            fallback_reason,uint64_t(int64_t(processor_result)));
                        if(item.nr_timed) item.nr_modeled=std::chrono::steady_clock::now();
                        if (sub_diag_mode >= 1) {
                            item.nr_modeled_qpc = qpc_now();
                            nrvk_get_last_cpu_substage(&item.had_net_fence, &item.vk_wait_start_qpc, &item.vk_wait_end_qpc);
                        }
                    }
                    else {
                        fallback_reason=FAILED(registered)?flashdiag::Reason::prepare_registration:
                            flashdiag::Reason::prepare_timeout;
                        flashdiag::recorder.event(flashdiag::EventKind::failure,item.diagnostic,
                            fallback_reason,uint64_t(uint32_t(registered)),waited);
                    }
                    if(event) CloseHandle(event);
                } else fallback_reason=flashdiag::Reason::prepare_event;
            } else fallback_reason=flashdiag::Reason::prepare_signal;
            HRESULT ready_wait_hr=E_FAIL;
            uint64_t ready_done=0, ready_value=0;
            if(valid) {
                fallback_reason=flashdiag::Reason::composite_or_wait;
                ready_wait_hr=queue->Wait(item.nr_ready.Get(),result.ready_value);
                ready_done=item.nr_ready?item.nr_ready->GetCompletedValue():0;
                ready_value=result.ready_value;
                const HRESULT ready_wait=ready_wait_hr;
                if(auto* proof=item.hdr->resource_proof()) proof->processor_ready(SUCCEEDED(ready_wait));
                bool composite_recorded=false;HRESULT composite_closed=S_FALSE;
                valid=SUCCEEDED(ready_wait) &&
                    (composite_recorded=
                    item.hdr->record_composite(item.composite_list.Get(),
                        item.original_color.Get(),item.nr_output.Get(),item.color_copy.Get(),
                        int(item.frame_controls.style),item.frame_controls.local_tone,
                        item.frame_controls.display_strength,
                        item.route==NRB_ROUTE_XESS?
                            static_cast<D3D12_RESOURCE_STATES>(
                                D3D12_RESOURCE_STATE_COPY_SOURCE |
                                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE):
                            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        item.route==NRB_ROUTE_XESS?
                            static_cast<D3D12_RESOURCE_STATES>(
                                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE |
                                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE):
                            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                        item.nr_prepared_bypassed)) &&
                    SUCCEEDED(composite_closed=item.composite_list->Close());
                if(!valid) fallback_reason=FAILED(ready_wait)?flashdiag::Reason::queue_wait:
                    !composite_recorded?flashdiag::Reason::composite_record:flashdiag::Reason::composite_close;
                if(valid) {
                    ID3D12CommandList* composite=item.composite_list.Get();
                    if(auto* proof=item.hdr->resource_proof()) {
                        proof->composite_closed();proof->composite_submitted();
                    }
                    original_execute(queue,1,&composite);
                    flashdiag::recorder.count(flashdiag::Counter::nr_composited);
                    item.diagnostic.stage_bits|=flashdiag::bit(flashdiag::Stage::composite_submitted);
                    if(item.nr_timed) item.nr_pre_sr=std::chrono::steady_clock::now();
                    if (sub_diag_mode >= 1) item.nr_pre_sr_qpc = qpc_now();
                    item.nr_processor=*processor;
                    item.nr_submitted=true;
                    item.nr_started=started;
                    previous_processor=&item;
                }
            }
            if(!valid) {
                if(log) {
                    char line[256];
                    sprintf_s(line,"deferred_nr_frame_failed reason=%d frame=%llu processor_called=%d wait_hr=0x%08lX completed=%llu value=%llu\r\n",
                        int(fallback_reason),(unsigned long long)item.diagnostic.nr_frame_id,
                        int(item.nr_processor_called),unsigned long(ready_wait_hr),
                        (unsigned long long)ready_done,(unsigned long long)ready_value);
                    log(line);
                }
                if(item.hdr->resource_proof()) item.hdr->quarantine_resource_lease();
                nr_failed.store(true);
                gpuhandoff::process_failed();
                flashdiag::recorder.count(flashdiag::Counter::failure);
                flashdiag::recorder.event(flashdiag::EventKind::failure,
                    item.diagnostic,fallback_reason);
            }
        }
        if(!item.nr_submitted) {
            item.diagnostic_reason=fallback_reason;
            flashdiag::recorder.fallback(item.diagnostic,fallback_reason);
        }
#else
        (void)processor;(void)controls;
        flashdiag::recorder.fallback(matches[m].item->diagnostic,
            flashdiag::Reason::disabled);
#endif
        ID3D12CommandList* deferred=matches[m].item->deferred_list.Get();
#ifdef NRB_LIVE_NR_TEST
        if(matches[m].item->hdr && matches[m].item->hdr->resource_proof())
            matches[m].item->hdr->resource_proof()->deferred_submitted();
#endif
        original_execute(queue,1,&deferred);
#ifdef NRB_LIVE_NR_TEST
        if (g_diag_mode.load(std::memory_order_relaxed) >= 1) matches[m].item->deferred_exec_end_qpc = qpc_now();
#endif
        matches[m].item->diagnostic.stage_bits|=flashdiag::bit(flashdiag::Stage::sr_submitted);
        flashdiag::recorder.progress(matches[m].item->diagnostic);
    }
    if(begin<count) original_execute(queue,count-begin,lists+begin);
    for(unsigned m=0;m<matched;m++) {
        auto& item=*matches[m].item;
        item.submitted=true;
        const HRESULT completion_signal=queue->Signal(item.completion.Get(),1);
#ifdef NRB_LIVE_NR_TEST
        if (g_diag_mode.load(std::memory_order_relaxed) >= 1) item.tail_submit_end_qpc = qpc_now();
#endif
#ifdef NRB_LIVE_NR_TEST
        if(item.hdr && item.hdr->resource_proof()) {
            item.hdr->resource_proof()->completion_signal(SUCCEEDED(completion_signal));
            if(FAILED(completion_signal)) item.hdr->quarantine_resource_lease();
        }
#endif
        if(FAILED(completion_signal)) {
            item.diagnostic_reason=flashdiag::Reason::completion_signal;
            signal_failures.fetch_add(1,std::memory_order_relaxed);
            flashdiag::recorder.count(flashdiag::Counter::failure);
            flashdiag::recorder.event(flashdiag::EventKind::failure,
                item.diagnostic,flashdiag::Reason::completion_signal);
            item.retired_done=true;
            flashdiag::recorder.finish(item.diagnostic,item.diagnostic_reason);
        }
#ifdef NRB_LIVE_NR_TEST
        else if(item.nr_submitted) {
            const bool use_late_latch = (g_diag_late_latch.load(std::memory_order_relaxed) != 0);
            item.diag_late_latch = use_late_latch ? 1u : 0u;
            if (use_late_latch) {
                if (item.nr_timed) item.submit_return_tp = std::chrono::steady_clock::now();
                item.submit_return_qpc = qpc_now();
            } else {
                retire_submitted_pending(item, queue, true, log);
            }
        }
        else {
            retire_submitted_pending(item, queue, false, log);
        }
#else
        else {
            retire_submitted_pending(item, queue, false, log);
        }
#endif
    }
    reap_locked();
    // A failed Signal retains the submitted objects until process exit; never
    // recycle an allocator or texture that might still be used by the GPU.
    return matched;
}
unsigned deferred_identity_recorded() {return recorded.load(std::memory_order_relaxed);}
uint64_t deferred_nr_processed() {
#ifdef NRB_LIVE_NR_TEST
    return nr_completed.load(std::memory_order_relaxed);
#else
    return 0;
#endif
}
double deferred_nr_last_ms() {
#ifdef NRB_LIVE_NR_TEST
    return nr_last_ms.load(std::memory_order_relaxed);
#else
    return 0;
#endif
}
double deferred_nr_average_ms() {
#ifdef NRB_LIVE_NR_TEST
    return nr_average_ms.load(std::memory_order_relaxed);
#else
    return 0;
#endif
}
bool deferred_hdr_cache_stats(NRB_HdrCacheStats* stats) {
    if(!stats || stats->abi_size!=sizeof(NRB_HdrCacheStats)) return false;
    NRB_HdrCacheStats out{};out.abi_size=sizeof(out);
#ifdef NRB_LIVE_NR_TEST
    const auto value=hdr_cache_session().stats();out.enabled=value.enabled?1:0;
    out.shader_compile_calls=value.shader_compile_calls;
    out.shader_compile_failures=value.shader_compile_failures;
    out.source_reads=value.source_reads;
    out.source_hashes=value.source_hashes;
    out.pso_creates=value.pso_creates;
    out.root_signature_creates=value.root_signature_creates;
    out.pipeline_hits=value.pipeline_hits;
    out.pipeline_misses=value.pipeline_misses;
    out.uncached_initializations=value.uncached_initializations;
    out.shader_generation=value.shader_generation;
    out.resident_device_entries=value.resident_device_entries;
#endif
    *stats=out;return true;
}
bool deferred_hdr_set_resource_pool_enabled(bool enabled) {
#ifdef NRB_LIVE_NR_TEST
    hdr_resource_pool_session().set_enabled(enabled);return true;
#else
    (void)enabled;return false;
#endif
}
bool deferred_hdr_resource_pool_stats(HdrResourcePoolStats& stats) {
#ifdef NRB_LIVE_NR_TEST
    stats=hdr_resource_pool_session().stats();return true;
#else
    (void)stats;return false;
#endif
}
void deferred_hdr_set_cache_enabled(bool enabled) {
#ifdef NRB_LIVE_NR_TEST
    hdr_cache_session().set_enabled(enabled);
#else
    (void)enabled;
#endif
}
bool deferred_hdr_warmup_shader() {
#ifdef NRB_LIVE_NR_TEST
    HMODULE module=nullptr;wchar_t path[MAX_PATH]{};
    if(!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
        GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        reinterpret_cast<LPCWSTR>(&record_deferred_identity),&module) ||
        !GetModuleFileNameW(module,path,MAX_PATH)) return false;
    std::wstring shader=path;const auto slash=shader.find_last_of(L"\\/");
    if(slash==std::wstring::npos) return false;
    shader.resize(slash+1);shader+=L"nr_hdr_proxy.hlsl";
    return hdr_cache_session().warmup_shader(shader);
#else
    return false;
#endif
}
bool deferred_nr_record_times(NRB_RecordTimes* times) {
    if(!times || times->abi_size!=sizeof(NRB_RecordTimes)) return false;
#ifdef NRB_LIVE_NR_TEST
    std::lock_guard guard(nr_timing_mutex);
    return record_timing_window.snapshot(times);
#else
    *times=NRB_RecordTimes{};times->abi_size=sizeof(NRB_RecordTimes);return false;
#endif
}
bool deferred_nr_stage_times(NRB_StageTimes* times) {
    if(!times || times->abi_size!=sizeof(NRB_StageTimes)) return false;
    NRB_StageTimes result{};
    result.abi_size=sizeof(result);
#ifdef NRB_LIVE_NR_TEST
    std::lock_guard guard(nr_timing_mutex);
    result.frames=nr_sample_count.load(std::memory_order_relaxed);
    result.prep_last_ms=prep_last_ms.load(std::memory_order_relaxed);
    result.prep_average_ms=prep_average_ms.load(std::memory_order_relaxed);
    result.host_last_ms=host_last_ms.load(std::memory_order_relaxed);
    result.host_average_ms=host_average_ms.load(std::memory_order_relaxed);
    result.composite_submit_last_ms=composite_submit_last_ms.load(std::memory_order_relaxed);
    result.composite_submit_average_ms=composite_submit_average_ms.load(std::memory_order_relaxed);
    result.tail_last_ms=tail_last_ms.load(std::memory_order_relaxed);
    result.tail_average_ms=tail_average_ms.load(std::memory_order_relaxed);
#endif
    *times=result;
    return result.frames>0;
}
namespace {
void reset_nr_timing_locked() {
#ifdef NRB_LIVE_NR_TEST
    nr_timing_count=0;
    record_timing_window.reset();
    nr_timing_sums.fill(0);
    for(auto& samples:nr_timing_samples) samples.fill(0);
    nr_last_ms.store(0,std::memory_order_relaxed);
    nr_average_ms.store(0,std::memory_order_relaxed);
    nr_sample_count.store(0,std::memory_order_relaxed);
    for(auto* value:{&prep_last_ms,&prep_average_ms,&host_last_ms,&host_average_ms,
                     &composite_submit_last_ms,&composite_submit_average_ms,
                     &tail_last_ms,&tail_average_ms})
        value->store(0,std::memory_order_relaxed);
#endif
}
}
bool deferred_nr_timing_enabled() {
#ifdef NRB_LIVE_NR_TEST
    return nr_timing_enabled.load(std::memory_order_acquire);
#else
    return false;
#endif
}
void deferred_nr_set_timing_enabled(bool enabled) {
#ifdef NRB_LIVE_NR_TEST
    std::lock_guard guard(nr_timing_mutex);
    if(nr_timing_enabled.load(std::memory_order_relaxed)==enabled) return;
    nr_timing_enabled.store(enabled,std::memory_order_release);
    nr_timing_epoch.fetch_add(1,std::memory_order_release);
    reset_nr_timing_locked();
#else
    (void)enabled;
#endif
}
void deferred_nr_reset_metrics() {
#ifdef NRB_LIVE_NR_TEST
    std::lock_guard guard(nr_timing_mutex);
    nr_timing_epoch.fetch_add(1,std::memory_order_release);
    reset_nr_timing_locked();
#endif
}
unsigned deferred_hdr_ready() {
#ifdef NRB_LIVE_NR_TEST
    return hdr_ready.load(std::memory_order_relaxed);
#else
    return 0;
#endif
}
bool deferred_nr_failed() {
#ifdef NRB_LIVE_NR_TEST
    return nr_failed.load(std::memory_order_relaxed);
#else
    return false;
#endif
}
unsigned deferred_identity_retired() {return retired.load(std::memory_order_relaxed);}
unsigned deferred_identity_signal_failures() {return signal_failures.load(std::memory_order_relaxed);}
unsigned deferred_identity_generation_mismatches() {return generation_mismatches.load(std::memory_order_relaxed);}
unsigned deferred_identity_invalidated() {return invalidated.load(std::memory_order_relaxed);}
void deferred_identity_list_reset(ID3D12GraphicsCommandList* list) {
    if(!list) return;
    std::lock_guard guard(pending_mutex);
    for(size_t i=0;i<pending.size();) {
        if(!pending[i]->submitted && pending[i]->game_list.Get()==list) {
#ifdef NRB_LIVE_NR_TEST
            // A recursive Reset during insertion must not erase GPU-exposed
            // ownership merely because the final submitted flag is not set.
            if(pending[i]->hdr && pending[i]->hdr->resource_proof() &&
               !pending[i]->hdr->resource_proof()->pristine_unsubmitted()) {
                pending[i]->hdr->quarantine_resource_lease();++i;continue;
            }
#endif
            pending[i]->diagnostic.stage_bits|=flashdiag::bit(flashdiag::Stage::invalidated);
            flashdiag::recorder.count(flashdiag::Counter::invalidated);
            flashdiag::recorder.event(flashdiag::EventKind::invalidated,
                pending[i]->diagnostic,flashdiag::Reason::list_reset);
            flashdiag::recorder.finish(pending[i]->diagnostic,flashdiag::Reason::list_reset);
            pending.erase(pending.begin()+i);
            invalidated.fetch_add(1,std::memory_order_relaxed);
        } else ++i;
    }
}
void observe_deferred_identity_tail(const TailSummary& s) {
    if(!s.valid || s.route==NRB_ROUTE_UNKNOWN || s.route>NRB_ROUTE_XESS) return;
    const bool safe=tail_safe_for_deferred_identity(s);
    if(safe) tail_valid_samples[s.route].fetch_add(1,std::memory_order_relaxed);
    tail_samples[s.route].fetch_add(1,std::memory_order_release);
}
void deferred_identity_abandon_all() {
    std::lock_guard<std::recursive_mutex> guard(pending_mutex);
    for(auto& item:pending) if(item) abandoned.push_back(std::move(item));
    pending.clear();
}
void deferred_identity_clear_failure() {
    // Only a full component replacement may drop this latch: the failure
    // belonged to the component that was just torn down.
    nr_failed.store(false,std::memory_order_relaxed);
}
void deferred_identity_set_suspended(bool on) {
    swap_suspended.store(on,std::memory_order_relaxed);
}
unsigned deferred_identity_pending() {
    std::lock_guard<std::recursive_mutex> guard(pending_mutex);
    return unsigned(pending.size());
}
void reset_deferred_identity_tail(NRB_Route route) {
    if(route==NRB_ROUTE_UNKNOWN || route>NRB_ROUTE_XESS) return;
    tail_samples[route].store(0,std::memory_order_release);
    tail_valid_samples[route].store(0,std::memory_order_release);
}
bool deferred_identity_tail_verified(NRB_Route route) {
    // Counters are reset on list/generation changes; require at least 8
    // observations rather than exactly 8. The exact form locked out a valid
    // tail once a ninth safe sample arrived before the next reset.
    return route!=NRB_ROUTE_UNKNOWN && route<=NRB_ROUTE_XESS &&
           tail_samples[route].load(std::memory_order_acquire)>=8 &&
           tail_valid_samples[route].load(std::memory_order_relaxed)>=8;
}

#ifdef NRB_LIVE_NR_TEST
static void flush_diag_files_locked(const std::wstring& out_dir) {
    if (out_dir.empty()) return;
    const auto& brecs = bridge_diag_records();
    const NrvkDiagFrameRecordMirror* nrecs = nullptr;
    uint32_t ncount = 0;
    uint64_t nd3d_unavail = 0, nvk_unavail = 0, noverflow = 0;
    nrvk_get_records(&nrecs, &ncount, &nd3d_unavail, &nvk_unavail, &noverflow);

    std::unordered_map<uint64_t, const NrvkDiagFrameRecordMirror*> nmap;
    nmap.reserve(ncount);
    for (uint32_t i = 0; i < ncount; ++i) {
        nmap[nrecs[i].frame_id] = &nrecs[i];
    }

    const std::wstring cpu_path = out_dir + L"\\CPU-FRAMES.jsonl";
    FILE* fc = nullptr;
    if (_wfopen_s(&fc, cpu_path.c_str(), L"wb") == 0 && fc) {
        for (const auto& r : brecs) {
            uint32_t slot_idx = 0;
            auto it = nmap.find(r.frame_id);
            if (it != nmap.end()) slot_idx = it->second->slot_index;
            std::fprintf(fc,
                "{\"frame_id\":%llu,\"pending_seq\":%llu,\"completed_index\":%llu,"
                "\"window_id\":%u,\"diag_mode\":%u,\"arm_reuse\":%u,\"late_latch\":%u,"
                "\"slot_index\":%u,\"prepared_bypassed\":%u,\"private_descriptors\":%u,"
                "\"pool_alloc_kind\":%u,\"composite_heap_kind\":%u,"
                "\"had_previous_processor\":%u,\"had_network_fence_pending\":%u,"
                "\"prep_ms\":%.6f,\"host_ms\":%.6f,\"composite_submit_ms\":%.6f,"
                "\"tail_ms\":%.6f,\"total_ms\":%.6f,"
                "\"record_total_ms\":%.6f,\"record_resources_ms\":%.6f,"
                "\"record_hdr_ms\":%.6f,\"record_xess_ms\":%.6f,"
                "\"submit_gap_ms\":%.6f,\"end_to_end_ms\":%.6f,"
                "\"qpc_freq_hz\":%llu,"
                "\"retire_prev_cpu_ms\":%.6f,\"prep_wait_cpu_ms\":%.6f,"
                "\"vk_wait_fences_cpu_ms\":%.6f,\"vk_record_submit_cpu_ms\":%.6f,"
                "\"composite_submit_cpu_ms\":%.6f,\"tail_sr_exec_cpu_ms\":%.6f,"
                "\"tail_post_sr_submit_cpu_ms\":%.6f,\"tail_submit_cpu_ms\":%.6f,"
                "\"tail_wait_cpu_ms\":%.6f,\"tail_retire_cpu_ms\":%.6f}\n",
                (unsigned long long)r.frame_id,
                (unsigned long long)r.pending_seq,
                (unsigned long long)r.completed_index,
                r.window_id, r.diag_mode, r.arm_reuse, r.late_latch,
                slot_idx, r.prepared_bypassed, r.private_descriptors,
                r.pool_alloc_kind, r.composite_heap_kind,
                r.had_previous_processor, r.had_network_fence_pending,
                r.prep_ms, r.host_ms, r.composite_submit_ms,
                r.tail_ms, r.total_ms,
                r.record_total_ms, r.record_resources_ms,
                r.record_hdr_ms, r.record_xess_ms,
                r.submit_gap_ms, r.end_to_end_ms,
                (unsigned long long)r.qpc_freq_hz,
                r.retire_prev_cpu_ms, r.prep_wait_cpu_ms,
                r.vk_wait_fences_cpu_ms, r.vk_record_submit_cpu_ms,
                r.composite_submit_cpu_ms, r.tail_sr_exec_cpu_ms,
                r.tail_post_sr_submit_cpu_ms, r.tail_submit_cpu_ms,
                r.tail_wait_cpu_ms, r.tail_retire_cpu_ms);
        }
        std::fclose(fc);
    }

    const std::wstring gpu_path = out_dir + L"\\GPU-STAGES.jsonl";
    FILE* fg = nullptr;
    if (_wfopen_s(&fg, gpu_path.c_str(), L"wb") == 0 && fg) {
        for (const auto& r : brecs) {
            if (r.diag_mode != 2) continue;
            const NrvkDiagFrameRecordMirror* nr = nullptr;
            auto it = nmap.find(r.frame_id);
            if (it != nmap.end()) nr = it->second;

            const uint64_t dfreq = r.d3d_freq_hz ? r.d3d_freq_hz : (nr ? nr->d3d_freq_hz : 0);
            const double vperiod = nr ? nr->vk_timestamp_period_ns : 0.0;
            const uint32_t vbits = nr ? nr->vk_valid_bits : 0;
            const uint32_t nd3d_ok = nr ? nr->d3d_valid : 0;
            const uint32_t nvk_ok = nr ? nr->vk_valid : 0;
            const uint64_t d0 = r.d_ticks[0], d1 = r.d_ticks[1], d2 = r.d_ticks[2], d3 = r.d_ticks[3];
            const uint64_t d4 = nr ? nr->d4_tick : 0, d5 = nr ? nr->d5_tick : 0;
            const uint64_t d6 = r.d_ticks[4], d7 = r.d_ticks[5], d8 = r.d_ticks[6], d9 = r.d_ticks[7];
            uint64_t v[7] {};
            if (nr) for (int k = 0; k < 7; ++k) v[k] = nr->v_ticks[k];

            auto d_ms = [&](uint64_t a, uint64_t b) -> double {
                if (!dfreq || !a || b < a) return 0.0;
                return (double)(b - a) * 1000.0 / (double)dfreq;
            };
            auto v_ms = [&](uint64_t a, uint64_t b) -> double {
                if (vperiod <= 0.0 || !a || b < a) return 0.0;
                return (double)(b - a) * vperiod / 1000000.0;
            };

            std::fprintf(fg,
                "{\"frame_id\":%llu,\"pending_seq\":%llu,\"returned_seq\":%llu,"
                "\"window_id\":%u,\"arm_reuse\":%u,\"slot_index\":%u,\"copy_slot\":%u,"
                "\"bridge_d3d_valid\":%u,\"nrvk_d3d_valid\":%u,\"vk_valid\":%u,"
                "\"d3d_freq_hz\":%llu,\"vk_timestamp_period_ns\":%.6f,\"vk_valid_bits\":%u,"
                "\"d0_tick\":%llu,\"d1_tick\":%llu,\"d2_tick\":%llu,\"d3_tick\":%llu,"
                "\"d4_tick\":%llu,\"d5_tick\":%llu,\"d6_tick\":%llu,\"d7_tick\":%llu,"
                "\"d8_tick\":%llu,\"d9_tick\":%llu,"
                "\"v0_tick\":%llu,\"v1_tick\":%llu,\"v2_tick\":%llu,\"v3_tick\":%llu,"
                "\"v4_tick\":%llu,\"v5_tick\":%llu,\"v6_tick\":%llu,"
                "\"d3d_color_copy_gpu_ms\":%.6f,\"d3d_prepare_gpu_ms\":%.6f,"
                "\"d3d_prep_list_gpu_ms\":%.6f,\"d3d_prep_to_nrvk_copy_gap_gpu_ms\":%.6f,"
                "\"d3d_nrvk_copy_gpu_ms\":%.6f,\"d3d_nrvk_copy_to_composite_gap_gpu_ms\":%.6f,"
                "\"d3d_composite_gpu_ms\":%.6f,\"d3d_composite_to_sr_gap_gpu_ms\":%.6f,"
                "\"d3d_deferred_sr_gpu_ms\":%.6f,\"d3d_total_span_gpu_ms\":%.6f,"
                "\"vk_acquire_gpu_ms\":%.6f,\"vk_history_controls_gpu_ms\":%.6f,"
                "\"vk_gather_gpu_ms\":%.6f,\"vk_network_266_gpu_ms\":%.6f,"
                "\"vk_scatter_gpu_ms\":%.6f,\"vk_release_gpu_ms\":%.6f,"
                "\"vk_total_cb_gpu_ms\":%.6f}\n",
                (unsigned long long)r.frame_id,
                (unsigned long long)r.pending_seq,
                (unsigned long long)(nr ? nr->returned_seq : 0),
                r.window_id, r.arm_reuse,
                nr ? nr->slot_index : 0u, nr ? nr->copy_slot : 0u,
                r.d3d_valid, nd3d_ok, nvk_ok,
                (unsigned long long)dfreq, vperiod, vbits,
                (unsigned long long)d0, (unsigned long long)d1, (unsigned long long)d2, (unsigned long long)d3,
                (unsigned long long)d4, (unsigned long long)d5, (unsigned long long)d6, (unsigned long long)d7,
                (unsigned long long)d8, (unsigned long long)d9,
                (unsigned long long)v[0], (unsigned long long)v[1], (unsigned long long)v[2], (unsigned long long)v[3],
                (unsigned long long)v[4], (unsigned long long)v[5], (unsigned long long)v[6],
                d_ms(d0, d1), d_ms(d2, d3),
                d_ms(d0, d3), d_ms(d3, d4),
                d_ms(d4, d5), d_ms(d5, d6),
                d_ms(d6, d7), d_ms(d7, d8),
                d_ms(d8, d9), d_ms(d0, d9),
                v_ms(v[0], v[1]), v_ms(v[1], v[2]),
                v_ms(v[2], v[3]), v_ms(v[3], v[4]),
                v_ms(v[4], v[5]), v_ms(v[5], v[6]),
                v_ms(v[0], v[6]));
        }
        std::fclose(fg);
    }
}
#endif

void deferred_diag_poll() {
#ifdef NRB_LIVE_NR_TEST
    HMODULE module = nullptr;
    wchar_t mod_path[MAX_PATH]{};
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
            GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCWSTR>(&record_deferred_identity), &module) ||
        !GetModuleFileNameW(module, mod_path, MAX_PATH)) return;
    std::wstring base_dir = mod_path;
    const auto slash = base_dir.find_last_of(L"\\/");
    if (slash == std::wstring::npos) return;
    base_dir.resize(slash);
    const std::wstring ctl_path = base_dir + L"\\dsh_diag_control.ini";
    const DWORD attr = GetFileAttributesW(ctl_path.c_str());
    if (attr == INVALID_FILE_ATTRIBUTES || (attr & FILE_ATTRIBUTE_DIRECTORY)) return;

    const uint32_t want_mode = (uint32_t)GetPrivateProfileIntW(L"diag", L"mode", 0, ctl_path.c_str());
    const uint32_t want_win = (uint32_t)GetPrivateProfileIntW(L"diag", L"window_id", 0, ctl_path.c_str());
    const uint32_t want_arm = (uint32_t)GetPrivateProfileIntW(L"diag", L"arm_reuse", 1, ctl_path.c_str());
    const uint32_t want_late_latch = (uint32_t)GetPrivateProfileIntW(L"diag", L"late_latch", 1, ctl_path.c_str());
    const uint32_t want_flush = (uint32_t)GetPrivateProfileIntW(L"diag", L"flush_seq", 0, ctl_path.c_str());
    wchar_t out_buf[MAX_PATH]{};
    GetPrivateProfileStringW(L"diag", L"out_dir", L"", out_buf, MAX_PATH, ctl_path.c_str());
    std::wstring out_dir = out_buf[0] ? std::wstring(out_buf) : base_dir;

    g_diag_active.store(true, std::memory_order_relaxed);
    hdr_set_composite_heap_reuse(want_arm != 0);
    g_diag_arm_reuse.store(want_arm, std::memory_order_relaxed);
    g_diag_late_latch.store(want_late_latch, std::memory_order_relaxed);
    g_diag_window_id.store(want_win, std::memory_order_relaxed);
    g_diag_mode.store(want_mode, std::memory_order_relaxed);
    nrvk_set_diag_state(want_mode, want_win, want_arm);

    std::lock_guard guard(nr_timing_mutex);
    g_diag_out_dir = out_dir;
    if (want_flush > g_last_flush_seq) {
        flush_diag_files_locked(out_dir);
        g_last_flush_seq = want_flush;
    }

    const NrvkDiagFrameRecordMirror* nrecs = nullptr;
    uint32_t ncount = 0;
    uint64_t nd3d_unavail = 0, nvk_unavail = 0, noverflow = 0;
    nrvk_get_records(&nrecs, &ncount, &nd3d_unavail, &nvk_unavail, &noverflow);
    uint64_t nrvk_d3d_freq = 0;
    double vk_period = 0.0;
    uint32_t vk_bits = 0;
    nrvk_get_device_info(&nrvk_d3d_freq, &vk_period, &vk_bits);
    const auto comp_stats = hdr_composite_heap_diag_stats();
    const auto pool_stats = hdr_resource_pool_session().stats();
    const uint64_t qf = ensure_qpc_freq();
    const uint64_t df = g_d3d_freq_hz ? g_d3d_freq_hz : nrvk_d3d_freq;

    const std::wstring status_path = out_dir + L"\\DIAG-LIVE-STATUS.json";
    FILE* fs = nullptr;
    if (_wfopen_s(&fs, status_path.c_str(), L"wb") == 0 && fs) {
        std::fprintf(fs,
            "{\"active\":true,\"mode\":%u,\"window_id\":%u,\"arm_reuse\":%u,\"late_latch\":%u,"
            "\"last_flush_seq\":%u,\"nr_completed\":%llu,"
            "\"bridge_records\":%u,\"nrvk_records\":%u,"
            "\"pooled_private_composite_count\":%llu,\"unique_composite_heap_creates\":%llu,"
            "\"pool_acquires\":%llu,\"pool_hits\":%llu,\"pool_creates\":%llu,"
            "\"pool_busy_fallbacks\":%llu,\"pool_quarantines\":%llu,"
            "\"bridge_d3d_unavail\":%llu,\"nrvk_d3d_unavail\":%llu,\"nrvk_vk_unavail\":%llu,"
            "\"bridge_overflow\":%llu,\"nrvk_overflow\":%llu,"
            "\"qpc_freq_hz\":%llu,\"d3d_freq_hz\":%llu,"
            "\"vk_timestamp_period_ns\":%.6f,\"vk_valid_bits\":%u}\n",
            want_mode, want_win, want_arm, want_late_latch,
            g_last_flush_seq, (unsigned long long)nr_completed.load(std::memory_order_relaxed),
            (unsigned)bridge_diag_records().size(), ncount,
            (unsigned long long)comp_stats.pooled_private_composite_count,
            (unsigned long long)comp_stats.unique_composite_heap_creates,
            (unsigned long long)pool_stats.acquires,
            (unsigned long long)pool_stats.hits,
            (unsigned long long)pool_stats.creates,
            (unsigned long long)pool_stats.busy_fallbacks,
            (unsigned long long)pool_stats.quarantines,
            (unsigned long long)g_bridge_d3d_unavail,
            (unsigned long long)nd3d_unavail,
            (unsigned long long)nvk_unavail,
            (unsigned long long)g_bridge_diag_overflow,
            (unsigned long long)noverflow,
            (unsigned long long)qf, (unsigned long long)df,
            vk_period, vk_bits);
        std::fclose(fs);
    }
#endif
}

void deferred_diag_shutdown() {
#ifdef NRB_LIVE_NR_TEST
    if (!g_diag_active.load(std::memory_order_relaxed)) return;
    std::lock_guard guard(nr_timing_mutex);
    if (!g_diag_out_dir.empty()) {
        flush_diag_files_locked(g_diag_out_dir);
    }
#endif
}
}


NRB_API int NRB_SetHdrResourcePoolEnabled(int enabled) {
    return nrb::deferred_hdr_set_resource_pool_enabled(enabled!=0)?1:0;
}
NRB_API int NRB_GetHdrResourcePoolStats(NRB_HdrResourcePoolStats* stats) {
    if(!stats || stats->abi_size!=sizeof(*stats)) return 0;
#ifdef NRB_LIVE_NR_TEST
    nrb::HdrResourcePoolStats value;
    if(!nrb::deferred_hdr_resource_pool_stats(value)) return 0;
    NRB_HdrResourcePoolStats out{};out.abi_size=sizeof(out);
    out.enabled=value.enabled;out.capacity=nrb::HdrProxyResourcePool::capacity;
    out.shutdown=value.shutdown;
    out.acquires=value.acquires;
    out.hits=value.hits;
    out.creates=value.creates;
    out.disabled_fallbacks=value.disabled_fallbacks;
    out.busy_fallbacks=value.busy_fallbacks;
    out.creation_failures=value.creation_failures;
    out.retire_returns=value.retire_returns;
    out.unsubmitted_returns=value.unsubmitted_returns;
    out.quarantines=value.quarantines;
    out.evictions=value.evictions;
    out.generation=value.generation;
    out.resident_slots=value.resident_slots;
    out.free_slots=value.free_slots;
    out.leased_slots=value.leased_slots;
    out.quarantined_slots=value.quarantined_slots;
    out.texture_bytes=value.texture_bytes;
    out.peak_texture_bytes=value.peak_texture_bytes;
    out.descriptor_bytes_estimate=value.descriptor_bytes_estimate;
    out.max_texture_bytes=nrb::HdrProxyResourcePool::max_texture_bytes;
    *stats=out;return 1;
#else
    return 0;
#endif
}
