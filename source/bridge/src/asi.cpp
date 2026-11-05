#include "nr_bridge.h"
#include "nr_gpu_handoff.h"
#include "nr_core.h"
#include "nr_diag.h"
#include "nr_python.h"
#include "nr_sync_probe.h"
#include "periodic_flash_native_diag.h"
#include <algorithm>
#ifdef NRB_XESS_IDENTITY_TEST
#include "xess_identity.h"
#endif
#ifdef NRB_DEFERRED_IDENTITY_TEST
#include "deferred_identity.h"
#endif
#ifdef NRB_TAIL_ORDER_PROBE
#include "nr_tail_probe.h"
#endif
#ifdef NRB_ONE_FRAME_COLOR_READBACK
#include "nr_one_frame_readback.h"
#endif
#include <nvsdk_ngx.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <windows.h>
#include <detours/detours.h>
#include <intrin.h>
#include <atomic>
#include <cstddef>
#include <mutex>
#include <cstring>
#include <vector>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdarg>
#include <string>
#include <filesystem>

namespace {
using Evaluate = NVSDK_NGX_Result (__cdecl *)(ID3D12GraphicsCommandList*,
    const NVSDK_NGX_Handle*, NVSDK_NGX_Parameter*, PFN_NVSDK_NGX_ProgressCallback);
Evaluate original = nullptr;
// Streamline's public slEvaluateFeature(feature=0) is the authoritative DLSS
// tag when the NGX call is nested synchronously from sl.common.dll.
using SLEvaluate = int32_t (__cdecl *)(uint32_t,const void*,const void**,
                                       uint32_t,void*);
SLEvaluate original_sl_evaluate=nullptr;
thread_local bool in_sl_dlss_evaluate=false;
// These tags are set only while the game's public XeSS/FSR entry is executing.
// OptiScaler's nested NGX call can then be attributed without relying on
// optional parameter names or on whichever upscaler was selected previously.
thread_local NRB_Route in_upscaler_api=NRB_ROUTE_UNKNOWN;
using XeSSExecute = int32_t (__cdecl *)(void*,ID3D12GraphicsCommandList*,const void*);
using FSR3Dispatch = int32_t (__cdecl *)(void*,const void*);
XeSSExecute original_xess_execute=nullptr;
FSR3Dispatch original_fsr3_dispatch=nullptr;
#ifdef NRB_TAIL_ORDER_PROBE
void schedule_list_hooks(ID3D12GraphicsCommandList* list);
#endif
// Prefix of the public FFX FSR3 dispatch ABI. This diagnostic reads the
// descriptor without changing the game's resources or dispatch arguments.
// The audited OptiScaler source used to build this ASI omits the FFX headers.
struct Fsr3ResourceProbe {
    void* resource;
    uint32_t type,format,width,height,depth,mip_count,flags,usage,state;
    wchar_t name[64];
};
static_assert(sizeof(Fsr3ResourceProbe)==176);
struct Fsr3DispatchProbe {
    void* command_list;
    Fsr3ResourceProbe resources[10];
    float jitter_x,jitter_y,motion_scale_x,motion_scale_y;
    uint32_t render_width,render_height;
};
static_assert(offsetof(Fsr3DispatchProbe,resources)==8);
static_assert(offsetof(Fsr3DispatchProbe,render_width)==1784);
HMODULE host = nullptr;
std::atomic<int> init_state{0};
std::mutex state_mutex;
NRB_Processor processor{};
ID3D12CommandQueue* queue = nullptr;
ID3D12Fence* completion = nullptr;
HANDLE completion_event = nullptr;
uint64_t completion_value = 0;
uint64_t frame_id = 0;
uint64_t last_processed_id = 0, control_epoch = 0, last_processed_epoch = 0;
uint32_t last_width = 0, last_height = 0;
NRB_Route last_route = NRB_ROUTE_UNKNOWN;
NRB_Controls controls{sizeof(NRB_Controls), 0, 360, 0, NRB_HISTORY_FUSED,
    1, 0, 0, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f};
NRB_Result borrowed{};
// A Boolean from a caller is not source-submission evidence. The signed host
// exposes no current-frame producer callback, so this remains fail closed.
std::atomic<bool> source_submission_proven{false};
std::atomic<NRB_Status> last_status{NRB_BYPASS_DISABLED};
std::atomic<uint64_t> counts[4]{};
std::atomic<uint64_t> processed_count{0};
std::atomic<double> last_process_ms{0.0}, average_process_ms{0.0};
std::array<double,32> process_samples{};
size_t process_sample_count=0, process_sample_cursor=0;
double process_sample_sum=0.0;
thread_local bool inside = false;
#ifdef NRB_XESS_IDENTITY_TEST
std::atomic<unsigned> identity_relay_logs{0};
#endif
std::mutex log_mutex, queue_mutex, hook_mutex;
nrb::DiagnosticGate diagnostic_gate;
void log_line(const char* message);
nrb::PythonProcessor python_processor{log_line};
bool runtime_assets_available=false;
bool built_in_processor=false;
bool processor_failed=false;
// [vk102 integration] Required nrvk component. No built-in/Python backend may
// process a frame while this component is absent or still mounting.
#define NRVK_REQUIRED 1
static char g_manifest_dir[MAX_PATH]="nr-runtime/manifest";
static char g_status_json[MAX_PATH]="component_status.json";
static char g_panel_ini[MAX_PATH]="nr_panel.ini";
static char g_panel_state[MAX_PATH]="panel_state.json";
#define NRVK_MANIFEST_DIR g_manifest_dir
#define NRVK_STATUS_JSON_PATH g_status_json
std::atomic<ID3D12Device*> observed_d3d12_device{nullptr};
HMODULE nrvk_bridge_module=nullptr;
std::atomic<bool> nrvk_bridge_mounted{false};
std::atomic<bool> nrvk_mount_attempted{false};
std::atomic<bool> nrvk_mount_failed{false};
// ---------------------------------------------------------------- [panel]
// Workspace control panel. work\vk102-small-310-9-16-20261019\panel\
// nr_control_panel.py writes nr_panel.ini; this block polls it off the render
// thread and folds the values into the NRB_Controls both frame paths already
// copy, so appearance controls need no new ABI. Only an explicit backend
// switch stops and re-mounts the nrvk component.
#define NR_PANEL_INI_PATH g_panel_ini
namespace panel {
struct Values {
    int enabled=1, style=0, auto_mask=1;
    // AMD's own defaults (nr_graph.cpp:2362-2367): --local-tone-strength 1,
    // --skin-structure-strength -1 (=-1 means "fall back to local_structure"). The shipped
    // nr_panel.ini carries the same numbers; these only apply when a key is missing, and a
    // skin of 0 is *not* equivalent to -1: it is a gradient value, worth a measured 40.94 dB
    // against the reference (07-进展.md, "配置其实有三个").
    float intensity=1.0f, local_structure=1.0f, local_tone=1.0f,
          skin_structure=-1.0f, display_strength=1.0f;
    std::string backend="small";
};
struct NrvkConfig {
    uint32_t abi_size;
    const char* manifest_dir;
    const char* cache_dir;
    const char* status_json_path;
    uint32_t output_ring_slots;
    uint32_t log_verbose;
};
static_assert(sizeof(NrvkConfig)==40,"NRVK_Config layout drift");
using RegFn=int(*)(const NRB_Processor*);
using AttachFn=int(*)(ID3D12Device*,const void*,RegFn);
using ShutdownFn=int(*)(void);

std::mutex mutex;
Values applied;
bool applied_valid=false;
// Published lock-free for the render thread: the outer Evaluate path already
// holds state_mutex when it reaches the frame build, so the frame path must
// never take that mutex again.
std::atomic<bool> live_valid{false};
std::atomic<uint32_t> live_enabled{1}, live_style{0}, live_auto_mask{1};
std::atomic<float> live_display{1.0f}, live_intensity{1.0f}, live_local_tone{0.0f},
    live_local_structure{1.0f}, live_skin{0.0f};
// The component mounted from the game directory is authoritative for the boot
// plan: switching back to it must re-attach that exact DLL, never a copy from
// a package bin directory.
std::string boot_key="small";
std::string boot_dll;
HMODULE boot_module=nullptr;
std::string backend_key="small";
std::string backend_manifest=NRVK_MANIFEST_DIR;
std::string backend_dll;
HMODULE backend_module=nullptr;
std::string unresolved_backend;
// A component can attach and still reject every frame (device init or contract
// mismatch). The first frame it is given decides: a hard process error before
// any success rolls the swap back and refuses the key for this session.
std::atomic<int> probe_state{0};   // 0 idle, 1 awaiting the first frame, 3 broken
// The component shallow-copies NRVK_Config and dereferences cfg.manifest_dir on
// the first frame, not at attach time. The buffer must therefore outlive the
// attach call; a temporary std::string here reads as a dangling pointer later.
char live_manifest[MAX_PATH]{};
std::string unusable;
std::string verify_key, verify_manifest, verify_dll;
HMODULE verify_module=nullptr;
std::string verify_prev_key, verify_prev_manifest, verify_prev_dll;
HMODULE verify_prev_module=nullptr;

// -------------------------------------------------------------- panel state
// The panel runs in another process: publish what only the bridge knows.
#define NR_PANEL_STATE_PATH g_panel_state
std::mutex event_mutex;
std::string last_event;
uint64_t last_event_ms=0;
void set_event(const char* fmt,...) {
    char text[192]{};
    va_list args; va_start(args,fmt);
    vsnprintf_s(text,sizeof text,_TRUNCATE,fmt,args);
    va_end(args);
    std::lock_guard guard(event_mutex);
    last_event=text;
    last_event_ms=GetTickCount64();
}
void write_state() {
    NRB_Controls c{};
    {std::lock_guard guard(state_mutex); c=controls;}
    NRB_StageTimes stage{};
    stage.abi_size=sizeof(stage);
#ifdef NRB_DEFERRED_IDENTITY_TEST
    nrb::deferred_nr_stage_times(&stage);
#endif
    std::string event;
    uint64_t event_ms=0;
    {std::lock_guard guard(event_mutex); event=last_event; event_ms=last_event_ms;}
    NRB_GpuHandoffStats handoff{};handoff.abi_size=sizeof(handoff);NRB_GetGpuHandoffStats(&handoff);
    const double total=stage.prep_average_ms+stage.host_average_ms+
        stage.composite_submit_average_ms+stage.tail_average_ms;
    const double last=stage.prep_last_ms+stage.host_last_ms+
        stage.composite_submit_last_ms+stage.tail_last_ms;
    FILE* file=nullptr;
    if(fopen_s(&file,NR_PANEL_STATE_PATH,"w")==0&&file) {
        fprintf(file,
            "{\"backend\":\"%s\",\"mounted\":%d,\"processor\":%d,\"switch\":%d,"
            "\"processed\":%llu,\"event\":\"%s\",\"event_ms\":%llu,"
            "\"applied\":{\"enabled\":%u,\"style\":%u,\"intensity\":%.4f,"
            "\"structure\":%.4f,\"tone\":%.4f,\"skin\":%.4f,\"mask\":%u,"
            "\"display\":%.4f},\"stage\":{\"frames\":%llu,\"prep\":%.3f,"
            "\"host\":%.3f,\"composite\":%.3f,\"tail\":%.3f,\"total\":%.3f,"
            "\"last\":%.3f},\"handoff\":{\"bypasses\":%llu,\"cpu_waits\":%llu,\"motion_waits\":%llu}}\n",
            backend_key.c_str(),int(nrvk_bridge_mounted.load(std::memory_order_relaxed)),
            int(processor.process!=nullptr),int(probe_state.load(std::memory_order_relaxed)),
            (unsigned long long)NRB_ProcessedCount(),event.c_str(),
            (unsigned long long)event_ms,
            c.enabled,c.style,c.model_intensity,c.local_structure,c.local_tone,
            c.skin_structure,c.auto_mask,c.display_strength,
            (unsigned long long)stage.frames,stage.prep_average_ms,
            stage.host_average_ms,stage.composite_submit_average_ms,
            stage.tail_average_ms,total,last,
            (unsigned long long)handoff.prepared_bypasses,(unsigned long long)handoff.prepared_cpu_waits,(unsigned long long)handoff.motion_readback_waits);
        fclose(file);
    }
}
bool same_controls(const Values& a,const Values& b) {
    return a.enabled==b.enabled && a.style==b.style && a.auto_mask==b.auto_mask &&
        a.intensity==b.intensity && a.local_structure==b.local_structure &&
        a.local_tone==b.local_tone && a.skin_structure==b.skin_structure &&
        a.display_strength==b.display_strength;
}
// Reads one key from the named INI section without allocating a parser; the
// file is a few hundred bytes and is polled four times a second.
std::string lookup(const std::string& text,const char* section,const char* key) {
    const std::string want=std::string("[")+section+"]";
    std::string current;
    size_t pos=0;
    while(pos<text.size()) {
        size_t end=text.find('\n',pos);
        if(end==std::string::npos) end=text.size();
        std::string line=text.substr(pos,end-pos);
        pos=end+1;
        while(!line.empty()&&(line.back()=='\r'||line.back()==' '||line.back()=='\t'))
            line.pop_back();
        const size_t lead=line.find_first_not_of(" \t");
        if(lead==std::string::npos) continue;
        line.erase(0,lead);
        if(line.empty()||line.front()==';'||line.front()=='#') continue;
        if(line.front()=='['&&line.back()==']') { current=line; continue; }
        const size_t eq=line.find('=');
        if(eq==std::string::npos) continue;
        std::string name=line.substr(0,eq);
        while(!name.empty()&&(name.back()==' '||name.back()=='\t')) name.pop_back();
        if(current!=want||name!=key) continue;
        std::string value=line.substr(eq+1);
        const size_t vs=value.find_first_not_of(" \t");
        return vs==std::string::npos?std::string():value.substr(vs);
    }
    return std::string();
}
bool read_ini(std::string& text) {
    HANDLE h=CreateFileA(NR_PANEL_INI_PATH,GENERIC_READ,
        FILE_SHARE_READ|FILE_SHARE_WRITE|FILE_SHARE_DELETE,nullptr,OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,nullptr);
    if(h==INVALID_HANDLE_VALUE) return false;
    char buffer[8193]{};
    DWORD got=0;
    const bool ok=ReadFile(h,buffer,sizeof(buffer)-1,&got,nullptr)!=0;
    CloseHandle(h);
    if(!ok) return false;
    text.assign(buffer,got);
    // An editor that saves with a BOM must not silently disable the panel.
    if(text.size()>=3 && static_cast<unsigned char>(text[0])==0xEF &&
       static_cast<unsigned char>(text[1])==0xBB && static_cast<unsigned char>(text[2])==0xBF)
        text.erase(0,3);
    return true;
}
bool parse(Values& v,std::string& text) {
    if(!read_ini(text)) return false;
    auto number=[&](const char* key,float fallback) {
        const std::string s=lookup(text,"NR",key);
        return s.empty()?fallback:static_cast<float>(atof(s.c_str()));
    };
    auto integer=[&](const char* key,int fallback) {
        const std::string s=lookup(text,"NR",key);
        return s.empty()?fallback:atoi(s.c_str());
    };
    v.enabled=integer("enabled",v.enabled)?1:0;
    v.style=integer("style",v.style);
    v.auto_mask=integer("auto_mask",v.auto_mask)?1:0;
    v.intensity=number("intensity",v.intensity);
    v.local_structure=number("local_structure",v.local_structure);
    v.local_tone=number("local_tone",v.local_tone);
    v.skin_structure=number("skin_structure",v.skin_structure);
    v.display_strength=number("display_strength",v.display_strength);
    const std::string backend=lookup(text,"NR","backend");
    v.backend=backend=="102"?"102":"small";
    return true;
}
void log_values(const char* what,const Values& v) {
    char line[320];
    sprintf_s(line,"[panel] %s enabled=%d style=%d auto_mask=%d intensity=%.3f "
        "local_structure=%.3f local_tone=%.3f skin=%.3f display=%.3f backend=%s\r\n",
        what,v.enabled,v.style,v.auto_mask,v.intensity,v.local_structure,
        v.local_tone,v.skin_structure,v.display_strength,v.backend.c_str());
    log_line(line);
}
// Appearance controls only. Nothing here bumps control_epoch: an epoch change
// forces reset_history, which would drop temporal history four times a second
// while the panel is being polled. Only an enable toggle resets history.
bool apply_controls(const Values& v) {
    bool accepted=false;
    {
        std::lock_guard guard(state_mutex);
        NRB_Controls c=controls;
        c.enabled=v.enabled?1u:0u;
        c.style=static_cast<uint32_t>(v.style);
        c.auto_mask=v.auto_mask?1u:0u;
        c.display_strength=v.display_strength;
        c.model_intensity=v.intensity;
        c.local_tone=v.local_tone;
        c.local_structure=v.local_structure;
        c.skin_structure=v.skin_structure;
        if(nrb::validate_controls(c)) {
            if(c.enabled!=controls.enabled) ++control_epoch;
            controls=c;
            live_enabled.store(c.enabled,std::memory_order_relaxed);
            live_style.store(c.style,std::memory_order_relaxed);
            live_auto_mask.store(c.auto_mask,std::memory_order_relaxed);
            live_display.store(c.display_strength,std::memory_order_relaxed);
            live_intensity.store(c.model_intensity,std::memory_order_relaxed);
            live_local_tone.store(c.local_tone,std::memory_order_relaxed);
            live_local_structure.store(c.local_structure,std::memory_order_relaxed);
            live_skin.store(c.skin_structure,std::memory_order_relaxed);
            live_valid.store(true,std::memory_order_release);
            accepted=true;
        }
    }
    return accepted;
}
// [Backends] <key> = <manifest dir>; the component beside that package is used.
bool resolve_backend(const std::string& key,const std::string& text,
                     std::string& manifest,std::string& dll) {
    manifest=lookup(text,"Backends",key.c_str());
    if(manifest.empty()) return false;
    std::string base=manifest;
    const size_t slash=base.find_last_of("/\\");
    if(slash==std::string::npos) return false;
    base.resize(slash);
    const char* candidates[]={"/bin/nrvk_bridge.dll","/nrvk_bridge.dll"};
    for(const char* suffix:candidates) {
        dll=base+suffix;
        if(GetFileAttributesA(dll.c_str())!=INVALID_FILE_ATTRIBUTES) return true;
    }
    return false;
}
// Returns true only when the requested backend is the one now attached. A
// component that still owes a frame reports rc=2; that is a retry, not a
// result, so the caller keeps the request pending instead of latching it.
bool swap_backend(const std::string& key,const std::string& text) {
    if(!nrvk_bridge_mounted.load(std::memory_order_acquire)||!backend_module) return false;
    if(!unusable.empty()&&key==unusable) {
        static std::string complained;
        if(complained!=key) {
            complained=key;
            char line[300];
            sprintf_s(line,"[panel] backend=%s is marked unusable this session; staying on %s\r\n",
                key.c_str(),backend_key.c_str());
            log_line(line);
        }
        return false;
    }
    std::string manifest,dll;
    HMODULE target=nullptr;
    if(key==boot_key) {
        if(!boot_module) return false;
        manifest=NRVK_MANIFEST_DIR;
        dll=boot_dll;
        target=boot_module;
    } else {
        if(!resolve_backend(key,text,manifest,dll)) {
            if(unresolved_backend!=key) {
                unresolved_backend=key;
                char line[360];
                sprintf_s(line,"[panel] backend=%s has no manifest/component; keeping %s\r\n",
                    key.c_str(),backend_key.c_str());
                log_line(line);
            }
            return false;
        }
        target=LoadLibraryExA(dll.c_str(),nullptr,LOAD_WITH_ALTERED_SEARCH_PATH);
        if(!target) {
            char line[360];
            sprintf_s(line,"[panel] backend=%s load failed err=%lu; keeping %s\r\n",
                key.c_str(),GetLastError(),backend_key.c_str());
            log_line(line);
            return false;
        }
    }
    ID3D12Device* device=observed_d3d12_device.load(std::memory_order_acquire);
    if(!device) return false;
    // Preflight the target with its own ComponentSelfTest before touching the
    // running component: it runs the same checked_plan() the attach would run,
    // so a component that cannot validate its manifest here never gets the
    // chance to replace a working backend.
    {
        auto self_test=(int(*)(const NrvkConfig*,void*))GetProcAddress(target,"NRVK_ComponentSelfTest");
        if(!self_test) {
            log_line("[panel] target component has no self test; keeping current backend\r\n");
            return false;
        }
        NrvkConfig probe{sizeof(NrvkConfig),manifest.c_str(),nullptr,NRVK_STATUS_JSON_PATH,3,1};
        const int pre=self_test(&probe,nullptr);
        if(pre!=0) {
            char line[400];
            sprintf_s(line,"[panel] backend=%s preflight rc=%d acp=%u manifest=%s; keeping %s\r\n",
                key.c_str(),pre,GetACP(),manifest.c_str(),backend_key.c_str());
            log_line(line);
            set_event("%s preflight rc=%d",key.c_str(),pre);
            return false;
        }
    }
    auto attach_target=(AttachFn)GetProcAddress(target,"NRVK_AttachToBridge");
    auto attach_old=(AttachFn)GetProcAddress(backend_module,"NRVK_AttachToBridge");
    auto stop_old=(ShutdownFn)GetProcAddress(backend_module,"NRVK_Shutdown");
    if(!attach_target||!attach_old||!stop_old) {
        log_line("[panel] nrvk exports missing; keeping current backend\r\n");
        return false;
    }
    // A deferred frame may be inside the component right now. Stop recording,
    // drain the pipeline, and only then tear the component down: destroying it
    // under a live process call is an instant crash.
#ifdef NRB_DEFERRED_IDENTITY_TEST
    nrb::deferred_identity_set_suspended(true);
    bool drained=false;
    for(int wait=0;wait<40&&!drained;++wait) {
        if(nrb::deferred_identity_pending()==0) drained=true; else Sleep(5);
    }
    if(!drained) {
        nrb::deferred_identity_set_suspended(false);
        static std::string drain_busy;
        if(drain_busy!=key) {
            drain_busy=key;
            char line[320];
            sprintf_s(line,"[panel] backend=%s drain timeout (pipeline busy); keeping %s\r\n",
                key.c_str(),backend_key.c_str());
            log_line(line);
        }
        return false;
    }
#endif
    const int stopped=stop_old();
    if(stopped==2) {   // a frame still owns the borrowed output; retry later
#ifdef NRB_DEFERRED_IDENTITY_TEST
        nrb::deferred_identity_set_suspended(false);
#endif
        static std::string deferred_busy;
        if(deferred_busy!=key) {
            deferred_busy=key;
            char line[240];
            sprintf_s(line,"[panel] backend=%s deferred: component busy; retrying\r\n",
                key.c_str());
            set_event("%s deferred (component busy)",key.c_str());
            log_line(line);
        }
        return false;
    }
    if(stopped!=0) {
#ifdef NRB_DEFERRED_IDENTITY_TEST
        nrb::deferred_identity_set_suspended(false);
#endif
        char line[220];
        sprintf_s(line,"[panel] nrvk shutdown rc=%d; keeping %s\r\n",stopped,backend_key.c_str());
        log_line(line);
        return false;
    }
#ifdef NRB_DEFERRED_IDENTITY_TEST
    // The component that owned these records is gone, so they can never
    // retire; leaving them would keep the 8-slot window full and silently
    // stop all later frames. Abandon them. The tail proof is a property of the
    // game's command lists, not of the component, so it must NOT be reset here:
    // clearing it makes identity_source_state_proven false and the deferred
    // path stops recording entirely.
    nrb::deferred_identity_abandon_all();
    // A pending from the old component latches the helper failure while it is
    // being retired; that latch would skip NR on every later frame even though
    // the new component is mounted and healthy.
    nrb::deferred_identity_clear_failure();
#endif
    nrvk_bridge_mounted.store(false,std::memory_order_release);
    strcpy_s(live_manifest,manifest.c_str());
    NrvkConfig cfg{sizeof(NrvkConfig),live_manifest,nullptr,NRVK_STATUS_JSON_PATH,3,1};
    const int attached=attach_target(device,&cfg,&NRB_RegisterProcessor);
    char line[420];
    sprintf_s(line,"[panel] backend %s -> %s attach rc=%d dll=%s\r\n",
        backend_key.c_str(),key.c_str(),attached,dll.c_str());
    log_line(line);
    if(attached==0) {
        verify_prev_key=backend_key;
        verify_prev_manifest=backend_manifest;
        verify_prev_dll=backend_dll;
        verify_prev_module=backend_module;
        verify_key=key;
        verify_manifest=manifest;
        verify_dll=dll;
        verify_module=target;
        backend_key=key;
        backend_manifest=manifest;
        backend_dll=dll;
        backend_module=target;
        nrvk_bridge_module=target;   // the atexit shutdown must reach the live component
        nrvk_mount_failed.store(false,std::memory_order_release);
        nrvk_bridge_mounted.store(true,std::memory_order_release);
        {std::lock_guard guard(state_mutex); ++control_epoch;}  // no history across plans
        probe_state.store(1,std::memory_order_release);
        set_event("%s -> %s ok",verify_prev_key.c_str(),verify_key.c_str());
#ifdef NRB_DEFERRED_IDENTITY_TEST
        nrb::deferred_identity_set_suspended(false);
#endif
        return true;
    }
    NrvkConfig restore{sizeof(NrvkConfig),backend_manifest.c_str(),nullptr,
                       NRVK_STATUS_JSON_PATH,3,1};
    if(attach_old(device,&restore,&NRB_RegisterProcessor)==0) {
        nrvk_bridge_mounted.store(true,std::memory_order_release);
        char back[240];
        sprintf_s(back,"[panel] backend switch failed; restored %s\r\n",backend_key.c_str());
        log_line(back);
        set_event("switch failed; restored %s",backend_key.c_str());
    } else {
        nrvk_mount_failed.store(true,std::memory_order_release);
        log_line("[nrvk] backend rollback failed; frames fail closed (no fallback)\r\n");
    }
#ifdef NRB_DEFERRED_IDENTITY_TEST
    nrb::deferred_identity_set_suspended(false);
#endif
    return false;
}
// Put the previous component back after a target failed its first frame.
// Returns false when the target is still busy or the previous one did not
// attach; the caller retries on the next poll.
bool restore_previous() {
    auto stop_target=(ShutdownFn)GetProcAddress(verify_module,"NRVK_Shutdown");
    if(!stop_target) return false;
    if(stop_target()!=0) return false;
    nrvk_bridge_mounted.store(false,std::memory_order_release);
    auto attach_prev=(AttachFn)GetProcAddress(verify_prev_module,"NRVK_AttachToBridge");
    ID3D12Device* device=observed_d3d12_device.load(std::memory_order_acquire);
    if(!attach_prev||!device) return false;
    strcpy_s(live_manifest,verify_prev_manifest.c_str());
    NrvkConfig cfg{sizeof(NrvkConfig),live_manifest,nullptr,
                   NRVK_STATUS_JSON_PATH,3,1};
    if(attach_prev(device,&cfg,&NRB_RegisterProcessor)!=0) return false;
    backend_key=verify_prev_key;
    backend_manifest=verify_prev_manifest;
    backend_dll=verify_prev_dll;
    backend_module=verify_prev_module;
    nrvk_bridge_module=verify_prev_module;
    nrvk_bridge_mounted.store(true,std::memory_order_release);
    nrvk_mount_failed.store(false,std::memory_order_release);
    {std::lock_guard guard(state_mutex); ++control_epoch;}
    return true;
}
void verify_swap() {
    if(probe_state.load(std::memory_order_acquire)!=3) return;
    char line[320];
    sprintf_s(line,"[panel] backend %s failed its first frame; rolling back to %s\r\n",
        verify_key.c_str(),verify_prev_key.c_str());
    log_line(line);
    if(!restore_previous()) return;   // retried on the next poll
    unusable=verify_key;
    probe_state.store(0,std::memory_order_release);
    sprintf_s(line,"[panel] backend %s marked unusable this session; back on %s\r\n",
        unusable.c_str(),backend_key.c_str());
    log_line(line);
    set_event("%s unusable; back on %s",unusable.c_str(),backend_key.c_str());
}
void apply_once() {
    verify_swap();
    Values want=applied;
    std::string text;
    if(!parse(want,text)) return;
    const bool backend_changed=want.backend!=backend_key;
    bool controls_changed=!applied_valid;
    if(!controls_changed) {
        std::lock_guard guard(mutex);
        controls_changed=!same_controls(applied,want);
    }
    if(!controls_changed && !backend_changed) return;
    // A switch that could not complete stays pending: only a verified attach
    // moves the request forward, otherwise the next poll tries again.
    if(backend_changed && !swap_backend(want.backend,text)) want.backend=backend_key;
    if(controls_changed) {
        if(apply_controls(want)) {
            Values shown=want;
            shown.backend=backend_key;   // log the backend actually in use
            log_values("apply",shown);
        } else {
            char line[320];
            sprintf_s(line,"[panel] rejected intensity=%.3f local_structure=%.3f "
                "local_tone=%.3f skin=%.3f display=%.3f; keeping previous controls\r\n",
                want.intensity,want.local_structure,want.local_tone,
                want.skin_structure,want.display_strength);
            log_line(line);
        }
    }
    std::lock_guard guard(mutex);
    applied=want;
    applied_valid=true;
}
DWORD WINAPI worker(void*) {
    unsigned ticks=0;
    for(;;) {
        apply_once();
        if((++ticks%2)==0) write_state();
#ifdef NRB_DEFERRED_IDENTITY_TEST
        if((ticks%4)==0) nrb::deferred_diag_poll();
#endif
        Sleep(250);
    }
    return 0;
}
void start() {
    if(HANDLE thread=CreateThread(nullptr,0,worker,nullptr,0,nullptr)) CloseHandle(thread);
}
} // namespace panel
using CreateDevice = HRESULT (WINAPI *)(IUnknown*,D3D_FEATURE_LEVEL,REFIID,void**);
using CreateQueue = HRESULT (STDMETHODCALLTYPE *)(ID3D12Device*,const D3D12_COMMAND_QUEUE_DESC*,REFIID,void**);
using ExecuteLists = void (STDMETHODCALLTYPE *)(ID3D12CommandQueue*,UINT,ID3D12CommandList* const*);
CreateDevice original_create_device=nullptr;
CreateDevice original_create_sl=nullptr;
CreateDevice original_create_host=nullptr;
std::array<CreateDevice,3> hooked_device_exports{};
CreateQueue original_create_queue=nullptr;
ExecuteLists original_execute=nullptr;
#ifdef NRB_DEFERRED_IDENTITY_TEST
std::recursive_mutex execute_serial_mutex;
#endif
struct QueueRecord { ID3D12CommandQueue* object; ID3D12Device* device; uint32_t ordinal; uint64_t submissions; };
struct ListRecord { ID3D12CommandList* object=nullptr; uint32_t queue_ordinal=0; uint64_t serial=0; };
std::vector<QueueRecord> queues;
std::array<ListRecord,128> recent_lists{};
uint64_t submit_serial=0;
uint64_t submit_batch=0;
#ifdef NRB_DLSS_COLOR_STATE_PROOF_V1
// Recorded legacy/direct scope only; not GPU execution-state proof.
// Enable the exact reviewed Cyberpunk profile without a special launcher.
nrb::ColorStateScopeMode select_color_state_scope() noexcept {
    char value[64]{};
    SetLastError(ERROR_SUCCESS);
    const auto size=GetEnvironmentVariableA("NRB_DLSS_COLOR_STATE_PROOF_SCOPE",value,sizeof(value));
    const auto error=GetLastError();
    const bool override_present=size!=0 || error!=ERROR_ENVVAR_NOT_FOUND;
    if(size>=sizeof(value)) return nrb::ColorStateScopeMode::disabled;
    wchar_t executable[1024]{};
    const auto length=GetModuleFileNameW(nullptr,executable,1024);
    const std::wstring_view path=length>0 && length<1024?
        std::wstring_view(executable,length):std::wstring_view{};
    return nrb::dlss_color_select_scope(override_present,
        std::string_view(value,size),path);
}
const auto color_state_scope_mode=select_color_state_scope();
bool color_state_scope_enabled() noexcept {
    return color_state_scope_mode!=nrb::ColorStateScopeMode::disabled;
}
const bool color_state_shadow_mode=color_state_scope_mode==nrb::ColorStateScopeMode::shadow;
nrb::ProofAcquire acquire_color_identity(void*,void* object,bool resource,nrb::ProofIdentity& out) noexcept {
    if(!object) return nrb::ProofAcquire::unknown;
    IUnknown* identity=nullptr;
    if(resource) {
        auto* r=static_cast<ID3D12Resource*>(object);
        if(!nrb::dlss_color_candidate(r->GetDesc())) return nrb::ProofAcquire::not_candidate;
        if(FAILED(r->QueryInterface(IID_PPV_ARGS(&identity)))) return nrb::ProofAcquire::unknown;
    } else {
        auto* list=static_cast<ID3D12CommandList*>(object);
        if(list->GetType()!=D3D12_COMMAND_LIST_TYPE_DIRECT ||
           FAILED(list->QueryInterface(IID_PPV_ARGS(&identity)))) return nrb::ProofAcquire::unknown;
    }
    if(!identity) return nrb::ProofAcquire::unknown;
    out.cookie=reinterpret_cast<uintptr_t>(identity);out.lease=identity;
    return nrb::ProofAcquire::acquired;
}
void release_color_identity(void*,nrb::ProofIdentity& identity) noexcept {
    if(identity.lease) static_cast<IUnknown*>(identity.lease)->Release();
    identity={};
}
nrb::SyncProbe sync_probe({nullptr,acquire_color_identity,release_color_identity},color_state_scope_enabled());
#else
nrb::SyncProbe sync_probe;
#endif
std::atomic<uint32_t> sync_eval_logs{0},sync_submit_logs{0};
using ListReset=HRESULT (STDMETHODCALLTYPE *)(ID3D12GraphicsCommandList*,ID3D12CommandAllocator*,ID3D12PipelineState*);
using ListClose=HRESULT (STDMETHODCALLTYPE *)(ID3D12GraphicsCommandList*);
using ListBarriers=void (STDMETHODCALLTYPE *)(ID3D12GraphicsCommandList*,UINT,const D3D12_RESOURCE_BARRIER*);
#ifdef NRB_TAIL_ORDER_PROBE
using ListDraw=void (STDMETHODCALLTYPE *)(ID3D12GraphicsCommandList*,UINT,UINT,UINT,UINT);
using ListDrawIndexed=void (STDMETHODCALLTYPE *)(ID3D12GraphicsCommandList*,UINT,UINT,UINT,INT,UINT);
using ListDispatch=void (STDMETHODCALLTYPE *)(ID3D12GraphicsCommandList*,UINT,UINT,UINT);
using ListCopyTexture=void (STDMETHODCALLTYPE *)(ID3D12GraphicsCommandList*,
    const D3D12_TEXTURE_COPY_LOCATION*,UINT,UINT,UINT,
    const D3D12_TEXTURE_COPY_LOCATION*,const D3D12_BOX*);
using ListCopyResource=void (STDMETHODCALLTYPE *)(ID3D12GraphicsCommandList*,
    ID3D12Resource*,ID3D12Resource*);
ListDraw original_list_draw=nullptr;
ListDrawIndexed original_list_draw_indexed=nullptr;
ListDispatch original_list_dispatch=nullptr;
ListCopyTexture original_list_copy_texture=nullptr;
ListCopyResource original_list_copy_resource=nullptr;
#endif
ListReset original_list_reset=nullptr;
ListClose original_list_close=nullptr;
ListBarriers original_list_barriers=nullptr;
struct ListMethods {ListReset reset=nullptr;ListClose close=nullptr;ListBarriers barriers=nullptr;
#ifdef NRB_TAIL_ORDER_PROBE
    ListDraw draw=nullptr;ListDrawIndexed draw_indexed=nullptr;ListDispatch dispatch=nullptr;
    ListCopyTexture copy_texture=nullptr;ListCopyResource copy_resource=nullptr;
#endif
};
ListMethods first_list_methods{};
std::atomic<int> list_hook_state{0}; // 0 absent, 1 installing, 2 installed, 3 failed
ID3D12CommandList* consumer_list=nullptr;
uint64_t consumer_submit_before=0;
uint32_t consumer_queue_ordinal=0;

bool attach_detour(void** target,void* replacement) {
    LONG rc=DetourTransactionBegin();
    if(rc==NO_ERROR) rc=DetourUpdateThread(GetCurrentThread());
    if(rc==NO_ERROR) rc=DetourAttach(target,replacement);
    if(rc==NO_ERROR) rc=DetourTransactionCommit();
    else DetourTransactionAbort();
    return rc==NO_ERROR;
}
int32_t __cdecl observed_xess_execute(void* context,ID3D12GraphicsCommandList* list,
                                      const void* params) {
    static std::atomic<unsigned> calls{0};
    if(calls.fetch_add(1,std::memory_order_relaxed)<3)
        log_line("native XeSS execute entered\r\n");
    const auto previous=in_upscaler_api;
    in_upscaler_api=NRB_ROUTE_XESS;
    const auto result=original_xess_execute(context,list,params);
    in_upscaler_api=previous;
    return result;
}
int32_t __cdecl observed_fsr3_dispatch(void* context,const void* params) {
    static std::atomic<unsigned> calls{0};
    const unsigned call=calls.fetch_add(1,std::memory_order_relaxed);
    if(call<3)
        log_line("native FSR3 dispatch entered\r\n");
    // Read only the public FFX descriptor. OptiScaler may pass this dispatch
    // straight through when its D3D12 device discovery fails, so an NGX
    // evaluate hook alone cannot describe the FSR source in that case.
    if(params && (call<3 || (call<=600 && call%120==0))) {
        const auto* frame=static_cast<const Fsr3DispatchProbe*>(params);
        const auto& color=frame->resources[0];
        const auto& depth=frame->resources[1];
        const auto& motion=frame->resources[2];
        const auto& output=frame->resources[9];
        char line[640];
        sprintf_s(line,
            "fsr3_source call=%u context=%p list=%p render=[%u,%u] "
            "color=[%p,%u,%u,%u,%u] motion=[%p,%u,%u,%u,%u] "
            "depth=[%p,%u,%u,%u,%u] output=[%p,%u,%u,%u,%u] "
            "mv_scale=[%.3f,%.3f] jitter=[%.3f,%.3f]\r\n",
            call,context,frame->command_list,frame->render_width,
            frame->render_height,color.resource,color.width,color.height,
            color.format,color.state,motion.resource,motion.width,motion.height,
            motion.format,motion.state,depth.resource,depth.width,depth.height,
            depth.format,depth.state,output.resource,output.width,output.height,
            output.format,output.state,frame->motion_scale_x,frame->motion_scale_y,
            frame->jitter_x,frame->jitter_y);
        log_line(line);
    }
    const auto* frame=static_cast<const Fsr3DispatchProbe*>(params);
#ifdef NRB_TAIL_ORDER_PROBE
    ID3D12GraphicsCommandList* observed_list=nullptr;
    if(frame && frame->command_list && frame->render_width==1280 &&
       frame->render_height==720 && frame->resources[0].resource &&
       frame->resources[0].format==13 && frame->resources[0].width==1280 &&
       frame->resources[0].height==720 && frame->resources[9].resource &&
       frame->resources[9].width==2560 && frame->resources[9].height==1440) {
        observed_list=static_cast<ID3D12GraphicsCommandList*>(frame->command_list);
        schedule_list_hooks(observed_list);
        nrb::tail_probe_rearm(NRB_ROUTE_FSR);
        nrb::tail_before_evaluate(observed_list,
            static_cast<ID3D12Resource*>(frame->resources[0].resource),
            static_cast<ID3D12Resource*>(frame->resources[9].resource),
            NRB_ROUTE_FSR);
    }
#endif
    const auto previous=in_upscaler_api;
    in_upscaler_api=NRB_ROUTE_FSR;
    const auto result=original_fsr3_dispatch(context,params);
    in_upscaler_api=previous;
#ifdef NRB_TAIL_ORDER_PROBE
    if(observed_list) nrb::tail_after_evaluate(observed_list);
#endif
    return result;
}
DWORD WINAPI install_upscaler_api_probes(void*) {
    // The game loads these modules when its menu selects the corresponding
    // renderer. Hook them off the render thread and leave the signed Opti DLL
    // unchanged. A missing/failed hook leaves that route fail-closed.
    for(unsigned tick=0;tick<600 && (!original_xess_execute || !original_fsr3_dispatch);++tick) {
        if(!original_xess_execute) if(auto module=GetModuleHandleW(L"libxess.dll")) {
            auto target=reinterpret_cast<XeSSExecute>(GetProcAddress(module,"xessD3D12Execute"));
            if(target) {
                original_xess_execute=target;
                if(attach_detour(reinterpret_cast<void**>(&original_xess_execute),
                                 reinterpret_cast<void*>(observed_xess_execute)))
                    log_line("XeSS public execute route tag attached\r\n");
                else {original_xess_execute=nullptr;
                    log_line("XeSS route tag unavailable; XeSS NR remains bypassed\r\n");}
            }
        }
        if(!original_fsr3_dispatch) if(auto module=GetModuleHandleW(L"ffx_fsr3upscaler_x64.dll")) {
            auto target=reinterpret_cast<FSR3Dispatch>(GetProcAddress(module,
                "ffxFsr3UpscalerContextDispatch"));
            if(target) {
                original_fsr3_dispatch=target;
                if(attach_detour(reinterpret_cast<void**>(&original_fsr3_dispatch),
                                 reinterpret_cast<void*>(observed_fsr3_dispatch)))
                    log_line("FSR3 public dispatch route tag attached\r\n");
                else {original_fsr3_dispatch=nullptr;
                    log_line("FSR3 route tag unavailable; FSR NR remains bypassed\r\n");}
            }
        }
        Sleep(1000);
    }
    return 0;
}
int32_t __cdecl observed_sl_evaluate(uint32_t feature,const void* frame,
    const void** inputs,uint32_t count,void* command_buffer) {
    const bool previous=in_sl_dlss_evaluate;
    if(feature==0 && command_buffer) in_sl_dlss_evaluate=true;
    const int32_t result=original_sl_evaluate(feature,frame,inputs,count,command_buffer);
    in_sl_dlss_evaluate=previous;
    return result;
}
void install_sl_feature_probe() {
    HMODULE sl=GetModuleHandleW(L"sl.interposer.dll");
    if(!sl) {log_line("Streamline feature entry absent; external route stays unknown\r\n");return;}
    original_sl_evaluate=reinterpret_cast<SLEvaluate>(
        GetProcAddress(sl,"slEvaluateFeature"));
    if(!original_sl_evaluate || !attach_detour(
            reinterpret_cast<void**>(&original_sl_evaluate),
            reinterpret_cast<void*>(observed_sl_evaluate))) {
        original_sl_evaluate=nullptr;
        log_line("Streamline feature hook unavailable; external route stays unknown\r\n");
        return;
    }
    log_line("Streamline slEvaluateFeature feature tag attached\r\n");
}

void STDMETHODCALLTYPE observed_execute(ID3D12CommandQueue* q,UINT count,ID3D12CommandList* const* lists) {
#ifdef NRB_DEFERRED_IDENTITY_TEST
    // Keep other intercepted ExecuteCommandLists calls from interleaving with
    // the prefix/private/suffix submissions on this diagnostic queue path.
    std::lock_guard execute_guard(execute_serial_mutex);
#endif
    uint64_t batch=0;
    std::array<nrb::SubmitObservation,64> correlated{};
    {
        std::lock_guard guard(queue_mutex);
        for(auto& entry:queues) if(entry.object==q) {
            ++entry.submissions;
            batch=++submit_batch;
            for(UINT i=0;i<count && lists;i++) {
                const uint64_t serial=++submit_serial;
                recent_lists[serial & (recent_lists.size()-1)]={lists[i],entry.ordinal,serial};
            }
            break;
        }
    }
    // Diagnostic correlation only. No queue pointer or submit observation
    // certifies that this frame's resource writes completed before Evaluate.
    for(UINT i=0;i<count && lists && i<correlated.size() && batch;i++)
        correlated[i]=sync_probe.submit(lists[i],batch);
    // Bind the list generation at call entry: another thread may Reset the
    // list immediately after ExecuteCommandLists returns. This is still only
    // an API-call observation, never a GPU-completion fence.
#ifdef NRB_DEFERRED_IDENTITY_TEST
    NRB_Controls deferred_controls{};
    NRB_Processor deferred_processor{};
    {std::lock_guard guard(state_mutex);
        deferred_controls=controls;deferred_processor=processor;}
    const unsigned deferred=nrb::submit_deferred_identity(q,count,lists,
        count<=correlated.size()?correlated.data():nullptr,original_execute,
        &deferred_processor,&deferred_controls,log_line);
    if(!deferred) original_execute(q,count,lists);
    else {
        static std::atomic<unsigned> deferred_logs{0};
        if(deferred_logs.fetch_add(1,std::memory_order_relaxed)<8) {
            char line[320];
            sprintf_s(line,"deferred_identity_submitted batch_count=%u inserted=%u recorded=%u retired=%u signal_failures=%u generation_mismatches=%u invalidated=%u hdr_ready=%u nr_frames=%llu nr_failed=%u nr_last_ms=%.2f\r\n",
                count,deferred,nrb::deferred_identity_recorded(),
                nrb::deferred_identity_retired(),
                nrb::deferred_identity_signal_failures(),
                nrb::deferred_identity_generation_mismatches(),
                nrb::deferred_identity_invalidated(),nrb::deferred_hdr_ready(),
                static_cast<unsigned long long>(nrb::deferred_nr_processed()),
                unsigned(nrb::deferred_nr_failed()),nrb::deferred_nr_last_ms());
            log_line(line);
        }
    }
#else
    original_execute(q,count,lists);
#endif
#ifdef NRB_XESS_IDENTITY_TEST
    const unsigned relayed=nrb::xess_identity_submitted(q,count,lists);
    if(relayed) {
        static std::atomic<unsigned> submit_logs{0};
        if(submit_logs.fetch_add(1,std::memory_order_relaxed)<8) {
            char line[160];
            sprintf_s(line,"ngx_identity_submitted count=%u retired=%u\r\n",
                relayed,nrb::xess_identity_completed());
            log_line(line);
        }
    }
#endif
#ifdef NRB_ONE_FRAME_COLOR_READBACK
    nrb::one_frame_readback_submitted(q,count,lists,correlated.data(),
                                     static_cast<UINT>(correlated.size()),log_line);
#endif
    for(UINT i=0;i<count && lists && i<correlated.size();i++) {
        const auto& observation=correlated[i];
        if(!observation.correlated || sync_submit_logs.fetch_add(1)>=16) continue;
        char line[320];
        sprintf_s(line,"{\"event\":\"execute_lists_call\",\"list\":\"%p\",\"generation\":%llu,\"batch\":%llu,\"batch_count\":%u,\"batch_index\":%u,\"evaluations\":%u,\"latest_evaluate_sequence\":%llu,\"proof\":\"unknown\"}\r\n",
            static_cast<void*>(lists[i]),observation.generation,observation.batch,
            count,i,observation.evaluations,observation.latest_evaluate_sequence);
        log_line(line);
    }
}
HRESULT STDMETHODCALLTYPE observed_list_reset(ID3D12GraphicsCommandList* list,
    ID3D12CommandAllocator* allocator,ID3D12PipelineState* state) {
    const HRESULT hr=original_list_reset(list,allocator,state);
    sync_probe.reset(list,SUCCEEDED(hr));
#ifdef NRB_TAIL_ORDER_PROBE
    if(SUCCEEDED(hr)) nrb::tail_reset(list);
#endif
#ifdef NRB_DEFERRED_IDENTITY_TEST
    if(SUCCEEDED(hr)) nrb::deferred_identity_list_reset(list);
#endif
    return hr;
}
HRESULT STDMETHODCALLTYPE observed_list_close(ID3D12GraphicsCommandList* list) {
    const HRESULT hr=original_list_close(list);
    sync_probe.close(list,SUCCEEDED(hr));
#ifdef NRB_TAIL_ORDER_PROBE
    if(SUCCEEDED(hr)) {
        const auto s=nrb::tail_close(list);
        if(s.valid) {
            char line[800];
            sprintf_s(line,"{\"event\":\"same_list_tail\",\"route\":%u,\"pre_actions\":%u,\"in_xess_actions\":%u,\"after_actions\":%u,\"after_dispatches\":%u,\"after_draws\":%u,\"last_dispatch\":[%u,%u,%u],\"after_color_barriers\":%u,\"after_output_barriers\":%u,\"output_read_barriers\":%u,\"actions_after_output_read\":%u,\"after_output_copies\":%u,\"color_state\":[%u,%u],\"output_state\":[%u,%u],\"color_flags\":%u,\"output_flags\":%u,\"color_subresource\":%u,\"output_subresource\":%u}\r\n",
                unsigned(s.route),s.pre_actions,s.in_evaluate_actions,s.after_actions,
                s.after_dispatches,s.after_draws,
                s.last_dispatch_x,s.last_dispatch_y,s.last_dispatch_z,
                s.after_color_barriers,s.after_output_barriers,
                s.output_read_barriers,s.actions_after_output_read,
                s.after_output_copies,s.color_before,s.color_after,
                s.output_before,s.output_after,s.color_flags,s.output_flags,
                s.color_subresource,s.output_subresource);
            log_line(line);
            if(s.route==NRB_ROUTE_XESS) {
                char copy_line[220];
                sprintf_s(copy_line,
                    "xess_tail_copy safe=%u from_color=%u to_color=%u to_output=%u src=[%u,%u,%u] dst=[%u,%u,%u]\r\n",
                    unsigned(nrb::tail_safe_for_deferred_identity(s)),
                    s.after_copy_from_color,s.after_copy_to_color,
                    s.after_copy_to_output,s.copy_source_format,
                    s.copy_source_width,s.copy_source_height,
                    s.copy_dest_format,s.copy_dest_width,s.copy_dest_height);
                log_line(copy_line);
            }
#ifdef NRB_DEFERRED_IDENTITY_TEST
            nrb::observe_deferred_identity_tail(s);
#endif
        }
    }
#endif
    return hr;
}
void STDMETHODCALLTYPE observed_list_barriers(ID3D12GraphicsCommandList* list,UINT count,
                                               const D3D12_RESOURCE_BARRIER* barriers) {
    original_list_barriers(list,count,barriers);
    sync_probe.barriers(list,count,barriers);
#ifdef NRB_TAIL_ORDER_PROBE
    nrb::tail_barriers(list,count,barriers);
#endif
}
#ifdef NRB_TAIL_ORDER_PROBE
void STDMETHODCALLTYPE observed_list_draw(ID3D12GraphicsCommandList* list,
    UINT vertices,UINT instances,UINT first_vertex,UINT first_instance) {
    original_list_draw(list,vertices,instances,first_vertex,first_instance);
    nrb::tail_action(list);
}
void STDMETHODCALLTYPE observed_list_draw_indexed(ID3D12GraphicsCommandList* list,
    UINT indices,UINT instances,UINT first_index,INT base_vertex,UINT first_instance) {
    original_list_draw_indexed(list,indices,instances,first_index,base_vertex,first_instance);
    nrb::tail_action(list);
}
void STDMETHODCALLTYPE observed_list_dispatch(ID3D12GraphicsCommandList* list,
    UINT x,UINT y,UINT z) {
    original_list_dispatch(list,x,y,z);
    nrb::tail_action(list,true,x,y,z);
}
void STDMETHODCALLTYPE observed_list_copy_texture(ID3D12GraphicsCommandList* list,
    const D3D12_TEXTURE_COPY_LOCATION* dest,UINT x,UINT y,UINT z,
    const D3D12_TEXTURE_COPY_LOCATION* source,const D3D12_BOX* box) {
    original_list_copy_texture(list,dest,x,y,z,source,box);
    nrb::tail_copy(list,source?source->pResource:nullptr,
        dest?dest->pResource:nullptr,true);
}
void STDMETHODCALLTYPE observed_list_copy_resource(ID3D12GraphicsCommandList* list,
    ID3D12Resource* dest,ID3D12Resource* source) {
    original_list_copy_resource(list,dest,source);
    nrb::tail_copy(list,source,dest,true);
}
#endif
DWORD WINAPI install_list_hooks_worker(void*) {
    std::lock_guard guard(hook_mutex);
    original_list_reset=first_list_methods.reset;
    original_list_close=first_list_methods.close;
    original_list_barriers=first_list_methods.barriers;
#ifdef NRB_TAIL_ORDER_PROBE
    original_list_draw=first_list_methods.draw;
    original_list_draw_indexed=first_list_methods.draw_indexed;
    original_list_dispatch=first_list_methods.dispatch;
    original_list_copy_texture=first_list_methods.copy_texture;
    original_list_copy_resource=first_list_methods.copy_resource;
#endif
    LONG rc=DetourTransactionBegin();
    if(rc==NO_ERROR) rc=DetourUpdateThread(GetCurrentThread());
    if(rc==NO_ERROR) rc=DetourAttach(reinterpret_cast<void**>(&original_list_reset),
                                     reinterpret_cast<void*>(observed_list_reset));
    if(rc==NO_ERROR) rc=DetourAttach(reinterpret_cast<void**>(&original_list_close),
                                     reinterpret_cast<void*>(observed_list_close));
    if(rc==NO_ERROR) rc=DetourAttach(reinterpret_cast<void**>(&original_list_barriers),
                                     reinterpret_cast<void*>(observed_list_barriers));
#ifdef NRB_TAIL_ORDER_PROBE
    if(rc==NO_ERROR) rc=DetourAttach(reinterpret_cast<void**>(&original_list_draw),
                                     reinterpret_cast<void*>(observed_list_draw));
    if(rc==NO_ERROR) rc=DetourAttach(reinterpret_cast<void**>(&original_list_draw_indexed),
                                     reinterpret_cast<void*>(observed_list_draw_indexed));
    if(rc==NO_ERROR) rc=DetourAttach(reinterpret_cast<void**>(&original_list_dispatch),
                                     reinterpret_cast<void*>(observed_list_dispatch));
    if(rc==NO_ERROR) rc=DetourAttach(reinterpret_cast<void**>(&original_list_copy_texture),
                                     reinterpret_cast<void*>(observed_list_copy_texture));
    if(rc==NO_ERROR) rc=DetourAttach(reinterpret_cast<void**>(&original_list_copy_resource),
                                     reinterpret_cast<void*>(observed_list_copy_resource));
#endif
    if(rc==NO_ERROR) rc=DetourTransactionCommit();
    else DetourTransactionAbort();
    list_hook_state.store(rc==NO_ERROR?2:3,std::memory_order_release);
    log_line(rc==NO_ERROR?"D3D12 list Reset/Close/ResourceBarrier metadata hooks installed\r\n":
                          "D3D12 list metadata hook installation failed; sync unknown\r\n");
    return 0;
}
void schedule_list_hooks(ID3D12GraphicsCommandList* list) {
    int expected=0;
    if(!list || !list_hook_state.compare_exchange_strong(expected,1)) return;
    auto** methods=*reinterpret_cast<void***>(list);
    first_list_methods.reset=reinterpret_cast<ListReset>(methods[10]);
    first_list_methods.close=reinterpret_cast<ListClose>(methods[9]);
    first_list_methods.barriers=reinterpret_cast<ListBarriers>(methods[26]);
#ifdef NRB_TAIL_ORDER_PROBE
    first_list_methods.draw=reinterpret_cast<ListDraw>(methods[12]);
    first_list_methods.draw_indexed=reinterpret_cast<ListDrawIndexed>(methods[13]);
    first_list_methods.dispatch=reinterpret_cast<ListDispatch>(methods[14]);
    first_list_methods.copy_texture=reinterpret_cast<ListCopyTexture>(methods[16]);
    first_list_methods.copy_resource=reinterpret_cast<ListCopyResource>(methods[17]);
#endif
    HANDLE worker=CreateThread(nullptr,0,install_list_hooks_worker,nullptr,0,nullptr);
    if(worker) CloseHandle(worker);
    else {list_hook_state.store(3,std::memory_order_release);
          log_line("D3D12 list metadata hook worker failed; sync unknown\r\n");}
}
void observe_queue(ID3D12CommandQueue* q) {
    if(!q || q->GetDesc().Type!=D3D12_COMMAND_LIST_TYPE_DIRECT) return;
    ID3D12Device* d=nullptr;
    if(FAILED(q->GetDevice(IID_PPV_ARGS(&d)))) return;
    {
        std::lock_guard guard(queue_mutex);
        for(auto& e:queues) if(e.object==q) {d->Release();return;}
        q->AddRef(); queues.push_back({q,d,static_cast<uint32_t>(queues.size()+1),0});
    }
    std::lock_guard hook_guard(hook_mutex);
    if(!original_execute) {
        auto** methods=*reinterpret_cast<void***>(q);
        original_execute=reinterpret_cast<ExecuteLists>(methods[10]);
        if(!attach_detour(reinterpret_cast<void**>(&original_execute),
                          reinterpret_cast<void*>(observed_execute))) {
            original_execute=nullptr; log_line("queue ExecuteCommandLists detour failed\r\n");
        }
    }
}
HRESULT STDMETHODCALLTYPE observed_create_queue(ID3D12Device* d,const D3D12_COMMAND_QUEUE_DESC* desc,
                                                 REFIID iid,void** output) {
    const HRESULT hr=original_create_queue(d,desc,iid,output);
    if(SUCCEEDED(hr) && output && *output && iid==__uuidof(ID3D12CommandQueue))
        observe_queue(static_cast<ID3D12CommandQueue*>(*output));
    return hr;
}
void observe_device(ID3D12Device* d) {
    if(!d) return;
    std::lock_guard hook_guard(hook_mutex);
    if(original_create_queue) return;
    auto** methods=*reinterpret_cast<void***>(d);
    original_create_queue=reinterpret_cast<CreateQueue>(methods[8]);
    if(!attach_detour(reinterpret_cast<void**>(&original_create_queue),
                      reinterpret_cast<void*>(observed_create_queue))) {
        original_create_queue=nullptr; log_line("device CreateCommandQueue detour failed\r\n");
    }
}
// [vk102 integration] One-shot mount of the frozen nrvk component after the
// first real D3D12 device is captured. The nrvk component registers its
// processor through the existing host callback, replacing the built-in one.
// ------------------------------------------------------------ bridge config
// Per-game paths are read from nr_bridge.ini next to this module so one build
// can serve several games. The component keeps these pointers past the attach
// call, so they live in static buffers.
static int g_tail_profile=0;
static void load_bridge_config() {
    g_tail_profile=0;
    char path[MAX_PATH]{};
    HMODULE self=nullptr;
    if(GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS|
            GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCSTR>(&load_bridge_config),&self)) {
        const DWORD n=GetModuleFileNameA(self,path,MAX_PATH);
        if(n && n<MAX_PATH) {
            const auto directory=std::filesystem::path(path).parent_path();
            strcpy_s(g_manifest_dir,(directory/"../nr-runtime/manifest").lexically_normal().string().c_str());
            strcpy_s(g_status_json,(directory/"component_status.json").string().c_str());
            strcpy_s(g_panel_ini,(directory/"nr_panel.ini").string().c_str());
            strcpy_s(g_panel_state,(directory/"panel_state.json").string().c_str());
            char* slash=strrchr(path,'\\');
            if(slash) {
                strcpy_s(slash+1,MAX_PATH-(slash+1-path),"nr_bridge.ini");
                HANDLE h=CreateFileA(path,GENERIC_READ,
                    FILE_SHARE_READ|FILE_SHARE_WRITE|FILE_SHARE_DELETE,nullptr,
                    OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,nullptr);
                if(h!=INVALID_HANDLE_VALUE) {
                    char buf[8192]{}; DWORD got=0;
                    ReadFile(h,buf,sizeof buf-1,&got,nullptr);
                    CloseHandle(h);
                    std::string text(buf,got);
                    std::string value=panel::lookup(text,"bridge","manifest_dir");
                    if(!value.empty()) strcpy_s(g_manifest_dir,sizeof g_manifest_dir,value.c_str());
                    value=panel::lookup(text,"bridge","status_json");
                    if(!value.empty()) strcpy_s(g_status_json,sizeof g_status_json,value.c_str());
                    value=panel::lookup(text,"bridge","panel_ini");
                    if(!value.empty()) strcpy_s(g_panel_ini,value.c_str());
                    value=panel::lookup(text,"bridge","panel_state");
                    if(!value.empty()) strcpy_s(g_panel_state,value.c_str());
                    panel::backend_manifest=g_manifest_dir;
                    value=panel::lookup(text,"bridge","tail_profile");
                    if(!value.empty()) g_tail_profile=atoi(value.c_str());
                }
            }
        }
    }
    char line[600];
    sprintf_s(line,"[nrvk] config manifest_dir=%s status_json=%s tail_profile=%d\r\n",
        g_manifest_dir,g_status_json,g_tail_profile);
    log_line(line);
}
static void nrvk_try_mount() {
    ID3D12Device* device=observed_d3d12_device.load(std::memory_order_acquire);
    if(!device) return;
    if(nrvk_mount_attempted.exchange(true,std::memory_order_acq_rel)) return;
    nrb::tail_set_profile(g_tail_profile);
    struct nrvk_config_mirror {
        uint32_t abi_size;
        const char* manifest_dir;
        const char* cache_dir;
        const char* status_json_path;
        uint32_t output_ring_slots;
        uint32_t log_verbose;
    };
    static_assert(sizeof(nrvk_config_mirror)==40,"NRVK_Config layout drift");
    log_line("[nrvk] loading component\r\n");
    HMODULE nb=LoadLibraryA("nrvk_bridge.dll");
    {char ld[128]; sprintf_s(ld,"[nrvk] LoadLibraryA=%p err=%lu\r\n",(void*)nb,GetLastError()); log_line(ld);}
    if(!nb) { nrvk_mount_failed.store(true,std::memory_order_release);
        log_line("[nrvk] nrvk_bridge.dll not found; required backend cannot run; no fallback\r\n"); return; }
    nrvk_bridge_module=nb;
    panel::backend_module=nb;
    panel::boot_module=nb;
    {char path[MAX_PATH]{}; const DWORD n=GetModuleFileNameA(nb,path,MAX_PATH);
     if(n) panel::backend_dll.assign(path,n);}
    panel::boot_dll=panel::backend_dll;
    using RegFn=int(*)(const NRB_Processor*);
    using AttachFn=int(*)(ID3D12Device*,const void*,RegFn);
    using ShutdownFn=int(*)(void);
    auto attach=(AttachFn)GetProcAddress(nb,"NRVK_AttachToBridge");
    auto stop=(ShutdownFn)GetProcAddress(nb,"NRVK_Shutdown");
    if(!attach||!stop) { nrvk_mount_failed.store(true,std::memory_order_release);
        log_line("[nrvk] required exports missing; no fallback\r\n"); return; }
    nrvk_config_mirror cfg{sizeof(nrvk_config_mirror)};
    cfg.manifest_dir=g_manifest_dir;
    cfg.status_json_path=g_status_json;
    cfg.output_ring_slots=3;
    cfg.log_verbose=1;
    // Bisect the attach: ComponentSelfTest only validates the plan, so a crash
    // before this returns points at the manifest probe, not at device objects.
    auto self_test=(int(*)(const void*,void*))GetProcAddress(nb,"NRVK_ComponentSelfTest");
    if(self_test) {
        const int stc=self_test(&cfg,nullptr);
        char st[96]; sprintf_s(st,"[nrvk] ComponentSelfTest rc=%d\r\n",stc); log_line(st);
    } else log_line("[nrvk] ComponentSelfTest export missing\r\n");
    log_line("[nrvk] calling AttachToBridge\r\n");
    int arc=attach(device,&cfg,&NRB_RegisterProcessor);
    {char at[96]; sprintf_s(at,"[nrvk] attach returned %d\r\n",arc); log_line(at);}
    char msg[192];
    std::snprintf(msg,sizeof msg,"[nrvk] attach rc=%d; component_required=1; fallback=0; manifest_dir=%s\r\n",arc,g_manifest_dir);
    log_line(msg);
    if(arc==0) {
        nrvk_bridge_mounted.store(true,std::memory_order_release);
        panel::set_event("boot mount ok");
        std::atexit([]{
#ifdef NRB_DEFERRED_IDENTITY_TEST
            nrb::deferred_diag_shutdown();
#endif
            if(nrvk_bridge_module) {
            auto shutdown=(ShutdownFn)GetProcAddress(nrvk_bridge_module,"NRVK_Shutdown");
            if(shutdown) shutdown();
        }});
    } else nrvk_mount_failed.store(true,std::memory_order_release);
}
static DWORD WINAPI nrvk_mount_worker(void*) { nrvk_try_mount(); return 0; }
// WARP (Microsoft Basic Render Driver) devices are created by engines that
// probe capabilities first and then release them. Attaching a component to such
// a device faults as soon as the engine drops it, so they are never captured.
static bool is_warp_device(ID3D12Device* d) {
    IDXGIDevice* g=nullptr;
    if(!d || FAILED(d->QueryInterface(IID_PPV_ARGS(&g)))) return false;
    bool warp=false;
    IDXGIAdapter* a=nullptr;
    if(SUCCEEDED(g->GetAdapter(&a)) && a) {
        DXGI_ADAPTER_DESC desc{};
        if(SUCCEEDED(a->GetDesc(&desc))) warp = desc.VendorId==0x1414;
        a->Release();
    }
    g->Release();
    return warp;
}
HRESULT record_created_device(HRESULT hr,void** output) {
    if(SUCCEEDED(hr) && output && *output) {
        ID3D12Device* d=nullptr;
        if(SUCCEEDED(static_cast<IUnknown*>(*output)->QueryInterface(IID_PPV_ARGS(&d)))) {
            observe_device(d);
            const bool warp=is_warp_device(d);
            ID3D12Device* current=observed_d3d12_device.load(std::memory_order_acquire);
            if(warp) {
                static std::atomic<bool> warned{false};
                if(!warned.exchange(true)) log_line("[nrvk] ignored a WARP device (engine probe)\r\n");
            } else if(current!=d) {
                d->AddRef();                    // the bridge keeps this device alive
                if(observed_d3d12_device.compare_exchange_strong(current,d)) {
                    if(current) current->Release();
                    nrvk_mount_attempted.store(false,std::memory_order_release);
                    nrvk_mount_failed.store(false,std::memory_order_release);
                    char line[160];
                    sprintf_s(line,"[nrvk] real device captured (vendor!=WARP); mount armed\r\n");
                    log_line(line);
                } else d->Release();
            }
            d->Release();
        }
    }
    return hr;
}
HRESULT WINAPI observed_create_device(IUnknown* adapter,D3D_FEATURE_LEVEL level,REFIID iid,void** output) {
    return record_created_device(original_create_device(adapter,level,iid,output),output);
}
HRESULT WINAPI observed_create_sl(IUnknown* adapter,D3D_FEATURE_LEVEL level,REFIID iid,void** output) {
    return record_created_device(original_create_sl(adapter,level,iid,output),output);
}
HRESULT WINAPI observed_create_host(IUnknown* adapter,D3D_FEATURE_LEVEL level,REFIID iid,void** output) {
    return record_created_device(original_create_host(adapter,level,iid,output),output);
}
void hook_device_export(HMODULE module,CreateDevice& slot,CreateDevice replacement) {
    if(!module) return;
    auto address=reinterpret_cast<CreateDevice>(GetProcAddress(module,"D3D12CreateDevice"));
    if(!address) return;
    for(auto installed:hooked_device_exports) if(installed==address) return;
    slot=address;
    if(!attach_detour(reinterpret_cast<void**>(&slot),reinterpret_cast<void*>(replacement))) {
        slot=nullptr;log_line("D3D12CreateDevice export detour failed\r\n");
    } else for(auto& installed:hooked_device_exports) if(!installed) {installed=address;break;}
}
void install_queue_capture() {
    hook_device_export(GetModuleHandleW(L"d3d12.dll"),original_create_device,observed_create_device);
    hook_device_export(GetModuleHandleW(L"sl.interposer.dll"),original_create_sl,observed_create_sl);
    if(!original_create_device && !original_create_sl)
        log_line("D3D12 queue capture: no device factory export loaded at ASI init\r\n");
}
struct QueueSnapshot { nrb::QueueMeta meta; ID3D12CommandQueue* selected=nullptr; uint64_t serial=0; };
QueueSnapshot queue_snapshot(ID3D12Device* d,ID3D12CommandList* list) {
    QueueSnapshot result{};
    std::lock_guard guard(queue_mutex);
    uint32_t matching=0;
    for(auto& e:queues) if(e.device==d) {
        ++matching; result.selected=e.object;
        result.meta.ordinal=e.ordinal;
        result.meta.type=static_cast<uint32_t>(e.object->GetDesc().Type);
        result.meta.submissions=e.submissions;
    }
    result.meta.candidate_count=matching;
    if(matching!=1) {result.selected=nullptr;result.meta.ordinal=0;}
    result.meta.same_device=matching>0;
    for(auto& item:recent_lists) if(item.object==list) {
        result.meta.consumer_list_previously_submitted=true;
        if(item.serial>result.serial) {
            result.serial=item.serial;
            result.meta.last_seen_consumer_queue_ordinal=item.queue_ordinal;
            result.meta.last_seen_consumer_submit_serial=item.serial;
        }
    }
    result.meta.producer_status=matching==0?"no_direct_queue_observed":
        matching>1?"ambiguous_direct_queues":"unproven_current_frame_resource_write";
    if(result.selected) result.selected->AddRef();
    return result;
}
bool consumer_submitted_after(ID3D12CommandList* list,uint64_t prior,uint32_t ordinal) {
    std::lock_guard guard(queue_mutex);
    for(auto& item:recent_lists) if(item.object==list && item.serial>prior &&
        item.queue_ordinal==ordinal) return true;
    return false;
}

void log_line(const char* message) {
    std::lock_guard guard(log_mutex);
    HMODULE self = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
        GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        reinterpret_cast<LPCWSTR>(&log_line), &self)) return;
    wchar_t path[MAX_PATH]{};
    DWORD size = GetModuleFileNameW(self, path, MAX_PATH);
    if (!size || size >= MAX_PATH) return;
    wchar_t* slash = wcsrchr(path, L'\\');
    if (!slash) return;
    wcscpy_s(slash + 1, MAX_PATH - (slash + 1 - path), L"CyberpunkNRBridge.log");
    HANDLE file = CreateFileW(path, FILE_APPEND_DATA,
        FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) return;
    DWORD written = 0;
    WriteFile(file, message, static_cast<DWORD>(strlen(message)), &written, nullptr);
    CloseHandle(file);
}

bool sync_queue() {
    // A failed retire has already poisoned the processor. Keep the borrowed
    // resource pinned and avoid retrying/logging the same Python error every
    // frame; recovery requires a fresh process with a known GPU state.
    if (processor_failed) return false;
    if (!queue) return false;
    if (borrowed.color && (!consumer_list || !consumer_queue_ordinal ||
        !consumer_submitted_after(consumer_list,consumer_submit_before,consumer_queue_ordinal)))
        return false;
    if (!completion) {
        ID3D12Device* d = nullptr;
        if (FAILED(queue->GetDevice(IID_PPV_ARGS(&d)))) return false;
        const HRESULT hr = d->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&completion));
        d->Release();
        if (FAILED(hr)) return false;
        completion_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (!completion_event) return false;
    }
    if (FAILED(queue->Signal(completion, ++completion_value)) ||
        FAILED(completion->SetEventOnCompletion(completion_value, completion_event)) ||
        WaitForSingleObject(completion_event, 10000) != WAIT_OBJECT_0) return false;
    if (borrowed.color) {
        try {
            if (!processor.retire || processor.retire(processor.context,completion,completion_value)!=0) {
                processor_failed=true;
                return false;
            }
        } catch (...) { processor_failed=true; return false; }
        borrowed.color->Release();
        borrowed.ready_fence->Release();
        borrowed = {};
        consumer_list=nullptr;
        consumer_queue_ordinal=0;
    }
    return true;
}

struct RouteDecision { NRB_Route route; NRB_RouteEvidence evidence; };
RouteDecision classify(void* caller, NVSDK_NGX_Parameter* params) {
    if(in_upscaler_api==NRB_ROUTE_XESS)
        return {NRB_ROUTE_XESS,NRB_EVIDENCE_XESS_API};
    if(in_upscaler_api==NRB_ROUTE_FSR)
        return {NRB_ROUTE_FSR,NRB_EVIDENCE_FSR3_API};
    if(in_sl_dlss_evaluate) {
        static std::atomic<bool> logged{false};
        if(!logged.exchange(true,std::memory_order_relaxed))
            log_line("NGX Evaluate nested in Streamline kFeatureDLSS=0\r\n");
        return {NRB_ROUTE_DLSS,NRB_EVIDENCE_STREAMLINE_FEATURE};
    }
    HMODULE caller_module = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
        GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        reinterpret_cast<LPCWSTR>(caller), &caller_module))
        return {NRB_ROUTE_UNKNOWN,NRB_EVIDENCE_NONE};
    if (caller_module == GetModuleHandleW(nullptr))
        return {NRB_ROUTE_DLSS,NRB_EVIDENCE_GAME_NGX};
    HMODULE streamline=GetModuleHandleW(L"sl.interposer.dll");
    if (streamline && caller_module == streamline)
        return {NRB_ROUTE_DLSS,NRB_EVIDENCE_STREAMLINE_NGX};
    if (caller_module != host)
        return {NRB_ROUTE_UNKNOWN,NRB_EVIDENCE_EXTERNAL_UNTAGGED};
    float fsr_time = 0;
    if (params && params->Get("FSR.frameTimeDelta", &fsr_time) == NVSDK_NGX_Result_Success)
        return {NRB_ROUTE_FSR,NRB_EVIDENCE_FSR_MARKER};
    int xess_marker=0;
    if (params && (params->Get("XeSS.ExposureScaleTexture",&xess_marker)==NVSDK_NGX_Result_Success ||
                   params->Get("XeSS.ResponsivePixelMask",&xess_marker)==NVSDK_NGX_Result_Success))
        return {NRB_ROUTE_XESS,NRB_EVIDENCE_XESS_MARKER};
    // Internal source without an explicit marker may be FSR, XeSS, or another
    // proxy. Do not infer XeSS by exclusion across different games.
    return {NRB_ROUTE_UNKNOWN,NRB_EVIDENCE_INTERNAL_UNTAGGED};
}

NVSDK_NGX_Result __cdecl intercept(ID3D12GraphicsCommandList* list,
    const NVSDK_NGX_Handle* handle, NVSDK_NGX_Parameter* params,
    PFN_NVSDK_NGX_ProgressCallback callback) {
    if (!original) return NVSDK_NGX_Result_Fail;
    if (inside || !list || !handle || !params) return original(list, handle, params, callback);
    void* caller = _ReturnAddress();
    inside = true;
    // Lazy component mount: by the first evaluate the engine has finished its
    // own startup, so loading the component here cannot deadlock the loader.
    if(!nrvk_bridge_mounted.load(std::memory_order_relaxed) &&
       !nrvk_mount_failed.load(std::memory_order_relaxed) &&
       !nrvk_mount_attempted.load(std::memory_order_relaxed)) {
        nrvk_try_mount();
    }
    const RouteDecision decision = classify(caller, params);
    const NRB_Route route = decision.route;
    if(route==NRB_ROUTE_FSR || route==NRB_ROUTE_XESS) {
        static std::atomic<unsigned> route_logs[4]{};
        if(route_logs[route].fetch_add(1,std::memory_order_relaxed)<8) {
            char line[180];
            sprintf_s(line,"upscaler_entry route=%u evidence=%u handle=%llu\r\n",
                unsigned(route),unsigned(decision.evidence),
                static_cast<unsigned long long>(handle->Id));
            log_line(line);
        }
    }
#if defined(NRB_XESS_IDENTITY_TEST) || defined(NRB_TAIL_ORDER_PROBE)
    HMODULE scope_module=nullptr;
    GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
        GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        reinterpret_cast<LPCWSTR>(caller),&scope_module);
    // Cyberpunk's Streamline invokes the signed OptiScaler evaluator from
    // sl.common.dll, outside slEvaluateFeature; keep production route UNKNOWN.
    // This diagnostic only gates the byte-identical relay on that caller.
    const bool diagnostic_dlss_scope=route==NRB_ROUTE_DLSS ||
        (decision.evidence==NRB_EVIDENCE_EXTERNAL_UNTAGGED &&
         scope_module && scope_module==GetModuleHandleW(L"sl.common.dll") &&
         GetModuleHandleW(L"Cyberpunk2077.exe") &&
         handle->Id>=1000000 && handle->Id<2000000);
    const bool trusted_other_sr_scope=
        (route==NRB_ROUTE_XESS && decision.evidence==NRB_EVIDENCE_XESS_API) ||
        (route==NRB_ROUTE_FSR && decision.evidence==NRB_EVIDENCE_FSR3_API);
#ifdef NRB_MULTI_ROUTE_LIVE
    const bool diagnostic_other_sr_scope=trusted_other_sr_scope;
#else
    const bool diagnostic_other_sr_scope=false;
#endif
    const bool diagnostic_sr_scope=diagnostic_dlss_scope || diagnostic_other_sr_scope;
#ifdef NRB_TAIL_ORDER_PROBE
    if(diagnostic_dlss_scope || trusted_other_sr_scope) {
        static std::atomic<NRB_Route> last_tail_route{NRB_ROUTE_UNKNOWN};
        const NRB_Route sample_route=diagnostic_dlss_scope?NRB_ROUTE_DLSS:route;
        const auto previous=last_tail_route.exchange(sample_route,std::memory_order_acq_rel);
        if(previous!=sample_route) {
            nrb::tail_probe_rearm(sample_route);
#ifdef NRB_DEFERRED_IDENTITY_TEST
            nrb::reset_deferred_identity_tail(sample_route);
#endif
        }
    }
#endif
#ifdef NRB_XESS_IDENTITY_TEST
    bool identity_source_state_proven=false;
    uint64_t identity_source_generation=0;
#endif
#endif
    nrb::flashdiag::Context flash{};
    flash.eval_id=nrb::flashdiag::recorder.begin();
    flash.list=reinterpret_cast<uintptr_t>(list);flash.feature_id=handle->Id;
    flash.route=unsigned(route);flash.evidence=unsigned(decision.evidence);
#if defined(NRB_XESS_IDENTITY_TEST) || defined(NRB_TAIL_ORDER_PROBE)
    const bool flash_tagged=decision.evidence==NRB_EVIDENCE_XESS_API ||
        decision.evidence==NRB_EVIDENCE_FSR3_API;
    flash.sr_slot=flash_tagged || (handle->Id>=1000000 && handle->Id<2000000);
    flash.eligible=diagnostic_sr_scope && flash.sr_slot;
    flash.effective_route=unsigned(diagnostic_dlss_scope?NRB_ROUTE_DLSS:route);
#endif
    if(flash.eligible) nrb::flashdiag::recorder.count(nrb::flashdiag::Counter::eligible);
    nrb::flashdiag::recorder.observe(flash);
    bool flash_inline_nr=false;
    auto flash_fallback_reason=nrb::flashdiag::Reason::source_state;
    const auto call_original=[&]() {
#ifdef NRB_DEFERRED_IDENTITY_TEST
        NVSDK_NGX_Result deferred_result=NVSDK_NGX_Result_Fail;
        if(route==NRB_ROUTE_XESS && nrb::deferred_identity_tail_verified(route)) {
            static std::atomic<unsigned> native_gate_logs{0};
            if(native_gate_logs.fetch_add(1,std::memory_order_relaxed)<8) {
                char gate_line[180];
                sprintf_s(gate_line,
                    "xess_native_gate scope=%u source=%u generation=%llu\r\n",
                    unsigned(diagnostic_sr_scope),unsigned(identity_source_state_proven),
                    static_cast<unsigned long long>(identity_source_generation));
                log_line(gate_line);
            }
        }
        // The outer Evaluate path already holds state_mutex here. A nested
        // NRB_GetControls call would deadlock the game's render thread.
        {
            static std::atomic<unsigned> probe_logs{0};
            if(probe_logs.fetch_add(1,std::memory_order_relaxed)<3) {
                char pl[160];
                sprintf_s(pl,"[nrvk] deferred probe profile=%d sr_scope=%d proven=%d route=%u\r\n",
                    g_tail_profile,diagnostic_sr_scope?1:0,
                    identity_source_state_proven?1:0,unsigned(route));
                log_line(pl);
            }
        }        const bool deferred_allowed = identity_source_state_proven ||
            (g_tail_profile>=1 && diagnostic_sr_scope);
        if(g_tail_profile>=1 && !identity_source_state_proven) {
            static std::atomic<bool> deferred_said{false};
            if(!deferred_said.exchange(true))
                log_line("[nrvk] tail_profile=1: recording deferred before the state proof exists\r\n");
        }
        if(diagnostic_sr_scope && deferred_allowed &&
           nrb::record_deferred_identity(list,handle,params,callback,original,
                                         identity_source_generation,controls,
                                         diagnostic_dlss_scope?NRB_ROUTE_DLSS:route,
                                         diagnostic_dlss_scope?NRB_EVIDENCE_STREAMLINE_NGX:
                                             decision.evidence,
                                         deferred_result,&flash)) {
#ifdef NRB_TAIL_ORDER_PROBE
            nrb::tail_after_evaluate(list);
#endif
            static std::atomic<unsigned> deferred_record_logs{0};
            if(deferred_record_logs.fetch_add(1,std::memory_order_relaxed)<8) {
                char line[160];
                sprintf_s(line,"deferred_identity_recorded count=%u retired=%u\r\n",
                    nrb::deferred_identity_recorded(),
                    nrb::deferred_identity_retired());
                log_line(line);
            }
            return deferred_result;
        }
        if(diagnostic_sr_scope && identity_source_state_proven)
            flash_fallback_reason=flash.record_reason?
                nrb::flashdiag::Reason(flash.record_reason):nrb::flashdiag::Reason::record_rejected;
#endif
#ifdef NRB_XESS_IDENTITY_TEST
        ID3D12Resource* source=nullptr;
        ID3D12Resource* copy=nullptr;
        if(diagnostic_dlss_scope && identity_source_state_proven &&
           params->Get(NVSDK_NGX_Parameter_Color,&source)==NVSDK_NGX_Result_Success && source) {
            copy=nrb::record_ngx_identity(list,source);
            if(copy) params->Set(NVSDK_NGX_Parameter_Color,copy);
        }
        if(diagnostic_dlss_scope &&
           identity_relay_logs.fetch_add(1,std::memory_order_relaxed)<8) {
            char line[256];
            sprintf_s(line,"ngx_identity_relay copied=%d state_proven=%d recorded=%u retired=%u\r\n",
                int(copy!=nullptr),int(identity_source_state_proven),
                nrb::xess_identity_recorded(),nrb::xess_identity_completed());
            log_line(line);
        }
#endif
        flash.stage_bits|=nrb::flashdiag::bit(nrb::flashdiag::Stage::original_called);
        const auto rc=original(list,handle,params,callback);
        if(!flash_inline_nr) {
            const auto why=flash.eligible?flash_fallback_reason:
                nrb::flashdiag::Reason::scope_or_feature;
            nrb::flashdiag::recorder.fallback(flash,why,unsigned(rc));
            nrb::flashdiag::recorder.finish(flash,why);
        }
#ifdef NRB_TAIL_ORDER_PROBE
        nrb::tail_after_evaluate(list);
#endif
#ifdef NRB_XESS_IDENTITY_TEST
        if(copy) params->Set(NVSDK_NGX_Parameter_Color,source);
#endif
        return rc;
    };
    if (decision.evidence==NRB_EVIDENCE_EXTERNAL_UNTAGGED) {
        static std::atomic<bool> logged_external{false};
        if (!logged_external.exchange(true,std::memory_order_relaxed)) {
            HMODULE module=nullptr;
            char path[MAX_PATH]{};
            if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                    GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                    static_cast<LPCSTR>(caller), &module) &&
                GetModuleFileNameA(module,path,MAX_PATH)) {
                char line[MAX_PATH+48]{};
                sprintf_s(line,"external NGX caller module=%s\r\n",path);
                log_line(line);
            }
        }
    }
    // OptiScaler's native DLSS and DLSSG IDs are not our XeSS SR slot.
    // FSR/XeSS IDs are independent: only an actual outer API tag can promote
    // one to a candidate. Unknown high IDs stay out of the FG path.
    const bool tagged_other_sr=
        decision.evidence==NRB_EVIDENCE_XESS_API ||
        decision.evidence==NRB_EVIDENCE_FSR3_API;
    if ((!tagged_other_sr && handle->Id < 1000000) ||
        (!tagged_other_sr && handle->Id >= 2000000 &&
         (route==NRB_ROUTE_DLSS || route==NRB_ROUTE_UNKNOWN))) {
        last_status.store(NRB_BYPASS_UNSUPPORTED, std::memory_order_relaxed);
        auto rc = call_original();
        inside = false;
        return rc;
    }
    counts[route].fetch_add(1, std::memory_order_relaxed);
    std::lock_guard guard(state_mutex);
    NRB_Frame frame{};
    nrb::extract_ngx(params, route, list, frame);
    {
        static std::atomic<bool> host_color_logged{false};
        if(!host_color_logged.exchange(true,std::memory_order_acq_rel) &&
           frame.color && frame.output) {
            const auto color_desc=frame.color->GetDesc();
            const auto output_desc=frame.output->GetDesc();
            const bool supported=color_desc.Format==DXGI_FORMAT_R16G16B16A16_FLOAT ||
                color_desc.Format==DXGI_FORMAT_R11G11B10_FLOAT;
            char line[224];
            sprintf_s(line,
                "host_color_dxgi=%u output_dxgi=%u color_supported=%d; supported=10(RGBA16F),26(R11G11B10)\r\n",
                unsigned(color_desc.Format),unsigned(output_desc.Format),int(supported));
            log_line(line);
        }
    }
    flash.enabled=controls.enabled;flash.history=unsigned(controls.history);flash.controls_known=1;
    flash.game_reset=frame.reset_history!=0;
    // Resource/list ordering is useful even when the external NGX caller has
    // not yet been classified. This probe cannot promote an unknown route or
    // authorize NR; the processor still sees the original fail-closed route.
    if ((route==NRB_ROUTE_DLSS || route==NRB_ROUTE_FSR ||
         route==NRB_ROUTE_XESS || route==NRB_ROUTE_UNKNOWN) &&
        frame.color && frame.motion && frame.output) {
        schedule_list_hooks(list);
#ifdef NRB_DLSS_COLOR_STATE_PROOF_V1
        auto** source_methods=*reinterpret_cast<void***>(list);
        const bool color_implementation_known=list_hook_state.load(std::memory_order_acquire)==2 &&
            source_methods[10]==reinterpret_cast<void*>(first_list_methods.reset) &&
            source_methods[9]==reinterpret_cast<void*>(first_list_methods.close) &&
            source_methods[26]==reinterpret_cast<void*>(first_list_methods.barriers);
        if(diagnostic_dlss_scope && sync_probe.color_state_enabled() && !color_implementation_known)
            sync_probe.unknown_recording(list);
#endif
        auto sync=sync_probe.evaluate(list,frame.color,frame.motion,diagnostic_dlss_scope);
#ifdef NRB_TAIL_ORDER_PROBE
        if(sync.reset_observed && !sync.closed &&
           (diagnostic_dlss_scope || trusted_other_sr_scope))
            nrb::tail_before_evaluate(list,frame.color,frame.output,
                diagnostic_dlss_scope?NRB_ROUTE_DLSS:route);
#endif
#ifdef NRB_XESS_IDENTITY_TEST
#ifdef NRB_DLSS_COLOR_STATE_PROOF_V1
        const bool dlss_source_proven=nrb::dlss_color_processing_source(sync,color_implementation_known,color_state_shadow_mode);
#else
        const bool dlss_source_proven=nrb::legacy_dlss_color_source(sync);
#endif
        // Native routes are deferred until the entire producer list has
        // executed. Their eight exact tail samples, rather than a preceding
        // color barrier on this list, establish the state of the color copy.
        // The same list-scoped tail proof is valid for the DLSS route: the
        // legacy color-age gate expires on unrelated command traffic, which is
        // the periodic-flash source. Prefer the tail proof when it is verified.
        bool native_deferred_source_proven=false;
#ifdef NRB_DEFERRED_IDENTITY_TEST
        native_deferred_source_proven=
            (trusted_other_sr_scope || diagnostic_dlss_scope) &&
            sync.reset_observed && !sync.closed &&
            nrb::deferred_identity_tail_verified(route);
#endif
        identity_source_state_proven=dlss_source_proven || native_deferred_source_proven;
        identity_source_generation=sync.generation;
#endif
        const int hook_state=list_hook_state.load(std::memory_order_acquire);
        if(hook_state==2) {
            auto** methods=*reinterpret_cast<void***>(list);
            if(methods[10]!=reinterpret_cast<void*>(first_list_methods.reset) ||
               methods[26]!=reinterpret_cast<void*>(first_list_methods.barriers))
                sync.incomplete=true; // another command-list implementation
        } else sync.incomplete=true;
        flash.generation=sync.generation;
        flash.evaluate_sequence=sync.evaluate_sequence;flash.color_sequence=sync.color.sequence;
        flash.color_after=unsigned(sync.color.after);
        flash.color_age=sync.color.seen && sync.evaluate_sequence>sync.color.sequence?
            static_cast<uint32_t>(std::min<uint64_t>(UINT32_MAX,
                sync.evaluate_sequence-sync.color.sequence)):UINT32_MAX;
        flash.source_bits=unsigned(sync.reset_observed) |
            (unsigned(sync.closed)<<1) | (unsigned(sync.oversized_barrier)<<2) |
            (unsigned(sync.color.seen)<<3) |
            (unsigned(sync.color.flags==D3D12_RESOURCE_BARRIER_FLAG_NONE)<<4) |
            (unsigned(sync.color.subresource==D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES)<<5) |
            (unsigned(sync.color.after==D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE)<<6) |
            (unsigned(flash.color_age<=64)<<7) | (unsigned(sync.incomplete)<<9);
#ifdef NRB_DLSS_COLOR_STATE_PROOF_V1
        // Additive source-mask fields; the frozen v2 buffer ABI is unchanged.
        if(sync.color_state.enabled) flash.source_bits|=(1u<<10) |
            (unsigned(sync.color_state.known)<<11) | (unsigned(color_state_shadow_mode)<<12) |
            (unsigned(sync.color_state.reason)<<16);
#endif
#ifdef NRB_XESS_IDENTITY_TEST
        flash.source_bits|=unsigned(identity_source_state_proven)<<8;
        if(flash.eligible && identity_source_state_proven)
            nrb::flashdiag::recorder.count(nrb::flashdiag::Counter::source_proven);
#endif
#ifdef NRB_ONE_FRAME_COLOR_READBACK
        nrb::one_frame_readback_record(list,frame.color,sync,log_line);
#endif
        // Keep a few early unknown samples, then reserve the bounded log for
        // generations whose successful Reset was actually observed.
        static std::atomic<uint32_t> unknown_logs{0};
        const bool report=sync.reset_observed?
            sync_eval_logs.fetch_add(1,std::memory_order_relaxed)<16:
            unknown_logs.fetch_add(1,std::memory_order_relaxed)<4;
        if(report) {
            char line[1000];
            sprintf_s(line,"{\"event\":\"sync_boundary\",\"list\":\"%p\",\"hook_state\":%d,\"generation\":%llu,\"evaluate_sequence\":%llu,\"reset_observed\":%d,\"closed\":%d,\"incomplete\":%d,\"submit_batch_before\":%llu,\"color\":{\"resource\":\"%p\",\"barrier_seen\":%d,\"sequence\":%llu,\"before\":%u,\"after\":%u,\"flags\":%u,\"subresource\":%u,\"full_non_pixel_read\":%d},\"motion\":{\"resource\":\"%p\",\"barrier_seen\":%d,\"sequence\":%llu,\"before\":%u,\"after\":%u,\"flags\":%u,\"subresource\":%u,\"full_non_pixel_read\":%d},\"source_submission_proven\":false,\"conclusion\":\"unknown\"}\r\n",
                static_cast<void*>(list),hook_state,sync.generation,sync.evaluate_sequence,
                int(sync.reset_observed),int(sync.closed),int(sync.incomplete),sync.submit_batch_before,
                static_cast<void*>(frame.color),int(sync.color.seen),sync.color.sequence,
                unsigned(sync.color.before),unsigned(sync.color.after),unsigned(sync.color.flags),
                sync.color.subresource,int(sync.color.full_non_pixel_read),
                static_cast<void*>(frame.motion),int(sync.motion.seen),sync.motion.sequence,
                unsigned(sync.motion.before),unsigned(sync.motion.after),unsigned(sync.motion.flags),
                sync.motion.subresource,int(sync.motion.full_non_pixel_read));
            log_line(line);
        }
    }
    if(flash.eligible && flash.game_reset) {
        nrb::flashdiag::recorder.count(nrb::flashdiag::Counter::game_reset);
        nrb::flashdiag::recorder.event(nrb::flashdiag::EventKind::reset,flash);
    }
    nrb::flashdiag::recorder.progress(flash);
    frame.route_evidence=decision.evidence;
    frame.frame_id = ++frame_id;
    // The outer Evaluate path holds state_mutex here, so the panel values
    // are read from lock-free atomics instead of taking the mutex again.
    NRB_Controls frame_controls=controls;
    if(panel::live_valid.load(std::memory_order_acquire)) {
        frame_controls.enabled=panel::live_enabled.load(std::memory_order_relaxed);
        frame_controls.style=panel::live_style.load(std::memory_order_relaxed);
        frame_controls.auto_mask=panel::live_auto_mask.load(std::memory_order_relaxed);
        frame_controls.display_strength=panel::live_display.load(std::memory_order_relaxed);
        frame_controls.model_intensity=panel::live_intensity.load(std::memory_order_relaxed);
        frame_controls.local_tone=panel::live_local_tone.load(std::memory_order_relaxed);
        frame_controls.local_structure=panel::live_local_structure.load(std::memory_order_relaxed);
        frame_controls.skin_structure=panel::live_skin.load(std::memory_order_relaxed);
    }
    frame.controls = frame_controls;
    frame.control_epoch = control_epoch;
    frame.reset_history |= frame_controls.history == NRB_HISTORY_RESET ||
        last_processed_id + 1 != frame.frame_id || last_processed_epoch != control_epoch ||
        last_width != frame.render_width || last_height != frame.render_height || last_route != route;
    auto status = nrb::validate_metadata(frame);
    // FSR.frameTimeDelta and XeSS optional markers are useful diagnostics,
    // not cross-game proof of the external API entry. Only the game's/SL's
    // direct NGX call currently establishes the DLSS route for processing.
    if (route == NRB_ROUTE_UNKNOWN ||
        decision.evidence == NRB_EVIDENCE_FSR_MARKER ||
        decision.evidence == NRB_EVIDENCE_XESS_MARKER)
        status=NRB_BYPASS_UNKNOWN_ROUTE;
    QueueSnapshot observed{};
    // Collect real resource descriptors even when HDR or another metadata
    // contract makes NR ineligible. The mock test has no resources and skips
    // COM inspection entirely.
    if (frame.color && frame.motion && frame.output) {
        ID3D12Device* d=nullptr;
        if (FAILED(list->GetDevice(IID_PPV_ARGS(&d)))) status=NRB_BYPASS_UNSUPPORTED;
        else {
            frame.device=d;
            observed=queue_snapshot(d,list);
            if (!queue && observed.selected) {
                queue=observed.selected; observed.selected=nullptr;
            }
            if (queue && observed.selected && queue!=observed.selected)
                observed.meta.same_device=false;
            if (observed.selected) observed.selected->Release();
        }
    }
    frame.queue=queue;
    if (frame.device && queue) {
        ID3D12Device* q_device=nullptr;
        if(FAILED(queue->GetDevice(IID_PPV_ARGS(&q_device))) || q_device!=frame.device)
            observed.meta.same_device=false;
        if(q_device) q_device->Release();
    }
    nrb::ResourceMeta color_meta{},motion_meta{},depth_meta{},exposure_meta{},output_meta{};
    if (frame.device) {
        const auto resource_meta=[](ID3D12Resource* r) {
            nrb::ResourceMeta meta{};
            if(r) {const auto desc=r->GetDesc();meta.format=static_cast<uint32_t>(desc.Format);
                meta.width=static_cast<uint32_t>(desc.Width);meta.height=desc.Height;}
            return meta;
        };
        color_meta=resource_meta(frame.color);
        motion_meta=resource_meta(frame.motion);
        depth_meta=resource_meta(frame.depth);
        exposure_meta=resource_meta(frame.exposure);
        output_meta=resource_meta(frame.output);
    }
    bool synced=!borrowed.color || sync_queue();
    if (status == NRB_OK && !frame_controls.enabled) status = NRB_BYPASS_DISABLED;
    // [vk102 integration] Fail closed until the required nrvk processor has
    // registered. Never let the built-in Python/mixed processor process frames.
    if (status == NRB_OK && !nrvk_bridge_mounted.load(std::memory_order_acquire)) {
        status = nrvk_mount_failed.load(std::memory_order_acquire) ? NRB_PROCESS_ERROR : NRB_BYPASS_RUNTIME_MISSING;
        static std::atomic<bool> gate_logged{false};
        if (!gate_logged.exchange(true,std::memory_order_acq_rel))
            log_line(nrvk_mount_failed.load(std::memory_order_acquire)
                ? "[nrvk] required backend unavailable; frames fail closed (no fallback)\r\n"
                : "[nrvk] waiting for required backend mount; frames bypassed (no fallback)\r\n");
    }
    if (status == NRB_OK && (!processor.process || !processor.retire)) status = NRB_BYPASS_DISABLED;
    if (status == NRB_OK && built_in_processor && !runtime_assets_available)
        status = NRB_BYPASS_RUNTIME_MISSING;
    if (status == NRB_OK && processor_failed) status=NRB_PROCESS_ERROR;
    if (status == NRB_OK && !synced) status=NRB_BYPASS_UNSYNCED;
    if (status == NRB_OK) status = nrb::validate_resources(frame);
    // The Python/RE8 contract only applies to the built-in Python processor.
    // With the Vulkan component mounted that path is dead (fail-closed), and
    // enforcing the RE8 adapter plan would reject every non-Cyberpunk engine.
    if (status == NRB_OK && !nrvk_bridge_mounted.load(std::memory_order_acquire))
        status=nrb::validate_python_contract(frame);
    // Production gate (profile 0): a registered queue ordinal plus the v5 source
    // proof. Profile 1 substitutes the tail proof for engines whose lists never
    // register an ordinal: the SR output must have been transitioned, read, and
    // never copied over (see tail_safe_generic).
    bool order_proven;
    if (g_tail_profile >= 1) {
        // Bootstrap for engines without a v5 proof: the tail counters only grow
        // once the deferred path runs, which itself needs this gate. Requiring
        // the proof here would deadlock forever, so profile 1 trusts the queue
        // binding and lets validate_resources plus the component's own contract
        // reject bad frames.
        order_proven = queue && observed.meta.same_device;
        static std::atomic<bool> said{false};
        if (!said.exchange(true))
            log_line("[nrvk] tail_profile=1: order proof relaxed to queue+device for this game\r\n");
    } else {
        order_proven = queue && observed.meta.ordinal && observed.meta.same_device &&
                       source_submission_proven.load(std::memory_order_acquire);
    }
    if (status == NRB_OK && !order_proven) status = NRB_BYPASS_UNSYNCED;
    if (status == NRB_OK && !sync_queue()) status=NRB_PROCESS_ERROR;
    NRB_Result result{};
    result.abi_size = sizeof(result);
    double process_ms=0.0;
    if (status == NRB_OK) {
        const auto started=std::chrono::steady_clock::now();
        try {
            if (processor.process(processor.context,&frame,&result)!=0 ||
                !nrb::validate_result(frame,result) ||
                FAILED(queue->Wait(result.ready_fence,result.ready_value)))
                status=NRB_PROCESS_ERROR;
        } catch (...) {status=NRB_PROCESS_ERROR;}
        process_ms=std::chrono::duration<double,std::milli>(
            std::chrono::steady_clock::now()-started).count();
        if(status!=NRB_OK) processor_failed=true;
        if(status!=NRB_OK && result.color) {
            // Keep the borrowed output pinned: its GPU work or consumer order
            // is unknown, so it is unsafe to recycle it into a later frame.
            borrowed=result;
            consumer_list=nullptr;
        }
    }
    // [panel] A component that was just swapped in and errors before its
    // first successful frame is unusable; the panel worker rolls back.
    if(panel::probe_state.load(std::memory_order_relaxed)==1) {
        if(status==NRB_OK) panel::probe_state.store(0,std::memory_order_relaxed);
        else if(status==NRB_PROCESS_ERROR)
            panel::probe_state.store(3,std::memory_order_relaxed);
    }
    if(frame.device) frame.device->Release();
    const uint32_t sample=diagnostic_gate.claim(route);
    if (sample) {
        const auto line=nrb::format_diagnostic(sample,frame,color_meta,motion_meta,
            depth_meta,exposure_meta,output_meta,observed.meta,status);
        log_line(line.c_str());
    }
    {
        // One line per status change: the frame diagnostic only samples the
        // first frames, which can all precede the component mount.
        static std::atomic<int> last_logged{-1};
        if(last_logged.load(std::memory_order_relaxed)!=int(status)) {
            char sl[128];
            sprintf_s(sl,"bypass_changed status=%d source=%ux%u output=%ux%u\r\n",
                int(status),frame.render_width,frame.render_height,
                frame.output_width,frame.output_height);
            log_line(sl);
            last_logged.store(int(status),std::memory_order_relaxed);
        }
    }
    last_status.store(status, std::memory_order_relaxed);
    NVSDK_NGX_Result rc;
    if (status == NRB_OK) {
        flash_inline_nr=true; // current result only; never count it as raw fallback
        params->Set(NVSDK_NGX_Parameter_Color, result.color);
        borrowed = result;
        consumer_list=list;
        {std::lock_guard queue_guard(queue_mutex);consumer_submit_before=submit_serial;}
        consumer_queue_ordinal=observed.meta.ordinal;
        rc = call_original();
        params->Set(NVSDK_NGX_Parameter_Color, frame.color);
        if (rc == NVSDK_NGX_Result_Success) {
            last_processed_id = frame.frame_id;
            last_processed_epoch = control_epoch;
            last_width = frame.render_width;
            last_height = frame.render_height;
            last_route = route;
            processed_count.fetch_add(1,std::memory_order_relaxed);
            if (process_sample_count==process_samples.size())
                process_sample_sum-=process_samples[process_sample_cursor];
            else ++process_sample_count;
            process_samples[process_sample_cursor]=process_ms;
            process_sample_sum+=process_ms;
            process_sample_cursor=(process_sample_cursor+1)%process_samples.size();
            last_process_ms.store(process_ms,std::memory_order_relaxed);
            average_process_ms.store(process_sample_sum/process_sample_count,
                                     std::memory_order_relaxed);
        } else {
            last_processed_id = 0;
            last_status.store(NRB_PROCESS_ERROR,std::memory_order_relaxed);
        }
    } else {
        rc = call_original();
    }
    inside = false;
    return rc;
}
}

NRB_API int NRB_RegisterProcessor(const NRB_Processor* p) {
    std::lock_guard guard(state_mutex);
    if (borrowed.color) return 0;
    if (!p) { processor = {}; built_in_processor=false; processor_failed=false;
        nrb::gpuhandoff::processor_changed(nullptr);return 1; }
    if (p->abi_size != sizeof(NRB_Processor) || !p->process || !p->retire) return 0;
    processor = *p;
    built_in_processor=p->context==&python_processor;
    processor_failed=false;
    nrb::gpuhandoff::processor_changed(p);
    return 1;
}
NRB_API int NRB_RegisterQueue(ID3D12CommandQueue* q) {
    if(q) observe_queue(q);
    std::lock_guard guard(state_mutex);
    if (borrowed.color || (q && q->GetDesc().Type != D3D12_COMMAND_LIST_TYPE_DIRECT)) return 0;
    if (q) q->AddRef();
    if (queue) queue->Release();
    queue = q;
    nrb::gpuhandoff::queue_changed();
    if (completion) { completion->Release(); completion = nullptr; }
    if (completion_event) { CloseHandle(completion_event); completion_event = nullptr; }
    completion_value = 0;
    return 1;
}
NRB_API void NRB_SetProducerReady(int ready) {
    // Kept for old callers. A Boolean contains no resource/queue/frame proof.
    static std::atomic<bool> warned=false;
    if(ready && !warned.exchange(true))
        log_line("NRB_SetProducerReady ignored: source submission proof required\r\n");
}
NRB_API int NRB_SetControls(const NRB_Controls* c) {
    if (!c || !nrb::validate_controls(*c)) return 0;
    std::lock_guard guard(state_mutex);
    controls = *c;
    ++control_epoch;
    process_sample_count=process_sample_cursor=0;
    process_sample_sum=0.0;
    last_process_ms.store(0.0,std::memory_order_relaxed);
    average_process_ms.store(0.0,std::memory_order_relaxed);
#ifdef NRB_LIVE_NR_TEST
    nrb::deferred_nr_reset_metrics();
#endif
    return 1;
}
NRB_API int NRB_GetControls(NRB_Controls* c) {
    if (!c || c->abi_size != sizeof(NRB_Controls)) return 0;
    std::lock_guard guard(state_mutex);
    *c = controls;
    return 1;
}
// Read-only, fixed-size CPU snapshot. No file write or GPU synchronization.
NRB_API int NRB_GetPeriodicFlashDiagV2(nrb::flashdiag::Snapshot* snapshot) {
    return nrb::flashdiag::recorder.snapshot(snapshot)?1:0;
}
NRB_API int NRB_GetPeriodicFlashContextV2(nrb::flashdiag::Context* context) {
    return nrb::flashdiag::current_context(context)?1:0;
}
NRB_API NRB_Status NRB_LastStatus() {
#ifdef NRB_LIVE_NR_TEST
    if(nrb::deferred_nr_failed()) return NRB_PROCESS_ERROR;
    if(nrb::deferred_nr_processed()) {
        std::lock_guard guard(state_mutex);
        return controls.enabled?NRB_OK:NRB_BYPASS_DISABLED;
    }
#endif
    return last_status.load(std::memory_order_relaxed);
}
NRB_API uint64_t NRB_InterceptCount(NRB_Route r) {
    return r <= NRB_ROUTE_XESS ? counts[r].load(std::memory_order_relaxed) : 0;
}
NRB_API uint64_t NRB_ProcessedCount() {
#ifdef NRB_LIVE_NR_TEST
    return processed_count.load(std::memory_order_relaxed)+nrb::deferred_nr_processed();
#else
    return processed_count.load(std::memory_order_relaxed);
#endif
}
NRB_API double NRB_LastProcessMs() {
#ifdef NRB_LIVE_NR_TEST
    if(!nrb::deferred_nr_timing_enabled()) return 0;
    const double t=nrb::deferred_nr_last_ms();
    if(t>0) return t;
#endif
    return last_process_ms.load(std::memory_order_relaxed);
}
NRB_API double NRB_AverageProcessMs() {
#ifdef NRB_LIVE_NR_TEST
    if(!nrb::deferred_nr_timing_enabled()) return 0;
    const double t=nrb::deferred_nr_average_ms();
    if(t>0) return t;
#endif
    return average_process_ms.load(std::memory_order_relaxed);
}
NRB_API int NRB_GetStageTimes(NRB_StageTimes* times) {
#ifdef NRB_DEFERRED_IDENTITY_TEST
    return nrb::deferred_nr_stage_times(times) ? 1 : 0;
#else
    if(!times || times->abi_size!=sizeof(NRB_StageTimes)) return 0;
    *times=NRB_StageTimes{};
    times->abi_size=sizeof(NRB_StageTimes);
    return 0;
#endif
}
NRB_API int NRB_GetRecordTimes(NRB_RecordTimes* times) {
#ifdef NRB_DEFERRED_IDENTITY_TEST
    return nrb::deferred_nr_record_times(times)?1:0;
#else
    if(!times || times->abi_size!=sizeof(NRB_RecordTimes)) return 0;
    *times=NRB_RecordTimes{};times->abi_size=sizeof(NRB_RecordTimes);return 0;
#endif
}
NRB_API int NRB_GetHdrCacheStats(NRB_HdrCacheStats* stats) {
#ifdef NRB_LIVE_NR_TEST
    return nrb::deferred_hdr_cache_stats(stats)?1:0;
#else
    if(!stats || stats->abi_size!=sizeof(NRB_HdrCacheStats)) return 0;
    *stats=NRB_HdrCacheStats{};stats->abi_size=sizeof(NRB_HdrCacheStats);return 0;
#endif
}
NRB_API int NRB_SetHdrCacheEnabled(int enabled) {
#ifdef NRB_LIVE_NR_TEST
    if(enabled!=0 && enabled!=1) return 0;
    nrb::deferred_hdr_set_cache_enabled(enabled!=0);return 1;
#else
    (void)enabled;return 0;
#endif
}
NRB_API int NRB_GetTimingEnabled() {
#ifdef NRB_LIVE_NR_TEST
    return nrb::deferred_nr_timing_enabled()?1:0;
#else
    return 0;
#endif
}
NRB_API int NRB_SetTimingEnabled(int enabled) {
#ifdef NRB_LIVE_NR_TEST
    if(enabled!=0 && enabled!=1) return 0;
    nrb::deferred_nr_set_timing_enabled(enabled!=0);
    return 1;
#else
    (void)enabled;
    return 0;
#endif
}

namespace {
DWORD WINAPI initialize_worker(void*) {
    // OptiScaler calls InitializeASI from its DLL_PROCESS_ATTACH. Keep all
    // Detours, file I/O and Python initialization outside the loader lock.
    try {
    NRB_Processor built_in{sizeof(NRB_Processor),&python_processor,
        nrb::PythonProcessor::process_callback,nrb::PythonProcessor::retire_callback};
#ifdef NRVK_REQUIRED
    (void)built_in;
    log_line("NRVK required; built-in Python processor intentionally not registered; no fallback\r\n");
#else
    NRB_RegisterProcessor(&built_in);
#endif
#ifdef NRB_LIVE_NR_TEST
    {std::lock_guard guard(state_mutex);
        controls.enabled=1;controls.input_height=720;
        controls.history=NRB_HISTORY_FUSED;controls.graph_replay=1;}
#endif
#ifdef NRB_LIVE_NR_TEST
    if(!nrb::deferred_hdr_warmup_shader())
        log_line("HDR shader cache warmup unavailable; first frame will retry\r\n");
#endif
    runtime_assets_available=python_processor.assets_available();
    if(!runtime_assets_available) log_line("Portable RE8 runtime assets missing; processor registered but disabled\r\n");
    nrb::deferred_identity_set_log(log_line);
    load_bridge_config(); // Resolve shared paths before hooks/panel threads use them.
    install_queue_capture();
    nrvk_try_mount();
    panel::start();
    const wchar_t* names[] = {L"OptiScaler.dll", L"dxgi.dll", L"version.dll",
        L"winmm.dll", L"winhttp.dll", L"OptiScaler.asi"};
    for (auto name : names) {
        if (original) break;
        HMODULE candidate = GetModuleHandleW(name);
        if (!candidate) continue;
        auto address = GetProcAddress(candidate, "NVSDK_NGX_D3D12_EvaluateFeature");
        if (address) { host = candidate; original = reinterpret_cast<Evaluate>(address); break; }
    }
    hook_device_export(host,original_create_host,observed_create_host);
    if (!original) {
        log_line("ASI loaded; OptiScaler NGX evaluator export missing\r\n");
        init_state.store(3, std::memory_order_release);
        return 0;
    }
    if (!attach_detour(reinterpret_cast<void**>(&original),reinterpret_cast<void*>(intercept))) {
        original = nullptr;
        log_line("ASI loaded; NGX evaluator detour failed\r\n");
        init_state.store(3, std::memory_order_release);
    } else {
        install_sl_feature_probe();
        if(HANDLE worker=CreateThread(nullptr,0,install_upscaler_api_probes,
                                      nullptr,0,nullptr)) CloseHandle(worker);
        log_line("ASI loaded; pre-SR NGX evaluator detour attached\r\n");
        init_state.store(2, std::memory_order_release);
        // Controls must be reachable even while every frame is bypassed.
        if (runtime_assets_available && !python_processor.start_controls())
            log_line("Portable NR control page startup failed\r\n");
    }
    } catch (...) {
        log_line("ASI deferred initialization threw an exception\r\n");
        init_state.store(3, std::memory_order_release);
    }
    return 0;
}
}

extern "C" __declspec(dllexport) void InitializeASI() {
    int expected=0;
    if (!init_state.compare_exchange_strong(expected,1,std::memory_order_acq_rel)) return;
    HANDLE worker=CreateThread(nullptr,0,initialize_worker,nullptr,0,nullptr);
    if (worker) CloseHandle(worker);
    else init_state.store(3,std::memory_order_release);
}
NRB_API int NRB_InitState() { return init_state.load(std::memory_order_acquire); }

BOOL WINAPI DllMain(HINSTANCE, DWORD, LPVOID) { return TRUE; }
