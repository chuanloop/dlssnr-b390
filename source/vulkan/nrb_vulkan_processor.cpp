// nrb_vulkan_processor.cpp — 独立 Vulkan FP16 NR 后端处理器（v3：接入版完整提交路径）
// 契约：include/nr_bridge.h v1（冻结）；设计：INTERFACE-CONTRACT.md v3。
//
// 逐帧时序（[GPU 路径 1-4] 对齐 T5 已验证配方，native720 真实两帧票同款）：
//   [桥] sync_queue()：CPU 等待游戏 queue 完成（asi.cpp:1093-1106）→ 输入 GPU 完备
//   [组件 D3D12] 自建 allocator/list：CopyTextureRegion 游戏→shared staging（格式原样）
//                游戏 queue ExecuteCommandList(拷贝) → 同队 Signal(f_cp, v)
//                → D3D12 copy_fence 经 copy_ready timeline 导入 Vulkan，vkQueueSubmit 端等待（无 f_cp CPU 等待）
//   [组件 Vulkan] submit{ [acquire EXTERNAL→family] → [gather+网络+scatter]
//                → [release family→EXTERNAL] }，signal(槽导出 timeline, sv),
//                fence=槽 cb_fence —— 输入无 wait 信号量；同步在 submit 组装（S2）
//   [输出] Vk timeline EXPORTABLE(OPAQUE_WIN32) → GetSemaphoreWin32HandleKHR
//                → D3D12 OpenSharedHandle → 宿主 queue->Wait(sv)（ready 值单调）
//   [桥] 下一帧 sync_queue() 后 retire(fence, v) → 槽归还
//   [S1] 槽复用前：vkGetFenceStatus(槽 cb_fence)（GPU 完成证据）→ reset cb
#include "nrb_vulkan_processor.h"
#include "port_network_bridge_backend.h"
#include "nr_bridge.h"
#include <d3d12.h>
#include <dxgi1_4.h>

#include <cstdio>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#define NRVK_LOG(...) do { if (g_ctx && g_ctx->cfg.log_verbose) fprintf(stderr, "[nrvk] " __VA_ARGS__); } while (0)

// ------------------------------------------------------------- 后端适配层（T5 真实链接）
// 固定包：102 expand-mid-direct-store（plan SHA 1ba536a5…，266 发/487 buffers）。
// 组件 DLL 与独立包内 backend LIB 精确链接；四占位 stub 已移除，强符号由 LIB 提供。

struct NrvkDiagFrameRecord {
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

namespace {

struct NrvkDiagPendingSlot {
    bool active = false;
    uint64_t copy_fence_val = 0;
    VkFence cb_fence = VK_NULL_HANDLE;
    NrvkDiagFrameRecord rec {};
};

struct ImportedImg {
    ID3D12Resource* res = nullptr;
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;   // images[] 元数据（后端 frame_supported 校验）
    DXGI_FORMAT dxgi_format = DXGI_FORMAT_UNKNOWN;
    uint32_t width = 0, height = 0;
};

struct OutputSlot {                        // 槽 = 输出纹理 + ready fence + (cb, cb_fence) 对
    ID3D12Resource* res = nullptr;
    ID3D12Fence* fence = nullptr;          // ready fence = Vk 导出 timeline 的 D3D12 侧视图
    uint64_t fence_value = 0;
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkSemaphore sem = VK_NULL_HANDLE;      // Vk 原生 timeline（EXPORTABLE OPAQUE_WIN32）
    VkFormat format = VK_FORMAT_UNDEFINED; // 槽实际格式/尺寸（元数据禁止用 SR 目标伪造）
    DXGI_FORMAT dxgi_format = DXGI_FORMAT_UNKNOWN;
    uint32_t width = 0, height = 0;
    VkCommandBuffer cb = VK_NULL_HANDLE;   // [S1] 槽私有 cb
    VkFence cb_fence = VK_NULL_HANDLE;     // [S1] 建为 SIGNALED；复用前 wait
    bool in_flight = false;
    uint64_t returned_seq = 0;             // Global return order, not local ready value.
};

struct StagingGroup {                      // 一组输入 staging（按游戏资源指针组缓存）
    ImportedImg color, motion, depth, exposure;
    uint64_t last_frame = 0;               // frame_id of last use, for bounded LRU eviction
    uint64_t bytes = 0;                    // D3D12 allocation estimate for status/audit
};

struct Ctx {
    NRVK_Config cfg {};
    ID3D12Device* d3d = nullptr;
    NRVK_Backend* backend = nullptr;          // T5 适配层上下文
    VkFence network_fence_pending = VK_NULL_HANDLE;  // 上一帧网络完成 fence（单上下文门控）
    uint64_t frame_seq = 0;
    NRB_RegisterFn host_register = nullptr;   // 宿主显式传入的注册回调（[任务1]）

    HMODULE vulkan_mod = nullptr;
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice phy = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t queue_family = 0;
    VkCommandPool pool = VK_NULL_HANDLE;

    // D3D12 侧拷贝设施（F2/F3）
    ID3D12CommandAllocator* copy_alloc = nullptr;
    ID3D12GraphicsCommandList* copy_list = nullptr;
    ID3D12Fence* copy_fence = nullptr;         // 拷贝完成 fence（CPU 等待，[GPU 路径 3]）
    HANDLE copy_fence_nt = nullptr;
    uint64_t copy_fence_value = 0;
    VkSemaphore copy_ready = VK_NULL_HANDLE; // persistent imported D3D12 copy timeline
    HANDLE copy_event = nullptr;               // copy_fence 完成事件（输入 CPU fence 等待）

    std::map<std::string, StagingGroup> staging;
    uint64_t staging_bytes = 0;               // sum of D3D12 allocations currently held
    std::vector<OutputSlot> slots;
    uint32_t slot_next = 0;
    std::mutex mu;
    bool init_ok = false;

    // 运行状态
    uint64_t frames_ok = 0, frames_bypass = 0, ring_full = 0, converted = 0;
    uint64_t gpu_timeout = 0;              // [S1 v4] 单列：槽 cb_fence 未完成旁路
    double   last_ms = 0, avg_ms = 0;
    uint32_t last_status = 0;
    // Minimal observed contract fields for the existing status JSON. These are
    // measurements of the actual imported/copied resources, not expected values.
    uint32_t in_color_fmt = 0, in_motion_fmt = 0, in_depth_fmt = 0, out_fmt = 0;
    uint32_t in_color_w = 0, in_color_h = 0, in_motion_w = 0, in_motion_h = 0;
    uint32_t in_depth_w = 0, in_depth_h = 0, out_w = 0, out_h = 0;
    uint32_t gather_x = 0, gather_y = 0, gather_z = 0;
    uint32_t scatter_x = 0, scatter_y = 0, scatter_z = 0;
    uint64_t last_frame_id = 0, last_control_epoch = 0;
    uint32_t last_reset = 0, last_history = 0;
    bool history_committed = false;
    uint64_t history_frame = 0, history_epoch = 0;
    VkQueryPool vk_query_pool = VK_NULL_HANDLE;
    double vk_timestamp_period_ns = 0.0;
    uint32_t vk_timestamp_valid_bits = 0;
    ID3D12QueryHeap* d3d_ts_heap = nullptr;
    ID3D12Resource* d3d_ts_readback = nullptr;
    uint64_t d3d_freq_hz = 0;
    uint32_t diag_mode = 0;
    uint32_t diag_window_id = 0;
    uint32_t diag_arm_reuse = 0;
    uint64_t last_cpu_frame_id = 0;
    uint32_t last_cpu_had_fence = 0;
    uint64_t last_cpu_wait_start_qpc = 0;
    uint64_t last_cpu_wait_end_qpc = 0;
    NrvkDiagPendingSlot diag_pending[16] {};
    std::vector<NrvkDiagFrameRecord> diag_records;
    uint64_t diag_d3d_unavail = 0;
    uint64_t diag_vk_unavail = 0;
    uint64_t diag_overflow = 0;
};
Ctx* g_ctx = nullptr;
HMODULE g_vulkan_mod = nullptr;
NRVK_Config g_cfg {};
static PFN_vkGetInstanceProcAddr g_gipa = nullptr;   // 实例级查询用（能力预检/内存属性）

static PFN_vkGetDeviceProcAddr g_gdpa = nullptr;
#define NRVK_DEV_FN(name) \
    static PFN_vk##name pfn_##name = nullptr; \
    if (!g_gdpa) g_gdpa = (PFN_vkGetDeviceProcAddr)GetProcAddress(g_vulkan_mod, "vkGetDeviceProcAddr"); \
    if (!g_gdpa) return -4; \
    if (!pfn_##name) pfn_##name = (PFN_vk##name)g_gdpa(c->device, "vk" #name); \
    if (!pfn_##name) return -4;

bool map_depth(DXGI_FORMAT f) { return f == DXGI_FORMAT_D32_FLOAT; }

bool map_format(DXGI_FORMAT f, VkFormat* out) {
    switch (f) {
        case DXGI_FORMAT_R16G16B16A16_FLOAT: *out = VK_FORMAT_R16G16B16A16_SFLOAT; return true;
        case DXGI_FORMAT_R16G16_FLOAT:       *out = VK_FORMAT_R16G16_SFLOAT;       return true;
        case DXGI_FORMAT_R32G32_FLOAT:       *out = VK_FORMAT_R32G32_SFLOAT;       return true;
        case DXGI_FORMAT_R32_FLOAT:          *out = VK_FORMAT_R32_SFLOAT;          return true;
        case DXGI_FORMAT_D32_FLOAT:          *out = VK_FORMAT_D32_SFLOAT;          return true;
        case DXGI_FORMAT_R32G32B32A32_FLOAT: *out = VK_FORMAT_R32G32B32A32_SFLOAT; return true;
        case DXGI_FORMAT_R10G10B10A2_UNORM:  *out = VK_FORMAT_A2B10G10R10_UNORM_PACK32; return true;
        case DXGI_FORMAT_R11G11B10_FLOAT:    *out = VK_FORMAT_B10G11R11_UFLOAT_PACK32; return true;
        case DXGI_FORMAT_R32_UINT:           *out = VK_FORMAT_R32_UINT;            return true;
        default: return false;
    }
}

#include "nrvk_build_identity.inc"

// Staging inputs are our own D3D12 committed resources plus the Vulkan
// external-memory import of each. They must be destroyed as a unit when a
// scene/resource set is evicted; otherwise the map grows for the whole session.
static void destroy_imported(Ctx* c, ImportedImg& im) {
    if (!c || c->device == VK_NULL_HANDLE || !g_gdpa) return;
    if (im.view) {
        auto f = (PFN_vkDestroyImageView)g_gdpa(c->device, "vkDestroyImageView");
        if (f) f(c->device, im.view, nullptr);
        im.view = VK_NULL_HANDLE;
    }
    if (im.image) {
        auto f = (PFN_vkDestroyImage)g_gdpa(c->device, "vkDestroyImage");
        if (f) f(c->device, im.image, nullptr);
        im.image = VK_NULL_HANDLE;
    }
    if (im.mem) {
        auto f = (PFN_vkFreeMemory)g_gdpa(c->device, "vkFreeMemory");
        if (f) f(c->device, im.mem, nullptr);
        im.mem = VK_NULL_HANDLE;
    }
    if (im.res) { im.res->Release(); im.res = nullptr; }
}
static void destroy_staging_group(Ctx* c, StagingGroup& g) {
    destroy_imported(c, g.color);
    destroy_imported(c, g.motion);
    destroy_imported(c, g.depth);
    destroy_imported(c, g.exposure);
}

void write_status(Ctx* c) {
    if (!c->cfg.status_json_path) return;
    uint32_t grid_w = 0, grid_h = 0;
    nrvk_backend_last_grid(&grid_w, &grid_h);
    if (grid_w && grid_h) c->gather_x = (grid_w * grid_h + 255) / 256;
    FILE* f = nullptr;
    if (!fopen_s(&f, c->cfg.status_json_path, "w") && f) {
        fprintf(f,
            "{\"init_state\":%d,\"processed_count\":%llu,\"last_process_ms\":%.3f,"
            "\"avg_process_ms\":%.3f,\"last_status\":%u,\"bypass\":%llu,"
            "\"ring_full\":%llu,\"converted_frames\":%llu,\"gpu_timeout\":%llu,"
            "\"slots_in_flight\":%u,\"slots_total\":%u,"
            "\"staging_groups\":%u,\"staging_bytes\":%llu,"
            "\"profile\":\"adapter-not-low-latency\","
            "\"required_nrvk\":1,\"fallback\":0,"
            "\"build_id\":\"%s\",\"manifest_sha256\":\"%s\",\"runtime_sha256\":\"%s\","
            "\"input_contract\":{\"color_dxgi\":%u,\"color_extent\":[%u,%u],"
            "\"motion_dxgi\":%u,\"motion_extent\":[%u,%u],"
            "\"depth_dxgi\":%u,\"depth_extent\":[%u,%u]},"
            "\"output_contract\":{\"dxgi\":%u,\"extent\":[%u,%u]},"
            "\"dispatch_contract\":{\"gather\":[%u,%u,%u],\"scatter\":[%u,%u,%u]},"
            "\"grid\":{\"active\":[%u,%u],\"scale\":%.4f},"
            "\"history\":{\"committed_before\":%u,\"used\":%u,\"reset_requested\":%u,"
            "\"frame_id\":%llu,\"control_epoch\":%llu}}\n",
            c->init_ok ? 2 : 1,
            (unsigned long long)c->frames_ok, c->last_ms, c->avg_ms, c->last_status,
            (unsigned long long)c->frames_bypass, (unsigned long long)c->ring_full,
            (unsigned long long)c->converted, (unsigned long long)c->gpu_timeout,
            (unsigned)std::count_if(c->slots.begin(), c->slots.end(),
                [](const OutputSlot& s) { return s.in_flight; }), (unsigned)c->slots.size(),
            (unsigned)c->staging.size(), (unsigned long long)c->staging_bytes,
            kFrozenBuildId,kFrozenManifestSha,kFrozenRuntimeSha,
            c->in_color_fmt,c->in_color_w,c->in_color_h,
            c->in_motion_fmt,c->in_motion_w,c->in_motion_h,
            c->in_depth_fmt,c->in_depth_w,c->in_depth_h,
            c->out_fmt,c->out_w,c->out_h,
            c->gather_x,c->gather_y,c->gather_z,
            c->scatter_x,c->scatter_y,c->scatter_z,
            grid_w, grid_h, grid_w ? double(grid_w) / 1280.0 : 0.0,
            c->history_committed?1u:0u,c->last_history,c->last_reset,
            (unsigned long long)c->last_frame_id,(unsigned long long)c->last_control_epoch);
        fclose(f);
    }
}

// ------------------------------------------------------------- Vulkan 初始化
// [GPU 路径 1] 活动 context 实际初始化——对齐 T5 已验证配方（native720 真实两帧票）：
//   instance 1.3 零扩展；设备扩展恰为 coopmat + external_memory_win32 +
//   external_semaphore_win32；feature 链 coopMatrix→shaderFloat16→
//   storageBuffer16Bit→subgroupSizeControl+computeFullSubgroups→synchronization2→
//   vulkanMemoryModel(DeviceScope/AvailabilityVisibilityChains=false)→timelineSemaphore。
//   LUID 不匹配直接失败（合同 §4，宁可不跑不静默拷贝）；COMPUTE 族队列；
//   命令池含 RESET_COMMAND_BUFFER_BIT（槽 cb 复用前显式 vkResetCommandBuffer）。
int create_vulkan(Ctx* c, ID3D12Device* d3d) {
    g_vulkan_mod = LoadLibraryA("vulkan-1.dll");
    if (!g_vulkan_mod) return -3;
    auto gipa = (PFN_vkGetInstanceProcAddr)GetProcAddress(g_vulkan_mod, "vkGetInstanceProcAddr");
    if (!gipa) return -3;
    g_gipa = gipa;
    auto vkCreateInstance = (PFN_vkCreateInstance)gipa(nullptr, "vkCreateInstance");
    VkApplicationInfo app { VK_STRUCTURE_TYPE_APPLICATION_INFO };
    app.apiVersion = VK_API_VERSION_1_3;      // T5 已验证（timeline 为 1.3 core）
    VkInstanceCreateInfo ici { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    ici.pApplicationInfo = &app;
    VkResult vir = vkCreateInstance(&ici, nullptr, &c->instance);
    if (vir != VK_SUCCESS) { fprintf(stderr, "[nrvk] vkCreateInstance fail %d\n", vir); return -4; }
    auto vkEnumPD = (PFN_vkEnumeratePhysicalDevices)gipa(c->instance, "vkEnumeratePhysicalDevices");
    auto vkGetPDProps2 = (PFN_vkGetPhysicalDeviceProperties2)gipa(c->instance, "vkGetPhysicalDeviceProperties2");
    auto vkCreateDevice = (PFN_vkCreateDevice)gipa(c->instance, "vkCreateDevice");
    uint32_t n = 0;
    if (vkEnumPD(c->instance, &n, nullptr) != VK_SUCCESS || !n) return -3;
    std::vector<VkPhysicalDevice> phys(n);
    vkEnumPD(c->instance, &n, phys.data());
    LUID want = d3d->GetAdapterLuid();
    for (auto p : phys) {                                    // LUID 匹配（合同 §4）
        VkPhysicalDeviceIDProperties idp { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES };
        VkPhysicalDeviceProperties2 pp { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &idp };
        vkGetPDProps2(p, &pp);
        if (idp.deviceLUIDValid && !memcmp(idp.deviceLUID, &want, sizeof(LUID)) && idp.deviceNodeMask) {
            c->phy = p;
            c->vk_timestamp_period_ns = (double)pp.properties.limits.timestampPeriod;
            break;
        }
    }
    if (!c->phy) { NRVK_LOG("no LUID match\n"); return -3; }
    auto vkEnumExt = (PFN_vkEnumerateDeviceExtensionProperties)gipa(c->instance, "vkEnumerateDeviceExtensionProperties");
    uint32_t ne = 0; vkEnumExt(c->phy, nullptr, &ne, nullptr);
    std::vector<VkExtensionProperties> ext(ne); vkEnumExt(c->phy, nullptr, &ne, ext.data());
    // [T5 已验证] 设备扩展恰三个；基础 external_memory/external_semaphore 由 win32
    // 变体的隐式依赖覆盖（1.3 层面驱动按需启用）
    const char* want_exts[] = { "VK_KHR_cooperative_matrix",
                                VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,
                                VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME };
    std::vector<const char*> dev_exts;
    for (auto want : want_exts)
        for (auto& e : ext) if (!strcmp(e.extensionName, want)) { dev_exts.push_back(want); break; }
    if (dev_exts.size() != 3) {
        for (auto& e : ext) fprintf(stderr, "[nrvk] ext: %s\n", e.extensionName);
        return -4;
    }
    float prio = 1.f;
    auto vkGetQF = (PFN_vkGetPhysicalDeviceQueueFamilyProperties)gipa(c->instance, "vkGetPhysicalDeviceQueueFamilyProperties");
    uint32_t nqf = 0; vkGetQF(c->phy, &nqf, nullptr);
    std::vector<VkQueueFamilyProperties> qfps(nqf); vkGetQF(c->phy, &nqf, qfps.data());
    uint32_t qfam = 0;                                       // T5 同款：首个 COMPUTE 族
    while (qfam < nqf && !(qfps[qfam].queueFlags & VK_QUEUE_COMPUTE_BIT)) ++qfam;
    if (qfam == nqf) return -4;
    c->queue_family = qfam;
    c->vk_timestamp_valid_bits = qfps[qfam].timestampValidBits;
    VkDeviceQueueCreateInfo qci { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
    qci.queueFamilyIndex = qfam; qci.queueCount = 1; qci.pQueuePriorities = &prio;
    // [T5 契约] feature 链全 TRUE；DeviceScope/AvailabilityVisibilityChains=false
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR cmf { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR };
    VkPhysicalDeviceShaderFloat16Int8Features f16f { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES };
    VkPhysicalDevice16BitStorageFeatures st16 { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES };
    VkPhysicalDeviceSubgroupSizeControlFeatures ssc { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES };
    VkPhysicalDeviceSynchronization2Features s2f { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES };
    VkPhysicalDeviceVulkanMemoryModelFeatures mmf { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES };
    VkPhysicalDeviceTimelineSemaphoreFeatures tf { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES };
    VkPhysicalDeviceFeatures2 f2q { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
    f2q.pNext = &cmf; cmf.pNext = &f16f; f16f.pNext = &st16; st16.pNext = &ssc;
    ssc.pNext = &s2f; s2f.pNext = &mmf; mmf.pNext = &tf;
    auto vkGetPDF2 = (PFN_vkGetPhysicalDeviceFeatures2)gipa(c->instance, "vkGetPhysicalDeviceFeatures2");
    vkGetPDF2(c->phy, &f2q);
    if (!cmf.cooperativeMatrix || !f16f.shaderFloat16 || !st16.storageBuffer16BitAccess ||
        !ssc.subgroupSizeControl || !ssc.computeFullSubgroups || !s2f.synchronization2 ||
        !mmf.vulkanMemoryModel || !tf.timelineSemaphore) {
        fprintf(stderr, "[nrvk] backend feature contract unmet (cm=%d f16=%d s16=%d ssc=%d cfs=%d s2=%d mm=%d tl=%d)\n",
                (int)cmf.cooperativeMatrix, (int)f16f.shaderFloat16, (int)st16.storageBuffer16BitAccess,
                (int)ssc.subgroupSizeControl, (int)ssc.computeFullSubgroups,
                (int)s2f.synchronization2, (int)mmf.vulkanMemoryModel, (int)tf.timelineSemaphore);
        return -4;
    }
    cmf.cooperativeMatrix = VK_TRUE; cmf.cooperativeMatrixRobustBufferAccess = VK_FALSE;
    f16f.shaderFloat16 = VK_TRUE; f16f.shaderInt8 = VK_FALSE;
    st16.storageBuffer16BitAccess = VK_TRUE;
    st16.uniformAndStorageBuffer16BitAccess = VK_FALSE;
    st16.storagePushConstant16 = VK_FALSE; st16.storageInputOutput16 = VK_FALSE;
    ssc.subgroupSizeControl = VK_TRUE; ssc.computeFullSubgroups = VK_TRUE;
    s2f.synchronization2 = VK_TRUE;
    mmf.vulkanMemoryModel = VK_TRUE;
    mmf.vulkanMemoryModelDeviceScope = VK_FALSE;
    mmf.vulkanMemoryModelAvailabilityVisibilityChains = VK_FALSE;
    tf.timelineSemaphore = VK_TRUE;
    VkDeviceCreateInfo dci { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &cmf };
    dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = (uint32_t)dev_exts.size();
    dci.ppEnabledExtensionNames = dev_exts.data();
    for (auto e : dev_exts) fprintf(stderr, "[nrvk] dev ext: %s\n", e);
    VkResult vdr = vkCreateDevice(c->phy, &dci, nullptr, &c->device);
    if (vdr != VK_SUCCESS) { fprintf(stderr, "[nrvk] vkCreateDevice fail %d\n", vdr); return -4; }
    auto vkGetQ = (PFN_vkGetDeviceQueue)gipa(c->instance, "vkGetDeviceQueue");
    vkGetQ(c->device, c->queue_family, 0, &c->queue);   // 族匹配 cb 池（族不匹配会挂起）
    // [GPU 路径 1] 命令池：RESET_COMMAND_BUFFER_BIT（槽 cb 显式 reset 复用）
    NRVK_DEV_FN(CreateCommandPool);
    VkCommandPoolCreateInfo pci { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = c->queue_family;
    if (pfn_CreateCommandPool(c->device, &pci, nullptr, &c->pool) != VK_SUCCESS) return -4;
    if (c->vk_timestamp_valid_bits > 0 && c->vk_timestamp_period_ns > 0.0) {
        auto pfn_CreateQueryPool = (PFN_vkCreateQueryPool)g_gdpa(c->device, "vkCreateQueryPool");
        if (pfn_CreateQueryPool) {
            VkQueryPoolCreateInfo qpci { VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
            qpci.queryType = VK_QUERY_TYPE_TIMESTAMP;
            qpci.queryCount = 128;
            if (pfn_CreateQueryPool(c->device, &qpci, nullptr, &c->vk_query_pool) != VK_SUCCESS)
                c->vk_query_pool = VK_NULL_HANDLE;
        }
    }
    return 0;
}

// [GPU 路径 2] shared D3D12 资源导入——T5 已验证配方：
//   能力预检（importable）→ samples=1 → Requirements2+DedicatedRequirements →
//   handle 属性交集选 memoryType → VkImportMemoryWin32HandleInfoKHR +
//   VkMemoryDedicatedAllocateInfo（da.image 指向该图）→ Bind → view → **完整保存**。
//   usage 按角色：输入 SAMPLED，输出 STORAGE（不是大杂烩 usage）。
int import_shared_texture(Ctx* c, ID3D12Resource* res, VkImageUsageFlags usage, ImportedImg* out) {
    NRVK_DEV_FN(CreateImage); NRVK_DEV_FN(GetMemoryWin32HandlePropertiesKHR);
    NRVK_DEV_FN(GetImageMemoryRequirements2); NRVK_DEV_FN(AllocateMemory);
    NRVK_DEV_FN(BindImageMemory); NRVK_DEV_FN(DestroyImage); NRVK_DEV_FN(FreeMemory);
    NRVK_DEV_FN(CreateImageView); NRVK_DEV_FN(DestroyImageView);
    if (!g_gipa) return -4;
    D3D12_RESOURCE_DESC d = res->GetDesc();
    VkFormat vf;
    if (!map_format(d.Format, &vf)) return 1;
    // 能力预检（T5 同款）：该 format/usage 的外部导入必须可用
    auto gifp2 = (PFN_vkGetPhysicalDeviceImageFormatProperties2)
        g_gipa(c->instance, "vkGetPhysicalDeviceImageFormatProperties2");
    if (!gifp2) return -4;
    VkPhysicalDeviceExternalImageFormatInfo ext { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO };
    ext.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT;
    VkPhysicalDeviceImageFormatInfo2 q { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2, &ext };
    q.format = vf; q.type = VK_IMAGE_TYPE_2D; q.tiling = VK_IMAGE_TILING_OPTIMAL; q.usage = usage;
    VkExternalImageFormatProperties ep { VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES };
    VkImageFormatProperties2 sp { VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2, &ep };
    if (gifp2(c->phy, &q, &sp) != VK_SUCCESS ||
        !(ep.externalMemoryProperties.externalMemoryFeatures & VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT) ||
        !(ep.externalMemoryProperties.compatibleHandleTypes & ext.handleType)) {
        NRVK_LOG("shared image not importable (fmt=%d usage=%u)\n", (int)vf, usage);
        return 3;
    }
    HANDLE nt = nullptr;
    if (FAILED(c->d3d->CreateSharedHandle(res, nullptr, GENERIC_ALL, nullptr, &nt))) return 2;
    VkExternalMemoryImageCreateInfo emi { VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO };
    emi.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT;
    VkImageCreateInfo ici { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, &emi };
    ici.imageType = VK_IMAGE_TYPE_2D; ici.format = vf;
    ici.extent = { (uint32_t)d.Width, (uint32_t)d.Height, 1 };
    ici.mipLevels = 1; ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;                 // [清单 2] 不得为零值
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = usage;                                   // 角色化 usage（SAMPLED/STORAGE）
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImage img = VK_NULL_HANDLE;
    if (pfn_CreateImage(c->device, &ici, nullptr, &img) != VK_SUCCESS) { CloseHandle(nt); return 2; }
    // Requirements2 + dedicated 需求（T5 同款）
    VkMemoryDedicatedRequirements ded { VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS };
    VkMemoryRequirements2 mr2 { VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2, &ded };
    VkImageMemoryRequirementsInfo2 mi { VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2 };
    mi.image = img;
    pfn_GetImageMemoryRequirements2(c->device, &mi, &mr2);
    VkMemoryWin32HandlePropertiesKHR hp { VK_STRUCTURE_TYPE_MEMORY_WIN32_HANDLE_PROPERTIES_KHR };
    if (pfn_GetMemoryWin32HandlePropertiesKHR(c->device,
            VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT, nt, &hp) != VK_SUCCESS) {
        CloseHandle(nt); pfn_DestroyImage(c->device, img, nullptr); return 2;
    }
    uint32_t inter = mr2.memoryRequirements.memoryTypeBits & hp.memoryTypeBits;
    auto gmp = (PFN_vkGetPhysicalDeviceMemoryProperties)g_gipa(c->instance, "vkGetPhysicalDeviceMemoryProperties");
    if (!gmp) { CloseHandle(nt); pfn_DestroyImage(c->device, img, nullptr); return -4; }
    VkPhysicalDeviceMemoryProperties mp2 {}; gmp(c->phy, &mp2);
    uint32_t type = 0;
    while (type < mp2.memoryTypeCount && !(inter & (1u << type))) ++type;
    if (type == mp2.memoryTypeCount) {
        NRVK_LOG("no imported memoryType intersection (MR=%x handle=%x)\n",
                 mr2.memoryRequirements.memoryTypeBits, hp.memoryTypeBits);
        CloseHandle(nt); pfn_DestroyImage(c->device, img, nullptr); return 2;
    }
    // 导入 + dedicated 链（da.image 指向该图；外部导入必须 dedicated）
    VkImportMemoryWin32HandleInfoKHR imp { VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR };
    imp.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT; imp.handle = nt;
    VkMemoryDedicatedAllocateInfo da { VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO };
    da.image = img; da.pNext = &imp;
    VkMemoryAllocateInfo mai { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &da };
    mai.allocationSize = mr2.memoryRequirements.size; mai.memoryTypeIndex = type;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkResult vr = pfn_AllocateMemory(c->device, &mai, nullptr, &mem);
    CloseHandle(nt);
    if (vr != VK_SUCCESS) { pfn_DestroyImage(c->device, img, nullptr); return 2; }
    if (pfn_BindImageMemory(c->device, img, mem, 0) != VK_SUCCESS) {
        pfn_FreeMemory(c->device, mem, nullptr); pfn_DestroyImage(c->device, img, nullptr); return 2;
    }
    VkImageViewCreateInfo vci { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    vci.image = img; vci.viewType = VK_IMAGE_VIEW_TYPE_2D; vci.format = vf;
    vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    VkImageView view = VK_NULL_HANDLE;
    if (pfn_CreateImageView(c->device, &vci, nullptr, &view) != VK_SUCCESS) {
        pfn_FreeMemory(c->device, mem, nullptr); pfn_DestroyImage(c->device, img, nullptr); return 2;
    }
    out->res = res; out->image = img; out->mem = mem; out->view = view;
    out->width = (uint32_t)d.Width; out->height = (uint32_t)d.Height;
    out->format = vf; out->dxgi_format = d.Format;
    return 0;
}

// [GPU 路径 2+3] 输出槽创建——T5 已验证配方：
//   D3D 侧 RGBA32F + SIMULTANEOUS_ACCESS|ALLOW_UNORDERED_ACCESS（STORAGE/UAV 对应）；
//   尺寸 = render（source-sized）；Vk 侧 STORAGE 导入并**保存** image/mem/view；
//   ready 信号 = Vk 原生 timeline（EXPORTABLE OPAQUE_WIN32）→ GetSemaphoreWin32HandleKHR
//   → D3D12 OpenSharedHandle（本机 D3D12_FENCE_BIT 只有导入声明，不得用于导出）。
// Adaptive input: on a render-size change the render-sized output slots must
// be rebuilt. Drains pending network/segment fences, then tears the slots down
// (staging groups referencing the old game textures are dropped too; both are
// recreated lazily on the next frame).
static void destroy_output_slots(Ctx* c) {
    if (!c || !g_gdpa) return;
    auto pfn_WaitForFences = (PFN_vkWaitForFences)g_gdpa(c->device, "vkWaitForFences");
    auto pfn_DestroyImageView = (PFN_vkDestroyImageView)g_gdpa(c->device, "vkDestroyImageView");
    auto pfn_DestroyImage = (PFN_vkDestroyImage)g_gdpa(c->device, "vkDestroyImage");
    auto pfn_FreeMemory = (PFN_vkFreeMemory)g_gdpa(c->device, "vkFreeMemory");
    auto pfn_DestroySemaphore = (PFN_vkDestroySemaphore)g_gdpa(c->device, "vkDestroySemaphore");
    auto pfn_DestroyFence = (PFN_vkDestroyFence)g_gdpa(c->device, "vkDestroyFence");
    auto pfn_FreeCommandBuffers = (PFN_vkFreeCommandBuffers)g_gdpa(c->device, "vkFreeCommandBuffers");
    if (!pfn_DestroyImageView || !pfn_DestroyImage || !pfn_FreeMemory) return;
    if (pfn_WaitForFences && c->network_fence_pending)
        pfn_WaitForFences(c->device, 1, &c->network_fence_pending, VK_TRUE, 10000000000ull);
    std::vector<VkCommandBuffer> cbs;
    for (auto& sl : c->slots) {
        if (pfn_WaitForFences && sl.cb_fence != VK_NULL_HANDLE)
            pfn_WaitForFences(c->device, 1, &sl.cb_fence, VK_TRUE, 10000000000ull);
        if (sl.view) pfn_DestroyImageView(c->device, sl.view, nullptr);
        if (sl.image) pfn_DestroyImage(c->device, sl.image, nullptr);
        if (sl.mem) pfn_FreeMemory(c->device, sl.mem, nullptr);
        if (sl.sem) pfn_DestroySemaphore(c->device, sl.sem, nullptr);
        if (sl.cb_fence) pfn_DestroyFence(c->device, sl.cb_fence, nullptr);
        if (sl.cb) cbs.push_back(sl.cb);
        if (sl.fence) { sl.fence->Release(); sl.fence = nullptr; }
        if (sl.res) { sl.res->Release(); sl.res = nullptr; }
        sl = OutputSlot{};
    }
    if (pfn_FreeCommandBuffers && !cbs.empty() && c->pool != VK_NULL_HANDLE)
        pfn_FreeCommandBuffers(c->device, c->pool, (uint32_t)cbs.size(), cbs.data());
    for (auto& kv : c->staging) destroy_staging_group(c, kv.second);
    c->staging.clear();
    c->staging_bytes = 0;
}

int create_output_slots(Ctx* c, uint32_t w, uint32_t h) {
    NRVK_DEV_FN(AllocateCommandBuffers); NRVK_DEV_FN(CreateFence);
    NRVK_DEV_FN(CreateSemaphore); NRVK_DEV_FN(GetSemaphoreWin32HandleKHR);
    D3D12_HEAP_PROPERTIES hp { D3D12_HEAP_TYPE_DEFAULT };
    D3D12_RESOURCE_DESC d {};
    d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    d.Width = w; d.Height = h; d.DepthOrArraySize = 1; d.MipLevels = 1;
    d.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;           // T5 已验证输出格式
    d.SampleDesc.Count = 1;
    d.Flags = D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS |
              D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    for (auto& s : c->slots) {
        if (FAILED(c->d3d->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_SHARED, &d,
                 D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&s.res)))) return -2;
        ImportedImg im {};
        int rc = import_shared_texture(c, s.res, VK_IMAGE_USAGE_STORAGE_BIT, &im);  // 导入并保存
        if (rc != 0) return rc;
        s.image = im.image; s.mem = im.mem; s.view = im.view;
        s.format = im.format; s.width = im.width; s.height = im.height;
        s.dxgi_format = d.Format;
        // [GPU 路径 3] 输出 ready：Vk 原生 timeline，创建即声明 EXPORTABLE OPAQUE_WIN32，
        // 导出 NT 句柄 → D3D12 OpenSharedHandle 为 ready fence（T5 已验证值 1/2 单调一致）
        VkSemaphoreTypeCreateInfo tci { VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO };
        tci.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE; tci.initialValue = 0;
        VkExportSemaphoreCreateInfo exp { VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO };
        exp.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
        tci.pNext = &exp;
        VkSemaphoreCreateInfo sci { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &tci };
        if (pfn_CreateSemaphore(c->device, &sci, nullptr, &s.sem) != VK_SUCCESS) return -4;
        VkSemaphoreGetWin32HandleInfoKHR shi { VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR };
        shi.semaphore = s.sem; shi.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
        HANDLE nt = nullptr;
        if (pfn_GetSemaphoreWin32HandleKHR(c->device, &shi, &nt) != VK_SUCCESS) return -4;
        HRESULT ohr = c->d3d->OpenSharedHandle(nt, IID_PPV_ARGS(&s.fence));
        CloseHandle(nt);
        if (FAILED(ohr)) { NRVK_LOG("OpenSharedHandle(exported timeline) fail hr=0x%08lX\n", (unsigned long)ohr); return -2; }
        s.fence_value = 0;
        VkCommandBufferAllocateInfo cai { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
        cai.commandPool = c->pool; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai.commandBufferCount = 1;
        if (pfn_AllocateCommandBuffers(c->device, &cai, &s.cb) != VK_SUCCESS) return -4;
        VkFenceCreateInfo fci { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;                // [S1] 首次 wait 直通
        if (pfn_CreateFence(c->device, &fci, nullptr, &s.cb_fence) != VK_SUCCESS) return -4;
    }
    return 0;
}

// ------------------------------------------------------------- D3D12 拷贝（F2/F3）
static void harvest_diag_slot(Ctx* c, uint32_t qslot, bool force_retire) {
    if (!c || qslot >= 16) return;
    NrvkDiagPendingSlot& p = c->diag_pending[qslot];
    if (!p.active) return;
    const bool copy_done = c->copy_fence && (c->copy_fence->GetCompletedValue() >= p.copy_fence_val);
    if (!copy_done && !force_retire) return;
    uint64_t vk_raw[14] {};
    VkResult qres = VK_ERROR_UNKNOWN;
    if (c->device != VK_NULL_HANDLE && c->vk_query_pool != VK_NULL_HANDLE && g_gdpa) {
        auto pfn_GetQueryPoolResults = (PFN_vkGetQueryPoolResults)g_gdpa(c->device, "vkGetQueryPoolResults");
        if (pfn_GetQueryPoolResults) {
            qres = pfn_GetQueryPoolResults(
                c->device, c->vk_query_pool, qslot * 8, 7,
                sizeof(vk_raw), vk_raw, sizeof(uint64_t) * 2,
                VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
        }
    }
    if (qres == VK_NOT_READY && !force_retire) return;
    bool all_vk_avail = (qres == VK_SUCCESS);
    for (int i = 0; i < 7; ++i) {
        p.rec.v_ticks[i] = vk_raw[i * 2];
        if (vk_raw[i * 2 + 1] == 0) all_vk_avail = false;
    }
    p.rec.vk_valid = all_vk_avail ? 1u : 0u;
    if (!all_vk_avail) ++c->diag_vk_unavail;
    p.rec.d3d_valid = 0;
    if (copy_done && c->d3d_ts_readback && c->d3d_freq_hz != 0) {
        const SIZE_T byte_off = (SIZE_T)(qslot * 2) * sizeof(uint64_t);
        D3D12_RANGE rr { byte_off, byte_off + 2 * sizeof(uint64_t) };
        void* mapped = nullptr;
        if (SUCCEEDED(c->d3d_ts_readback->Map(0, &rr, &mapped)) && mapped) {
            const uint64_t* ticks = (const uint64_t*)((const uint8_t*)mapped + byte_off);
            p.rec.d4_tick = ticks[0];
            p.rec.d5_tick = ticks[1];
            D3D12_RANGE wr { 0, 0 };
            c->d3d_ts_readback->Unmap(0, &wr);
            if (p.rec.d4_tick != 0 && p.rec.d5_tick >= p.rec.d4_tick)
                p.rec.d3d_valid = 1u;
        }
    }
    if (!p.rec.d3d_valid) ++c->diag_d3d_unavail;
    if (c->diag_records.size() < 32768) {
        c->diag_records.push_back(p.rec);
    } else {
        ++c->diag_overflow;
    }
    p.active = false;
}

static void harvest_all_retired_diag(Ctx* c, bool force_all) {
    if (!c) return;
    for (uint32_t step = 0; step < 16; ++step) {
        uint32_t best = 16;
        uint64_t best_seq = UINT64_MAX;
        for (uint32_t i = 0; i < 16; ++i) {
            if (c->diag_pending[i].active && c->diag_pending[i].rec.returned_seq < best_seq) {
                best_seq = c->diag_pending[i].rec.returned_seq;
                best = i;
            }
        }
        if (best >= 16) break;
        harvest_diag_slot(c, best, force_all);
        if (c->diag_pending[best].active) break;
    }
}

int record_and_submit_copies(Ctx* c, ID3D12CommandQueue* game_q,
                             const NRB_Frame* f, StagingGroup& st) {
    auto do_copy = [](ID3D12GraphicsCommandList* cl, ID3D12Resource* src, ImportedImg& dst) {
        if (!src || !dst.res) return;
        // Native normalized proxy contract: producer restored sampled inputs
        // to NON_PIXEL_SHADER_RESOURCE; shared staging returns to COMMON.
        D3D12_RESOURCE_BARRIER before[2] {};
        for (auto& b : before) b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        before[0].Transition = { src, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_SOURCE };
        before[1].Transition = { dst.res, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
            D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST };
        cl->ResourceBarrier(2, before);
        D3D12_TEXTURE_COPY_LOCATION dl;
        dl.pResource = dst.res; dl.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        dl.SubresourceIndex = 0;
        D3D12_TEXTURE_COPY_LOCATION sl;
        sl.pResource = src; sl.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        sl.SubresourceIndex = 0;
        cl->CopyTextureRegion(&dl, 0, 0, 0, &sl, nullptr);   // 全资源逐位（base 语义在后端 gather 处理）
        for (auto& b : before) std::swap(b.Transition.StateBefore, b.Transition.StateAfter);
        cl->ResourceBarrier(2, before);
    };
    const bool record_d3d_ts = (c->diag_mode == 2 && c->d3d_ts_heap && c->d3d_ts_readback);
    const uint32_t qslot = (uint32_t)(c->frame_seq % 16);
    if (record_d3d_ts) {
        if (c->d3d_freq_hz == 0 && game_q) game_q->GetTimestampFrequency(&c->d3d_freq_hz);
        c->copy_list->EndQuery(c->d3d_ts_heap, D3D12_QUERY_TYPE_TIMESTAMP, qslot * 2 + 0);
    }
    do_copy(c->copy_list, f->color, st.color);
    do_copy(c->copy_list, f->motion, st.motion);
    do_copy(c->copy_list, f->depth, st.depth);
    do_copy(c->copy_list, f->exposure, st.exposure);
    if (record_d3d_ts) {
        c->copy_list->EndQuery(c->d3d_ts_heap, D3D12_QUERY_TYPE_TIMESTAMP, qslot * 2 + 1);
        c->copy_list->ResolveQueryData(
            c->d3d_ts_heap, D3D12_QUERY_TYPE_TIMESTAMP,
            qslot * 2, 2, c->d3d_ts_readback, (UINT64)(qslot * 2) * sizeof(uint64_t));
    }
    if (FAILED(c->copy_list->Close())) return -1;
    ID3D12CommandList* lists[] = { c->copy_list };
    game_q->ExecuteCommandLists(1, lists);               // 游戏队列：拷贝先于 XeSS 消费
    if (FAILED(game_q->Signal(c->copy_fence, ++c->copy_fence_value))) return -1;
    return 0;                                            // 同队同序：列表先、Signal 后（F2）
}

// ------------------------------------------------------------- 控制解析（现役映射）
// 逐项对照 AMD 现役 nr_graph.cpp:1991-2007（style/tone/structure/skin/mask/
// noise/const 的解析）与本工程 ioport_images.glsl ctl(0..19) 用法。
// 无合成默认值；NRB 未提供的旋钮用 AMD CLI 现役默认并在注释标注。
static void resolve_model_controls(float c20[20], const NRB_Frame* f) {
    const NRB_Controls& n = f->controls;
    const bool mask = n.auto_mask != 0;
    const float structure = n.local_structure;         // AMD --local-structure-strength（默认 1）
    const float skin = n.skin_structure;               // AMD --skin-structure-strength（默认 -1）
    const float mask_skin  = mask ? (skin < 0 ? structure : skin) : -1.0f;
    const float mask_other = mask ? structure : -1.0f;
    const int style = n.style <= 2 ? (int)n.style : 0; // standard/natural/film → 0/1/2
    c20[0]  = float(style) / 128.0f;                   // nr_graph.cpp:2002 s434
    c20[1]  = n.local_tone;                            // :2003 aux0（--local-tone-strength 默认 1）
    c20[2]  = mask ? 1.0f : structure;                 // :2004 aux1
    c20[3]  = mask_skin;                               // :2005 gate0
    c20[4]  = mask_other;                              // :2006 gate1
    c20[5]  = 1.0f;                                    // :2017 --img-noise 现役默认 1
    c20[6]  = 1.0f;                                    // :2026 --img-const 现役默认 1
    c20[7]  = n.model_intensity;                       // surface: min(ctl7,1) 强度，>1 走特殊路径
    c20[8]  = f->reset_history ? 0.0f : 1.0f;          // temporal 开关（首帧/reset/epoch 变化由后端再压 0）
    c20[9]  = 1280.0f;                                 // history W（模型域，后端校验）
    c20[10] = 720.0f;                                  // history H
    c20[11] = 0.0f;                                    // depth-neighborhood（现役默认关）
    c20[12] = f->depth_inverted ? 1.0f : 0.0f;         // inverted selection（帧参数直传）
    c20[13] = f->motion_scale_x / 1280.0f;             // 接线清单：current-to-previous 源像素 → UV 因子
    c20[14] = f->motion_scale_y / 720.0f;
    c20[15] = 1.0f;                                    // motion UV scale（native 采样，恒等）
    c20[16] = 1.0f;
    c20[17] = 0.0f;                                    // motion UV bias（native 恒等）
    c20[18] = 0.0f;
    c20[19] = 1.0f;                                    // history_strength（AMD surface 默认全历史；NRB 无 UI 旋钮，显式常量）
}

// ------------------------------------------------------------- NRB_Processor
int nrvk_process(void* context, const NRB_Frame* f, NRB_Result* out) {
    Ctx* c = (Ctx*)context;
    // The first frame initializes this context below; init_ok starts false.
    if (!c || !f || !out) return -1;
    if (f->abi_size != sizeof(NRB_Frame) || out->abi_size != sizeof(NRB_Result)) return -1;
    std::lock_guard<std::mutex> lk(c->mu);

    // [GPU 路径 1] 首次 process：活动 context 的 Vulkan 设备/命令池初始化。
    // probe rc0 不等价于初始化（清单 1）；成功后才允许 backend init / 报 ready。
    if (c->device == VK_NULL_HANDLE) {
        int vrc = create_vulkan(c, c->d3d);
        if (vrc != 0) {
            NRVK_LOG("active vulkan init failed rc=%d\n", vrc);
            c->last_status = 2; write_status(c);
            return NRB_PROCESS_ERROR;
        }
        fprintf(stderr, "[nrvk] stage=active_vulkan_create rc=0\n");
    }

    // [T5] 后端 init：Vulkan 设备在首次 process 前创建完成；init 只允许成功一次
    if (!c->backend) {
        int irc = nrvk_backend_init(c->instance, c->device, c->phy, c->queue,
                                    c->queue_family, &c->cfg, &c->backend);
        if (irc != 0) {
            NRVK_LOG("backend init failed rc=%d (%s)\n", irc,
                     port_network_bridge_backend_last_error());
            c->last_status = 2; write_status(c);
            return NRB_PROCESS_ERROR;
        }
        fprintf(stderr, "[nrvk] stage=backend_init rc=0\n");
    }
    if (!c->copy_ready) {
        NRVK_DEV_FN(CreateSemaphore); NRVK_DEV_FN(ImportSemaphoreWin32HandleKHR);
        VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
        type.semaphoreType=VK_SEMAPHORE_TYPE_TIMELINE;
        VkSemaphoreCreateInfo create{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,&type};
        if(pfn_CreateSemaphore(c->device,&create,nullptr,&c->copy_ready)!=VK_SUCCESS) return NRB_PROCESS_ERROR;
        VkImportSemaphoreWin32HandleInfoKHR import{VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_WIN32_HANDLE_INFO_KHR};
        import.semaphore=c->copy_ready;import.handleType=VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE_BIT;import.handle=c->copy_fence_nt;
        if(pfn_ImportSemaphoreWin32HandleKHR(c->device,&import)!=VK_SUCCESS) {
            NRVK_DEV_FN(DestroySemaphore);pfn_DestroySemaphore(c->device,c->copy_ready,nullptr);c->copy_ready=VK_NULL_HANDLE;
            return NRB_PROCESS_ERROR;
        }
    }
    c->init_ok = true; // Device/pool and actual backend both initialized.

    // [单上下文] 上一帧网络完成前不得覆写共享 staging / 更新描述符 /
    // 录制下一帧——CPU 等待上一网络 fence（= 上一提交的槽 cb_fence）。
    c->last_cpu_frame_id = f->frame_id;
    c->last_cpu_had_fence = (c->network_fence_pending != VK_NULL_HANDLE) ? 1u : 0u;
    c->last_cpu_wait_start_qpc = 0;
    c->last_cpu_wait_end_qpc = 0;
    if (c->diag_mode >= 1) {
        LARGE_INTEGER q0 {}; QueryPerformanceCounter(&q0);
        c->last_cpu_wait_start_qpc = (uint64_t)q0.QuadPart;
    }
    if (c->network_fence_pending) {
        NRVK_DEV_FN(WaitForFences);
        if (pfn_WaitForFences(c->device, 1, &c->network_fence_pending, VK_TRUE,
                              10000000000ull) != VK_SUCCESS) {
            ++c->frames_bypass; ++c->gpu_timeout; c->last_status = NRB_BYPASS_UNSYNCED;
            write_status(c);
            return NRB_BYPASS_UNSYNCED;
        }
        c->network_fence_pending = VK_NULL_HANDLE;
    }
    if (c->diag_mode >= 1) {
        LARGE_INTEGER q1 {}; QueryPerformanceCounter(&q1);
        c->last_cpu_wait_end_qpc = (uint64_t)q1.QuadPart;
    }
    const uint32_t diag_qslot = (uint32_t)(c->frame_seq % 16);
    if (c->diag_pending[diag_qslot].active) {
        harvest_diag_slot(c, diag_qslot, true);
    }

    // Adaptive input: a render-size change rebuilds the render-sized output
    // slots (and drops stale staging groups); the backend breaks temporal
    // history on the same condition.
    if (!c->slots.empty() && c->slots[0].res &&
        (c->slots[0].width != f->render_width || c->slots[0].height != f->render_height)) {
        NRVK_LOG("adaptive render size %ux%u -> %ux%u; rebuilding output slots\r\n",
                 c->slots[0].width, c->slots[0].height, f->render_width, f->render_height);
        destroy_output_slots(c);
    }
    // 惰性初始化：输出槽（尺寸=首帧 render 尺寸）+ 槽私有 cb/fence（[S1]/[T4]）
    if (c->slots.empty() || !c->slots[0].res) {
        c->slots.resize(c->cfg.output_ring_slots ? c->cfg.output_ring_slots : 3);
        int rc = create_output_slots(c, f->render_width, f->render_height);
        if (rc != 0) { c->last_status = 2; write_status(c); return rc; }
    }
    OutputSlot& slot = c->slots[c->slot_next];
    if (slot.in_flight) {
        ++c->ring_full; ++c->frames_bypass; c->last_status = NRB_BYPASS_UNSYNCED;
        write_status(c); return NRB_BYPASS_UNSYNCED;
    }

    // [F3] staging 组：游戏纹理非 shared，拷贝从 D3D12 侧开始写组件自建 staging
    char key[160];
    sprintf_s(key, "%p_%p_%p_%p_%ux%u", (void*)f->color, (void*)f->motion,
              (void*)f->depth, (void*)f->exposure, f->render_width, f->render_height);
    auto it = c->staging.find(key);
    if (it == c->staging.end()) {
        // Bounded LRU: one current + one previous resource set. The previous
        // frame's network/copy fences have already been waited above, so an
        // evicted group is safe to destroy. Without this cap the map grows for
        // every new game resource pointer set (scene/UI/DLSS changes).
        while (c->staging.size() >= 2) {
            auto victim = c->staging.begin();
            for (auto i = c->staging.begin(); i != c->staging.end(); ++i)
                if (i->second.last_frame < victim->second.last_frame) victim = i;
            c->staging_bytes -= victim->second.bytes;
            destroy_staging_group(c, victim->second);
            c->staging.erase(victim);
        }
        StagingGroup g {};
        auto mk = [&](ID3D12Resource* src, ImportedImg& dst) -> int {
            if (!src) return 0;
            D3D12_RESOURCE_DESC d = src->GetDesc();
            D3D12_RESOURCE_DESC sd = d;                  // 同格式同尺寸（逐位拷贝）
            sd.Flags = D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS;   // 输入仅 SAMPLED（T5 已验证）
            D3D12_HEAP_PROPERTIES hp { D3D12_HEAP_TYPE_DEFAULT };
            if (FAILED(c->d3d->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_SHARED, &sd,
                     D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&dst.res)))) return -2;
            const auto ai = c->d3d->GetResourceAllocationInfo(0, 1, &sd);
            if (ai.SizeInBytes != UINT64_MAX) g.bytes += ai.SizeInBytes;
            const int irc = import_shared_texture(c, dst.res, VK_IMAGE_USAGE_SAMPLED_BIT, &dst);
            if (irc != 0) {
                if (dst.res) { dst.res->Release(); dst.res = nullptr; }
                return -2;
            }
            return 0;
        };
        int rc = mk(f->color, g.color);
        if (!rc) rc = mk(f->motion, g.motion);
        if (!rc) rc = mk(f->depth, g.depth);
        if (!rc) rc = mk(f->exposure, g.exposure);
        if (rc) {
            destroy_staging_group(c, g);
            c->last_status = 2; write_status(c); return NRB_BYPASS_UNSUPPORTED;
        }
        g.last_frame = f->frame_id;
        it = c->staging.emplace(key, std::move(g)).first;
        c->staging_bytes += it->second.bytes;
    }
    StagingGroup& st = it->second;
    st.last_frame = f->frame_id;
    c->in_color_fmt=(uint32_t)st.color.dxgi_format;c->in_color_w=st.color.width;c->in_color_h=st.color.height;
    c->in_motion_fmt=(uint32_t)st.motion.dxgi_format;c->in_motion_w=st.motion.width;c->in_motion_h=st.motion.height;
    c->in_depth_fmt=(uint32_t)st.depth.dxgi_format;c->in_depth_w=st.depth.width;c->in_depth_h=st.depth.height;
    c->out_fmt=(uint32_t)slot.dxgi_format;c->out_w=slot.width;c->out_h=slot.height;
    c->gather_x=3600;c->gather_y=1;c->gather_z=1;c->scatter_x=(slot.width*slot.height+255)/256;c->scatter_y=1;c->scatter_z=1;
    c->last_frame_id=f->frame_id;c->last_control_epoch=f->control_epoch;c->last_reset=f->reset_history?1u:0u;
    c->last_history=(c->history_committed && !f->reset_history &&
        f->controls.history!=NRB_HISTORY_RESET && f->control_epoch==c->history_epoch &&
        f->frame_id==c->history_frame+1)?1u:0u;

    // [F2] 拷贝在游戏 queue 上执行 + 同队 Signal(f_cp)（列表先入队、Signal 后入队；
    // 实际运行于 deferred submit 路径：上一帧 copy_fence 完成已由上一帧 network_fence_pending 保证）
    // [单上下文] Reset copy allocator 前：确认上一帧拷贝已完成（sync_queue 已
    // 上一帧 network_fence_pending 已完成，此处为显式非阻塞校验）
    if (c->copy_fence->GetCompletedValue() < c->copy_fence_value) {
        ++c->frames_bypass; ++c->gpu_timeout; c->last_status = NRB_BYPASS_UNSYNCED;
        write_status(c);
        return NRB_BYPASS_UNSYNCED;
    }
    if (FAILED(c->copy_alloc->Reset()) || FAILED(c->copy_list->Reset(c->copy_alloc, nullptr))) { c->last_status = 2; write_status(c); return NRB_PROCESS_ERROR; }
    if (record_and_submit_copies(c, f->queue, f, st) != 0) {
        c->last_status = 2; write_status(c); return NRB_PROCESS_ERROR;
    }
    // The copy queue signals copy_fence_value after all input copies. Vulkan
    // waits for that exact value in its submit; no CPU input wait is needed.

    // [S1 v4] 槽可用性检查：vkGetFenceStatus **0 等待**——上一轮 Vulkan 未完成时
    // 该槽直接视为不可用（旁路并单列 gpu_timeout 计数），**热路径不等满 1s**。
    // fence 在上次 submit 前已被 ResetFences、由该次提交 signal——本次确认
    // signaled 后才允许 ResetFences+复用（ResetFences 每次提交前正确执行）。
    {
        NRVK_DEV_FN(GetFenceStatus); NRVK_DEV_FN(ResetCommandBuffer);
        if (pfn_GetFenceStatus(c->device, slot.cb_fence) != VK_SUCCESS) {
            ++c->frames_bypass; ++c->gpu_timeout; c->last_status = NRB_BYPASS_UNSYNCED;
            write_status(c);
            return NRB_BYPASS_UNSYNCED;
        }
        pfn_ResetCommandBuffer(slot.cb, 0);
    }

    // [F1] 帧参数全透传
    NRVK_FrameParams fp { sizeof(NRVK_FrameParams) };
    fp.jitter_x = f->jitter_x; fp.jitter_y = f->jitter_y;
    fp.motion_scale_x = f->motion_scale_x; fp.motion_scale_y = f->motion_scale_y;
    fp.motion_scale_origin = f->motion_scale_origin;
    fp.pre_exposure = f->pre_exposure; fp.exposure_scale = f->exposure_scale;
    fp.auto_exposure = f->auto_exposure; fp.hdr_input = f->hdr_input;
    fp.motion_jittered = f->motion_jittered;
    fp.low_resolution_motion = f->low_resolution_motion;
    fp.depth_inverted = f->depth_inverted; fp.reset_history = f->reset_history;
    fp.render_width = f->render_width; fp.render_height = f->render_height;
    fp.output_width = f->output_width; fp.output_height = f->output_height;
    fp.color_base_x = f->color_base_x; fp.color_base_y = f->color_base_y;
    fp.motion_base_x = f->motion_base_x; fp.motion_base_y = f->motion_base_y;
    fp.depth_base_x = f->depth_base_x; fp.depth_base_y = f->depth_base_y;
    fp.output_base_x = f->output_base_x; fp.output_base_y = f->output_base_y;
    fp.graph_replay = f->controls.graph_replay;
    fp.model_intensity = f->controls.model_intensity;
    fp.local_tone = f->controls.local_tone;
    fp.local_structure = f->controls.local_structure;

    // [S2] 录制：begin → acquire → backend 网络段 → release → end（外部同步不进录制）
    {
        NRVK_DEV_FN(BeginCommandBuffer); NRVK_DEV_FN(EndCommandBuffer);
        NRVK_DEV_FN(CmdPipelineBarrier);
        VkCommandBufferBeginInfo bi { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (pfn_BeginCommandBuffer(slot.cb, &bi) != VK_SUCCESS) return NRB_PROCESS_ERROR;
        const bool record_vk_ts = (c->diag_mode == 2 && c->vk_query_pool != VK_NULL_HANDLE);
        auto pfn_CmdResetQueryPool = record_vk_ts ? (PFN_vkCmdResetQueryPool)g_gdpa(c->device, "vkCmdResetQueryPool") : nullptr;
        auto pfn_CmdWriteTimestamp = record_vk_ts ? (PFN_vkCmdWriteTimestamp)g_gdpa(c->device, "vkCmdWriteTimestamp") : nullptr;
        if (record_vk_ts && pfn_CmdResetQueryPool && pfn_CmdWriteTimestamp) {
            pfn_CmdResetQueryPool(slot.cb, c->vk_query_pool, diag_qslot * 8, 8);
            pfn_CmdWriteTimestamp(slot.cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, c->vk_query_pool, diag_qslot * 8 + 0);
        }
        // [GPU 路径 4] 真实 ownership 屏障（T5 已验证）：每帧 EXTERNAL→family acquire。
        // 数据保持 GENERAL（GENERAL→GENERAL）——UNDEFINED 转换会丢弃 D3D12 生产者
        // 内容，禁用；metadata 标 GENERAL 不构成屏障。
        VkImageMemoryBarrier acq[5] {}; uint32_t na = 0;
        auto acquire = [&](VkImage img, bool is_out) {
            if (!img) return;
            VkImageMemoryBarrier& m = acq[na++];
            m.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            m.image = img;
            m.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
            m.oldLayout = m.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            m.srcQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
            m.dstQueueFamilyIndex = c->queue_family;
            m.srcAccessMask = 0;
            m.dstAccessMask = is_out ? VK_ACCESS_SHADER_WRITE_BIT : VK_ACCESS_SHADER_READ_BIT;
        };
        acquire(st.color.image, false); acquire(st.motion.image, false);
        acquire(st.depth.image, false); acquire(st.exposure.image, false);
        acquire(slot.image, true);
        pfn_CmdPipelineBarrier(slot.cb,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, nullptr, 0, nullptr, na, acq);
        // [T5] 扩展 tail：v3 前缀 + frame_id/control_epoch/controls20/noise_seed/images[5]
        PortNetworkBridgeFrameIO ext {};
        ext.v3.abi_size = sizeof(PortNetworkBridgeFrameIO);   // 独立 tail，非 sizeof(v3)
        VkImageView views[4] = { st.color.view, st.motion.view, st.depth.view, st.exposure.view };
        ext.v3.io_view_count = 4; ext.v3.io_views = views; ext.v3.out_view = slot.view;
        ext.v3.params = &fp;
        ext.frame_id = f->frame_id;
        ext.control_epoch = f->control_epoch;
        ext.nrb_controls = f->controls;
        resolve_model_controls(ext.model_controls, f);
        // noise_seed 映射（显式定义，非截断）：frame_id 64 位经 PCG32 步进取高 32 位；
        // seed0 仍走 PCG 公式（与后端约定一致）
        ext.noise_seed = uint32_t((f->frame_id * 6364136223846793005ULL +
                                   1442695040888963407ULL) >> 33);
        auto img_meta = [&](const ImportedImg& g, VkImageUsageFlags us) {
            PortNetworkBridgeImage im {};
            im.image = g.image; im.format = g.format;
            im.extent = { g.width, g.height };
            im.valid_rect = { { 0, 0 }, { g.width, g.height } };  // 整幅（crop base 由参数一次性施加）
            im.usage = us; im.layout = VK_IMAGE_LAYOUT_GENERAL;
            im.queue_family = c->queue_family;
            return im; };
        ext.images[0] = img_meta(st.color,   VK_IMAGE_USAGE_SAMPLED_BIT);
        ext.images[1] = img_meta(st.motion,  VK_IMAGE_USAGE_SAMPLED_BIT);
        ext.images[2] = img_meta(st.depth,   VK_IMAGE_USAGE_SAMPLED_BIT);
        ext.images[3] = img_meta(st.exposure,VK_IMAGE_USAGE_SAMPLED_BIT);
        {   PortNetworkBridgeImage om {};
            om.image = slot.image;
            om.format = slot.format;                     // RGBA32F（T5 已验证；非旧 RGBA16F）
            om.extent = { slot.width, slot.height };     // 槽实际 extent（= render，禁止 SR 目标伪造）
            om.valid_rect = { { 0, 0 }, { slot.width, slot.height } };
            om.usage = VK_IMAGE_USAGE_STORAGE_BIT;
            om.layout = VK_IMAGE_LAYOUT_GENERAL;
            om.queue_family = c->queue_family;
            ext.images[4] = om; }
        if (record_vk_ts && pfn_CmdResetQueryPool && pfn_CmdWriteTimestamp) {
            nrvk_backend_set_timestamp_query(c->backend, c->vk_query_pool, diag_qslot * 8);
        } else {
            nrvk_backend_set_timestamp_query(c->backend, VK_NULL_HANDLE, 0);
        }
        int brc = nrvk_backend_frame(c->backend, slot.cb, &ext.v3);
        nrvk_backend_set_timestamp_query(c->backend, VK_NULL_HANDLE, 0);
        fprintf(stderr, "[nrvk] stage=backend_frame frame=%llu rc=%d\n", (unsigned long long)f->frame_id, brc);
        if (brc != 0) {
            // 录制失败：弃录（Reset/discard cb），绝不提交
            NRVK_DEV_FN(ResetCommandBuffer);
            pfn_ResetCommandBuffer(slot.cb, 0);
            NRVK_LOG("backend frame rc=%d (%s)\n", brc, port_network_bridge_backend_last_error());
            c->last_status = NRB_PROCESS_ERROR; write_status(c);
            return NRB_PROCESS_ERROR;
        }
        // [GPU 路径 4] family→EXTERNAL release（T5 已验证）：消费者完成证据
        // 由 ready timeline 承担，外部所有权归还。
        VkImageMemoryBarrier rel[5] {}; uint32_t nr = 0;
        auto release = [&](VkImage img, bool is_out) {
            if (!img) return;
            VkImageMemoryBarrier& m = rel[nr++];
            m.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            m.image = img;
            m.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
            m.oldLayout = m.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            m.srcQueueFamilyIndex = c->queue_family;
            m.dstQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
            m.srcAccessMask = is_out ? VK_ACCESS_SHADER_WRITE_BIT : VK_ACCESS_SHADER_READ_BIT;
            m.dstAccessMask = 0;
        };
        release(st.color.image, false); release(st.motion.image, false);
        release(st.depth.image, false); release(st.exposure.image, false);
        release(slot.image, true);
        pfn_CmdPipelineBarrier(slot.cb,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            0, 0, nullptr, 0, nullptr, nr, rel);
        if (record_vk_ts && pfn_CmdResetQueryPool && pfn_CmdWriteTimestamp) {
            pfn_CmdWriteTimestamp(slot.cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, c->vk_query_pool, diag_qslot * 8 + 6);
        }
        if (pfn_EndCommandBuffer(slot.cb) != VK_SUCCESS) return NRB_PROCESS_ERROR;
    }

    // [S2] submit 组装（[GPU 路径 3] T5 已验证）：waitSemaphore=copy_ready(copy_fence_value)（
    // GPU timeline 门控输入拷贝完成）；signal(槽导出 timeline, sv) + fence=槽 cb_fence
    // （[S1] 完成证据）。D3D12 消费者经 ready fence（导出句柄 OpenSharedHandle）
    // queue->Wait(sv)。
    {
        NRVK_DEV_FN(QueueSubmit); NRVK_DEV_FN(ResetFences);
        VkTimelineSemaphoreSubmitInfo tsi { VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO };
        uint64_t sv = slot.fence_value + 1;
        tsi.signalSemaphoreValueCount = 1; tsi.pSignalSemaphoreValues = &sv;
        tsi.waitSemaphoreValueCount=1;tsi.pWaitSemaphoreValues=&c->copy_fence_value;
        VkSubmitInfo si { VK_STRUCTURE_TYPE_SUBMIT_INFO, &tsi };
        si.commandBufferCount = 1; si.pCommandBuffers = &slot.cb;
        VkPipelineStageFlags input_stage=VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        si.waitSemaphoreCount=1;si.pWaitSemaphores=&c->copy_ready;si.pWaitDstStageMask=&input_stage;
        si.signalSemaphoreCount = 1; si.pSignalSemaphores = &slot.sem;
        pfn_ResetFences(c->device, 1, &slot.cb_fence);
        VkResult sub_rc = pfn_QueueSubmit(c->queue, 1, &si, slot.cb_fence);
        fprintf(stderr, "[nrvk] stage=network_submit frame=%llu rc=%d ready=%llu\n",
            (unsigned long long)f->frame_id, (int)sub_rc, (unsigned long long)sv);
        // [T5] 提交结果立即通知后端：成功转移记录到真实网络完成 fence；
        // 失败取消记录（历史代不前进）。两类都必须通知。
        port_network_bridge_backend_submit_result(c->backend, slot.cb_fence,
            sub_rc == VK_SUCCESS ? VK_SUCCESS : sub_rc);
        if (sub_rc != VK_SUCCESS) {
            c->last_status = NRB_PROCESS_ERROR; write_status(c);
            return NRB_PROCESS_ERROR;
        }
        c->network_fence_pending = slot.cb_fence;   // 下一帧 staging 覆写前 CPU 等待
        c->history_committed=true;c->history_frame=f->frame_id;c->history_epoch=f->control_epoch;
        out->abi_size = sizeof(NRB_Result);
        // The host takes ownership of these two references (Attach + Release).
        // The slot keeps its own reference so reuse after consumer retirement
        // cannot leave a dangling ID3D12Resource/ID3D12Fence in the slot.
        out->color = slot.res;
        if (out->color) out->color->AddRef();
        out->ready_fence = slot.fence;
        if (out->ready_fence) out->ready_fence->AddRef();
        out->ready_value = sv;                               // 与 signal 值一致
        slot.fence_value = sv;
        slot.returned_seq = ++c->frame_seq;
        if (c->diag_mode == 2 && c->vk_query_pool != VK_NULL_HANDLE && c->d3d_ts_heap && c->d3d_ts_readback) {
            NrvkDiagPendingSlot& p = c->diag_pending[diag_qslot];
            p = NrvkDiagPendingSlot {};
            p.active = true;
            p.copy_fence_val = c->copy_fence_value;
            p.cb_fence = slot.cb_fence;
            p.rec.frame_id = f->frame_id;
            p.rec.returned_seq = slot.returned_seq;
            p.rec.window_id = c->diag_window_id;
            p.rec.arm_reuse = c->diag_arm_reuse;
            p.rec.slot_index = c->slot_next;
            p.rec.copy_slot = diag_qslot;
            p.rec.had_network_fence_pending = c->last_cpu_had_fence;
            p.rec.vk_valid_bits = c->vk_timestamp_valid_bits;
            p.rec.wait_fences_start_qpc = c->last_cpu_wait_start_qpc;
            p.rec.wait_fences_end_qpc = c->last_cpu_wait_end_qpc;
            p.rec.d3d_freq_hz = c->d3d_freq_hz;
            p.rec.vk_timestamp_period_ns = c->vk_timestamp_period_ns;
        }
        slot.in_flight = true;
        c->slot_next = (c->slot_next + 1) % (uint32_t)c->slots.size();
        ++c->frames_ok; c->last_status = NRB_OK;
        write_status(c);
        return NRB_OK;
    }
}

int nrvk_retire(void* context, ID3D12Fence* fence, uint64_t value) {
    Ctx* c = (Ctx*)context;
    if (!c || !fence) return -1;
    const uint64_t completed = fence->GetCompletedValue();
    if (completed == UINT64_MAX) return NRB_PROCESS_ERROR; // D3D12 device removed.
    if (completed < value) return NRB_BYPASS_UNSYNCED;
    std::lock_guard<std::mutex> lk(c->mu);
    // NRB supplies a completed consumer fence, not the output ready fence.
    // Same-queue host callbacks retire outputs in process order. A per-slot
    // ready value is not a global order across separate timeline semaphores.
    OutputSlot* oldest = nullptr;
    for (auto& s : c->slots)
        if (s.in_flight && (!oldest || s.returned_seq < oldest->returned_seq)) oldest = &s;
    if (!oldest) return -1;
    NRVK_DEV_FN(GetSemaphoreCounterValue);
    uint64_t vk_ready = 0;
    const VkResult rc = pfn_GetSemaphoreCounterValue(c->device, oldest->sem, &vk_ready);
    if (rc != VK_SUCCESS) return NRB_PROCESS_ERROR;
    fprintf(stderr, "[nrvk] stage=consumer_retire seq=%llu expected_ready=%llu vk_ready=%llu d3d_ready=%llu consumer_completed=%llu consumer_value=%llu\n",
        (unsigned long long)oldest->returned_seq, (unsigned long long)oldest->fence_value,
        (unsigned long long)vk_ready, (unsigned long long)oldest->fence->GetCompletedValue(),
        (unsigned long long)completed, (unsigned long long)value);
    oldest->in_flight = false; write_status(c); return NRB_OK;
}

int nrvk_register(ID3D12Device* d3d, const NRVK_Config* cfg, NRB_RegisterFn host_register) {
    if (g_ctx) return NRB_OK;
    if (!cfg || cfg->abi_size != sizeof(NRVK_Config) || !cfg->manifest_dir || !d3d || !host_register) return -1;
    int prc = nrvk_backend_probe(cfg->manifest_dir);
    if (prc != 0) { NRVK_LOG("backend probe failed rc=%d\n", prc); return -2; }
    NRVK_LOG("backend probe ok (adapter linked, manifest accepted)\n");
    Ctx* c = new Ctx();
    c->cfg = *cfg; c->d3d = d3d;
    c->host_register = host_register;   // [修复2] 保存回调：Shutdown 注销必须用同一 callback
    c->slots.resize(cfg->output_ring_slots ? cfg->output_ring_slots : 3);
    if (FAILED(d3d->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
             IID_PPV_ARGS(&c->copy_alloc)))) { delete c; return -2; }
    if (FAILED(d3d->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
             c->copy_alloc, nullptr, IID_PPV_ARGS(&c->copy_list)))) { delete c; return -2; }
    c->copy_list->Close();
    if (FAILED(d3d->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&c->copy_fence)))) { delete c; return -2; }
    if (FAILED(d3d->CreateSharedHandle(c->copy_fence, nullptr, GENERIC_ALL, nullptr, &c->copy_fence_nt))) { delete c; return -2; }
    // [GPU 路径 3] 输入方向 CPU 等待用完成事件
    c->copy_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!c->copy_event) { delete c; return -2; }
    c->diag_records.reserve(32768);
    {
        D3D12_QUERY_HEAP_DESC qhd {};
        qhd.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
        qhd.Count = 32;
        qhd.NodeMask = 0;
        if (SUCCEEDED(d3d->CreateQueryHeap(&qhd, IID_PPV_ARGS(&c->d3d_ts_heap)))) {
            D3D12_HEAP_PROPERTIES hp_rb { D3D12_HEAP_TYPE_READBACK };
            D3D12_RESOURCE_DESC rd {};
            rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
            rd.Width = 32 * sizeof(uint64_t);
            rd.Height = 1;
            rd.DepthOrArraySize = 1;
            rd.MipLevels = 1;
            rd.SampleDesc.Count = 1;
            rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            if (FAILED(d3d->CreateCommittedResource(&hp_rb, D3D12_HEAP_FLAG_NONE, &rd,
                    D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&c->d3d_ts_readback)))) {
                c->d3d_ts_heap->Release();
                c->d3d_ts_heap = nullptr;
            }
        }
    }
    g_ctx = c;
    // [任务1] 宿主显式传入注册回调——不再从游戏 EXE 查找 ASI 导出
    NRB_Processor p { sizeof(NRB_Processor), g_ctx, &nrvk_process, &nrvk_retire };
    // [修复1] 宿主注册回调约定（asi.cpp:1177-1186）：成功=1，失败=0。
    // 不能按 NRB_OK(=0) 判成功——否则宿主返回 1 时会误删已注册成功的 Ctx。
    // 失败回传宿主失败码 0 会与 Attach 成功码冲突（asi.cpp:1329 按 arc==0
    // 注册 atexit），故映射为 -3。
    int pr = host_register(&p);
    if (pr != 1) { delete c; g_ctx = nullptr; return pr ? pr : -3; }
    (void)prc;
    NRVK_LOG("processor registered (adapter linked)\n");
    write_status(c);
    return NRB_OK;
}

// [退出入口] 显式关停：宿主停止回调调用。
// [任务2] 锁纪律：注销在锁外（宿主 RegisterProcessor 取宿主 state_mutex；
// 若持 c->mu 调它而一帧回调整体持有 c->mu 等宿主锁 → 死锁）；Ctx 在锁作用域
// 结束后才销毁（避免解锁已销毁 mutex）。
extern "C" __declspec(dllexport)
int NRVK_Shutdown(void) {
    if (!g_ctx) return NRB_OK;
    Ctx* c = g_ctx;
    // (1) 注销处理器（锁外）：asi.cpp:1174 若帧借出中返回 0 → 稍后重试
    if (c->host_register) {
        if (c->host_register(nullptr) != 1)
            return 2;                            // borrowed.color：稍后重试
    }
    // (2) 锁内：停止新回调 + 确认在途回调结束 + 退休 GPU 工作
    // [修复3] 设备守卫：注册后未跑首帧（create_vulkan 未执行）时 device 为空，
    // NRVK_DEV_FN 会因 g_vulkan_mod==nullptr 返回 -4——注销已成功却泄漏 Ctx、
    // g_ctx 悬空且重试恒 -4。无设备即无在途 GPU 工作，整段跳过。
    {
        std::lock_guard<std::mutex> lk(c->mu);
        // Backend borrows this device/queue and owns buffers/descriptors.
        // Complete and destroy it before the component destroys VkDevice.
        if (c->backend) { nrvk_backend_release(c->backend); c->backend = nullptr; }
        if (c->device != VK_NULL_HANDLE) {
            NRVK_DEV_FN(WaitForFences);
            if (c->network_fence_pending)
                pfn_WaitForFences(c->device, 1, &c->network_fence_pending, VK_TRUE, 10000000000ull);
            for (auto& sl : c->slots)
                if (sl.cb_fence != VK_NULL_HANDLE)
                    pfn_WaitForFences(c->device, 1, &sl.cb_fence, VK_TRUE, 10000000000ull);
            // [GPU 路径收尾] 活动 context 全部 Vulkan 对象遥毁（与创建严格对应：
            // 槽 image/view/mem/sem/cb_fence、staging 组、命令池、device、instance）
            NRVK_DEV_FN(DestroyImageView); NRVK_DEV_FN(DestroyImage); NRVK_DEV_FN(FreeMemory);
            NRVK_DEV_FN(DestroySemaphore); NRVK_DEV_FN(DestroyFence); NRVK_DEV_FN(DestroyCommandPool);
            NRVK_DEV_FN(DestroyDevice);
            if(c->copy_ready){pfn_DestroySemaphore(c->device,c->copy_ready,nullptr);c->copy_ready=VK_NULL_HANDLE;}
            for (auto& sl : c->slots) {
                if (sl.view) pfn_DestroyImageView(c->device, sl.view, nullptr);
                if (sl.image) pfn_DestroyImage(c->device, sl.image, nullptr);
                if (sl.mem) pfn_FreeMemory(c->device, sl.mem, nullptr);
                if (sl.sem) pfn_DestroySemaphore(c->device, sl.sem, nullptr);
                if (sl.cb_fence) pfn_DestroyFence(c->device, sl.cb_fence, nullptr);
                if (sl.fence) { sl.fence->Release(); sl.fence = nullptr; }
                if (sl.res) { sl.res->Release(); sl.res = nullptr; }
            }
            for (auto& kv : c->staging) destroy_staging_group(c, kv.second);
            c->staging.clear();
            c->staging_bytes = 0;
            if (c->pool) { pfn_DestroyCommandPool(c->device, c->pool, nullptr); c->pool = VK_NULL_HANDLE; }
            harvest_all_retired_diag(c, true);
            if (c->vk_query_pool != VK_NULL_HANDLE) {
                auto pfn_DestroyQueryPool = (PFN_vkDestroyQueryPool)g_gdpa(c->device, "vkDestroyQueryPool");
                if (pfn_DestroyQueryPool) pfn_DestroyQueryPool(c->device, c->vk_query_pool, nullptr);
                c->vk_query_pool = VK_NULL_HANDLE;
            }
            pfn_DestroyDevice(c->device, nullptr);
            if (g_gipa) {
                auto vkDestroyInstance = (PFN_vkDestroyInstance)g_gipa(c->instance, "vkDestroyInstance");
                if (vkDestroyInstance && c->instance != VK_NULL_HANDLE)
                    vkDestroyInstance(c->instance, nullptr);
            }
            c->device = VK_NULL_HANDLE; c->instance = VK_NULL_HANDLE; c->queue = VK_NULL_HANDLE;
            c->phy = VK_NULL_HANDLE;
            if (g_vulkan_mod) { FreeLibrary(g_vulkan_mod); g_vulkan_mod = nullptr; }
            g_gdpa = nullptr; g_gipa = nullptr;
        }
        if (c->copy_fence_nt) { CloseHandle(c->copy_fence_nt); c->copy_fence_nt = nullptr; }
        if (c->copy_event) { CloseHandle(c->copy_event); c->copy_event = nullptr; }
        if (c->d3d_ts_readback) { c->d3d_ts_readback->Release(); c->d3d_ts_readback = nullptr; }
        if (c->d3d_ts_heap) { c->d3d_ts_heap->Release(); c->d3d_ts_heap = nullptr; }
        c->init_ok = 0;
        write_status(c);
    }                                        // 锁作用域结束
    // (3) 锁外销毁 Ctx
    g_ctx = nullptr;
    delete c;
    return NRB_OK;
}

} // namespace

// DllMain 保持最小：所有等待/释放在 NRVK_Shutdown（宿主停止回调调用）
BOOL WINAPI DllMain(HINSTANCE, DWORD, LPVOID) { return TRUE; }

extern "C" __declspec(dllexport)
int NRVK_ComponentSelfTest(const NRVK_Config* cfg, ID3D12Device* /*d3d*/) {
    if (!cfg || cfg->abi_size != sizeof(NRVK_Config) || !cfg->manifest_dir) return -1;
    int rc = nrvk_backend_probe(cfg->manifest_dir);
    if (rc == -5) return -5;
    return rc != 0 ? -2 : (cfg->output_ring_slots > 8 ? -1 : 0);
}
extern "C" __declspec(dllexport)
int NRVK_AttachToBridge(ID3D12Device* d3d, const NRVK_Config* cfg, NRB_RegisterFn host_register) {
    return nrvk_register(d3d, cfg, host_register);
}

extern "C" __declspec(dllexport)
int NRVK_PreparedGpuOrderedV1(void* context,ID3D12Device* device,ID3D12CommandQueue* queue) {
    Ctx* c=static_cast<Ctx*>(context);
    if(!c || c!=g_ctx || !device || !queue) return 0;
    std::lock_guard<std::mutex> lock(c->mu);
    // On every call process() enqueues copies on frame.queue, which is the
    // same producer queue. The imported timeline gates every Vulkan command.
    return c->d3d==device && c->init_ok && c->copy_ready!=VK_NULL_HANDLE;
}

extern "C" __declspec(dllexport)
void NRVK_DiagSetState(uint32_t mode, uint32_t window_id, uint32_t arm_reuse) {
    if (!g_ctx) return;
    std::lock_guard<std::mutex> lock(g_ctx->mu);
    g_ctx->diag_mode = mode;
    g_ctx->diag_window_id = window_id;
    g_ctx->diag_arm_reuse = arm_reuse;
}

extern "C" __declspec(dllexport)
void NRVK_DiagGetLastCpuSubstage(uint64_t* frame_id, uint32_t* had_fence, uint64_t* wait_start_qpc, uint64_t* wait_end_qpc) {
    if (!g_ctx) return;
    std::lock_guard<std::mutex> lock(g_ctx->mu);
    if (frame_id) *frame_id = g_ctx->last_cpu_frame_id;
    if (had_fence) *had_fence = g_ctx->last_cpu_had_fence;
    if (wait_start_qpc) *wait_start_qpc = g_ctx->last_cpu_wait_start_qpc;
    if (wait_end_qpc) *wait_end_qpc = g_ctx->last_cpu_wait_end_qpc;
}

extern "C" __declspec(dllexport)
void NRVK_DiagHarvestRetired(void) {
    if (!g_ctx) return;
    std::lock_guard<std::mutex> lock(g_ctx->mu);
    harvest_all_retired_diag(g_ctx, false);
}

extern "C" __declspec(dllexport)
void NRVK_DiagGetRecords(const void** out_data, uint32_t* out_count, uint64_t* out_d3d_unavail, uint64_t* out_vk_unavail, uint64_t* out_overflow) {
    if (!g_ctx) {
        if (out_data) *out_data = nullptr;
        if (out_count) *out_count = 0;
        return;
    }
    std::lock_guard<std::mutex> lock(g_ctx->mu);
    harvest_all_retired_diag(g_ctx, false);
    if (out_data) *out_data = g_ctx->diag_records.empty() ? nullptr : g_ctx->diag_records.data();
    if (out_count) *out_count = (uint32_t)g_ctx->diag_records.size();
    if (out_d3d_unavail) *out_d3d_unavail = g_ctx->diag_d3d_unavail;
    if (out_vk_unavail) *out_vk_unavail = g_ctx->diag_vk_unavail;
    if (out_overflow) *out_overflow = g_ctx->diag_overflow;
}

extern "C" __declspec(dllexport)
void NRVK_DiagGetDeviceInfo(uint64_t* out_d3d_freq_hz, double* out_vk_period_ns, uint32_t* out_vk_valid_bits) {
    if (!g_ctx) return;
    std::lock_guard<std::mutex> lock(g_ctx->mu);
    if (out_d3d_freq_hz) *out_d3d_freq_hz = g_ctx->d3d_freq_hz;
    if (out_vk_period_ns) *out_vk_period_ns = g_ctx->vk_timestamp_period_ns;
    if (out_vk_valid_bits) *out_vk_valid_bits = g_ctx->vk_timestamp_valid_bits;
}

int NRVK_InteropSelfTestD3D12(ID3D12Device* d3d, const char* receipt_path) {
    if (!d3d || !receipt_path) return (fprintf(stderr, "[interop] setup fail rc=-1 @%d\n", __LINE__), -1);
    auto t0 = GetTickCount64();
    constexpr uint32_t W = 32, H = 32;                    // 32×32 RGBA16F = 8192 B（32行×256 pitch=8192）
    constexpr size_t   NB = 8192;

    // ---- D3D12 资源 ----
    D3D12_HEAP_PROPERTIES hp_def { D3D12_HEAP_TYPE_DEFAULT };
    D3D12_HEAP_PROPERTIES hp_up { D3D12_HEAP_TYPE_UPLOAD };
    D3D12_HEAP_PROPERTIES hp_rb { D3D12_HEAP_TYPE_READBACK };
    D3D12_RESOURCE_DESC td {}, bd {};
    td.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    td.Width = W; td.Height = H; td.DepthOrArraySize = 1; td.MipLevels = 1;
    td.Format = DXGI_FORMAT_R16G16B16A16_FLOAT; td.SampleDesc.Count = 1;
    td.Flags = D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS;
    bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    bd.Width = NB; bd.Height = 1; bd.DepthOrArraySize = 1; bd.MipLevels = 1;
    bd.SampleDesc.Count = 1; bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ID3D12Resource *staging_tex = nullptr, *output_tex = nullptr;
    ID3D12Resource *upload = nullptr, *readback = nullptr;
    if (FAILED(d3d->CreateCommittedResource(&hp_def, D3D12_HEAP_FLAG_NONE, &td,
             D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&staging_tex)))) return (fprintf(stderr, "[interop] setup fail rc=-2 @%d\n", __LINE__), -2);
    if (FAILED(d3d->CreateCommittedResource(&hp_def, D3D12_HEAP_FLAG_NONE, &td,
             D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&output_tex)))) return (fprintf(stderr, "[interop] setup fail rc=-2 @%d\n", __LINE__), -2);
    if (FAILED(d3d->CreateCommittedResource(&hp_up, D3D12_HEAP_FLAG_NONE, &bd,
             D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&upload)))) return (fprintf(stderr, "[interop] setup fail rc=-2 @%d\n", __LINE__), -2);
    if (FAILED(d3d->CreateCommittedResource(&hp_rb, D3D12_HEAP_FLAG_NONE, &bd,
             D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback)))) return (fprintf(stderr, "[interop] setup fail rc=-2 @%d\n", __LINE__), -2);

    std::vector<uint8_t> pattern(NB);
    for (uint32_t i = 0; i < NB / 2; ++i) {
        uint16_t v = (uint16_t)(0x3800 + (i & 0x7FF));
        pattern[i * 2] = v & 0xFF; pattern[i * 2 + 1] = v >> 8;
    }
    {   void* m = nullptr; D3D12_RANGE z { 0, 0 };
        if (FAILED(upload->Map(0, &z, &m))) return (fprintf(stderr, "[interop] setup fail rc=-2 @%d\n", __LINE__), -2);
        memcpy(m, pattern.data(), NB);
        D3D12_RANGE w { 0, NB }; upload->Unmap(0, &w); }

    ID3D12CommandAllocator* alloc = nullptr;
    ID3D12GraphicsCommandList* cl = nullptr;
    ID3D12CommandQueue* q = nullptr;
    ID3D12Fence* done = nullptr; HANDLE done_ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (FAILED(d3d->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc)))) return (fprintf(stderr, "[interop] setup fail rc=-2 @%d\n", __LINE__), -2);
    if (FAILED(d3d->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc, nullptr,
             IID_PPV_ARGS(&cl)))) return (fprintf(stderr, "[interop] setup fail rc=-2 @%d\n", __LINE__), -2);
    cl->Close();                                         // 新建列表处于 recording 态，先 Close 才可 Reset
    D3D12_COMMAND_QUEUE_DESC qd {};
    if (FAILED(d3d->CreateCommandQueue(&qd, IID_PPV_ARGS(&q)))) return (fprintf(stderr, "[interop] setup fail rc=-2 @%d\n", __LINE__), -2);
    if (FAILED(d3d->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&done)))) return (fprintf(stderr, "[interop] setup fail rc=-2 @%d\n", __LINE__), -2);
    uint64_t done_v = 0;
    auto submit_and_wait = [&]() -> int {
        if (FAILED(cl->Close())) return (fprintf(stderr, "[interop] setup fail rc=-2 @%d\n", __LINE__), -2);
        ID3D12CommandList* lists[] = { cl };
        q->ExecuteCommandLists(1, lists);
        if (FAILED(q->Signal(done, ++done_v))) return (fprintf(stderr, "[interop] setup fail rc=-3 @%d\n", __LINE__), -3);
        if (FAILED(done->SetEventOnCompletion(done_v, done_ev))) return (fprintf(stderr, "[interop] setup fail rc=-3 @%d\n", __LINE__), -3);
        if (WaitForSingleObject(done_ev, 10000) != WAIT_OBJECT_0) return 2;
        if (FAILED(alloc->Reset())) return (fprintf(stderr, "[interop] setup fail rc=-2 @%d\n", __LINE__), -2);
        return 0;
    };
    auto trans = [&](ID3D12Resource* r, D3D12_RESOURCE_STATES from, D3D12_RESOURCE_STATES to) {
        D3D12_RESOURCE_BARRIER b { D3D12_RESOURCE_BARRIER_TYPE_TRANSITION };
        b.Transition.pResource = r;
        b.Transition.StateBefore = from;
        b.Transition.StateAfter = to;
        b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        cl->ResourceBarrier(1, &b);
    };
    auto set_footprint = [&](D3D12_TEXTURE_COPY_LOCATION& l) {
        l.PlacedFootprint.Footprint.Width = W;
        l.PlacedFootprint.Footprint.Height = H;
        l.PlacedFootprint.Footprint.Depth = 1;
        l.PlacedFootprint.Footprint.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
        l.PlacedFootprint.Footprint.RowPitch = 256;
    };

    auto rbar = [&](ID3D12Resource* r, D3D12_RESOURCE_STATES from, D3D12_RESOURCE_STATES to) {
        static PFN_vkVoidFunction unused = nullptr; (void)unused;
        D3D12_RESOURCE_BARRIER b { D3D12_RESOURCE_BARRIER_TYPE_TRANSITION };
        b.Transition.pResource = r;
        b.Transition.StateBefore = from;
        b.Transition.StateAfter = to;
        b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        cl->ResourceBarrier(1, &b);
    };
    // ---- [A] D3D12：图案 → staging 纹理 → READBACK ----
    bool skip_d3d12 = getenv("NRVK_NO_D3D12") != nullptr;
    int rc = 0;                                          // 函数级（[B]/[C]/[D] 共用）
    if (skip_d3d12) fprintf(stderr, "[interop] NRVK_NO_D3D12: skip D3D12 stages\n");
    fprintf(stderr, "[interop] stage A enter\n");
    if (!skip_d3d12) {
    if (FAILED(cl->Reset(alloc, nullptr))) return (fprintf(stderr, "[interop] setup fail rc=-2 @%d\n", __LINE__), -2);
    {   trans(staging_tex, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
        trans(readback, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
        D3D12_TEXTURE_COPY_LOCATION dl { staging_tex, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {0} };
        D3D12_TEXTURE_COPY_LOCATION sl { upload, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {} };
        set_footprint(sl);
        cl->CopyTextureRegion(&dl, 0, 0, 0, &sl, nullptr);
        D3D12_TEXTURE_COPY_LOCATION rl { readback, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {} };
        set_footprint(rl);
        D3D12_TEXTURE_COPY_LOCATION d2 { staging_tex, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {0} };
        rbar(staging_tex, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
        cl->CopyTextureRegion(&rl, 0, 0, 0, &d2, nullptr);
        // [修复 R4] staging 此时处于 COPY_SOURCE（669 行已转），StateBefore 必须一致
        trans(staging_tex, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
        trans(readback, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON); }
    rc = skip_d3d12 ? 0 : submit_and_wait(); if (rc) return rc;
    } // end !skip_d3d12

    // ---- [B] CPU：READBACK → Vulkan HOST_VISIBLE ----
    Ctx tmp {}; tmp.d3d = d3d; tmp.cfg.output_ring_slots = 1;
    int vrc = create_vulkan(&tmp, d3d);
    if (vrc != 0) return vrc;
    HMODULE vm = g_vulkan_mod;
    auto gdpa = (PFN_vkGetDeviceProcAddr)GetProcAddress(vm, "vkGetDeviceProcAddr");
    if (!gdpa) return (fprintf(stderr, "[interop] setup fail rc=-4 @%d\n", __LINE__), -4);
#define VFN(name) auto pfn_##name = (PFN_vk##name)gdpa(tmp.device, "vk" #name)
    VFN(AllocateMemory); VFN(FreeMemory); VFN(CreateBuffer); VFN(DestroyBuffer);
    VFN(GetBufferMemoryRequirements); VFN(BindBufferMemory); VFN(MapMemory); VFN(UnmapMemory);
    VFN(CreateCommandPool); VFN(AllocateCommandBuffers); VFN(BeginCommandBuffer);
    VFN(EndCommandBuffer); VFN(QueueSubmit); VFN(CreateFence); VFN(WaitForFences);
    VFN(DestroyFence); VFN(DestroyCommandPool); VFN(CmdCopyBuffer);
    auto gmp_fn = (PFN_vkGetPhysicalDeviceMemoryProperties)GetProcAddress(vm, "vkGetPhysicalDeviceMemoryProperties");
    if (!gmp_fn) return (fprintf(stderr, "[interop] setup fail rc=-4 @%d\n", __LINE__), -4);
#undef VFN
    auto mk_buf = [&](VkDeviceSize sz, VkMemoryPropertyFlags mp, VkBufferUsageFlags us,
                      VkBuffer* b, VkDeviceMemory* m2) -> int {
        VkBufferCreateInfo bi { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        bi.size = sz; bi.usage = us; bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (pfn_CreateBuffer(tmp.device, &bi, nullptr, b) != VK_SUCCESS) return (fprintf(stderr, "[interop] setup fail rc=-2 @%d\n", __LINE__), -2);
        VkMemoryRequirements mr; pfn_GetBufferMemoryRequirements(tmp.device, *b, &mr);
        VkMemoryAllocateInfo mai { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
        mai.allocationSize = mr.size; mai.memoryTypeIndex = 0xFFFFFFFFu;
        VkPhysicalDeviceMemoryProperties mp2 {};
        gmp_fn(tmp.phy, &mp2);
        for (uint32_t t = 0; t < mp2.memoryTypeCount; ++t)
            if ((mr.memoryTypeBits & (1u << t)) &&
                (mp2.memoryTypes[t].propertyFlags & mp) == mp) { mai.memoryTypeIndex = t; break; }
        if (mai.memoryTypeIndex == 0xFFFFFFFFu) return (fprintf(stderr, "[interop] setup fail rc=-2 @%d\n", __LINE__), -2);
        if (pfn_AllocateMemory(tmp.device, &mai, nullptr, m2) != VK_SUCCESS)
            return (fprintf(stderr, "[interop] setup fail rc=-2 @%d\n", __LINE__), -2);
        // [修复 R1] 分配后必须绑定（Khronos 有效使用要求：未绑定 buffer 不得参与拷贝）
        if (pfn_BindBufferMemory(tmp.device, *b, *m2, 0) != VK_SUCCESS)
            return (fprintf(stderr, "[interop] setup fail rc=-2 @%d\n", __LINE__), -2);
        return 0;
    };
    VkBuffer h_in, d_mid, h_out; VkDeviceMemory m_in, m_mid, m_out;
    if ((rc = mk_buf(NB, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &h_in, &m_in))) return rc;
    if ((rc = mk_buf(NB, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &d_mid, &m_mid))) return rc;
    if ((rc = mk_buf(NB, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &h_out, &m_out))) return rc;
    {   void* m = nullptr;
        if (pfn_MapMemory(tmp.device, m_in, 0, NB, 0, &m) != VK_SUCCESS) return (fprintf(stderr, "[interop] setup fail rc=-2 @%d\n", __LINE__), -2);
        void* src = nullptr; D3D12_RANGE z { 0, 0 };
        if (FAILED(readback->Map(0, &z, &src))) return (fprintf(stderr, "[interop] setup fail rc=-2 @%d\n", __LINE__), -2);
        memcpy(m, src, NB);
        D3D12_RANGE w { 0, NB }; readback->Unmap(0, &w);
        pfn_UnmapMemory(tmp.device, m_in); }

    fprintf(stderr, "[interop] stage B done (host fill)\n");
    // ---- [C] Vulkan：HOST → LOCAL → HOST ----
    fprintf(stderr, "[interop] stage C enter\n");
    VkCommandPoolCreateInfo pci { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    pci.queueFamilyIndex = tmp.queue_family;
    if (pfn_CreateCommandPool(tmp.device, &pci, nullptr, &tmp.pool) != VK_SUCCESS) return (fprintf(stderr, "[interop] setup fail rc=-4 @%d\n", __LINE__), -4);
    VkCommandBufferAllocateInfo cai { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    cai.commandPool = tmp.pool; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount = 1;
    VkCommandBuffer cb;
    if (pfn_AllocateCommandBuffers(tmp.device, &cai, &cb) != VK_SUCCESS) return (fprintf(stderr, "[interop] setup fail rc=-4 @%d\n", __LINE__), -4);
    VkFenceCreateInfo fci { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    VkFence vf = VK_NULL_HANDLE;
    if (pfn_CreateFence(tmp.device, &fci, nullptr, &vf) != VK_SUCCESS) return (fprintf(stderr, "[interop] setup fail rc=-4 @%d\n", __LINE__), -4);
    VkCommandBufferBeginInfo bi { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    pfn_BeginCommandBuffer(cb, &bi);
    VkBufferCopy c1 { 0, 0, NB }, c2 { 0, 0, NB };
    pfn_CmdCopyBuffer(cb, h_in, d_mid, 1, &c1);
    // [修复 R2] 写后读屏障：第二拷读 d_mid 依赖第一拷的写入（官方同步示例同款）。
    // 同一 command buffer 内拷贝之间无隐式顺序，必须显式依赖。
    auto pfn_CmdPipelineBarrier = (PFN_vkCmdPipelineBarrier)gdpa(tmp.device, "vkCmdPipelineBarrier");
    if (!pfn_CmdPipelineBarrier) return (fprintf(stderr, "[interop] setup fail rc=-4 @%d\n", __LINE__), -4);
    VkBufferMemoryBarrier bmb { VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER };
    bmb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    bmb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    bmb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bmb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bmb.buffer = d_mid;
    bmb.offset = 0; bmb.size = NB;
    pfn_CmdPipelineBarrier(cb,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_DEPENDENCY_BY_REGION_BIT, 0, nullptr, 1, &bmb, 0, nullptr);
    pfn_CmdCopyBuffer(cb, d_mid, h_out, 1, &c2);
    pfn_EndCommandBuffer(cb);
    VkSubmitInfo si { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.commandBufferCount = 1; si.pCommandBuffers = &cb;
    fprintf(stderr, "[interop] submitting cb\n");
    VkResult qsr = pfn_QueueSubmit(tmp.queue, 1, &si, vf);
    fprintf(stderr, "[interop] submit rc=%d\n", qsr);
    if (qsr != VK_SUCCESS) return -3;
    auto qwi = (PFN_vkQueueWaitIdle)gdpa(tmp.device, "vkQueueWaitIdle");
    fprintf(stderr, "[interop] queueWaitIdle enter\n");
    VkResult wir = qwi(tmp.queue);
    fprintf(stderr, "[interop] queueWaitIdle rc=%d\n", wir);
    if (wir != VK_SUCCESS) return 2;
    if (pfn_WaitForFences(tmp.device, 1, &vf, VK_TRUE, 10000000000ull) != VK_SUCCESS) return 2;

    fprintf(stderr, "[interop] stage C done\n");
    // ---- [D] CPU：Vulkan HOST_OUT → D3D12 UPLOAD → 纹理 → READBACK → 比对 ----
    fprintf(stderr, "[interop] stage D enter\n");
    {   void* m = nullptr;
        if (pfn_MapMemory(tmp.device, m_out, 0, NB, 0, &m) != VK_SUCCESS) return (fprintf(stderr, "[interop] setup fail rc=-2 @%d\n", __LINE__), -2);
        void* dst = nullptr; D3D12_RANGE z { 0, 0 };
        if (FAILED(upload->Map(0, &z, &dst))) return (fprintf(stderr, "[interop] setup fail rc=-2 @%d\n", __LINE__), -2);
        memcpy(dst, m, NB);
        D3D12_RANGE w { 0, NB }; upload->Unmap(0, &w);
        pfn_UnmapMemory(tmp.device, m_out); }
    if (FAILED(cl->Reset(alloc, nullptr))) return (fprintf(stderr, "[interop] setup fail rc=-2 @%d\n", __LINE__), -2);
    {   trans(output_tex, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
        trans(readback, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
        D3D12_TEXTURE_COPY_LOCATION dl { output_tex, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {0} };
        D3D12_TEXTURE_COPY_LOCATION sl { upload, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {} };
        set_footprint(sl);
        cl->CopyTextureRegion(&dl, 0, 0, 0, &sl, nullptr);
        D3D12_TEXTURE_COPY_LOCATION rl { readback, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {} };
        set_footprint(rl);
        D3D12_TEXTURE_COPY_LOCATION d3 { output_tex, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {0} };
        // [修复 R3] output_tex 作拷贝源前必须转 COPY_SOURCE（此前仍处 COPY_DEST）
        trans(output_tex, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
        cl->CopyTextureRegion(&rl, 0, 0, 0, &d3, nullptr);
        trans(output_tex, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
        trans(readback, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON); }
    rc = submit_and_wait(); if (rc) return rc;
    void* m2 = nullptr; D3D12_RANGE zr { 0, 0 };
    if (FAILED(readback->Map(0, &zr, &m2))) return (fprintf(stderr, "[interop] setup fail rc=-2 @%d\n", __LINE__), -2);
    int cmp = memcmp(m2, pattern.data(), NB);
    D3D12_RANGE rr { 0, NB }; readback->Unmap(0, &rr);

    // ---- 清理与回执 ----
    pfn_DestroyBuffer(tmp.device, h_in, nullptr); pfn_FreeMemory(tmp.device, m_in, nullptr);
    pfn_DestroyBuffer(tmp.device, d_mid, nullptr); pfn_FreeMemory(tmp.device, m_mid, nullptr);
    pfn_DestroyBuffer(tmp.device, h_out, nullptr); pfn_FreeMemory(tmp.device, m_out, nullptr);
    pfn_DestroyFence(tmp.device, vf, nullptr); pfn_DestroyCommandPool(tmp.device, tmp.pool, nullptr);
    auto t1 = GetTickCount64();
    FILE* fp = nullptr;
    if (!fopen_s(&fp, receipt_path, "w") && fp) {
        fprintf(fp,
            "{\"interop_test\":{\"path\":\"host-staging full round trip (R1-R4 fixes applied; import path separate)\","
            "\"bytes\":%d,\"bitwise_equal\":%s,\"round_trip_ms\":%llu,"
            "\"gpu_copies\":6,\"cpu_memcpys\":2,"
            "\"icd_finding\":\"old segfault NOT attributed to driver: probe shows device-level external_memory_win32 OK; component had WIN32 names at INSTANCE level (device extensions misused) — NOT reproducible in current build (NRVK_INST_EXT=1 round trip also passes); crash-era config = timeline-only feature chain + 6 device exts + instance-level win32 names, exact trigger not isolated; import path still gated behind NRVK_EXT_MEM=1 and remains UNVERIFIED by this CPU-staging round trip\"}}\n",
            (int)NB, cmp == 0 ? "true" : "false", (unsigned long long)(t1 - t0));
        fclose(fp);
    }
    return cmp == 0 ? 0 : 1;
}
