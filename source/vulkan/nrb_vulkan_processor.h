// nrb_vulkan_processor.h — 独立 Vulkan FP16 NR 后端处理器（契约 v3）
// 冻结 NRB 契约不变（include/nr_bridge.h v1，sha 前16 b3d34c8e268992ec）。
//
// v3 修正（2026-10-13 三点）：
//   S1 命令缓冲生命周期：每输出槽配 (cb, fence) 对，fence 建为已 signalling；
//      槽复用前 vkWaitForFences(槽 fence)（GPU 完成证据）后才 reset——不靠 CPU
//      调用串行推断。cb 生命周期与槽环同构（F4 语义延续）。
//   S2 同步位置：等待/信号在 **queue submit**（VkSubmitInfo pWait/pSignal
//      Semaphores），不在命令录制内。backend 只录网络段；外部同步对象由组件
//      在 submit 组装（copy_sem wait → [网络段] → 槽 sem signal + 槽 fence）。
//   S3 最小互操作测试：NRVK_InteropSelfTestD3D12() 一次完整往返
//      （D3D12 写→Vulkan 读/写→D3D12 读回比对），含状态转换与恢复
//      （D3D12 侧保持 COMMON、Vulkan 侧仅 GENERAL——外部内存解耦状态规则）。
//
// 诚实口径（沿用 v2）：现役桥 sync_queue() 的 CPU 等待保留、输入拷贝为新增成本；
// 本组件 = **接入版（可运行、可换后端）**，不称低延迟桥；是否更快由完整链路实测。
#pragma once
#include <stdint.h>
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan.h>

// NRB 帧参数透传子集（与 NRB_Frame 一一对应；后端不做语义再解释）
struct NRVK_FrameParams {
    uint32_t abi_size;                 // = sizeof(NRVK_FrameParams)
    float    jitter_x, jitter_y;
    float    motion_scale_x, motion_scale_y;
    uint32_t motion_scale_origin;      // NRB_MotionScaleOrigin 原值
    float    pre_exposure, exposure_scale;
    uint32_t auto_exposure;
    uint32_t hdr_input;
    uint32_t motion_jittered;
    uint32_t low_resolution_motion;
    uint32_t depth_inverted;
    uint32_t reset_history;
    uint32_t render_width, render_height;
    uint32_t output_width, output_height;
    uint32_t color_base_x, color_base_y;
    uint32_t motion_base_x, motion_base_y;
    uint32_t depth_base_x, depth_base_y;
    uint32_t output_base_x, output_base_y;
    // 控制参数（NRB_Controls 的后端相关子集；显示侧旋钮不进后端）
    uint32_t graph_replay;
    float    model_intensity;
    float    local_tone;
    float    local_structure;
};

struct NRVK_Config {
    uint32_t abi_size;                 // = sizeof(NRVK_Config)
    const char* manifest_dir;          // 后端冷 manifest 目录（模型/哈希只在此目录）
    const char* cache_dir;             // pipeline/cache（可空）
    const char* status_json_path;      // 运行状态 JSON 输出路径（status.html 消费；可空）
    uint32_t output_ring_slots;        // 输出槽环深（默认 3）
    uint32_t log_verbose;
    // The frame the caller is about to submit, if it already knows it. 0 means "not stated" and
    // keeps the historical behaviour (use whatever extent the installed plan declares). When it is
    // stated and differs from the installed plan's frame, init activates the matching ladder plan
    // before the first dispatch -- buffers and descriptor sets are shared across the ladder, so
    // this costs one plan parse and no reallocation.
    uint32_t render_width;
    uint32_t render_height;
};

// 后端帧 IO（v3）：
//   输入侧：io_views[] = 组件 shared staging 的导入视图（内容 = D3D12 侧
//   CopyTextureRegion 逐位拷贝的游戏纹理，格式保持原样）。
//   buffer 生成：backend 在 cb 内记录 gather 核，把 staging 图像按 manifest io
//   布局（io_at layout 0/1/3）写入 manifest io buffer（staging buffer 池由
//   backend 持有，尺寸=manifest io 资源大小）。
//   输出侧：backend 把输出 buffer 按布局写入 out_view（scatter/写回核）。
//   同步（S2）：backend **不碰**外部同步对象——wait(copy_sem)/signal(槽 sem)
//   由组件在 VkSubmitInfo 组装；backend 返回后组件只负责 submit。
struct NRVK_BackendFrameIO {
    uint32_t abi_size;                 // = sizeof(NRVK_BackendFrameIO)
    uint32_t io_view_count;
    VkImageView* io_views;             // shared staging 导入视图（顺序见 manifest r0 端口）
    VkImageView out_view;
    const NRVK_FrameParams* params;
};
struct NRVK_Backend;
struct ID3D12Device;
extern "C" {
// 返回 0=成功；-1 参数/-2 manifest/-3 设备不匹配/-4 扩展缺失/-5 适配层未链接
int  nrvk_backend_probe(const char* manifest_dir);
int  nrvk_backend_init(VkInstance instance, VkDevice device, VkPhysicalDevice phy,
                       VkQueue queue, uint32_t queue_family, const NRVK_Config* cfg,
                       NRVK_Backend** out);
int  nrvk_backend_frame(NRVK_Backend*, VkCommandBuffer cb, const NRVK_BackendFrameIO*);
void nrvk_backend_release(NRVK_Backend*);
}

extern "C" __declspec(dllexport)
int NRVK_ComponentSelfTest(const NRVK_Config* cfg, ID3D12Device* d3d);
using NRB_RegisterFn = int (*)(const void*);   // 宿主提供的 NRB_RegisterProcessor
extern "C" __declspec(dllexport)
int NRVK_AttachToBridge(ID3D12Device* d3d, const NRVK_Config* cfg, NRB_RegisterFn host_register);
// 宿主停止回调调用；返回 0=已关停，2=帧借出中（稍后重试），<0=错误
extern "C" __declspec(dllexport)
int NRVK_Shutdown(void);
// S3：最小互操作往返（无网络段）：D3D12 写 staging → Vulkan 拷贝到输出 →
// D3D12 读回比对。需要 GPU；在 QUEUE 独立票中执行，返回 0=往返逐位一致。
extern "C" __declspec(dllexport)
int NRVK_InteropSelfTestD3D12(ID3D12Device* d3d, const char* receipt_path);
