// External-device adapter extracted from the frozen basicVMM model runner.
// No frame submit, external-image import, device creation or game installation.
#define VK_NO_PROTOTYPES
#define NOMINMAX
#define _CRT_SECURE_NO_WARNINGS
#include "port_network_bridge_backend.h"
// The host's log sink, so that a build that never finishes says where it stopped: this file is
// compiled with windows/src/core on the include path (see build_optiscaler_nr.sh), which is where
// nr_log.hpp lives. nr::logf falls back to the terminal when no host set a sink.
#include "nr_log.hpp"
#include <bcrypt.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "port_network_bridge_backend_model_identity.inc"
#include "port_network_bridge_backend_plan.inc"
// kFrameWidth/kFrameHeight and the ioport float offsets, generated from the plan on disk
// by tools/freeze_plan.py. The network's contract is the plan, not 1280x720: the output
// port carries the frame extent in its own constants (p0.zw splits the invocation id,
// p5.xy is the motion-uv denominator), so every size check and push block below has to
// follow the plan instead of contradicting it.
#include "port_network_bridge_backend_frame.inc"
// Which plan each working scale loads, with the hash that makes it the right one. The scale is a
// runtime choice: every ladder plan declares the same BUFFER and BIND lines (measured), so a
// switch only swaps the per-pass pushes and the history offsets - no reallocation, no new
// descriptor sets, no rebuild.
#include "port_network_bridge_backend_ladder.inc"

// Everything the frame path needs from the plan's extent. These used to be compile-time
// constants; they are per-instance now, initialised from the plan that was actually loaded.
struct BridgeGeometry {
    uint32_t frame_w=0,frame_h=0,canvas_w=0,canvas_h=0,ioport_motion=0,ioport_depth=0;
    double scale=0.0;
};
static BridgeGeometry geometry_of(const Plan& p) {
    BridgeGeometry g;
    // The output port is `ioport_block70_2` in the full layout and `r75_ioport_block30_2` in the
    // small one; matching the literal name threw "plan does not declare its extent" for Small.
    // Recognise it by role: any pass whose name ends in `ioport_block<N>_2`.
    for(const auto& pass:p.passes) {
        if(pass.push.size()<4) continue;
        const std::string& n=pass.name;
        if(n.find("ioport_block")==std::string::npos || n.size()<2 || n.compare(n.size()-2,2,"_2")!=0) continue;
        g.frame_w=pass.push[2];g.frame_h=pass.push[3];break;
    }
    if(!p.passes.empty()&&p.passes[0].push.size()>17){g.canvas_w=p.passes[0].push[16];g.canvas_h=p.passes[0].push[17];}
    if(!g.frame_w||!g.frame_h||!g.canvas_w||!g.canvas_h)throw std::runtime_error("plan does not declare its extent");
    const uint64_t pixels=uint64_t(g.frame_w)*uint64_t(g.frame_h);
    g.ioport_motion=uint32_t(8*pixels+20);g.ioport_depth=uint32_t(10*pixels+20);
    return g;
}

// Shader push-block sizes (6x uvec4 gather, 3x uvec4 scatter). The pipeline
// layout ranges are built from these and static_asserts bind them to the
// arrays pushed per frame, so a struct change cannot silently under-range:
// an under-sized range feeds the tail of the block from stale push memory and
// corrupts gather/scatter addressing with no API error.
constexpr uint32_t kGatherPushBytes=96;
constexpr uint32_t kScatterPushBytes=96;
// Descriptor writes per frame: gather 0-2 (colour/motion/depth), 3-4 (pre/post), 5 (base plane);
// scatter 0 (output plane), 1 (frame), 2 (pre), 3 (source colour), 4 (base plane).
constexpr uint32_t kDescriptorWrites=11;
#include "port_network_bridge_backend_frozen.inc"

static thread_local std::string last_error;
extern "C" const char* port_network_bridge_backend_last_error() { return last_error.c_str(); }
#define CHECK(x) do { const VkResult port_bridge_result_=(x); if(port_bridge_result_!=VK_SUCCESS) throw std::runtime_error(std::string(#x)+" VkResult="+std::to_string(port_bridge_result_)); } while(0)
// NRVK_DUMP_PUSH=1 prints a hash of every pass push plus its grid, so an installed plan and a
// switched one can be diffed line by line: the first differing line is where the two paths
// diverge.
static bool g_dump_push=false;
#define DEVICE_FUNCTIONS(F) \
 F(vkCreateCommandPool) F(vkDestroyCommandPool) F(vkAllocateCommandBuffers) \
 F(vkResetCommandBuffer) F(vkBeginCommandBuffer) F(vkEndCommandBuffer) \
 F(vkCreateBuffer) F(vkDestroyBuffer) F(vkGetBufferMemoryRequirements2) \
 F(vkAllocateMemory) F(vkFreeMemory) F(vkBindBufferMemory) F(vkMapMemory) F(vkUnmapMemory) \
 F(vkCreateDescriptorSetLayout) F(vkDestroyDescriptorSetLayout) F(vkCreateDescriptorPool) \
 F(vkDestroyDescriptorPool) F(vkAllocateDescriptorSets) F(vkUpdateDescriptorSets) \
 F(vkCreateShaderModule) F(vkDestroyShaderModule) F(vkCreatePipelineLayout) \
 F(vkDestroyPipelineLayout) F(vkCreateComputePipelines) F(vkDestroyPipeline) \
 F(vkCmdBindPipeline) F(vkCmdBindDescriptorSets) F(vkCmdPushConstants) F(vkCmdDispatch) \
 F(vkCmdFillBuffer) F(vkCmdUpdateBuffer) F(vkCmdCopyBuffer) F(vkCmdPipelineBarrier) \
 F(vkCreateFence) F(vkDestroyFence) F(vkResetFences) F(vkWaitForFences) F(vkGetFenceStatus) \
 F(vkQueueSubmit) F(vkQueueWaitIdle) F(vkCreateSampler) F(vkDestroySampler)
struct NRVK_Backend {
    std::string manifest_dir;
    HMODULE loader=nullptr;
    VkDevice device=VK_NULL_HANDLE; VkQueue queue=VK_NULL_HANDLE; uint32_t family=0;
    Plan plan; VkPhysicalDeviceMemoryProperties memory{};
    // Active extent, derived from `plan` at init and re-derived on a scale switch.
    BridgeGeometry geo;
    // Plan passes per ladder entry, parsed on first use. A variant carries only passes/history/
    // outputs: BUFFER and BIND are identical across scales, so buffers and descriptor sets are
    // shared and never rebuilt.
    std::vector<Plan> variants;
    std::vector<VkDescriptorSet> pass_sets;
    std::vector<size_t> pass_pipelines;
    double scale_req=0.0;   // from nr_tune.txt `scale=`; 0 means "leave it alone"
    std::vector<VkDeviceMemory> pool_memory; std::vector<Pipeline> pipelines;
    VkDescriptorPool descriptors=VK_NULL_HANDLE;
    Pipeline gather{},scatter{},scatter32{}; VkDescriptorSet gather_set=VK_NULL_HANDLE,scatter_set=VK_NULL_HANDLE;
    VkSampler sampler=VK_NULL_HANDLE;
    VkCommandPool startup_pool=VK_NULL_HANDLE; VkCommandBuffer startup_cb=VK_NULL_HANDLE;
    VkFence startup_fence=VK_NULL_HANDLE;
    Buffer staging; void* mapped=nullptr;
    // The frame as the model saw it, at the model's resolution. The scatter needs it to publish
    // the answer as an *edit* over the frame instead of a replacement (see scatter.comp); the
    // ioport planes are in-place scratch by the end of the pass list, so gather publishes the
    // resampled frame here instead.
    VkBuffer base_copy=VK_NULL_HANDLE; VkDeviceMemory base_copy_mem=VK_NULL_HANDLE; VkDeviceSize base_copy_bytes=0;
    float detail=1.0f, colour=1.0f, guard=2.0f, acutance=0.0f, rcas=0.0f, intensity=1.0f;
    // `mvs=` multiplies the motion scale handed to the model. 1 is the package value; anything
    // else is a sweep of the history-tap distance, which is the one term a moving camera exposes
    // and a still frame cannot: a wrong scale taps history short or long and the answer's overall
    // level drifts, while a zero motion vector makes the same mistake invisible.
    float mv_scale=1.0f;
    int delta=-1;      // -1 auto (edit mode whenever the model runs below render resolution), 0 replace, 1 edit
    int hist=-1;       // -1 auto (geometry gate), 0 no temporal history, 1 forced on
    int hist_logged=-1; // last logged decision, so a flip after the first frames is visible
    uint32_t ticks=0;   // frames since init: drives the twice-a-second ini re-read
    size_t pre=0,post=0,output=0;
    // The port's final compose multiplies the model's answer by this 4-byte weight. Nothing
    // ever wrote it, so it read as 0 and the port published its *input* unchanged: the network
    // could compute anything and the frame still came out identical, which is exactly what
    // every A/B so far measured.
    int blend_idx=-1; float blend=1.0f;   // NRVK_BLEND overrides for A/B
    // Whether anything actually asked for a blend value. The package ships its own constant in
    // `ioport_30_ioport_blend_scale` (4 bytes, fp16 0x3A23 = 0.7670898 for the Small model - AMD's
    // kPostBlendScaleSmall, nr_runtime.cpp:246) and the pass reads it; writing 1.0 over it every
    // frame threw that coefficient away, so the network published 100% of each fresh answer and 0%
    // of the accumulated history. Off unless a knob says otherwise.
    bool blend_given=false;
    int easu=1;         // tune.z: 1 = EASU-weighted edit upscale, 0 = bilinear
    // A scale switch reuses the pool aliases of the installed plan. Scratch buffers that the new
    // plan reads before writing still hold the previous plan's data, which is why a switched
    // 1344x768 run produced a broken answer while the same plan *installed* was clean. Clearing
    // every non-resident buffer once after a switch reproduces the freshly-allocated state.
    bool clear_all_pending=false;
    int fill_canvas=0;  // fill=1: gather/scatter cover the whole canvas, not the frame rect
    // Publish weights on the edit: tone scales its canvas-scale mean (the model's global tone shift),
    // texture scales everything above that. 1/1 == publish the edit unchanged.
    float tone_w=1.0f, texture_w=1.0f;
    // Last modification stamps of the two knob files. Per backend, not function-static: a second
    // instance would otherwise see "unchanged" and run on the compiled-in defaults forever.
    std::filesystem::file_time_type ini_stamp{}, tune_stamp{};
    bool knobs_seen=false;
    // In-game frame capture (tune `dump=1`): writes the network's input bundle and its answer
    // to disk once, so quality work can be measured on real content - depth and motion included -
    // instead of the offline harness, which has both planes zeroed.
    int dump_req=0; bool dump_pending=false; uint64_t dump_at=0;
    VkBuffer dump_buf=VK_NULL_HANDLE; VkDeviceMemory dump_mem=VK_NULL_HANDLE; void* dump_map=nullptr;
    VkCommandPool dump_pool=VK_NULL_HANDLE; VkCommandBuffer dump_cb=VK_NULL_HANDLE; VkFence dump_fence=VK_NULL_HANDLE;
    uint32_t installed_canvas_w=0,installed_canvas_h=0;
    bool record_pending=false,history_committed=false;
    // Consecutive frames refused because a record was never handed back (see the -7 gate).
    unsigned record_stuck=0;
    unsigned hist_probe=0;   // first few frames after a switch log *why* history is on or off
    uint32_t scale_miss_w=0,scale_miss_h=0;   // last missing working extent, so the warning prints once
    uint64_t pending_frame=0,pending_epoch=0,committed_frame=0,committed_epoch=0;
    uint32_t last_render_width=0,last_render_height=0;
    // Last geometry reported to nr_tune_read.txt, so one line is written per actual change and
    // not per frame.
    uint32_t geom_render_w=0,geom_render_h=0,geom_active_w=0,geom_active_h=0; bool geom_delta=false;
    VkFence last_network_fence=VK_NULL_HANDLE; VkQueryPool ts_query_pool=VK_NULL_HANDLE; uint32_t ts_base_query=0; PFN_vkCmdWriteTimestamp vkCmdWriteTimestamp=nullptr;
#define FN(f) PFN_##f f=nullptr;
    DEVICE_FUNCTIONS(FN)
#undef FN
};

static std::string sha256(const std::filesystem::path& path) {
    auto bytes=read_bytes(path.string()); BCRYPT_ALG_HANDLE alg=nullptr; BCRYPT_HASH_HANDLE hash=nullptr;
    if(BCryptOpenAlgorithmProvider(&alg,BCRYPT_SHA256_ALGORITHM,nullptr,0)<0)throw std::runtime_error("SHA256 provider");
    unsigned char digest[32];
    NTSTATUS rc=BCryptCreateHash(alg,&hash,nullptr,0,nullptr,0,0);
    if(rc>=0)rc=BCryptHashData(hash,reinterpret_cast<PUCHAR>(bytes.data()),ULONG(bytes.size()),0);
    if(rc>=0)rc=BCryptFinishHash(hash,digest,sizeof(digest),0);
    if(hash)BCryptDestroyHash(hash);BCryptCloseAlgorithmProvider(alg,0);
    if(rc<0)throw std::runtime_error("SHA256 failed");
    std::ostringstream s;for(auto x:digest)s<<std::hex<<std::setw(2)<<std::setfill('0')<<unsigned(x);return s.str();
}
// Defined below; init calls it to activate a ladder entry before the first dispatch.
static bool switch_scale(NRVK_Backend* b,size_t idx);
static Plan checked_plan(const char* directory) {
    if(!directory)throw std::runtime_error("manifest_dir missing");
    auto dir=std::filesystem::path(directory);
    if(sha256(dir/"port_network_manifest.json")!=frozen_manifest_sha ||
       sha256(dir/"port_network_runtime.txt")!=frozen_runtime_sha)
        throw std::runtime_error("requires exact build-frozen manifest/runtime (backend-adapter/port_queue_backend_frozen.json)");
    for(const auto& f:frozen_files)if(sha256(dir/f.path)!=f.sha)throw std::runtime_error(std::string("frozen file hash: ")+f.path);
    Plan p=read_plan((dir/"port_network_runtime.txt").string());
    if(p.passes.size()!=kPlanPasses || p.outputs.size()!=kPlanOutputs || p.history.size()!=kPlanHistory || p.buffers.size()!=kPlanBuffers)
        throw std::runtime_error("prepared plan dimensions differ from this build profile");
    return p;
}
extern "C" int nrvk_backend_probe(const char* dir) {
    if(!dir){last_error="manifest_dir missing";return -1;}
    try {
        auto p=checked_plan(dir);
        const BridgeGeometry g=geometry_of(p);
        // The ladder is what a scale switch loads later, so report it here: a package deployed
        // without `ladder/` would only fail at the moment someone asks for another scale.
        size_t ok=0;
        for(const auto& e:ladder_plans) {
            const auto path=std::filesystem::path(dir)/e.path;
            std::error_code ec;
            if(std::filesystem::is_regular_file(path,ec)&&sha256(path)==e.sha)++ok;
        }
        std::printf("[probe] frame=%ux%u canvas=%ux%u ioport=%u/%u ladder=%zu/%zu\n",
            g.frame_w,g.frame_h,g.canvas_w,g.canvas_h,g.ioport_motion,g.ioport_depth,ok,kLadderCount);
        last_error.clear();return 0;
    }
    catch(const std::exception& e){last_error=e.what();return -2;}
}
static void barrier(NRVK_Backend* b,VkCommandBuffer cb,VkPipelineStageFlags src,VkPipelineStageFlags dst,VkAccessFlags a,VkAccessFlags z) {
    VkMemoryBarrier m{VK_STRUCTURE_TYPE_MEMORY_BARRIER};m.srcAccessMask=a;m.dstAccessMask=z;
    b->vkCmdPipelineBarrier(cb,src,dst,0,1,&m,0,nullptr,0,nullptr);
}
static void make_buffer(NRVK_Backend* b,Buffer& buf,VkBufferUsageFlags usage,VkMemoryPropertyFlags flags) {
    VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};ci.size=buf.bytes;ci.usage=usage;
    CHECK(b->vkCreateBuffer(b->device,&ci,nullptr,&buf.handle));
    VkMemoryDedicatedRequirements dedicated{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS};
    VkMemoryRequirements2 req{VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};req.pNext=&dedicated;
    VkBufferMemoryRequirementsInfo2 query{VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2};query.buffer=buf.handle;
    b->vkGetBufferMemoryRequirements2(b->device,&query,&req);
    const auto& need=req.memoryRequirements;uint32_t type=0;
    while(type<b->memory.memoryTypeCount && !((need.memoryTypeBits&(1u<<type)) &&
        (b->memory.memoryTypes[type].propertyFlags&flags)==flags))++type;
    if(type==b->memory.memoryTypeCount)throw std::runtime_error("required memory type: "+buf.name);
    buf.memoryType=type;buf.allocated=need.size;buf.alignment=need.alignment;buf.typeBits=need.memoryTypeBits;
    buf.dedicatedRequired=dedicated.requiresDedicatedAllocation;buf.dedicatedPreferred=dedicated.prefersDedicatedAllocation;
    if((flags&VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)||buf.dedicatedRequired) {
        VkMemoryDedicatedAllocateInfo di{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};di.buffer=buf.handle;
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};ai.allocationSize=need.size;ai.memoryTypeIndex=type;ai.pNext=buf.dedicatedRequired?&di:nullptr;
        CHECK(b->vkAllocateMemory(b->device,&ai,nullptr,&buf.memory));CHECK(b->vkBindBufferMemory(b->device,buf.handle,buf.memory,0));
    }
}
// make_buffer() only binds memory for host-visible or dedicated allocations; everything else is
// meant to be pooled by pool_buffers() over `plan.buffers`. A buffer allocated beside the plan (the
// base plane) therefore has to be bound here, or its descriptor names an unbound buffer: Intel
// accepts the descriptor and the kernel reads garbage, which is a device loss away from silent.
static void bind_dedicated(NRVK_Backend* b,Buffer& buf) {
    if(buf.memory||!buf.handle)return;
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize=buf.allocated;ai.memoryTypeIndex=buf.memoryType;
    CHECK(b->vkAllocateMemory(b->device,&ai,nullptr,&buf.memory));
    CHECK(b->vkBindBufferMemory(b->device,buf.handle,buf.memory,0));
}
static void pool_buffers(NRVK_Backend* b) {
    struct Slot {VkDeviceSize offset,size;uint32_t last;};
    // The temporal planes and the published frame cross the frame boundary: the port writes
    // frame_out at the end of frame N and the history copy reads it, while the next frame reads
    // the history planes before anything wrote them. That use is invisible to the plan's
    // firstPass/lastPass numbers, so aliasing them with scratch is only accidentally safe when
    // the two canvases have the same extent. Pin them to their own slots.
    std::map<std::pair<uint32_t,bool>,std::vector<size_t>> groups;
    for(size_t i=0;i<b->plan.buffers.size();++i){auto& x=b->plan.buffers[i];if(!x.memory)groups[{x.memoryType,x.resident}].push_back(i);}
    for(auto& group:groups) {
        auto ids=group.second;std::sort(ids.begin(),ids.end(),[&](size_t a,size_t c){return b->plan.buffers[a].firstPass<b->plan.buffers[c].firstPass;});
        VkDeviceSize size=0;std::vector<Slot> slots;
        for(auto id:ids) {
            auto& x=b->plan.buffers[id];Slot* selected=nullptr;
            if(!x.resident)for(auto& s:slots)if(s.last<x.firstPass && s.size>=x.allocated && s.offset%x.alignment==0)
                if(!selected||s.size<selected->size)selected=&s;
            if(selected){x.memoryOffset=selected->offset;selected->last=x.lastPass;}
            else{x.memoryOffset=(size+x.alignment-1)/x.alignment*x.alignment;slots.push_back({x.memoryOffset,x.allocated,x.lastPass});size=x.memoryOffset+x.allocated;}
            x.ownsMemory=false;
        }
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};ai.allocationSize=size;ai.memoryTypeIndex=group.first.first;
        VkDeviceMemory mem;CHECK(b->vkAllocateMemory(b->device,&ai,nullptr,&mem));b->pool_memory.push_back(mem);
        for(auto id:ids){auto& x=b->plan.buffers[id];x.memory=mem;
            CHECK(b->vkBindBufferMemory(b->device,x.handle,mem,x.memoryOffset));}
    }
}
static Pipeline pipeline(NRVK_Backend* b,const std::string& spv,const std::vector<VkDescriptorSetLayoutBinding>& bindings,uint32_t push_bytes,uint32_t subgroup) {
    Pipeline p{};
    try {
    VkDescriptorSetLayoutCreateInfo di{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};di.bindingCount=uint32_t(bindings.size());di.pBindings=bindings.data();
    CHECK(b->vkCreateDescriptorSetLayout(b->device,&di,nullptr,&p.descriptors));
    VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT,0,push_bytes};
    VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};li.setLayoutCount=1;li.pSetLayouts=&p.descriptors;li.pushConstantRangeCount=push_bytes?1:0;li.pPushConstantRanges=&range;
    CHECK(b->vkCreatePipelineLayout(b->device,&li,nullptr,&p.layout));
    auto bytes=read_bytes(spv);std::vector<uint32_t> words(bytes.size()/4);memcpy(words.data(),bytes.data(),bytes.size());
    VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};mi.codeSize=bytes.size();mi.pCode=words.data();VkShaderModule module;
    CHECK(b->vkCreateShaderModule(b->device,&mi,nullptr,&module));
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo sg{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};sg.requiredSubgroupSize=subgroup;
    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};ci.stage.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ci.stage.pNext=&sg;ci.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;ci.stage.module=module;ci.stage.pName="main";ci.layout=p.layout;
    VkResult r=b->vkCreateComputePipelines(b->device,VK_NULL_HANDLE,1,&ci,nullptr,&p.handle);b->vkDestroyShaderModule(b->device,module,nullptr);CHECK(r);return p;
    } catch(...) {
        if(p.handle)b->vkDestroyPipeline(b->device,p.handle,nullptr);
        if(p.layout)b->vkDestroyPipelineLayout(b->device,p.layout,nullptr);
        if(p.descriptors)b->vkDestroyDescriptorSetLayout(b->device,p.descriptors,nullptr);
        throw;
    }
}
static VkDescriptorSet allocate_set(NRVK_Backend* b,VkDescriptorSetLayout layout) {
    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};ai.descriptorPool=b->descriptors;ai.descriptorSetCount=1;ai.pSetLayouts=&layout;
    VkDescriptorSet set;CHECK(b->vkAllocateDescriptorSets(b->device,&ai,&set));return set;
}
static void init_resources(NRVK_Backend* b) {
    // Progress lines, because this is the whole time of a build and it runs off the render thread:
    // a build that stalls has no other symptom than "frames pass through" for ever.
    const auto t_start=std::chrono::steady_clock::now();
    auto elapsed=[&]{return std::chrono::duration<double>(std::chrono::steady_clock::now()-t_start).count();};
    nr::logf("[nr] port: %zu plan buffers, allocating them",b->plan.buffers.size());
    for(auto& x:b->plan.buffers){make_buffer(b,x,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);b->staging.bytes=std::max(b->staging.bytes,x.bytes);}
    nr::logf("[nr] port: plan buffers allocated (%.1f s)",elapsed());
    // Sized for the largest canvas any ladder plan can ask for, so switching scale never needs a
    // bigger buffer (and never reallocates in the middle of a session).
    uint64_t canvas_pixels=uint64_t(kCanvasWidth)*uint64_t(kCanvasHeight);
    for(const auto& e:ladder_plans) canvas_pixels=std::max<uint64_t>(canvas_pixels,uint64_t(e.canvas_w)*e.canvas_h);
    b->base_copy_bytes=VkDeviceSize(canvas_pixels)*4*sizeof(float);
    { Buffer t; t.name="base_copy"; t.bytes=b->base_copy_bytes;
      make_buffer(b,t,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
      bind_dedicated(b,t);
      b->base_copy=t.handle; b->base_copy_mem=t.memory; }
    // NRVK_NO_POOL=1 gives every plan buffer its own allocation. The pool aliases scratch buffers
    // by their liveness interval, and that assignment is baked in at init - if a ladder plan is
    // switched to later and its liveness differs, the aliases it inherits are wrong. Dedicated
    // allocation removes aliasing entirely, which makes it a clean way to test that theory.
    // Braces matter here: an unbraced `for (...) if (...) stmt; else ...` binds the else to the
    // inner if, so pooling silently ran only when a buffer already had memory - every device
    // local buffer stayed unbound and the first upload died as DEVICE_LOST on the startup
    // fence. That is what crashed the game on 2026-10-31 22:31.
    if(const char* nopool=std::getenv("NRVK_NO_POOL");nopool&&std::atoi(nopool)!=0) {
        for(auto& x:b->plan.buffers) if(!x.memory) bind_dedicated(b,x);
    } else {
    pool_buffers(b);
    }
    for(auto& x:b->plan.buffers) if(!x.memory)
        throw std::runtime_error("buffer left unbound: "+x.name);
    b->staging.name="startup_staging";make_buffer(b,b->staging,VK_BUFFER_USAGE_TRANSFER_SRC_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    CHECK(b->vkMapMemory(b->device,b->staging.memory,0,VK_WHOLE_SIZE,0,&b->mapped));
    VkCommandPoolCreateInfo cp{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};cp.queueFamilyIndex=b->family;cp.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    CHECK(b->vkCreateCommandPool(b->device,&cp,nullptr,&b->startup_pool));
    VkCommandBufferAllocateInfo ca{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};ca.commandPool=b->startup_pool;ca.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;ca.commandBufferCount=1;
    CHECK(b->vkAllocateCommandBuffers(b->device,&ca,&b->startup_cb));VkFenceCreateInfo fc{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};CHECK(b->vkCreateFence(b->device,&fc,nullptr,&b->startup_fence));
    size_t upload_total=0;for(auto& x:b->plan.buffers) if(x.resident||x.dedicatedRequired)++upload_total;
    size_t upload_done=0;
    nr::logf("[nr] port: uploading %zu weight buffers, %.1f s in",upload_total,elapsed());
    for(auto& x:b->plan.buffers) {
        if(!x.resident&&!x.dedicatedRequired)continue;
        if(x.path=="-")memset(b->mapped,0,size_t(x.bytes));
        else{auto data=read_bytes(x.path);if(data.size()!=x.bytes)throw std::runtime_error("cold size: "+x.name);memcpy(b->mapped,data.data(),data.size());}
        CHECK(b->vkResetCommandBuffer(b->startup_cb,0));VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};bi.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        CHECK(b->vkBeginCommandBuffer(b->startup_cb,&bi));VkBufferCopy copy{0,0,x.bytes};b->vkCmdCopyBuffer(b->startup_cb,b->staging.handle,x.handle,1,&copy);
        barrier(b,b->startup_cb,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT);
        CHECK(b->vkEndCommandBuffer(b->startup_cb));CHECK(b->vkResetFences(b->device,1,&b->startup_fence));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};si.commandBufferCount=1;si.pCommandBuffers=&b->startup_cb;CHECK(b->vkQueueSubmit(b->queue,1,&si,b->startup_fence));
        CHECK(b->vkWaitForFences(b->device,1,&b->startup_fence,VK_TRUE,20ULL*1000*1000*1000));
        if(++upload_done%8==0||upload_done==upload_total)
            nr::logf("[nr] port: uploaded %zu/%zu (%.1f s)",upload_done,upload_total,elapsed());
    }
    b->vkUnmapMemory(b->device,b->staging.memory);b->mapped=nullptr;
    b->vkDestroyBuffer(b->device,b->staging.handle,nullptr);b->staging.handle=VK_NULL_HANDLE;
    b->vkFreeMemory(b->device,b->staging.memory,nullptr);b->staging.memory=VK_NULL_HANDLE;
    b->vkDestroyFence(b->device,b->startup_fence,nullptr);b->startup_fence=VK_NULL_HANDLE;
    b->vkDestroyCommandPool(b->device,b->startup_pool,nullptr);b->startup_pool=VK_NULL_HANDLE;b->startup_cb=VK_NULL_HANDLE;
    // 7 = the bridge's own storage buffers: gather 3 (pre, post, base plane) + scatter 4
    // (output plane, pre fallback, base plane, colour is a sampler). Every plan pass adds its own.
    std::map<std::string,size_t> cache;uint32_t count=7;
    nr::logf("[nr] port: creating pipelines for %zu passes (%.1f s)",b->plan.passes.size(),elapsed());
    for(auto& pass:b->plan.passes) {
        count+=uint32_t(pass.bindings.size());std::string key=pass.spv+":"+std::to_string(pass.push.size())+":"+std::to_string(pass.subgroup);
        for(auto& bind:pass.bindings)key+=":"+std::to_string(bind.binding);
        if(cache.count(key)){pass.pipeline=cache[key];continue;}
        std::vector<VkDescriptorSetLayoutBinding> bindings;for(auto& bind:pass.bindings)bindings.push_back({bind.binding,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr});
        pass.pipeline=b->pipelines.size();cache[key]=pass.pipeline;b->pipelines.push_back(pipeline(b,pass.spv,bindings,uint32_t(pass.push.size()*4),pass.subgroup));
        if(b->pipelines.size()%8==0) nr::logf("[nr] port: %zu pipelines (%.1f s)",b->pipelines.size(),elapsed());
    }
    nr::logf("[nr] port: %zu pipelines for %zu passes (%.1f s)",b->pipelines.size(),b->plan.passes.size(),elapsed());
    // Push-constant ranges MUST match the shader block sizes exactly: a range
    // shorter than the block silently feeds the tail of the struct from
    // whatever is in push-constant memory, which corrupts gather/scatter
    // addressing without any API error. Static asserts bind them to the arrays.
    auto sb=[](uint32_t binding){return VkDescriptorSetLayoutBinding{binding,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};};
    auto cis=[](uint32_t binding){return VkDescriptorSetLayoutBinding{binding,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};};
    auto si=[](uint32_t binding){return VkDescriptorSetLayoutBinding{binding,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};};
    // gather: colour/motion/depth, the two ioport input bundles, and the base plane it publishes
    // for the scatter. scatter: the network output plane, the frame, the pre bundle (the
    // composition fallback when the base plane is not used), the source colour and the base plane.
    b->gather=pipeline(b,(std::filesystem::path(b->manifest_dir)/gather_spv).string(),{cis(0),cis(1),cis(2),sb(3),sb(4),sb(5)},kGatherPushBytes,32);
    b->scatter=pipeline(b,(std::filesystem::path(b->manifest_dir)/scatter_spv).string(),{sb(0),si(1),sb(2),cis(3),sb(4)},kScatterPushBytes,32);
    b->scatter32=pipeline(b,(std::filesystem::path(b->manifest_dir)/scatter32_spv).string(),{sb(0),si(1),sb(2),cis(3),sb(4)},kScatterPushBytes,32);
    // Every storage buffer the two bridge sets can name has to fit: gather 3 + scatter 4 are in
    // `count` already, and each plan pass contributes one per binding.
    VkDescriptorPoolSize sizes[]={{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,count},{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,4},{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,1}};
    VkDescriptorPoolCreateInfo di{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};di.maxSets=uint32_t(b->plan.passes.size()+2);di.poolSizeCount=3;di.pPoolSizes=sizes;CHECK(b->vkCreateDescriptorPool(b->device,&di,nullptr,&b->descriptors));
    for(auto& pass:b->plan.passes) {
        pass.set=allocate_set(b,b->pipelines[pass.pipeline].descriptors);std::vector<VkDescriptorBufferInfo> info(pass.bindings.size());std::vector<VkWriteDescriptorSet> writes(pass.bindings.size());
        for(size_t i=0;i<pass.bindings.size();++i){auto& x=pass.bindings[i];info[i]={b->plan.buffers[x.resource].handle,x.offset,x.bytes};writes[i]={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};writes[i].dstSet=pass.set;writes[i].dstBinding=x.binding;writes[i].descriptorCount=1;writes[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;writes[i].pBufferInfo=&info[i];}
        b->vkUpdateDescriptorSets(b->device,uint32_t(writes.size()),writes.data(),0,nullptr);
    }
    b->gather_set=allocate_set(b,b->gather.descriptors);b->scatter_set=allocate_set(b,b->scatter.descriptors);
    // The per-pass descriptor sets belong to the buffers, not to a scale: every ladder plan
    // declares the same BIND lines, so a variant reuses these sets as-is.
    b->pass_sets.clear();b->pass_sets.reserve(b->plan.passes.size());
    b->pass_pipelines.clear();b->pass_pipelines.reserve(b->plan.passes.size());
    for(auto& pass:b->plan.passes) { b->pass_sets.push_back(pass.set); b->pass_pipelines.push_back(pass.pipeline); }
    VkSamplerCreateInfo sc{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};sc.magFilter=VK_FILTER_NEAREST;sc.minFilter=VK_FILTER_NEAREST;sc.mipmapMode=VK_SAMPLER_MIPMAP_MODE_NEAREST;sc.addressModeU=sc.addressModeV=sc.addressModeW=VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;sc.maxLod=0;
    CHECK(b->vkCreateSampler(b->device,&sc,nullptr,&b->sampler));
    // 102's first pass is the F1 transport. Binding 1 is its aux pack, not the
    // image bundle. Select the frozen IO bundles by their semantic names.
    bool found_pre=false,found_post=false;
    for(size_t i=0;i<b->plan.buffers.size();++i) {
        if(b->plan.buffers[i].name==kPreImageName){b->pre=i;found_pre=true;}
        if(b->plan.buffers[i].name==kPostImageName){b->post=i;found_post=true;}
    }
    if(!found_pre||!found_post)throw std::runtime_error("Small ioport image bundles missing");
    // The bundles have to hold the *largest* plan the ladder can switch to, not just the one that
    // happens to be installed: a switch must never run out of room in the middle of a session.
    uint64_t frame_px=uint64_t(b->geo.frame_w)*uint64_t(b->geo.frame_h);
    uint64_t canvas_px=uint64_t(b->geo.canvas_w)*uint64_t(b->geo.canvas_h);
    for(const auto& e:ladder_plans){
        frame_px=std::max<uint64_t>(frame_px,uint64_t(e.frame_w)*e.frame_h);
        canvas_px=std::max<uint64_t>(canvas_px,uint64_t(e.canvas_w)*e.canvas_h);
    }
    const uint64_t ioport_min=12u*frame_px+40u;   // four planes + their 20-byte tails
    if(b->plan.buffers[b->pre].bytes<ioport_min || b->plan.buffers[b->post].bytes<ioport_min)
        throw std::runtime_error("ioport bundle smaller than the largest ladder canvas");
    b->installed_canvas_w=b->geo.canvas_w;b->installed_canvas_h=b->geo.canvas_h;
    b->output=b->plan.outputs.front().resource;
    for(size_t i=0;i<b->plan.buffers.size();++i) {
        // `ioport_70_ioport_blend_scale` in the full layout, `ioport_30_...` in the small one:
        // the strength control has to follow the port's block number, so match the suffix.
        const std::string& n=b->plan.buffers[i].name;
        const std::string suffix="_ioport_blend_scale";
        if(n.size()>suffix.size() && n.compare(n.size()-suffix.size(),suffix.size(),suffix)==0) b->blend_idx=int(i);
    }
    if(const char* e=std::getenv("NRVK_BLEND")){b->blend=float(std::atof(e));b->blend_given=true;}
    if(const char* e=std::getenv("NRVK_DUMP_PUSH"))g_dump_push=std::atoi(e)!=0;
    if(b->plan.buffers[b->output].bytes<4u*canvas_px*sizeof(float))throw std::runtime_error("frame_out smaller than the largest ladder canvas");
    nr::logf("[nr] port: resources ready (%zu buffers, %zu pipelines, %zu passes) in %.1f s",
             b->plan.buffers.size(),b->pipelines.size(),b->plan.passes.size(),elapsed());
}

// Device gate. B580 (0xe20b, Xe2 dGPU) is the verified target. The Arc B390
// (Panther Lake Xe3 iGPU) ships with PCI deviceID 0xb080 and is adapted: the
// same Intel Vulkan driver family exposes the 8x16x16 fp16 cooperative-matrix
// recipe, 32-wide full subgroups and >=48KiB SLM that the frozen shaders and
// weight layout are bound to (tools/b390_vk_contract_probe.cpp validates this
// on real hardware). Other Panther Lake SKUs can be opted in explicitly with
// NRVK_DEVICE_ID=<hex>; everything else fails closed.
static bool supported_intel_device(uint32_t deviceID) {
    if(deviceID==0xe20b||deviceID==0xb080)return true;
    char env[32]{};const DWORD n=GetEnvironmentVariableA("NRVK_DEVICE_ID",env,sizeof env);
    return n>0&&n<sizeof env&&strtoul(env,nullptr,16)==deviceID;
}

// Composition knobs, named after the panel that owns them (OptiScaler DLSS-NR -> "Apply NR edit"):
// Detail strength 0..2, Colour strength 0..4, Highlight guard 1..8 -- the same three AMD composes
// with in windows/shaders/passes/runtime_transfer.comp. They never reach the NGX parameter block,
// so they are read from files. Two sources, in order:
//   1. OptiScaler.ini [DlssNr] TransferStrength / ColourStrength -- what the panel writes, when it
//      writes. It does not save on exit in this build, so this source is often stale;
//   2. nr_tune.txt next to it (detail=, colour=, guard=) -- ours, always live.
// Both are polled (stat only) every 30 frames, so a change lands in about half a second.
static void read_strengths(NRVK_Backend* b) {
    const std::filesystem::path dir=std::filesystem::path(b->manifest_dir).parent_path();
    const std::filesystem::path tune_path=dir/"nr_tune.txt";
    // OptiScaler.ini lives next to the game executable, not next to the manifest. Reading only the
    // package copy is why every line in nr_tune_read.txt said `ini=missing`: the panel's
    // TransferStrength / ColourStrength were written to a file this code never opened. Look beside
    // the exe first, fall back to the package.
    std::filesystem::path ini_path=dir/"OptiScaler.ini";
    {
        std::error_code e0;
        char exe[MAX_PATH]{};const DWORD n=GetModuleFileNameA(nullptr,exe,sizeof exe);
        if(n>0&&n<sizeof exe){
            const auto beside=std::filesystem::path(exe).parent_path()/"OptiScaler.ini";
            if(std::filesystem::exists(beside,e0))ini_path=beside;
        }
    }
    std::error_code e1,e2;
    const auto t1=std::filesystem::last_write_time(ini_path,e1);
    const auto t2=std::filesystem::last_write_time(tune_path,e2);
    // First call always reads: the compiled-in defaults are not a measurement.
    const bool changed=!b->knobs_seen||(!e1&&t1!=b->ini_stamp)||(!e2&&t2!=b->tune_stamp);
    if(!changed)return;
    b->knobs_seen=true;b->ini_stamp=t1;b->tune_stamp=t2;

    float detail=1.0f,colour=1.0f,guard=2.0f,acutance=0.0f,rcas=0.0f,delta=-1.0f;
    auto scan=[&](const std::filesystem::path& path,const char* const* keys,float* out,int n,bool ini) {
        FILE* f=std::fopen(path.string().c_str(),"rb"); if(!f)return;
        char line[256];
        while(std::fgets(line,sizeof line,f)) {
            for(int i=0;i<n;++i) {
                const size_t k=std::strlen(keys[i]);
                if(std::strncmp(line,keys[i],k)!=0)continue;
                const char* eq=std::strchr(line,ini?'=':'=');if(!eq)return;
                char* end=nullptr;const float v=std::strtof(eq+1,&end);
                if(end==eq+1)continue;                 // "auto"/empty: leave the default
                out[i]=v;
            }
        }
        std::fclose(f);
    };
    static const char* ini_keys[]={"TransferStrength","ColourStrength"};
    float ini_vals[]={1.0f,1.0f};
    scan(ini_path,ini_keys,ini_vals,2,true);
    detail=ini_vals[0];colour=ini_vals[1];
    static const char* tune_keys[]={"detail=","colour=","guard=","acutance=","delta=","scale=","hist=",
        "intensity=","blend=","easu=","fill=","dump=","tone=","texture=","rcas=","mvs="};
    // blend defaults to -1 = "leave the package's own blend_scale alone" (see blend_given).
    float tune_vals[]={detail,colour,guard,0.0f,delta,0.0f,-1.0f,1.0f,-1.0f,1.0f,0.0f,0.0f,1.0f,1.0f,0.0f,1.0f};
    scan(tune_path,tune_keys,tune_vals,16,false);
    detail=tune_vals[0];colour=tune_vals[1];guard=tune_vals[2];acutance=tune_vals[3];delta=tune_vals[4];
    // `scale=0.5` asks for a ladder plan; 0 (or absent, or "auto") leaves the installed one alone.
    b->scale_req=tune_vals[5]<0.05f?0.0:(tune_vals[5]>1.0f?1.0:double(tune_vals[5]));

    b->detail=detail<0.0f?0.0f:(detail>2.0f?2.0f:detail);
    b->colour=colour<0.0f?0.0f:(colour>4.0f?4.0f:colour);
    b->guard =guard<1.0f?1.0f:(guard>8.0f?8.0f:guard);
    b->acutance=acutance<0.0f?0.0f:(acutance>2.0f?2.0f:acutance);
    // RCAS strength 0..1. 0 keeps the historic fixed-weight high-pass byte-identical, so an
    // existing install is unaffected until the key is set. Offline sweep on a captured 1120x700
    // frame: 0.5 reaches acut=0.25's flat-area detail at roughly a third of its edge overshoot.
    b->rcas=tune_vals[14]<0.0f?0.0f:(tune_vals[14]>1.0f?1.0f:tune_vals[14]);
    b->mv_scale=tune_vals[15]<0.05f?0.05f:(tune_vals[15]>8.0f?8.0f:tune_vals[15]);
    // -1 (or "delta=auto") keeps the automatic rule: edit mode only when the model runs below
    // the render resolution.
    b->delta=delta<0.0f?-1:(delta>=0.5f?1:0);
    b->hist =tune_vals[6]<0.0f?-1:(tune_vals[6]>=0.5f?1:0);
    // ctl7 is the model's intensity and it is *linear*, not a 0..1 collapse: swept on the real
    // 960x600 frame (offline harness, NRVK_CTL=7:<v>), |answer - input| is 0.00/255 at 0,
    // 0.95 at 0.25, 1.90 at 0.5, 3.80 at 1.0, 5.58 at 1.5, 7.31 at 2.0 and 10.64 at 3.0, with
    // the frame mean staying at 0.825 the whole way. The old note ("at 1.0 the frame collapses
    // into flat bars") does not reproduce, and the clamp at 1.0 threw away more than half of the
    // model's range - which is why NR looked like it did nothing. Range is 0..3.
    b->intensity=tune_vals[7]<0.0f?0.0f:(tune_vals[7]>3.0f?3.0f:tune_vals[7]);
    // NRVK_INTENSITY=<0..1>: the same override NRVK_BLEND has, added because the offline runner
    // never gets a tune file (its manifest_dir is the plan directory, so `read_strengths` finds
    // neither nr_tune.txt nor OptiScaler.ini) and intensity is the one knob that decides whether
    // this network edits the frame at all: the note above measured |answer-input| at 2.9/255 for
    // c[7]=0 against 33/255 at 0.5. Without the override the harness could not A/B it.
    if(const char* e=std::getenv("NRVK_INTENSITY")){
        const float v=float(std::atof(e)); b->intensity=v<0.0f?0.0f:(v>3.0f?3.0f:v);
    }
    b->blend_given=tune_vals[8]>=0.0f;
    b->blend=tune_vals[8]<0.0f?1.0f:(tune_vals[8]>1.0f?1.0f:tune_vals[8]);
    // EASU weights on the edit instead of bilinear; hot switch so the cost can be A/B-ed in game.
    b->easu=tune_vals[9]>=0.5f?1:0;
    b->fill_canvas=tune_vals[10]>=0.5f?1:0;
    b->dump_req=tune_vals[11]>=0.5f?1:0;
    b->tone_w   =tune_vals[12]<0.0f?0.0f:(tune_vals[12]>2.0f?2.0f:tune_vals[12]);
    b->texture_w=tune_vals[13]<0.0f?0.0f:(tune_vals[13]>2.0f?2.0f:tune_vals[13]);
    // Diagnostics for the composition knobs. They have been "set" and silently ineffective once
    // already, and from the outside there is no way to tell a wrong directory from a value being
    // overwritten by the defaults. One line per re-read, to stdout and to a file next to them
    // (the game has no console, so the file is the one that matters).
    {
        std::string line=std::string("dir=")+dir.string()+
            " ini="+(e1?"missing":"ok")+" tune="+(e2?"missing":"ok")+
            " detail="+std::to_string(b->detail)+" colour="+std::to_string(b->colour)+
            " guard="+std::to_string(b->guard)+" acut="+std::to_string(b->acutance)+
        " rcas="+std::to_string(b->rcas)+
            " delta="+(b->delta<0?std::string("auto"):std::to_string(b->delta))+
            " scale="+(b->scale_req>0.0?std::to_string(b->scale_req):std::string("installed"))+
            " hist="+(b->hist<0?std::string("auto"):std::to_string(b->hist))+
            " intensity="+std::to_string(b->intensity)+" blend="+(b->blend_given?std::to_string(b->blend):std::string("pkg"))+
            " tone="+std::to_string(b->tone_w)+" texture="+std::to_string(b->texture_w)+
            " ini_path="+ini_path.string()+"\n";
        std::printf("[tune] %s",line.c_str());
        if(FILE* lf=std::fopen((dir/"nr_tune_read.txt").string().c_str(),"a")){std::fputs(line.c_str(),lf);std::fclose(lf);}
    }
}

extern "C" int nrvk_backend_init(VkInstance instance,VkDevice device,VkPhysicalDevice phy,VkQueue queue,uint32_t family,const NRVK_Config* cfg,NRVK_Backend** out) {
    if(!out||!cfg||cfg->abi_size!=sizeof(*cfg)||!instance||!device||!phy||!queue){last_error="init external device/config missing";return -1;}*out=nullptr;
    NRVK_Backend* b=nullptr;int failure=-2;
    const auto t_init=std::chrono::steady_clock::now();
    auto since_init=[&]{return std::chrono::duration<double>(std::chrono::steady_clock::now()-t_init).count();};
    nr::logf("[nr] port: init from '%s'",cfg->manifest_dir);
    try {
        auto plan=checked_plan(cfg->manifest_dir);b=new NRVK_Backend();b->manifest_dir=cfg->manifest_dir;b->plan=std::move(plan);b->device=device;b->queue=queue;b->family=family;
        nr::logf("[nr] port: plan read, %zu buffers and %zu passes, %.0f ms",b->plan.buffers.size(),
                 b->plan.passes.size(),since_init()*1000.0);
        // The extent comes from the plan that is actually installed, not from a compile-time
        // constant: that is what makes a scale switch a runtime operation.
        b->geo=geometry_of(b->plan);b->variants.resize(kLadderCount);
        for(size_t i=0;i<kLadderCount;++i)
            if(ladder_plans[i].frame_w==b->geo.frame_w&&ladder_plans[i].frame_h==b->geo.frame_h) b->geo.scale=ladder_plans[i].scale;
        read_strengths(b);
        // Reject an extent the ladder does not carry *before* the device work below. This check used
        // to sit at the end of init, after 39 pipelines had been compiled and ~444 MB had been
        // allocated; the host then builds a fresh network for every extent the game reports, and
        // with dynamic resolution on (Witcher 3: [Rendering/DRS] Enable) that is a new extent every
        // few seconds - the 05:26-09:33 session ended with seven resident networks and 7 x 39 shader
        // compiles. That churn is the one load pattern that matches the Intel codegen abort in the
        // crash logs, so failing here in a millisecond instead of compiling is worth far more than
        // the nicer message it costs.
        int ladder_pick=-1;
        if(cfg->render_width>=8&&cfg->render_height>=8&&
           (cfg->render_width!=b->geo.frame_w||cfg->render_height!=b->geo.frame_h)){
            for(size_t i=0;i<kLadderCount;++i)
                if(ladder_plans[i].frame_w==cfg->render_width&&ladder_plans[i].frame_h==cfg->render_height){ladder_pick=int(i);break;}
            if(ladder_pick<0){
                std::string have;
                for(size_t i=0;i<kLadderCount;++i)
                    have+=(i?", ":"")+std::to_string(ladder_plans[i].frame_w)+"x"+std::to_string(ladder_plans[i].frame_h);
                failure=-8;
                throw std::runtime_error("no ladder plan for "+std::to_string(cfg->render_width)+"x"+
                    std::to_string(cfg->render_height)+"; have "+(have.empty()?std::string("none (ladder is empty)"):have));
            }
        }
        failure=-4;b->loader=LoadLibraryA("vulkan-1.dll");if(!b->loader)throw std::runtime_error("vulkan-1.dll missing");
        auto gipa=reinterpret_cast<PFN_vkGetInstanceProcAddr>(GetProcAddress(b->loader,"vkGetInstanceProcAddr"));
        auto gdpa=reinterpret_cast<PFN_vkGetDeviceProcAddr>(gipa(instance,"vkGetDeviceProcAddr"));
#define LOAD(f) b->f=reinterpret_cast<PFN_##f>(gdpa(device,#f));if(!b->f)throw std::runtime_error("missing device function " #f);
        DEVICE_FUNCTIONS(LOAD) b->vkCmdWriteTimestamp=reinterpret_cast<PFN_vkCmdWriteTimestamp>(gdpa(device,"vkCmdWriteTimestamp"));
#undef LOAD
        failure=-3;auto props=reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(gipa(instance,"vkGetPhysicalDeviceProperties2"));
        VkPhysicalDeviceProperties2 physical{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};props(phy,&physical);
        if(physical.properties.vendorID!=0x8086||!supported_intel_device(physical.properties.deviceID)||physical.properties.limits.maxComputeSharedMemorySize<49152)
            throw std::runtime_error("expected Intel Arc B580 (0xe20b) or B390 (0xb080), or NRVK_DEVICE_ID=<hex>; shared>=48KiB");
        failure=-4;auto features_fn=reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2>(gipa(instance,"vkGetPhysicalDeviceFeatures2"));
        VkPhysicalDeviceCooperativeMatrixFeaturesKHR cm{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
        VkPhysicalDeviceShaderFloat16Int8Features half{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES};
        VkPhysicalDevice16BitStorageFeatures storage{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES};
        VkPhysicalDeviceSubgroupSizeControlFeatures sg{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES};
        VkPhysicalDeviceSynchronization2Features sync{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES};
        VkPhysicalDeviceVulkanMemoryModelFeatures mm{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES};
        VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};features.pNext=&cm;cm.pNext=&half;half.pNext=&storage;storage.pNext=&sg;sg.pNext=&sync;sync.pNext=&mm;features_fn(phy,&features);
        if(!cm.cooperativeMatrix||!half.shaderFloat16||!storage.storageBuffer16BitAccess||!sg.subgroupSizeControl||!sg.computeFullSubgroups||!sync.synchronization2||!mm.vulkanMemoryModel)throw std::runtime_error("required FP16/SG32/coop/basicVMM support missing");
        auto getmem=reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties>(gipa(instance,"vkGetPhysicalDeviceMemoryProperties"));getmem(phy,&b->memory);
        nr::logf("[nr] port: device accepted (%04x:%04x, %.0f ms)",physical.properties.vendorID,
                 physical.properties.deviceID,since_init()*1000.0);
        failure=-8;init_resources(b);last_error.clear();*out=b;
        // The caller may already know which frame it is about to submit. When the ladder has a plan
        // for that exact extent and it is not the installed one, make it active now: every ladder
        // entry declares the same BUFFER lines, so this is one plan parse, not a reallocation.
        // A size the ladder does not carry is an explicit failure -- falling back to the installed
        // plan would dispatch a different extent than the frames arriving, which is the one thing
        // measured to corrupt the picture (see the scaled-dispatch note further down).
        if(ladder_pick>=0){
            if(!switch_scale(b,size_t(ladder_pick)))throw std::runtime_error("ladder switch failed: "+last_error);
            nr::logf("[nr] port: ladder -> %ux%u, canvas %ux%u",b->geo.frame_w,b->geo.frame_h,
                     b->geo.canvas_w,b->geo.canvas_h);
        }
        nr::logf("[nr] port: ready in %.1f s",since_init());
        return 0;
    } catch(const std::exception& e) {
        // release() overwrites last_error, which used to hide the real init failure behind a
        // "release queue completion failed" that had nothing to do with the cause.
        const std::string why=e.what(); std::fprintf(stderr,"[init] EXCEPTION: %s\n",why.c_str()); nrvk_backend_release(b); last_error=why;
        nr::logf("[nr] port: init failed after %.1f s: %s",since_init(),why.c_str());
        return failure;
    }
}

static bool crop_valid(const PortNetworkBridgeImage& x,uint32_t bx,uint32_t by,uint32_t w,uint32_t h) {
    return x.image && x.layout==VK_IMAGE_LAYOUT_GENERAL && x.valid_rect.offset.x>=0 && x.valid_rect.offset.y>=0 &&
        bx>=uint32_t(x.valid_rect.offset.x)&&by>=uint32_t(x.valid_rect.offset.y)&&w&&h&&
        uint64_t(bx)+w<=uint64_t(x.valid_rect.offset.x)+x.valid_rect.extent.width&&
        uint64_t(by)+h<=uint64_t(x.valid_rect.offset.y)+x.valid_rect.extent.height&&
        uint64_t(x.valid_rect.offset.x)+x.valid_rect.extent.width<=x.extent.width&&
        uint64_t(x.valid_rect.offset.y)+x.valid_rect.extent.height<=x.extent.height;
}
static bool frame_supported(NRVK_Backend* b,const PortNetworkBridgeFrameIO& io) {
    const auto& p=*io.v3.params;
    if(p.abi_size!=sizeof(p)||p.render_width<8||p.render_height<8||
       p.output_width<p.render_width||p.output_height<p.render_height||
       (!p.low_resolution_motion&&(p.output_width!=p.render_width||p.output_height!=p.render_height))||
       p.motion_jittered||p.jitter_x!=0.0f||p.jitter_y!=0.0f||
       p.motion_scale_origin<NRB_MV_DLSS_NGX||p.motion_scale_origin>NRB_MV_XESS_TO_NGX)return false;
    uint32_t bx[]={p.color_base_x,p.motion_base_x,p.depth_base_x,p.output_base_x};uint32_t by[]={p.color_base_y,p.motion_base_y,p.depth_base_y,p.output_base_y};
    for(unsigned i=0;i<4;++i){unsigned index=i==3?4:i;const auto& x=io.images[index];
        if(i==2&&!x.image)continue;
        const bool guide=(i==1||i==2);
        const uint32_t need_w=guide?x.extent.width:p.render_width;
        const uint32_t need_h=guide?x.extent.height:p.render_height;
        if(!crop_valid(x,bx[i],by[i],need_w,need_h)||x.queue_family!=b->family||!(x.usage&(i==3?VK_IMAGE_USAGE_STORAGE_BIT:VK_IMAGE_USAGE_SAMPLED_BIT)))return false;}
    const auto c=io.images[0].format,m=io.images[1].format;
    const auto d=io.images[2].format;
    const bool depth_ok=d==VK_FORMAT_R32_SFLOAT||d==VK_FORMAT_D32_SFLOAT||
                        d==VK_FORMAT_D32_SFLOAT_S8_UINT||d==VK_FORMAT_D24_UNORM_S8_UINT;
    // R11G11B10_FLOAT (DXGI 26) is the one addition to the shipped whitelist: the RE engine and
    // Forza both hand it over as the colour plane, and the old D3D12/ASI bridge always accepted
    // it -- only this Vulkan path listed RGBA16F/32F alone, which refused every frame with -6.
    const bool color_ok=c==VK_FORMAT_R16G16B16A16_SFLOAT||c==VK_FORMAT_R32G32B32A32_SFLOAT||
                        c==VK_FORMAT_B10G11R11_UFLOAT_PACK32;
    return color_ok&&
        (m==VK_FORMAT_R16G16_SFLOAT||m==VK_FORMAT_R32G32_SFLOAT)&&(!io.images[2].image||depth_ok)&&
        (io.images[4].format==VK_FORMAT_R16G16B16A16_SFLOAT||io.images[4].format==VK_FORMAT_R32G32B32A32_SFLOAT)&&io.nrb_controls.abi_size==sizeof(NRB_Controls)&&
        io.nrb_controls.input_height>0u&&io.nrb_controls.input_height<=p.render_height;
}
extern "C" void nrvk_backend_set_timestamp_query(NRVK_Backend* b,VkQueryPool pool,uint32_t base_query) { if(!b)return; b->ts_query_pool=pool; b->ts_base_query=base_query; }
static std::atomic<uint32_t> g_last_grid_w{0}, g_last_grid_h{0};
// Runtime experiment switch, read once per frame from <package>/config/nr_grid.txt
// so the reduced-grid modes can be A/B-ed inside one game session:
//   auto (default) - active = min(source, grid), dispatch scaled to active
//   floor          - as auto but floor() on the scaled grid counts
//   x              - scale only gx, keep gy at the plan value
//   y              - scale only gy, keep gx at the plan value
//   gz             - as auto plus gz scaled by the y ratio
//   full           - active = min(source, grid), dispatch the whole plan grid
//   off            - active = full 1280x720 contract grid (pre-adaptive path)
static std::string grid_mode(const std::string& manifest_dir) {
    if (manifest_dir.empty()) return {};
    std::ifstream file(std::filesystem::path(manifest_dir).parent_path() / "config" / "nr_grid.txt");
    std::string mode;
    if (!(file >> mode) || mode.empty()) return {};
    for (auto& ch : mode) ch = char(std::tolower(static_cast<unsigned char>(ch)));
    return mode;
}
extern "C" void nrvk_backend_last_grid(uint32_t* width, uint32_t* height) {
    if (width) *width = g_last_grid_w.load(std::memory_order_relaxed);
    if (height) *height = g_last_grid_h.load(std::memory_order_relaxed);
}
// Offline diagnostic, no effect unless NRVK_SNAP is set: `<pass index>:<buffer name>`
// publishes that buffer as the frame right after the named pass. It bounces through
// `ioport_70_ioport_blended` because the final pass overwrites `frame_out` and there is
// no spare allocation in the plan - the blended port is written by the last pass anyway,
// so borrowing it cannot change what we are looking at. Everything downstream of the
// snapshot sees corrupted input and is meant to; scatter publishes the snapshot.
static int g_snap_pass = -2;
static std::string g_snap_buf;
static bool g_snap_parsed = false;
// Make ladder entry `idx` the active plan. Buffers and descriptor sets are shared - every ladder
// plan declares the same BUFFER and BIND lines - so what changes is the per-pass pushes, the
// history offsets, the output binding and the extent. Temporal state is canvas-sized, so it is
// dropped: a switch is a scene cut as far as the network is concerned.
static bool switch_scale(NRVK_Backend* b,size_t idx) {
    if(!b||idx>=kLadderCount)return false;
    Plan& v=b->variants[idx];
    if(v.passes.empty()) {
        const auto path=std::filesystem::path(b->manifest_dir)/ladder_plans[idx].path;
        std::error_code ec;
        if(!std::filesystem::is_regular_file(path,ec)||sha256(path)!=ladder_plans[idx].sha) {
            last_error=std::string("ladder plan missing or changed: ")+ladder_plans[idx].path;return false;
        }
        try { v=read_plan(path.string()); }
        catch(const std::exception& e) { last_error=e.what();return false; }
        if(v.passes.size()!=b->pass_sets.size()||v.history.size()!=kPlanHistory||v.outputs.size()!=kPlanOutputs) {
            last_error="ladder plan shape differs from the installed one";return false;
        }
        v.buffers.clear();   // shared with the active plan
    }
    b->plan.passes=v.passes;
    b->plan.history=v.history;
    b->plan.outputs=v.outputs;
    for(size_t i=0;i<b->plan.passes.size();++i) { b->plan.passes[i].set=b->pass_sets[i]; b->plan.passes[i].pipeline=b->pass_pipelines[i]; }
    b->geo=geometry_of(b->plan);
    b->geo.scale=ladder_plans[idx].scale;
    b->history_committed=false;b->last_render_width=0;b->last_render_height=0;b->hist_probe=0;
    b->clear_all_pending=true;
    return true;
}
// Offline diagnostics: hand a plan buffer's handle to the caller so it can copy the raw bytes
// out with its own device. Reading a buffer through the scatter hides exactly the defects worth
// looking at (fp16 rounding, clamping, the composite itself).
extern "C" int nrvk_backend_buffer_info(NRVK_Backend* b, const char* name, void** handle,
                                        unsigned long long* bytes) {
    if(!b||!name) return -1;
    if(name[0]=='@'){   // @output / @base: the buffers the scatter actually samples
        const size_t idx = (name[1]=='o') ? b->output : 0;
        if(name[1]=='b'){ if(handle)*(void**)handle=(void*)b->base_copy; if(bytes)*bytes=(unsigned long long)b->base_copy_bytes; return 0; }
        auto& x=b->plan.buffers[idx]; if(handle)*(void**)handle=(void*)x.handle; if(bytes)*bytes=(unsigned long long)x.bytes; return 0; }
    for(auto& x : b->plan.buffers)
        if(x.name == name) { if(handle) *handle = (void*)x.handle; if(bytes) *bytes = (unsigned long long)x.bytes; return 0; }
    return -2;
}


// In-game capture: `dump=1` in nr_tune.txt writes one frame's data next to the manifest -
//   nr_dump_input.bin   the ioport bundle the network reads (colour / motion / depth planes)
//   nr_dump_answer.bin  frame_out, i.e. what the port published
//   nr_dump.txt         geometry and knob values, so the dump explains itself
// It runs at the top of the *next* frame on its own command buffer and fence, because by then the
// previous frame's buffers are final. One shot: the request is consumed, so leaving dump=1 in the
// file does not keep writing.
static void capture_dump(NRVK_Backend* b) {
    struct Item { const char* name; VkBuffer buf; VkDeviceSize bytes; };
    const Item items[2] = {
        {"input",  b->plan.buffers[b->pre].handle,    b->plan.buffers[b->pre].bytes},
        {"answer", b->plan.buffers[b->output].handle, b->plan.buffers[b->output].bytes},
    };
    VkDeviceSize need = 0;
    for (const auto& it : items) if (it.bytes > need) need = it.bytes;
    if (b->dump_buf == VK_NULL_HANDLE) {
        Buffer t; t.name = "dump_staging"; t.bytes = need;
        make_buffer(b, t, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        b->dump_buf = t.handle; b->dump_mem = t.memory;
        if (b->vkMapMemory(b->device, b->dump_mem, 0, VK_WHOLE_SIZE, 0, &b->dump_map) != VK_SUCCESS) {
            b->dump_map = nullptr; return;
        }
    }
    if (b->dump_pool == VK_NULL_HANDLE) {
        VkCommandPoolCreateInfo cp{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        cp.queueFamilyIndex = b->family; cp.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        if (b->vkCreateCommandPool(b->device, &cp, nullptr, &b->dump_pool) != VK_SUCCESS) return;
        VkCommandBufferAllocateInfo ca{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ca.commandPool = b->dump_pool; ca.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ca.commandBufferCount = 1;
        if (b->vkAllocateCommandBuffers(b->device, &ca, &b->dump_cb) != VK_SUCCESS) return;
        VkFenceCreateInfo fc{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        if (b->vkCreateFence(b->device, &fc, nullptr, &b->dump_fence) != VK_SUCCESS) return;
    }
    const auto dir = std::filesystem::path(b->manifest_dir).parent_path();
    // The dump used to copy the buffers straight away and then wait only for its *own* fence. That
    // fence orders the copy against the queue, but the network's command buffer is submitted by the
    // bridge on the game's timeline, so on a busy frame the copy could read a buffer that was still
    // being written - which is exactly the "one row in N is right, the rest is scrambled" picture
    // every dump has shown. A one-shot diagnostic may as well stall: wait for the queue to go idle
    // (the network records on it) before taking the snapshot.
    if (b->vkQueueWaitIdle) b->vkQueueWaitIdle(b->queue);
    for (const auto& it : items) {
        if (!it.buf || it.bytes > need) continue;
        b->vkResetCommandBuffer(b->dump_cb, 0);
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (b->vkBeginCommandBuffer(b->dump_cb, &bi) != VK_SUCCESS) return;
        VkBufferCopy copy{0, 0, it.bytes};
        b->vkCmdCopyBuffer(b->dump_cb, it.buf, b->dump_buf, 1, &copy);
        if (b->vkEndCommandBuffer(b->dump_cb) != VK_SUCCESS) return;
        b->vkResetFences(b->device, 1, &b->dump_fence);
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1; si.pCommandBuffers = &b->dump_cb;
        if (b->vkQueueSubmit(b->queue, 1, &si, b->dump_fence) != VK_SUCCESS) return;
        if (b->vkWaitForFences(b->device, 1, &b->dump_fence, VK_TRUE, 5ull * 1000 * 1000 * 1000) != VK_SUCCESS) return;
        const auto path = (dir / (std::string("nr_dump_") + it.name + ".bin")).string();
        if (FILE* f = std::fopen(path.c_str(), "wb")) { std::fwrite(b->dump_map, 1, size_t(it.bytes), f); std::fclose(f); }
    }
    if (FILE* f = std::fopen((dir / "nr_dump.txt").string().c_str(), "w")) {
        std::fprintf(f, "frame=%ux%u canvas=%ux%u scale=%.2f delta=%d intensity=%.3f easu=%d\n",
                     b->geo.frame_w, b->geo.frame_h, b->geo.canvas_w, b->geo.canvas_h,
                     b->geo.scale, b->delta, double(b->intensity), b->easu);
        std::fprintf(f, "input_bytes=%llu answer_bytes=%llu\n",
                     (unsigned long long)b->plan.buffers[b->pre].bytes,
                     (unsigned long long)b->plan.buffers[b->output].bytes);
        std::fclose(f);
    }
    std::printf("[dump] wrote input + answer (frame %llu)\n", (unsigned long long)b->dump_at);
}

// One-shot audit line per working scale. The offline runner is fed the same plan file, so if the
// hashes below differ from the ones computed off that file, the runtime supplied a different push
// set than the plan says - which is the only way a switched frame can differ from the same plan run
// offline. Printed once per scale (and again after every switch), through nr::logf so it lands in
// the game's log instead of a console nobody sees.
static double g_audit_scale=-1.0;
static unsigned long long g_audit_gather=0;
// One frame's worth of per-pass push hashes, so the log can be diffed line by line against the
// plan file (tools: the same FNV-1a over each PUSH block). Anything the runtime supplies that the
// plan does not say shows up as the first differing line.
static int g_push_log_budget=0;
static double g_push_log_scale=-2.0;
static void audit_scale_once(NRVK_Backend* b,uint32_t active_w,uint32_t active_h,
                             const uint32_t* sp,size_t sn) {
    if(!b||b->plan.passes.empty()) return;
    if(b->geo.scale==g_audit_scale) return;
    g_audit_scale=b->geo.scale;
    auto h=[&](const uint32_t* p,size_t n){unsigned long long x=1469598103934665603ull;
        for(size_t i=0;i<n;++i){x^=p[i];x*=1099511628211ull;}return x;};
    const auto& f=b->plan.passes.front();const auto& l=b->plan.passes.back();
    nr::logf("[nr] port: audit scale=%.2f frame=%ux%u canvas=%ux%u active=%ux%u ioport m=%u d=%u "
             "gather=%016llx scatter=%016llx push0=%016llx g0=%ux%u pushN=%016llx gN=%ux%u",
             b->geo.scale,b->geo.frame_w,b->geo.frame_h,b->geo.canvas_w,b->geo.canvas_h,
             active_w,active_h,b->geo.ioport_motion,b->geo.ioport_depth,
             g_audit_gather,h(sp,sn),h(f.push.data(),f.push.size()),f.gx,f.gy,
             h(l.push.data(),l.push.size()),l.gx,l.gy);
}

extern "C" int nrvk_backend_frame(NRVK_Backend* b,VkCommandBuffer cb,const NRVK_BackendFrameIO* base) {
    if(!b||!cb||!base||!base->params){last_error="frame context/cb/params missing";return -1;}
    if(base->abi_size!=sizeof(PortNetworkBridgeFrameIO)){last_error="v3 lacks actual image metadata/frame_id/control_epoch/controls20: use independent PortNetworkBridgeFrameIO tail";return -6;}
    if(base->io_view_count!=4||!base->io_views||!base->io_views[0]||!base->io_views[1]||!base->out_view){last_error="color/motion/output views missing";return -1;}
    const auto& io=*reinterpret_cast<const PortNetworkBridgeFrameIO*>(base);
    if(!frame_supported(b,io)){last_error="requires adaptive render-size input (>=8x8)/input_height720 controls, float inputs/RGBA16F or32F output at render size, GENERAL crops, normalized LDR proxy(exposure1,jitter0,non-jittered native MV), known NGX motion origin";return -6;}
    if(io.images[2].image&&!base->io_views[2]){last_error="actual depth image requires its sampled view";return -1;}
    // `record_pending` alone is the honest gate. The completion fence is *not* usable here: the
    // signal the host submits is queued behind the cut command buffer, which the bridge submits on
    // the game's timeline *after* `nrvk_backend_frame` returns, so asking the fence "is the network
    // done" at the top of the next frame answers no for as long as that list has not run - and once
    // one frame is refused, nothing else is submitted, so it never runs: a permanent -7.
    // (Measured 05:40-07:55: 7 frames ran, then 4386 refusals in a row - NR silently off, which is
    // exactly the "the picture got worse again" report.)
    if(b->record_pending){
        // A record that was never handed back is the one thing that must block the next frame, but
        // it must not block it forever: if the host lost a frame (game paused on a loading screen,
        // feature recreated before submit), drop the stale record after ~4 s and keep the network
        // alive instead of running without NR for the rest of the session.
        if(++b->record_stuck > 120){
            b->record_stuck=0; b->record_pending=false;
            nr::logf("[nr] port: dropping a record that was never submitted after 120 frames, continuing");
        } else {
            last_error="prior single-context network not completed or record not submitted";return -7;
        }
    } else b->record_stuck=0;
    // The dump waits for the network to leave its building phase: an early frame would just
    // snapshot the passthrough copy and read as "the network changes nothing".
    if(b->dump_pending&&b->ticks>=b->dump_at){b->dump_pending=false;capture_dump(b);}
    const bool have_depth=io.images[2].image!=VK_NULL_HANDLE;
    if (!g_snap_parsed) {
        g_snap_parsed = true;
        if (const char* v = std::getenv("NRVK_SNAP")) {
            if (const char* c = std::strchr(v, ':')) {
                g_snap_pass = std::atoi(v);
                g_snap_buf.assign(c + 1);
            }
        }
    }
    VkDescriptorImageInfo images[4]={{b->sampler,base->io_views[0],VK_IMAGE_LAYOUT_GENERAL},{b->sampler,base->io_views[1],VK_IMAGE_LAYOUT_GENERAL},{b->sampler,have_depth?base->io_views[2]:base->io_views[0],VK_IMAGE_LAYOUT_GENERAL},{VK_NULL_HANDLE,base->out_view,VK_IMAGE_LAYOUT_GENERAL}};
    VkDescriptorBufferInfo buffers[4]={{b->plan.buffers[b->pre].handle,0,b->plan.buffers[b->pre].bytes},{b->plan.buffers[b->post].handle,0,b->plan.buffers[b->post].bytes},{b->plan.buffers[b->output].handle,0,b->plan.buffers[b->output].bytes},{b->base_copy,0,b->base_copy_bytes}};
    // 11 writes: gather 0-2 images, gather 3-5 buffers (pre, post, base plane), scatter 0-4
    // (output plane, frame image, pre fallback, colour sampler, base plane). The count is a named
    // constant because this has already been shipped one short: `vkUpdateDescriptorSets` with a
    // count smaller than the loop that filled the array leaves the tail bindings unwritten, and a
    // descriptor that was never written reads as a constant zero with no API error anywhere.
    VkWriteDescriptorSet writes[kDescriptorWrites]{};
    for(unsigned i=0;i<kDescriptorWrites;++i){auto& w=writes[i];w.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;w.descriptorCount=1;
        if(i<3){w.dstSet=b->gather_set;w.dstBinding=i;w.descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;w.pImageInfo=&images[i];}
        else if(i==3){w.dstSet=b->gather_set;w.dstBinding=3;w.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;w.pBufferInfo=&buffers[0];}
        else if(i==4){w.dstSet=b->gather_set;w.dstBinding=4;w.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;w.pBufferInfo=&buffers[1];}
        else if(i==5){w.dstSet=b->gather_set;w.dstBinding=5;w.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;w.pBufferInfo=&buffers[3];}
        else if(i==6){w.dstSet=b->scatter_set;w.dstBinding=0;w.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;w.pBufferInfo=&buffers[2];}
        else if(i==7){w.dstSet=b->scatter_set;w.dstBinding=1;w.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;w.pImageInfo=&images[3];}
        else if(i==8){w.dstSet=b->scatter_set;w.dstBinding=2;w.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;w.pBufferInfo=&buffers[0];}
        else if(i==9){w.dstSet=b->scatter_set;w.dstBinding=3;w.descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;w.pImageInfo=&images[0];}
        else{w.dstSet=b->scatter_set;w.dstBinding=4;w.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;w.pBufferInfo=&buffers[3];}}
    b->vkUpdateDescriptorSets(b->device,kDescriptorWrites,writes,0,nullptr);
    auto write_ts=[&](VkPipelineStageFlagBits stage,uint32_t offset){ if(b->ts_query_pool!=VK_NULL_HANDLE&&b->vkCmdWriteTimestamp) b->vkCmdWriteTimestamp(cb,stage,b->ts_query_pool,b->ts_base_query+offset); };
    write_ts(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,1);
    // Global compute execution dependency preserves pool alias WAR ordering
    // across submissions. WB's producer wait/acquire precedes this segment.
    barrier(b,cb,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT|VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_MEMORY_READ_BIT|VK_ACCESS_MEMORY_WRITE_BIT,VK_ACCESS_TRANSFER_READ_BIT|VK_ACCESS_TRANSFER_WRITE_BIT|VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT);
    // Runtime scale. This has to run before the scratch clear below, otherwise the frame that
    // performs the switch still computes with the previous plan's memory and pollutes the
    // history the next frame starts from.
    if((++b->ticks%30u)==1u) read_strengths(b);
    // The model extent is exactly AMD's rule, with no alignment: round(render * scale)
    // (windows/src/core/nr_renodx.cpp:1108 - `*nw = lround(rw * resolution_scale)`). The table is
    // searched for that extent, and the switch happens when the *plan* behind it differs - so
    // `scale=1.0` can replace the installed padded plan with an unpadded one of the same extent
    // (the 12% canvas tax disappears without touching the picture).
    if(b->scale_req>0.0){
        const uint32_t want_w=uint32_t(std::lround(double(base->params->render_width )*b->scale_req));
        const uint32_t want_h=uint32_t(std::lround(double(base->params->render_height)*b->scale_req));
        // Two plans can share an extent: the installed follow plan (padded canvas) and a
        // working-scale plan (canvas == frame). Prefer the unpadded one - it is the same picture
        // for less work, and it is the one whose HISTORY/plane strides the model's own extent
        // matches exactly.
        size_t best=kLadderCount;
        for(size_t i=0;i<kLadderCount;++i){
            if(ladder_plans[i].frame_w!=want_w||ladder_plans[i].frame_h!=want_h)continue;
            if(ladder_plans[i].canvas_w==want_w&&ladder_plans[i].canvas_h==want_h){best=i;break;}
            if(best==kLadderCount)best=i;
        }
        if(best>=kLadderCount){
            if(b->scale_miss_w!=want_w||b->scale_miss_h!=want_h){   // once per missing extent
                b->scale_miss_w=want_w;b->scale_miss_h=want_h;
                nr::logf("[nr] port: [scale] no ladder plan for %.2f -> %ux%u",b->scale_req,want_w,want_h);
            }
        } else if(ladder_plans[best].canvas_w!=b->geo.canvas_w||ladder_plans[best].canvas_h!=b->geo.canvas_h){
            const double was=b->geo.scale;
            if(ladder_plans[best].canvas_w*ladder_plans[best].canvas_h>
               uint64_t(b->installed_canvas_w)*b->installed_canvas_h)
                nr::logf("[nr] port: [scale] warning: canvas grows past the installed plan; if the "
                         "image looks wrong, reinstall this scale with tools/set_scale.py");
            if(switch_scale(b,best)) nr::logf("[nr] port: [scale] %.2f -> %.2f (%ux%u canvas %ux%u)",
                was,ladder_plans[best].scale,b->geo.frame_w,b->geo.frame_h,b->geo.canvas_w,b->geo.canvas_h);
            else nr::logf("[nr] port: [scale] refused: %s",last_error.c_str());
        }
    }
    if(b->clear_all_pending){
        b->clear_all_pending=false;
        // Zero-init means exactly what init_resources() did: every buffer with no cold file.
        // The temporal planes (HISTORY source/destination) are *resident* with path "-", so a
        // filter on !resident alone left them holding the previous canvas' contents and the
        // network read a mis-strided history - the switched frame came out with the lower third
        // turned into stripes while the same plan *installed* was clean.
        barrier(b,cb,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_ACCESS_MEMORY_WRITE_BIT,VK_ACCESS_TRANSFER_WRITE_BIT);
        for(auto& x:b->plan.buffers) if(x.handle&&(!x.resident||x.path=="-")) b->vkCmdFillBuffer(cb,x.handle,0,VK_WHOLE_SIZE,0);
        barrier(b,cb,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT);
    }
    float controls[20];memcpy(controls,io.model_controls,sizeof(controls));
    controls[7]=b->intensity;   // see intensity= above: the host's 1.0 is past the model's range
    {   // NRVK_INTENSITY also applies when the caller never ran `read_strengths` (the offline
        // runner builds the backend directly, so the tune-file override above never runs there and
        // the harness could not A/B this control at all).
        static const float override_value=[](){const char* e=std::getenv("NRVK_INTENSITY");
            return e?float(std::atof(e)):-1.0f;}();
        if(override_value>=0.0f) controls[7]=override_value;
    }
    {   // NRVK_DETAIL: the scatter's TransferStrength (pc.hdr.y in the composite). Unlike c[7] this
        // one provably moves the picture (detail>1 is a power on the luma ratio, <1 mixes back
        // toward the frame), so the harness needs it to answer "how strong is NR really".
        static const float detail_override=[](){const char* e=std::getenv("NRVK_DETAIL");
            return e?float(std::atof(e)):-1.0f;}();
        if(detail_override>=0.0f) b->detail=detail_override;
    }
    {   // NRVK_BLEND: the package's own blend_scale (4-byte cold buffer, fp16 0x3A23 = 0.7670898
        // for the Small model - mismatched by kPostBlendScale 0.73974609 for the full one). AMD
        // feeds `history_strength * post_blend_scale` to the pass that uses it; we overwrite the
        // whole scalar, so the default has to be the model's constant, not 1.
        static const float blend_override=[](){const char* e=std::getenv("NRVK_BLEND");
            return e?float(std::atof(e)):-1.0f;}();
        if(blend_override>=0.0f) b->blend=blend_override;
    }
    const auto& p=*base->params;
    // HDR framing (gather.comp/neutral). The loader decides whether the frame is HDR and
    // hands over its exposure in pre_exposure; 0 means "display-referred, leave the pixels
    // alone", which is the entire SDR path and must stay bit-identical. An HDR frame is
    // tone-shaped on the way in and un-shaped on the way out, so it keeps its own units.
    float hdr_gain=0.0f;
    if(p.hdr_input){
        hdr_gain=p.pre_exposure>0.0f?p.pre_exposure:1.0f;
        if(p.exposure_scale>0.0f) hdr_gain*=p.exposure_scale;
        if(!(hdr_gain>1e-6f)||!(hdr_gain<1e6f)) hdr_gain=1.0f;
        if(const char* e=std::getenv("NRVK_HDR_GAIN")){const float v=float(std::atof(e));if(v>0.0f)hdr_gain=v;}
    }
    uint32_t hdr_word=0;memcpy(&hdr_word,&hdr_gain,sizeof(hdr_word));
    // NR grid adaptation. The network keeps its frozen 1280x720 canvas and
    // stride; only the active top-left region of that canvas is gathered,
    // dispatched and scattered. Source sizes at or below the contract place the
    // frame 1:1 and cost pixels*ratio^2 of the dispatches; larger sources keep
    // the full grid (stretch contract unchanged). active==1280x720 reproduces
    // the upstream identity path bit-exactly, so this cannot regress it.
    // Aspect-preserving placement: the frame is scaled uniformly to the largest
    // rect that fits the frozen canvas and centred. A 16:10 source therefore
    // keeps its proportions instead of being squashed into 16:9, which shows up
    // as a rubbery/jelly look after scatter stretches it back. Sources at or
    // below the contract are never upscaled (fit is capped at 1.0).
    const uint32_t canvas_w=b->geo.canvas_w, canvas_h=b->geo.canvas_h;
    // The bridge's own extent is the **frame**, never the canvas. The plan's canvas is the
    // network's internal dispatch domain (padded to 64-multiples), while the ioport planes the port
    // reads and writes are frame-sized and frame-strided - measured: the port's plane offsets are
    // 4*frame_px, and the offline runner injects a frame at the origin with a frame-width stride.
    // Filling the canvas instead only looks right while canvas_w == frame_w (0.5x and 0.8x); at
    // 1.0x, where 1280 pads to 1344, it sheared every row by 64 texels.
    uint32_t active_w=b->geo.frame_w,active_h=b->geo.frame_h,active_x=0,active_y=0;
    const std::string mode=grid_mode(b->manifest_dir);
    if(const char* grid_env=std::getenv("NRVK_GRID_SCALE")){
        const double scale=std::atof(grid_env);
        if(scale>0.05&&scale<1.0){
            active_w=std::max(16u,uint32_t(std::lround(double(active_w)*scale)));
            active_h=std::max(16u,uint32_t(std::lround(double(active_h)*scale)));
        }
    }
    (void)mode;   // dispatch scaling experiments belonged to the canvas-era geometry
    // `fill=1` makes the bridge cover the whole canvas instead of only the frame rect. At
    // 0.5/0.8 the two are the same width and it is a no-op; at 1.0 the canvas is 1344x768 and
    // the model's pyramid reads across the 1280x720 rect either way, so this decides whether
    // the padding is edge-free garbage or a stretched copy of the frame.
    if(b->fill_canvas){active_w=canvas_w;active_h=canvas_h;active_x=0;active_y=0;}
    // NRB scales are already current->previous source pixels, including route
    // sign/unit normalization. Apply them once; origin is provenance, no flip.
    controls[11]=have_depth?1.0f:0.0f;controls[12]=p.depth_inverted?1.0f:0.0f;
    // ctl(13)/ctl(14) mean "texture value -> model pixels" and the caller has already resolved that
    // word (resolve_controls: the motion scale in source pixels divided by the extent it is
    // expressed in). The runtime seam states p.motion_scale_x *normalised* - ngx_scale/extent,
    // 960/960 = 1.0 - so recomputing it here divided by frame_w a second time and every history tap
    // moved 1/960 of the truth: a static view was fine (mv==0) while any camera motion ghosted,
    // which is exactly the symptom this backend was blamed for. Prefer the caller's word; keep the
    // old expression only for a caller that left the slot empty. mv_scale stays the hot knob
    // (clamped to [0.05, 8.0] above), applied last so a tune file can still scale the true value.
    const bool zero_motion=io.nrb_controls.history==NRB_HISTORY_ZERO_MOTION;
    const float ctl13=io.model_controls[13]!=0.0f?io.model_controls[13]:p.motion_scale_x/float(b->geo.frame_w);
    const float ctl14=io.model_controls[14]!=0.0f?io.model_controls[14]:p.motion_scale_y/float(b->geo.frame_h);
    controls[13]=zero_motion?0.0f:ctl13*b->mv_scale;
    controls[14]=zero_motion?0.0f:ctl14*b->mv_scale;
    // One line every 120 frames: the running values, so "does this flip mid-scene" is answerable
    // from the log instead of from another blind A/B. ctl13 is printed beside c13 so the caller's
    // resolution and the applied (after mv_scale) value can never be confused again.
    if((b->ticks%120u)==0u){
        nr::logf("[nr] mvprobe: nrb=%u mvs=%.2f ms=%.6f,%.6f ctl13=%.6f c13=%.8f c14=%.8f fw=%u fh=%u cw=%u ch=%u",
                 unsigned(io.nrb_controls.history),b->mv_scale,
                 p.motion_scale_x,p.motion_scale_y,ctl13,controls[13],controls[14],
                 b->geo.frame_w,b->geo.frame_h,b->geo.canvas_w,b->geo.canvas_h);
    }
    // ctl(9)/ctl(10) are the *history image* extent, not the dispatch canvas. Only f01 reads them
    // (nrf_rgba_clamped: W/H build the history uv and the clamp rect), and the history plane it
    // samples is one RGBA f32 image at the **frame** extent - that is what the HISTORY copy
    // (16*frame_px bytes) and the gather's frame-strided planes write. Feeding the padded canvas
    // (1344x768 for a 1280x720 frame) made every history tap read 5% too far right and shear a
    // row every frame, which is exactly the "denoise looks like it is not running" symptom.
    // The offline runner (the reference this port is validated against) passes the frame here.
    controls[9]=float(b->geo.frame_w); controls[10]=float(b->geo.frame_h);
    controls[15]=controls[16]=1.0f;controls[17]=controls[18]=0.0f;
    // Temporal subsampling ("skip2"): run the network on alternate frames and
    // let scatter re-publish the retained result on the frames in between. The
    // plan, its geometry and its push constants are untouched - this only
    // changes how often the chain runs. Guarded so the first frame after a
    // reset, a render-size change or a mode switch always processes.
    const bool skip_eligible=(mode=="skip2")&&b->history_committed&&
        p.render_width==b->last_render_width&&p.render_height==b->last_render_height;
    const bool skip_network=skip_eligible&&((io.frame_id&1u)==0u);
    // History continues only on consecutive frames of the same source size; a
    // render-size change invalidates the temporal state. Reduced grids also
    // disable it: the history buffers span the full canvas and the region
    // outside the active rect is never refreshed.
    // Temporal history is per-frame-sized (the plane it accumulates is), so it is no longer tied to
    // the canvas covering the active rect: that gate switched history off at every native scale,
    // which is exactly where a model that builds detail over frames needs it.
    bool history_allowed=true;
    // `hist=` overrides the geometry gate: at the native contract the canvas is padded (1344x768 for
    // a 1280x720 frame), so the gate is false and the model runs with no temporal accumulation at
    // all. That is a suspicious default for a model whose detail is supposed to build up over
    // frames, so the override exists to test it: -1 auto, 0 off, 1 forced on.
    if(b->hist>=0) history_allowed=b->hist!=0;
    if(const char* hist_env=std::getenv("NRVK_GRID_HISTORY"))history_allowed=std::atoi(hist_env)!=0;
    // Under skip2 the committed frame advances by two, so the continuation test
    // must expect a step of two; otherwise every processed frame looks like a
    // discontinuity and the model resets its temporal state each time.
    const uint64_t history_step=(mode=="skip2")?2ull:1ull;
    // The continuation test used to be `frame_id == committed_frame + 1` *and* `history_committed`.
    // Both halves were fatal in the port host: it spelled `ext.frame_id = 0` on every frame, so the
    // equality could never hold, and `history_committed` is only set by `submit_result` - one
    // missing bookkeeping call and the network spends its life as a single-frame spatial filter
    // (the "NR is running but nothing is denoised" symptom, `history=0` in every log line).
    // What the model actually needs is: this is a *later* frame than the last one we ran, at the
    // same extent, with no reset in between. `last_render_width` is cleared on every switch, so the
    // first frame after a switch still runs without history and the second one starts accumulating.
    // `hist=1` means *force*: the host asks for a reset on frames where the picture is still a
    // valid continuation (the NGX feature is re-created on every scale change, and W3 asks again
    // after each cutscene), and one reset per frame is indistinguishable from "history off for
    // ever" - every log line then reads history=0 with every other term correct. The override
    // exists so the temporal path can be measured at all; `hist=auto` still honours the reset.
    const bool history=!skip_network&&history_allowed&&
        (b->hist>0||(!p.reset_history&&io.nrb_controls.history!=NRB_HISTORY_RESET))&&
        io.control_epoch==b->committed_epoch&&
        b->last_render_width!=0&&io.frame_id>b->committed_frame&&
        p.render_width==b->last_render_width&&p.render_height==b->last_render_height;
    if(!history)controls[8]=0.0f;
    // Offline diagnostic (never set in the game): NRVK_CTL=<index>:<value> overwrites one word of
    // the 20-word control block, so each slot can be swept to find which one the network actually
    // reads. The `intensity=` knob was measured dead (0/0.25/0.5/1.0 produce byte-identical
    // answers), which means c[7] is not the slider this model listens to - this finds the one it is.
    if(const char* ctl=std::getenv("NRVK_CTL")){
        int idx=-1; float val=0.0f;
        if(std::sscanf(ctl,"%d:%f",&idx,&val)==2&&idx>=0&&idx<20)controls[idx]=val;
    }
    // A switched frame runs without history for one frame and then commits; if it never comes back
    // on, the picture silently loses all temporal denoise while every other log line still looks
    // healthy. Print the decision inputs for the first frames after each extent change.
    // The probe used to stop after 8 frames, which hid the case this exists for: history is on for
    // the first frames and then drops out for the rest of the session (one term flips, the log stays
    // silent). Now it also prints whenever the decision changes and once every ~600 frames, so a
    // session can be read end to end.
    if(b->hist_probe<8||int(history)!=b->hist_logged||(b->ticks%600u)==0u){
        ++b->hist_probe;
        b->hist_logged=int(history);
        // Every term of the decision, because "history=0" alone cannot tell a reset from a
        // mis-set knob and both look identical on screen (no temporal accumulation at all).
        nr::logf("[nr] port: history=%d (committed=%d frame %llu vs %llu+%llu, epoch %d/%d, "
                 "%ux%u vs %ux%u) reset=%d nrb=%d hist_knob=%d allowed=%d skip=%d",
                 int(history),int(b->history_committed),
                 (unsigned long long)io.frame_id,(unsigned long long)b->committed_frame,
                 (unsigned long long)history_step,io.control_epoch,b->committed_epoch,
                 p.render_width,p.render_height,b->last_render_width,b->last_render_height,
                 int(p.reset_history),int(io.nrb_controls.history),b->hist,int(history_allowed),
                 int(skip_network));
    }
    if(b->blend_idx>=0&&b->blend_given){float alpha=b->blend;
        b->vkCmdUpdateBuffer(cb,b->plan.buffers[b->blend_idx].handle,0,sizeof(alpha),&alpha);}
    for(auto& h:b->plan.history){if(history){VkBufferCopy copy{h.sourceOffset,h.destinationOffset,h.bytes};b->vkCmdCopyBuffer(cb,b->plan.buffers[h.source].handle,b->plan.buffers[h.destination].handle,1,&copy);}
        b->vkCmdUpdateBuffer(cb,b->plan.buffers[h.destination].handle,h.controlsOffset,sizeof(controls),controls);}
    barrier(b,cb,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT);
    write_ts(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,2);
    auto& pre=b->plan.passes.front().push;auto& post=b->plan.passes.back().push;
    // IO-buffer offsets from the ioport contract. These are layout constants, not
    // indices into the unrelated first-pass push constants: the planes sit at
    // kU, kU+20, 2*kU+20 and 2.5*kU+20 with kU = frame pixels in floats, each
    // writing kU pixels of RGBA32F / RG16F / R32F. freeze_plan.py derives them from
    // the plan so a smaller frame does not read another frame's planes.
    constexpr uint32_t ioport_color_float_offset=0;
    const uint32_t ioport_motion_float_offset=b->geo.ioport_motion;
    const uint32_t ioport_depth_float_offset=b->geo.ioport_depth;
    // A missing or unreadable switch file must land on the safe mode, never on
    // the scaled one: scaling the plan's dispatch is known to corrupt the image.
    // skip2 lowers the run rate only; it must never enable the scaled dispatch
    // path, which is known to corrupt the image. Only the explicit experiment
    // modes (auto/floor/x/y/gz) scale the plan grid.
    const bool dispatch_full=(mode.empty()||mode=="full"||mode=="skip2");
    const bool scale_x=!dispatch_full&&(mode!="y");
    const bool scale_y=!dispatch_full&&(mode!="x");
    const bool use_floor=(mode=="floor");
    const bool scale_z=(mode=="gz");
    const auto scaled_count=[&](uint32_t plan_count,double ratio,bool enabled){
        if(!enabled) return plan_count;
        const double scaled=double(plan_count)*ratio;
        const uint32_t want=use_floor?uint32_t(std::floor(scaled)):uint32_t(std::ceil(scaled));
        return std::max(1u,std::min<uint32_t>(plan_count,want));
    };
    const double grid_sx=scale_x?double(active_w)/double(b->geo.frame_w):1.0;
    const double grid_sy=scale_y?double(active_h)/double(b->geo.frame_h):1.0;
    g_last_grid_w.store(active_w,std::memory_order_relaxed);g_last_grid_h.store(active_h,std::memory_order_relaxed);
    if(!skip_network){
    if(b->geo.scale!=g_push_log_scale){g_push_log_scale=b->geo.scale;g_push_log_budget=int(b->plan.passes.size());}
    uint32_t gather_push[]={p.color_base_x,p.color_base_y,p.motion_base_x,p.motion_base_y,p.depth_base_x,p.depth_base_y,active_w,active_h,ioport_color_float_offset,ioport_motion_float_offset,ioport_depth_float_offset,have_depth?1u:0u,ioport_color_float_offset,ioport_motion_float_offset,ioport_depth_float_offset,hdr_word,p.render_width,p.render_height,io.images[1].extent.width,io.images[1].extent.height,active_w,active_h,0,0};
    static_assert(sizeof(gather_push)==kGatherPushBytes,"gather push block must match its pipeline layout range");
    if(g_dump_push){unsigned long long h=1469598103934665603ull;for(auto v:gather_push){h^=v;h*=1099511628211ull;}
        std::printf("[push] gather n=%zu h=%016llx\n",sizeof(gather_push)/4,h);}
    // Diagnostic marker: the frame that is about to be snapshotted gets its output buffer filled
    // with 0.5 first. Whatever the dump shows as "not 0.5" was written by the pass list, and
    // whatever is still 0.5 was never written - which separates "the output pass under-covers" from
    // "the dump raced", the two explanations every screenshot so far is consistent with.
    if(b->dump_req>0&&!b->dump_pending&&b->dump_at>0&&b->ticks+1u>=b->dump_at){
        b->vkCmdFillBuffer(cb,b->plan.buffers[b->output].handle,0,VK_WHOLE_SIZE,0x3f000000u);
        barrier(b,cb,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT);
    }

    b->vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,b->gather.handle);b->vkCmdBindDescriptorSets(cb,VK_PIPELINE_BIND_POINT_COMPUTE,b->gather.layout,0,1,&b->gather_set,0,nullptr);
    b->vkCmdPushConstants(cb,b->gather.layout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(gather_push),gather_push);b->vkCmdDispatch(cb,(active_w*active_h+255)/256,1,1);
    {   // keep the gather push's hash for the one-shot audit line at the end of the frame
        unsigned long long x=1469598103934665603ull;
        for(size_t i=0;i<sizeof(gather_push)/sizeof(gather_push[0]);++i){x^=gather_push[i];x*=1099511628211ull;}
        g_audit_gather=x;
    }
    barrier(b,cb,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT);
    write_ts(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,3);
    for(size_t i=0;i<b->plan.passes.size();++i){auto& pass=b->plan.passes[i];bool clearing=false;
        for(auto& x:b->plan.buffers)if(!x.resident&&x.firstPass==i&&x.clearFirst){if(!clearing)barrier(b,cb,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_TRANSFER_WRITE_BIT);clearing=true;b->vkCmdFillBuffer(cb,x.handle,0,VK_WHOLE_SIZE,0);}
        if(clearing)barrier(b,cb,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT);
        auto& pipe=b->pipelines[pass.pipeline];b->vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,pipe.handle);b->vkCmdBindDescriptorSets(cb,VK_PIPELINE_BIND_POINT_COMPUTE,pipe.layout,0,1,&pass.set,0,nullptr);
        auto push=pass.push;if(i==0){push[22]=io.noise_seed;push[23]&=~4u;}
        if(!push.empty())b->vkCmdPushConstants(cb,pipe.layout,VK_SHADER_STAGE_COMPUTE_BIT,0,uint32_t(push.size()*4),push.data());
        // Dispatches scale with the active rect: the plan's grids describe the
        // full 1280x720 canvas, so the active region is the same fraction of
        // every pass grid. Round UP: an under-covered grid leaves a strip that
        // no pass ever writes, which shows as black/stale bands that flicker
        // with the history state. Over-coverage only touches canvas outside the
        // active rect, which scatter never samples. Never exceed the plan grid
        // and never drop a pass entirely.
        const uint32_t group_x=scaled_count(pass.gx,grid_sx,scale_x);
        const uint32_t group_y=scaled_count(pass.gy,grid_sy,scale_y);
        const uint32_t group_z=scale_z?scaled_count(pass.gz,grid_sy,true):pass.gz;
        if(g_push_log_budget>0){--g_push_log_budget;unsigned long long h=1469598103934665603ull;
            for(auto v:push){h^=v;h*=1099511628211ull;}
            nr::logf("[nr] push %zu %-34s %016llx g=%u,%u,%u",i,pass.name.c_str(),h,group_x,group_y,group_z);}
        if(g_dump_push){unsigned long long h=1469598103934665603ull;
            for(auto v:push){h^=v;h*=1099511628211ull;}
            std::printf("[push] %zu n=%zu h=%016llx g=%u,%u,%u\n",i,push.size(),h,group_x,group_y,group_z);}
        b->vkCmdDispatch(cb,group_x,group_y,group_z);
        barrier(b,cb,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT);
        if (g_snap_pass >= 0 && int(i) == g_snap_pass && !g_snap_buf.empty()) {
            VkBuffer src = VK_NULL_HANDLE, scratch = VK_NULL_HANDLE;
            VkDeviceSize n = 0;
            for (auto& x : b->plan.buffers) {
                if (x.name == g_snap_buf) { src = x.handle; n = x.bytes; }
                if (x.name == "ioport_0_ioport_images") scratch = x.handle;
            }
            if (src && scratch) {
                VkBufferCopy r{0, 0, std::min(n, b->plan.buffers[b->output].bytes)};
                barrier(b,cb,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_TRANSFER_READ_BIT);
                b->vkCmdCopyBuffer(cb, src, scratch, 1, &r);
                barrier(b,cb,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT);
                std::printf("[snap] pass %d <- %s (%llu bytes)\n", g_snap_pass, g_snap_buf.c_str(),
                            (unsigned long long)r.size);
            }
        }
    }
    }
    if (g_snap_pass >= 0 && !g_snap_buf.empty()) {
        VkBuffer scratch = VK_NULL_HANDLE;
        for (auto& x : b->plan.buffers) if (x.name == "ioport_0_ioport_images") scratch = x.handle;
        if (scratch) {
            VkBufferCopy r{0, 0, b->plan.buffers[b->output].bytes};
            barrier(b,cb,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_ACCESS_MEMORY_WRITE_BIT,VK_ACCESS_TRANSFER_READ_BIT);
            b->vkCmdCopyBuffer(cb, scratch, b->plan.buffers[b->output].handle, 1, &r);
            barrier(b,cb,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_MEMORY_READ_BIT);
        }
    }
    write_ts(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,4);
    const Pipeline& scatter=io.images[4].format==VK_FORMAT_R32G32B32A32_SFLOAT?b->scatter32:b->scatter;
    // base_size = output origin + extent. grid_size = (0, 0, grid_w, grid_h):
    // the shader uses grid_size.z as the row stride and grid_size.zw as the
    // canvas extent, so the grid dims belong in z/w, not x/y. Leaving z=0 made
    // every row read row 0 - a horizontally smeared output that reads as black
    // bands and flicker, with no API error anywhere.
    //
    // **The row stride is the plan's, not the ioport canvas.** `frame_out` is a
    // planar tensor whose plane is the plan's output extent (1280x720 at the
    // native size, 1024x576 at the adaptive one - see the output pass's push
    // and its plane offsets in port_network_runtime.txt), while kFrameWidth is
    // the `ioport` canvas the *input* side is frozen to. They coincide at the
    // native size and diverge at every other one, and reading a 1024-wide
    // plane at a 1280 stride shifts each row by 256 texels - four rows wrap a
    // whole row, which is exactly the four-row-period stripe pattern the
    // 1024x576 session showed. The plan is generated for the render size
    // (frame_supported() enforces it), so that size is the plane's width.
    const uint32_t plane_w=b->geo.frame_w;
    const uint32_t plane_h=b->geo.frame_h;
    uint32_t acut_bits=0;memcpy(&acut_bits,&b->acutance,4);
    uint32_t rcas_bits=0;memcpy(&rcas_bits,&b->rcas,4);
    uint32_t easu_bits=b->easu?1u:0u;
    uint32_t blend_t=0,blend_c=0,blend_g=0;memcpy(&blend_t,&b->detail,4);memcpy(&blend_c,&b->colour,4);memcpy(&blend_g,&b->guard,4);
    // Below 1x the answer was computed from a downsampled frame, so it carries no high frequency
    // of its own; publishing it whole is what erased fine texture. In that case the scatter keeps
    // the frame's own texel and adds only the answer's deviation (tune.y). delta: -1 auto,
    // 0 replace, 1 edit. At active==render the two are the same value, and replace is what the
    // 1x path has always done, so auto never changes it.
    const bool scaled=active_w<p.render_width||active_h<p.render_height;
    const bool delta_mode=b->delta<0?scaled:(b->delta!=0);
    // Publish the geometry once per change. Everything about the working scale is invisible from
    // outside: the game log reports the NR feature at the render size while the network actually
    // runs on a smaller canvas, and "is the edit path even engaged" cannot be answered from the
    // render size alone. One line per change, in the same file as the tuning readback.
    if(p.render_width!=b->geom_render_w||p.render_height!=b->geom_render_h||
       active_w!=b->geom_active_w||active_h!=b->geom_active_h||delta_mode!=b->geom_delta) {
        b->geom_render_w=p.render_width;b->geom_render_h=p.render_height;
        b->geom_active_w=active_w;b->geom_active_h=active_h;b->geom_delta=delta_mode;
        char line[160];
        std::snprintf(line,sizeof line,"geom render=%ux%u active=%ux%u canvas=%ux%u scale=%.2f delta=%d depth=%d\n",
            p.render_width,p.render_height,active_w,active_h,canvas_w,canvas_h,b->geo.scale,int(delta_mode),have_depth?1:0);
        std::printf("[%s",line);
        const auto dir=std::filesystem::path(b->manifest_dir).parent_path();
        if(FILE* lf=std::fopen((dir/"nr_tune_read.txt").string().c_str(),"a")){std::fputs(line,lf);std::fclose(lf);}
    }
    // grid_size.x carries the RCAS strength (see rcas_delta); it was an unused zero, so the block
    // size and every other offset stay as they are.
    uint32_t scatter_push[]={p.output_base_x,p.output_base_y,p.render_width,p.render_height,rcas_bits,0,plane_w,plane_h,active_w,active_h,0u,0u,hdr_word,blend_t,blend_c,blend_g,p.color_base_x,p.color_base_y,p.render_width,p.render_height,acut_bits,delta_mode?1u:0u,easu_bits,
        (uint32_t(std::lround(b->tone_w*255.0f))&0xFFFFu)|((uint32_t(std::lround(b->texture_w*255.0f))&0xFFFFu)<<16)};
    if(g_dump_push){unsigned long long h=1469598103934665603ull;for(auto v:scatter_push){h^=v;h*=1099511628211ull;}
        std::printf("[push] scatter n=%zu h=%016llx\n",sizeof(scatter_push)/4,h);}
    static_assert(sizeof(scatter_push)==kScatterPushBytes,"scatter push block must match its pipeline layout range");b->vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,scatter.handle);b->vkCmdBindDescriptorSets(cb,VK_PIPELINE_BIND_POINT_COMPUTE,scatter.layout,0,1,&b->scatter_set,0,nullptr);
    b->vkCmdPushConstants(cb,scatter.layout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(scatter_push),scatter_push);b->vkCmdDispatch(cb,(p.render_width*p.render_height+255)/256,1,1);
    barrier(b,cb,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_MEMORY_READ_BIT|VK_ACCESS_MEMORY_WRITE_BIT);
    if(b->dump_req>0&&!b->dump_pending){if(b->dump_at==0)b->dump_at=b->ticks+600u;if(b->ticks>=b->dump_at){b->dump_req=0;b->dump_pending=true;}}
    write_ts(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,5);
    audit_scale_once(b,active_w,active_h,scatter_push,sizeof(scatter_push)/sizeof(scatter_push[0]));
    b->record_pending=true;b->pending_frame=io.frame_id;b->pending_epoch=io.control_epoch;b->last_render_width=p.render_width;b->last_render_height=p.render_height;last_error.clear();return 0;
}
extern "C" int port_network_bridge_backend_submit_result(NRVK_Backend* b,VkFence fence,VkResult result) {
    if(!b||!b->record_pending||(result==VK_SUCCESS&&!fence)){last_error="submit_result requires recorded frame and actual completion fence";return -1;}
    b->record_pending=false;
    if(result==VK_SUCCESS){b->history_committed=true;b->committed_frame=b->pending_frame;b->committed_epoch=b->pending_epoch;b->last_network_fence=fence;}
    return 0;
}
extern "C" void nrvk_backend_release(NRVK_Backend* b) {
    if(!b)return;
    // Caller serializes queue access and resolves/discards unsubmitted CBs.
    // Full queue completion is explicit here; external device/queue stay alive.
    if(b->vkQueueWaitIdle && b->vkQueueWaitIdle(b->queue)!=VK_SUCCESS) {
        last_error="release queue completion failed; context retained, external device owner must resolve completion";
        return;
    }
    if(b->descriptors)b->vkDestroyDescriptorPool(b->device,b->descriptors,nullptr);
    auto free_pipe=[&](Pipeline& p){if(p.handle)b->vkDestroyPipeline(b->device,p.handle,nullptr);if(p.layout)b->vkDestroyPipelineLayout(b->device,p.layout,nullptr);if(p.descriptors)b->vkDestroyDescriptorSetLayout(b->device,p.descriptors,nullptr);};
    for(auto& p:b->pipelines)free_pipe(p);free_pipe(b->gather);free_pipe(b->scatter);free_pipe(b->scatter32);
    if(b->sampler)b->vkDestroySampler(b->device,b->sampler,nullptr);
    if(b->mapped)b->vkUnmapMemory(b->device,b->staging.memory);
    if(b->staging.handle)b->vkDestroyBuffer(b->device,b->staging.handle,nullptr);if(b->staging.memory)b->vkFreeMemory(b->device,b->staging.memory,nullptr);
    if(b->base_copy)b->vkDestroyBuffer(b->device,b->base_copy,nullptr);if(b->base_copy_mem)b->vkFreeMemory(b->device,b->base_copy_mem,nullptr);
    if(b->dump_map)b->vkUnmapMemory(b->device,b->dump_mem);if(b->dump_buf)b->vkDestroyBuffer(b->device,b->dump_buf,nullptr);if(b->dump_mem)b->vkFreeMemory(b->device,b->dump_mem,nullptr);if(b->dump_fence)b->vkDestroyFence(b->device,b->dump_fence,nullptr);if(b->dump_pool)b->vkDestroyCommandPool(b->device,b->dump_pool,nullptr);
    for(auto& x:b->plan.buffers){if(x.handle)b->vkDestroyBuffer(b->device,x.handle,nullptr);if(x.memory&&x.ownsMemory)b->vkFreeMemory(b->device,x.memory,nullptr);}
    for(auto mem:b->pool_memory)b->vkFreeMemory(b->device,mem,nullptr);
    if(b->startup_fence)b->vkDestroyFence(b->device,b->startup_fence,nullptr);
    if(b->startup_pool)b->vkDestroyCommandPool(b->device,b->startup_pool,nullptr);
    if(b->loader)FreeLibrary(b->loader);delete b;
}
