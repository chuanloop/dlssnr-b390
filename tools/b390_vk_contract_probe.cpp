// First-run Vulkan contract probe for the B390 adaptation (no model needed).
// Verifies on real hardware exactly what the runtime enforces:
//   - nrb_vulkan_processor.cpp: Vulkan 1.3, the three device extensions and
//     the feature chain, LUID validity, COMPUTE queue family.
//   - port_network_bridge_backend.cpp device gate: Intel vendor, B580 (0xe20b)
//     or Arc B390 (PCI deviceID 0xb080; NRVK_DEVICE_ID override), SLM >= 48 KiB.
//   - port_network_bridge_backend_plan.inc: every pass requires subgroup 32
//     and shared <= 49152.
//   - The frozen shader recipe: 8x16x16 fp16 cooperative-matrix A/B shapes
//     with fp16 and fp32 accumulators at subgroup scope (this is what lets the
//     B580-proven weight layout load unchanged on Xe3).
// Build (x64 Native Tools prompt or after vcvarsall):
//   cl /nologo /utf-8 /O2 /std:c++17 /I "%VULKAN_SDK%\Include" ^
//      tools\b390_vk_contract_probe.cpp /Fe:build\b390_vk_contract_probe.exe ^
//      /link /LIBPATH:"%VULKAN_SDK%\Lib" vulkan-1.lib
#define VK_NO_PROTOTYPES
#define NOMINMAX
#include <vulkan/vulkan.h>
#include <windows.h>
#include <cstdio>
#include <cstring>
#include <vector>

static int failures = 0;
static void check(bool ok, const char* what) {
    if (!*what) return;
    std::printf("[%s] %s\n", ok ? "OK  " : "FAIL", what);
    if (!ok) ++failures;
}

int main() {
    HMODULE dll = LoadLibraryA("vulkan-1.dll");
    if (!dll) { std::printf("FAIL vulkan-1.dll missing (Intel driver runtime)\n"); return 1; }
    auto gipa = (PFN_vkGetInstanceProcAddr)GetProcAddress(dll, "vkGetInstanceProcAddr");
    auto create = (PFN_vkCreateInstance)gipa(nullptr, "vkCreateInstance");
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VkInstance instance;
    if (create(&ici, nullptr, &instance) != VK_SUCCESS) { std::printf("FAIL vkCreateInstance\n"); return 1; }
    auto enumPD = (PFN_vkEnumeratePhysicalDevices)gipa(instance, "vkEnumeratePhysicalDevices");
    auto props2 = (PFN_vkGetPhysicalDeviceProperties2)gipa(instance, "vkGetPhysicalDeviceProperties2");
    auto feats2 = (PFN_vkGetPhysicalDeviceFeatures2)gipa(instance, "vkGetPhysicalDeviceFeatures2");
    auto enumExt = (PFN_vkEnumerateDeviceExtensionProperties)gipa(instance, "vkEnumerateDeviceExtensionProperties");
    auto getQF = (PFN_vkGetPhysicalDeviceQueueFamilyProperties)gipa(instance, "vkGetPhysicalDeviceQueueFamilyProperties");
    auto getCMP = (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)
        gipa(instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");
    uint32_t n = 0;
    if (enumPD(instance, &n, nullptr) != VK_SUCCESS || !n) { std::printf("FAIL no physical devices\n"); return 1; }
    std::vector<VkPhysicalDevice> devs(n);
    enumPD(instance, &n, devs.data());
    std::printf("physical devices: %u\n", n);
    bool intel_found = false;
    for (auto dev : devs) {
        VkPhysicalDeviceIDProperties idp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
        VkPhysicalDeviceSubgroupSizeControlProperties sscp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES};
        VkPhysicalDeviceProperties2 pp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &sscp};
        idp.pNext = pp.pNext; pp.pNext = &idp;
        props2(dev, &pp);
        const auto& p = pp.properties;
        std::printf("- %s | vendor=0x%04x device=0x%04x driver=%u.%u.%u api=%u.%u.%u luid_valid=%d\n",
            p.deviceName, p.vendorID, p.deviceID,
            VK_API_VERSION_MAJOR(p.driverVersion), VK_API_VERSION_MINOR(p.driverVersion), VK_API_VERSION_PATCH(p.driverVersion),
            VK_API_VERSION_MAJOR(p.apiVersion), VK_API_VERSION_MINOR(p.apiVersion), VK_API_VERSION_PATCH(p.apiVersion),
            (int)idp.deviceLUIDValid);
        if (p.vendorID != 0x8086) continue;
        intel_found = true;
        char label[128];
        std::snprintf(label, sizeof label, "deviceID 0x%04x is B390 (0xb080) / B580 (0xe20b), or set NRVK_DEVICE_ID", p.deviceID);
        check(p.deviceID == 0xb080 || p.deviceID == 0xe20b, label);
        check(p.apiVersion >= VK_API_VERSION_1_3, "Vulkan 1.3");
        std::snprintf(label, sizeof label, "SLM >= 48 KiB (have %u B)", p.limits.maxComputeSharedMemorySize);
        check(p.limits.maxComputeSharedMemorySize >= 49152, label);
        std::snprintf(label, sizeof label, "subgroup range [%u, %u] covers required 32",
                      sscp.minSubgroupSize, sscp.maxSubgroupSize);
        check(sscp.minSubgroupSize <= 32 && sscp.maxSubgroupSize >= 32, label);
        check(idp.deviceLUIDValid, "deviceLUID valid (D3D12 handoff match)");

        uint32_t ne = 0; enumExt(dev, nullptr, &ne, nullptr);
        std::vector<VkExtensionProperties> ext(ne); enumExt(dev, nullptr, &ne, ext.data());
        bool cm = false, em = false, es = false;
        for (auto& e : ext) {
            if (!std::strcmp(e.extensionName, "VK_KHR_cooperative_matrix")) cm = true;
            if (!std::strcmp(e.extensionName, "VK_KHR_external_memory_win32")) em = true;
            if (!std::strcmp(e.extensionName, "VK_KHR_external_semaphore_win32")) es = true;
        }
        check(cm, "extension VK_KHR_cooperative_matrix");
        check(em, "extension VK_KHR_external_memory_win32");
        check(es, "extension VK_KHR_external_semaphore_win32");

        VkPhysicalDeviceCooperativeMatrixFeaturesKHR cmf{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
        VkPhysicalDeviceShaderFloat16Int8Features f16{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES};
        VkPhysicalDevice16BitStorageFeatures st16{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES};
        VkPhysicalDeviceSubgroupSizeControlFeatures ssc{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES};
        VkPhysicalDeviceSynchronization2Features s2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES};
        VkPhysicalDeviceVulkanMemoryModelFeatures mm{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES};
        VkPhysicalDeviceTimelineSemaphoreFeatures tl{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES};
        VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        f2.pNext = &cmf; cmf.pNext = &f16; f16.pNext = &st16; st16.pNext = &ssc;
        ssc.pNext = &s2; s2.pNext = &mm; mm.pNext = &tl;
        feats2(dev, &f2);
        check(cmf.cooperativeMatrix, "feature cooperativeMatrix");
        check(f16.shaderFloat16, "feature shaderFloat16");
        check(st16.storageBuffer16BitAccess, "feature storageBuffer16BitAccess");
        check(ssc.subgroupSizeControl, "feature subgroupSizeControl");
        check(ssc.computeFullSubgroups, "feature computeFullSubgroups");
        check(s2.synchronization2, "feature synchronization2");
        check(mm.vulkanMemoryModel, "feature vulkanMemoryModel");
        check(tl.timelineSemaphore, "feature timelineSemaphore");

        uint32_t nqf = 0; getQF(dev, &nqf, nullptr);
        std::vector<VkQueueFamilyProperties> qf(nqf); getQF(dev, &nqf, qf.data());
        bool compute_ts = false;
        for (auto& q : qf) if ((q.queueFlags & VK_QUEUE_COMPUTE_BIT) && q.timestampValidBits) compute_ts = true;
        check(compute_ts, "COMPUTE queue family with timestamps");

        if (getCMP) {
            uint32_t nc = 0;
            if (getCMP(dev, &nc, nullptr) == VK_SUCCESS && nc) {
                std::vector<VkCooperativeMatrixPropertiesKHR> cs(nc, VkCooperativeMatrixPropertiesKHR{VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR});
                getCMP(dev, &nc, cs.data());
                std::printf("  cooperative matrix configs: %u\n", nc);
                bool mma_h = false, mma_f = false;
                for (auto& c : cs) {
                    std::printf("  M=%u N=%u K=%u A=%d B=%d C=%d R=%d scope=%d\n",
                        c.MSize, c.NSize, c.KSize, (int)c.AType, (int)c.BType, (int)c.CType, (int)c.ResultType, (int)c.scope);
                    if (c.MSize == 8 && c.NSize == 16 && c.KSize == 16 && c.scope == VK_SCOPE_SUBGROUP_KHR &&
                        c.AType == VK_COMPONENT_TYPE_FLOAT16_KHR && c.BType == VK_COMPONENT_TYPE_FLOAT16_KHR) {
                        if (c.CType == VK_COMPONENT_TYPE_FLOAT16_KHR && c.ResultType == VK_COMPONENT_TYPE_FLOAT16_KHR) mma_h = true;
                        if (c.CType == VK_COMPONENT_TYPE_FLOAT32_KHR && c.ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR) mma_f = true;
                    }
                }
                check(mma_h, "recipe fp16 accumulator 8x16x16 subgroup config");
                check(mma_f, "recipe fp32 accumulator 8x16x16 subgroup config");
            } else {
                check(false, "cooperative matrix properties queryable");
            }
        } else {
            check(false, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR available");
        }
    }
    check(intel_found, "Intel GPU present");
    std::printf("\n%s (%d failure(s))\n", failures ? "CONTRACT UNMET" : "Vulkan contract satisfied for the B390 adaptation", failures);
    return failures ? 1 : 0;
}
