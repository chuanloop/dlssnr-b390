// Minimal stand-in for the NGX core (nvngx.dll).
//
// Why this exists
// ---------------
// OptiScaler's DLSS-NR path does not call the provider directly. It first brings
// up an NGX core:
//
//     EnsureForwarder  -> loads nvngx.dll_dlssnr.dll (our provider)
//     NVNGXProxy::InitNVNGX
//         _nvngx.dll   -> not found
//         nvngx.dll    -> not found
//         registry     -> not found
//     [E] DLSS-NR unavailable: NGX core initialization failed
//
// On an Intel machine nothing provides that core, so the channel dies before the
// provider is ever called. This module fills the hole: it answers Init with
// success and hands out feature handles, which is all the NR path needs before it
// starts talking to the provider.
//
// It deliberately does NOT implement DLSS: it is deployed as `_nvngx.dll`, the
// name OptiScaler tries first and games do not load, so the game's own upscaler
// path is untouched.
//
// Everything is a stub. Nothing is rendered, no image changes.

#include <windows.h>

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cwchar>
#include <share.h>

#include <nvsdk_ngx_params.h>

namespace {

FILE* g_log = nullptr;
SRWLOCK g_lock = SRWLOCK_INIT;

void log_line(const char* fmt, ...) {
    if (!g_log) return;
    char buf[512];
    va_list args;
    va_start(args, fmt);
    _vsnprintf_s(buf, sizeof(buf), _TRUNCATE, fmt, args);
    va_end(args);
    AcquireSRWLockExclusive(&g_lock);
    fputs(buf, g_log);
    fflush(g_log);
    ReleaseSRWLockExclusive(&g_lock);
}

void open_log() {
    if (g_log) return;
    wchar_t path[MAX_PATH] = {0};
    HMODULE self = nullptr;
    GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                           GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                       reinterpret_cast<LPCWSTR>(&open_log), &self);
    GetModuleFileNameW(self, path, MAX_PATH);
    wchar_t* slash = wcsrchr(path, L'\\');
    if (slash) *slash = 0;
    wchar_t file[MAX_PATH] = {0};
    swprintf_s(file, L"%s\\dlssnr_probe.log", path);
    g_log = _wfsopen(file, L"a", _SH_DENYNO);
}

NVSDK_NGX_Handle g_handle{1};
unsigned g_init_count = 0;

}  // namespace

// The NGX surface, declared loosely on purpose: every argument is an opaque
// pointer or integer, nothing is dereferenced except the out-handle of
// CreateFeature. Signatures matter only for register placement, and on x64 the
// caller's prototype decides that, not ours.

#define NGX_EXPORT extern "C" __declspec(dllexport)

// ------------------------------------------------------------------ init/shutdown

NGX_EXPORT NVSDK_NGX_Result __cdecl NVSDK_NGX_D3D12_Init(
    unsigned long long, const wchar_t*, void*, const void*, int) {
    ++g_init_count;
    log_line("[ngx_stub] D3D12_Init\n");
    return NVSDK_NGX_Result_Success;
}

NGX_EXPORT NVSDK_NGX_Result __cdecl NVSDK_NGX_D3D12_Init_Ext(
    unsigned long long, const wchar_t*, void*, int, const void*) {
    ++g_init_count;
    log_line("[ngx_stub] D3D12_Init_Ext\n");
    return NVSDK_NGX_Result_Success;
}

NGX_EXPORT NVSDK_NGX_Result __cdecl NVSDK_NGX_D3D12_Shutdown() {
    log_line("[ngx_stub] D3D12_Shutdown\n");
    return NVSDK_NGX_Result_Success;
}

NGX_EXPORT NVSDK_NGX_Result __cdecl NVSDK_NGX_D3D12_Shutdown1(void*) {
    log_line("[ngx_stub] D3D12_Shutdown1\n");
    return NVSDK_NGX_Result_Success;
}

// ------------------------------------------------------------------- feature calls

NGX_EXPORT NVSDK_NGX_Result __cdecl NVSDK_NGX_D3D12_CreateFeature(
    void*, int feature_id, void*, void** out_handle) {
    log_line("[ngx_stub] D3D12_CreateFeature feature=%d out=%p\n", feature_id,
             out_handle);
    if (out_handle) {
        __try {
            *out_handle = &g_handle;
        } __except (EXCEPTION_EXECUTE_HANDLER) {
            log_line("[ngx_stub] CreateFeature: out pointer not writable\n");
            return NVSDK_NGX_Result_FAIL_InvalidParameter;
        }
    }
    return NVSDK_NGX_Result_Success;
}

NGX_EXPORT NVSDK_NGX_Result __cdecl NVSDK_NGX_D3D12_EvaluateFeature(
    void*, const void*, void*, void*) {
    log_line("[ngx_stub] D3D12_EvaluateFeature\n");
    return NVSDK_NGX_Result_Success;
}

NGX_EXPORT NVSDK_NGX_Result __cdecl NVSDK_NGX_D3D12_ReleaseFeature(void*) {
    log_line("[ngx_stub] D3D12_ReleaseFeature\n");
    return NVSDK_NGX_Result_Success;
}

NGX_EXPORT NVSDK_NGX_Result __cdecl NVSDK_NGX_D3D12_GetScratchBufferSize(
    int, const void*, size_t* out_size) {
    log_line("[ngx_stub] D3D12_GetScratchBufferSize\n");
    if (out_size) {
        __try {
            *out_size = 0;
        } __except (EXCEPTION_EXECUTE_HANDLER) {
            return NVSDK_NGX_Result_FAIL_InvalidParameter;
        }
    }
    return NVSDK_NGX_Result_Success;
}

NGX_EXPORT NVSDK_NGX_Result __cdecl NVSDK_NGX_D3D12_GetFeatureRequirements(
    void*, const void*, void*) {
    log_line("[ngx_stub] D3D12_GetFeatureRequirements\n");
    return NVSDK_NGX_Result_Success;
}

NGX_EXPORT NVSDK_NGX_Result __cdecl NVSDK_NGX_D3D12_PopulateParameters_Impl(
    void*) {
    return NVSDK_NGX_Result_Success;
}

// ------------------------------------------------------- D3D11 / Vulkan / CUDA
// Exported so a method-address sweep finds a complete surface. Never called in a
// D3D12 title; they only record that they were reached.

#define NGX_STUB_CALL(name)                                       \
    NGX_EXPORT NVSDK_NGX_Result __cdecl name(                     \
        void*, void*, void*, void*, void*) {                      \
        log_line("[ngx_stub] " #name "\n");                       \
        return NVSDK_NGX_Result_Success;                          \
    }

NGX_STUB_CALL(NVSDK_NGX_D3D11_Init)
NGX_STUB_CALL(NVSDK_NGX_D3D11_Init_Ext)
NGX_STUB_CALL(NVSDK_NGX_D3D11_CreateFeature)
NGX_STUB_CALL(NVSDK_NGX_D3D11_EvaluateFeature)
NGX_STUB_CALL(NVSDK_NGX_D3D11_ReleaseFeature)
NGX_STUB_CALL(NVSDK_NGX_D3D11_GetScratchBufferSize)
NGX_STUB_CALL(NVSDK_NGX_D3D11_GetFeatureRequirements)
NGX_STUB_CALL(NVSDK_NGX_D3D11_Shutdown)
NGX_STUB_CALL(NVSDK_NGX_D3D11_Shutdown1)
NGX_STUB_CALL(NVSDK_NGX_D3D11_PopulateParameters_Impl)

NGX_STUB_CALL(NVSDK_NGX_VULKAN_Init)
NGX_STUB_CALL(NVSDK_NGX_VULKAN_Init_Ext)
NGX_STUB_CALL(NVSDK_NGX_VULKAN_Init_Ext2)
NGX_STUB_CALL(NVSDK_NGX_VULKAN_CreateFeature)
NGX_STUB_CALL(NVSDK_NGX_VULKAN_CreateFeature1)
NGX_STUB_CALL(NVSDK_NGX_VULKAN_EvaluateFeature)
NGX_STUB_CALL(NVSDK_NGX_VULKAN_ReleaseFeature)
NGX_STUB_CALL(NVSDK_NGX_VULKAN_GetScratchBufferSize)
NGX_STUB_CALL(NVSDK_NGX_VULKAN_GetFeatureRequirements)
NGX_STUB_CALL(NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements)
NGX_STUB_CALL(NVSDK_NGX_VULKAN_GetFeatureInstanceExtensionRequirements)
NGX_STUB_CALL(NVSDK_NGX_VULKAN_Shutdown)
NGX_STUB_CALL(NVSDK_NGX_VULKAN_Shutdown1)
NGX_STUB_CALL(NVSDK_NGX_VULKAN_PopulateParameters_Impl)

NGX_STUB_CALL(NVSDK_NGX_CUDA_Init)
NGX_STUB_CALL(NVSDK_NGX_CUDA_Init1)
NGX_STUB_CALL(NVSDK_NGX_CUDA_Init_Ext)
NGX_STUB_CALL(NVSDK_NGX_CUDA_Init_Ext1)
NGX_STUB_CALL(NVSDK_NGX_CUDA_CreateFeature)
NGX_STUB_CALL(NVSDK_NGX_CUDA_CreateFeature1)
NGX_STUB_CALL(NVSDK_NGX_CUDA_EvaluateFeature)
NGX_STUB_CALL(NVSDK_NGX_CUDA_ReleaseFeature)
NGX_STUB_CALL(NVSDK_NGX_CUDA_GetScratchBufferSize)
NGX_STUB_CALL(NVSDK_NGX_CUDA_GetFeatureRequirements)
NGX_STUB_CALL(NVSDK_NGX_CUDA_Shutdown)
NGX_STUB_CALL(NVSDK_NGX_CUDA_Shutdown1)
NGX_STUB_CALL(NVSDK_NGX_CUDA_PopulateParameters_Impl)

// ------------------------------------------------------------------ misc surface

NGX_EXPORT NVSDK_NGX_Result __cdecl NVSDK_NGX_GetAPIVersion(int* out_version) {
    if (out_version) {
        __try {
            *out_version = 0x0000000A;
        } __except (EXCEPTION_EXECUTE_HANDLER) {
            return NVSDK_NGX_Result_FAIL_InvalidParameter;
        }
    }
    return NVSDK_NGX_Result_Success;
}

NGX_EXPORT NVSDK_NGX_Result __cdecl NVSDK_NGX_GetApplicationId(
    unsigned long long* out_id) {
    if (out_id) {
        __try {
            *out_id = 0;
        } __except (EXCEPTION_EXECUTE_HANDLER) {
            return NVSDK_NGX_Result_FAIL_InvalidParameter;
        }
    }
    return NVSDK_NGX_Result_Success;
}

NGX_EXPORT NVSDK_NGX_Result __cdecl NVSDK_NGX_GetSnippetVersion(void*) {
    return NVSDK_NGX_Result_Success;
}
NGX_EXPORT NVSDK_NGX_Result __cdecl NVSDK_NGX_GetGPUArchitecture(void*) {
    return NVSDK_NGX_Result_Success;
}
NGX_EXPORT NVSDK_NGX_Result __cdecl NVSDK_NGX_GetDriverVersionEx(void*) {
    return NVSDK_NGX_Result_Success;
}
NGX_EXPORT NVSDK_NGX_Result __cdecl NVSDK_NGX_SetOverrideStatusCallback(void*) {
    return NVSDK_NGX_Result_Success;
}
NGX_EXPORT NVSDK_NGX_Result __cdecl NVSDK_NGX_SetRuntimeParamsCallback(void*) {
    return NVSDK_NGX_Result_Success;
}
NGX_EXPORT NVSDK_NGX_Result __cdecl NVSDK_NGX_SetTelemetryEvaluateCallback(
    void*) {
    return NVSDK_NGX_Result_Success;
}

BOOL WINAPI DllMain(HINSTANCE, DWORD reason, LPVOID) {
    if (reason == DLL_PROCESS_ATTACH) {
        open_log();
        log_line("=== ngx_stub loaded ===\n");
    }
    return TRUE;
}
