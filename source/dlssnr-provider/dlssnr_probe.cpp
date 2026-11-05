// Observation-only provider for OptiScaler's DLSS-NR channel.
//
// Round 1 of M1: export the 25 dlssnr_* names, log every call with its raw
// arguments, dump whatever DLSSNR.* keys the parameter object exposes, and
// return values that let OptiScaler continue far enough to reach evaluate.
// No rendering, no backend, no image change.
//
// Build: see CMakeLists.txt. Deploy as nvngx.dll_dlssnr.dll next to OptiScaler.
// Log:   dlssnr_probe.log next to this module.

// WIN32_LEAN_AND_MEAN / NOMINMAX come from the target definitions.
#pragma warning(push)
#pragma warning(disable : 4996)

#include <windows.h>
#include <intrin.h>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <share.h>

#include <nvsdk_ngx_params.h>

#include "dlssnr_abi.h"

namespace {

FILE* g_log = nullptr;
wchar_t g_dir[MAX_PATH] = {0};
SRWLOCK g_lock = SRWLOCK_INIT;
LONG g_seq = 0;

// Values OptiScaler pushes through the float-slot channel (style, strengths...).
float g_slots[32] = {0};
float g_ratio = 1.0f;          // reported WorkingScale
void* g_handle = nullptr;      // fake feature handle we hand back

// ------------------------------------------------------------------ logging

void open_log() {
    if (g_log) return;

    HMODULE self = nullptr;
    GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                           GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                       reinterpret_cast<LPCWSTR>(&open_log), &self);
    wchar_t path[MAX_PATH] = {0};
    GetModuleFileNameW(self, path, MAX_PATH);
    wchar_t* slash = wcsrchr(path, L'\\');
    if (slash) *slash = 0;
    wcscpy_s(g_dir, path);

    wchar_t file[MAX_PATH] = {0};
    swprintf_s(file, L"%s\\dlssnr_probe.log", path);
    // _SH_DENYNO: the log has to be readable while the game is still running,
    // otherwise every observation costs a game restart.
    g_log = _wfsopen(file, L"a", _SH_DENYNO);
    if (g_log) {
        SYSTEMTIME st;
        GetLocalTime(&st);
        fprintf(g_log, "\n=== probe attached %04d-%02d-%02d %02d:%02d:%02d ===\n",
                st.wYear, st.wMonth, st.wDay, st.wHour, st.wMinute, st.wSecond);
        fflush(g_log);
    }
}

void log_raw(const char* text) {
    open_log();
    if (!g_log) return;
    AcquireSRWLockExclusive(&g_lock);
    fputs(text, g_log);
    fflush(g_log);
    ReleaseSRWLockExclusive(&g_lock);
}

// Name of the module that called us: proves who is driving. The address is
// captured in the entry point - taking it here would report our own module.
void caller_module(void* ret, char* out, size_t bytes) {
    out[0] = 0;
    HMODULE mod = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPCWSTR>(ret), &mod))
        return;
    wchar_t full[MAX_PATH] = {0};
    GetModuleFileNameW(mod, full, MAX_PATH);
    const wchar_t* base = wcsrchr(full, L'\\');
    base = base ? base + 1 : full;
    WideCharToMultiByte(CP_ACP, 0, base, -1, out, int(bytes), nullptr, nullptr);
}

// ------------------------------------------------------- parameter inspection

enum class VKind { NotParam, Param };

bool readable(const void* p, size_t bytes) {
    if (!p) return false;
    MEMORY_BASIC_INFORMATION mbi{};
    if (!VirtualQuery(p, &mbi, sizeof(mbi))) return false;
    if (mbi.State != MEM_COMMIT) return false;
    if (mbi.Protect & (PAGE_NOACCESS | PAGE_GUARD)) return false;
    return reinterpret_cast<uintptr_t>(p) + bytes <=
           reinterpret_cast<uintptr_t>(mbi.BaseAddress) + mbi.RegionSize;
}

bool in_image(const void* p) {
    if (!p) return false;
    MEMORY_BASIC_INFORMATION mbi{};
    if (!VirtualQuery(p, &mbi, sizeof(mbi))) return false;
    return mbi.Type == MEM_IMAGE;
}

// A parameter object is a C++ object owned by OptiScaler: its vtable pointer,
// and the code it points at, both live inside a mapped image. Requiring that
// keeps us from calling through a random 8-byte value.
bool looks_like_param(const void* p) {
    if (!p || (reinterpret_cast<uintptr_t>(p) & 7)) return false;
    if (!readable(p, sizeof(void*))) return false;
    const void* vtable = *reinterpret_cast<void* const*>(p);
    if (!in_image(vtable)) return false;
    if (!readable(vtable, sizeof(void*))) return false;
    const void* first = *reinterpret_cast<void* const*>(vtable);
    return in_image(first);
}

__declspec(noinline) bool try_get_uint(NVSDK_NGX_Parameter* p, const char* key,
                                       unsigned* out) {
    __try {
        return p->Get(key, out) == NVSDK_NGX_Result_Success;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return false;
    }
}

__declspec(noinline) bool try_get_int(NVSDK_NGX_Parameter* p, const char* key,
                                      int* out) {
    __try {
        return p->Get(key, out) == NVSDK_NGX_Result_Success;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return false;
    }
}

__declspec(noinline) bool try_get_float(NVSDK_NGX_Parameter* p, const char* key,
                                        float* out) {
    __try {
        return p->Get(key, out) == NVSDK_NGX_Result_Success;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return false;
    }
}

__declspec(noinline) bool try_get_ptr(NVSDK_NGX_Parameter* p, const char* key,
                                      void** out) {
    __try {
        return p->Get(key, out) == NVSDK_NGX_Result_Success;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return false;
    }
}

__declspec(noinline) bool try_write_ptr(void* slot, void* value) {
    __try {
        *reinterpret_cast<void**>(slot) = value;
        return true;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return false;
    }
}

__declspec(noinline) bool try_write_float(float* slot, float value) {
    __try {
        *slot = value;
        return true;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return false;
    }
}

// Writes a handle only where it is safe: the slot must be writable, must not be
// the parameter object itself, and must currently be null. Anything else means
// we guessed the argument wrong and corrupting it would take OptiScaler down.
bool try_write_handle_if_empty(void* slot, void* value) {
    if (!slot || looks_like_param(slot)) return false;
    if (!readable(slot, sizeof(void*))) return false;
    void* current = nullptr;
    __try {
        current = *reinterpret_cast<void**>(slot);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return false;
    }
    if (current) return false;
    return try_write_ptr(slot, value);
}

// Recognises the parameter object among the raw arguments and dumps what we can.
// Returns true when the object was identified.
bool dump_params(const char* who, uintptr_t a1, uintptr_t a2, uintptr_t a3,
                 uintptr_t a4) {
    const uintptr_t args[4] = {a1, a2, a3, a4};
    for (int i = 0; i < 4; ++i) {
        void* candidate = reinterpret_cast<void*>(args[i]);
        if (!looks_like_param(candidate)) continue;

        auto* p = static_cast<NVSDK_NGX_Parameter*>(candidate);
        unsigned width = 0, height = 0;
        if (!try_get_uint(p, "DLSSNR.Width", &width)) continue;
        if (width < 16 || width > 32768) continue;
        try_get_uint(p, "DLSSNR.Height", &height);

        int enabled = -1, reset = -1, preset = -1, inverted = -1, automask = -1,
            uicorr = -1;
        float intensity = 0.f, ratio = 0.f, local_s = 0.f, local_t = 0.f,
              skin = 0.f, probe = 0.f, mvx = 0.f, mvy = 0.f;
        try_get_int(p, "DLSSNR.Enabled", &enabled);
        try_get_int(p, "DLSSNR.Reset", &reset);
        try_get_int(p, "DLSSNR.Hint.Render.Preset", &preset);
        try_get_int(p, "DLSSNR.DepthInverted", &inverted);
        try_get_int(p, "DLSSNR.UseAutoMask", &automask);
        try_get_int(p, "DLSSNR.UICorrection", &uicorr);
        try_get_float(p, "DLSSNR.Intensity", &intensity);
        try_get_float(p, "DLSSNR.ScalingRatio", &ratio);
        try_get_float(p, "DLSSNR.LocalStructureStrength", &local_s);
        try_get_float(p, "DLSSNR.LocalToneStrength", &local_t);
        try_get_float(p, "DLSSNR.SkinStructureStrength", &skin);
        try_get_float(p, "DLSSNR.OptiScalerFloatProbe", &probe);
        try_get_float(p, "DLSSNR.MVecScaleX", &mvx);
        try_get_float(p, "DLSSNR.MVecScaleY", &mvy);

        void* color = nullptr;
        void* depth = nullptr;
        void* mvec = nullptr;
        void* output = nullptr;
        void* ui = nullptr;
        void* ui_alpha = nullptr;
        void* backbuffer = nullptr;
        try_get_ptr(p, "DLSSNR.Color", &color);
        try_get_ptr(p, "DLSSNR.Depth", &depth);
        try_get_ptr(p, "DLSSNR.MVec", &mvec);
        try_get_ptr(p, "DLSSNR.Output", &output);
        try_get_ptr(p, "DLSSNR.UI", &ui);
        try_get_ptr(p, "DLSSNR.UIAlpha", &ui_alpha);
        try_get_ptr(p, "DLSSNR.Backbuffer", &backbuffer);

        char buf[1024];
        _snprintf_s(buf, sizeof(buf), _TRUNCATE,
                    "       %s: param@arg%d w=%u h=%u en=%d rst=%d pre=%d inv=%d "
                    "mask=%d uic=%d int=%.3f ratio=%.4f ls=%.2f lt=%.2f sk=%.2f "
                    "probe=%.4f mv=%.4f/%.4f\n"
                    "       %s: res color=%p depth=%p mvec=%p out=%p ui=%p "
                    "alpha=%p bb=%p\n",
                    who, i + 1, width, height, enabled, reset, preset, inverted,
                    automask, uicorr, intensity, ratio, local_s, local_t, skin,
                    probe, mvx, mvy, who, color, depth, mvec, output, ui,
                    ui_alpha, backbuffer);
        log_raw(buf);
        return true;
    }
    return false;
}

// --------------------------------------------------------------- entry macro

void log_call(const char* name, void* ret_addr, uintptr_t a1, uintptr_t a2,
              uintptr_t a3, uintptr_t a4, uintptr_t a5, uintptr_t a6) {
    char caller[64];
    caller_module(ret_addr, caller, sizeof(caller));
    char buf[512];
    const LONG n = InterlockedIncrement(&g_seq);
    _snprintf_s(buf, sizeof(buf), _TRUNCATE,
                "[%04ld] %-28s caller=%-16s a1=%p a2=%p a3=%p a4=%p a5=%p a6=%p\n",
                n, name, caller, (void*)a1, (void*)a2, (void*)a3, (void*)a4,
                (void*)a5, (void*)a6);
    log_raw(buf);
    dump_params(name, a1, a2, a3, a4);
}

#define PROBE_INT(name)                                                     \
    DLSSNR_EXPORT int __cdecl name(uintptr_t a1, uintptr_t a2, uintptr_t a3, \
                                   uintptr_t a4, uintptr_t a5, uintptr_t a6) \
    {                                                                       \
        log_call(#name, _ReturnAddress(), a1, a2, a3, a4, a5, a6);                            \
        return 0;                                                           \
    }

#define PROBE_STUB(name) PROBE_INT(name)

}  // namespace

// ------------------------------------------------------------- D3D12 entries

// create: the out-handle is the most likely second argument. Handing back a
// non-null pointer lets OptiScaler proceed to evaluate; we never dereference it.
DLSSNR_EXPORT int __cdecl dlssnr_call_create(uintptr_t a1, uintptr_t a2,
                                            uintptr_t a3, uintptr_t a4) {
    log_call("dlssnr_call_create", _ReturnAddress(), a1, a2, a3, a4, 0, 0);
    if (!g_handle) g_handle = reinterpret_cast<void*>(&g_ratio);
    // Only the second argument is considered; write only into an empty slot that
    // is not the parameter object, so a wrong guess cannot corrupt anything.
    try_write_handle_if_empty(reinterpret_cast<void*>(a2), g_handle);
    try_write_handle_if_empty(reinterpret_cast<void*>(a3), g_handle);
    return 0;
}

DLSSNR_EXPORT int __cdecl dlssnr_call_evaluate(uintptr_t a1, uintptr_t a2,
                                              uintptr_t a3, uintptr_t a4) {
    log_call("dlssnr_call_evaluate", _ReturnAddress(), a1, a2, a3, a4, 0, 0);
    return 0;
}

DLSSNR_EXPORT int __cdecl dlssnr_call_release(uintptr_t a1, uintptr_t a2) {
    log_call("dlssnr_call_release", _ReturnAddress(), a1, a2, 0, 0, 0, 0);
    return 0;
}

DLSSNR_EXPORT int __cdecl dlssnr_call_set_extras(uintptr_t a1, uintptr_t a2,
                                                uintptr_t a3) {
    log_call("dlssnr_call_set_extras", _ReturnAddress(), a1, a2, a3, 0, 0, 0);
    return 0;
}

// Floats arrive in XMM registers, so this one cannot use the generic prototype.
DLSSNR_EXPORT int __cdecl dlssnr_call_set_float_slot(int slot, float value) {
    if (slot >= 0 && slot < 32) g_slots[slot] = value;
    char buf[192];
    char caller[64];
    caller_module(_ReturnAddress(), caller, sizeof(caller));
    const LONG n = InterlockedIncrement(&g_seq);
    _snprintf_s(buf, sizeof(buf), _TRUNCATE,
                "[%04ld] %-28s caller=%-16s slot=%d value=%.5f\n", n,
                "dlssnr_call_set_float_slot", caller, slot, value);
    log_raw(buf);
    return 0;
}

DLSSNR_EXPORT float __cdecl dlssnr_call_probe_float(uintptr_t a1, uintptr_t a2) {
    log_call("dlssnr_call_probe_float", _ReturnAddress(), a1, a2, 0, 0, 0, 0);
    float value = 0.f;
    void* candidate = reinterpret_cast<void*>(a1);
    if (looks_like_param(candidate))
        try_get_float(static_cast<NVSDK_NGX_Parameter*>(candidate),
                      "DLSSNR.OptiScalerFloatProbe", &value);
    char buf[160];
    _snprintf_s(buf, sizeof(buf), _TRUNCATE, "       probe -> returning %.6f\n",
                value);
    log_raw(buf);
    return value;
}

DLSSNR_EXPORT void* __cdecl dlssnr_call_last_create() {
    log_call("dlssnr_call_last_create", _ReturnAddress(), 0, 0, 0, 0, 0, 0);
    if (!g_handle) g_handle = reinterpret_cast<void*>(&g_ratio);
    return g_handle;
}

DLSSNR_EXPORT void* __cdecl dlssnr_call_last_init() {
    log_call("dlssnr_call_last_init", _ReturnAddress(), 0, 0, 0, 0, 0, 0);
    return g_handle;
}

// ------------------------------------------------------------ ratio querying

DLSSNR_EXPORT float __cdecl dlssnr_query_scaling_ratio(float* out_ratio) {
    log_call("dlssnr_query_scaling_ratio", _ReturnAddress(), (uintptr_t)out_ratio, 0, 0, 0, 0, 0);
    if (out_ratio && readable(out_ratio, sizeof(float))) try_write_float(out_ratio, g_ratio);
    return g_ratio;
}

DLSSNR_EXPORT int __cdecl dlssnr_last_ratio_stage() {
    log_call("dlssnr_last_ratio_stage", _ReturnAddress(), 0, 0, 0, 0, 0, 0);
    return 1;
}

DLSSNR_EXPORT float __cdecl dlssnr_last_ratio_result() {
    log_call("dlssnr_last_ratio_result", _ReturnAddress(), 0, 0, 0, 0, 0, 0);
    return g_ratio;
}

// --------------------------------------------------- D3D11 / Vulkan variants
// Present so OptiScaler finds every export it probes for; the current target is
// D3D12, so these only record that they were touched.

PROBE_STUB(dlssnr_d3d11_init)
PROBE_STUB(dlssnr_d3d11_create)
PROBE_STUB(dlssnr_d3d11_release)
PROBE_STUB(dlssnr_d3d11_requirements)
PROBE_STUB(dlssnr_d3d11_probe)
PROBE_STUB(dlssnr_d3d11_last_init)
PROBE_STUB(dlssnr_d3d11_last_create)

PROBE_STUB(dlssnr_vk_init)
PROBE_STUB(dlssnr_vk_create)
PROBE_STUB(dlssnr_vk_evaluate)
PROBE_STUB(dlssnr_vk_release)
PROBE_STUB(dlssnr_vk_probe)
PROBE_STUB(dlssnr_vk_last_init)
PROBE_STUB(dlssnr_vk_last_create)

// ------------------------------------------------------------------- startup

BOOL WINAPI DllMain(HINSTANCE, DWORD reason, LPVOID) {
    if (reason == DLL_PROCESS_ATTACH) {
        open_log();
        if (g_log) {
            char buf[512];
            _snprintf_s(buf, sizeof(buf), _TRUNCATE,
                        "=== dlssnr probe loaded, dir=%ls ===\n", g_dir);
            log_raw(buf);
        }
    }
    return TRUE;
}
