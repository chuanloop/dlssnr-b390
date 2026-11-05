// Private ABI between OptiScaler's DLSS-NR feature and the "nvngx.dll_dlssnr.dll"
// provider module.
//
// Reconstructed from the shipped binaries, not from any documentation:
//
//   OptiScaler.dll (26.8 MB)          calls 25 exports named dlssnr_*
//   nvngx.dll_dlssnr.dll (121 KB)     the reference provider; exports exactly those 25
//   nvngx_dlssnr.dll (165 MB)         the real model container; exports 55 standard
//                                     NVSDK_NGX_* entry points
//
// The reference provider is a thin adapter: it reads the DLSSNR.* keys OptiScaler
// writes into an NVSDK_NGX_Parameter object and forwards to the standard NGX entry
// points of the model container. A provider that renders those same 25 exports
// cannot be distinguished from it, which is why this ABI - not NGX - is what an
// Intel implementation has to satisfy.
//
// Signatures below marked "probed" are guesses confirmed at runtime by
// dlssnr_probe.cpp; anything still marked "unknown" is logged and returns a
// benign default. Only the D3D12 path matters for the current target.
#pragma once

#include <cstdint>

#define DLSSNR_EXPORT extern "C" __declspec(dllexport)

// D3D12 path (the ones that matter now).
typedef int  (__cdecl *PFN_dlssnr_call_create)(void* params, void* out_handle);   // probed
typedef int  (__cdecl *PFN_dlssnr_call_evaluate)(void* params, void* handle);     // probed
typedef int  (__cdecl *PFN_dlssnr_call_release)(void* handle);                   // unknown
typedef int  (__cdecl *PFN_dlssnr_call_set_extras)(void* params);                // unknown
typedef int  (__cdecl *PFN_dlssnr_call_set_float_slot)(int slot, float value);   // probed
typedef float(__cdecl *PFN_dlssnr_call_probe_float)(void* params);               // unknown
typedef float(__cdecl *PFN_dlssnr_query_scaling_ratio)(float* out_ratio);        // unknown
typedef int  (__cdecl *PFN_dlssnr_last_ratio_stage)(void);                       // unknown
typedef float(__cdecl *PFN_dlssnr_last_ratio_result)(void);                      // unknown
typedef void* (__cdecl *PFN_dlssnr_call_last_create)(void);                       // unknown
typedef void* (__cdecl *PFN_dlssnr_call_last_init)(void);                         // unknown
