#pragma once

#include <d3d12.h>
#include <nvsdk_ngx.h>
#include "nr_sync_probe.h"
#include "nr_tail_probe.h"
#include "nr_bridge.h"
#include "periodic_flash_native_diag.h"

namespace nrb {
using NgxEvaluate = NVSDK_NGX_Result (__cdecl *)(ID3D12GraphicsCommandList*,
    const NVSDK_NGX_Handle*,NVSDK_NGX_Parameter*,PFN_NVSDK_NGX_ProgressCallback);

// Optional host log sink for first-frame color-format and record-reject
// diagnostics. Must be set once during InitializeASI; may be null.
using DeferredLog = void (*)(const char*);
void deferred_identity_set_log(DeferredLog log);

// Diagnostic only: record XeSS against a private command list. It will be
// submitted immediately after the game's list that contains this Evaluate.
bool record_deferred_identity(ID3D12GraphicsCommandList* game_list,
    const NVSDK_NGX_Handle* handle,NVSDK_NGX_Parameter* params,
    PFN_NVSDK_NGX_ProgressCallback callback,NgxEvaluate original,
    uint64_t game_list_generation,const NRB_Controls& frame_controls,
    NRB_Route route,NRB_RouteEvidence evidence,
    NVSDK_NGX_Result& result,flashdiag::Context* diagnostic=nullptr);

// Returns the number of private lists inserted into this queue batch.
unsigned submit_deferred_identity(ID3D12CommandQueue* queue,UINT count,
    ID3D12CommandList* const* lists,const SubmitObservation* observations,
    void (STDMETHODCALLTYPE *original_execute)(ID3D12CommandQueue*,UINT,
        ID3D12CommandList* const*),const NRB_Processor* processor,
    const NRB_Controls* controls,void (*log)(const char*));
uint64_t deferred_nr_processed();
double deferred_nr_last_ms();
double deferred_nr_average_ms();
bool deferred_nr_stage_times(NRB_StageTimes* times);
bool deferred_nr_record_times(NRB_RecordTimes* times);
bool deferred_hdr_cache_stats(NRB_HdrCacheStats* stats);
void deferred_hdr_set_cache_enabled(bool enabled);
bool deferred_hdr_warmup_shader();
bool deferred_nr_timing_enabled();
void deferred_nr_set_timing_enabled(bool enabled);
void deferred_nr_reset_metrics();
unsigned deferred_hdr_ready();
bool deferred_nr_failed();
unsigned deferred_identity_recorded();
unsigned deferred_identity_retired();
unsigned deferred_identity_signal_failures();
unsigned deferred_identity_generation_mismatches();
unsigned deferred_identity_invalidated();
void deferred_identity_list_reset(ID3D12GraphicsCommandList* list);
void observe_deferred_identity_tail(const TailSummary& sample);
void reset_deferred_identity_tail(NRB_Route route);
// Drops pending records whose owning component was torn down. Such a
// record can never retire; keeping it would fill the 8-slot window and
// silently stop every later frame after a backend switch.
void deferred_identity_abandon_all();
// Clears the latched NR failure. Using it is only valid when the whole
// component that owned the failure was replaced.
void deferred_identity_clear_failure();
// Pauses recording so the pipeline can drain before a component swap.
void deferred_identity_set_suspended(bool suspended);
unsigned deferred_identity_pending();
bool deferred_identity_tail_verified(NRB_Route route);
void deferred_diag_poll();
void deferred_diag_shutdown();
}
