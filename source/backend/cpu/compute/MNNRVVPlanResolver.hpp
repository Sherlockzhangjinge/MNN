#ifndef MNN_RVV_PLAN_RESOLVER_HPP
#define MNN_RVV_PLAN_RESOLVER_HPP

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int MNNRVVPlanResolve(int wbits, int plane, int oc, int ic, int blockNum, int blockL, int contractOk, int accumAvail);
int MNNRVVPlanIsAccum(int planId);
int MNNRVVPlanIsStrided(int planId);
int MNNRVVPlanSlot(int planId);
const char* MNNRVVPlanName(int planId, char* buffer, size_t size);
void MNNRVVPlanInstalled(int planId, int installOk, const char* fallbackReason);

#ifdef __cplusplus
}
#endif

#endif
