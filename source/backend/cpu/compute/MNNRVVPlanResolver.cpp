#include "MNNRVVPlanResolver.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mutex>

#if defined(__riscv) && defined(MNN_USE_RVV)
#include <riscv_vector.h>
extern "C" bool MNNRVVUseAccumOverH();
extern "C" bool MNNRVVUseAccumOverH_W4();
extern "C" bool MNNRVVAccumImplStrided();
#endif

namespace {

constexpr int kMaxPlans = 256;
constexpr int kMaxCache = 64;
constexpr int kMaxInstallRecords = 32;

struct PlanRecord {
    int64_t vendor;
    int64_t march;
    int64_t vlen;
    int wbits;
    int regime;
    int oc;
    int ic;
    int blockNum;
    int blockL;
    int planId;
};

struct PlanCacheEntry {
    uint64_t key;
    int planId;
    int wbits;
    int regime;
    int oc;
    int ic;
    unsigned hits;
};

struct InstallRecord {
    int planId;
    int installOk;
    char reason[96];
    unsigned count;
};

struct ResolverState {
    bool loaded;
    bool trace;
    int planCount;
    PlanRecord plans[kMaxPlans];
    int cacheCount;
    PlanCacheEntry cache[kMaxCache];
    int installCount;
    InstallRecord installs[kMaxInstallRecords];
};

ResolverState& state() {
    static ResolverState value = {};
    return value;
}

std::mutex& stateMutex() {
    static std::mutex value;
    return value;
}

uint64_t readCpuId(const char* name) {
#if defined(__riscv)
    FILE* file = fopen("/proc/cpuinfo", "r");
    if (file == nullptr) {
        return 0;
    }
    char line[512];
    uint64_t value = 0;
    while (fgets(line, sizeof(line), file) != nullptr) {
        if (strstr(line, name) != nullptr) {
            char* colon = strchr(line, ':');
            if (colon != nullptr) {
                value = strtoull(colon + 1, nullptr, 0);
            }
            break;
        }
    }
    fclose(file);
    return value;
#else
    (void)name;
    return 0;
#endif
}

int currentVlen() {
#if defined(__riscv) && defined(MNN_USE_RVV)
    return static_cast<int>(__riscv_vsetvlmax_e8m1()) * 8;
#else
    return 0;
#endif
}

int currentRegime(int plane) {
    return plane > 1 ? 1 : 0;
}

bool parseInt64(const char* text, int64_t* value) {
    if (strcmp(text, "*") == 0) {
        *value = -1;
        return true;
    }
    char* end = nullptr;
    *value = static_cast<int64_t>(strtoll(text, &end, 0));
    return end != text && *end == '\0';
}

bool parseInt(const char* text, int* value) {
    if (strcmp(text, "*") == 0) {
        *value = -1;
        return true;
    }
    char* end = nullptr;
    *value = static_cast<int>(strtol(text, &end, 0));
    return end != text && *end == '\0';
}

bool parseRegime(const char* text, int* value) {
    if (strcmp(text, "*") == 0) {
        *value = -1;
        return true;
    }
    if (strcmp(text, "prefill") == 0) {
        *value = 1;
        return true;
    }
    if (strcmp(text, "decode") == 0) {
        *value = 0;
        return true;
    }
    return parseInt(text, value);
}

void loadPlanFile() {
    ResolverState& s = state();
    if (s.loaded) {
        return;
    }
    s.loaded = true;
    const char* trace = getenv("MNN_RVV_PLAN_TRACE");
    s.trace = trace != nullptr && trace[0] == '1';
    const char* path = getenv("MNN_RVV_PLAN_FILE");
    if (path == nullptr || path[0] == '\0') {
        return;
    }
    FILE* file = fopen(path, "r");
    if (file == nullptr) {
        if (s.trace) {
            fprintf(stderr, "[MNN][RVV-PLAN] table=open-fail path=%s\n", path);
        }
        return;
    }
    char line[512];
    while (s.planCount < kMaxPlans && fgets(line, sizeof(line), file) != nullptr) {
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') {
            continue;
        }
        char vendor[32], march[32], vlen[32], wbits[32], regime[32];
        char oc[32], ic[32], blockNum[32], blockL[32], planId[32];
        const int count = sscanf(line, "%31s %31s %31s %31s %31s %31s %31s %31s %31s %31s", vendor, march,
                                 vlen, wbits, regime, oc, ic, blockNum, blockL, planId);
        if (count != 10) {
            continue;
        }
        PlanRecord record = {};
        if (!parseInt64(vendor, &record.vendor) || !parseInt64(march, &record.march) ||
            !parseInt64(vlen, &record.vlen) || !parseInt(wbits, &record.wbits) ||
            !parseRegime(regime, &record.regime) || !parseInt(oc, &record.oc) || !parseInt(ic, &record.ic) ||
            !parseInt(blockNum, &record.blockNum) || !parseInt(blockL, &record.blockL) ||
            !parseInt(planId, &record.planId) || record.planId < 0) {
            continue;
        }
        s.plans[s.planCount++] = record;
    }
    fclose(file);
    if (s.trace) {
        fprintf(stderr, "[MNN][RVV-PLAN] table=loaded path=%s entries=%d\n", path, s.planCount);
    }
}

bool matches(int64_t expected, int64_t actual) {
    return expected < 0 || expected == actual;
}

bool matches(int expected, int actual) {
    return expected < 0 || expected == actual;
}

int specificity(const PlanRecord& record) {
    int score = 0;
    score += record.vendor >= 0;
    score += record.march >= 0;
    score += record.vlen >= 0;
    score += record.wbits >= 0;
    score += record.regime >= 0;
    score += record.oc >= 0;
    score += record.ic >= 0;
    score += record.blockNum >= 0;
    score += record.blockL >= 0;
    return score;
}

uint64_t planKey(int wbits, int regime, int oc, int ic, int blockNum, int blockL) {
    const uint64_t fields[] = {readCpuId("mvendorid"), readCpuId("marchid"), static_cast<uint64_t>(currentVlen()),
                               static_cast<uint64_t>(wbits), static_cast<uint64_t>(regime), static_cast<uint64_t>(oc),
                               static_cast<uint64_t>(ic), static_cast<uint64_t>(blockNum), static_cast<uint64_t>(blockL)};
    uint64_t hash = 1469598103934665603ull;
    for (size_t i = 0; i < sizeof(fields) / sizeof(fields[0]); ++i) {
        hash ^= fields[i];
        hash *= 1099511628211ull;
    }
    return hash;
}

int fallbackPlan(int wbits) {
#if defined(__riscv) && defined(MNN_USE_RVV)
    if (getenv("MNN_RVV_FORCE_REDUCTION") != nullptr && getenv("MNN_RVV_FORCE_REDUCTION")[0] == '1') {
        return 0;
    }
    const bool useAccum = wbits == 4 ? MNNRVVUseAccumOverH_W4() : MNNRVVUseAccumOverH();
    if (!useAccum) {
        return 0;
    }
    const char* grain = getenv("MNN_RVV_ACCUM_GRAIN");
    const uint64_t vendor = readCpuId("mvendorid");
    const uint64_t march = readCpuId("marchid");
    const int defaultLmul = vendor == 0x710 && (march & 0xffu) == 0x01u ? 4 : 8;
    const int lmul = grain != nullptr && grain[0] == '2' ? 2 :
                     (grain != nullptr && grain[0] == '4' ? 4 :
                      (grain != nullptr && grain[0] == '8' ? 8 : defaultLmul));
    const char* unitCap = getenv("MNN_RVV_ACCUM_UNIT");
    const int unit = lmul == 8 && unitCap != nullptr && atoi(unitCap) == 1 ? 1 : (lmul == 8 ? 2 : 4);
    const bool strided = wbits == 8 && MNNRVVAccumImplStrided();
    return (strided ? 1000 : 0) + 10 * lmul + unit;
#else
    (void)wbits;
    return 0;
#endif
}

bool validPlan(int planId) {
    switch (planId) {
        case 0:
        case 24:
        case 44:
        case 81:
        case 82:
        case 1024:
        case 1044:
        case 1081:
        case 1082:
            return true;
        default:
            return false;
    }
}

int tablePlan(int wbits, int regime, int oc, int ic, int blockNum, int blockL) {
    loadPlanFile();
    ResolverState& s = state();
    const int64_t vendor = static_cast<int64_t>(readCpuId("mvendorid"));
    const int64_t march = static_cast<int64_t>(readCpuId("marchid"));
    const int64_t vlen = currentVlen();
    int selected = -1;
    int selectedScore = -1;
    for (int i = 0; i < s.planCount; ++i) {
        const PlanRecord& record = s.plans[i];
        if ((record.wbits >= 0 && record.wbits != wbits) || (record.regime >= 0 && record.regime != regime) ||
            !matches(record.vendor, vendor) || !matches(record.march, march) || !matches(record.vlen, vlen) ||
            !matches(record.oc, oc) || !matches(record.ic, ic) || !matches(record.blockNum, blockNum) ||
            !matches(record.blockL, blockL) || (wbits == 4 && record.planId >= 1000) || !validPlan(record.planId)) {
            continue;
        }
        const int score = specificity(record);
        if (score > selectedScore) {
            selected = record.planId;
            selectedScore = score;
        }
    }
    return selected;
}

}

extern "C" int MNNRVVPlanIsAccum(int planId) {
    return planId != 0;
}

extern "C" int MNNRVVPlanIsStrided(int planId) {
    return planId >= 1000;
}

extern "C" int MNNRVVPlanSlot(int planId) {
    switch (planId) {
        case 24: return 1;
        case 44: return 2;
        case 81: return 3;
        case 82: return 4;
        case 1024: return 5;
        case 1044: return 6;
        case 1081: return 7;
        case 1082: return 8;
        default: return 0;
    }
}

extern "C" const char* MNNRVVPlanName(int planId, char* buffer, size_t size) {
    if (planId == 0) {
        snprintf(buffer, size, "reduction");
    } else {
        snprintf(buffer, size, "m%d/U%d%s", (planId % 1000) / 10, planId % 10, planId >= 1000 ? "/strided" : "");
    }
    return buffer;
}

extern "C" int MNNRVVPlanResolve(int wbits, int plane, int oc, int ic, int blockNum, int blockL, int contractOk,
                                  int accumAvail) {
    std::lock_guard<std::mutex> lock(stateMutex());
    ResolverState& s = state();
    loadPlanFile();
    const int regime = currentRegime(plane);
    const uint64_t key = planKey(wbits, regime, oc, ic, blockNum, blockL);
    for (int i = 0; i < s.cacheCount; ++i) {
        if (s.cache[i].key == key) {
            ++s.cache[i].hits;
            return s.cache[i].planId;
        }
    }
    int planId = 0;
    const int table = tablePlan(wbits, regime, oc, ic, blockNum, blockL);
    if (table >= 0 && contractOk && accumAvail) {
        planId = table;
    } else if (contractOk && accumAvail) {
        planId = fallbackPlan(wbits);
    }
    if (!contractOk || !accumAvail) {
        planId = 0;
    }
    if (s.cacheCount < kMaxCache) {
        PlanCacheEntry& entry = s.cache[s.cacheCount++];
        entry.key = key;
        entry.planId = planId;
        entry.wbits = wbits;
        entry.regime = regime;
        entry.oc = oc;
        entry.ic = ic;
        entry.hits = 0;
    }
    if (s.trace) {
        char name[32];
        fprintf(stderr, "[MNN][RVV-PLAN] key=0x%016llx dtype=W%d regime=%s shape=oc%d,ic%d,bk%d,bl%d selected=%s source=%s cache=miss\n",
                static_cast<unsigned long long>(key), wbits, regime ? "prefill" : "decode", oc, ic, blockNum, blockL,
                MNNRVVPlanName(planId, name, sizeof(name)), table >= 0 ? "table" : "fallback");
        fflush(stderr);
    }
    return planId;
}

extern "C" void MNNRVVPlanInstalled(int planId, int installOk, const char* fallbackReason) {
    std::lock_guard<std::mutex> lock(stateMutex());
    ResolverState& s = state();
    loadPlanFile();
    if (!s.trace) {
        return;
    }
    const char* reason = fallbackReason == nullptr ? "none" : fallbackReason;
    for (int i = 0; i < s.installCount; ++i) {
        if (s.installs[i].planId == planId && s.installs[i].installOk == installOk &&
            strcmp(s.installs[i].reason, reason) == 0) {
            ++s.installs[i].count;
            return;
        }
    }
    if (s.installCount < kMaxInstallRecords) {
        InstallRecord& record = s.installs[s.installCount++];
        record.planId = planId;
        record.installOk = installOk;
        strncpy(record.reason, reason, sizeof(record.reason) - 1);
        record.reason[sizeof(record.reason) - 1] = '\0';
        record.count = 1;
    }
    char name[32];
    fprintf(stderr, "[MNN][RVV-PLAN] install=%s executed=%s fallback=%s\n", installOk ? "ok" : "fail",
            installOk ? MNNRVVPlanName(planId, name, sizeof(name)) : "reduction", reason);
    fflush(stderr);
}
