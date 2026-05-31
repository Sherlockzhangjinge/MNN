#include <stddef.h>
#include <stdint.h>
#include <string.h>

template <typename T>
static void MNNPackC8Common(T* dst, const T* src, size_t area, size_t depth, int* areaOffset) {
    const size_t srcAreaStride = (size_t)areaOffset[0];
    const size_t dstAreaStride = (size_t)areaOffset[1];
    const size_t depthC8 = depth / 8;
    const size_t remain = depth - depthC8 * 8;

    for (size_t z = 0; z < depthC8; ++z) {
        T* dstZ = dst + z * dstAreaStride * 8;
        for (size_t x = 0; x < area; ++x) {
            for (size_t y = 0; y < 8; ++y) {
                dstZ[x * 8 + y] = src[(z * 8 + y) * srcAreaStride + x];
            }
        }
    }

    if (remain > 0) {
        T* dstZ = dst + depthC8 * dstAreaStride * 8;
        for (size_t x = 0; x < area; ++x) {
            for (size_t y = 0; y < remain; ++y) {
                dstZ[x * 8 + y] = src[(depthC8 * 8 + y) * srcAreaStride + x];
            }
            for (size_t y = remain; y < 8; ++y) {
                dstZ[x * 8 + y] = (T)0;
            }
        }
    }
}

template <typename T>
static void MNNUnpackC8Common(T* dst, const T* src, size_t area, size_t depth, int* areaOffset) {
    const size_t srcAreaStride = (size_t)areaOffset[0];
    const size_t dstAreaStride = (size_t)areaOffset[1];
    const size_t depthC8 = depth / 8;
    const size_t remain = depth - depthC8 * 8;

    for (size_t z = 0; z < depthC8; ++z) {
        const T* srcZ = src + z * srcAreaStride * 8;
        for (size_t y = 0; y < 8; ++y) {
            T* dstY = dst + (z * 8 + y) * dstAreaStride;
            for (size_t x = 0; x < area; ++x) {
                dstY[x] = srcZ[x * 8 + y];
            }
        }
    }

    if (remain > 0) {
        const T* srcZ = src + depthC8 * srcAreaStride * 8;
        for (size_t y = 0; y < remain; ++y) {
            T* dstY = dst + (depthC8 * 8 + y) * dstAreaStride;
            for (size_t x = 0; x < area; ++x) {
                dstY[x] = srcZ[x * 8 + y];
            }
        }
    }
}

template <typename T>
static void MNNUnpackTransposeC8Common(T* dst, const T* src, size_t area, size_t depth, int* areaOffset) {
    const size_t dstAreaStride = (size_t)areaOffset[1];
    const size_t cDiv8 = depth / 8;
    const size_t cAlign = cDiv8 * 8;

    for (size_t hi = 0; hi < area; ++hi) {
        const T* srcHeight = src + hi * depth;
        T* dstHeight = dst + hi * 8;
        for (size_t ci = 0; ci < cDiv8; ++ci) {
            for (size_t i = 0; i < 8; ++i) {
                dstHeight[ci * dstAreaStride * 8 + i] = srcHeight[ci * 8 + i];
            }
        }
    }

    if (cAlign == depth) {
        return;
    }

    const size_t cRemain = depth - cAlign;
    const T* srcAlign = src + cAlign;
    T* dstAlign = dst + dstAreaStride * cAlign;
    for (size_t hi = 0; hi < area; ++hi) {
        const T* srcHeight = srcAlign + hi * depth;
        T* dstHeight = dstAlign + hi * 8;
        for (size_t i = 0; i < 8; ++i) {
            dstHeight[i] = (T)0;
        }
        for (size_t ci = 0; ci < cRemain; ++ci) {
            dstHeight[ci] = srcHeight[ci];
        }
    }
}

template <typename T>
static void MNNPackTransposeC8Common(T* dst, const T* src, size_t area, size_t depth, int* areaOffset) {
    const size_t srcAreaStride = (size_t)areaOffset[0];
    const size_t dstDepthStride = (size_t)areaOffset[1];
    const size_t cDiv8 = depth / 8;
    const size_t cAlign = cDiv8 * 8;

    for (size_t hi = 0; hi < area; ++hi) {
        const T* srcHeight = src + hi * 8;
        T* dstHeight = dst + hi * dstDepthStride;
        for (size_t ci = 0; ci < cDiv8; ++ci) {
            for (size_t i = 0; i < 8; ++i) {
                dstHeight[ci * 8 + i] = srcHeight[ci * srcAreaStride * 8 + i];
            }
        }
    }

    if (cAlign == depth) {
        return;
    }

    const size_t cRemain = depth - cAlign;
    const T* srcAlign = src + srcAreaStride * cAlign;
    T* dstAlign = dst + cAlign;
    for (size_t hi = 0; hi < area; ++hi) {
        const T* srcHeight = srcAlign + hi * 8;
        T* dstHeight = dstAlign + hi * dstDepthStride;
        for (size_t ci = 0; ci < cRemain; ++ci) {
            dstHeight[ci] = srcHeight[ci];
        }
    }
}

void MNNPackC8Int8_RVV(int8_t* dst, const int8_t* src, size_t area, size_t depth, int* areaOffset) {
    MNNPackC8Common(dst, src, area, depth, areaOffset);
}

void MNNUnpackC8Int8_RVV(int8_t* dst, const int8_t* src, size_t area, size_t depth, int* areaOffset) {
    MNNUnpackC8Common(dst, src, area, depth, areaOffset);
}

void MNNPackC8Int16_RVV(int16_t* dst, const int16_t* src, size_t area, size_t depth, int* areaOffset) {
    MNNPackC8Common(dst, src, area, depth, areaOffset);
}

void MNNUnpackC8Int16_RVV(int16_t* dst, const int16_t* src, size_t area, size_t depth, int* areaOffset) {
    MNNUnpackC8Common(dst, src, area, depth, areaOffset);
}

void MNNUnpackTransposeC8_RVV(float* dst, const float* src, size_t area, size_t depth, int* areaOffset) {
    MNNUnpackTransposeC8Common(dst, src, area, depth, areaOffset);
}

void MNNPackTransposeC8_RVV(float* dst, const float* src, size_t area, size_t depth, int* areaOffset) {
    MNNPackTransposeC8Common(dst, src, area, depth, areaOffset);
}

void MNNUnpackTransposeC8Int8_RVV(int8_t* dst, const int8_t* src, size_t area, size_t depth, int* areaOffset) {
    MNNUnpackTransposeC8Common(dst, src, area, depth, areaOffset);
}

void MNNPackTransposeC8Int8_RVV(int8_t* dst, const int8_t* src, size_t area, size_t depth, int* areaOffset) {
    MNNPackTransposeC8Common(dst, src, area, depth, areaOffset);
}

void MNNUnpackTransposeC8Int16_RVV(int16_t* dst, const int16_t* src, size_t area, size_t depth, int* areaOffset) {
    MNNUnpackTransposeC8Common(dst, src, area, depth, areaOffset);
}

void MNNPackTransposeC8Int16_RVV(int16_t* dst, const int16_t* src, size_t area, size_t depth, int* areaOffset) {
    MNNPackTransposeC8Common(dst, src, area, depth, areaOffset);
}
