#if defined(MNN_USE_RVV) && defined(__riscv)
#include "MNNTestSuite.h"
#include "backend/cpu/compute/CommonOptFunction.h"
#include <cmath>
#include <cstring>
#include <vector>

namespace {
int weightValue(int bits, int channel, int block, int depth) {
    const int value = channel * 3 + block * 5 + depth * 7;
    return bits == 4 ? value % 16 : value % 19 - 9;
}

bool checkAccum(const MNN::MatmulRelatedFunctions& functions, int bits, int plan, int tokens, int channels,
                int blockLength, bool asymmetric, bool split) {
    const int channelUnit = 4;
    const int depthUnit = 16;
    const int blocks = 3;
    const int paddedChannels = (channels + channelUnit - 1) / channelUnit * channelUnit;
    const int depthCount = blockLength * depthUnit;
    const int weightBytes = blockLength * channelUnit * depthUnit / (bits == 4 ? 2 : 1);
    const int blockBytes = weightBytes + 2 * channelUnit * sizeof(float);
    std::vector<int8_t> packed(paddedChannels / channelUnit * blocks * blockBytes, 0);
    std::vector<float> weightSums(paddedChannels * blocks, 0.f);
    std::vector<float> bias(paddedChannels, 0.f);
    for (int channel = 0; channel < channels; ++channel) {
        const int group = channel / channelUnit;
        const int lane = channel % channelUnit;
        bias[channel] = (channel % 5 - 2) / 8.f;
        for (int block = 0; block < blocks; ++block) {
            int8_t* base = packed.data() + (group * blocks + block) * blockBytes;
            float* scale = reinterpret_cast<float*>(base + weightBytes);
            float* offset = scale + channelUnit;
            scale[lane] = (channel % 3 + 1) / 32.f;
            offset[lane] = (block - 1) / 16.f;
            float sum = 0.f;
            for (int depth = 0; depth < depthCount; ++depth) {
                const int value = weightValue(bits, channel, block, depth);
                const int depthGroup = depth / depthUnit;
                const int depthLane = depth % depthUnit;
                if (bits == 4) {
                    uint8_t* nibble = reinterpret_cast<uint8_t*>(base) + depthGroup * 32 + (lane % 2) * 16 + depthLane;
                    *nibble |= static_cast<uint8_t>(lane < 2 ? value << 4 : value);
                } else {
                    base[depthGroup * channelUnit * depthUnit + lane * depthUnit + depthLane] = value;
                }
                sum += value * scale[lane] + offset[lane];
            }
            weightSums[(group * blocks + block) * channelUnit + lane] = sum;
        }
    }
    std::vector<int8_t> source(blocks * depthCount * tokens);
    std::vector<float> inputScale(asymmetric ? blocks * tokens : tokens);
    std::vector<float> inputBias(blocks * tokens, 0.f);
    std::vector<float> sourceSum(blocks * tokens, 0.f);
    for (int token = 0; token < tokens; ++token) {
        if (!asymmetric) {
            inputScale[token] = (token % 3 + 1) / 8.f;
        }
        for (int block = 0; block < blocks; ++block) {
            const float scale = asymmetric ? (block + token % 3 + 1) / 8.f : inputScale[token];
            if (asymmetric) {
                inputScale[block * tokens + token] = scale;
                inputBias[block * tokens + token] = (token - block) / 32.f;
            }
            for (int depth = 0; depth < depthCount; ++depth) {
                const int value = (token * 5 + block * 3 + depth * 7) % 15 - 7;
                const int index =
                    ((block * blockLength + depth / depthUnit) * tokens + token) * depthUnit + depth % depthUnit;
                source[index] = value;
                sourceSum[block * tokens + token] += value * scale;
            }
        }
    }
    const auto reorderSize =
        bits == 4 ? functions.MNNReorderWeightAccumH_W4_Size_Plan : functions.MNNReorderWeightAccumH_Size_Plan;
    const auto reorder = bits == 4 ? functions.MNNReorderWeightAccumH_W4_Plan : functions.MNNReorderWeightAccumH_Plan;
    const auto kernel = bits == 4 ? functions.Int8GemmKernel_AccumH_W4 : functions.Int8GemmKernel_AccumH;
    if (reorderSize == nullptr || reorder == nullptr || kernel == nullptr) {
        MNN_ERROR("RVV accum functions are not registered for W%d\n", bits);
        return false;
    }
    const size_t bytes = reorderSize(channels, paddedChannels, blocks, blockLength, plan);
    std::vector<int8_t> reordered(bytes);
    reorder(reordered.data(), packed.data(), weightSums.data(), channels, paddedChannels, blocks, blockLength, plan);
    std::vector<float> output(paddedChannels * tokens + 16, -123456.f);
    QuanPostTreatParameters post = {};
    post.useInt8 = 0;
    post.biasFloat = bias.data();
    post.srcKernelSum = sourceSum.data();
    post.inputScale = inputScale.data();
    post.inputBias = asymmetric ? inputBias.data() : nullptr;
    post.blockNum = blocks;
    float limits[2] = {-10000.f, 10000.f};
    post.fp32minmax = limits;
    for (int start = 0; start < paddedChannels;) {
        const int count = split && start == 0 ? channelUnit : paddedChannels - start;
        post.accumHOcStart = start;
        post.biasFloat = bias.data() + start;
        int8_t* destination = reinterpret_cast<int8_t*>(output.data() + start * tokens);
        kernel(destination, source.data(), reordered.data(), blockLength, tokens * channelUnit * sizeof(float),
               count / channelUnit, &post, tokens);
        start += count;
    }
    for (int channel = 0; channel < paddedChannels; ++channel) {
        for (int token = 0; token < tokens; ++token) {
            float expected = bias[channel];
            if (channel < channels) {
                const float weightScale = (channel % 3 + 1) / 32.f;
                for (int block = 0; block < blocks; ++block) {
                    int accumulator = 0;
                    for (int depth = 0; depth < depthCount; ++depth) {
                        const int input = (token * 5 + block * 3 + depth * 7) % 15 - 7;
                        accumulator += input * weightValue(bits, channel, block, depth);
                    }
                    const int scaleIndex = asymmetric ? block * tokens + token : token;
                    const float weightOffset = (block - 1) / 16.f;
                    expected += accumulator * weightScale * inputScale[scaleIndex] +
                                sourceSum[block * tokens + token] * weightOffset;
                    if (asymmetric) {
                        const int sumIndex =
                            (channel / channelUnit * blocks + block) * channelUnit + channel % channelUnit;
                        expected += inputBias[block * tokens + token] * weightSums[sumIndex];
                    }
                }
            }
            expected = expected < limits[0] ? limits[0] : (expected > limits[1] ? limits[1] : expected);
            const int index = (channel / channelUnit * tokens + token) * channelUnit + channel % channelUnit;
            const float actual = output[index];
            if (!std::isfinite(actual) || std::fabs(actual - expected) > 1e-4f) {
                MNN_ERROR(
                    "RVV accum mismatch W%d plan=%d tokens=%d channels=%d blockL=%d asym=%d split=%d "
                    "channel=%d token=%d expected=%f actual=%f\n",
                    bits, plan, tokens, channels, blockLength, asymmetric, split, channel, token, expected, actual);
                return false;
            }
        }
    }
    for (int index = paddedChannels * tokens; index < paddedChannels * tokens + 16; ++index) {
        if (output[index] != -123456.f) {
            MNN_ERROR("RVV accum output overrun W%d plan=%d tokens=%d\n", bits, plan, tokens);
            return false;
        }
    }
    return true;
}
} // namespace

class RVVAccumHTest : public MNNTestCase {
public:
    bool run(int precision) override {
        const auto core = MNN::MNNGetCoreFunctions();
        if (core == nullptr || !core->supportRVV) {
            MNN_ERROR("RVV accum test requires an initialized RVV CPU backend\n");
            return false;
        }
        const auto& functions = core->int8MatmulRelatedFunctions;
        const int plans[] = {24, 44, 82};
        const int channelCounts[] = {13, 36, 68};
        for (int bits = 4; bits <= 8; bits += 4) {
            for (int plan : plans) {
                for (int tokens = 1; tokens <= GEMM_INT8_DST_XUNIT_RVV; ++tokens) {
                    for (int channels : channelCounts) {
                        for (int blockLength = 1; blockLength <= 2; ++blockLength) {
                            for (int asymmetric = 0; asymmetric <= 1; ++asymmetric) {
                                for (int split = 0; split <= 1; ++split) {
                                    if (!checkAccum(functions, bits, plan, tokens, channels, blockLength, asymmetric,
                                                    split)) {
                                        return false;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return true;
    }
};

MNNTestSuiteRegister(RVVAccumHTest, "backend/riscv/accum_h");
#endif
