// A Hadamard-rotated projection over weights kept in a GGUF release's own block formats.
//
// The ternary releases rotate every body projection's input, and an artifact may hold those rows in
// the source's ggml blocks (tools/convert/ternary.py's rotated_source). This test drives the
// production composition that executes such a Use -- execution::project_add, i.e. rotate the
// activation with the Use's signs and add the GGUF product to the residual -- and checks it against
// an FP64 oracle built from this tree's independent block decode and a naive Sylvester transform,
// so neither the production decoder nor the production Hadamard kernel is the reference.

#include "core/device.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "models/qwen3_5/execution/linear.h"

#include "cuda_availability.h"
#include "ops/gguf_reference.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::test;

namespace execution = ninfer::models::qwen3_5::execution;

constexpr std::int32_t kHidden  = 5120;
constexpr std::int32_t kHadamard = 1024;

// The gated projection of a Hadamard-rotated checkpoint: its represented values are the source's,
// and its product quantises the activation to ggml's q8_1 as that format's arithmetic requires.
constexpr ReductionCriterion kA8Tolerance{/*relative_l2=*/0.04,
                                          /*gross_absolute=*/1.0 / 256.0,
                                          /*gross_relative=*/0.06};

struct Case {
    QType qtype;
    const char* name;
    std::int32_t k;
    std::vector<int> half_fields; // binary16 multipliers inside one block
};

Weight gguf_weight(QType qtype, void* data, std::int32_t n, std::int32_t k) {
    const auto block = gguf_block_shape(qtype);
    Weight weight{};
    weight.payload         = data;
    weight.payload_bytes   = std::uint64_t(n) * std::uint64_t(k / block.elements) * block.bytes;
    weight.qtype           = qtype;
    weight.layout          = QuantLayout::GgufBlocks;
    weight.ndim            = 2;
    weight.n               = n;
    weight.k               = k;
    weight.shape[0]        = n;
    weight.shape[1]        = k;
    weight.padded_shape[0] = n;
    weight.padded_shape[1] = k;
    weight.qdata           = data;
    weight.group_size      = static_cast<std::uint32_t>(block.elements);
    weight.group           = block.elements;
    return weight;
}

// Random blocks whose binary16 multipliers are finite, so every stored value is representable.
std::vector<std::uint8_t> random_blocks(QType qtype, std::int32_t rows, std::int32_t k,
                                        const std::vector<int>& half_fields, std::uint32_t seed) {
    const auto block        = gguf_block_shape(qtype);
    const std::int64_t count = std::int64_t(rows) * (k / block.elements);
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(count) * block.bytes);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> scale(0.002F, 0.02F);
    for (auto& value : bytes) { value = static_cast<std::uint8_t>(byte(rng)); }
    for (std::int64_t i = 0; i < count; ++i) {
        std::uint8_t* base = bytes.data() + i * block.bytes;
        for (const int offset : half_fields) {
            const std::uint16_t bits = f32_to_bf16(scale(rng)) & 0x7f80U;
            std::memcpy(base + offset, &bits, sizeof(bits));
        }
    }
    return bytes;
}

// The forward transform of hadamard_transform.h: per 1024-block of the fastest axis, the
// normalized Sylvester Walsh-Hadamard matrix applied to the sign-weighted input.
void forward_transform(std::vector<double>& column) {
    for (std::size_t begin = 0; begin < column.size(); begin += kHadamard) {
        double* block = column.data() + begin;
        for (int length = 1; length < kHadamard; length <<= 1) {
            for (int i = 0; i < kHadamard; i += length << 1) {
                for (int j = 0; j < length; ++j) {
                    const double a = block[i + j], b = block[i + j + length];
                    block[i + j]          = a + b;
                    block[i + j + length] = a - b;
                }
            }
        }
        for (int i = 0; i < kHadamard; ++i) { block[i] *= 1.0 / 32.0; }
    }
}

// The activation the Op's rotated input holds, at the BF16 boundary it is stored through.
std::vector<std::uint16_t> rotated_activation(const std::vector<std::uint16_t>& activation,
                                              const std::vector<std::uint16_t>& signs,
                                              std::int32_t k, std::int32_t tokens) {
    std::vector<std::uint16_t> rotated(static_cast<std::size_t>(k) * tokens);
    std::vector<double> column(static_cast<std::size_t>(k));
    for (std::int32_t t = 0; t < tokens; ++t) {
        for (std::int32_t i = 0; i < k; ++i) {
            column[static_cast<std::size_t>(i)] =
                double(bf16_to_f32(activation[static_cast<std::size_t>(t) * k + i])) *
                double(bf16_to_f32(signs[static_cast<std::size_t>(i)]));
        }
        forward_transform(column);
        for (std::int32_t i = 0; i < k; ++i) {
            rotated[static_cast<std::size_t>(t) * k + i] = f32_to_bf16(float(column[i]));
        }
    }
    return rotated;
}

int run_case(const Case& c, DeviceContext& device) {
    const std::int32_t tokens_samples[] = {1, 8, 9, 32};
    const std::vector<std::int32_t> rows{0, 1, 3, 17, 64, 128, 255, 1000, 4096, kHidden - 1};
    std::mt19937 rng(0x51u + static_cast<std::uint32_t>(c.k));
    std::normal_distribution<float> normal(0.0F, 1.0F);

    const auto blocks = random_blocks(c.qtype, kHidden, c.k, c.half_fields, 0x91u);
    DeviceBuffer weight_device(blocks.size());
    weight_device.copy_from_host(blocks.data(), blocks.size());

    std::vector<std::uint16_t> signs(static_cast<std::size_t>(c.k));
    for (std::int32_t i = 0; i < c.k; ++i) {
        // An asymmetric sign pattern, so a dropped or duplicated sign changes the result.
        const std::uint32_t mixed = (static_cast<std::uint32_t>(i) * 2654435761U) ^ 0x5bd1e995U;
        signs[static_cast<std::size_t>(i)] = f32_to_bf16((mixed >> 13) % 3u == 0 ? -1.0F : 1.0F);
    }
    DeviceBuffer signs_device(signs.size() * sizeof(std::uint16_t));
    signs_device.copy_from_host(signs.data(), signs_device.bytes);

    const Weight weight = gguf_weight(c.qtype, weight_device.p, kHidden, c.k);
    execution::LinearParameters parameters;
    parameters.weight          = weight;
    parameters.policy          = ops::LinearPolicy::AllowA8;
    parameters.hadamard_signs =
        Tensor(signs_device.p, DType::BF16, {c.k});

    int failures = 0;
    for (const std::int32_t tokens : tokens_samples) {
        std::vector<std::uint16_t> activation(static_cast<std::size_t>(c.k) * tokens);
        for (std::int32_t t = 0; t < tokens; ++t) {
            for (std::int32_t i = 0; i < c.k; ++i) {
                activation[static_cast<std::size_t>(t) * c.k + i] =
                    f32_to_bf16(normal(rng));
            }
        }
        std::vector<std::uint16_t> residual(static_cast<std::size_t>(kHidden) * tokens);
        for (auto& value : residual) { value = f32_to_bf16(normal(rng)); }

        DeviceBuffer activation_device(activation.size() * sizeof(std::uint16_t));
        DeviceBuffer residual_device(residual.size() * sizeof(std::uint16_t));
        activation_device.copy_from_host(activation.data(), activation_device.bytes);
        residual_device.copy_from_host(residual.data(), residual_device.bytes);
        Tensor x(activation_device.p, DType::BF16, {c.k, tokens});
        Tensor residual_view(residual_device.p, DType::BF16, {kHidden, tokens});

        const std::size_t inner = ops::linear_add_workspace_capacity_bytes(
            c.qtype, kHidden, c.k, parameters.policy, tokens, tokens);
        const std::size_t capacity = execution::rotated_workspace_bytes(
            parameters.hadamard_signs, c.k, tokens, inner);
        WorkspaceArena workspace(std::max<std::size_t>(capacity, 256));
        execution::project_add(x, parameters, residual_view, workspace, device.stream);
        cuda_synchronize(device.stream);
        if (workspace.used() != 0 || workspace.peak_used() > capacity) {
            std::cerr << c.name << " T=" << tokens << ": workspace exceeds its capacity\n";
            ++failures;
        }

        // The oracle: the rotation the Op stores in BF16, then one FP64 dot product per sampled row
        // against the independently decoded weights, plus the residual.
        const auto rotated = rotated_activation(activation, signs, c.k, tokens);
        const auto residual_after =
            from_device<std::uint16_t>(residual_device.p, static_cast<std::size_t>(kHidden) * tokens);
        std::vector<double> got, want;
        const std::int32_t per_row = c.k / gguf_reference::kSuperBlock;
        const auto block           = gguf_block_shape(c.qtype);
        std::vector<double> decoded(gguf_reference::kSuperBlock);
        for (const std::int32_t row : rows) {
            for (std::int32_t t = 0; t < tokens; ++t) {
                double accumulator = 0;
                for (std::int32_t b = 0; b < per_row; ++b) {
                    gguf_reference::decode_block(
                        c.qtype,
                        blocks.data() + static_cast<std::size_t>(row) * per_row * block.bytes +
                            b * block.bytes,
                        decoded.data());
                    const std::size_t base =
                        static_cast<std::size_t>(t) * c.k +
                        static_cast<std::size_t>(b) * gguf_reference::kSuperBlock;
                    for (int i = 0; i < gguf_reference::kSuperBlock; ++i) {
                        accumulator += decoded[static_cast<std::size_t>(i)] *
                                       double(bf16_to_f32(rotated[base + i]));
                    }
                }
                want.push_back(double(bf16_to_f32(
                                  residual[static_cast<std::size_t>(t) * kHidden + row])) +
                              accumulator);
                got.push_back(double(
                    bf16_to_f32(residual_after[static_cast<std::size_t>(t) * kHidden + row])));
            }
        }
        failures += verify_reduction(std::string(c.name) + " rotated GGUF projection T=" +
                                         std::to_string(tokens),
                                     got, want, kA8Tolerance);
        // The Op updates only the residual.
        const auto activation_after = from_device<std::uint16_t>(activation_device.p,
                                                                 activation.size());
        const auto signs_after =
            from_device<std::uint16_t>(signs_device.p, signs.size());
        if (activation_after != activation || signs_after != signs) {
            std::cerr << c.name << " T=" << tokens << ": inputs were modified\n";
            ++failures;
        }
    }
    return failures;
}

} // namespace

int main() {
    int count                   = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (cuda_unavailable(count_err) || count == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    cuda_check(count_err, "cudaGetDeviceCount");
    try {
        DeviceContext device(0);
        const Case cases[] = {
            {QType::GGUF_Q2_K, "gguf_q2_k", 6144, {80, 82}},
            {QType::GGUF_Q3_K, "gguf_q3_k", 17408, {108}},
        };
        int failures = 0;
        for (const auto& c : cases) { failures += run_case(c, device); }
        if (failures == 0) { std::cout << "ok\n"; }
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: unexpected exception: " << error.what() << '\n';
        return 1;
    }
}
