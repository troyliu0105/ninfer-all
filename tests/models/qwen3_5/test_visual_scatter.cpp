#include "guarded_main.h"
#include "ops/op_tester.h"
#include "models/qwen3_5/execution/visual_scatter.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

int run_visual_scatter_checks() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    constexpr std::int32_t d = 5120;
    constexpr std::int32_t t = 4;
    constexpr std::int32_t v = 3;
    std::vector<float> token_embeddings(static_cast<std::size_t>(d) * t);
    std::vector<float> visual_embeddings(static_cast<std::size_t>(d) * v);
    for (int col = 0; col < t; ++col) {
        std::fill_n(token_embeddings.data() + static_cast<std::size_t>(col) * d, d,
                    static_cast<float>(10 + col));
    }
    for (int col = 0; col < v; ++col) {
        std::fill_n(visual_embeddings.data() + static_cast<std::size_t>(col) * d, d,
                    static_cast<float>(100 + col));
    }
    round_to_bf16(token_embeddings);
    round_to_bf16(visual_embeddings);
    DeviceBuffer dvisual = to_device_bf16(visual_embeddings);
    Tensor visual(dvisual.p, DType::BF16, {d, v});
    const std::vector<std::int32_t> scatter_indices{2, 4, 7};
    WorkspaceArena work(1ULL << 20);

    auto run = [&](std::int32_t prompt_tokens, const char* label) {
        DeviceBuffer dinput = to_device_bf16(token_embeddings);
        Tensor input(dinput.p, DType::BF16, {d, t});
        work.reset();
        const auto window = models::qwen3_5::plan_mtp_alignment_window(
            static_cast<std::uint32_t>(prompt_tokens), 0, t);
        const auto overlap = models::qwen3_5::shifted_visual_overlap(
            scatter_indices, static_cast<std::uint32_t>(prompt_tokens), window);
        Tensor destination_indices =
            work.alloc(DType::I32, {static_cast<std::int32_t>(overlap.size())});
        models::qwen3_5::detail::scatter_shifted_visual_embeddings(input, visual, overlap,
                                                                   destination_indices, nullptr);
        cudaDeviceSynchronize();

        std::vector<double> reference(token_embeddings.begin(), token_embeddings.end());
        for (int row = 0; row < d; ++row) {
            reference[static_cast<std::size_t>(1) * d + row] = visual_embeddings[row];
            if (prompt_tokens > 4) {
                reference[static_cast<std::size_t>(3) * d + row] =
                    visual_embeddings[static_cast<std::size_t>(d) + row];
            }
        }
        return verify_exact(label, from_device_bf16(dinput, reference.size()), reference);
    };

    int failures = 0;
    failures += run(8, "MTP shifted visual composition crosses chunk boundary");
    failures += run(4, "MTP shifted visual composition keeps bonus token embedding");

    // A text chunk that ends exactly on an item's first visual token scatters none of its own
    // (count == 0) yet the shifted MTP input of its last token names that column, so the staged
    // overlay tensor must still exist at the model's hidden width. Regression for the worker
    // crash "scatter: expected src [D,V], indices [V], dst [D,T]" (overlay Vision residency).
    {
        const std::vector<std::int32_t> boundary_scatter{t, t + 1, t + 2};
        const auto window = models::qwen3_5::plan_mtp_alignment_window(
            static_cast<std::uint32_t>(t * 4), 0, t);
        const auto overlap =
            models::qwen3_5::shifted_visual_overlap(boundary_scatter, t * 4, window);
        const bool boundary_named = overlap.size() == 1 && overlap.destination_columns.front() == t - 1;
        if (!boundary_named) {
            std::cout << "FAIL boundary chunk names the next item's first visual column\n";
            ++failures;
        } else {
            // The pre-fix workspace allocated no visual_embeddings for this chunk (count == 0),
            // so the shifted scatter read a default [1,1] tensor and failed the scatter guard.
            std::vector<float> first_column(visual_embeddings.begin(),
                                            visual_embeddings.begin() + d);
            round_to_bf16(first_column);
            std::vector<std::uint16_t> column_bits(first_column.size());
            for (std::size_t i = 0; i < column_bits.size(); ++i) {
                column_bits[i] = f32_to_bf16(first_column[i]);
            }
            DeviceBuffer dinput = to_device_bf16(token_embeddings);
            Tensor input(dinput.p, DType::BF16, {d, t});
            work.reset();
            // The fixed workspace stages one column even though this chunk scatters none.
            Tensor staged = work.alloc(DType::BF16, {d, 1});
            cuda_check(cudaMemcpy(staged.data, column_bits.data(), staged.bytes(),
                                  cudaMemcpyHostToDevice),
                       "staging the boundary column");
            Tensor destination_indices =
                work.alloc(DType::I32, {static_cast<std::int32_t>(overlap.size())});
            models::qwen3_5::detail::scatter_shifted_visual_embeddings(
                input, staged, overlap, destination_indices, nullptr);
            cudaDeviceSynchronize();
            std::vector<double> reference(token_embeddings.begin(), token_embeddings.end());
            for (int row = 0; row < d; ++row) {
                reference[static_cast<std::size_t>(t - 1) * d + row] = first_column[row];
            }
            failures += verify_exact("MTP shifted visual scatter on a boundary chunk",
                                     from_device_bf16(dinput, reference.size()), reference);
        }
    }
    std::cout << (failures ? "FAIL" : "OK") << " shifted visual composition\n";
    return failures ? 1 : 0;
}

NINFER_GUARDED_TEST_MAIN(run_visual_scatter_checks)
