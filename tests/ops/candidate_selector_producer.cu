#include "core/pdl.cuh"

#include <cuda_runtime.h>
#include <cstdint>

namespace ninfer::test {
namespace {

// Publish an early PDL trigger, then leave the old, valid token ids visible long enough for a
// consumer missing its dependency wait to read them. This is also an ordinary producer without PDL.
__global__ void delayed_selector_inputs(const std::int32_t* ids, int count,
                                        const std::int32_t* anchors, int batch,
                                        std::int32_t* out_ids, std::int32_t* out_anchors) {
    pdl::enter();
    if (threadIdx.x == 0) {
        const unsigned long long begin = clock64();
        while (clock64() - begin < 1000000ULL) {}
    }
    __syncthreads();
    for (int i = threadIdx.x; i < count; i += blockDim.x) { out_ids[i] = ids[i]; }
    if (threadIdx.x < batch) { out_anchors[threadIdx.x] = anchors[threadIdx.x]; }
}

} // namespace

cudaError_t produce_selector_inputs(const std::int32_t* ids, int count,
                                     const std::int32_t* anchors, int batch,
                                     std::int32_t* out_ids, std::int32_t* out_anchors,
                                     cudaStream_t stream) {
    return pdl::launch_consumer({dim3(1), dim3(256), 0, stream}, delayed_selector_inputs,
                                ids, count, anchors, batch, out_ids, out_anchors);
}

} // namespace ninfer::test
