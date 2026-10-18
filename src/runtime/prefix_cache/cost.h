#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>

namespace ninfer::runtime::prefix_cache {

// Machine model used only to rank admission sources and to value snapshots for eviction. It never
// decides feasibility. The Program fills it from the calibrated prefill coefficients and the
// transfer bandwidth measured at startup.
struct CacheCostModel {
    double chunk_seconds          = 0.0;
    std::uint32_t chunk_tokens    = 2048;
    double token_seconds          = 1.0 / 4000.0;
    double attention_pair_seconds = 0.0;
    double h2d_bytes_per_second   = 50.0e9;
    // Conservative uncalibrated disk throughput; unlike H2D it is not measured at startup.
    // A nonpositive rate disables disk candidates without affecting Host-only restores.
    double disk_read_bytes_per_second = 1.0e9;
    double transfer_batch_seconds = 20.0e-6;

    // Prefill of `tokens` tokens appended after `base` reused tokens.
    [[nodiscard]] double prefill_seconds(std::uint32_t base, std::uint32_t tokens) const noexcept {
        if (tokens == 0) { return 0.0; }
        const double s            = static_cast<double>(tokens);
        const double pairs        = static_cast<double>(base) * s + s * (s + 1.0) / 2.0;
        const std::uint32_t chunk = std::max<std::uint32_t>(chunk_tokens, 1U);
        const double chunks       = static_cast<double>((tokens + chunk - 1U) / chunk);
        return chunks * chunk_seconds + s * token_seconds + pairs * attention_pair_seconds;
    }

    [[nodiscard]] double restore_seconds(std::uint64_t bytes) const noexcept {
        if (bytes == 0) { return 0.0; }
        return transfer_batch_seconds + static_cast<double>(bytes) / h2d_bytes_per_second;
    }

    // A restore that moves `host_bytes` over H2D and reads `disk_bytes` from the disk tier into
    // pinned staging. The staged bytes then ride the same H2D path, so only their slower disk
    // read is charged: the disk read is the serialization point, not an added transfer. A
    // nonpositive rate disables the tier: every disk candidate ranks as never worth choosing,
    // while Host-only restores stay finite.
    [[nodiscard]] double restore_seconds(std::uint64_t host_bytes,
                                         std::uint64_t disk_bytes) const noexcept {
        if (disk_bytes == 0) { return restore_seconds(host_bytes); }
        if (disk_read_bytes_per_second <= 0.0) {
            return std::numeric_limits<double>::infinity();
        }
        return transfer_batch_seconds + static_cast<double>(disk_bytes) / disk_read_bytes_per_second;
    }
};

} // namespace ninfer::runtime::prefix_cache
