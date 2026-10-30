#pragma once

#include "runtime/prefix_cache/disk_catalog.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <span>
#include <thread>
#include <unordered_set>
#include <variant>
#include <vector>

namespace ninfer::runtime::prefix_cache {

// The catalog's sole runtime I/O executor. Submission and completion adoption never acquire a
// payload/journal lock. Borrowed buffers remain leased until their completion is adopted.
class PrefixDiskWorker final {
public:
    enum class Kind : std::uint8_t { WriteBlock, WriteImage, ReadBlock, ReadImage, ReadTail, Remove };

    struct BlockWrite {
        NodeRef node;
        std::uint32_t record = kNoId;
        PrefixDiskCatalog::BlockIdentityRec identity;
        std::span<const std::byte> source;
    };
    struct ImageWrite {
        SnapshotRef snapshot;
        std::uint32_t record = kNoId;
        PrefixDiskCatalog::SnapIdentityRec identity;
        std::vector<const std::byte*> segments;
        std::span<const std::byte> tail;
    };
    struct Read {
        Kind kind = Kind::ReadBlock;
        NodeRef node;
        SnapshotRef snapshot;
        std::uint32_t record = kNoId;
        std::uint32_t slot = 0;
        std::span<std::byte> destination;
    };
    struct Completion {
        Kind kind;
        NodeRef node;
        SnapshotRef snapshot;
        std::uint32_t record = kNoId;
        std::uint32_t slot = 0;
        std::uint64_t bytes = 0;
        bool ok = false;
    };

    // The catalog must already be open and outlive this worker. The byte bound includes running
    // writes and completed writes whose source leases the Engine has not released yet.
    PrefixDiskWorker(PrefixDiskCatalog& catalog, PrefixDiskCatalog::Geometry geometry,
                     std::uint64_t max_write_bytes, std::uint32_t max_reads);
    ~PrefixDiskWorker();
    PrefixDiskWorker(const PrefixDiskWorker&) = delete;
    PrefixDiskWorker& operator=(const PrefixDiskWorker&) = delete;
    // Initial submissions may be batched before starting the executor. shutdown() also settles
    // work accepted before start(), so no borrowed buffer can outlive its owner.
    void start();

    [[nodiscard]] bool can_write(std::uint64_t bytes) const;
    [[nodiscard]] bool submit(BlockWrite&& job);
    [[nodiscard]] bool submit(ImageWrite&& job);
    [[nodiscard]] bool submit(const Read& job);
    void remove(std::uint32_t record);
    [[nodiscard]] std::vector<Completion> take_completions();
    [[nodiscard]] bool busy() const;
    // Recovery/shutdown barrier only: finish accepted I/O before its source owner is cleared.
    // Completed records are still adopted by the Engine, not by this wait.
    void wait_idle();
    [[nodiscard]] std::uint64_t completion_revision() const noexcept {
        return completion_revision_.load(std::memory_order_acquire);
    }
    void shutdown() noexcept;

private:
    void run();
    Completion execute(const BlockWrite& job);
    Completion execute(const ImageWrite& job);
    Completion execute(const Read& job);
    Completion execute_removal(std::uint32_t record);

    PrefixDiskCatalog& catalog_;
    PrefixDiskCatalog::Geometry geometry_;
    std::uint64_t max_write_bytes_;
    std::uint32_t max_reads_;
    mutable std::mutex mutex_;
    std::condition_variable wake_;
    std::condition_variable idle_;
    std::deque<std::variant<BlockWrite, ImageWrite>> writes_;
    std::deque<Read> reads_;
    std::deque<std::uint32_t> removals_;
    std::unordered_set<std::uint32_t> removing_;
    std::vector<Completion> completed_;
    std::uint64_t write_bytes_ = 0;
    std::uint32_t read_count_ = 0;
    std::atomic<std::uint64_t> completion_revision_{0};
    bool running_ = false;
    bool stopping_ = false;
    std::thread thread_;
};

} // namespace ninfer::runtime::prefix_cache
