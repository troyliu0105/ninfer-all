#pragma once

// The hybrid prefix cache's disk tier binding (docs/maintainer/hybrid-prefix-cache-disk-tier.md
// §2, §4, §5): one background I/O worker that demotes Host-resident closures to disk under
// Host pressure, and prepares disk-resident sources for a queued FIFO head.
//
// Thread ownership (A4 design note, local://hybrid-disk-tier-plan.md): the Engine worker owns
// every index and catalog metadata decision — victim selection, `finish_block`,
// `commit_snapshot`, `mark_disk_backed`, staging promotions. The I/O worker only moves bytes
// between pinned Host slabs and the catalog. A demotion closure is submitted whole: its block
// records, image segments and tail leave as independent jobs, but the closure commits only on
// the Engine worker once every one of its records reported completion. Jobs carry the pinned
// slab pointers captured at submission; the pin (taken with the job) keeps the bytes valid
// until completion publishes them.
//
// Demotion never blocks admission, restores or `drain()`: it runs from the same `poll()`
// cadence the Host write-through uses, and neither `drain()` nor `settle_prefetch()` ever
// waits on this tier. Staging (block payloads plus one whole StateImage) is pinned once with
// `PinnedHostBuffer` and accounted against the advertised Host budget by
// `plan_hybrid_disk_staging`; a staging slot recycles only after its H2D event (observed as the
// node turning device-Resident on the poll cadence).
//
// The tier also owns the catalog's reclamation: the index's `release_disk_record` /
// `release_snapshot_disk_records` callbacks arrive here and drop the record (journaling the
// removal, so a restart never resurrects it), and a full catalog is reclaimed with the index's
// own bounded disk victim selection instead of failing every later demotion.

#include "core/arena.h"
#include "models/qwen3_5/program/prefix/hybrid_cache.h"
#include "runtime/prefix_cache/disk_catalog.h"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

namespace pc = runtime::prefix_cache;

class HybridDiskTier final {
public:
    struct Options {
        std::filesystem::path directory;
        std::uint64_t block_capacity_bytes  = 0;
        std::uint64_t image_capacity_bytes  = 0;
        // Host slab watermarks (fractions of the slab pool): demotion starts at or above the
        // high watermark and submissions continue until the pool falls below the low one.
        // Demotion must start before the index's own GDSF eviction would drop Host copies:
        // the write-through path evicts as soon as no slab is free, so the tier warms at a
        // lower watermark and stops once pressure clears.
        double host_high_watermark = 0.75;
        double host_low_watermark  = 0.50;
        // Demotion closures considered per poll cycle.
        std::uint32_t demotion_batch_records = 16;
        // Staging block payloads, from `plan_hybrid_disk_staging`: the affordable part of the
        // advertised Host budget after the slab pool's own minimum geometry and the image/tail
        // staging. A configuration with no block slot must not open a tier.
        std::uint32_t staging_blocks = 0;
        // Bound on queued-but-unstarted demotion bytes.
        std::uint64_t max_inqueue_bytes = std::uint64_t{64} << 20U;
    };

    struct Stats {
        std::uint64_t demoted_blocks    = 0;
        std::uint64_t demoted_images    = 0;
        std::uint64_t committed_closures = 0;
        // Closures whose writes failed (or whose snapshot left the index): never committed.
        std::uint64_t failed_closures   = 0;
        std::uint64_t disk_read_bytes   = 0;
        std::uint64_t disk_write_bytes  = 0;
        std::uint64_t restored_blocks   = 0;
        std::uint64_t queue_rejects     = 0;
        // Catalog records the index released and bounded victim eviction reclaimed.
        std::uint64_t released_records  = 0;
        std::uint64_t evicted_records   = 0;
        // Release/removal retries still owed to the catalog after an I/O failure.
        std::uint32_t pending_removals  = 0;
        std::uint32_t catalog_blocks    = 0;
        std::uint32_t catalog_committed = 0;
    };

    // `fingerprint` is the hybrid cache fingerprint: a foreign catalog is wiped and reopened
    // empty (the directory is artifact-scoped).
    HybridDiskTier(Options options, const HybridHostLayout& host,
                   const std::string& fingerprint);
    ~HybridDiskTier();

    HybridDiskTier(const HybridDiskTier&)            = delete;
    HybridDiskTier& operator=(const HybridDiskTier&) = delete;

    struct RestoreSummary {
        bool ok = false;
        std::uint64_t blocks    = 0;
        std::uint64_t snapshots = 0;
        std::uint64_t kv_bytes  = 0;
    };
    // Replays the catalog and restores its committed closures into `index` (parents first,
    // anchors before their snapshots, nothing evicted to make room). `ok` is false when the
    // catalog was foreign (already wiped empty; the caller proceeds without disk content).
    [[nodiscard]] RestoreSummary restore_into(pc::PrefixCacheIndex& index);

    // ---- demotion (Engine worker) ----------------------------------------------------------------
    // One bounded step of Host-pressure demotion: no I/O, only victim selection and job
    // submission with their pins. Called from the poll() cadence points in hybrid_quote and
    // hybrid_publish_blocks.
    void maybe_demote(pc::PrefixCacheIndex& index, const HybridPrefixCache& cache);

    // Engine-worker completion step: finishes landed block records, marks disk-backed nodes,
    // and commits closures whose every record landed (reclaiming the snapshot's Host slabs
    // through the ordinary index path once the closure is safe).
    void publish_completions(pc::PrefixCacheIndex& index);

    // Whether any demotion write or staging read is outstanding.
    [[nodiscard]] bool busy() const noexcept;

    // Stops the worker after every accepted job settles, drops the records of closures that
    // never committed, and checkpoints the catalog. Never throws: it runs from destructors and
    // from the noexcept shutdown path, and reports a checkpoint failure on stderr.
    void shutdown() noexcept;

    // Wipes the catalog to empty and restarts the worker (Engine recovery).
    void reset();

    [[nodiscard]] Stats stats() const noexcept;

    // ---- cold preparation (Engine worker) ---------------------------------------------------------
    // Queues block records for staging reads on the I/O worker; bytes surface through
    // take_staged_blocks() on later poll cycles. False when the queue is full.
    [[nodiscard]] bool stage_blocks(const std::vector<pc::NodeRef>& nodes,
                                    const pc::PrefixCacheIndex& index);
    struct StagedBlock {
        pc::NodeRef node;
        const std::byte* bytes = nullptr;
        // Cycles since these bytes landed without an admission consuming them.
        std::uint32_t age = 0;
    };
    // Blocks whose staging bytes landed; the tier holds their slots until release_staged()
    // names them — by entry, because taking moves the entries out of the tier's list. Slots
    // recycle only after the consuming H2D copy landed (the node is device-Resident again).
    [[nodiscard]] std::vector<StagedBlock> take_staged_blocks();
    void release_staged(const std::vector<StagedBlock>& blocks);
    // Queues a whole snapshot image for staging; at most one at a time.
    [[nodiscard]] bool stage_image(pc::SnapshotRef snapshot,
                                   const pc::PrefixCacheIndex& index);
    // The staged image bytes once landed for this snapshot (nullptr otherwise); valid until
    // release_image().
    [[nodiscard]] const std::byte* take_staged_image(pc::SnapshotRef snapshot);
    // Drops a staged image that belongs to no snapshot the quotes still name.
    void release_staged_image_except(pc::SnapshotRef snapshot);
    void release_image();
    // The staged tail bytes once landed for this snapshot (nullptr otherwise).
    [[nodiscard]] const std::byte* staged_tail(pc::SnapshotRef snapshot) const noexcept;
    // Drops a staged tail that belongs to no snapshot the quotes still name.
    void release_staged_tail_except(pc::SnapshotRef snapshot);
    // Queues a snapshot's tail bundle for staging; at most one at a time.
    [[nodiscard]] bool stage_tail(pc::SnapshotRef snapshot, const pc::PrefixCacheIndex& index);
    // Returns a staged tail nobody consumed (the tier's single tail slot).
    void release_tail();

    // The pinned staging footprint (for the Program's startup ledger).
    [[nodiscard]] std::uint64_t staging_bytes() const noexcept;

    // ---- record reclamation (Engine worker) --------------------------------------------------------
    // The index dropped its last reference to `record`: the catalog drops the record and its
    // payloads so a restart never resurrects it. Engine-worker only; the record's own write has
    // completed by construction (the index only releases records that finished or aborted a fill
    // whose I/O returned). A catalog failure is owed back and retried at the next cadence point.
    void release_disk_record(std::uint32_t record) noexcept;
    // A snapshot the index removed: the closure's image segments and tail leave with it. The
    // anchor-path block records are released through release_disk_record by the index itself.
    void release_snapshot_disk_records(std::uint32_t record) noexcept;

private:
    // A block record write: the worker copies `source` (the node's pinned Host slab payload)
    // into catalog record `record`. Completion runs finish_block + mark_disk_backed.
    struct BlockJob {
        pc::NodeRef node;
        std::uint32_t record = pc::PrefixDiskCatalog::kNoRecord;
        pc::PrefixDiskCatalog::BlockIdentityRec identity;
        const std::byte* source = nullptr;
    };

    // A snapshot closure write: image segments (the snapshot's pinned Host slab payloads, in
    // slab order) and, when a tail exists, its tail slab payload. `records` names every block
    // record of the closure's anchor path (already finished or finishing independently); the
    // Engine worker commits once the segment writes and those blocks all landed.
    struct ImageJob {
        pc::SnapshotRef snapshot;
        std::uint32_t record = pc::PrefixDiskCatalog::kNoRecord;
        std::vector<const std::byte*> segments;
        const std::byte* tail = nullptr;
        pc::PrefixDiskCatalog::SnapIdentityRec identity;
    };

    // A staged read: the worker reads the record into a staging slot.
    struct ReadJob {
        pc::NodeRef node;
        std::uint32_t disk_ref = pc::kNoId;
        std::uint32_t slot     = 0;
    };

    struct ImageReadJob {
        pc::SnapshotRef snapshot;
        std::uint32_t disk_ref = pc::kNoId;
    };

    struct TailReadJob {
        pc::SnapshotRef snapshot;
        std::uint32_t disk_ref = pc::kNoId;
    };

    struct CompletedBlock {
        pc::NodeRef node;
        std::uint32_t record = pc::PrefixDiskCatalog::kNoRecord;
        pc::PrefixDiskCatalog::BlockIdentityRec identity;
        bool ok = false;
    };

    struct CompletedImage {
        pc::SnapshotRef snapshot;
        std::uint32_t record = pc::PrefixDiskCatalog::kNoRecord;
        pc::PrefixDiskCatalog::SnapIdentityRec identity;
        bool ok = false;
    };

    // A closure whose writes are all in flight: commits once every block and the image landed.
    struct PendingClosure {
        pc::SnapshotRef snapshot;
        pc::PrefixDiskCatalog::SnapIdentityRec identity;
        std::vector<std::uint32_t> records; // this closure's queued writes, awaited on commit
        // Every chain record (queued, in-flight from a sibling closure, or committed earlier):
        // the parent link for new writes and the committed identity's anchor come from here.
        std::vector<std::uint32_t> chain_records;
        // The whole anchor chain: after the commit these nodes' Host copies are duplicates of
        // the closure's records (release_redundant_host_copies).
        std::vector<pc::NodeRef> chain_nodes;
        std::uint32_t image_record = pc::PrefixDiskCatalog::kNoRecord;
        bool image_landed          = false;
        // The image write returned (landed or failed). Only then may its record be reclaimed.
        bool image_returned        = false;
        // A block write of this closure failed or was lost: no commit can follow, and the image
        // record must wait for its own write to return before it is reclaimed.
        bool failed                = false;
    };

    void worker_loop();
    void run_write_block(const BlockJob& job);
    void run_write_image(const ImageJob& job);
    void run_read(const ReadJob& job);
    void run_image_read(const ImageReadJob& job);
    void run_tail_read(const TailReadJob& job);

    // Drops `record` from the catalog, queueing it for retry when the removal fails.
    void release_record(std::uint32_t record) noexcept;
    // Recycles a full catalog: bounded disk victim eviction plus the collection of reserved
    // records no write will finish. True when anything was reclaimed.
    bool reclaim_records(pc::PrefixCacheIndex& index);
    // Retries the removals a catalog failure left owed. Engine worker, no index access.
    void retry_removals();
    // Ends a closure: unpins its snapshot and releases the chain Host copies the committed
    // records made redundant.
    void finish_closure(pc::PrefixCacheIndex& index, PendingClosure& closure, bool committed);
    // Returns a closure's still-open snapshot disk fill, which reclaims the image record it
    // reserved through the backend's release callback. A closure whose snapshot already left the
    // index owns no fill; its record goes back to the catalog directly.
    void abort_open_snapshot_fill(pc::PrefixCacheIndex& index, const PendingClosure& closure);
    // Frees one staging slot, ignoring a pointer that belongs to no block slot and one already
    // returned (a duplicate release must not hand the same slot out twice).
    void free_staging_slot(const std::byte* bytes) noexcept;
    [[nodiscard]] std::byte* block_staging(std::uint32_t slot) const noexcept;

    Options options_;
    HybridHostLayout host_;
    std::unique_ptr<pc::PrefixDiskCatalog> catalog_;

    // Staging (pinned once, accounted inside the advertised Host budget).
    std::optional<PinnedHostBuffer> block_staging_;
    std::optional<PinnedHostBuffer> image_staging_;
    std::optional<PinnedHostBuffer> tail_staging_;
    std::vector<std::uint32_t> free_staging_;
    // Whether each block staging slot is on free_staging_; a duplicate release is ignored.
    std::vector<bool> staging_slot_free_;

    // Engine-worker submission queues (consumed by the worker under mutex_).
    std::deque<BlockJob> write_blocks_;
    std::deque<ImageJob> write_images_;
    std::deque<ReadJob> read_blocks_;
    std::optional<ImageReadJob> read_image_;
    std::optional<TailReadJob> read_tail_;
    // Nodes whose read the worker popped but has not landed yet (staged_blocks_ gets them
    // only on completion): a quote re-staging in that window would read into the same slot
    // twice.
    std::vector<pc::NodeRef> reads_running_;
    bool image_read_running_ = false;
    bool tail_read_running_  = false;
    // Worker → Engine-worker completions, and closures awaiting their last record.
    std::vector<CompletedBlock> completed_blocks_;
    std::vector<CompletedImage> completed_images_;
    std::vector<PendingClosure> pending_closures_;
    std::vector<StagedBlock> staged_blocks_;
    pc::SnapshotRef staged_image_;
    pc::SnapshotRef staged_tail_;
    // Records the catalog refused to drop (I/O failure): retried at the next cadence point.
    std::vector<std::uint32_t> pending_removals_;
    // Records the index released while closures still named them: a closure that names one of
    // these can no longer commit (its anchor would dangle).
    std::vector<std::uint32_t> removed_records_;

    mutable std::mutex mutex_;
    std::condition_variable wake_;
    std::thread worker_;
    std::atomic<bool> stopping_{false};

    Stats stats_;
    std::uint64_t inqueue_bytes_ = 0;
};

} // namespace ninfer::models::qwen3_5::detail
