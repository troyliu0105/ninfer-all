// The hybrid prefix cache's disk tier binding — see hybrid_disk.h and the A4 design note in
// local://hybrid-disk-tier-plan.md.
//
// The two threads: the Engine worker picks victims, snapshots their metadata and Host slab
// pointers into immutable jobs, and pins the sources; the I/O worker only moves bytes between
// those pinned slabs and the catalog. Completions return to the Engine worker, which alone
// runs finish_block/commit_snapshot and mutates the index. A closure commits only once every
// one of its block records and its image record landed.

#include "models/qwen3_5/program/prefix/hybrid_disk.h"

#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace ninfer::models::qwen3_5::detail {

namespace {

using pc::CopyState;
using pc::NodeRef;
using pc::NodeView;
using pc::PrefixCacheIndex;
using pc::PrefixDiskCatalog;
using pc::SnapshotRef;
using pc::SnapshotView;

} // namespace

HybridDiskTier::HybridDiskTier(Options options, const HybridHostLayout& host,
                               const std::string& fingerprint)
    : options_(std::move(options)), host_(host) {
    if (options_.directory.empty()) {
        throw std::invalid_argument("hybrid disk tier directory is empty");
    }
    if (options_.staging_blocks == 0 || host_.block_payload_bytes == 0 ||
        host_.image_bytes == 0 || host_.image_slabs == 0 || host_.slab_bytes == 0) {
        throw std::invalid_argument("hybrid disk tier requires a positive geometry");
    }
    // One image segment per slab: a segment write is exactly the slab payload the pin holds.
    PrefixDiskCatalog::Geometry geometry;
    geometry.block_payload_bytes = host_.block_payload_bytes;
    geometry.image_segment_bytes = host_.slab_bytes;
    geometry.image_segments      = host_.image_slabs;
    PrefixDiskCatalog::Options catalog_options;
    catalog_options.directory            = options_.directory;
    catalog_options.block_capacity_bytes = options_.block_capacity_bytes;
    catalog_options.image_capacity_bytes = options_.image_capacity_bytes;
    catalog_ = std::make_unique<PrefixDiskCatalog>(std::move(catalog_options), geometry,
                                                   fingerprint);
    block_staging_ = std::make_unique<std::byte[]>(
        static_cast<std::size_t>(options_.staging_blocks) * host_.block_payload_bytes);
    // The disk stores whole slabs (image_slabs of slab_bytes); only the first image_bytes are
    // the StateImage. The staging buffer must hold every physical segment the reads write.
    image_staging_ = std::make_unique<std::byte[]>(
        static_cast<std::size_t>(host_.image_slabs) * host_.slab_bytes);
    tail_staging_  = std::make_unique<std::byte[]>(host_.block_payload_bytes);
    free_staging_.resize(options_.staging_blocks);
    for (std::uint32_t i = 0; i < options_.staging_blocks; ++i) {
        free_staging_[i] = options_.staging_blocks - 1U - i;
    }
    worker_ = std::thread([this] { worker_loop(); });
}

HybridDiskTier::~HybridDiskTier() { shutdown(); }

bool HybridDiskTier::restore_into(PrefixCacheIndex& index) {
    if (!catalog_->open()) {
        catalog_->wipe();
        return false;
    }
    // Replayed ids are parents-first by construction: a record's finish_block preceded any
    // dependent commit, and the catalog replays in id order.
    std::unordered_map<std::uint32_t, NodeRef> restored;
    for (const PrefixDiskCatalog::ReplayedBlock& block : catalog_->replayed_blocks()) {
        NodeRef parent;
        if (block.identity.parent != PrefixDiskCatalog::kNoRecord) {
            const auto found = restored.find(block.identity.parent);
            if (found == restored.end()) { continue; } // torn dependency: skip this subtree
            parent = found->second;
        }
        const auto node = index.restore_disk_block(
            parent, block.identity.lookup_hash,
            std::span<const TokenId>(block.identity.tokens.data(),
                                     block.identity.tokens.size()),
            block.identity.extra, block.id);
        if (!node) { break; } // node capacity: deeper entries cannot restore either
        restored.emplace(block.id, *node);
    }
    for (const PrefixDiskCatalog::ReplayedSnapshot& snapshot : catalog_->replayed_snapshots()) {
        NodeRef anchor;
        if (snapshot.identity.anchor != PrefixDiskCatalog::kNoRecord) {
            const auto found = restored.find(snapshot.identity.anchor);
            if (found == restored.end()) {
                continue;
            }
            anchor = found->second;
        }
        (void)index.restore_disk_snapshot(
            anchor, snapshot.identity.frontier,
            std::span<const TokenId>(snapshot.identity.tail.data(), snapshot.identity.tail_len),
            snapshot.identity.kind, snapshot.identity.hits, snapshot.id, true);
    }
    return true;
}

// ---- demotion (Engine worker) ----------------------------------------------------------------------

void HybridDiskTier::maybe_demote(PrefixCacheIndex& index, const HybridPrefixCache& cache) {
    if (!cache.host_tier()) { return; }
    const std::uint32_t slabs = index.config().host_slabs;
    if (slabs == 0) { return; }
    const pc::PrefixIndexStats stats = index.stats();
    const std::uint32_t occupied     = slabs - stats.host_free_slabs;
    if (static_cast<double>(occupied) <
        static_cast<double>(slabs) * options_.host_high_watermark) {
        return; // below the high watermark
    }
    const auto candidates = index.collect_demotion_candidates(options_.demotion_batch_records);
    for (const SnapshotRef snapshot : candidates) {
        if (inqueue_bytes_ >= options_.max_inqueue_bytes) {
            ++stats_.queue_rejects;
            return; // backpressure: stop submitting, never block
        }
        const SnapshotView view = index.snapshot(snapshot);
        if (view.host != CopyState::Resident) { continue; }

        PendingClosure closure;
        closure.snapshot = snapshot;
        closure.identity = PrefixDiskCatalog::SnapIdentityRec{
            .anchor   = PrefixDiskCatalog::kNoRecord,
            .frontier = view.frontier,
            .tail_len = view.tail_len,
            .tail     = {},
            .kind     = view.kind,
            .hits     = view.hits};
        std::copy(view.tail.begin(), view.tail.end(), closure.identity.tail.begin());

        // The anchor chain root→anchor; each block without a disk record becomes a write job.
        std::vector<NodeRef> chain;
        for (NodeRef walk = view.anchor; walk.valid(); walk = index.node(walk).parent) {
            chain.push_back(walk);
        }
        std::reverse(chain.begin(), chain.end());
        bool abort_closure = false;
        index.pin_snapshot(snapshot);
        for (const NodeRef node : chain) {
            const NodeView node_view = index.node(node);
            if (node_view.disk == CopyState::Resident || node_view.disk == CopyState::Filling) {
                // Already disk-backed (or a sibling closure's write is in flight): the anchor
                // chain keeps those records without recopying. An in-flight write is a real
                // dependency — it stays in the wait set so a sibling failure fails this
                // closure too; a committed record never completes again and must not.
                if (node_view.disk_ref != pc::kNoId) {
                    closure.chain_records.push_back(node_view.disk_ref);
                    if (node_view.disk == CopyState::Filling) {
                        closure.records.push_back(node_view.disk_ref);
                    }
                }
                continue;
            }
            if (node_view.host != CopyState::Resident) {
                abort_closure = true; // no Host source to read
                break;
            }
            const pc::BlockIdentity identity = index.block_identity(node);
            const std::uint32_t parent_record =
                closure.chain_records.empty() ? PrefixDiskCatalog::kNoRecord
                                              : closure.chain_records.back();
            if (const auto existing = catalog_->find_block(parent_record, identity.lookup_hash,
                                                           identity.tokens, identity.extra)) {
                closure.chain_records.push_back(*existing);
                continue; // a committed ancestor is reused, not recopied
            }
            const std::uint32_t record = catalog_->begin_block_write();
            if (record == PrefixDiskCatalog::kNoRecord) {
                abort_closure = true; // disk full: stop demoting this cycle
                break;
            }
            index.pin_node(node);
            index.begin_node_disk_fill(node, record);
            BlockJob job;
            job.node     = node;
            job.record   = record;
            job.identity = PrefixDiskCatalog::BlockIdentityRec{
                .parent      = parent_record,
                .lookup_hash = identity.lookup_hash,
                .extra       = identity.extra,
                .tokens      = {}};
            std::copy(identity.tokens.begin(), identity.tokens.end(),
                      job.identity.tokens.begin());
            job.source = cache.slab_bytes_of(node_view.host_slab);
            {
                std::lock_guard lock(mutex_);
                write_blocks_.push_back(job);
                inqueue_bytes_ += host_.block_payload_bytes;
                wake_.notify_one();
            }
            closure.records.push_back(record);
            closure.chain_records.push_back(record);
        }
        if (abort_closure) {
            // Queued write jobs keep their pins: publish_completions unpins each node as its
            // write lands, so unpinning here would release a pin the job still owns.
            index.unpin_snapshot(snapshot);
            continue;
        }

        // The image (and tail) job; the snapshot stays pinned until the closure commits.
        const std::uint32_t record = catalog_->begin_snapshot_write();
        if (record == PrefixDiskCatalog::kNoRecord) {
            // Same ownership as the abort path: the queued writes release their own pins.
            index.unpin_snapshot(snapshot);
            ++stats_.queue_rejects;
            continue;
        }
        index.begin_snapshot_disk_fill(snapshot, record);
        ImageJob job;
        job.snapshot = snapshot;
        job.record   = record;
        for (const std::uint32_t slab : view.host_slabs) {
            job.segments.push_back(cache.slab_bytes_of(slab));
        }
        // The tail slab is the snapshot's last host slab (blocks and images share the pool and
        // the tail occupies the slab after the image run).
        if (view.tail_len != 0 && view.host_slabs.size() > host_.image_slabs) {
            job.tail = cache.slab_bytes_of(view.host_slabs.back());
        }
        job.identity = closure.identity;
        // The committed identity names the chain's last record: the deepest block this
        // closure named, whether it wrote the record or a sibling's write lands it.
        job.identity.anchor = closure.chain_records.empty()
                                  ? PrefixDiskCatalog::kNoRecord
                                  : closure.chain_records.back();
        closure.identity.anchor = job.identity.anchor;
        closure.image_record = record;
        {
            std::lock_guard lock(mutex_);
            write_images_.push_back(std::move(job));
            inqueue_bytes_ += host_.image_bytes;
            wake_.notify_one();
        }
        pending_closures_.push_back(std::move(closure));
    }
}

void HybridDiskTier::publish_completions(PrefixCacheIndex& index) {
    std::vector<CompletedBlock> blocks;
    std::vector<CompletedImage> images;
    {
        std::lock_guard lock(mutex_);
        blocks = std::move(completed_blocks_);
        images = std::move(completed_images_);
        completed_blocks_.clear();
        completed_images_.clear();
    }
    // Block records: finish, mark the node disk-backed, release the source pin.
    for (const CompletedBlock& done : blocks) {
        if (!done.ok || !index.valid(done.node)) {
            if (done.record != PrefixDiskCatalog::kNoRecord) { catalog_->abandon(done.record); }
            if (index.valid(done.node)) { index.unpin_node(done.node); }
            if (index.valid(done.node)) { index.abort_node_disk_fill(done.node); }
            // The closure that waited on this record can never commit; drop it and reset its
            // snapshot so a later cycle retries the demotion from scratch.
            std::erase_if(pending_closures_, [&](PendingClosure& closure) {
                if (std::find(closure.records.begin(), closure.records.end(), done.record) ==
                    closure.records.end()) {
                    return false;
                }
                index.abort_snapshot_disk_fill(closure.snapshot);
                index.unpin_snapshot(closure.snapshot);
                return true;
            });
            continue;
        }
        catalog_->finish_block(done.record, done.identity);
        index.mark_disk_backed(done.node, done.record);
        ++stats_.demoted_blocks;
        stats_.disk_write_bytes += host_.block_payload_bytes;
        index.unpin_node(done.node);
    }
    // Image records: their payload landed; the closure commits when its blocks all did.
    std::vector<std::uint32_t> landed_images;
    for (const CompletedImage& done : images) {
        if (!done.ok || !index.valid(done.snapshot)) {
            if (done.record != PrefixDiskCatalog::kNoRecord) { catalog_->abandon(done.record); }
            if (index.valid(done.snapshot)) { index.unpin_snapshot(done.snapshot); }
            if (index.valid(done.snapshot)) { index.abort_snapshot_disk_fill(done.snapshot); }
            std::erase_if(pending_closures_, [&](const PendingClosure& closure) {
                return closure.image_record == done.record;
            });
            continue;
        }
        stats_.disk_write_bytes += host_.image_bytes;
        landed_images.push_back(done.record);
    }
    for (PendingClosure& closure : pending_closures_) {
        if (std::find(landed_images.begin(), landed_images.end(), closure.image_record) !=
            landed_images.end()) {
            closure.image_landed = true;
        }
        std::erase_if(closure.records, [&](std::uint32_t record) {
            return std::any_of(blocks.begin(), blocks.end(), [&](const CompletedBlock& done) {
                return done.ok && done.record == record;
            });
        });
    }
    // Commit every closure whose wait list emptied and whose image landed. The records vector
    // held the awaited block writes; committed ancestors were never queued, so an empty list
    // with the image landed means the whole closure is durable.
    for (std::size_t i = 0; i < pending_closures_.size();) {
        PendingClosure& closure = pending_closures_[i];
        if (!closure.records.empty() || !closure.image_landed) {
            ++i;
            continue;
        }
        const SnapshotRef snapshot = closure.snapshot;
        try {
            catalog_->commit_snapshot(closure.image_record, closure.identity);
            index.mark_snapshot_disk_backed(snapshot, closure.image_record);
            // The committed closure duplicates the Host copy; releasing it is the tier's
            // whole point, and it happens now rather than under the next eviction pressure.
            index.release_snapshot_host_copy(snapshot);
            ++stats_.demoted_images;
            ++stats_.committed_closures;
        } catch (...) {
            catalog_->abandon(closure.image_record);
        }
        index.unpin_snapshot(snapshot);
        pending_closures_.erase(pending_closures_.begin() + static_cast<std::ptrdiff_t>(i));
    }
}

std::uint64_t HybridDiskTier::staging_bytes() const noexcept {
    return static_cast<std::uint64_t>(options_.staging_blocks) * host_.block_payload_bytes +
           host_.image_bytes + host_.block_payload_bytes;
}

bool HybridDiskTier::busy() const noexcept {
    std::lock_guard lock(mutex_);
    return !write_blocks_.empty() || !write_images_.empty() || !read_blocks_.empty() ||
           read_image_.has_value() || read_tail_.has_value() || !completed_blocks_.empty() ||
           !completed_images_.empty();
}

HybridDiskTier::Stats HybridDiskTier::stats() const noexcept {
    Stats out       = stats_;
    const auto catalog = catalog_->stats();
    out.catalog_blocks    = catalog.blocks;
    out.catalog_committed = catalog.committed;
    return out;
}

void HybridDiskTier::shutdown() {
    if (stopping_.exchange(true, std::memory_order_acq_rel)) { return; }
    {
        std::lock_guard lock(mutex_);
        wake_.notify_all();
    }
    if (worker_.joinable()) { worker_.join(); }
    catalog_->checkpoint();
}

void HybridDiskTier::reset() {
    shutdown();
    catalog_->wipe();
    stats_          = Stats{};
    inqueue_bytes_  = 0;
    pending_closures_.clear();
    staged_blocks_.clear();
    write_blocks_.clear();
    write_images_.clear();
    read_blocks_.clear();
    read_image_.reset();
    staged_image_   = SnapshotRef{};
    staged_tail_    = SnapshotRef{};
    read_tail_.reset();
    free_staging_.clear();
    free_staging_.resize(options_.staging_blocks);
    for (std::uint32_t i = 0; i < options_.staging_blocks; ++i) {
        free_staging_[i] = options_.staging_blocks - 1U - i;
    }
    stopping_.store(false, std::memory_order_release);
    worker_ = std::thread([this] { worker_loop(); });
}

// ---- cold preparation (Engine worker) ----------------------------------------------------------------

bool HybridDiskTier::stage_blocks(const std::vector<NodeRef>& nodes,
                                  const PrefixCacheIndex& index) {
    std::lock_guard lock(mutex_);
    if (free_staging_.size() < nodes.size()) { return false; }
    std::vector<ReadJob> jobs;
    jobs.reserve(nodes.size());
    for (const NodeRef node : nodes) {
        if (!index.valid(node)) { continue; }
        const auto queued = std::find_if(read_blocks_.begin(), read_blocks_.end(),
                                         [&](const ReadJob& job) { return job.node == node; });
        if (queued != read_blocks_.end()) { continue; } // already reading
        if (std::find(reads_running_.begin(), reads_running_.end(), node) !=
            reads_running_.end()) {
            continue; // the worker popped it; landing is imminent
        }
        if (std::find_if(staged_blocks_.begin(), staged_blocks_.end(),
                         [&](const StagedBlock& staged) { return staged.node == node; }) !=
            staged_blocks_.end()) {
            continue; // already landed
        }
        const NodeView view = index.node(node);
        if (view.disk != CopyState::Resident || view.device == CopyState::Resident ||
            view.host == CopyState::Resident) {
            continue; // only disk-only blocks stage
        }
        ReadJob job;
        job.node     = node;
        job.disk_ref = view.disk_ref;
        job.slot     = free_staging_.back();
        free_staging_.pop_back();
        jobs.push_back(job);
    }
    if (jobs.empty()) { return false; }
    for (const ReadJob& job : jobs) { read_blocks_.push_back(job); }
    wake_.notify_one();
    return true;
}

std::vector<HybridDiskTier::StagedBlock> HybridDiskTier::take_staged_blocks() {
    std::lock_guard lock(mutex_);
    return std::move(staged_blocks_);
}

void HybridDiskTier::release_staged(const std::vector<StagedBlock>& blocks) {
    std::lock_guard lock(mutex_);
    for (const StagedBlock& block : blocks) {
        const std::uint32_t slot = static_cast<std::uint32_t>(
            (block.bytes - block_staging_.get()) / host_.block_payload_bytes);
        free_staging_.push_back(slot);
    }
}

bool HybridDiskTier::stage_image(SnapshotRef snapshot, const PrefixCacheIndex& index) {
    std::lock_guard lock(mutex_);
    if (read_image_.has_value() || image_read_running_ || staged_image_.valid()) {
        return false;
    }
    if (!index.valid(snapshot)) { return false; }
    const SnapshotView view = index.snapshot(snapshot);
    if (view.disk != CopyState::Resident) { return false; }
    read_image_ = ImageReadJob{snapshot, view.disk_ref};
    wake_.notify_one();
    return true;
}

const std::byte* HybridDiskTier::take_staged_image(SnapshotRef snapshot) {
    std::lock_guard lock(mutex_);
    return staged_image_ == snapshot ? image_staging_.get() : nullptr;
}

void HybridDiskTier::release_staged_image_except(SnapshotRef snapshot) {
    std::lock_guard lock(mutex_);
    if (staged_image_.valid() && !(staged_image_ == snapshot)) { staged_image_ = SnapshotRef{}; }
}

void HybridDiskTier::release_image() {
    std::lock_guard lock(mutex_);
    staged_image_ = SnapshotRef{};
}

const std::byte* HybridDiskTier::staged_tail(SnapshotRef snapshot) const noexcept {
    std::lock_guard lock(mutex_);
    return staged_tail_ == snapshot ? tail_staging_.get() : nullptr;
}

void HybridDiskTier::release_staged_tail_except(SnapshotRef snapshot) {
    std::lock_guard lock(mutex_);
    if (staged_tail_.valid() && !(staged_tail_ == snapshot)) { staged_tail_ = SnapshotRef{}; }
}

void HybridDiskTier::release_tail() {
    std::lock_guard lock(mutex_);
    staged_tail_ = SnapshotRef{};
}

bool HybridDiskTier::stage_tail(SnapshotRef snapshot, const PrefixCacheIndex& index) {
    std::lock_guard lock(mutex_);
    if (read_tail_.has_value() || tail_read_running_ || staged_tail_.valid()) { return false; }
    if (!index.valid(snapshot)) { return false; }
    const SnapshotView view = index.snapshot(snapshot);
    if (view.tail_len == 0 || view.disk != CopyState::Resident ||
        view.host == CopyState::Resident) {
        return false;
    }
    read_tail_ = TailReadJob{snapshot, view.disk_ref};
    wake_.notify_one();
    return true;
}

// ---- I/O worker ----------------------------------------------------------------------------------------

void HybridDiskTier::worker_loop() {
    for (;;) {
        std::unique_lock lock(mutex_);
        wake_.wait(lock, [&] {
            return stopping_.load(std::memory_order_acquire) || !write_blocks_.empty() ||
                   !write_images_.empty() || !read_blocks_.empty() || read_image_.has_value() ||
                   read_tail_.has_value();
        });
        if (write_blocks_.empty() && write_images_.empty() && read_blocks_.empty() &&
            !read_image_.has_value() && !read_tail_.has_value()) {
            if (stopping_.load(std::memory_order_acquire)) { return; }
            continue;
        }
        if (!write_blocks_.empty()) {
            const BlockJob job = write_blocks_.front();
            write_blocks_.pop_front();
            lock.unlock();
            run_write_block(job);
            lock.lock();
        } else if (!write_images_.empty()) {
            const ImageJob job = std::move(write_images_.front());
            write_images_.pop_front();
            lock.unlock();
            run_write_image(job);
            lock.lock();
        } else if (!read_blocks_.empty()) {
            const ReadJob job = read_blocks_.front();
            read_blocks_.pop_front();
            reads_running_.push_back(job.node);
            lock.unlock();
            run_read(job);
            lock.lock();
            std::erase(reads_running_, job.node);
        } else if (read_image_.has_value()) {
            const ImageReadJob job = *read_image_;
            read_image_.reset();
            image_read_running_ = true;
            lock.unlock();
            run_image_read(job);
            lock.lock();
            image_read_running_ = false;
        } else {
            const TailReadJob job = *read_tail_;
            read_tail_.reset();
            tail_read_running_ = true;
            lock.unlock();
            run_tail_read(job);
            lock.lock();
            tail_read_running_ = false;
        }
    }
}

void HybridDiskTier::run_write_block(const BlockJob& job) {
    bool ok = false;
    try {
        ok = catalog_->write_block(
            job.record, std::span<const std::byte>(job.source, host_.block_payload_bytes));
    } catch (...) { ok = false; }
    std::lock_guard lock(mutex_);
    completed_blocks_.push_back(
        CompletedBlock{job.node, job.record, job.identity, ok});
    inqueue_bytes_ = inqueue_bytes_ >= host_.block_payload_bytes
                         ? inqueue_bytes_ - host_.block_payload_bytes
                         : 0U;
}

void HybridDiskTier::run_write_image(const ImageJob& job) {
    bool ok = true;
    try {
        if (job.segments.size() < host_.image_slabs) {
            // The snapshot's Host copy lost slabs between submission and the write (a
            // concurrent reclaim under pressure cannot happen — the pin holds — but a
            // partially-filled snapshot reaches here when its fill aborted).
            ok = false;
        }
        for (std::uint32_t segment = 0; ok && segment != host_.image_slabs; ++segment) {
            ok = catalog_->write_image_segment(
                job.record, segment,
                std::span<const std::byte>(job.segments[segment], host_.slab_bytes));
        }
        if (ok && job.tail != nullptr) {
            ok = catalog_->write_tail(
                job.record, std::span<const std::byte>(job.tail, host_.block_payload_bytes));
        }
    } catch (...) { ok = false; }
    std::lock_guard lock(mutex_);
    completed_images_.push_back(CompletedImage{job.snapshot, job.record, job.identity, ok});
    inqueue_bytes_ = inqueue_bytes_ >= host_.image_bytes ? inqueue_bytes_ - host_.image_bytes
                                                         : 0U;
}

void HybridDiskTier::run_read(const ReadJob& job) {
    std::byte* destination =
        block_staging_.get() + static_cast<std::size_t>(job.slot) * host_.block_payload_bytes;
    bool ok = false;
    try {
        catalog_->read_block(job.disk_ref,
                             std::span<std::byte>(destination, host_.block_payload_bytes));
        ok = true;
    } catch (...) { ok = false; }
    std::lock_guard lock(mutex_);
    if (ok) {
        staged_blocks_.push_back(StagedBlock{job.node, destination});
        stats_.disk_read_bytes += host_.block_payload_bytes;
        ++stats_.restored_blocks;
    } else {
        free_staging_.push_back(job.slot);
    }
}

void HybridDiskTier::run_image_read(const ImageReadJob& job) {
    bool ok = true;
    for (std::uint32_t segment = 0; segment != host_.image_slabs && ok; ++segment) {
        try {
            catalog_->read_image_segment(
                job.disk_ref, segment,
                std::span<std::byte>(image_staging_.get() +
                                         static_cast<std::size_t>(segment) * host_.slab_bytes,
                                     host_.slab_bytes));
        } catch (...) { ok = false; }
    }
    std::lock_guard lock(mutex_);
    if (ok) {
        staged_image_ = job.snapshot;
        stats_.disk_read_bytes += host_.image_bytes;
    }
}

void HybridDiskTier::run_tail_read(const TailReadJob& job) {
    bool ok = false;
    try {
        catalog_->read_tail(job.disk_ref,
                            std::span<std::byte>(tail_staging_.get(),
                                                 host_.block_payload_bytes));
        ok = true;
    } catch (...) { ok = false; }
    std::lock_guard lock(mutex_);
    if (ok) {
        staged_tail_ = job.snapshot;
        stats_.disk_read_bytes += host_.block_payload_bytes;
    }
}

} // namespace ninfer::models::qwen3_5::detail
