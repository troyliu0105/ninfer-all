#include "runtime/prefix_cache/disk_worker.h"

#include <stdexcept>
#include <utility>

namespace ninfer::runtime::prefix_cache {

PrefixDiskWorker::PrefixDiskWorker(PrefixDiskCatalog& catalog, PrefixDiskCatalog::Geometry geometry,
                                   std::uint64_t max_write_bytes, std::uint32_t max_reads)
    : catalog_(catalog), geometry_(geometry), max_write_bytes_(max_write_bytes),
      max_reads_(max_reads) {
    if (max_write_bytes == 0 || max_reads == 0) {
        throw std::invalid_argument("disk worker requires positive queue bounds");
    }
}

PrefixDiskWorker::~PrefixDiskWorker() { shutdown(); }

void PrefixDiskWorker::start() {
    std::lock_guard lock(mutex_);
    if (thread_.joinable() || stopping_) {
        throw std::logic_error("disk worker cannot be started twice");
    }
    thread_ = std::thread([this] { run(); });
}

bool PrefixDiskWorker::can_write(std::uint64_t bytes) const {
    std::lock_guard lock(mutex_);
    return !stopping_ && bytes <= max_write_bytes_ - write_bytes_;
}

bool PrefixDiskWorker::submit(BlockWrite&& job) {
    if (job.source.size() != geometry_.block_payload_bytes) {
        throw std::invalid_argument("disk block write has the wrong payload size");
    }
    std::lock_guard lock(mutex_);
    if (stopping_ || job.source.size() > max_write_bytes_ - write_bytes_) { return false; }
    writes_.emplace_back(std::move(job));
    write_bytes_ += geometry_.block_payload_bytes;
    wake_.notify_one();
    return true;
}

bool PrefixDiskWorker::submit(ImageWrite&& job) {
    if (job.segments.size() != geometry_.image_segments ||
        job.tail.size() != (job.identity.tail_len == 0 ? 0U : geometry_.block_payload_bytes)) {
        throw std::invalid_argument("disk image write has incomplete image or tail bytes");
    }
    const auto bytes = geometry_.image_segment_bytes * geometry_.image_segments + job.tail.size();
    std::lock_guard lock(mutex_);
    if (stopping_ || bytes > max_write_bytes_ - write_bytes_) { return false; }
    writes_.emplace_back(std::move(job));
    write_bytes_ += bytes;
    wake_.notify_one();
    return true;
}

bool PrefixDiskWorker::submit(const Read& job) {
    const auto bytes = job.kind == Kind::ReadImage
                           ? geometry_.image_segment_bytes * geometry_.image_segments
                           : geometry_.block_payload_bytes;
    if ((job.kind != Kind::ReadBlock && job.kind != Kind::ReadImage && job.kind != Kind::ReadTail) ||
        job.destination.size() != bytes) {
        throw std::invalid_argument("disk read has the wrong kind or destination size");
    }
    std::lock_guard lock(mutex_);
    if (stopping_ || read_count_ >= max_reads_) { return false; }
    reads_.push_back(job);
    ++read_count_;
    wake_.notify_one();
    return true;
}

void PrefixDiskWorker::remove(std::uint32_t record) {
    std::lock_guard lock(mutex_);
    if (stopping_ || !removing_.insert(record).second) { return; }
    try { removals_.push_back(record); }
    catch (...) { removing_.erase(record); throw; }
    wake_.notify_one();
}

std::vector<PrefixDiskWorker::Completion> PrefixDiskWorker::take_completions() {
    std::lock_guard lock(mutex_);
    std::vector<Completion> out;
    out.swap(completed_);
    for (const auto& done : out) {
        if (done.kind == Kind::WriteBlock || done.kind == Kind::WriteImage) {
            write_bytes_ -= done.bytes;
        } else if (done.kind == Kind::Remove) {
            removing_.erase(done.record);
        } else {
            --read_count_;
        }
    }
    return out;
}

bool PrefixDiskWorker::busy() const {
    std::lock_guard lock(mutex_);
    return running_ || !writes_.empty() || !reads_.empty() ||
           !removals_.empty() || !completed_.empty();
}

void PrefixDiskWorker::shutdown() noexcept {
    {
        std::lock_guard lock(mutex_);
        stopping_ = true;
        wake_.notify_one();
    }
    if (thread_.joinable()) { thread_.join(); }
    else { run(); }
}

PrefixDiskWorker::Completion PrefixDiskWorker::execute(const BlockWrite& job) {
    Completion out{Kind::WriteBlock, job.node, {}, job.record, 0, job.source.size(), false};
    try {
        if (!catalog_.write_block(job.record, job.source)) { return out; }
        catalog_.finish_block(job.record, job.identity);
        out.ok = true;
    } catch (...) {}
    return out;
}

PrefixDiskWorker::Completion PrefixDiskWorker::execute(const ImageWrite& job) {
    const auto bytes = geometry_.image_segment_bytes * geometry_.image_segments + job.tail.size();
    Completion out{Kind::WriteImage, {}, job.snapshot, job.record, 0, bytes, false};
    try {
        for (std::uint32_t i = 0; i < geometry_.image_segments; ++i) {
            if (!catalog_.write_image_segment(job.record, i,
                                              {job.segments[i], geometry_.image_segment_bytes})) {
                return out;
            }
        }
        if (!job.tail.empty() && !catalog_.write_tail(job.record, job.tail)) { return out; }
        catalog_.commit_snapshot(job.record, job.identity);
        out.ok = true;
    } catch (...) {}
    return out;
}

PrefixDiskWorker::Completion PrefixDiskWorker::execute(const Read& job) {
    Completion out{job.kind, job.node, job.snapshot, job.record, job.slot,
                   job.destination.size(), false};
    try {
        if (job.kind == Kind::ReadBlock) {
            catalog_.read_block(job.record, job.destination);
        } else if (job.kind == Kind::ReadTail) {
            catalog_.read_tail(job.record, job.destination);
        } else {
            for (std::uint32_t i = 0; i < geometry_.image_segments; ++i) {
                catalog_.read_image_segment(
                    job.record, i,
                    job.destination.subspan(i * geometry_.image_segment_bytes,
                                            geometry_.image_segment_bytes));
            }
        }
        out.ok = true;
    } catch (...) {
        // Report failure before any cleanup I/O. The Engine invalidates this record generation
        // and queues removal separately, so a blocked journal cannot delay root fallback.
    }
    return out;
}

PrefixDiskWorker::Completion PrefixDiskWorker::execute_removal(std::uint32_t record) {
    Completion out{Kind::Remove, {}, {}, record, 0, 0, false};
    try { catalog_.remove(record); out.ok = true; }
    catch (...) {}
    return out;
}

void PrefixDiskWorker::run() {
    for (;;) {
        std::unique_lock lock(mutex_);
        wake_.wait(lock, [&] {
            return stopping_ || !reads_.empty() || !removals_.empty() || !writes_.empty();
        });
        if (reads_.empty() && removals_.empty() && writes_.empty()) {
            if (stopping_) { return; }
            continue;
        }
        running_ = true;
        Completion done{Kind::Remove, {}, {}, kNoId, 0, 0, false};
        // A demanded read overtakes all unstarted background jobs. Never hold this queue lock
        // across catalog, payload, journal or checkpoint I/O.
        if (!reads_.empty()) {
            const Read job = reads_.front();
            reads_.pop_front();
            lock.unlock();
            done = execute(job);
            lock.lock();
        } else if (!removals_.empty()) {
            const auto record = removals_.front();
            removals_.pop_front();
            lock.unlock();
            done = execute_removal(record);
            lock.lock();
        } else {
            // Preserve write submission order: a continuing block stream must not starve an
            // older image commit and prevent it from releasing its leased Host slabs.
            auto job = std::move(writes_.front());
            writes_.pop_front();
            lock.unlock();
            done = std::visit([&](const auto& write) { return execute(write); }, job);
            lock.lock();
        }
        completed_.push_back(done);
        running_ = false;
    }
}

} // namespace ninfer::runtime::prefix_cache
