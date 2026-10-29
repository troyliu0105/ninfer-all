#include "runtime/prefix_cache/disk_worker.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <filesystem>
#include <future>
#include <iostream>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

#ifdef __linux__
#include <fcntl.h>
#include <poll.h>
#include <sys/inotify.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace {
using namespace ninfer::runtime::prefix_cache;
constexpr std::uint64_t kBlock = 4096;
constexpr std::uint64_t kSegment = 8192;
constexpr std::uint32_t kSegments = 2;
const PrefixDiskCatalog::Geometry kGeometry{kBlock, kSegment, kSegments};
using Kind = PrefixDiskWorker::Kind;

void check(bool value, const char* message) {
    if (!value) { throw std::runtime_error(message); }
}
struct Scratch {
    std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("ninfer_disk_worker_" + std::to_string(std::random_device{}()));
    Scratch() { std::filesystem::create_directories(path); }
    ~Scratch() { std::error_code error; std::filesystem::remove_all(path, error); }
};
PrefixDiskCatalog::Options options(const std::filesystem::path& path) {
    return {.directory = path, .block_capacity_bytes = 16 * kBlock,
            .image_capacity_bytes = 16 * kSegment, .journal_flush_records = 256};
}
std::vector<std::byte> payload(std::uint64_t bytes, unsigned seed) {
    std::vector<std::byte> out(bytes);
    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] = static_cast<std::byte>((i * 197 + seed) & 255);
    }
    return out;
}
PrefixDiskCatalog::BlockIdentityRec identity(std::uint32_t parent, unsigned seed) {
    PrefixDiskCatalog::BlockIdentityRec out;
    out.parent = parent;
    out.lookup_hash = seed;
    out.tokens.fill(static_cast<ninfer::TokenId>(seed));
    return out;
}
std::uint32_t put(PrefixDiskCatalog& catalog, unsigned seed) {
    const auto ref = catalog.begin_block_write();
    check(ref != kNoId, "block reservation");
    check(catalog.write_block(ref, payload(kBlock, seed)), "block bytes");
    catalog.finish_block(ref, identity(kNoId, seed));
    return ref;
}
std::vector<PrefixDiskWorker::Completion> await(PrefixDiskWorker& worker, std::size_t count) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    std::vector<PrefixDiskWorker::Completion> out;
    while (out.size() < count && std::chrono::steady_clock::now() < deadline) {
        auto batch = worker.take_completions();
        out.insert(out.end(), batch.begin(), batch.end());
        if (out.size() < count) { std::this_thread::sleep_for(std::chrono::milliseconds(1)); }
    }
    check(out.size() == count, "accepted disk jobs must each complete exactly once");
    return out;
}

void bounds_and_read_priority(const std::filesystem::path& path) {
    PrefixDiskCatalog catalog(options(path), kGeometry, "worker-bounds");
    check(catalog.open(), "open");
    const auto existing = put(catalog, 7);
    auto a = payload(kBlock, 11);
    auto b = payload(kBlock, 12);
    std::vector<std::byte> read(kBlock);
    PrefixDiskWorker worker(catalog, kGeometry, 2 * kBlock, 1);
    const auto first = catalog.begin_block_write();
    const auto second = catalog.begin_block_write();
    check(worker.submit(PrefixDiskWorker::BlockWrite{{}, first, identity(kNoId, 11), a}),
          "first accepted write");
    check(worker.submit(PrefixDiskWorker::BlockWrite{{}, second, identity(first, 12), b}),
          "second accepted write");
    check(!worker.can_write(1), "the byte bound includes every accepted write");
    check(!worker.submit(PrefixDiskWorker::BlockWrite{{}, first, identity(kNoId, 11), a}),
          "a whole next payload must not overshoot the bound");
    check(worker.submit(PrefixDiskWorker::Read{Kind::ReadBlock, {}, {}, existing, 0, read}),
          "demanded read accepted beside a full write queue");
    check(!worker.submit(PrefixDiskWorker::Read{Kind::ReadBlock, {}, {}, existing, 0, read}),
          "read destinations stay bounded until their completion is adopted");
    worker.remove(existing);
    worker.start();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (catalog.stats().abandoned != 0 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    check(catalog.stats().abandoned == 0, "both accepted payloads must finish publication");
    check(!worker.can_write(1),
          "completed but unadopted writes still lease their full source-byte budget");
    const auto done = await(worker, 4);
    check(done[0].kind == Kind::ReadBlock && done[0].ok,
          "a demanded read must precede all unstarted background writes and removals");
    check(read == payload(kBlock, 7), "the read sees its intact leased record");
    check(std::all_of(done.begin(), done.end(), [](const auto& item) { return item.ok; }),
          "queue pressure cannot reclaim either accepted write reservation");
    std::vector<std::byte> actual(kBlock);
    catalog.read_block(second, actual);
    check(actual == b, "a queued descendant preserves its exact payload");
    check(worker.can_write(2 * kBlock), "adopting completions returns the entire write budget");
}

void complete_image_and_restart(const std::filesystem::path& path) {
    auto image0 = payload(kSegment, 21);
    auto image1 = payload(kSegment, 22);
    auto tail = payload(kBlock, 23);
    std::uint32_t snap = kNoId;
    {
        PrefixDiskCatalog catalog(options(path), kGeometry, "worker-image");
        check(catalog.open(), "open image store");
        const auto anchor = put(catalog, 20);
        snap = catalog.begin_snapshot_write(true);
        PrefixDiskWorker worker(catalog, kGeometry, 2 * kSegment + kBlock, 1);
        PrefixDiskWorker::ImageWrite job;
        job.record = snap;
        job.identity.anchor = anchor;
        job.identity.frontier = kBlockTokens + 3;
        job.identity.tail_len = 3;
        job.identity.tail[0] = 31;
        job.identity.tail[1] = 32;
        job.identity.tail[2] = 33;
        job.segments = {image0.data(), image1.data()};
        job.tail = tail;
        check(worker.submit(std::move(job)), "exact image-and-tail byte bound accepts the closure");
        check(!worker.can_write(1), "physical image segments and tail are fully charged");
        worker.start();
        const auto done = await(worker, 1);
        check(done[0].ok && done[0].kind == Kind::WriteImage,
              "completion publishes the committed whole closure, not just image bytes");
        worker.shutdown();
        catalog.checkpoint();
    }
    PrefixDiskCatalog reopened(options(path), kGeometry, "worker-image");
    check(reopened.open(), "reopen image store");
    check(reopened.replayed_snapshots().size() == 1 && reopened.replayed_snapshots()[0].id == snap,
          "the committed snapshot survives restart");
    std::vector<std::byte> actual(kSegment);
    reopened.read_image_segment(snap, 0, actual);
    check(actual == image0, "first StateImage segment survives exactly");
    reopened.read_image_segment(snap, 1, actual);
    check(actual == image1, "last StateImage segment survives exactly");
    actual.resize(kBlock);
    reopened.read_tail(snap, actual);
    check(actual == tail, "partial-block tail survives exactly");
}

void older_image_precedes_younger_blocks(const std::filesystem::path& path) {
    PrefixDiskCatalog catalog(options(path), kGeometry, "worker-write-fairness");
    check(catalog.open(), "open fairness store");
    const auto anchor = put(catalog, 30);
    const auto snapshot = catalog.begin_snapshot_write(false);
    auto image = payload(kSegment, 31);
    auto block = payload(kBlock, 32);
    PrefixDiskWorker worker(catalog, kGeometry, 2 * kSegment + kBlock, 1);
    PrefixDiskWorker::ImageWrite job;
    job.record = snapshot;
    job.identity.anchor = anchor;
    job.identity.frontier = kBlockTokens;
    job.segments = {image.data(), image.data()};
    check(worker.submit(std::move(job)), "older complete image accepted");
    const auto younger = catalog.begin_block_write();
    check(worker.submit(PrefixDiskWorker::BlockWrite{{}, younger, identity(anchor, 32), block}),
          "a later block stream fits beside the accepted image");
    worker.start();
    const auto done = await(worker, 2);
    check(done[0].kind == Kind::WriteImage && done[0].ok && done[1].ok,
          "new block streams must not postpone an already accepted closure and its Host release");
}

void failed_read_and_broken_closure(const std::filesystem::path& path) {
    PrefixDiskCatalog catalog(options(path), kGeometry, "worker-failure");
    check(catalog.open(), "open failure store");
    const auto parent = put(catalog, 40);
    const auto child = catalog.begin_block_write();
    check(catalog.write_block(child, payload(kBlock, 41)), "child payload");
    catalog.finish_block(child, identity(parent, 41));
    const auto snap = catalog.begin_snapshot_write(false);
    auto segment = payload(kSegment, 42);
    std::vector<std::byte> read(kBlock);
    PrefixDiskWorker worker(catalog, kGeometry, 2 * kSegment, 1);
    PrefixDiskWorker::ImageWrite job;
    job.record = snap;
    job.identity.anchor = child;
    job.identity.frontier = 2 * kBlockTokens;
    job.segments = {segment.data(), segment.data()};
    check(worker.submit(std::move(job)), "image job accepted");
    worker.remove(parent);
    check(worker.submit(PrefixDiskWorker::Read{Kind::ReadBlock, {}, {}, kNoId - 1, 0, read}),
          "missing-record read accepted");
    worker.start();
    const auto done = await(worker, 3);
    check(done[0].kind == Kind::ReadBlock && !done[0].ok,
          "a read error surfaces as a completion instead of a silent retry");
    check(done[2].kind == Kind::WriteImage && !done[2].ok,
          "a closure with a removed ancestor must never commit");
    worker.remove(snap);
    check(await(worker, 1)[0].ok, "failed image reservation is explicitly reclaimable");
    check(catalog.stats().committed == 0, "no restartable closure advertises a broken chain");
}

#ifdef __linux__
void metadata_does_not_wait_for_io(const std::filesystem::path& path, bool failed_read = false) {
    PrefixDiskCatalog catalog(options(path), kGeometry, "worker-no-engine-io");
    check(catalog.open(), "open nonblocking store");
    const auto ref = failed_read ? put(catalog, 50) : catalog.begin_block_write();
    catalog.checkpoint();
    const auto journal = path / "catalog.journal";
    if (failed_read) {
        check(std::filesystem::create_directory(journal), "inject a first removal failure");
        bool rejected = false;
        try { catalog.remove(ref); } catch (const std::exception&) { rejected = true; }
        std::filesystem::remove(journal);
        check(rejected, "a failed removal retains the now-unavailable source record");
    }
    check(::mkfifo(journal.c_str(), 0600) == 0, "create real blocked journal");
    const int reader = ::open(journal.c_str(), O_RDONLY | O_NONBLOCK);
    check(reader != -1, "open FIFO reader");
    const int writer = ::open(journal.c_str(), O_WRONLY | O_NONBLOCK);
    check(writer != -1, "open FIFO filler");
    std::array<char, 4096> buffer{};
    while (::write(writer, buffer.data(), buffer.size()) > 0) {}
    check(errno == EAGAIN, "fill the real journal pipe to backpressure");
    ::close(writer);
    const int notify = ::inotify_init1(IN_CLOEXEC);
    check(notify != -1 && ::inotify_add_watch(notify, journal.c_str(), IN_OPEN) != -1,
          "watch the worker opening the journal");
    auto source = payload(kBlock, 50);
    PrefixDiskWorker worker(catalog, kGeometry, kBlock, 1);
    if (failed_read) {
        check(worker.submit(PrefixDiskWorker::Read{Kind::ReadBlock, {}, {}, ref, 0, source}),
              "accept read of unavailable record");
        worker.remove(ref);
    } else {
        check(worker.submit(PrefixDiskWorker::BlockWrite{{}, ref, identity(kNoId, 50), source}),
              "accept blocked-journal job");
    }
    worker.start();
    // IN_OPEN proves the worker reached the real journal. The full pipe then holds its fflush
    // until the reader drains it: metadata progress is measured against a deterministic I/O gate.
    pollfd event{notify, POLLIN, 0};
    const bool opened = ::poll(&event, 1, 10000) == 1;
    std::future<std::vector<PrefixDiskWorker::Completion>> failure;
    if (failed_read) {
        failure = std::async(std::launch::async, [&] { return await(worker, 1); });
    }
    auto metadata = std::async(std::launch::async, [&] {
        (void)catalog.stats();
        (void)catalog.find_block(kNoId, 50, identity(kNoId, 50).tokens, 0);
        return catalog.begin_block_write();
    });
    const auto ready = metadata.wait_for(std::chrono::seconds(2));
    const auto failure_status = failed_read ? failure.wait_for(std::chrono::seconds(2))
                                           : std::future_status::ready;
    while (::read(reader, buffer.data(), buffer.size()) > 0) {}
    const auto another = metadata.get();
    const auto done = failed_read ? failure.get() : await(worker, 1);
    worker.shutdown();
    const auto removals = worker.take_completions();
    ::close(reader);
    ::close(notify);
    std::filesystem::remove(journal);
    check(opened, "the worker must reach the blocked journal before metadata is inspected");
    check(ready == std::future_status::ready && another != kNoId,
          "Engine find/reserve/stats must not wait behind blocked journal I/O");
    check(failure_status == std::future_status::ready,
          "a read failure must reach the Engine before its cleanup journal can complete");
    check(done[0].ok != failed_read, "unblocking journal preserves the original job outcome");
    check(std::all_of(removals.begin(), removals.end(), [](const auto& item) { return item.ok; }),
          "deferred cleanup eventually finishes");
    catalog.abandon(another);
    catalog.checkpoint();
}
#endif
}

int main() {
    try {
        Scratch scratch;
        bounds_and_read_priority(scratch.path / "bounds");
        complete_image_and_restart(scratch.path / "image");
        older_image_precedes_younger_blocks(scratch.path / "fairness");
        failed_read_and_broken_closure(scratch.path / "failure");
#ifdef __linux__
        metadata_does_not_wait_for_io(scratch.path / "nonblocking");
        metadata_does_not_wait_for_io(scratch.path / "failed-read-nonblocking", true);
#endif
        std::cout << "disk worker: bounded streaming, demanded-read priority, exact restart, "
                     "failed-read completion and nonblocking metadata passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
