#include "models/qwen3_5/program/prefix/hybrid_disk.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
using namespace ninfer;
using namespace ninfer::runtime::prefix_cache;
using ninfer::models::qwen3_5::detail::HybridDiskTier;
constexpr std::size_t kSlab = 4096;
constexpr std::size_t kPayload = 3072; // padded Host slabs must not inflate block payloads
constexpr std::uint32_t kImageSlabs = 2;
using Bytes = std::vector<std::byte>;

void check(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}
struct Scratch {
    std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("ninfer_disk_cpu_" + std::to_string(std::random_device{}()));
    Scratch() { std::filesystem::create_directories(path); }
    ~Scratch() { std::error_code error; std::filesystem::remove_all(path, error); }
};
Bytes block_oracle(std::span<const TokenId> tokens) {
    Bytes out(kPayload);
    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] = std::byte((static_cast<unsigned>(tokens[i % tokens.size()]) * 17U +
                           i * 197U + i / 13U) & 255U);
    }
    return out;
}
Bytes image_oracle(unsigned key) {
    Bytes out(kImageSlabs * kSlab);
    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] = std::byte((key * 83U + i * 193U + i / 7U) & 255U);
    }
    return out;
}
std::vector<TokenId> prompt(unsigned key, std::uint32_t blocks, std::uint32_t tail = 5) {
    std::vector<TokenId> out(blocks * kBlockTokens + tail + 1);
    for (std::size_t i = 0; i < out.size(); ++i) { out[i] = (key * 997U + i * 313U) % 50000U; }
    return out;
}
PrefixIndexConfig config(std::uint32_t slabs = 6) {
    PrefixIndexConfig out;
    out.max_nodes = 128;
    out.max_snapshots = 32;
    out.host_slabs = slabs;
    out.image_slabs = kImageSlabs;
    out.device_snapshot_slots = 8;
    out.block_bytes = kPayload;
    out.image_bytes = kImageSlabs * kSlab;
    out.disk_capacity_records = 128;
    out.cost.token_seconds = 0.001;
    out.cost.h2d_bytes_per_second = 1.0e9;
    out.cost.disk_read_bytes_per_second = 1.0e8;
    return out;
}
HybridDiskTier::Options options(const std::filesystem::path& path) {
    HybridDiskTier::Options out;
    out.directory = path;
    out.block_capacity_bytes = 64 * kPayload;
    out.image_capacity_bytes = 16 * kSlab;
    out.staging_blocks = 2;
    out.host_high_watermark = 0.5;
    out.host_low_watermark = 0.0;
    out.max_inqueue_bytes = kImageSlabs * kSlab + kPayload;
    return out;
}

// Only the physical copies are test-owned. All demotion, publication, staging, record recovery,
// failure invalidation and reclamation below execute the production HybridDiskTier/index/worker.
struct Memory final : PrefixIndexBackend {
    struct Copy { NodeRef node; SnapshotRef snapshot; std::uint32_t device = kNoId; };
    PrefixCacheIndex* index = nullptr;
    HybridDiskTier* disk = nullptr;
    std::vector<Bytes> host;
    std::map<std::uint32_t, Bytes> device;
    std::map<std::uint32_t, Bytes> images;
    std::deque<Copy> pending;
    std::uint32_t next_device = 1;
    bool double_release = false;
    explicit Memory(std::uint32_t slabs) : host(slabs, Bytes(kSlab, std::byte{0xDD})) {}
    void release_device_block(std::uint32_t id) noexcept override {
        if (device.erase(id) != 1) { double_release = true; }
    }
    void release_snapshot(SnapshotRef ref) noexcept override { images.erase(ref.index); }
    void drop_snapshot_device_image(SnapshotRef ref) noexcept override { images.erase(ref.index); }
    void release_disk_record(std::uint32_t record) noexcept override {
        if (disk) { disk->release_disk_record(record); }
    }
    void release_snapshot_disk_records(SnapshotRef, std::uint32_t record, bool) noexcept override {
        if (disk) { disk->release_snapshot_disk_records(record); }
    }
    std::uint32_t allocate(Bytes bytes) {
        const auto id = next_device++;
        device.emplace(id, std::move(bytes));
        return id;
    }
    bool snapshot_write(SnapshotRef ref) {
        const auto before = index->snapshot(ref);
        if (before.host != CopyState::Absent || before.device_slot == kNoId ||
            (before.tail_len && before.tail_device_copy != CopyState::Resident)) { return false; }
        check(images.contains(ref.index), "Device image must really exist");
        index->pin_snapshot(ref);
        if (!index->begin_snapshot_host_fill(ref)) { index->unpin_snapshot(ref); return false; }
        pending.push_back({{}, ref});
        return true;
    }
    void block_write(NodeRef ref, bool repair = true) {
        const auto before = index->node(ref);
        if (before.host != CopyState::Absent || before.device != CopyState::Resident) { return; }
        index->pin_node(ref);
        const auto slab = index->begin_host_fill(ref, repair);
        if (!slab) { index->unpin_node(ref); return; }
        pending.push_back({ref, {}, before.device_id});
    }
    void land_one() {
        check(!pending.empty(), "a physical copy was actually submitted");
        const auto copy = pending.front();
        pending.pop_front();
        if (copy.node.valid()) {
            const auto view = index->node(copy.node);
            const auto& source = device.at(copy.device);
            std::copy(source.begin(), source.end(), host.at(view.host_slab).begin());
            index->complete_host_fill(copy.node);
            index->unpin_node(copy.node);
        } else {
            const auto view = index->snapshot(copy.snapshot);
            const auto& image = images.at(copy.snapshot.index);
            for (std::uint32_t i = 0; i < kImageSlabs; ++i) {
                std::copy_n(image.begin() + i * kSlab, kSlab, host.at(view.host_slabs[i]).begin());
            }
            if (view.tail_len) {
                const auto& tail = device.at(view.tail_device);
                std::copy(tail.begin(), tail.end(), host.at(view.host_slabs.back()).begin());
            }
            index->complete_snapshot_host_fill(copy.snapshot);
            index->unpin_snapshot(copy.snapshot);
        }
    }
    HybridDiskTier::HostTransfers transfers() {
        return {this,
            [](void* p, std::uint32_t slab) { return static_cast<const std::byte*>(
                static_cast<Memory*>(p)->host.at(slab).data()); },
            [](void* p, SnapshotRef ref) { return static_cast<Memory*>(p)->snapshot_write(ref); },
            [](void* p, NodeRef ref) { static_cast<Memory*>(p)->block_write(ref); }};
    }
};
struct Capture {
    std::vector<TokenId> tokens;
    std::vector<NodeRef> path;
    SnapshotRef snapshot;
    unsigned image_key = 0;
};
struct Harness {
    Memory memory;
    PrefixCacheIndex index;
    Bytes blocks{2 * kPayload};
    Bytes image{kImageSlabs * kSlab};
    Bytes tail{kPayload};
    HybridDiskTier disk;
    std::vector<SnapshotRef> captures;
    HybridDiskTier::RestoreSummary restored;
    Harness(const std::filesystem::path& path, std::uint32_t slabs = 6,
            std::optional<HybridDiskTier::Options> overrides = std::nullopt)
        : memory(slabs), index(config(slabs), memory),
          disk(overrides ? *overrides : options(path), {kSlab, kPayload, kImageSlabs},
               {blocks, image, tail}, "cpu-production-disk-v1") {
        memory.index = &index;
        memory.disk = &disk;
        restored = disk.restore_into(index);
    }
    ~Harness() { disk.shutdown(); memory.disk = nullptr; }
    Capture capture(std::vector<TokenId> tokens, unsigned image_key) {
        Capture out{std::move(tokens), {}, {}, image_key};
        const auto hashes = block_lookup_hashes(out.tokens, {});
        NodeRef parent;
        for (std::size_t i = 0; i < hashes.size(); ++i) {
            const std::span<const TokenId> block{out.tokens.data() + i * kBlockTokens, kBlockTokens};
            const auto id = memory.allocate(block_oracle(block));
            const auto result = index.insert_block(parent, hashes[i], block, 0, id);
            if (!result.device_attached) { memory.release_device_block(id); }
            out.path.push_back(result.node);
            parent = result.node;
        }
        const auto frontier = static_cast<std::uint32_t>(out.tokens.size() - 1);
        const auto tail_len = frontier % kBlockTokens;
        const std::span<const TokenId> tail_tokens{out.tokens.data() + hashes.size() * kBlockTokens, tail_len};
        const auto slot = index.acquire_device_slot();
        check(slot.has_value(), "physical image slot available");
        const auto tail_id = tail_len ? std::optional(memory.allocate(block_oracle(tail_tokens)))
                                      : std::nullopt;
        const auto published = index.publish_snapshot(parent, frontier, tail_tokens, tail_id,
                                                       *slot, SnapshotKind::Endpoint);
        check(published.created, "distinct snapshot identity");
        out.snapshot = published.snapshot;
        captures.push_back(out.snapshot);
        memory.images[out.snapshot.index] = image_oracle(image_key);
        index.release_path(out.path);
        check_state();
        return out;
    }
    void check_state() {
        index.check_invariants();
        check(!memory.double_release, "each physical Device buffer returns exactly once");
        check(index.stats().host_free_slabs <= memory.host.size(), "Host capacity conserved");
    }
    void step(bool copy = true, bool demote = true) {
        if (copy && !memory.pending.empty()) { memory.land_one(); }
        disk.publish_completions(index);
        if (demote) { disk.maybe_demote(index, memory.transfers()); }
        check_state();
    }
    template<class Predicate> void until(Predicate ready, bool copy = true, bool demote = true) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (!ready() && std::chrono::steady_clock::now() < deadline) {
            step(copy, demote);
            std::this_thread::yield();
        }
        if (!ready()) {
            const auto stats = disk.stats();
            std::cerr << "Host free=" << index.stats().host_free_slabs
                      << " copies=" << memory.pending.size()
                      << " blocks=" << stats.demoted_blocks << " images=" << stats.demoted_images
                      << " failed=" << stats.failed_closures << " rejects=" << stats.queue_rejects << '\n';
            for (const auto ref : captures) {
                if (!index.valid(ref)) { std::cerr << "snapshot retired\n"; continue; }
                const auto view = index.snapshot(ref);
                std::cerr << "snapshot " << ref.index << " host=" << static_cast<int>(view.host)
                          << " disk=" << static_cast<int>(view.disk) << " pins=" << view.pins
                          << " committed=" << view.disk_committed << '\n';
            }
            throw std::runtime_error("production orchestration failed to progress within its bounded workload");
        }
    }
    void fill_image(const Capture& capture) {
        check(memory.snapshot_write(capture.snapshot), "submit initial image write");
        memory.land_one();
    }
    void fill_block(NodeRef node) {
        memory.block_write(node, false);
        check(!memory.pending.empty(), "submit initial block write");
        memory.land_one();
    }
    SnapshotRef find(const Capture& capture) {
        const auto match = index.match(capture.tokens, block_lookup_hashes(capture.tokens, {}), {},
                                       capture.tokens.size());
        for (const auto& candidate : match.candidates) {
            if (candidate.frontier == capture.tokens.size() - 1) { return candidate.snapshot; }
        }
        throw std::runtime_error("exact persisted snapshot is not reusable");
    }
    void read_exact(const Capture& capture) {
        const auto snapshot = find(capture);
        const auto match = index.match(capture.tokens, block_lookup_hashes(capture.tokens, {}), {},
                                       capture.tokens.size());
        check(match.path.size() == capture.path.size(), "complete exact ancestor path after restart");
        std::size_t received = 0;
        while (received < match.path.size()) {
            // The remaining cold nodes may already own queued reads from the preceding batch.
            (void)disk.stage_blocks(match.path, index);
            std::vector<HybridDiskTier::StagedBlock> batch;
            until([&] {
                auto landed = disk.take_staged_blocks();
                batch.insert(batch.end(), landed.begin(), landed.end());
                return !batch.empty();
            }, false, false);
            for (const auto& staged : batch) {
                const auto found = std::find(match.path.begin(), match.path.end(), staged.node);
                check(found != match.path.end(), "staged node belongs to this exact path");
                const auto n = static_cast<std::size_t>(found - match.path.begin());
                const auto expected = block_oracle({capture.tokens.data() + n * kBlockTokens, kBlockTokens});
                check(std::equal(expected.begin(), expected.end(), staged.bytes), "exact block payload oracle");
                index.pin_node(staged.node);
                const auto destination = memory.allocate(Bytes(kPayload, std::byte{0xAC}));
                index.begin_device_fill(staged.node, destination);
                std::copy_n(staged.bytes, kPayload, memory.device.at(destination).begin());
                index.complete_device_fill(staged.node);
                index.unpin_node(staged.node);
                ++received;
            }
            disk.release_staged(batch);
            disk.release_staged(batch); // duplicate release cannot allocate a slot twice
            check_state();
        }
        check(disk.stage_image(snapshot, index), "stage exact image");
        until([&] { return disk.take_staged_image(snapshot) != nullptr; }, false, false);
        const auto expected_image = image_oracle(capture.image_key);
        check(std::equal(expected_image.begin(), expected_image.end(), disk.take_staged_image(snapshot)),
              "exact complete recurrent StateImage oracle");
        disk.release_image();
        const auto view = index.snapshot(snapshot);
        if (view.tail_len) {
            check(disk.stage_tail(snapshot, index), "stage exact tail");
            until([&] { return disk.staged_tail(snapshot) != nullptr; }, false, false);
            const auto expected_tail = block_oracle({capture.tokens.data() + capture.path.size() * kBlockTokens,
                                                     view.tail_len});
            check(std::equal(expected_tail.begin(), expected_tail.end(), disk.staged_tail(snapshot)),
                  "exact nonaligned tail oracle");
            disk.release_tail();
        }
        check_state();
    }
};

void streaming_branches_and_restart(const std::filesystem::path& path) {
    Capture first, second, shallow;
    {
        Harness h(path);
        first = h.capture(prompt(11, 12), 111);
        auto branch = prompt(22, 10, 3);
        std::copy_n(first.tokens.begin(), 4 * kBlockTokens, branch.begin());
        second = h.capture(std::move(branch), 222);
        auto prefix = std::vector<TokenId>(first.tokens.begin(), first.tokens.begin() + 3 * kBlockTokens + 1);
        shallow = h.capture(std::move(prefix), 333);
        h.fill_image(first);
        h.fill_block(first.path[0]);
        h.fill_block(first.path[1]);
        check(h.index.stats().host_free_slabs == 1, "ordinary Host capacity is full before spill");
        h.disk.maybe_demote(h.index, h.memory.transfers());
        h.until([&] { return h.index.snapshot(first.snapshot).disk_committed &&
                             h.index.snapshot(second.snapshot).disk_committed &&
                             h.index.snapshot(shallow.snapshot).disk_committed; });
        check(h.disk.stats().demoted_images == 3, "all three complete closures actually committed");
        check(h.disk.stats().demoted_blocks == 18, "shared ancestors are not stored twice");
        h.until([&] { return !h.disk.busy() && h.memory.pending.empty(); });
        check(h.index.stats().host_free_slabs == 6, "streaming drains every source lease");
    }
    Harness h(path);
    check(h.restored.blocks == 18 && h.restored.snapshots == 3, "restart restores branches and shallow boundary");
    h.read_exact(first);
    // Shared blocks are already warm: evict their Device duplicates so each read is forced cold.
    h.index.evict_backed_device_blocks(128);
    h.read_exact(second);
    h.index.evict_backed_device_blocks(128);
    h.read_exact(shallow);
}

void accepted_write_budget(const std::filesystem::path& path) {
    Harness h(path, 12);
    const auto captured = h.capture(prompt(31, 8), 334);
    h.fill_image(captured);
    for (const auto node : captured.path) { h.fill_block(node); }
    check(h.index.stats().host_free_slabs == 1, "fill ordinary Host capacity with a long ready chain");
    for (unsigned pass = 0; pass < 12; ++pass) {
        h.disk.maybe_demote(h.index, h.memory.transfers()); // deliberately do not adopt completions
        std::uint32_t leased_blocks = 0;
        for (const auto node : captured.path) {
            const auto view = h.index.node(node);
            if (view.disk == CopyState::Filling) {
                ++leased_blocks;
                check(view.host == CopyState::Resident && view.pins != 0,
                      "each accepted disk source stays owned until completion adoption");
            }
        }
        check(leased_blocks == 3 && leased_blocks * kPayload <= options(path).max_inqueue_bytes,
              "each cadence respects the exact payload-byte bound across the entire chain");
        check(h.index.snapshot(captured.snapshot).disk == CopyState::Absent,
              "an image cannot reserve ahead of its unfinished ancestors");
        h.check_state();
    }
    check(h.disk.stats().queue_rejects != 0, "the workload actually exercised queue backpressure");
    h.until([&] { return h.index.snapshot(captured.snapshot).disk_committed; });
    check(h.index.stats().host_free_slabs == 12, "adoption returns every byte and source lease");
}

void delayed_sources_and_cancellation(const std::filesystem::path& path) {
    Harness h(path);
    const auto captured = h.capture(prompt(41, 8), 444);
    h.fill_image(captured);
    h.fill_block(captured.path[0]);
    h.fill_block(captured.path[1]);
    h.disk.maybe_demote(h.index, h.memory.transfers());
    check(h.index.snapshot(captured.snapshot).disk == CopyState::Absent,
          "partial ancestor writes cannot publish or reserve an image closure");
    // A simulated H2D reader borrows a real slab until its explicit completion, independently
    // of the worker finishing a disk write and the request abandoning further preparation.
    const auto node = captured.path[0];
    const auto source_slab = h.index.node(node).host_slab;
    h.index.pin_node(node);
    h.until([&] { return h.index.node(node).disk == CopyState::Resident &&
                         h.index.node(node).pins == 1; }, true, false);
    check(h.index.node(node).host == CopyState::Resident, "disk completion cannot reuse a reader's slab");
    const auto expected = block_oracle({captured.tokens.data(), kBlockTokens});
    check(std::equal(expected.begin(), expected.end(), h.memory.host[source_slab].begin()),
          "the delayed reader still sees its original bytes after request cancellation");
    h.index.unpin_node(node);
    check(h.index.node(node).host == CopyState::Absent, "final reader returns the deferred source slab");
    std::fill(h.memory.host[source_slab].begin(), h.memory.host[source_slab].end(), std::byte{0xEF});
    // Stop physical completions at a real Filling state. Repeated demotion cadences must not
    // publish it, leak another pin or enqueue duplicate repairs while the event is held.
    h.until([&] { return !h.memory.pending.empty(); }, false, true);
    const auto copy = h.memory.pending.front();
    const auto pins = copy.node.valid() ? h.index.node(copy.node).pins
                                       : h.index.snapshot(copy.snapshot).pins;
    for (unsigned i = 0; i < 12; ++i) { h.step(false, true); }
    check(h.memory.pending.size() == 1, "a held physical event cannot enqueue duplicate repairs");
    check((copy.node.valid() ? h.index.node(copy.node).pins : h.index.snapshot(copy.snapshot).pins) == pins,
          "backpressure must not accumulate repair pins");
    h.until([&] { return h.index.snapshot(captured.snapshot).disk == CopyState::Filling; });
    h.index.pin_snapshot(captured.snapshot); // delayed Host image/tail reader
    const auto source = h.index.snapshot(captured.snapshot);
    std::vector<std::uint32_t> slabs(source.host_slabs.begin(), source.host_slabs.end());
    h.until([&] { return h.index.snapshot(captured.snapshot).disk_committed; }, false, false);
    check(h.index.snapshot(captured.snapshot).host == CopyState::Resident,
          "image commit cannot free a delayed image/tail reader's source");
    Bytes actual;
    for (std::uint32_t i = 0; i < kImageSlabs; ++i) {
        actual.insert(actual.end(), h.memory.host[slabs[i]].begin(), h.memory.host[slabs[i]].end());
    }
    check(actual == image_oracle(captured.image_key), "delayed image reader sees the complete original state");
    const auto tail = block_oracle({captured.tokens.data() + captured.path.size() * kBlockTokens, 5});
    check(std::equal(tail.begin(), tail.end(), h.memory.host[slabs.back()].begin()), "delayed tail reader bytes");
    h.index.unpin_snapshot(captured.snapshot);
    check(h.index.stats().host_free_slabs == 6, "cancelled interests leave no source leases after completion");
    h.check_state();
}

void cancelled_staging_readers(const std::filesystem::path& path) {
    Capture first, second;
    {
        Harness h(path);
        first = h.capture(prompt(71, 6), 771);
        second = h.capture(prompt(72, 6), 772);
        h.fill_image(first);
        h.fill_block(first.path[0]);
        h.fill_block(first.path[1]);
        h.until([&] { return h.index.snapshot(first.snapshot).disk_committed &&
                             h.index.snapshot(second.snapshot).disk_committed; });
    }
    Harness h(path);
    const auto first_path = h.index.match(first.tokens, block_lookup_hashes(first.tokens, {}), {},
                                          first.tokens.size()).path;
    const auto second_path = h.index.match(second.tokens, block_lookup_hashes(second.tokens, {}), {},
                                           second.tokens.size()).path;
    check(h.disk.stage_blocks(first_path, h.index), "accept the two initial staging reads");
    std::vector<HybridDiskTier::StagedBlock> held;
    h.until([&] {
        auto batch = h.disk.take_staged_blocks();
        held.insert(held.end(), batch.begin(), batch.end());
        return held.size() == 2;
    }, false, false);
    std::array<std::uint32_t, 2> destinations{};
    for (std::size_t i = 0; i < held.size(); ++i) {
        h.index.pin_node(held[i].node);
        destinations[i] = h.memory.allocate(Bytes(kPayload, std::byte{0xBA}));
        h.index.begin_device_fill(held[i].node, destinations[i]);
    }
    // The request is now cancelled, but both submitted H2D copies still read their staging.
    check(!h.disk.stage_blocks(second_path, h.index), "cancellation cannot recycle accepted copy sources");
    check(h.index.evict_backed_device_blocks(128) == 0,
          "Device eviction cannot reclaim either unfinished destination");
    std::copy_n(held[0].bytes, kPayload, h.memory.device.at(destinations[0]).begin());
    h.index.complete_device_fill(held[0].node);
    h.index.unpin_node(held[0].node);
    const std::vector<HybridDiskTier::StagedBlock> returned{held[0]};
    h.disk.release_staged(returned);
    check(h.disk.stage_blocks({second_path.front()}, h.index), "one finished copy funds exactly one new read");
    h.disk.release_staged(returned); // stale release after this physical slot changed owners
    check(!h.disk.stage_blocks({second_path[1]}, h.index),
          "a stale release must not free a slot now owned by a different read generation");
    std::vector<HybridDiskTier::StagedBlock> newer;
    h.until([&] {
        auto batch = h.disk.take_staged_blocks();
        newer.insert(newer.end(), batch.begin(), batch.end());
        return !newer.empty();
    }, false, false);
    const auto expected_new = block_oracle({second.tokens.data(), kBlockTokens});
    check(std::equal(expected_new.begin(), expected_new.end(), newer.front().bytes),
          "the new slot owner receives its own exact bytes");
    const auto expected_held = block_oracle({first.tokens.data() + kBlockTokens, kBlockTokens});
    check(std::equal(expected_held.begin(), expected_held.end(), held[1].bytes),
          "another cancelled copy keeps its original bytes while the first slot is reused");
    std::copy_n(held[1].bytes, kPayload, h.memory.device.at(destinations[1]).begin());
    h.index.complete_device_fill(held[1].node);
    h.index.unpin_node(held[1].node);
    h.disk.release_staged({held[1]});
    h.disk.release_staged(newer);
    check(h.memory.device.at(destinations[1]) == expected_held,
          "the delayed cancelled copy lands with intact source contents");
    check(h.index.node(held[0].node).pins == 0 && h.index.node(held[1].node).pins == 0,
          "accepted copy leases finish independently of cancelled request interest");
    h.check_state();
}

void corrupt_payload(const std::filesystem::path& file, const Bytes& expected) {
    std::ifstream input(file, std::ios::binary);
    Bytes contents(static_cast<std::size_t>(std::filesystem::file_size(file)));
    input.read(reinterpret_cast<char*>(contents.data()), contents.size());
    check(static_cast<std::size_t>(input.gcount()) == contents.size(), "read real payload file for fault injection");
    const auto found = std::search(contents.begin(), contents.end(), expected.begin(), expected.end());
    check(found != contents.end(), "locate actual payload by contents, not private file offsets");
    const auto offset = found - contents.begin();
    const char damaged = static_cast<char>(std::to_integer<unsigned>(*found) ^ 0x80U);
    std::fstream output(file, std::ios::binary | std::ios::in | std::ios::out);
    output.seekp(offset);
    output.write(&damaged, 1);
    output.flush();
    check(static_cast<bool>(output), "inject one real payload checksum error");
}

void failed_read_retires_old_generation(const std::filesystem::path& path) {
    Capture captured;
    {
        Harness h(path);
        captured = h.capture(prompt(51, 5), 555);
        h.fill_image(captured);
        h.fill_block(captured.path[0]);
        h.fill_block(captured.path[1]);
        h.until([&] { return h.index.snapshot(captured.snapshot).disk_committed; });
    }
    Harness h(path);
    auto match = h.index.match(captured.tokens, block_lookup_hashes(captured.tokens, {}), {}, captured.tokens.size());
    check(match.candidates.size() == 1, "cold restart has a real disk-only candidate");
    const auto old_root = match.path.front();
    corrupt_payload(path / "blocks.bin", block_oracle({captured.tokens.data(), kBlockTokens}));
    check(h.disk.stage_blocks(match.path, h.index), "accept read of damaged parent and another old node");
    h.until([&] { return !h.index.valid(old_root); }, false, false);
    check(h.index.match(captured.tokens, block_lookup_hashes(captured.tokens, {}), {},
                        captured.tokens.size()).candidates.empty(), "bad disk-only parent immediately falls back to root");
    const auto replacement = h.capture(captured.tokens, 556);
    check(replacement.path.front() != old_root, "descriptor reuse carries a new generation");
    h.until([&] { return !h.disk.busy(); }, false, false);
    const auto stale = h.disk.take_staged_blocks();
    check(stale.empty(), "old accepted read completions cannot publish into replacement descriptors");
    check(h.index.valid(replacement.snapshot) && h.index.node(replacement.path.front()).device == CopyState::Resident,
          "late old-generation failure preserves the replacement's volatile sources");
    h.check_state();
}

void retry_removal_without_engine_io(const std::filesystem::path& path) {
    Capture captured;
    {
        Harness h(path);
        captured = h.capture(prompt(61, 6), 666);
        h.fill_image(captured);
        h.fill_block(captured.path[0]);
        h.fill_block(captured.path[1]);
        h.until([&] { return h.index.snapshot(captured.snapshot).disk_committed; });
    }
    Harness h(path);
    const auto snapshot = h.find(captured);
    const auto journal = path / "catalog.journal";
    check(std::filesystem::create_directory(journal), "inject a real tombstone-open failure");
    h.index.supersede(snapshot);
    check(h.index.evict_disk_entries(1) == 7, "retire the complete exclusive closure");
    h.until([&] { return h.disk.stats().pending_removals != 0; }, false, false);
    check(h.index.stats().nodes == 0 && h.index.stats().snapshots == 0,
          "logical retirement and root fallback do not wait for failed cleanup I/O");
    check(std::filesystem::remove(journal), "remove the injected journal failure");
    h.until([&] { return !h.disk.busy(); }, false, false);
    check(h.disk.stats().catalog_blocks == 0 && h.disk.stats().catalog_committed == 0,
          "production completion retries eventually remove every retained record");
    const auto replacement = h.capture(prompt(62, 6), 667);
    h.fill_image(replacement);
    h.fill_block(replacement.path[0]);
    h.fill_block(replacement.path[1]);
    h.until([&] { return h.index.snapshot(replacement.snapshot).disk_committed; });
    check(h.disk.stats().failed_closures == 0, "reclaimed capacity supports a new complete closure");
}
}

int main() {
    try {
        Scratch scratch;
        streaming_branches_and_restart(scratch.path / "streaming");
        accepted_write_budget(scratch.path / "budget");
        delayed_sources_and_cancellation(scratch.path / "leases");
        cancelled_staging_readers(scratch.path / "staging-readers");
        failed_read_retires_old_generation(scratch.path / "bad-read");
        retry_removal_without_engine_io(scratch.path / "removal");
        std::cout << "hybrid disk CPU: production streaming, branches, exact restart, delayed copies, "
                     "cancelled interests, checksum fallback, stale generations and removal retry passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "hybrid disk CPU: " << error.what() << '\n';
        return 1;
    }
}
