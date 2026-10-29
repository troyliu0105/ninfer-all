// Flat reference oracle: exact token prefixes and independently tracked source ownership.
// No GDSF/LRU/slab-allocation algorithm is reproduced. Controlled drains make victim order free.
#include "runtime/prefix_cache/prefix_index.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace ninfer;
using namespace ninfer::runtime::prefix_cache;
constexpr std::uint64_t kCollision = 42;
constexpr std::uint64_t kBlockBytes = 3072;
constexpr std::uint64_t kImageBytes = 8192;
std::uint32_t seed = 0;
unsigned epoch = 0;
std::vector<std::string> operations;
void check(bool value, const char* why) { if (!value) { throw std::runtime_error(why); } }
void trace(std::string operation) { operations.push_back(std::move(operation)); }

struct Backend final : PrefixIndexBackend {
    std::set<std::uint32_t> devices, records;
    std::uint32_t next_device = 1, next_record = 1000;
    bool duplicate_release = false;
    std::uint32_t device() { const auto id = next_device++; devices.insert(id); return id; }
    std::uint32_t record() { const auto id = next_record++; records.insert(id); return id; }
    void release_device_block(std::uint32_t id) noexcept override {
        if (devices.erase(id) != 1) { duplicate_release = true; }
    }
    void release_disk_record(std::uint32_t id) noexcept override {
        if (records.erase(id) != 1) { duplicate_release = true; }
    }
    void release_snapshot_disk_records(SnapshotRef, std::uint32_t id, bool) noexcept override {
        release_disk_record(id);
    }
};
struct Node {
    NodeRef ref;
    int parent = -1;
    std::vector<TokenId> tokens;
    std::uint64_t extra = 0;
    bool allocated = true, device = true, host = false, disk = false;
    unsigned pins = 0;
    std::uint32_t device_id = kNoId, record = kNoId;
};
struct Snapshot {
    SnapshotRef ref;
    std::vector<TokenId> prefix;
    std::vector<unsigned> path;
    bool allocated = true, image = true, tail_device = false, host = false, disk = false;
    unsigned pins = 0;
    std::uint32_t tail = 0, tail_id = kNoId, record = kNoId;
};
struct Query { std::vector<TokenId> tokens; std::vector<std::uint64_t> extras; };
PrefixIndexConfig config() {
    PrefixIndexConfig out;
    out.max_nodes = 16;
    out.max_snapshots = 8;
    out.device_snapshot_slots = 4;
    out.host_slabs = 48;
    out.image_slabs = 2;
    out.block_bytes = kBlockBytes;
    out.image_bytes = kImageBytes;
    out.disk_capacity_records = 32;
    out.cost.token_seconds = 0.001;
    out.cost.h2d_bytes_per_second = 1.0e9;
    out.cost.disk_read_bytes_per_second = 1.0e8;
    return out;
}
std::vector<TokenId> tokens(std::mt19937& random, unsigned count) {
    std::vector<TokenId> out(count);
    for (auto& token : out) { token = random() % 50000U; }
    return out;
}

struct Reference {
    Backend backend;
    PrefixCacheIndex index{config(), backend};
    std::vector<Node> nodes;
    std::vector<Snapshot> snapshots;
    std::vector<Query> queries;
    std::vector<NodeRef> retired_nodes;
    std::vector<SnapshotRef> retired_snapshots;

    bool under(unsigned child, unsigned ancestor) const {
        for (int current = static_cast<int>(child); current >= 0; current = nodes[current].parent) {
            if (static_cast<unsigned>(current) == ancestor) { return true; }
        }
        return false;
    }
    bool eligible(const Snapshot& snapshot, const Query& query) const {
        if (!snapshot.allocated || !(snapshot.image || snapshot.host || snapshot.disk) ||
            (snapshot.tail && !(snapshot.tail_device || snapshot.host || snapshot.disk)) ||
            snapshot.prefix.size() >= query.tokens.size() ||
            !std::equal(snapshot.prefix.begin(), snapshot.prefix.end(), query.tokens.begin())) { return false; }
        for (unsigned i = 0; i < snapshot.path.size(); ++i) {
            const auto& node = nodes[snapshot.path[i]];
            const auto extra = query.extras.empty() ? 0 : query.extras.at(i);
            if (!node.allocated || !(node.device || node.host || node.disk) || node.extra != extra) { return false; }
        }
        return true;
    }
    void verify() {
        index.check_invariants();
        check(!backend.duplicate_release, "physical ownership released twice");
        std::set<std::uint32_t> devices, records;
        unsigned node_count = 0, snapshot_count = 0, disk_nodes = 0, disk_snapshots = 0;
        unsigned host_used = 0, slots_used = 0;
        for (const auto& node : nodes) {
            check(index.valid(node.ref) == node.allocated, "node lifetime disagrees with reference");
            if (!node.allocated) { continue; }
            ++node_count;
            disk_nodes += node.disk;
            host_used += node.host;
            if (node.device) { devices.insert(node.device_id); }
            if (node.disk) { records.insert(node.record); }
            const auto view = index.node(node.ref);
            check((view.device == CopyState::Resident) == node.device &&
                  (view.host == CopyState::Resident) == node.host &&
                  (view.disk == CopyState::Resident) == node.disk && view.pins == node.pins,
                  "node source availability or pins disagree with reference");
        }
        for (const auto& snapshot : snapshots) {
            check(index.valid(snapshot.ref) == snapshot.allocated, "snapshot lifetime disagrees with reference");
            if (!snapshot.allocated) { continue; }
            ++snapshot_count;
            disk_snapshots += snapshot.disk;
            slots_used += snapshot.image;
            host_used += snapshot.host ? 2 + (snapshot.tail != 0) : 0;
            if (snapshot.tail_device) { devices.insert(snapshot.tail_id); }
            if (snapshot.disk) { records.insert(snapshot.record); }
            const auto view = index.snapshot(snapshot.ref);
            check((view.device_slot != kNoId) == snapshot.image &&
                  (view.tail_device_copy == CopyState::Resident) == snapshot.tail_device &&
                  (view.host == CopyState::Resident) == snapshot.host &&
                  view.disk_committed == snapshot.disk && view.pins == snapshot.pins,
                  "snapshot source availability or pins disagree with reference");
        }
        const auto stats = index.stats();
        check(stats.nodes == node_count && stats.snapshots == snapshot_count &&
              stats.disk_resident_blocks == disk_nodes && stats.disk_committed_snapshots == disk_snapshots,
              "logical or persistent capacity disagrees with reference");
        check(stats.host_free_slabs + host_used == config().host_slabs &&
              stats.free_device_slots + slots_used == config().device_snapshot_slots,
              "Host slabs or image slots are not conserved");
        check(backend.devices == devices && backend.records == records,
              "backend ownership disagrees with independently surviving sources");
        for (const auto ref : retired_nodes) { check(!index.valid(ref), "stale node generation became valid"); }
        for (const auto ref : retired_snapshots) { check(!index.valid(ref), "stale snapshot generation became valid"); }
        for (const auto& query : queries) {
            std::vector<std::uint64_t> hashes(query.tokens.size() / kBlockTokens, kCollision);
            const auto result = index.match(query.tokens, hashes, query.extras, query.tokens.size());
            unsigned expected_count = 0;
            for (const auto& snapshot : snapshots) {
                if (!eligible(snapshot, query)) { continue; }
                ++expected_count;
                const auto found = std::find_if(result.candidates.begin(), result.candidates.end(),
                    [&](const MatchCandidate& candidate) { return candidate.snapshot == snapshot.ref; });
                check(found != result.candidates.end(), "an exact eligible prefix disappeared");
                unsigned host_blocks = 0, disk_blocks = 0;
                std::uint64_t transfer = 0, disk_read = 0;
                for (const auto n : snapshot.path) {
                    const auto& node = nodes[n];
                    if (node.device) { continue; }
                    transfer += kBlockBytes;
                    if (node.host) { ++host_blocks; }
                    else { ++disk_blocks; disk_read += kBlockBytes; }
                }
                if (!snapshot.image) {
                    transfer += kImageBytes;
                    if (!snapshot.host) { disk_read += kImageBytes; }
                }
                if (snapshot.tail && !snapshot.tail_device) {
                    transfer += kBlockBytes;
                    if (!snapshot.host) { disk_read += kBlockBytes; }
                }
                check(found->frontier == snapshot.prefix.size() && found->path_blocks == snapshot.path.size() &&
                      found->host_only_blocks == host_blocks && found->disk_only_blocks == disk_blocks &&
                      found->filling_blocks == 0 && found->restore_bytes == transfer &&
                      found->disk_read_bytes == disk_read,
                      "candidate frontier or source-byte accounting disagrees with reference");
                check(found->image_on_device == snapshot.image && found->image_on_host == snapshot.host &&
                      found->image_on_disk == snapshot.disk,
                      "candidate image sources disagree with reference");
            }
            check(result.candidates.size() == expected_count, "an ineligible or duplicate prefix was offered");
            if (expected_count == 0) { check(!index.choose(result, query.tokens.size()).candidate, "empty source set must choose root"); }
        }
    }
    void node_disk(unsigned n) {
        auto& node = nodes[n];
        if (node.disk) { return; }
        node.record = backend.record();
        index.pin_node(node.ref);
        index.begin_node_disk_fill(node.ref, node.record);
        index.mark_disk_backed(node.ref, node.record);
        index.unpin_node(node.ref);
        node.disk = true;
    }
    void snapshot_disk(unsigned s) {
        auto& snapshot = snapshots[s];
        if (snapshot.disk) { return; }
        snapshot.record = backend.record();
        index.pin_snapshot(snapshot.ref);
        index.begin_snapshot_disk_fill(snapshot.ref, snapshot.record);
        index.complete_snapshot_disk_fill(snapshot.ref);
        index.unpin_snapshot(snapshot.ref);
        snapshot.disk = true;
    }
    void repair() {
        for (unsigned n = 0; n < nodes.size(); ++n) { if (nodes[n].allocated) { node_disk(n); } }
        for (unsigned s = 0; s < 3; ++s) { snapshot_disk(s); }
    }
    void host_node(unsigned n) {
        auto& node = nodes[n];
        if (node.host) { return; }
        index.pin_node(node.ref);
        check(index.begin_host_fill(node.ref).has_value(), "reference workload has ample Host room");
        index.complete_host_fill(node.ref);
        index.unpin_node(node.ref);
        node.host = true;
    }
    void host_snapshot(unsigned s) {
        auto& snapshot = snapshots[s];
        if (snapshot.host) { return; }
        index.pin_snapshot(snapshot.ref);
        check(index.begin_snapshot_host_fill(snapshot.ref), "reference image has ample Host room");
        index.complete_snapshot_host_fill(snapshot.ref);
        index.unpin_snapshot(snapshot.ref);
        snapshot.host = true;
    }
    void pin_node(unsigned n, bool pin) {
        if (pin) { index.pin_node(nodes[n].ref); } else { index.unpin_node(nodes[n].ref); }
        for (int current = static_cast<int>(n); current >= 0; current = nodes[current].parent) {
            if (pin) { ++nodes[current].pins; } else { --nodes[current].pins; }
        }
    }
    void release_host(unsigned n) {
        auto& node = nodes[n];
        if (!node.host) { return; }
        trace("pin/drop Host/unpin node " + std::to_string(n));
        pin_node(n, true);
        check(index.release_redundant_host_copies({&node.ref, 1}) == 0, "pinned Host source released early");
        verify();
        pin_node(n, false);
        node.host = false;
        verify();
    }
    void invalidate(unsigned n) {
        trace("invalidate node " + std::to_string(n));
        index.invalidate_disk_block(nodes[n].ref, nodes[n].record);
        for (unsigned child = 0; child < nodes.size(); ++child) {
            if (nodes[child].allocated && under(child, n)) { nodes[child].disk = false; }
        }
        for (auto& snapshot : snapshots) {
            if (!snapshot.allocated || std::find(snapshot.path.begin(), snapshot.path.end(), n) == snapshot.path.end()) { continue; }
            snapshot.disk = false;
            if (!snapshot.image && !snapshot.host && snapshot.pins == 0) {
                snapshot.allocated = false;
                retired_snapshots.push_back(snapshot.ref);
            }
        }
    }
    void disk_only_snapshot(unsigned s, Snapshot replacement) {
        const auto anchor = nodes[replacement.path.back()].ref;
        replacement.image = false;
        replacement.tail_device = false;
        replacement.host = false;
        replacement.disk = true;
        replacement.allocated = true;
        replacement.pins = 0;
        replacement.record = backend.record();
        const auto tail = std::span<const TokenId>(replacement.prefix).subspan(replacement.path.size() * kBlockTokens);
        const auto ref = index.restore_disk_snapshot(anchor, replacement.prefix.size(), tail,
                                                      SnapshotKind::Boundary, 1, replacement.record, true);
        check(ref.has_value(), "reference disk snapshot has available descriptor capacity");
        replacement.ref = *ref;
        if (s == snapshots.size()) { snapshots.push_back(std::move(replacement)); }
        else { snapshots[s] = std::move(replacement); }
    }

    void run_epoch(std::mt19937& random) {
        trace("create branching epoch " + std::to_string(epoch));
        nodes.clear(); snapshots.clear(); queries.clear();
        const auto common = tokens(random, 2 * kBlockTokens);
        auto left = common, right = common;
        const auto a = tokens(random, 2 * kBlockTokens + 6);
        const auto b = tokens(random, 2 * kBlockTokens + 4);
        left.insert(left.end(), a.begin(), a.end());
        right.insert(right.end(), b.begin(), b.end());
        auto insert = [&](int parent, const std::vector<TokenId>& input, unsigned block) {
            Node node;
            node.parent = parent;
            node.tokens.assign(input.begin() + block * kBlockTokens, input.begin() + (block + 1) * kBlockTokens);
            node.device_id = backend.device();
            const auto result = index.insert_block(parent < 0 ? NodeRef{} : nodes[parent].ref,
                                                   kCollision, node.tokens, 0, node.device_id);
            check(result.inserted && result.device_attached, "distinct colliding tokens must not merge");
            node.ref = result.node;
            nodes.push_back(std::move(node));
        };
        insert(-1, left, 0); insert(0, left, 1); insert(1, left, 2); insert(2, left, 3);
        const std::array<NodeRef, 2> shared{nodes[0].ref, nodes[1].ref};
        index.acquire_path(shared);
        insert(1, right, 2); insert(4, right, 3);
        const std::array<NodeRef, 4> lrefs{nodes[0].ref, nodes[1].ref, nodes[2].ref, nodes[3].ref};
        const std::array<NodeRef, 4> rrefs{nodes[0].ref, nodes[1].ref, nodes[4].ref, nodes[5].ref};
        index.release_path(lrefs); index.release_path(rrefs);
        for (unsigned n = 0; n < 6; ++n) { node_disk(n); }
        auto publish = [&](std::vector<TokenId> prefix, std::vector<unsigned> path) {
            Snapshot snapshot;
            snapshot.prefix = std::move(prefix);
            snapshot.path = std::move(path);
            snapshot.tail = snapshot.prefix.size() % kBlockTokens;
            snapshot.tail_device = snapshot.tail != 0;
            if (snapshot.tail) { snapshot.tail_id = backend.device(); }
            const auto slot = index.acquire_device_slot();
            check(slot.has_value(), "reference snapshots fit their Device slots");
            const auto tail = std::span<const TokenId>(snapshot.prefix).subspan(snapshot.path.size() * kBlockTokens);
            snapshot.ref = index.publish_snapshot(nodes[snapshot.path.back()].ref, snapshot.prefix.size(), tail,
                snapshot.tail ? std::optional(snapshot.tail_id) : std::nullopt, *slot, SnapshotKind::Boundary).snapshot;
            snapshots.push_back(std::move(snapshot));
        };
        publish({left.begin(), left.end() - 1}, {0, 1, 2, 3});
        publish({right.begin(), right.end() - 1}, {0, 1, 4, 5});
        publish(common, {0, 1});
        for (unsigned s = 0; s < 3; ++s) { snapshot_disk(s); }
        Snapshot disk_root;
        disk_root.prefix.assign(left.begin(), left.begin() + kBlockTokens + 3);
        disk_root.path = {0}; disk_root.tail = 3;
        disk_only_snapshot(3, disk_root);
        Node extra;
        extra.tokens = nodes[0].tokens; extra.extra = 99; extra.device = false; extra.disk = true;
        extra.record = backend.record();
        const auto extra_ref = index.restore_disk_block({}, kCollision, extra.tokens, extra.extra, extra.record);
        check(extra_ref.has_value(), "same tokens with a different extra key must stay distinct");
        extra.ref = *extra_ref;
        nodes.push_back(extra);
        Snapshot extra_snapshot;
        extra_snapshot.prefix = extra.tokens; extra_snapshot.path = {6};
        disk_only_snapshot(4, extra_snapshot);
        queries = {{left, {}}, {right, {}}, {left, std::vector<std::uint64_t>(4, 99)}};
        auto changed = left; changed[0] ^= 1; queries.push_back({changed, {}});
        auto divergent = left; divergent[2 * kBlockTokens + 1] ^= 1; queries.push_back({divergent, {}});
        auto wrong_extra = std::vector<std::uint64_t>(4, 0); wrong_extra[2] = 17;
        queries.push_back({left, wrong_extra});
        verify();

        // Last image-copy loss is invisible immediately but its descriptor stays leased.
        trace("invalidate leased disk-only image");
        const auto old_snapshot = snapshots[3].ref;
        const auto old_record = snapshots[3].record;
        index.pin_snapshot(old_snapshot); snapshots[3].pins = 1;
        index.invalidate_disk_snapshot(old_snapshot, old_record); snapshots[3].disk = false;
        verify();
        index.unpin_snapshot(old_snapshot); snapshots[3].pins = 0; snapshots[3].allocated = false;
        retired_snapshots.push_back(old_snapshot);
        verify();
        disk_only_snapshot(3, disk_root);
        check(snapshots[3].ref != old_snapshot, "replacement image needs a new generation");
        index.invalidate_disk_snapshot(snapshots[3].ref, old_record);
        index.invalidate_disk_snapshot(old_snapshot, snapshots[3].record);
        verify();

        for (unsigned step = 0; step < 24; ++step) {
            const unsigned n = random() % 6;
            const unsigned s = random() % 3;
            switch (random() % 4) {
            case 0: trace("Host backup node " + std::to_string(n)); host_node(n); break;
            case 1: release_host(n); break;
            case 2: {
                const auto stale = nodes[n].record;
                invalidate(n); verify(); repair();
                if (!snapshots[3].allocated) { disk_only_snapshot(3, disk_root); }
                index.invalidate_disk_block(nodes[n].ref, stale);
                break;
            }
            default: {
                trace("invalidate volatile-backed image " + std::to_string(s));
                index.invalidate_disk_snapshot(snapshots[s].ref, snapshots[s].record);
                snapshots[s].disk = false;
                verify(); snapshot_disk(s); host_snapshot(s); break;
            }
            }
            verify();
        }
        trace("release Host sources, drain backed Device copies");
        for (unsigned n = 0; n < 6; ++n) { release_host(n); }
        for (unsigned s = 0; s < 3; ++s) {
            auto& snapshot = snapshots[s];
            if (!snapshot.host) { continue; }
            index.pin_snapshot(snapshot.ref); snapshot.pins = 1;
            index.release_snapshot_host_copy(snapshot.ref); verify();
            index.unpin_snapshot(snapshot.ref); snapshot.pins = 0; snapshot.host = false;
            verify();
        }
        unsigned expected_devices = 6;
        for (const auto& snapshot : snapshots) { expected_devices += snapshot.tail_device; }
        check(index.evict_backed_device_blocks(64) == expected_devices, "backed drain releases exactly the protected Device copies");
        for (auto& node : nodes) { node.device = false; }
        for (auto& snapshot : snapshots) { snapshot.tail_device = false; }
        verify();
        trace("last-copy leaf loss, then unique orphan suffix reclamation");
        invalidate(3);
        nodes[3].allocated = false; retired_nodes.push_back(nodes[3].ref);
        snapshots[0].allocated = false; retired_snapshots.push_back(snapshots[0].ref);
        verify();
        check(index.release_disk_record_range(1) == 1, "the exclusive orphan suffix must yield without deleting the live shallow boundary");
        nodes[2].allocated = false; nodes[2].disk = false; retired_nodes.push_back(nodes[2].ref);
        verify();
        trace("drain all remaining complete closures");
        unsigned expected_records = 0;
        for (const auto& node : nodes) { expected_records += node.allocated && node.disk; }
        for (const auto& snapshot : snapshots) { expected_records += snapshot.allocated && snapshot.disk; }
        check(index.evict_disk_entries(std::numeric_limits<std::uint32_t>::max()) == expected_records,
              "full closure drain must reclaim the complete remaining record set independent of victim order");
        for (auto& node : nodes) { if (node.allocated) { retired_nodes.push_back(node.ref); } node.allocated = false; }
        for (auto& snapshot : snapshots) { if (snapshot.allocated) { retired_snapshots.push_back(snapshot.ref); } snapshot.allocated = false; }
        verify();
    }
};
}

int main() {
    try {
        for (const auto fixed_seed : {0x1234U, 0xA11CEU, 0xBADC0DEU}) {
            seed = fixed_seed;
            std::mt19937 random(seed);
            Reference reference;
            for (epoch = 0; epoch < 24; ++epoch) {
                operations.clear();
                reference.run_epoch(random);
            }
        }
        std::cout << "prefix disk reference: 3 seeds, 72 branching epochs, 1728 randomized operations; "
                     "exact identities, tier sources, stale generations, pins and capacity conserved\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "prefix disk reference: seed=" << seed << " epoch=" << epoch << ": " << error.what() << '\n';
        for (const auto& operation : operations) { std::cerr << "  " << operation << '\n'; }
        return 1;
    }
}
