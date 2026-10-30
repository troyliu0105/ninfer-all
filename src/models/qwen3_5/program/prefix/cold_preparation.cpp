#include "models/qwen3_5/program/prefix/cold_preparation.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace ninfer::models::qwen3_5::detail {

bool ColdPreparation::contains(pc::NodeRef node) const {
    return std::find(path_.begin(), path_.end(), node) != path_.end();
}

void ColdPreparation::release_unused(HybridDiskTier& disk) {
    if (!held_.empty()) { disk.release_staged(held_); held_.clear(); }
}

void ColdPreparation::select(std::uint64_t head, pc::SnapshotRef snapshot,
                              std::span<const pc::NodeRef> path, std::uint64_t now_ns,
                              std::uint64_t budget_ns, HybridDiskTier& disk) {
    if (head == 0) { throw std::invalid_argument("cold preparation has no FIFO head identity"); }
    if (head_ != head) {
        release_unused(disk);
        head_ = head;
        copied_blocks_ = 0;
        deadline_ns_ = now_ns + std::min(budget_ns, std::numeric_limits<std::uint64_t>::max() - now_ns);
        state_ = State::Preparing;
    } else if (state_ == State::Fallback) {
        return;
    }
    if (!(snapshot_ == snapshot) || path_.size() != path.size() ||
        !std::equal(path_.begin(), path_.end(), path.begin())) {
        snapshot_ = snapshot;
        path_.assign(path.begin(), path.end());
        cold_.reserve(path_.size());
        released_.clear();
        std::erase_if(held_, [&](const Block& block) {
            if (contains(block.node)) { return false; }
            released_.push_back(block);
            return true;
        });
        disk.release_staged(released_);
        state_ = State::Preparing;
    }
    dirty_ = true;
}

void ColdPreparation::abandon(std::uint64_t head, HybridDiskTier& disk) {
    if (!owns(head)) { return; }
    release_unused(disk);
    head_ = 0;
    snapshot_ = {};
    path_.clear();
    state_ = State::Idle;
    dirty_ = true;
}

void ColdPreparation::clear_after_settlement(HybridDiskTier& disk) {
    release_unused(disk);
    disk.release_staged(inflight_);
    disk.release_staged(disk.take_staged_blocks());
    disk.release_image();
    disk.release_tail();
    *this = ColdPreparation{};
}

bool ColdPreparation::pending(const HybridDiskTier& disk) const noexcept {
    return dirty_ || state_ == State::Preparing || !inflight_.empty() || disk.reads_pending();
}

bool ColdPreparation::progress(pc::PrefixCacheIndex& index, HybridDiskTier& disk,
                                std::uint64_t now_ns, const Copies& copies) {
    const auto previous = state_;
    const auto completion = disk.completion_revision();
    const auto room = copies.room(copies.context);
    if (!dirty_ && inflight_.empty() && completion == observed_completion_ &&
        copies.source_revision == observed_source_revision_ && room <= observed_room_ &&
        (state_ != State::Preparing || now_ns < deadline_ns_)) {
        return false;
    }
    observed_completion_ = completion;
    observed_source_revision_ = copies.source_revision;
    observed_room_ = room;
    dirty_ = false;
    disk.publish_completions(index);

    released_.clear();
    std::erase_if(inflight_, [&](const Block& block) {
        if (index.valid(block.node) && index.node(block.node).device == pc::CopyState::Filling) {
            return false;
        }
        released_.push_back(block);
        return true;
    });
    for (const auto& block : disk.take_staged_blocks()) {
        if (state_ != State::Idle && state_ != State::Fallback && contains(block.node)) {
            held_.push_back(block);
        } else {
            released_.push_back(block);
        }
    }
    std::erase_if(held_, [&](Block& block) {
        bool discard = !index.valid(block.node);
        if (!discard) {
            const auto node = index.node(block.node);
            // A different copy may have won while this read ran. Its Filling destination
            // is not ours to overwrite; a completed Host source uses normal pipelined restore.
            discard = node.device != pc::CopyState::Absent || node.host == pc::CopyState::Resident;
        }
        if (discard) { released_.push_back(block); }
        return discard;
    });
    disk.release_staged(released_);
    if (state_ == State::Idle) { return previous != state_; }
    const auto fallback = [&] {
        release_unused(disk);
        state_ = State::Fallback;
        return previous != state_;
    };
    if (!index.valid(snapshot_)) { return fallback(); }
    const auto snapshot = index.snapshot(snapshot_);
    const bool disk_image = snapshot.disk == pc::CopyState::Resident &&
                            snapshot.disk_committed && !snapshot.disk_invalidated;
    if ((snapshot.device_slot == pc::kNoId && snapshot.host != pc::CopyState::Resident && !disk_image) ||
        (snapshot.tail_len != 0 && snapshot.tail_device_copy != pc::CopyState::Resident &&
         snapshot.host != pc::CopyState::Resident && !disk_image)) {
        return fallback();
    }
    bool ready = true;
    cold_.clear();
    for (const auto node : path_) {
        if (!index.valid(node)) { return fallback(); }
        const auto view = index.node(node);
        if (view.device == pc::CopyState::Resident || view.host == pc::CopyState::Resident) { continue; }
        ready = false;
        if (view.device == pc::CopyState::Filling || view.host == pc::CopyState::Filling) { continue; }
        if (view.disk != pc::CopyState::Resident) { return fallback(); }
        const bool held = std::any_of(held_.begin(), held_.end(),
                                      [&](const Block& block) { return block.node == node; }) ||
                          std::any_of(inflight_.begin(), inflight_.end(),
                                      [&](const Block& block) { return block.node == node; });
        if (!held) { cold_.push_back(node); }
    }
    const bool auxiliary_ready =
        copies.prepare_auxiliary(copies.context, snapshot_,
                                 previous != State::Fallback && now_ns < deadline_ns_);
    if (ready && auxiliary_ready && inflight_.empty()) {
        state_ = State::Ready;
        return previous != state_;
    }
    // Expiry forbids new work, not use of an already accepted copy that lands late.
    if (previous == State::Fallback || now_ns >= deadline_ns_) { return fallback(); }
    state_ = State::Preparing;
    // A landed batch is consumed immediately; the entire cold chain never occupies staging.
    if (!held_.empty() && inflight_.empty() && room != 0) {
        const auto submitted = copies.submit(copies.context, held_, path_);
        if (submitted > held_.size()) { throw std::logic_error("cold copy accepted nonexistent sources"); }
        inflight_.insert(inflight_.end(), held_.begin(), held_.begin() + submitted);
        copied_blocks_ += submitted;
        held_.erase(held_.begin(), held_.begin() + submitted);
    }
    if (!cold_.empty()) { (void)disk.stage_blocks(cold_, index); }
    return previous != state_;
}

} // namespace ninfer::models::qwen3_5::detail
