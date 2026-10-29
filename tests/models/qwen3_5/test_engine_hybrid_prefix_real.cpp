// Real-artifact scenarios for the hybrid prefix cache (docs/maintainer/hybrid-prefix-cache-spec.md
// §13.3). Requires NINFER_TEST_ARTIFACT; NINFER_HYBRID_REAL_SCENARIO selects one scenario.

#include "ninfer/engine.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

namespace {

constexpr std::uint32_t kPrefillChunk = 512;
// Tokens per cached KV block.
constexpr std::uint32_t kBlock = 64;

// NINFER_HYBRID_KV_DTYPE selects the KV storage every scenario runs with (default bf16), so the
// Host tier's page records and restores are exercised for each profile.
ninfer::KvCacheStorage kv_storage = ninfer::KvCacheStorage::BFloat16;

bool select_kv_storage(std::string_view name) {
    if (name == "bf16") {
        kv_storage = ninfer::KvCacheStorage::BFloat16;
    } else if (name == "int8") {
        kv_storage = ninfer::KvCacheStorage::Int8Group64;
    } else if (name == "fp8") {
        kv_storage = ninfer::KvCacheStorage::Fp8E4M3Row256;
    } else if (name == "nvfp4") {
        kv_storage = ninfer::KvCacheStorage::Nvfp4Group16;
    } else if (name == "k8v4") {
        kv_storage = ninfer::KvCacheStorage::Fp8KeyNvfp4Value;
    } else {
        return false;
    }
    return true;
}

// Deterministic ordinary-vocabulary tokens; the content only has to be reproducible.
std::vector<ninfer::TokenId> synthetic_tokens(std::size_t count, std::uint32_t seed) {
    std::vector<ninfer::TokenId> tokens;
    tokens.reserve(count);
    std::uint32_t state = seed * 2654435761U + 1U;
    for (std::size_t index = 0; index < count; ++index) {
        state = state * 1664525U + 1013904223U;
        tokens.push_back(static_cast<ninfer::TokenId>(1000U + (state >> 8U) % 30000U));
    }
    return tokens;
}

ninfer::EngineOptions hybrid_options(const char* artifact, ninfer::SpeculativeBackend backend,
                                     std::uint32_t kv_tokens, std::size_t host_bytes,
                                     std::uint32_t device_slots) {
    ninfer::EngineOptions options;
    options.artifact_path        = artifact;
    options.max_context          = 4096;
    options.kv_capacity          = ninfer::KvCapacityPolicy::explicit_capacity(kv_tokens);
    options.prefill_chunk        = kPrefillChunk;
    options.kv_cache             = kv_storage;
    options.speculative.backend  = backend;
    options.max_concurrency      = 1;
    options.max_pending_requests = 1;
    if (backend == ninfer::SpeculativeBackend::Mtp) {
        options.speculative.draft_tokens  = 3;
        options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
    } else if (backend == ninfer::SpeculativeBackend::DFlash2) {
        options.speculative.draft_tokens = 7;
    }
    options.context_cache.mode                         = ninfer::ContextCacheMode::Hybrid;
    options.context_cache.host_cache_budget_bytes      = host_bytes;
    options.context_cache.hybrid.device_snapshot_slots = device_slots;
    return options;
}

ninfer::RequestOptions greedy(std::uint32_t outputs) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = outputs;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = true;
    options.stop.include_model_defaults       = false;
    return options;
}

template <typename Call>
bool fails_unavailable(Call&& call) {
    try {
        std::forward<Call>(call)();
    } catch (const ninfer::RequestError& error) {
        return error.kind() == ninfer::RequestErrorKind::Unavailable;
    }
    return false;
}

struct Observed {
    std::vector<ninfer::TokenId> tokens;
    std::uint32_t reused = 0;
    ninfer::RuntimeStats stats;
};

// A later request restores a flexible prompt-tail snapshot the first request captured at its last
// prefill chunk boundary. The Device-resident engine restores it with Device copies; the
// constrained engine is first forced to evict the snapshot image and most of its blocks, so the
// same snapshot comes back through the Host slabs. Both restores carry identical bytes and resume
// at the same frontier with the same chunking, so greedy generation must match token for token.
int exercise_restore_exact(const char* artifact, ninfer::SpeculativeBackend backend) {
    const std::vector<ninfer::TokenId> first = synthetic_tokens(1500, 1);
    std::vector<ninfer::TokenId> second(first.begin(), first.begin() + 1400);
    const std::vector<ninfer::TokenId> suffix = synthetic_tokens(300, 2);
    second.insert(second.end(), suffix.begin(), suffix.end());
    // 3900 prompt tokens need 61 of the constrained engine's 64 pages, evicting all but the
    // first few of the first request's cached blocks.
    const std::vector<ninfer::TokenId> pressure = synthetic_tokens(3900, 3);

    const auto run = [&](ninfer::EngineOptions options, bool apply_pressure) {
        ninfer::Engine engine(std::move(options));
        (void)engine.generate(engine.prepare_tokens(first), greedy(8));
        if (apply_pressure) { (void)engine.generate(engine.prepare_tokens(pressure), greedy(4)); }
        const ninfer::GenerationResult result =
            engine.generate(engine.prepare_tokens(second), greedy(24));
        return Observed{result.generated_token_ids, result.reused_prompt_tokens,
                        engine.runtime_stats()};
    };
    const Observed device = run(hybrid_options(artifact, backend, 16384, 1ULL << 30, 8), false);
    const Observed host   = run(hybrid_options(artifact, backend, 4096, 1ULL << 30, 1), true);

    // The first prompt's chunks end at 512 and 1024; the final chunk [1024, 1500) holds the
    // flexible prompt-tail tap, realized at its start.
    constexpr std::uint32_t kExpectedReuse = 2 * kPrefillChunk;
    int failures                           = 0;
    if (device.reused != kExpectedReuse || host.reused != kExpectedReuse) {
        std::cerr << "restore-exact: reused device=" << device.reused << " host=" << host.reused
                  << ", expected the prompt-tail snapshot at " << kExpectedReuse << '\n';
        ++failures;
    }
    if (device.stats.hybrid_host_image_restores != 0 ||
        device.stats.hybrid_host_block_restores != 0) {
        std::cerr << "restore-exact: the Device-resident engine restored from Host\n";
        ++failures;
    }
    if (host.stats.hybrid_host_image_restores == 0 || host.stats.hybrid_host_block_restores == 0) {
        std::cerr << "restore-exact: the constrained engine did not restore through Host slabs"
                  << " (images=" << host.stats.hybrid_host_image_restores
                  << " blocks=" << host.stats.hybrid_host_block_restores << ")\n";
        ++failures;
    }
    if (device.tokens.size() != 24 || device.tokens != host.tokens) {
        std::cerr << "restore-exact: Host and Device restores generated different tokens\n";
        ++failures;
    }
    return failures;
}

// A restarted Engine resumes from the Host tier its predecessor saved when stopped mid-generation:
// the restored snapshot is the same bytes, so greedy generation matches an Engine that never
// restarted. A file written for a different build identity is ignored, a Host tier one slab
// smaller than the file restores some but not all of its snapshots, and an abandoned save keeps
// the previous file and leaves no temporary one.
int exercise_persist(const char* artifact) {
    const std::filesystem::path file =
        std::filesystem::temp_directory_path() / "ninfer-hybrid-persist-real-test.bin";
    std::error_code ignored;
    std::filesystem::remove(file, ignored);
    const std::vector<ninfer::TokenId> first = synthetic_tokens(1500, 4);
    std::vector<ninfer::TokenId> second(first.begin(), first.begin() + 1400);
    const std::vector<ninfer::TokenId> suffix = synthetic_tokens(300, 5);
    second.insert(second.end(), suffix.begin(), suffix.end());
    const auto options = [&](const char* identity) {
        ninfer::EngineOptions result =
            hybrid_options(artifact, ninfer::SpeculativeBackend::None, 16384, 1ULL << 30, 8);
        if (identity != nullptr) {
            result.context_cache.hybrid.persistent_file     = file;
            result.context_cache.hybrid.persistent_identity = identity;
        }
        return result;
    };

    std::vector<ninfer::TokenId> reference;
    {
        ninfer::Engine engine(options(nullptr));
        (void)engine.generate(engine.prepare_tokens(first), greedy(8));
        reference = engine.generate(engine.prepare_tokens(second), greedy(24)).generated_token_ids;
    }
    int failures = 0;
    {
        // The Engine stops the way ninfer-serve does: stop() while a generation is still far from
        // its output limit. It fails as Unavailable instead of running on, new work is refused,
        // and the Host tier is still saved.
        ninfer::Engine saver(options("persist-test"));
        (void)saver.generate(saver.prepare_tokens(first), greedy(8));
        ninfer::GenerationHandle running =
            saver.submit(saver.prepare_tokens(synthetic_tokens(200, 6)), greedy(3500));
        std::this_thread::sleep_for(std::chrono::seconds(2));
        saver.stop();
        if (!fails_unavailable([&] { (void)running.wait(); })) {
            std::cerr << "persist: a running generation was not ended as Unavailable by stop()\n";
            ++failures;
        }
        // The answer comes before the Host tier is saved (seconds for a large one), so the file
        // is not in place yet.
        if (std::filesystem::exists(file)) {
            std::cerr << "persist: stop() answered the running generation only after the save\n";
            ++failures;
        }
        if (!fails_unavailable([&] { (void)saver.submit(saver.prepare_tokens(second), greedy(4)); })) {
            std::cerr << "persist: a stopped Engine accepted a request\n";
            ++failures;
        }
    }
    if (!std::filesystem::exists(file)) {
        std::cerr << "persist: the Engine did not save its Host tier\n";
        return 1;
    }
    // The saver's file, for the smaller Host tier below (the loader saves over `file`).
    const std::filesystem::path partial =
        std::filesystem::temp_directory_path() / "ninfer-hybrid-persist-real-test-partial.bin";
    std::filesystem::copy_file(file, partial, std::filesystem::copy_options::overwrite_existing);
    // The file read is published as one startup phase: the whole file, never a failure (a failed
    // phase would be logged as a failed startup).
    std::vector<ninfer::StartupEvent> load_events;
    const auto observe = [&](ninfer::EngineOptions result) {
        load_events.clear();
        result.startup_observer.callback = [&](const ninfer::StartupEvent& event) {
            if (event.phase == ninfer::StartupPhase::PrefixCacheLoad) {
                load_events.push_back(event);
            }
        };
        return result;
    };
    const auto observed = [&](const char* identity) { return observe(options(identity)); };
    const auto one_phase = [&](std::uint64_t expected_bytes) {
        bool monotonic = true;
        for (std::size_t index = 1; index < load_events.size(); ++index) {
            monotonic = monotonic && load_events[index].current >= load_events[index - 1].current &&
                        load_events[index].current <= expected_bytes;
        }
        return load_events.size() >= 2 &&
               load_events.front().status == ninfer::StartupStatus::Begin &&
               load_events.front().total == expected_bytes &&
               load_events.back().status == ninfer::StartupStatus::Complete &&
               load_events.back().current == expected_bytes && monotonic &&
               std::all_of(load_events.begin() + 1, load_events.end() - 1,
                           [](const ninfer::StartupEvent& event) {
                               return event.status == ninfer::StartupStatus::Progress;
                           });
    };
    std::uint64_t required_host_bytes = 0;
    std::uint64_t saved_snapshots     = 0;
    {
        const std::uint64_t file_bytes = std::filesystem::file_size(file);
        ninfer::Engine loader(observed("persist-test"));
        if (!one_phase(file_bytes)) {
            std::cerr << "persist: the file read was not published as one complete phase ("
                      << load_events.size() << " events)\n";
            ++failures;
        }
        const ninfer::LoadSummary summary = loader.load_summary();
        const ninfer::GenerationResult resumed =
            loader.generate(loader.prepare_tokens(second), greedy(24));
        const ninfer::RuntimeStats stats = loader.runtime_stats();
        const ninfer::LoadSummary::PrefixCacheRestore& cache = summary.prefix_cache;
        if (!cache.restored || cache.snapshots == 0 || cache.snapshots != cache.saved_snapshots ||
            cache.blocks != cache.saved_blocks || cache.required_host_bytes > cache.host_bytes) {
            std::cerr << "persist: the saved Host tier was not wholly restored ("
                      << cache.message << ")\n";
            ++failures;
        }
        required_host_bytes = cache.required_host_bytes;
        saved_snapshots     = cache.saved_snapshots;
        if (resumed.reused_prompt_tokens != 2 * kPrefillChunk ||
            stats.hybrid_host_block_restores == 0 || stats.hybrid_host_image_restores == 0) {
            std::cerr << "persist: the restarted Engine reused " << resumed.reused_prompt_tokens
                      << " tokens (blocks restored " << stats.hybrid_host_block_restores << ")\n";
            ++failures;
        }
        if (resumed.generated_token_ids != reference) {
            std::cerr << "persist: the restarted Engine generated different tokens\n";
            ++failures;
        }
    }
    {
        ninfer::Engine foreign(observed("another-build"));
        const ninfer::LoadSummary summary = foreign.load_summary();
        if (!load_events.empty()) {
            std::cerr << "persist: a file from another build published a load phase\n";
            ++failures;
        }
        if (summary.prefix_cache.restored || summary.prefix_cache.message.empty() ||
            foreign.generate(foreign.prepare_tokens(second), greedy(4)).reused_prompt_tokens != 0) {
            std::cerr << "persist: a file from another build was not ignored\n";
            ++failures;
        }
    }
    if (saved_snapshots < 2) {
        std::cerr << "persist: the saved tier holds " << saved_snapshots
                  << " snapshots; a smaller tier cannot be checked\n";
        ++failures;
    } else {
        // One slab short of the file: whole snapshots are dropped, never all of them, and only the
        // chosen entries are read.
        ninfer::EngineOptions small = observe(hybrid_options(
            artifact, ninfer::SpeculativeBackend::None, 16384, required_host_bytes - 1, 8));
        small.context_cache.hybrid.persistent_file     = partial;
        small.context_cache.hybrid.persistent_identity = "persist-test";
        const std::uint64_t file_bytes                 = std::filesystem::file_size(partial);
        ninfer::Engine engine(std::move(small));
        const ninfer::LoadSummary::PrefixCacheRestore cache = engine.load_summary().prefix_cache;
        if (!cache.restored || cache.snapshots == 0 || cache.snapshots >= cache.saved_snapshots ||
            cache.blocks > cache.saved_blocks || cache.required_host_bytes <= cache.host_bytes) {
            std::cerr << "persist: a smaller Host tier restored " << cache.snapshots << " of "
                      << cache.saved_snapshots << " snapshots (" << cache.message << ")\n";
            ++failures;
        }
        if (!one_phase(cache.bytes) || cache.bytes >= file_bytes) {
            std::cerr << "persist: a smaller Host tier did not read only what it restored ("
                      << cache.bytes << " of " << file_bytes << " bytes)\n";
            ++failures;
        }
        (void)engine.generate(engine.prepare_tokens(second), greedy(4));
    }
    // A Ctrl+C during ninfer-serve's stop abandons the save: the unfinished file is deleted and
    // the previous file stays as it was, whether the save was writing or had not begun.
    std::filesystem::path temporary = file;
    temporary += ".tmp";
    const auto unchanged = [&, bytes = std::filesystem::file_size(file),
                            time = std::filesystem::last_write_time(file)] {
        std::error_code error;
        return !std::filesystem::exists(temporary, error) &&
               std::filesystem::file_size(file, error) == bytes &&
               std::filesystem::last_write_time(file, error) == time;
    };
    using Abandon = ninfer::PrefixCacheSaveControl::Abandon;
    {
        ninfer::EngineOptions writing = options("persist-test");
        const ninfer::PrefixCacheSaveControl control = writing.context_cache.hybrid.persistent_save;
        std::optional<Abandon> result;
        {
            ninfer::Engine engine(std::move(writing));
            (void)engine.generate(engine.prepare_tokens(first), greedy(8));
            std::thread interrupt([&] {
                const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
                std::error_code error;
                while (!std::filesystem::exists(temporary, error) &&
                       std::chrono::steady_clock::now() < deadline) {}
                result = control.abandon(std::chrono::seconds(5));
            });
            engine.stop();
            interrupt.join();
        }
        if (result != Abandon::Unsaved || !unchanged()) {
            std::cerr << "persist: a save abandoned while writing left its file or replaced the "
                         "previous one\n";
            ++failures;
        }
    }
    {
        ninfer::EngineOptions early = options("persist-test");
        const ninfer::PrefixCacheSaveControl control = early.context_cache.hybrid.persistent_save;
        Abandon result = Abandon::StillWriting;
        {
            ninfer::Engine engine(std::move(early));
            (void)engine.generate(engine.prepare_tokens(first), greedy(8));
            result = control.abandon(std::chrono::milliseconds(0));
        }
        if (result != Abandon::Unsaved || !unchanged()) {
            std::cerr << "persist: a save abandoned before it began still wrote a file\n";
            ++failures;
        }
    }
    std::filesystem::remove(file, ignored);
    std::filesystem::remove(partial, ignored);
    return failures;
}

// Two requests submitted together share a prefix nothing has cached yet. The second waits for
// the first's snapshot where the prompts diverge and resumes from it instead of prefilling the
// prefix again (spec §12.2).
// - On a chunk boundary both runs compute the same chunks, and the leader stops after its first
//   token so the follower never shares a decode round: the follower's output must equal an
//   uncached run's.
// - Inside a block, the snapshot goes on the block boundary below, where it publishes at once.
int exercise_coalesce(const char* artifact, std::uint32_t shared_tokens, bool compare_output) {
    const std::vector<ninfer::TokenId> shared        = synthetic_tokens(shared_tokens, 6);
    std::vector<ninfer::TokenId> first               = shared;
    std::vector<ninfer::TokenId> second              = shared;
    const std::vector<ninfer::TokenId> first_suffix  = synthetic_tokens(40, 7);
    const std::vector<ninfer::TokenId> second_suffix = synthetic_tokens(40, 8);
    first.insert(first.end(), first_suffix.begin(), first_suffix.end());
    second.insert(second.end(), second_suffix.begin(), second_suffix.end());
    const std::uint32_t expected = shared_tokens / kBlock * kBlock;

    ninfer::EngineOptions options =
        hybrid_options(artifact, ninfer::SpeculativeBackend::None, 16384, 1ULL << 30, 8);
    options.max_concurrency = 2;
    // The follower is admitted only while the leader still prefills.
    options.concurrent_prefill = true;
    std::vector<ninfer::TokenId> reference;
    if (compare_output) {
        ninfer::Engine engine(options);
        reference = engine.generate(engine.prepare_tokens(second), greedy(16)).generated_token_ids;
    }
    ninfer::Engine engine(options);
    ninfer::GenerationHandle leader    = engine.submit(engine.prepare_tokens(first), greedy(1));
    ninfer::GenerationHandle follower  = engine.submit(engine.prepare_tokens(second), greedy(16));
    const ninfer::GenerationResult led = leader.wait();
    const ninfer::GenerationResult followed = follower.wait();
    int failures                            = 0;
    if (led.reused_prompt_tokens != 0 || followed.reused_prompt_tokens != expected) {
        std::cerr << "coalesce " << shared_tokens << ": the leader reused "
                  << led.reused_prompt_tokens << " tokens and the follower "
                  << followed.reused_prompt_tokens << " (expected 0 and " << expected << ")\n";
        ++failures;
    }
    if (compare_output && followed.generated_token_ids != reference) {
        std::cerr << "coalesce " << shared_tokens
                  << ": the follower generated different tokens than an uncached run\n";
        ++failures;
    }
    return failures;
}

// Two requests submitted together with unrelated prompts: the second is admitted between the
// first one's prefill chunks, so their chunks interleave on shared execution scalars. Each must
// generate what it generates alone. The second stops after its first token so the first decodes
// alone, as in its reference.
int exercise_interleaved(const char* artifact) {
    const std::vector<ninfer::TokenId> first  = synthetic_tokens(3 * kPrefillChunk + 100, 9);
    const std::vector<ninfer::TokenId> second = synthetic_tokens(3 * kPrefillChunk + 60, 10);
    ninfer::EngineOptions options =
        hybrid_options(artifact, ninfer::SpeculativeBackend::None, 16384, 1ULL << 30, 8);
    options.max_concurrency    = 2;
    options.concurrent_prefill = true;
    std::vector<ninfer::TokenId> first_alone;
    std::vector<ninfer::TokenId> second_alone;
    {
        ninfer::Engine engine(options);
        first_alone = engine.generate(engine.prepare_tokens(first), greedy(16)).generated_token_ids;
        second_alone =
            engine.generate(engine.prepare_tokens(second), greedy(1)).generated_token_ids;
    }
    ninfer::Engine engine(options);
    ninfer::GenerationHandle a = engine.submit(engine.prepare_tokens(first), greedy(16));
    ninfer::GenerationHandle b = engine.submit(engine.prepare_tokens(second), greedy(1));
    const ninfer::GenerationResult first_together  = a.wait();
    const ninfer::GenerationResult second_together = b.wait();
    int failures                                   = 0;
    if (first_together.generated_token_ids != first_alone ||
        second_together.generated_token_ids != second_alone) {
        std::cerr
            << "interleaved: concurrent prefills generated different tokens than alone (first "
            << (first_together.generated_token_ids == first_alone ? "same" : "differs")
            << ", second "
            << (second_together.generated_token_ids == second_alone ? "same" : "differs") << ")\n";
        ++failures;
    }
    return failures;
}

ninfer::ChatMessage text_message(ninfer::ChatRole role, std::string text) {
    ninfer::ChatMessage message;
    message.role = role;
    message.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = std::move(text), .media = {}});
    return message;
}

// Protocol adapters shape the hints the way OpenAI default caching does: the Engine's automatic
// Legacy shared-prefix candidates are disabled and an automatic marker follows the last message.
// Hybrid taps must keep their semantic placement under those hints.
bool protocol_hints = false;

ninfer::PromptInput conversation(std::vector<ninfer::ChatMessage> messages) {
    ninfer::PromptInput input;
    input.messages                = std::move(messages);
    input.options.enable_thinking = false;
    if (protocol_hints) {
        input.context_cache.allow_engine_automatic_shared_prefixes = false;
        input.context_cache.markers.push_back(ninfer::PromptCacheMarker{
            .after_message_count = static_cast<std::uint32_t>(input.messages.size()),
            .kind                = ninfer::PromptCacheMarkerKind::SharedStablePrefix,
            .evidence            = ninfer::SharedCandidateEvidence::DefaultAutomatic,
            .location            = ninfer::PromptCacheMarkerLocation::MessageBoundary,
        });
    }
    return input;
}

// Exact semantic taps: the next turn of a conversation resumes at the previous turn's generation
// opener (not 64-token-floored), and a new conversation with the same leading system block resumes
// at that block's end.
int exercise_turns(const char* artifact, std::size_t host_bytes) {
    ninfer::EngineOptions options =
        hybrid_options(artifact, ninfer::SpeculativeBackend::None, 8192, host_bytes, 0);
    options.context_cache.hybrid.device_snapshot_slots.reset();
    ninfer::Engine engine(std::move(options));

    std::string system = "You are a careful assistant for a small engineering team.";
    for (int sentence = 0; sentence < 60; ++sentence) {
        system += " Rule " + std::to_string(sentence) +
                  ": answer precisely, cite the relevant component, and keep replies short.";
    }
    const ninfer::ChatMessage leading = text_message(ninfer::ChatRole::System, system);
    const ninfer::ChatMessage first_user =
        text_message(ninfer::ChatRole::User, "Name three prime numbers.");

    const ninfer::GenerationResult first =
        engine.generate(engine.prepare(conversation({leading, first_user})), greedy(24));
    const ninfer::GenerationResult second = engine.generate(
        engine.prepare(conversation({leading, first_user,
                                     text_message(ninfer::ChatRole::Assistant, first.content),
                                     text_message(ninfer::ChatRole::User, "Now name two more.")})),
        greedy(8));
    const ninfer::GenerationResult other = engine.generate(
        engine.prepare(
            conversation({leading, text_message(ninfer::ChatRole::User, "What is a hash table?")})),
        greedy(8));
    const std::uint32_t other_user_tokens =
        engine
            .prepare(conversation({text_message(ninfer::ChatRole::User, "What is a hash table?")}))
            .summary()
            .prompt_tokens;

    int failures = 0;
    // The next turn resumes at the first turn's generation opener, a few tokens before its end, or
    // at the earliest boundary of the opener's cluster (the system-block end after a short user
    // turn), at most the 64-token tap separation earlier.
    if (second.reused_prompt_tokens + 64U + 16U < first.prompt.prompt_tokens ||
        second.reused_prompt_tokens >= second.prompt.prompt_tokens) {
        std::cerr << "turns: second turn reused " << second.reused_prompt_tokens
                  << " of the first turn's " << first.prompt.prompt_tokens
                  << " prompt tokens; expected the generation opener\n";
        ++failures;
    }
    // The shared system block ends where the new user turn begins.
    if (other.reused_prompt_tokens + other_user_tokens + 16U < other.prompt.prompt_tokens) {
        std::cerr << "turns: a new conversation reused " << other.reused_prompt_tokens << " of "
                  << other.prompt.prompt_tokens << " tokens; expected the system block end\n";
        ++failures;
    }
    const ninfer::RuntimeStats stats = engine.runtime_stats();
    if (stats.hybrid_taps_created == 0 || stats.hybrid_snapshot_hits < 2) {
        std::cerr << "turns: taps=" << stats.hybrid_taps_created
                  << " hits=" << stats.hybrid_snapshot_hits << '\n';
        ++failures;
    }
    return failures;
}

// Typed decision reads (src/serve/systemone.cpp) without the drafter: one-token reads of one
// state, Disposable, with explicit boundaries after the system turn and after the state part, on
// an Engine drafting with `backend`. The drafter never changes what the target computes, so the
// reads' first tokens and log probabilities equal those of drafting reads, bit for bit, and they
// resume where drafting reads resume. The state snapshot the first read leaves keeps the drafter's
// state: a drafting chat over the same state then drafts and accepts exactly as it does after
// drafting reads. A drafting-free request for more than one token is refused.
int exercise_drafting_off(const char* artifact, ninfer::SpeculativeBackend backend) {
    const char* name = backend == ninfer::SpeculativeBackend::Mtp ? "drafting-off-mtp"
                                                                  : "drafting-off-dflash2";
    const ninfer::ChatMessage system = text_message(
        ninfer::ChatRole::System,
        "You answer typed questions about a state with one label only: the letter of the best "
        "option, yes or no, or the number of the level. No words before or after the label.");
    std::string state = "State:\n";
    for (int line = 0; line < 24; ++line) {
        state += "Entry " + std::to_string(line) + ": the parcel passed hub " +
                 std::to_string(line * 7 % 13) + " and waited " + std::to_string(line % 5 + 1) +
                 " hours.\n";
    }
    std::string long_question = "Question: Which hub held the parcel longest? Options:\n";
    for (int hub = 0; hub < 13; ++hub) {
        long_question += static_cast<char>('A' + hub);
        long_question += ". hub " + std::to_string(hub) + "\n";
    }
    long_question += "Reply with a single letter and nothing else.";
    const std::array<std::string, 3> questions{
        "Question: Did the parcel pass hub 3? Answer with one word, yes or no, in lower case.",
        long_question,
        "Question: How many hours did it wait at most? Reply with a single number."};
    const auto prompt = [&](const std::string& question, bool read) {
        ninfer::ChatMessage user = text_message(ninfer::ChatRole::User, state);
        user.parts.push_back(ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Text, .text = question, .media = {}});
        ninfer::PromptInput input;
        input.messages                = {system, user};
        input.options.enable_thinking = false;
        if (read) {
            input.context_cache.retention            = ninfer::CacheRetentionHint::Disposable;
            input.context_cache.update_session_index = false;
        }
        input.context_cache.markers = {
            ninfer::PromptCacheMarker{
                .after_message_count = 1,
                .kind                = ninfer::PromptCacheMarkerKind::SharedStablePrefix,
                .evidence            = ninfer::SharedCandidateEvidence::ExplicitBoundary,
                .location            = ninfer::PromptCacheMarkerLocation::MessageBoundary},
            ninfer::PromptCacheMarker{
                .after_message_count      = 2,
                .kind                     = ninfer::PromptCacheMarkerKind::SharedStablePrefix,
                .evidence                 = ninfer::SharedCandidateEvidence::ExplicitBoundary,
                .location                 = ninfer::PromptCacheMarkerLocation::MessagePartBoundary,
                .after_message_part_count = 1}};
        return input;
    };
    const auto read_options = [](bool drafting, std::uint32_t outputs) {
        ninfer::RequestOptions options          = greedy(outputs);
        options.execution.logprobs              = true;
        options.execution.speculative_drafting = drafting;
        return options;
    };
    struct Run {
        std::vector<ninfer::TokenLogprob> reads;
        std::vector<std::uint32_t> reused;
        ninfer::GenerationResult chat;
    };
    const auto run = [&](bool drafting) {
        ninfer::Engine engine(hybrid_options(artifact, backend, 16384, 1ULL << 30, 8));
        Run out;
        for (const std::string& question : questions) {
            const ninfer::GenerationResult result =
                engine.generate(engine.prepare(prompt(question, true)), read_options(drafting, 1));
            out.reads.push_back(result.content_logprobs.at(0));
            out.reused.push_back(result.reused_prompt_tokens);
        }
        out.chat = engine.generate(
            engine.prepare(prompt("Summarize the parcel's route in two sentences.", false)),
            greedy(32));
        return out;
    };
    const Run drafting = run(true);
    const Run off      = run(false);

    int failures = 0;
    for (std::size_t read = 0; read < questions.size(); ++read) {
        const ninfer::TokenLogprob& a = drafting.reads[read];
        const ninfer::TokenLogprob& b = off.reads[read];
        if (a.id != b.id || a.logprob != b.logprob || a.top_ids != b.top_ids ||
            a.top_values != b.top_values) {
            std::cerr << name << ": read " << read
                      << " differs from the drafting read (token " << a.id << " vs " << b.id
                      << ")\n";
            ++failures;
        }
    }
    if (off.reused != drafting.reused || off.reused[1] == 0 || off.reused[2] != off.reused[1] ||
        off.reused[0] >= off.reused[1]) {
        std::cerr << name << ": reads reused " << off.reused[0] << '/' << off.reused[1] << '/'
                  << off.reused[2] << " tokens, drafting reads " << drafting.reused[0] << '/'
                  << drafting.reused[1] << '/' << drafting.reused[2]
                  << "; expected the later reads at the state boundary\n";
        ++failures;
    }
    const ninfer::SpeculativeStats& a = drafting.chat.speculative;
    const ninfer::SpeculativeStats& b = off.chat.speculative;
    if (off.chat.generated_token_ids != drafting.chat.generated_token_ids ||
        off.chat.reused_prompt_tokens != drafting.chat.reused_prompt_tokens ||
        off.chat.reused_prompt_tokens < off.reused[1] || b.drafted_tokens == 0 ||
        a.rounds != b.rounds || a.drafted_tokens != b.drafted_tokens ||
        a.accepted_tokens != b.accepted_tokens ||
        a.accepted_per_position != b.accepted_per_position) {
        std::cerr << name << ": the chat after the reads reused " << off.chat.reused_prompt_tokens
                  << " tokens and accepted " << b.accepted_tokens << " of " << b.drafted_tokens
                  << " drafts in " << b.rounds << " rounds; after drafting reads "
                  << drafting.chat.reused_prompt_tokens << ", " << a.accepted_tokens << " of "
                  << a.drafted_tokens << " in " << a.rounds
                  << (off.chat.generated_token_ids == drafting.chat.generated_token_ids
                          ? " (same tokens)\n"
                          : " (different tokens)\n");
        ++failures;
    }
    bool refused = false;
    try {
        ninfer::Engine engine(hybrid_options(artifact, backend, 16384, 1ULL << 30, 8));
        (void)engine.generate(engine.prepare(prompt(questions[0], true)), read_options(false, 2));
    } catch (const std::exception&) { refused = true; }
    if (!refused) {
        std::cerr << name << ": a drafting-free request for two tokens was accepted\n";
        ++failures;
    }
    return failures;
}

// Typed decision reads submitted as one group: their last prompt chunks (the questions, after the
// cached state boundary) run in one shared prefill pass, admitted while the first one's last
// chunk is staged (without concurrent prefill too) or with concurrent prefill (the MTP Engine, its
// reads drafting-free). Each read must produce what it produces alone: the same first token and
// reuse, its log probability and readout values within the rounding the wider projections allow.
int exercise_shared_prefill(const char* artifact, ninfer::SpeculativeBackend backend) {
    const char* name =
        backend == ninfer::SpeculativeBackend::Mtp ? "shared-prefill-mtp" : "shared-prefill";
    const ninfer::ChatMessage system = text_message(
        ninfer::ChatRole::System,
        "You answer typed questions about a state with one label only. No words before or after "
        "the label.");
    std::string state = "State:\n";
    for (int line = 0; line < 20; ++line) {
        state += "Entry " + std::to_string(line) + ": the parcel passed hub " +
                 std::to_string(line * 5 % 11) + " and waited " + std::to_string(line % 4 + 1) +
                 " hours.\n";
    }
    const std::array<std::string, 4> questions{
        "Question: Did the parcel pass hub 3? Answer yes or no.",
        "Question: Did the parcel pass hub 12? Answer yes or no.",
        "Question: Which hub came first? Options:\nA. hub 0\nB. hub 5\nC. hub 10\nReply with "
        "one letter.",
        "Question: How many hours did the parcel wait at most? Reply with a single number."};
    const auto prompt = [&](const std::string& question) {
        ninfer::ChatMessage user = text_message(ninfer::ChatRole::User, state);
        user.parts.push_back(ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Text, .text = question, .media = {}});
        ninfer::PromptInput input;
        input.messages                                   = {system, user};
        input.options.enable_thinking                    = false;
        input.context_cache.retention                    = ninfer::CacheRetentionHint::Disposable;
        input.context_cache.update_session_index         = false;
        input.context_cache.markers = {
            ninfer::PromptCacheMarker{
                .after_message_count = 1,
                .kind                = ninfer::PromptCacheMarkerKind::SharedStablePrefix,
                .evidence            = ninfer::SharedCandidateEvidence::ExplicitBoundary,
                .location            = ninfer::PromptCacheMarkerLocation::MessageBoundary},
            ninfer::PromptCacheMarker{
                .after_message_count      = 2,
                .kind                     = ninfer::PromptCacheMarkerKind::SharedStablePrefix,
                .evidence                 = ninfer::SharedCandidateEvidence::ExplicitBoundary,
                .location                 = ninfer::PromptCacheMarkerLocation::MessagePartBoundary,
                .after_message_part_count = 1}};
        return input;
    };
    ninfer::RequestOptions read          = greedy(1);
    read.execution.logprobs              = true;
    read.execution.speculative_drafting = false;
    read.execution.readout_tokens        = {9, 13, 220, 1000, 4000};
    ninfer::EngineOptions options = hybrid_options(artifact, backend, 16384, 1ULL << 30, 8);
    options.max_concurrency       = 4;
    options.max_pending_requests  = 8;
    options.concurrent_prefill    = backend == ninfer::SpeculativeBackend::Mtp;

    struct Read {
        ninfer::TokenLogprob logprob;
        std::uint32_t reused = 0;
    };
    const auto first_read = [&](const ninfer::GenerationResult& result) {
        return Read{.logprob = result.content_logprobs.at(0),
                    .reused  = result.reused_prompt_tokens};
    };
    // Alone: the first read leaves the state boundary, then each read runs by itself.
    std::vector<Read> alone;
    {
        ninfer::Engine engine(options);
        (void)engine.generate(engine.prepare(prompt(questions[0])), read);
        for (const std::string& question : questions) {
            alone.push_back(first_read(engine.generate(engine.prepare(prompt(question)), read)));
        }
    }
    // Together: after the same first read, the reads are submitted as one group.
    std::vector<Read> together;
    ninfer::RuntimeStats before;
    ninfer::RuntimeStats after;
    {
        ninfer::Engine engine(options);
        (void)engine.generate(engine.prepare(prompt(questions[0])), read);
        before = engine.runtime_stats();
        std::vector<ninfer::GenerationSubmission> group;
        for (const std::string& question : questions) {
            group.push_back(ninfer::GenerationSubmission{.prompt  = engine.prepare(prompt(question)),
                                                         .options = read});
        }
        for (ninfer::GenerationHandle& handle : engine.submit_group(std::move(group))) {
            together.push_back(first_read(handle.wait()));
        }
        after = engine.runtime_stats();
    }

    int failures = 0;
    const std::uint64_t passes   = after.shared_prefill_passes - before.shared_prefill_passes;
    const std::uint64_t requests = after.shared_prefill_requests - before.shared_prefill_requests;
    if (passes != 1 || requests != questions.size()) {
        std::cerr << name << ": the group shared " << passes << " passes (" << requests
                  << " requests); expected one pass of " << questions.size() << '\n';
        ++failures;
    }
    // The pass's wider projections round differently and the head's logits are BF16: a value may
    // move by one BF16 step of its logit (0.125 between 16 and 32, 0.25 up to 64; measured on an
    // RTX 3090: at most 0.125).
    constexpr float kTolerance = 0.26F;
    for (std::size_t index = 0; index < questions.size(); ++index) {
        const Read& a = alone[index];
        const Read& b = together[index];
        float worst   = std::fabs(a.logprob.logprob - b.logprob.logprob);
        bool shaped   = a.logprob.readout.size() == read.execution.readout_tokens.size() &&
                      b.logprob.readout.size() == a.logprob.readout.size();
        for (std::size_t k = 0; shaped && k < a.logprob.readout.size(); ++k) {
            worst = std::max(worst, std::fabs(a.logprob.readout[k] - b.logprob.readout[k]));
        }
        std::cerr << name << ": read " << index << " token " << b.logprob.id << " reused "
                  << b.reused << ", largest log probability change " << worst << '\n';
        if (a.logprob.id != b.logprob.id || a.reused != b.reused || !shaped ||
            !(worst <= kTolerance)) {
            std::cerr << name << ": read " << index << " together gave token " << b.logprob.id
                      << " (logprob " << b.logprob.logprob << ", reused " << b.reused
                      << "), alone token " << a.logprob.id << " (logprob " << a.logprob.logprob
                      << ", reused " << a.reused << ")\n";
            ++failures;
        }
    }
    return failures;
}

std::vector<std::uint8_t> gradient_ppm(int size, int phase) {
    std::vector<std::uint8_t> ppm;
    const std::string header =
        "P6\n" + std::to_string(size) + ' ' + std::to_string(size) + "\n255\n";
    ppm.insert(ppm.end(), header.begin(), header.end());
    for (int index = 0; index < size * size; ++index) {
        ppm.push_back(static_cast<std::uint8_t>((index + phase) & 0xff));
        ppm.push_back(static_cast<std::uint8_t>(((index / size) * 3 + phase) & 0xff));
        ppm.push_back(static_cast<std::uint8_t>((index * 7 + phase * 5) & 0xff));
    }
    return ppm;
}

// ---- disk tier scenarios -------------------------------------------------------------------------

// A per-scenario scratch directory under /tmp; wiped so reruns start from an empty catalog.
std::filesystem::path disk_tier_dir(const char* scenario) {
    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() / ("ninfer-hybrid-disk-" + std::string(scenario));
    std::error_code ignored;
    std::filesystem::remove_all(dir, ignored);
    std::filesystem::create_directories(dir);
    return dir;
}

// Host pressure demotes cold snapshot closures to the catalog; the blocks keep serving matches
// from their disk copies and generation stays token-identical to a no-disk reference engine.
int exercise_disk_demotion(const char* artifact) {
    const std::filesystem::path dir = disk_tier_dir("demotion");
    std::vector<ninfer::TokenId> first = synthetic_tokens(1500, 11);
    std::vector<ninfer::TokenId> second(first.begin(), first.begin() + 1400);
    const std::vector<ninfer::TokenId> suffix = synthetic_tokens(300, 12);
    second.insert(second.end(), suffix.begin(), suffix.end());
    const std::vector<ninfer::TokenId> pressure  = synthetic_tokens(3900, 13);
    const std::vector<ninfer::TokenId> pressure2 = synthetic_tokens(3900, 34);
    const std::vector<ninfer::TokenId> pressure3 = synthetic_tokens(3900, 35);
    const auto options = [&](std::size_t host_bytes, bool disk) {
        ninfer::EngineOptions result =
            hybrid_options(artifact, ninfer::SpeculativeBackend::None, 16384, host_bytes, 8);
        if (disk) {
            result.context_cache.disk_kv_path          = dir;
            result.context_cache.disk_kv_capacity_bytes = 1ULL << 30;
        }
        return result;
    };
    const auto run = [&](ninfer::EngineOptions opts) {
        ninfer::Engine engine(std::move(opts));
        (void)engine.generate(engine.prepare_tokens(first), greedy(8));
        (void)engine.generate(engine.prepare_tokens(pressure), greedy(4));
        (void)engine.generate(engine.prepare_tokens(pressure2), greedy(4));
        (void)engine.generate(engine.prepare_tokens(pressure3), greedy(4));
        const ninfer::GenerationResult result =
            engine.generate(engine.prepare_tokens(second), greedy(24));
        return Observed{result.generated_token_ids, result.reused_prompt_tokens,
                        engine.runtime_stats()};
    };
    const Observed reference = run(options(1ULL << 30, false));
    // The advertised Host budget is split between the disk tier's pinned staging and the slab
    // pool (docs/maintainer/hybrid-prefix-cache-disk-tier.md §3), and the staging takes the
    // surplus above the slab pool's own minimum geometry. The slab pool therefore lands on its
    // minimum — one snapshot plus a couple of slabs — so the pressure lineages must overflow it
    // and drive demotion, while the no-disk reference keeps its larger pool.
    const Observed disk = run(options(1ULL << 30, true));

    int failures = 0;
    if (disk.stats.hybrid_disk_demoted_blocks == 0 || disk.stats.hybrid_disk_demoted_images == 0) {
        std::cerr << "disk-demotion: no complete disk closure was published (blocks="
                  << disk.stats.hybrid_disk_demoted_blocks << " images="
                  << disk.stats.hybrid_disk_demoted_images << ")\n";
        ++failures;
    }
    if (disk.stats.hybrid_disk_write_bytes == 0) {
        std::cerr << "disk-demotion: the catalog saw no bytes\n";
        ++failures;
    }
    if (disk.reused != reference.reused) {
        std::cerr << "disk-demotion: reuse changed with the disk tier ("
                  << disk.reused << " vs " << reference.reused << ")\n";
        ++failures;
    }
    if (disk.tokens != reference.tokens) {
        std::cerr << "disk-demotion: generation diverged from the no-disk reference\n";
        ++failures;
    }
    return failures;
}

// A request reusing a demoted (disk-only) prefix restores through the tier's staging: the same
// bytes as a Host restore, so greedy generation stays identical and reuse is preserved.
int exercise_disk_restore(const char* artifact) {
    const std::filesystem::path dir = disk_tier_dir("restore");
    const std::vector<ninfer::TokenId> first = synthetic_tokens(1500, 21);
    std::vector<ninfer::TokenId> second(first.begin(), first.begin() + 1400);
    const std::vector<ninfer::TokenId> suffix = synthetic_tokens(300, 22);
    second.insert(second.end(), suffix.begin(), suffix.end());

    // The Device pool stays tight so the pressure lineages evict the first prompt's Device
    // blocks; the Host budget covers the tier's pinned staging and the slab pool's minimum, so
    // the pressure lineages' write-through overflows it and the first prompt's Host copies are
    // demoted while its committed disk records keep the prefix restorable. The reusing prompt
    // restores through staged disk bytes.
    ninfer::EngineOptions opts =
        hybrid_options(artifact, ninfer::SpeculativeBackend::None, 7168, 1ULL << 30, 2);
    opts.context_cache.disk_kv_path           = dir;
    opts.context_cache.disk_kv_capacity_bytes = 1ULL << 30;
    ninfer::Engine engine(std::move(opts));
    (void)engine.generate(engine.prepare_tokens(first), greedy(8));
    // Three distinct legal pressure lineages (prompt + generation stay inside max_context)
    // accumulate past the Device pool, evicting the first prompt's copies tier by tier.
    (void)engine.generate(engine.prepare_tokens(synthetic_tokens(3800, 23)), greedy(296));
    (void)engine.generate(engine.prepare_tokens(synthetic_tokens(1024, 24)), greedy(296));
    (void)engine.generate(engine.prepare_tokens(synthetic_tokens(3800, 25)), greedy(296));
    // A re-issue of the original prompt drives the Engine's poll while the demotion closure
    // is still in flight (an idle Engine does not poll), so by the next admission the closure
    // has committed and released the Host copy.
    (void)engine.generate(engine.prepare_tokens(first), greedy(512));
    // The warm attempt stages the disk-only blocks for the reusing prefix; admission retries
    // while the staging reads land, and the measured attempt restores through them. Both send
    // the reusing prompt, so the candidate is the demoted terminal snapshot itself.
    (void)engine.generate(engine.prepare_tokens(second), greedy(8));
    const ninfer::GenerationResult result =
        engine.generate(engine.prepare_tokens(second), greedy(24));
    const ninfer::RuntimeStats stats = engine.runtime_stats();

    int failures = 0;
    if (stats.hybrid_disk_demoted_blocks == 0 || stats.hybrid_disk_demoted_images == 0) {
        std::cerr << "disk-restore: no complete disk closure was published\n";
        ++failures;
        return failures;
    }
    if (result.reused_prompt_tokens < 2 * kPrefillChunk) {
        std::cerr << "disk-restore: reuse=" << result.reused_prompt_tokens
                  << " lost the demoted prefix\n";
        ++failures;
    }
    if (stats.hybrid_disk_read_bytes == 0) {
        std::cerr << "disk-restore: no staged disk source was read\n";
        ++failures;
    }
    return failures;
}

// Engine A demotes and exits; engine B reopens the same directory and reconstructs the radix
// tree from the catalog, serving the same reuse without a prefill of the shared prefix.
int exercise_disk_restart(const char* artifact) {
    const std::filesystem::path dir = disk_tier_dir("restart");
    const std::vector<ninfer::TokenId> first = synthetic_tokens(1500, 31);
    std::vector<ninfer::TokenId> second(first.begin(), first.begin() + 1400);
    const std::vector<ninfer::TokenId> suffix = synthetic_tokens(300, 32);
    second.insert(second.end(), suffix.begin(), suffix.end());
    const std::vector<ninfer::TokenId> pressure = synthetic_tokens(3900, 33);
    const std::vector<ninfer::TokenId> pressure2 = synthetic_tokens(3900, 36);
    const auto options = [&] {
        ninfer::EngineOptions result =
            hybrid_options(artifact, ninfer::SpeculativeBackend::None, 16384, 1ULL << 30, 8);
        result.context_cache.disk_kv_path           = dir;
        result.context_cache.disk_kv_capacity_bytes = 1ULL << 30;
        return result;
    };

    std::vector<ninfer::TokenId> reference;
    std::vector<ninfer::TokenId> first_reference;
    {
        ninfer::Engine saver(options());
        first_reference =
            saver.generate(saver.prepare_tokens(first), greedy(8)).generated_token_ids;
        (void)saver.generate(saver.prepare_tokens(pressure), greedy(4));
        (void)saver.generate(saver.prepare_tokens(pressure2), greedy(4));
        reference = saver.generate(saver.prepare_tokens(second), greedy(24)).generated_token_ids;
        const ninfer::RuntimeStats before = saver.runtime_stats();
        if (before.hybrid_disk_demoted_blocks == 0 || before.hybrid_disk_demoted_images == 0) {
            std::cerr << "disk-restart: no complete disk closure was published\n";
            return 1;
        }
    }
    ninfer::Engine loader(options());
    // The first attempt stages the disk-only sources (blocks, image, tail); admission retries
    // the FIFO head while the staging reads land, and the second attempt consumes them.
    const std::vector<ninfer::TokenId> first_loaded =
        loader.generate(loader.prepare_tokens(first), greedy(8)).generated_token_ids;
    const ninfer::RuntimeStats first_stats = loader.runtime_stats();
    const ninfer::GenerationResult result =
        loader.generate(loader.prepare_tokens(second), greedy(24));
    const ninfer::RuntimeStats stats = loader.runtime_stats();

    int failures = 0;
    if (stats.hybrid_disk_resident_blocks == 0 || stats.hybrid_snapshots == 0) {
        std::cerr << "disk-restart: the tree was not reconstructed from disk (blocks="
                  << stats.hybrid_disk_resident_blocks << " snapshots="
                  << stats.hybrid_snapshots << ")\n";
        ++failures;
    }
    if (first_stats.hybrid_disk_restored_blocks == 0 || first_stats.hybrid_disk_read_bytes == 0) {
        std::cerr << "disk-restart: the first reuse read no block from disk (blocks="
                  << first_stats.hybrid_disk_restored_blocks << " bytes="
                  << first_stats.hybrid_disk_read_bytes << ")\n";
        ++failures;
    }
    if (first_loaded != first_reference) {
        std::cerr << "disk-restart: the first reuse diverged from the pre-restart reference\n";
        ++failures;
    }
    if (result.reused_prompt_tokens < 2 * kPrefillChunk) {
        std::cerr << "disk-restart: reuse=" << result.reused_prompt_tokens
                  << " did not hit the reconstructed prefix\n";
        ++failures;
    }
    if (result.generated_token_ids != reference) {
        std::cerr << "disk-restart: generation diverged from the pre-restart reference\n";
        ++failures;
    }
    return failures;
}

// A snapshot whose frontier is not a 64-token multiple needs its tail, and a demoted one needs it
// from the disk tier's staging (the Host copy is gone). Resuming one extra token past the
// endpoint requires exactly that tail: the fill must accept the disk-committed tail source.
int exercise_disk_tail_restore(const char* artifact) {
    const std::filesystem::path dir = disk_tier_dir("tail-restore");
    // 1500 tokens: the terminal endpoint carries a 92-token tail.
    const std::vector<ninfer::TokenId> first = synthetic_tokens(1500, 41);
    std::vector<ninfer::TokenId> extended(first);
    extended.push_back(static_cast<ninfer::TokenId>(first.back() + 1U));

    const auto options = [&](bool disk) {
        ninfer::EngineOptions result =
            hybrid_options(artifact, ninfer::SpeculativeBackend::None, 16384, 1ULL << 30, 8);
        if (disk) {
            result.context_cache.disk_kv_path           = dir;
            result.context_cache.disk_kv_capacity_bytes = 1ULL << 30;
        }
        return result;
    };
    // The reference keeps the endpoint's Host copy and resumes over it; the disk engine must
    // restore the same bytes from its committed closure.
    std::vector<ninfer::TokenId> reference;
    std::uint32_t reference_reuse = 0;
    {
        ninfer::Engine engine(options(false));
        (void)engine.generate(engine.prepare_tokens(first), greedy(8));
        const ninfer::GenerationResult result =
            engine.generate(engine.prepare_tokens(extended), greedy(24));
        reference         = result.generated_token_ids;
        reference_reuse   = result.reused_prompt_tokens;
    }
    ninfer::Engine engine(options(true));
    (void)engine.generate(engine.prepare_tokens(first), greedy(8));
    // Push the endpoint's closure to the disk tier and its Host copy out of the slab pool.
    for (std::uint32_t index = 0; index < 4; ++index) {
        if (engine.runtime_stats().hybrid_disk_demoted_images != 0) { break; }
        (void)engine.generate(engine.prepare_tokens(synthetic_tokens(3900, 42 + index)), greedy(4));
    }
    const ninfer::RuntimeStats demoted = engine.runtime_stats();
    const ninfer::GenerationResult result =
        engine.generate(engine.prepare_tokens(extended), greedy(24));
    const ninfer::RuntimeStats stats = engine.runtime_stats();

    int failures = 0;
    if (demoted.hybrid_disk_demoted_blocks == 0 || demoted.hybrid_disk_demoted_images == 0) {
        std::cerr << "disk-tail-restore: no complete disk closure was published\n";
        ++failures;
        return failures;
    }
    // 1408 is the last full block of the shared prefix: reuse past it can only come from the
    // endpoint's tail.
    constexpr std::uint32_t kLastFullBlock = 1408;
    if (result.reused_prompt_tokens <= kLastFullBlock) {
        std::cerr << "disk-tail-restore: reuse=" << result.reused_prompt_tokens
                  << " did not resume over the snapshot tail (reference=" << reference_reuse
                  << ")\n";
        ++failures;
    }
    if (stats.hybrid_disk_read_bytes == 0) {
        std::cerr << "disk-tail-restore: the tail was not read from the disk tier\n";
        ++failures;
    }
    if (result.generated_token_ids != reference) {
        std::cerr << "disk-tail-restore: generation diverged from the no-disk reference\n";
        ++failures;
    }
    return failures;
}

// A small disk capacity (512 MiB) fills its record directories after a few lineages: the bounded
// victim eviction and the reserved-record collection must keep serving instead of failing every
// later demotion, and a restart on the same directory must not resurrect what the running cache
// removed.
int exercise_disk_small_capacity(const char* artifact) {
    const std::filesystem::path dir = disk_tier_dir("small-capacity");
    std::vector<ninfer::TokenId> first = synthetic_tokens(1500, 61);
    std::vector<ninfer::TokenId> second(first.begin(), first.begin() + 1400);
    const std::vector<ninfer::TokenId> suffix = synthetic_tokens(300, 62);
    second.insert(second.end(), suffix.begin(), suffix.end());
    const auto options = [&] {
        ninfer::EngineOptions result =
            hybrid_options(artifact, ninfer::SpeculativeBackend::None, 16384, 1ULL << 30, 8);
        result.context_cache.disk_kv_path = dir;
        // Half the block directory and half the image directory of a 512 MiB tier.
        result.context_cache.disk_kv_capacity_bytes = 512ULL << 20;
        return result;
    };
    std::vector<ninfer::TokenId> expected;
    {
        ninfer::Engine engine(options());
        (void)engine.generate(engine.prepare_tokens(first), greedy(8));
        for (std::uint32_t index = 0; index < 4; ++index) {
            if (engine.runtime_stats().hybrid_disk_demoted_images != 0) { break; }
            (void)engine.generate(engine.prepare_tokens(synthetic_tokens(3900, 63 + index)),
                                  greedy(4));
        }
        expected = engine.generate(engine.prepare_tokens(second), greedy(24)).generated_token_ids;
        const ninfer::RuntimeStats stats = engine.runtime_stats();
        if (stats.hybrid_disk_demoted_blocks == 0 || stats.hybrid_disk_demoted_images == 0) {
            std::cerr << "disk-small-capacity: no complete disk closure was published\n";
            return 1;
        }
        if (stats.hybrid_disk_write_bytes == 0) {
            std::cerr << "disk-small-capacity: the catalog saw no bytes\n";
            return 1;
        }
    }
    // The previous Engine's removals are journaled: the replay must reconstruct a consistent
    // tree (no dangling closure) and serve the same bytes.
    ninfer::Engine reopened(options());
    const ninfer::GenerationResult result =
        reopened.generate(reopened.prepare_tokens(second), greedy(24));
    int failures = 0;
    if (result.generated_token_ids != expected) {
        std::cerr << "disk-small-capacity: the reopened Engine diverged from the reference\n";
        ++failures;
    }
    return failures;
}

// A cold chain longer than any possible staging batch (the tier's block staging is capped below
// 256 blocks) and longer than one prefetch batch (256 blocks) must still be prepared and
// admitted: bounded streaming stages and copies a batch per cycle with progress, instead of
// requesting more than the pool holds and never being admitted.
int exercise_disk_long_chain(const char* artifact) {
    const std::filesystem::path dir = disk_tier_dir("long-chain");
    constexpr std::uint32_t kContext = 20480;
    // 17000 prompt tokens share 16640 with the second prompt: 260 cold blocks, more than the
    // staging cap and more than one prefetch batch.
    std::vector<ninfer::TokenId> first = synthetic_tokens(17000, 71);
    std::vector<ninfer::TokenId> second(first.begin(), first.begin() + 16640);
    const std::vector<ninfer::TokenId> suffix = synthetic_tokens(512, 72);
    second.insert(second.end(), suffix.begin(), suffix.end());
    const auto options = [&](bool disk) {
        ninfer::EngineOptions result =
            hybrid_options(artifact, ninfer::SpeculativeBackend::None, kContext, 1ULL << 30, 8);
        result.max_context = kContext;
        if (disk) {
            result.context_cache.disk_kv_path           = dir;
            result.context_cache.disk_kv_capacity_bytes = 4ULL << 30;
        }
        return result;
    };
    std::vector<ninfer::TokenId> reference;
    {
        ninfer::Engine engine(options(false));
        (void)engine.generate(engine.prepare_tokens(first), greedy(4));
        reference = engine.generate(engine.prepare_tokens(second), greedy(24)).generated_token_ids;
    }
    ninfer::Engine engine(options(true));
    (void)engine.generate(engine.prepare_tokens(first), greedy(4));
    // Distinct long lineages fill the slab pool and shed the shared chain's Device copies, so the
    // chain reaches the disk tier.
    for (std::uint32_t index = 0; index < 2; ++index) {
        (void)engine.generate(engine.prepare_tokens(synthetic_tokens(18000, 73 + index)), greedy(4));
    }
    const ninfer::GenerationResult result =
        engine.generate(engine.prepare_tokens(second), greedy(24));
    const ninfer::RuntimeStats stats = engine.runtime_stats();

    int failures = 0;
    if (stats.hybrid_disk_demoted_blocks == 0 || stats.hybrid_disk_demoted_images == 0) {
        std::cerr << "disk-long-chain: no complete disk closure was published\n";
        ++failures;
        return failures;
    }
    // 16640 shared tokens: the resume point must sit deep in the shared chain, not at a shallow
    // survivor of the pressure.
    if (result.reused_prompt_tokens + 256U < 16640U) {
        std::cerr << "disk-long-chain: reuse=" << result.reused_prompt_tokens
                  << " lost the long shared chain\n";
        ++failures;
    }
    if (result.generated_token_ids != reference) {
        std::cerr << "disk-long-chain: generation diverged from the no-disk reference\n";
        ++failures;
    }
    return failures;
}

// The persistence file and the disk tier are mutually exclusive at the Engine boundary.
int exercise_disk_persist_exclusive(const char* artifact) {
    ninfer::EngineOptions opts =
        hybrid_options(artifact, ninfer::SpeculativeBackend::None, 16384, 1ULL << 30, 8);
    opts.context_cache.disk_kv_path           = std::filesystem::temp_directory_path();
    opts.context_cache.disk_kv_capacity_bytes = 1ULL << 30;
    opts.context_cache.hybrid.persistent_file =
        std::filesystem::temp_directory_path() / "ninfer-disk-exclusive.bin";
    try {
        ninfer::Engine engine(std::move(opts));
    } catch (const std::invalid_argument&) { return 0; }
    std::cerr << "disk-persist-exclusive: the Engine accepted both a persistence file and the "
                 "disk tier\n";
    return 1;
}

// Vision identity: the same image and text resume at the generation opener, while identical text
// tokens with a different image cannot reuse anything from the image onwards (its blocks carry the
// image's content key) and never resume inside the image's span.
int exercise_vision(const char* artifact) {
    ninfer::EngineOptions options =
        hybrid_options(artifact, ninfer::SpeculativeBackend::None, 8192, 1ULL << 30, 0);
    options.context_cache.hybrid.device_snapshot_slots.reset();
    options.enable_vision = true;
    ninfer::Engine engine(std::move(options));

    std::string system = "You describe images for an accessibility service.";
    for (int sentence = 0; sentence < 30; ++sentence) {
        system +=
            " Rule " + std::to_string(sentence) + ": name colours, shapes and layout plainly.";
    }
    const auto prompt = [&](const std::vector<std::uint8_t>& image, const char* question) {
        ninfer::ChatMessage user;
        user.role = ninfer::ChatRole::User;
        ninfer::MessagePart media;
        media.kind              = ninfer::MessagePartKind::Media;
        media.media.kind        = ninfer::MediaKind::Image;
        media.media.bytes       = image;
        media.media.media_type  = "image/x-portable-pixmap";
        media.media.source_name = "image.ppm";
        user.parts.push_back(std::move(media));
        user.parts.push_back(ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Text, .text = question, .media = {}});
        return conversation({text_message(ninfer::ChatRole::System, system), std::move(user)});
    };
    const std::vector<std::uint8_t> first_image  = gradient_ppm(512, 0);
    const std::vector<std::uint8_t> second_image = gradient_ppm(512, 97);

    const ninfer::GenerationResult first =
        engine.generate(engine.prepare(prompt(first_image, "Describe the image.")), greedy(4));
    const ninfer::GenerationResult same =
        engine.generate(engine.prepare(prompt(first_image, "Describe the image.")), greedy(4));
    const ninfer::GenerationResult other =
        engine.generate(engine.prepare(prompt(second_image, "Describe the image.")), greedy(4));

    int failures = 0;
    if (same.reused_prompt_tokens + 64U < first.prompt.prompt_tokens) {
        std::cerr << "vision: the same image and text reused " << same.reused_prompt_tokens
                  << " of " << first.prompt.prompt_tokens << " tokens\n";
        ++failures;
    }
    // A 512x512 image is 256 merged tokens; a different image must lose at least those.
    if (other.prompt.prompt_tokens != first.prompt.prompt_tokens ||
        other.reused_prompt_tokens + 256U > first.prompt.prompt_tokens) {
        std::cerr << "vision: a different image reused " << other.reused_prompt_tokens << " of "
                  << other.prompt.prompt_tokens << " tokens\n";
        ++failures;
    }
    if (first.generated_token_ids.size() != 4 || other.generated_token_ids.size() != 4) {
        std::cerr << "vision: generation did not complete\n";
        ++failures;
    }
    return failures;
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "skip: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }
    if (const char* storage = std::getenv("NINFER_HYBRID_KV_DTYPE");
        storage != nullptr && *storage != '\0' && !select_kv_storage(storage)) {
        std::cerr << "unknown NINFER_HYBRID_KV_DTYPE " << storage << '\n';
        return 1;
    }
    const char* selected            = std::getenv("NINFER_HYBRID_REAL_SCENARIO");
    const std::string_view scenario = selected != nullptr && *selected != '\0' ? selected : "all";
    constexpr std::array<std::string_view, 22> kScenarios{"all",
                                                          "restore-exact",
                                                          "restore-exact-mtp",
                                                          "restore-exact-dflash2",
                                                          "drafting-off-mtp",
                                                          "drafting-off-dflash2",
                                                          "shared-prefill",
                                                          "shared-prefill-mtp",
                                                          "turns",
                                                          "turns-device-only",
                                                          "vision",
                                                          "persist",
                                                          "turns-protocol",
                                                          "coalesce",
                                                          "interleaved",
                                                          "disk-demotion",
                                                          "disk-restore",
                                                          "disk-restart",
                                                          "disk-persist-exclusive",
                                                          "disk-tail-restore",
                                                          "disk-small-capacity",
                                                          "disk-long-chain"};
    if (std::find(kScenarios.begin(), kScenarios.end(), scenario) == kScenarios.end()) {
        std::cerr << "unknown NINFER_HYBRID_REAL_SCENARIO " << scenario << '\n';
        return 1;
    }
    int failures = 0;
    try {
        const bool all = scenario == "all";
        if (all || scenario == "restore-exact") {
            failures += exercise_restore_exact(artifact, ninfer::SpeculativeBackend::None);
        }
        if (all || scenario == "restore-exact-mtp") {
            failures += exercise_restore_exact(artifact, ninfer::SpeculativeBackend::Mtp);
        }
        if (all || scenario == "restore-exact-dflash2") {
            failures += exercise_restore_exact(artifact, ninfer::SpeculativeBackend::DFlash2);
        }
        if (all || scenario == "drafting-off-mtp") {
            failures += exercise_drafting_off(artifact, ninfer::SpeculativeBackend::Mtp);
        }
        if (all || scenario == "drafting-off-dflash2") {
            failures += exercise_drafting_off(artifact, ninfer::SpeculativeBackend::DFlash2);
        }
        if (all || scenario == "shared-prefill") {
            failures += exercise_shared_prefill(artifact, ninfer::SpeculativeBackend::None);
        }
        if (all || scenario == "shared-prefill-mtp") {
            failures += exercise_shared_prefill(artifact, ninfer::SpeculativeBackend::Mtp);
        }
        if (all || scenario == "turns") { failures += exercise_turns(artifact, 1ULL << 30); }
        if (all || scenario == "turns-device-only") { failures += exercise_turns(artifact, 0); }
        if (all || scenario == "vision") { failures += exercise_vision(artifact); }
        if (all || scenario == "persist") { failures += exercise_persist(artifact); }
        if (all || scenario == "interleaved") { failures += exercise_interleaved(artifact); }
        if (all || scenario == "disk-demotion") { failures += exercise_disk_demotion(artifact); }
        if (all || scenario == "disk-restore") { failures += exercise_disk_restore(artifact); }
        if (all || scenario == "disk-restart") { failures += exercise_disk_restart(artifact); }
        if (all || scenario == "disk-persist-exclusive") {
            failures += exercise_disk_persist_exclusive(artifact);
        }
        if (all || scenario == "disk-tail-restore") {
            failures += exercise_disk_tail_restore(artifact);
        }
        if (all || scenario == "disk-small-capacity") {
            failures += exercise_disk_small_capacity(artifact);
        }
        if (all || scenario == "disk-long-chain") {
            failures += exercise_disk_long_chain(artifact);
        }
        if (all || scenario == "coalesce") {
            failures += exercise_coalesce(artifact, 6 * kPrefillChunk, true);
            failures += exercise_coalesce(artifact, 3000, false);
        }
        if (all || scenario == "turns-protocol") {
            protocol_hints = true;
            failures += exercise_turns(artifact, 1ULL << 30);
            protocol_hints = false;
        }
    } catch (const std::exception& error) {
        std::cerr << "hybrid prefix real test failed: " << error.what() << '\n';
        return 1;
    }
    if (failures == 0) { std::cout << "hybrid prefix real tests passed\n"; }
    return failures == 0 ? 0 : 1;
}
