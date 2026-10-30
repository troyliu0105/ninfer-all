# Hybrid prefix cache: disk tier

**Status: implemented.** This document defines the disk-tier extension to the
[Hybrid prefix cache](hybrid-prefix-cache-spec.md). Hybrid mode accepts `--disk-kv-path` with
`--disk-kv-gib` (default 64 GiB) and opens its own catalog under the artifact-scoped directory;
the Legacy-only `--disk-kv-restore` and `--disk-kv-directstorage` switches remain Legacy-only.
Host-only tests cover production demotion, cold-preparation control, staging and completion
orchestration, source leases, bounded repair, branched reclamation, failed-source fallback,
catalog replay and real worker I/O, including blocked-journal isolation. The real-engine
scenarios (`disk-demotion`, `disk-restore`, `disk-restart`, `disk-tail-restore`,
`disk-small-capacity`, `disk-long-chain`) require completed image demotion. `disk-long-chain`
additionally requires the first request after restart to restore more than the maximum staging
batch, without a warm retry. These changed CUDA bindings and scenarios remain GPU-unqualified;
neither numerical restore nor the performance contract (§9) follows from CPU coverage.

The extension retains cold prefixes after the pinned Host cache comes under pressure, serves
requests from Device, Host or Disk copies, and reconstructs the radix tree from committed disk
records after restart. It keeps one logical tree and isolates storage and asynchronous jobs from
model execution so that subsequent upstream rebases have a bounded integration surface.

The Hybrid specification remains the authority for implemented behavior. This document owns the
disk-specific contracts, including the command-line behavior in §8. The
[Engine architecture](engine-architecture.md) and
[paged KV contracts](paged-kv-cache.md) continue to govern request publication and physical pages.

## 1. Scope and decisions

The target is the existing single-device Qwen3.5 Hybrid path, including its supported KV formats,
Vision identity and speculative backends. Disk is a cold prefix-cache tier, not active-request
preemption, expert residency, a second model execution path or a shared cache across Engines.
All model arithmetic stays on the GPU.

The required behavior is:

- Start bounded asynchronous Host-to-Disk demotion before the Host cache exhausts its space.
- Match exact prefixes once, then source each required object from Device, Host or Disk in that
  order. Select a resume point by total predicted remaining time, not by frontier alone.
- Rebuild Disk-resident nodes and complete snapshots after restart without loading all payloads
  into RAM.
- Keep kernels, CUDA Graph keys, prefill decomposition, protocol cache hints and the request
  scheduler unchanged. Confine integration to cache residency, background preparation and final
  materialization.
- Reject simultaneous use of the disk tier and `--prefix-cache-file`.

A Qwen3.5 resume point needs a complete recurrent StateImage. KV alone cannot restore GDN state.
A reusable snapshot therefore requires its full ancestor path, its optional partial-block tail
and its StateImage, including the selected backend's state. Backend page commitment rules,
including MTP's frontier lag, remain unchanged.

### Architecture choice

Correctness, bounded interference and one source of residency truth take precedence over the
smallest initial diff. Rebase cost favors explicit modules and narrow integration points.

| Choice | Integration cost | Ongoing cost | Decision |
|---|---|---|---|
| One radix tree with per-tier copies | Extend residency, eviction and restoration in the existing index | One identity, dependency and pin model; disk mechanics stay in separate modules | Selected |
| Separate hot and cold trees | Preserve more of the two-tier index initially | Merge candidates, deduplicate publications and coordinate migration between two logical indexes | Rejected |

The selected design requires changes to existing two-tier assumptions. It does not introduce a
cache base class, storage-plugin framework or a copy of the Legacy owner/catalog planner.

## 2. Ownership and integration boundaries

| Owner | Responsibility |
|---|---|
| Core storage | Physical records, bounded I/O, payload checksums and read claims |
| `runtime/prefix_cache` | Exact tree identity, snapshot dependencies, per-tier residency and victim selection; `PrefixDiskCatalog` owns persistent metadata and `PrefixDiskWorker` executes runtime storage I/O |
| Qwen3.5 Program / `HybridPrefixCache` / `HybridDiskTier` | Program owns pinned staging and `ColdPreparation`; cache owns Host slabs and CUDA copy events; the CUDA-independent tier borrows staging, requests physical fills and adopts disk completions on the Engine thread |
| Program admission | Prepare bounded cold batches outside the context transaction, then quote ready sources, validate resources, bind prepared data and activate a lane |
| Engine | FIFO, deadlines, cancellation, execution order and publication |

The Engine worker remains the only mutator of the tree and residency state. I/O workers receive
immutable job descriptions and return completion records. They do not release logical pages,
edit nodes or publish snapshots. The worker publishes an atomic completion revision, and the
physical cache separately versions Host/Device copy completions. A queued preparation keeps
Engine progress enabled even with no active lane. The bounded idle cadence observes both:
image/tail availability can change without new disk I/O or more free KV pages. Only readiness
changes re-arm admission; an unchanged poll does not repeat matching/resource inspection or
depend on another request.

Existing code offers these reuse boundaries:

- [`DiskKVStore`](../../src/core/disk_kv_store.h) is CUDA-independent and supplies checksummed
  records and read claims. Its automatic page LRU is unsuitable for live Hybrid dependencies;
  Hybrid needs caller-controlled allocation/reclamation that fails when no reserved space exists.
  Legacy behavior remains separate.
- [`DiskKVBridge`](../../src/core/disk_kv_bridge.h) supplies bounded queued writes and completion
  tickets, but its reads execute on the calling thread and its capacity split is fixed by family.
  Reuse physical mechanisms where appropriate, not this policy as the Hybrid tier policy.
- [`HybridPrefixCache`](../../src/models/qwen3_5/program/prefix/hybrid_cache.h) already owns Host
  slabs and Device transfers. Preserve its existing Host write-through and layer-pipelined
  restore paths.
- [`PrefixCacheIndex`](../../src/runtime/prefix_cache/prefix_index.h) already owns tokens, extra
  keys, references and eviction. Extend this index rather than recreating those facts elsewhere.

`runtime/prefix_cache/disk_catalog.*` owns metadata encoding/replay and pure-memory reservations.
`disk_worker.*` is its sole runtime I/O executor: payload reads/writes, publication journals,
removals and checkpoints. `models/qwen3_5/program/prefix/hybrid_disk.*` owns model binding,
demotion interests and staging leases, not a second storage executor. Startup replay and shutdown
settlement may perform synchronous I/O before/after serving.

`HybridDiskTier::Layout` carries only byte geometry; `Staging` borrows the three accounted
buffers for the tier's full lifetime. `HostTransfers` supplies slab addresses and requests block
or snapshot D2H fills. A successful submission is not completion: the physical owner keeps its
pins and publishes Host residency only when the copy lands. Production callbacks invoke the
existing CUDA paths. CPU tests supply distinct memory buffers and explicitly delayed completion
events, while executing the same tier/index/catalog/worker code, not a second demotion algorithm.
Program destroys the tier and settles its worker before freeing staging or Host source slabs.
Pinned staging also outlives `HybridPrefixCache`: its detached prefetches use a private restore
stream, not the Program's compute/transfer streams. Member destruction therefore stops disk
jobs, synchronizes the cache's CUDA copies, then frees their staging. Synchronizing only the
Program streams does not establish this source-lifetime boundary.

Recovery also settles accepted disk source jobs and CUDA copies before clearing Host slabs,
staging leases or index descriptors. The rebuilt cache reattaches the existing disk tier.
This settlement barrier is confined to failure/rebuild and teardown, never normal admission.

## 3. Residency, identity and capacity

### Copies and validity

Each node keeps an opaque Disk record reference in addition to its Device and Host references.
Snapshot metadata identifies its disk image parts and optional tail. Each location tracks
Absent, Filling or Resident independently; Disk is a location, not a fourth transfer state.

A Filling destination is not an available copy. A resident Disk record has completed its write
and identity/checksum publication; that does not imply power-loss durability. Reads validate the
record before any consumer uses its bytes.

The index distinguishes:

- Dropping one tier's copy while another valid copy remains.
- Removing a logical snapshot and updating its dependencies.
- Losing the last copy of a required object and invalidating affected resume points.

A node with only a Disk copy remains in the tree. A snapshot can resume through mixed residency:
its ancestors, tail and image need not occupy the same tier. Losing a corrupt Disk copy preserves
any intact Device/Host copies. Hashes remain lookup accelerators; exact tokens and extra keys,
including Vision identity, decide equality.

Centralize these predicates and transitions instead of scattering disk-enabled branches through
admission, capture and release. Existing backed/unbacked Device LRU membership must recognize a
valid Disk backup. Supersession and Host pressure must distinguish a tier drop from deliberate
logical deletion; neither may silently delete a recoverable cold entry as a side effect.

### Dependency accounting

Global snapshot reachability and a tier's reclaimable bytes are different facts. A block may be
live because a snapshot needs it but still yield its Host slab because Disk holds a copy.
Maintain enough per-tier dependency information to compute that marginal reclaimable space.
Do not keep a Host slab merely because a disk-backed descendant exists.

A committed disk snapshot is a stronger condition than mixed-tier runtime validity: its entire
path, image and tail must exist on Disk. Shared ancestors occupy one disk payload record.
Incomplete disk closures may exist while jobs run, but cannot advertise restart reuse.

### Budgets

Resolve and report these bounds at startup:

- Cache-owned Device pages and snapshot slots, respecting active execution entitlements.
- Host slabs available for cache storage after reserving disk staging.
- Staging bytes, queued/in-flight bytes and maximum Host bytes leased to offload jobs.
- Disk payload and metadata capacity, including journal/checkpoint and temporary-work headroom.
- In-memory node, snapshot and disk-directory capacity.

Account pinned staging within the advertised Host budget; do not add an unreported second RAM
cache. A conservative staging layout holds one full StateImage and a bounded number of KV copy
batches. Its size depends on model geometry, not prefix length. The remaining Host pool must
still satisfy its minimum geometry. Reject an insufficient configuration before serving.

The implementation reserves that staging out of `--host-cache-mib` *before* it sizes the slab
pool: [`plan_hybrid_disk_staging`](../../src/models/qwen3_5/program/prefix/hybrid_host_layout.cpp)
takes one StateImage in its physical slab segments, one block payload for the snapshot-tail slot,
and up to `kHybridDiskMaxStagingBlocks` (256) block payloads — where the block batch gets only
what the budget leaves after the slab pool's own minimum geometry, so a Host budget that is legal
without the disk tier stays legal with it. The tier pins the result once with `PinnedHostBuffer`
(no `make_unique` side allocation), `staging_bytes()` reports the same rounded bytes, the
`HostKvPin` startup phase's total is the advertised budget (slab pool plus staging), and a budget
that cannot hold a single block payload beside the slab minimum is rejected at startup with the
MiB it needs — the plan's own `min_budget_bytes` (image segments, tail payload, slab minimum and
one block payload), never a re-derived approximation. Disk capacity is not part of the Host
budget: the catalog's byte capacity stays real bytes.

With Disk enabled, ordinary Host block and snapshot fills leave one slab available for a missing
ancestor's Device-to-Host repair. Only disk backfill can consume it. This is a reservation within
the existing slab pool, not another allocation: ordinary hot retention loses one slab of capacity,
while the advertised Host-plus-staging budget and the minimum startup geometry stay the same.

The node and snapshot descriptor bounds derive from the capacity plan in
[`create_hybrid_prefix_cache`](../../src/models/qwen3_5/program/prefix/hybrid_program.cpp):
Device pages plus Host slabs plus, with the tier enabled, the record budgets of
[`plan_hybrid_disk_records`](../../src/models/qwen3_5/program/prefix/hybrid_host_layout.cpp) —
one block record per disk-only node and `image_slabs` image segments per committed closure, the
catalog Store's own arithmetic, with representability (kNoId/2) rejected at startup. A full disk
tier can therefore never exhaust the index: `insert_block`'s capacity invariant remains a canary
for descriptor leaks, not a runtime bound, and every restore or publication path already degrades
(skip or evict) at its own capacity. The descriptors cost host RAM proportional to the tier's
record count (about 0.5 KiB per node slot, tokens and links included). Hot victim selection must
not scan the entire cold catalog as the disk tier grows.

## 4. Host pressure and asynchronous demotion

### Watermarks and victims

Use high/low occupancy watermarks with hysteresis. Start demotion above the high watermark and
stop selecting more work below the low watermark. Resolve the gap from snapshot/copy geometry
and the allowed in-flight bytes so that new Host writes have headroom. Percentages alone do not
protect a small pool that must receive a large StateImage. Qualify the actual thresholds; this
design does not claim a measured optimum.

Reclaim in this order:

1. Discard dead data that no snapshot or pending publication can use.
2. Release cold Host copies that already have valid Disk backups.
3. Schedule demotion of cold, useful data without a Disk copy.

Use existing supersession and GDSF concepts to choose cold candidates, while separating the
value of keeping a Host copy from the value of keeping a Disk copy. Host retention avoids a slow
restore; Disk retention avoids recomputation. The current Host score is not automatically the
right score for both tiers.

A finished block record is already a valid mixed-tier source, even before any descendant image
commits. Each block completion requests release of its Host duplicate immediately, funding the
next missing ancestor. Image/tail duplicates release after the image job commits. Both release
requests defer while another reader or writer holds a source pin and complete at the last
unpin; publication never authorizes reuse of bytes an accepted H2D or disk job still reads.

A full record directory first drains loose records through `release_disk_record_range`. It visits
every branch deepest-first, including orphan suffixes beyond a live shallow snapshot. A record
yields only when no snapshot needs it, no transfer pins it and no child's persistent reference
still needs it. A disk-only dead node retires with its unreachable subtree; valid volatile copies
otherwise keep the node. This prevents abandoned branches from permanently sealing the directory.
The tier then selects committed closure victims with `evict_disk_entries`: superseded closures
first, in lowest-GDSF order, followed by retained ones only when no dead closure remains.
Shared ancestors and mixed-tier runtime dependencies stay protected. A chosen closure may return
more than the nominal record budget because its exclusive chain is reclaimed together.

Runtime pressure never calls `reclaim_abandoned`: queued and running jobs own their reservations.
Only after the I/O worker has stopped may shutdown sweep provisional records. Runtime removals
are asynchronous; a refused reservation retries on a later cadence after the worker has journaled
the removals and refunded capacity. Volatile-tier pressure drops duplicates or refuses retention,
rather than destroying a committed closure.

Every write submission checks the remaining byte bound separately, before taking a reservation.
The default bound is three physical image-and-tail jobs. It covers queued, running and completed
but unadopted writes, including padded image segments and the full tail bundle. Image reservations
charge every image slot and the optional tail slot before execution; refusals consume no ID or
capacity. Demanded reads precede unstarted removals and writes. Writes themselves keep submission
order, so new block streams cannot starve an older image commit and its Host release. No
priority rule preempts I/O that is already running.

There is no closure-wide repair pin. Each accepted block write pins only for that job; waiting
for backfill or queue capacity owns neither an image reservation nor a snapshot pin. Once every
ancestor is disk-Resident, the tier reserves and pins the image/tail job. A bounded set of unpinned
demotion interests lets started closures finish below the low watermark. An image refill that
cannot yet fit keeps its started interest: other jobs may return space only on a later cadence.
Failed or stale fills abort on completion, release their records and retain any valid volatile
source. No accepted job's record is reclaimed early, so late work cannot recreate an abandoned
payload key.

Within each candidate batch, drainable Host-resident images precede complete Device-only images
requiring refill. Each group orders superseded snapshots oldest-first, then retained ones by GDSF.
In-flight, pinned and invalidated snapshots are excluded. A snapshot superseded within the last
two supersession ticks gets a grace window while Host occupancy stays below 95%; above that,
drainage wins. An image/tail is submitted only with its complete Host slab set.

A missing Device-resident ancestor uses the reserved repair slab on the existing write-through
streams. After its D2H event and disk write complete, returning that slab funds the next repair.
There is no need to hold the whole ancestor chain on Host. A complete Device-only image may also
refill on those streams; if it cannot yet fit, its Host-resident ancestors can still stream to
Disk and fund image space. Every candidate handle is revalidated before use, since an allocation
may evict a later candidate. Retention stays best-effort when no valid source or sufficient
capacity remains.

### Write and publication protocol

A demotion plan targets useful snapshot closures but writes independent block bundles and image
parts in bounded batches:

1. Reserve disk space and copy the required immutable identity metadata into the job plan.
2. Lease only the Host payloads the batch reads. Require completed D2H writes before submitting
   those payloads to disk workers.
3. Write missing records, reusing committed ancestors rather than copying the whole prefix for
   each snapshot.
4. The worker finishes the persistent record and returns a completion. The Engine alone publishes
   index residency and releases that job's Host source pin; a pending duplicate release runs only
   when the last source reader has returned.
5. Publish a restartable snapshot record only after its complete Disk closure exists. Protect
   all records required by that commit until publication completes.

Blocks can reach Disk before the associated snapshot closes. This permits a prefix larger than
the Host cache to accumulate on Disk as Host write-through advances. Bound provisional records
and pending metadata references; collect abandoned records when their publication can no longer
complete. Payloads without a committed snapshot never create restart hits by themselves.

Accepted offloads own independent source pins until their completion is adopted. Block jobs use
the index's conservative path-wide pins; image jobs pin their snapshot only while its image/tail
is actually queued or running. No pin spans an entire streaming repair or capacity wait. The
Device-eviction and latency cost of those bounded job pins remains part of GPU qualification.

### Backpressure

A full queue stops new demotion submissions. It does not make Host allocation wait for SSD
completion. If Host pressure outpaces the permitted write rate, discard lower-value cache data
or skip new retention according to policy, while preserving in-flight sources. Active requests
keep priority over retention.

Sustained incoming spill bytes above the allowed disk write rate cannot coexist with bounded
memory and lossless retention. This design chooses bounded memory and best-effort retention.
Normal Host-to-Disk demotion reads existing Host bytes; it adds no model pass or snapshot tap.

## 5. Lookup, cost and cold preparation

### Lookup and source selection

Walk the radix tree once and compare exact identities as today. For each required object, prefer
Device, then Host, then Disk. A complete hot hit performs no filesystem probe, synchronous disk
checksum or wait for a metadata checkpoint. The worker-owned in-memory catalog supplies Disk
references without a lock held across storage operations.

This source order is not a rule to stop at the first shallow hot snapshot. Compare feasible
resume points, including root, by their remaining work:

```text
remaining_time(s) = remaining_preparation_time(s)
                  + prefill_time(frontier(s), remaining_prompt_tokens)
                  + activation_and_first_token_work(s)
```

Account for the disk read queue, outstanding preparation, source bytes, checksum work and H2D.
Use measured end-to-end restore coefficients with margin, not SSD sequential-read specifications
or the Host H2D coefficient. A useful Disk choice must beat the best other source by enough to
cover prediction uncertainty. Predictions rank choices; actual pages, slots and validity decide
whether they can execute.

Quote excludes infeasible candidates before calling `choose`; both quote and prefetch honor its
root result. An absent candidate means root: a fitting cold prefix does not override a cheaper
recomputation decision, and no Disk reads are staged for a root choice.

### Prepare outside the global context transaction

Disk reads and cold preparation run while the FIFO head remains queued. They do not occupy the
Engine's global context transaction or an active lane. Preserve FIFO; do not add backfill or run a
second model prefill in a race against the read.
Hybrid skips speculative backfill inspections altogether: its FIFO contract cannot grant those
candidates admission, and inspection must not start reads or replace the true head's preparation.

This boundary is required by the current Engine. In
[`engine_core.h`](../../src/runtime/engine/engine_core.h), `snapshot_cancellations`,
`settle_terminal_requests` and `cancel_active_requests` defer topology changes while a context
transaction exists. A long-lived disk materialization would delay other requests' cancellation,
completion and resource release even if their decode kernels continued to run.

Cold preparation proceeds as follows:

1. Establish a preparation owned by the FIFO head's publication identity, selected snapshot and
   exact path. Equal prompt lengths are not request or prefix identity. `Preparing` is distinct
   from a capacity-blocked quote: an idle Engine progresses it instead of rejecting overload.
   Disk read leases protect storage records and metadata, not the Device LRU path.
2. Read and verify bounded block jobs into pinned staging. Give demanded reads priority over
   background demotion writes; this implementation does not coalesce separate block records.

   The batch is the staging pool, never the whole cold path: `stage_blocks` queues what the pool
   can hold now and the next cadence point queues more, so a chain longer than the pool (which is
   the normal case for a long conversation) advances a batch per cycle instead of requesting more
   slots than exist and making no progress. Requests are deduplicated against both the tier's
   queued/running/landed reads and the bytes the Program already holds for a landing batch. A
   staging slot recycles only after the copy that consumed it lands — the node is device-Resident
   again, or no batch reads it — and a duplicate release is ignored, so two nodes can never be
   handed the same slot.
   Readiness never requires the whole cold chain to occupy staging simultaneously, and polling
   age never releases an unconsumed payload. `ColdPreparation` is the sole block-staging owner;
   the physical Program callback consumes each ready batch and reports accepted Device fills.
   Each delivered block is revalidated: another consumer's Filling/Resident Device destination
   is not overwritten, and a newly ready Host copy takes the ordinary Host restore path.
3. When bytes are ready and spare cache capacity exists, allocate cache-owned Device
   destinations, submit H2D on the restore stream and mark those destinations Filling.
   No Device destination stays pinned while waiting for an unsubmitted disk read.
4. Publish copies after their CUDA events complete. Completed prefetched entries remain ordinary
   cache and may yield to active growth. Do not hold a whole cold path unevictable throughout a
   slow preparation.
5. Once the path is Device/Host-ready and its image/tail bytes are ready in their ordinary or
   disk-staging locations, request a fresh admission quote. The short transaction revalidates
   sources, reserves the execution entitlement and binds/activates the lane. It waits on no
   unfinished disk read.

The complete prefix need not fit in Host. Batches pass through staging to Device; optional Host
promotion follows normal retention policy rather than becoming a prerequisite for admission.
StateImage staging may hold a whole image so existing image copy primitives remain usable.
Device snapshot slots are acquired only for ready image copies, not while waiting on SSD reads.

Preparation is opportunistic, not a guarantee that a request can acquire its whole Device path
beside active lanes. If capacity is unavailable, leave the head queued, retry when resources
change, or choose another feasible source. No cold reservation may borrow active entitlements.
After any eviction, stale quote or changed source, re-evaluate the ready subset instead of
assuming prefetch reserved a future admission.

A monotonic per-head budget limits waiting: the minimum of the root estimate and feasible
non-disk offer costs, minus the chosen suffix's predicted prefill cost, clamped at zero.
Requoting the same head does not renew this budget. Already-ready sources win at expiry; checking
them may not start another read. Otherwise expiry or a lost required source selects `Fallback`
and excludes still-disk-dependent candidates for that head; a new head gets a new budget.
Cancellation detaches only its matching publication identity. Accepted I/O and CUDA work retain
their buffers until completion, including across replacement by an equal-length request.
Completed fills remain
ordinary evictable cache entries; no whole-path reservation spans a disk wait.
An accepted copy that lands after expiry can still make the queued head ready: `Fallback`
forbids renewed work/waiting, not reuse of completed data. That transition starts no new reads,
does not renew the budget and cannot revive an abandoned head.

### Transfer isolation

Keep disk preparation jobs separate from existing GPU transfer draining. Current `hybrid_quote`
and `hybrid_make_room` can call `HybridPrefixCache::drain()` in
[`hybrid_program.cpp`](../../src/models/qwen3_5/program/prefix/hybrid_program.cpp). Neither
`drain()`, `settle_prefetch()` nor an admission retry may start waiting for disk workers.

Only a batch whose disk bytes are ready may join the bounded CUDA copy lifecycle. Recycle its
staging only after the corresponding event. The compute stream receives no wait on an event
whose production depends on a future disk read. Preserve the existing Host-only layer-pipelined
restore; activation does not require introducing layer-pipelined disk consumption.

### Attribution and interpretation

Request-log schema 30 separates a KV match, a usable snapshot, the selection price and actual
execution. Hybrid materialization diagnostics describe the latest inspection, retained even if
the request is cancelled during preparation; they do not imply admission or executed prefill.

| Field | Meaning |
|---|---|
| `cached_prefix_tokens` | Deepest exact KV-block match, independent of snapshot usability |
| `best_reuse_prompt_tokens` | Deepest valid matched snapshot before feasibility filters |
| `prefix_source_decision` | `none`, `preparing`, `selected`, `root_no_snapshot`, `root_filtered`, or `root_priced` |
| `matched_snapshots`, `eligible_snapshots` | Candidate counts before and after filters |
| `filtered_filling_snapshots`, `filtered_vision_snapshots`, `filtered_capacity_snapshots`, `filtered_preparation_snapshots` | Mutually exclusive first-rejection counts, in that order; the last is exhausted/failed preparation |
| `selection_root_ns`, `selection_best_offered_ns` | Predicted root and cheapest eligible snapshot costs, including the offered price when root wins; use `eligible_snapshots` to distinguish no offer from a zero-cost offer |
| `selected_restore_bytes`, `selected_disk_read_bytes` | Source-work estimates for the selected quote, not completed traffic |

A gap between matched KV and actual reused tokens does not by itself identify snapshot placement,
eviction or a pricing error. The old logs cannot distinguish these causes; identical matched
lengths also do not establish identical prompts. The diagnostics add bounded scalar attribution,
not prompt text or per-token traces, and do not retune selection or retention.

`restored_host_bytes` includes Host/staging-to-Device copies attributed to cold preparation and
admission; `restored_disk_bytes` is the disk-delivered subset. Preparation can precede prefill.
Periodic disk `read_bytes` includes all completed logical payload reads, including image/tail-only
intervals and retries, not physical SSD traffic. Image-only demotion and byte-only reads appear
in the human throughput log even when the whole-block count is zero.

`timings_seconds.ttft` is `null` when no first token was emitted, and a real immediate first token
may have numeric zero. Console TTFT means exclude absent samples. `computed_prefill_tokens`
counts executed work, including a partially prefilled cancellation, rather than the planned
prompt-minus-hit suffix. Qwen3.5 `prefill` is summed pass time, not the entire elapsed interval
from first to last prefill; queue wait and other lanes' gaps must not be inferred from it.
Analysis excludes absent TTFTs from timing means but keeps those requests in request/work totals;
its `n_ttft` and `n_no_first_token` counts and validity warnings expose that denominator.
Benchmark cohorts requiring completed generations reject a missing first token instead of
silently dropping the failed sample from their average.

## 6. Persistent records and restart

### Format and identity

Store original KV and StateImage bytes without requantization. The persistent catalog needs:

| Record | Contents |
|---|---|
| Store header | Format version, store identity, execution fingerprint, payload geometry |
| Block | Stable object ID, parent ID, exact 64 tokens and extra key, payload reference |
| Snapshot | Anchor ID, exact frontier and tail tokens, kind, image parts and optional tail reference |
| Commit | Snapshot publication and the record generations it depends on |
| Removal | Retired snapshot or copy references and the generations they invalidate |

Runtime `NodeRef` generations, Device page IDs and Host slab indices are not persistent IDs.
Allocate stable record identities that do not alias a reused slot; recover allocator state
across restart, including records orphaned by an interrupted commit. A digest selects metadata
candidates but does not replace exact identity comparison. CRC detects payload damage, not token
identity collisions.

Use a versioned, checksummed append journal for catalog updates and a bounded periodic checkpoint
for restart cost. Batch updates; do not rewrite the entire catalog per KV block. Metadata I/O and
compaction run outside the Engine worker. Checkpoint replacement must preserve a recoverable
checkpoint/journal pair across interruption, including the journal position the checkpoint
covers. Reserve metadata/temporary space inside the configured store budget.

The fingerprint covers the artifact identity, KV/backend layouts, prefill/execution signature,
RoPE settings and implementation identity. Start with the existing Hybrid persistence's
conservative invalidation policy: a binary rebuild may invalidate reuse. Supporting reuse across
binary changes requires a separate demonstrated layout/semantics contract; this design does not
promise it. Use a distinct Hybrid store namespace/format, not the Legacy disk files. Report an
incompatible writable store and require a compatible location rather than overwriting foreign
records or interpreting their bytes under the new format.

### Commit, recovery and durability

Write payloads before their snapshot commit. A runtime Disk copy becomes readable only after
its record is complete; a restartable snapshot becomes visible only after its whole closure and
metadata commit complete. Removal records precede reuse of their physical storage. Generations
and checksums still validate references if an interrupted write leaves stale metadata.

On startup:

1. Verify the header and fingerprint before interpreting stored objects.
2. Replay the last valid checkpoint and complete journal records, ignoring an incomplete tail.
3. Validate referenced record identities/generations and dependency closure. Rebuild parents
   before children, then attach complete Disk snapshots.
4. Recompute runtime handles, reachability, per-tier accounting and eviction structures. Preserve
   bounded useful history such as hit counts, not pointers or old in-flight states.
5. Reclaim provisional payloads whose metadata never published and prune records with missing
   ancestry, image segments or a required tail. Finished loose blocks may replay without creating
   a snapshot hit; runtime pressure reclaims them across all branches. Admit requests only after
   this catalog is coherent.

Read headers/index metadata at startup; verify full payload checksums on demand rather than
preloading the disk cache into Host. A damaged or missing payload later invalidates that copy
and affected disk closures before reuse. Other intact copies and shallower snapshots remain
eligible. A corrupt catalog must not publish a partial tree with unchecked ancestry.

A read failure returns its own completion before cleanup I/O. The Engine invalidates the expected
record ID only, preserving a newer replacement and any intact Device/Host copies. Losing a parent
also invalidates descendant persistent parent links; already accepted fills keep their leases but
must abort instead of publishing those old links. A last-copy loss excludes the affected resume
point immediately, then retires its descriptors after the last transfer lease ends. Fresh matching
selects another valid source or root instead of retrying the same broken record until timeout.
Record removal is queued independently, so a blocked cleanup journal does not delay that fallback.

Removal first makes the record unavailable, then journals its tombstone, and only then releases
physical slots and metadata. An append or eviction failure preserves enough state for an
idempotent retry. Neither capacity accounting nor a crash replay treats an unjournaled deletion
as completed. The metadata mutex covers short in-memory operations only; a separate I/O mutex
serializes payload, journal and checkpoint operations without blocking Engine `find`, `begin_*`
or `stats` behind filesystem work.

Normal shutdown stops new submissions, settles accepted jobs, commits complete closures and
flushes catalog updates. An interrupted shutdown may lose pending cache work. There is no
power-loss durability promise: missing, torn or inconsistent records become misses, never
unchecked model inputs. The existing `DiskKVStore::flush_index` is not a durability barrier.

Only committed Disk closures survive restart. Host-only hot content is not automatically exported
at shutdown; this is not a replacement full-Host snapshot under another name.

## 7. Disk pressure and failure handling

The index chooses reclamation; the physical store cannot evict an arbitrary referenced page.
Reserve space before accepting writes and fail a new cache insertion when no safe space exists.
A Hybrid-specific no-automatic-eviction storage operation may extend the Core primitive without
changing Legacy LRU behavior.

Choose low-value, unpinned retention to release, update the affected disk closure records, and
then reclaim payloads that are no longer required. Mixed-tier runtime snapshots also count:
a Host image can still depend on a Disk-only ancestor. Removing a disk snapshot does not by
itself make every record on its path reclaimable. Shared ancestors, active dependencies and
in-flight read/write claims protect the records they still need.

| Failure | Required result |
|---|---|
| Queue full, slow writes or budget exhaustion | Stop submissions, skip retention or reclaim lower-value cache; keep serving active work |
| Missing record, checksum failure or read error | Invalidate the bad copy, release completed job resources and reselect a valid source before activation |
| Cancelled preparation | Detach request interest; retain accepted I/O buffers and copy destinations until safe completion |
| Stale completion after deletion/reuse | Reject it by runtime handle and record generation; release only resources owned by that job. A closure whose block write failed or whose anchor record was evicted never commits; the index's removal callbacks reach the catalog, which journals the removal so a restart never resurrects it. A failed closure's own image record is reclaimed only after that write returned, so a queued write can never recreate a deleted payload. |
| CUDA transfer failure | Preserve existing Engine-wide CUDA failure semantics; do not disguise it as a disk miss |
| Invalid path, incompatible store or insufficient startup geometry | Fail startup with the incompatible prerequisite identified |

Disk-disabled operation must preserve the existing two-tier behavior. Failure isolation must not
suppress invariant violations or convert arbitrary Program exceptions into cache misses.

## 8. Configuration and rebase surface

The Hybrid configuration reuses the existing spellings `--disk-kv-path` and `--disk-kv-gib` for
an opt-in path and positive bounded capacity (default 64 GiB). Hybrid opens and rebuilds its own
catalog whenever this tier is enabled. The Legacy `--disk-kv-restore` and
`--disk-kv-directstorage` switches remain Legacy-only; Hybrid adds no DirectStorage or
direct-to-GPU backend and rejects switches it ignores.

Validate the same combinations at the CLI and Engine boundary:

- Disk requires enabled Hybrid reuse and a nonzero Host budget large enough for staging and the
  remaining Host cache geometry.
- Disk and `hybrid.persistent_file` (`--prefix-cache-file`) are mutually exclusive. Reject both;
  neither takes precedence and neither silently exports/imports the other format.
- Disk path and capacity must resolve together. Existing Hybrid single-device restrictions and
  rejection of unrelated Legacy policies remain in force.

Watermarks, worker limits and batch sizes should be resolved from resource geometry and
qualification, with effective values in the startup ledger. Add user controls only for a
measured deployment requirement, not an unqualified knob for every internal limit.

The implementation patch is concentrated in:

- `runtime/prefix_cache`: per-tier copy predicates, retention/accounting and persistent catalog.
- `models/qwen3_5/program/prefix`: disk jobs, staging and prepared-source publication.
- `hybrid_program.cpp` and its quote/transaction data: ready-source validation and final binding.
- The existing head-prefetch/progress boundary: completion wakeup and bounded preparation before
  admission, without a second scheduler.
- Option normalization/validation, explicit CMake source lists and the affected tests/docs.

No loader weight changes, model mathematics or new family hierarchy belong to this feature. The
two-tier Hybrid reference remains authoritative where `--disk-kv-path` is absent.

## 9. Performance contract

The goal is to reduce recomputation after Host eviction while protecting hot requests and active
decode. Separate streams and CPU workers provide overlap, not isolation from shared bandwidth.
No absolute TTFT or zero-overhead claim follows from this design.

### TTFT

Protect Device/Host hits from disk work, catalog locks and checkpoints. Count queue wait and
unhidden preparation separately so overlap does not get counted twice. Select Disk only when
its remaining preparation plus prefill beats the best alternative with a safety margin.
Cold-start and OS-page-cache-warm reads need distinct coefficients and results.

FIFO-head prefetch may hide preparation under existing queue wait. It must not reserve a whole
cold path against active growth, synchronously settle an unfinished disk job, or wait forever
because a predicted cache hit exists.

### Prefill

Preserve tap density, chunk boundaries and arithmetic. Offload batches existing Host bytes;
it does not add exact taps, forward passes, conversion or compression. Bound worker-side
completion adoption and avoid catalog-wide scans at each commit/admission.

Report actually computed prefill tokens per second separately from effective input throughput
that includes reused tokens. Include time lost to pressure and CPU bookkeeping, not just GPU
kernel duration.

### Decode

Preserve the current prefill/decode service schedule. Give demanded reads precedence over
background writes inside the cache I/O subsystem without changing request FIFO or product QoS.
Use bounded queues and H2D batches; qualify byte/time limits while other lanes decode. Idle
execution can use a larger transfer budget than active decode.

H2D competes for PCIe and HBM bandwidth even on copy engines. Host-to-Disk writes compete for CPU,
RAM and storage bandwidth. Include expert host/disk streaming on the same devices in qualification
when that placement is used. Protect decode tail latency by slowing preparation or discarding
retention, not by extending GPU wait chains into SSD latency.

### Initial acceptance targets

These qualification targets are not measured results or previously accepted hardware-independent
SLAs. Fix the workload and thresholds before collecting implementation results; disclose and
justify any later change.

| Workload | Baseline | Initial target |
|---|---|---|
| Device/Host hits | Current Hybrid at matched effective hot capacity | TTFT p50 and p95 regress by no more than 2% |
| Root/cold prefill with background offload | Same model, executed tokens, chunk and concurrency without disk jobs | Actual prefill throughput regresses by no more than 2% |
| Decode with background offload | Same active sequences/backend without offload | Accepted-token throughput regresses by no more than 2%; streaming-gap p99 by no more than 5% |
| Decode with demanded cold preparation | Matched decode workload without preparation | Meet the same decode targets while reporting the cold request's TTFT and worst stalls separately |
| Disk-restored requests | Current Hybrid recomputation at the same Host budget | Demonstrate TTFT benefit across the selected workload distribution, including tails and exceptions |

Run both fixed-total-memory and matched-effective-hot-capacity comparisons. The first exposes
staging's reduction of hot retention; the second isolates mechanism overhead. Report cold fills'
eviction of other hot prefixes rather than attributing all hit-rate changes to workload alone.

A comparison records model/artifact, GPU and link, SSD/filesystem, toolchain, KV/backend,
concurrency, prompt/output lengths, shared-prefix distribution, cache state, budgets and offered
load. Include repeated comparable runs, sample counts, p50/p95/p99 where applicable, worst
changes, and every failed cell. Do not hide pauses behind average throughput or dismiss a small
regression as noise without evidence.

## 10. Verification and delivery

Implement in dependency order: persistent record/catalog correctness, bounded Host demotion,
pre-admission cold preparation, then public Engine qualification. These are integration steps,
not separately shippable incomplete substitutes for the four required behaviors.

### CPU behavior

Extend existing behavioral coverage for:

- Exact token/extra-key identity under forced lookup collisions; stable IDs across restart.
- Disk-only and mixed-tier validity, shared ancestors, nonaligned tails and last-copy loss.
- Tier-local reclaimable bytes, bounded metadata, and no automatic eviction of referenced records.
- Provisional blocks, complete snapshot commit, removal, interrupted checkpoints/journals,
  missing payloads and checksum failures.
- A streaming closure larger than ordinary Host block capacity, one reserved repair slab,
  per-block Host release before image commit, and the complete image/tail commit boundary.
- Source leases across commit: another reader keeps image, tail and block slabs until its last
  unpin; duplicate release requests return no bytes twice.
- Exact queue bytes through unadopted completions, read-destination limits, demanded-read priority
  and write ordering that prevents new block streams from starving an older closure.
- Exact image/tail reservations, rejected late writes after abandonment, removal publication
  failures and retries, crash replay without resurrection, and incomplete-ancestry pruning.
- Real blocked-journal I/O while Engine metadata operations and failed-read completion still
  progress; these checks use a Linux FIFO, not a mocked catalog or elapsed-time sleep.
- Failed-generation isolation, preserved volatile resumes, root fallback after last-copy loss,
  invalidated in-flight fills and deferred descriptor retirement until accepted leases end.
- The disk-committed tail as a device-fill source after device pressure shed its Device copy, and
  refusal when neither Host nor a committed disk closure can supply it.
- Host-first candidates, reachable complete Device-only refills, the source-ranking cost model
  including root, and orphan reclamation across forks and suffixes behind live boundaries.
- The staging budget split: `plan_hybrid_disk_staging` caps the block batch, never shrinks the
  slab pool below its own minimum, and reports the exact rejection figure.
- Queue saturation, stale completions, cancellation and source lifetime through accepted I/O.
- CLI and Engine option validation, including persistence-file mutual exclusion.

`ninfer_prefix_cache_index_test` covers index transitions; `ninfer_prefix_disk_catalog_test`
covers real persistent framing and recovery; `ninfer_prefix_disk_worker_test` exercises actual
queued I/O and blocking isolation. `ninfer_disk_kv_store_test` and `ninfer_disk_kv_bridge_test`
cover reusable storage primitives. Native I/O queue tests may skip in restricted environments;
report that gap rather than treating a skip as asynchronous-read qualification.

`ninfer_hybrid_disk_cpu_test` drives the production tier through long branching paths with a Host
pool smaller than the prefix, bounded cold-read staging and restart. Independent payloads compare
all block, full StateImage and nonaligned tail bytes exactly. Controlled copy completion checks
that cancellation of request interest never permits source reuse, accepted writes keep their byte
budget until adoption, and held repair events do not accumulate pins. Real payload corruption,
stale read completions and a failed tombstone journal exercise fallback and cleanup retries.

The same executable also drives production `ColdPreparation` with a 12-block disk-only chain
through two staging slots. It checks first-attempt readiness, exactly one read per block, and
independent destination bytes. Delayed copy completion, cancellation, an equal-length different
head, repeated idle polls, monotonic budget exhaustion and settled cleanup exercise the actual
preparation state machine. Auxiliary-copy completion must wake preparation even with unchanged
disk completion revision and free-page count. The physical callbacks are CPU buffers, not CUDA
event/kernel qualification; the shared Engine readiness decision also checks that idle preparation
is not mistaken for an impossible placement.
Late completion after budget exhaustion must also reach Ready without another read; a repeated
Fallback poll does not re-arm admission unless readiness actually changes.
A competing Device fill during an accepted disk read must keep its destination and bytes; the
preparation owner releases the unused read buffer instead of attempting a second fill.

The staging-slot lease includes a monotonically assigned generation, carried through Program
queues and checked on release. A delayed duplicate release after the slot has changed owners
cannot free the new reader's buffer; a free/busy bit alone does not protect this reuse boundary.
This adds one 64-bit owner token per staging slot and per returned block lease, not another payload
buffer or RAM cache. Pinned payload sizes and the advertised Host staging budget are unchanged.

`ninfer_prefix_disk_crash_test` uses pipe-synchronized Linux child processes terminated by SIGKILL
or `_exit`, so catalog destructors cannot manufacture recovery state. It checks unpublished
payloads, uncommitted images/tails, exact committed closures, tombstones and allocator recovery.
Narrow removal/checkpoint interruption windows are assembled from real production files captured
at explicit barriers; they test the resulting recoverable file pair, not an observed interruption
inside an atomic rename. Torn uncovered journal tails and leftover checkpoint temporaries are
also covered. Other platforms report this subprocess test as skipped, not qualified.

`ninfer_prefix_disk_reference_test` compares the index with a flat reference of exact token
prefixes, extra keys and independently tracked available sources. Three fixed seeds drive 72
branching epochs and 1,728 randomized operations, with candidate eligibility/source bytes, lease
counts, backend releases and Host/Device/disk accounting checked after each logical transition.
All lookup hashes collide deliberately; different tokens and extra keys still select different
prefixes. Repeated descriptor reuse checks stale generations, and controlled full drains avoid
duplicating private GDSF/LRU victim selection. Failures report the seed, epoch and operation trace.

The standalone [CPU gate](../../tests/host/prefix_disk/CMakeLists.txt) builds these paths with a
C++20 compiler and Threads only: no CUDA toolkit, driver, model artifact or Python is required.

```bash
cmake -S tests/host/prefix_disk -B ~/tmp/ninfer-prefix-disk-cpu -DCMAKE_BUILD_TYPE=Debug
cmake --build ~/tmp/ninfer-prefix-disk-cpu -j 4
TMPDIR=~/tmp ctest --test-dir ~/tmp/ninfer-prefix-disk-cpu --output-on-failure
```

For GCC/Clang ownership and undefined-behavior checks, configure a separate build with
`-DNINFER_PREFIX_DISK_SANITIZERS=ON`. The same sources also run in the normal product CTest suite.
These checks establish exact storage/recovery bytes, dependency and capacity semantics, and the
ownership protocol under delayed transfers; CUDA bindings and numerical execution remain the
separate qualification below.

Product integration also needs the CI configuration (`BUILD_TESTING=ON`, `NINFER_MULTICALL=ON`)
and a build of all default targets before CTest. Building only `ninfer_tests`, `ninfer` and
`ninfer-serve` omits independent tools such as `ninfer-multi-gpu-probe`, whose directly compiled
Device runtime must carry its peer-preflight implementation too. A passing CPU disk gate does not
establish that this wider source/link closure builds.

### Real execution and numerical semantics

Exercise the public Engine with a real artifact and force placement so a supposed Disk hit cannot
silently come from Device, Host or the old persistence file. Verify a prefix larger than the Host
pool, restart reconstruction, multiple shared branches, pressure during active growth and
cancellation while disk reads or CUDA copies remain outstanding. Include an idle Engine so job
completion itself must wake admission.

`disk-long-chain` now closes the saving Engine, reopens the persisted cache in a fresh Engine and
submits the extended prompt exactly once. It requires reuse of at least 17,000 tokens, more than
256 restored blocks, bounded completed read bytes and greedy outputs matching its no-disk
reference. A warm second attempt cannot satisfy these assertions. This scenario still needs a
real artifact/GPU run.

Compare restored KV and StateImage bytes exactly with their captured representation. Compare
Disk and hot restores of the same snapshot under the same continuation path. Root-vs-resumed
model checks retain the Hybrid specification's chunk-decomposition-aware numerical criteria;
byte-exact storage does not imply identical root-prefill arithmetic. Cover the promised KV and
speculative backend combinations, including MTP commitment lag and DFlash local state, and
preserve Vision's exact identity and forbidden resume spans.

### Performance and resource evidence

Measure hot hits, root prefill, offload-only, restore-only and simultaneous read/write pressure.
Include one and multiple lanes, long prefixes with short suffixes, sequential and scattered
records, storage-cold and OS-page-cache-warm reads, disk saturation and the relevant expert
placements. Track source-specific bytes, preparation/queue time, actual prefill work, accepted
decode tokens, streaming gaps, staging/leased Host peaks and cache drops. Extend existing logs
and benchmarks rather than creating a separate observability project.

CPU tests establish logical recovery, exact stored bytes and the transfer ownership protocol
under controlled completion schedules. They do not exercise actual CUDA event/stream ordering,
model-specific GPU packing/copy kernels or resumed model arithmetic. GPU runs qualify those
bindings, numerical outputs, TTFT and interference. Until those runs satisfy the selected
acceptance conditions, keep the feature opt-in and GPU-unqualified.
