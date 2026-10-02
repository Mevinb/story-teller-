# Story Teller: quota-aware generation optimization plan

Prepared 2026-10-01. Planning only: this document does not change runtime settings,
enable additional providers, download models, or consume inference quota.

## Intended behavior and limits

Keep Qwen on Groq as the preferred prose writer. Run at the fastest sustainable
pace within the actual organization limits. Queue work before exceeding limits,
keep progress and cancellation responsive, and preserve completed work through
transport failures, quota resets, and application restarts.

Zero failures or uninterrupted Qwen output cannot be guaranteed by client code.
Finite quota imposes a throughput ceiling; other applications can consume the same
organization quota, and the provider/network can fail. The achievable contract is:
no preventable quota storms, no lost completed scenes, no falsely frozen interface,
and automatic continuation when capacity returns. To keep generating during daily
exhaustion or outages, a separately validated backup must have available capacity.

## Findings from the current checkout

| Finding | Consequence | Implementation target |
| --- | --- | --- |
| `_track_id()` identifies quota by model plus key | Key identity alone does not establish the actual organization/project quota domain | `models/groq_model.py` and a quota-group scheduler |
| TPM limit is configured as 6,000 with a 15% margin | App budgets 5,100 tokens/minute; this is not proof of the account's actual ceiling | Configuration plus provider-header reconciliation |
| Fixed eight-second spacing applies to each key | Aggregate requests and tokens can still exceed a shared ceiling | One admission queue per quota group |
| Successful response headers are not incorporated into scheduling | Scheduler learns too late, after a 429 | Both normal and streaming response handling |
| Prompt estimation uses characters/4 on the original prompt | Added schema, message instructions, and chat framing are omitted | Account for the final serialized request |
| Streaming uses text-length estimates and skips chunks without choices | Actual usage metadata can be missed and reservations can be understated | Streaming usage parser, including terminal metadata chunks |
| Global active key may change between reservation and client use | Concurrent agents can reserve against a different identity from the actual request | Pin the request's key, client, and reservation together |
| Admission lock is held across pacing waits | One waiting request can block unrelated eligible work | Short atomic reservations and condition-based scheduling |
| Budgets/cooldowns are in process memory | Restart discards accounting while provider quota remains consumed | Persist counters, active reservations, cooldowns, and checkpoints |
| Proactive rotation runs only in non-streaming generation | Normal and streaming requests behave differently | Shared request execution path |
| `HYBRID_ROUTING=true`, but configured local model file is absent | Supporting agents also consume Groq quota | Validate a real support backend before enabling hybrid routing |
| Architect accepts a corrected plan even when violations remain | Invalid plans can trigger costly downstream revisions | Repair premise coverage and introduction state before drafting |

The base retry wrapper already avoids retrying a propagated HTTP 429. Preserve
that behavior; do not introduce another outer retry loop around scheduler-managed
requests. Audit remaining schema-repair and writer-repair calls so they also use
the same admission path.

## 1. Establish the actual quota identity and limits

The user confirmed the seven keys come from different accounts. Exact organization
identities and Console limits have not yet been provided. Confirm the
organization/project associated with each configured key without logging secret
values; do not assume either that all seven share one quota or that seven separate
accounts necessarily provide seven independent organization budgets.

For each authorized workload, account against its verified organization/model
quota domain, with project restrictions layered on top. Keys in the same domain
share accounting; credential rotation changes the credential, not its budget.
Until metadata is verified, use a conservative active-workload budget.

Groq's published acceptable-use policy prohibits exceeding its parameters or rate
limits by registering multiple accounts or orchestrating usage between multiple
organizations. This plan therefore does not depend on stacking account quotas for
a single continuous workload. Use approved capacity, a separate provider, or local
inference to meet the desired sustained throughput.

Record Console limits for Qwen: RPM, TPM, RPD, TPD and, when present, separate input
and output TPM. Include project restrictions and any usage from other applications.
Do not increase limits automatically from a public documentation table.

Groq currently publishes Qwen free-plan baselines of 30 RPM, 8,000 TPM, 1,000 RPD,
and 200,000 TPD. Actual account limits can differ. Keep these as documentation
context, not as verified configuration for this installation.

Start at 80% of the verified limiting dimensions. Increase utilization gradually
only after telemetry demonstrates stable estimates and no unexpected 429s.
Provider resets and request bursts mean a steady-state average alone is insufficient.

## 2. Implement one scheduler for every Groq call

Create `models/quota_scheduler.py`, shared by architect, planner, writer, critic,
editor, verifier, schema repairs, continuations, and utility calls.
One scheduler owns admission, with separate ledgers for verified quota domains;
all keys within a domain use the same ledger.

Admission sequence:

1. Build the complete messages, schema instructions, and output allowance.
2. Estimate input tokens with a locally available compatible tokenizer/chat
   template where possible; otherwise use an upper-bound estimator calibrated
   against observed usage. No model download is required to begin conservatively.
3. Reserve one request and the input/output allowance atomically against all
   applicable quota dimensions. Include outstanding requests in the budget.
4. If capacity is unavailable, calculate the earliest eligible time. Release the
   scheduler lock while waiting; emit a structured waiting event and allow Cancel.
5. Pin the chosen key and client to that reservation. Send exactly one request.
6. Read available headers on successful responses and failures. Reconcile remaining
   budget conservatively with local in-flight reservations, including out-of-order
   responses. A stale response must not restore already-reserved capacity.
7. Settle using actual input/output usage, including streaming terminal metadata.
   Verify the installed SDK and Groq usage format in a small integration test.
   If usage is missing or the connection dies after dispatch, retain a conservative
   charge until expiry/reconciliation; do not assume the request consumed zero.

Track rolling windows and provider reset deadlines. A 429 sets a cooldown for the
entire affected quota group/model, honors `Retry-After`, and queues one controlled
retry. A timeout after partial output checkpoints the draft; restarting the whole
request automatically must not duplicate prose or repeated charges unnoticed.

For multiple projects, use fair scheduling with priorities: active scene writing,
necessary repairs, supporting work, then optional polish. Do not parallelize
dependent story steps. Parallelize only independent work with sufficient budget
and compatible local hardware capacity.

## 3. Reduce useful-work cost without silently lowering quality

Keep Qwen responsible for scene prose. Move supporting tasks to a verified local
backend or another authorized provider with its own available capacity. The current
configured local model is missing; local fallback is a prerequisite, not a working
capability. Before choosing a model, test loading, structured output, latency,
memory usage, and story-quality checks on the actual machine.

First reduce waste even if every task remains on Groq:

- Fix character-introduction state and premise coverage before writing a scene.
  The recent missing Sentinel event and premature Kavya arc are regression cases.
  Recover introduction evidence from retained chapters when necessary rather than
  assuming a missing state field means a character never appeared.
- Preserve mandatory user beats, established facts, and recent endings when
  compacting context. Remove duplicated instructions and irrelevant history.
- Cache planning/analysis results by exact inputs plus relevant story-state version.
  Invalidate on edits, premise changes, chapter deletion, or changed dependencies.
- Perform deterministic checks first. Where review tasks can safely share context,
  test a single structured review covering consistency and editing suggestions.
- Patch the passages causing an issue instead of rewriting an entire scene.
  Revalidate afterward and escalate unresolved issues visibly.
- Give each task a measured output allowance. Architect/planner limits must still
  fit the expected schema and number of scenes; writer limits must preserve target
  length and complete endings. Learn from observed usage and truncation rates.
- Add one per-scene budget covering ALL repair sources, continuations, verifier
  rewrites, and best-of-N drafts. Avoid independent repair loops multiplying calls.
  On exhaustion, checkpoint and mark for review rather than silently calling a
  known-invalid scene verified or spending indefinitely on the same issue.

Proposed normal path: one architect call and one scene decomposition per chapter;
one Qwen draft per scene; local/deterministic checks; an optional targeted repair
only when needed. Cloud review remains a measured quality option if no suitable
support backend exists. The exact call savings must be measured, not assumed.

Groq currently documents prompt caching for GPT-OSS models, not Qwen. Do not count
on Qwen cache savings. For a validated supporting GPT-OSS route, keep stable prompt
prefixes and measure cached usage, without assuming cache hits or avoiding admission
reservations before the provider confirms them.

## 4. Continue jobs through pauses and failures

Persist run state, completed scenes, the current draft, pending task identity,
quota accounting, and next eligible time in SQLite under the application workspace.
Use transactional admission and a single scheduler owner/lease so multiple workers
cannot spend the same reservation. Use monotonic clocks for active waits and
persisted wall-clock deadlines for recovery across process restarts.

Revalidate in-flight tasks after a crash. A possibly dispatched request is not
equivalent to a request that was never sent. Resume from saved scene boundaries,
or explicitly repair a partial draft, rather than silently repeating whole chapters.

Show: writing, checking, waiting for quota with a countdown, using backup, or waiting
for reset. Include model, measured tokens, remaining capacity, and next task.
Continue SSE heartbeats throughout waits and preserve cancellation. The recent SSE
buffer/reconnect fixes remain part of this design; they do not create API capacity.

On Qwen's daily exhaustion or a sustained outage, automatically use an explicitly
configured and live-tested backup if its capacity is available. Show the switch
and record the provider for each scene. Retain story context and validate output.
If no capable backup is available, save the job and resume automatically when
capacity returns. Indefinite immediate output requires sufficient backup/local
capacity or an approved service tier with adequate limits.

## 5. Validate before rollout

Use a fake clock and simulated provider first; no real quota needs to be exhausted.

- Seven same-organization keys reserve against one budget in both stream modes.
- Concurrent projects cannot overspend RPM/TPM or starve an eligible project.
- Reservations include schemas, injected instructions, and pending output.
- Normal and streamed actual usage settle correctly, including metadata-only chunks.
- Shared 429 cooldowns honor reset headers; only one controlled retry occurs.
- Missing headers, malformed headers, stale responses, partial output, cancellation,
  authentication errors, provider outages, and daily exhaustion preserve state.
- Restart recovers active jobs without resetting quota or duplicating completed scenes.
- Long streams and browser reconnects keep latest progress/completion visible.
- Known premise/character violations are repaired without unbounded cloud rewrites.

Then run a small isolated story with real Groq and the proposed support backend.
Measure calls/scene, input/output tokens per accepted scene, revision count,
time-to-first-text, scene latency, quota utilization, 429 frequency, and quality.
Compare against the same premise and acceptance checks. Production stories stay
untouched until this passes. No benchmark pass establishes universal zero failures.

## Delivery order

1. Baseline telemetry and verified quota metadata.
2. Shared scheduler, request pinning, headers/usage accounting, unified retries.
3. Plan/context repairs and a single scene revision budget.
4. Validate support/fallback routing and persist automatic resume.
5. Simulated stress tests, small live benchmark, and gradual rollout.

## Capacity example

Assume, solely for illustration, the actual TPM ceiling is 6,000. At 80% utilization,
the sustainable allowance is 4,800 tokens/minute. A pipeline costing 6,000 total
input/output tokens per accepted scene has a steady-state upper bound of
4,800 / 6,000 = 0.8 scenes/minute, or at least 75 seconds per scene before other
latency. At 3,000 tokens/accepted scene it is 1.6 scenes/minute, or 37.5 seconds.
Burst reservations and other limits may slow it further. Token reduction and
separate support capacity make a measurable difference; seven keys sharing that
same quota do not change this ceiling.

## Primary references

- [Groq rate limits and response headers](https://console.groq.com/docs/rate-limits)
- [Groq prompt caching and supported models](https://console.groq.com/docs/prompt-caching)
- [Groq Qwen model reference](https://console.groq.com/docs/model/qwen/qwen3.8-27b)
- [Groq acceptable-use policy](https://console.groq.com/docs/legal/ai-policy)

## Implementation status (2026-10-02)

Implemented the persistent SQLite workload scheduler for RPM/TPM/RPD/TPD and
optional input/output limits; complete-message reservations; pinned credentials;
success/error header reconciliation; actual normal/stream usage; unknown-spend
accounting; cancellable FIFO waiting outside locks; one controlled 429 retry.

Added structured support validation and Gemini routing, finite support HTTP timeouts
and a provider circuit breaker. Gemini Flash 2.5 support requests disable thinking
so the allocated completion budget is available to the requested output. Exact
input/state caches and partial drafts survive restarts. Leased automatic and batch
manual jobs resume remaining chapters; Stop persists cancellation. Interactive
manual sessions continue to use their existing WIP resume flow.

Added deterministic mandatory-beat and introduction repairs, a shared scene repair
allowance, targeted paragraph patches, and visible review flags. Progressively
saved WIP chapter files are excluded from the completed-chapter pointer recovery.
Checkpointed plans/scenes are reused rather than replanned over completed prose.

Validation: 261 offline Python tests and four SSE client tests pass. Live normal and streaming Qwen requests returned actual usage
(122 and 129 total tokens respectively). A temporary lighthouse chapter completed
with 839 words. Additional live runs encountered Gemini 503/429 responses and
Groq cooldowns; another run required a review flag. Those observations prompted
the finite timeout, circuit breaker, targeted patch, and draft-preservation fixes.
They demonstrate recovery paths, not a zero-error or zero-delay throughput result.
The last raw live observation is in RATE_LIMIT_LIVE_CHECK.json; it was collected
while the subsequent resilience fixes were still being made. No existing user
story was generated, deleted, rewound, or edited by these checks.

Known limits: actual RPM/TPD entitlements and organization IDs remain unverified;
use conservative configured ceilings. Groq authenticated headers observed 8,000
TPM and 1,000 RPD, while the configured workload TPM cap remains 6,000. Other
applications can consume quota outside this ledger. Gemini support currently
experiences provider errors/quota pressure, and the configured local GGUF is absent.
OpenRouter has no selected model and is not used as an invented backup.

The running Gunicorn worker must reload these source changes. An old active job
was started before durable job tracking and cannot be retroactively leased.
Restart with ./start.sh after that job finishes or is stopped, then refresh the
browser. No browser surface was available for interactive GUI verification.
