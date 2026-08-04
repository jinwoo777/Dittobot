# OpenAI integration

## Supported boundary

`OpenAIClientFactory` constructs the official Python SDK client only for `OPENAI_MODE=live`, with
the configured timeout and retry limit. Default and test modes use `MockOpenAIClient` and make no
network call. Structured semantic outputs call
`client.responses.parse(..., text_format=PydanticModel, store=False)` rather than manually parsing
a JSON string. The supported dependency range is declared in `pyproject.toml`; model IDs remain
environment settings and must be verified against the operator's account before live use.

`DemonstrationAnalyzer`, `RecordingSkillDraftAnalyzer`, `SkillGraphComposer`, and
`RuntimeIntentResolver` accept only structured, bounded inputs and outputs. After schema parsing,
every returned entity, primitive, profile, and binding identifier is checked for membership in the
caller-supplied local catalogs; an unknown ID raises `SemanticCatalogViolationError` before the
result reaches graph or runtime code. `RecordingSkillDraftAnalyzer` receives only server-selected,
checksum-verified chronological RGB frames paired with locally rendered aligned-depth colormaps as
base64 data URLs. Each pair is RGB then depth. The active GPT-5.6 Luna configuration accepts images
but not video input, so raw video containers are not sent. Raw depth NPZ remains local. The strict
prompt treats two intentionally extended fingertips as gripper jaw tips and their midpoint as a
qualitative TCP proxy; the output schema forbids robot pose availability and contains no
coordinates. It never receives client-selected paths, API keys, force values, or execution
permission. `TranscriptionService`
calls Audio Transcriptions with a Korean hint, domain
prompt, verbose JSON, and segment timestamps. In mock mode, `robot-skill transcribe AUDIO_PATH`
uses a sibling `.txt` sidecar when present and never constructs a live client.
`SkillEmbeddingService` calls the Embeddings API in live mode and a deterministic local hash in
mock mode. The application persists one validated vector per skill version/model in SQLite and
uses NumPy cosine ranking; an embedding-provider failure can fall back to keyword retrieval
without granting execution authority.

The current wipe induction application invokes `DemonstrationAnalyzer` for bounded semantics and
the reteach decision, then materializes a locally defined safe wipe topology and inserts locally
measured, surface-relative path samples. `SkillGraphComposer` is implemented and
schema/catalog-tested as a service boundary, but it is not yet wired into the application induction
flow. Therefore the current application does not claim that an OpenAI graph proposal drove the
generated geometry or executable graph.

The UI recording workflow is deliberately a semantic-draft boundary. It evenly selects at most
`OPENAI_MAX_KEYFRAMES` aligned pairs (hard-limited to 300) from a finalized RGB-D manifest and persists the
validated `RecordingSkillDraft` beside that recording. Direct image inputs remain the primary
transport. If OpenAI rejects that image payload with an image/media/payload status error, the
server creates a local ZIP archive of the exact chronological RGB/depth-preview set and renders the
pairs as a labeled PDF contact sheet. The PDF is retried as an `input_file`, because normal file inputs do
not expose images embedded in ZIP archives to vision models. The ZIP is provenance/recovery data,
not a claimed vision input. The schema requires one TCP audit record for every supplied keyframe,
normalized two-fingertip image landmarks or an explicit failure reason, and semantic hand/tool/work-
surface regions. The schema fixes `executable=false` and `requires_pose_trajectory=true`;
the OpenAI response itself does not register, compile, validate, activate, or execute a SkillGraph.
Separate, explicitly invoked local endpoints use the normalized regions only as hints. Local raw
aligned depth and recorded camera intrinsics own plane fitting and fingertip deprojection. They may
add operator-confirmed surface calibration and two-fingertip RGB-D path evidence, register a
surface-relative Candidate, and run compile/Mock validation. They do not grant hardware authority.

Persisted recording drafts are available from `GET /skills/drafts` and
`GET /skills/drafts/{draft_id}`. Their promotion-readiness view is fail-closed. The UI/API can add
local evidence through:

- `POST /skills/drafts/{draft_id}/surface-calibration`
- `POST /skills/drafts/{draft_id}/surface-calibration/auto`
- `POST /skills/drafts/{draft_id}/tcp-trajectory`
- `POST /skills/drafts/{draft_id}/candidate`

Candidate registration remains disabled until the first two artifacts pass local validation. The
result stays a non-active, hardware-incompatible Candidate even after Mock validation. Real replay
still requires an independently calibrated robot-base TF chain, robot FK/trajectory evidence, and
the existing hardware safety path. A failed or absent legacy hand-eye NPY is advisory for this Mock
candidate path and is never attached as transform provenance.

## Observability and privacy

Every live integration creates a trace ID and sends it as a client request header. The structured
Responses path logs model/schema, latency, response ID, attempts, and available token usage; retry
logs contain trace ID and error type. API keys, raw image/audio bytes, and full prompts are not
logged by this code. The official SDK has its configured retry policy and the integration adds
bounded exponential backoff with jitter around retryable timeout/connection/rate-limit/server
errors.

## Function-tool status

`SAFE_FUNCTION_TOOLS` defines strict descriptors for six bounded, read-only operations: registry
search, manifest lookup, primitive catalog lookup, Scene entity query, Scene summary, and immutable
version comparison. `SafeFunctionDispatcher` validates every argument and provider result against
Pydantic schemas and maps names through explicit branches. Motion, force, shell, arbitrary Python,
safety disabling, compiled-artifact loading, and active-skill overwrite are explicitly unavailable.

When a caller supplies that dispatcher, `parse_structured_response` passes the strict descriptors
to Responses, disables parallel calls, permits one call per round, bounds the number of rounds,
returns only validated local JSON, and still requires the final Pydantic structured output. The
application's live runtime-intent path builds a sanitized snapshot from its validated registry and
current Scene and injects the dispatcher into `RuntimeIntentResolver`. Demonstration analysis and
graph composition remain structured-output-only. Enabling tools never adds robot-execution
authority; every final response must still satisfy the local `RuntimeIntent` schema and catalog
validation.

## Live mode

Set `OPENAI_API_KEY` only in the environment or an untracked `.env`, then set `OPENAI_MODE=live`.
Mock tests deliberately do not load `.env`. A live smoke test is an explicit operator action and
is not part of the default test suite. No real API key was used and Responses, Audio
Transcriptions, Embeddings, image input, and retry behavior were tested only with mocks/injected
fake clients in the current environment. A missing/invalid semantic result required by a command,
or schema/catalog validation failure, stops that operation before execution. Embedding-only search
failure may explicitly fall back to deterministic local keyword search; that fallback still grants
no execution authority and cannot bypass binding/preflight/runtime checks.
