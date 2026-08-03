# OpenAI integration

## Supported boundary

`OpenAIClientFactory` constructs the official Python SDK client only for `OPENAI_MODE=live`, with
the configured timeout and retry limit. Default and test modes use `MockOpenAIClient` and make no
network call. Structured semantic outputs call
`client.responses.parse(..., text_format=PydanticModel, store=False)` rather than manually parsing
a JSON string. The supported dependency range is declared in `pyproject.toml`; model IDs remain
environment settings and must be verified against the operator's account before live use.

`DemonstrationAnalyzer`, `SkillGraphComposer`, and `RuntimeIntentResolver` accept only structured,
bounded inputs and outputs. After schema parsing, every returned entity, primitive, profile, and
binding identifier is checked for membership in the caller-supplied local catalogs; an unknown ID
raises `SemanticCatalogViolationError` before the result reaches graph or runtime code.
Representative RGB/depth-colormap keyframes may be data-URL image inputs; raw videos and raw depth
arrays are not sent. `TranscriptionService` calls Audio Transcriptions with a Korean hint, domain
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
