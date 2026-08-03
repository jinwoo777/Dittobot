"""Audio-file transcription via the official OpenAI Audio API."""

from __future__ import annotations

from pathlib import Path

from openai import OpenAI

from robot_skill_system.settings import OpenAIMode, Settings

from .client import OpenAIClientFactory, RetryExecutor, new_trace_id
from .mock_client import MockOpenAIClient
from .schemas import TranscriptResult, TranscriptSegment


class TranscriptionService:
    """Transcribe an audio file; live microphone capture remains an external adapter."""

    def __init__(self, settings: Settings, *, client: OpenAI | None = None) -> None:
        self.settings = settings
        self._client = client
        self._mock = MockOpenAIClient()
        self._retry = RetryExecutor(settings.openai_api_max_retries)

    def transcribe(self, audio_path: Path) -> TranscriptResult:
        """Transcribe *audio_path* with Korean and robot-domain hints."""

        path = audio_path.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if self.settings.openai_mode is OpenAIMode.MOCK:
            return self._mock.transcribe(path, self.settings.openai_transcribe_model)
        client = self._client or OpenAIClientFactory(self.settings).create()
        trace_id = new_trace_id()
        with path.open("rb") as audio_file:

            def operation() -> object:
                audio_file.seek(0)
                return client.audio.transcriptions.create(
                    file=audio_file,
                    model=self.settings.openai_transcribe_model,
                    language="ko",
                    prompt=self.settings.stt_domain_prompt,
                    response_format="verbose_json",
                    timestamp_granularities=["segment"],
                    extra_headers={"X-Client-Request-Id": trace_id},
                )

            response, _attempts = self._retry.call(operation, trace_id)
        text = str(getattr(response, "text", response))
        segments = []
        for item in getattr(response, "segments", None) or []:
            segments.append(
                TranscriptSegment(
                    start_s=float(getattr(item, "start", 0.0)),
                    end_s=float(getattr(item, "end", 0.0)),
                    text=str(getattr(item, "text", "")),
                )
            )
        return TranscriptResult(
            text=text,
            language=getattr(response, "language", "ko"),
            duration_s=getattr(response, "duration", None),
            segments=segments,
            model=self.settings.openai_transcribe_model,
            response_id=getattr(response, "id", None),
            source_audio_uri=path.name,
        )
