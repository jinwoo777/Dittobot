"""Skill embedding generation and local cosine similarity."""

from __future__ import annotations

import math
from dataclasses import dataclass

from openai import OpenAI

from robot_skill_system.settings import OpenAIMode, Settings

from .client import OpenAIClientFactory, RetryExecutor, new_trace_id
from .mock_client import MockOpenAIClient
from .schemas import EmbeddingResult


@dataclass(frozen=True)
class SkillSearchDocument:
    """Canonical metadata normalized before embedding."""

    name: str
    description: str
    target_objects: tuple[str, ...] = ()
    tools: tuple[str, ...] = ()
    style: str = "normal"
    preconditions: tuple[str, ...] = ()

    def normalized_text(self) -> str:
        return "\n".join(
            (
                f"name: {self.name.strip().lower()}",
                f"description: {self.description.strip().lower()}",
                f"targets: {', '.join(sorted(value.lower() for value in self.target_objects))}",
                f"tools: {', '.join(sorted(value.lower() for value in self.tools))}",
                f"style: {self.style.strip().lower()}",
                "preconditions: "
                f"{', '.join(sorted(value.lower() for value in self.preconditions))}",
            )
        )


class SkillEmbeddingService:
    """Generate embeddings through the official SDK or a deterministic offline hash."""

    def __init__(self, settings: Settings, *, client: OpenAI | None = None) -> None:
        self.settings = settings
        self._client = client
        self._mock = MockOpenAIClient()
        self._retry = RetryExecutor(settings.openai_api_max_retries)

    def embed_text(self, text: str) -> EmbeddingResult:
        """Embed text without persisting or logging its raw contents."""

        if self.settings.openai_mode is OpenAIMode.MOCK:
            return self._mock.embed(text, self.settings.openai_embedding_model)
        client = self._client or OpenAIClientFactory(self.settings).create()
        trace_id = new_trace_id()
        response, _attempts = self._retry.call(
            lambda: client.embeddings.create(
                model=self.settings.openai_embedding_model,
                input=text,
                encoding_format="float",
                extra_headers={"X-Client-Request-Id": trace_id},
            ),
            trace_id,
        )
        data = response.data[0]
        usage = getattr(response, "usage", None)
        return EmbeddingResult(
            vector=list(data.embedding),
            model=response.model,
            response_id=getattr(response, "id", None),
            input_tokens=getattr(usage, "prompt_tokens", None),
        )

    def embed_document(self, document: SkillSearchDocument) -> EmbeddingResult:
        return self.embed_text(document.normalized_text())


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Return cosine similarity, rejecting mismatched or zero-length vectors."""

    if not left or len(left) != len(right):
        raise ValueError("embedding vectors must be non-empty and have equal dimensions")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        raise ValueError("embedding vectors must be non-zero")
    return sum(a * b for a, b in zip(left, right, strict=True)) / (
        left_norm * right_norm
    )
