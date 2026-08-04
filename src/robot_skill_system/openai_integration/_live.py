"""Shared Responses API structured-output implementation."""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypeVar, cast

from openai import OpenAI
from pydantic import BaseModel

from .client import SAFE_FUNCTION_TOOLS, RetryExecutor, usage_metadata
from .function_tools import FunctionToolError, SafeFunctionDispatcher
from .schemas import APICallMetadata

SchemaT = TypeVar("SchemaT", bound=BaseModel)
LOGGER = logging.getLogger(__name__)


def _image_part(path: Path, detail: str) -> dict[str, str]:
    media_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return {
        "type": "input_image",
        "image_url": f"data:{media_type};base64,{payload}",
        "detail": detail,
    }


def _file_part(path: Path, detail: str) -> dict[str, str]:
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return {
        "type": "input_file",
        "filename": path.name,
        "file_data": f"data:{media_type};base64,{payload}",
        "detail": detail,
    }


def parse_structured_response(
    *,
    client: OpenAI,
    retry: RetryExecutor,
    model: str,
    instructions: str,
    payload: BaseModel | dict[str, Any],
    output_type: type[SchemaT],
    trace_id: str,
    image_paths: list[Path] | None = None,
    file_paths: list[Path] | None = None,
    image_detail: str = "auto",
    tool_dispatcher: SafeFunctionDispatcher | None = None,
    maximum_tool_rounds: int = 4,
) -> tuple[SchemaT, APICallMetadata]:
    """Call Responses Parse and require a Pydantic-validated structured output.

    Supplying ``tool_dispatcher`` enables only the six strict read-only local tools. Tool calls
    are serialized, locally validated, and returned to the model before the final structured
    output is accepted. The default path supplies no tools and is behaviorally unchanged.
    """

    if maximum_tool_rounds < 1:
        raise ValueError("maximum_tool_rounds must be positive")

    body = payload.model_dump(mode="json") if isinstance(payload, BaseModel) else payload
    content: list[dict[str, str]] = [
        {
            "type": "input_text",
            "text": json.dumps(body, ensure_ascii=False, separators=(",", ":")),
        }
    ]
    for path in image_paths or []:
        content.append(_image_part(path, image_detail))
    for path in file_paths or []:
        content.append(_file_part(path, image_detail))

    started = time.monotonic()
    LOGGER.info(
        "openai_request_started trace_id=%s model=%s schema=%s image_count=%d file_count=%d",
        trace_id,
        model,
        output_type.__name__,
        len(image_paths or []),
        len(file_paths or []),
    )

    response_input: list[dict[str, Any]] = [{"role": "user", "content": content}]
    total_attempts = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0
    usage_observed = False
    tool_rounds = 0

    while True:

        def operation() -> object:
            common = {
                "model": model,
                "instructions": instructions,
                "input": cast(Any, response_input),
                "text_format": output_type,
                "store": False,
                "extra_headers": {"X-Client-Request-Id": trace_id},
            }
            if tool_dispatcher is None:
                return client.responses.parse(**cast(Any, common))
            return client.responses.parse(
                **cast(Any, common),
                tools=cast(Any, SAFE_FUNCTION_TOOLS),
                parallel_tool_calls=False,
                max_tool_calls=1,
            )

        response, attempts = retry.call(operation, trace_id)
        total_attempts += attempts
        usage = getattr(response, "usage", None)
        if usage is not None:
            usage_observed = True
            total_input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
            total_output_tokens += int(getattr(usage, "output_tokens", 0) or 0)
            total_tokens += int(getattr(usage, "total_tokens", 0) or 0)

        parsed = getattr(response, "output_parsed", None)
        if isinstance(parsed, output_type):
            break
        if tool_dispatcher is None:
            raise ValueError(f"Responses API did not return {output_type.__name__}")
        calls = _function_calls(response)
        if len(calls) != 1:
            raise FunctionToolError("Responses API must return exactly one serialized tool call")
        if tool_rounds >= maximum_tool_rounds:
            raise FunctionToolError("Responses API exceeded the bounded tool-call rounds")
        tool_rounds += 1
        call = calls[0]
        name = _required_tool_field(call, "name")
        arguments = _required_tool_field(call, "arguments")
        call_id = _required_tool_field(call, "call_id")
        output_json = tool_dispatcher.dispatch_json(name, arguments)
        response_input.extend(_response_output_history(response))
        response_input.append(
            {
                "type": "function_call_output",
                "call_id": call_id,
                "output": output_json,
            }
        )

    raw_metadata = usage_metadata(response, trace_id, total_attempts)
    if usage_observed:
        raw_metadata.update(
            {
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "total_tokens": total_tokens,
            }
        )
    metadata = APICallMetadata.model_validate(raw_metadata)
    LOGGER.info(
        "openai_request_finished trace_id=%s response_id=%s model=%s schema=%s "
        "schema_valid=true latency_ms=%.1f input_tokens=%s output_tokens=%s",
        trace_id,
        metadata.response_id,
        model,
        output_type.__name__,
        (time.monotonic() - started) * 1000.0,
        metadata.input_tokens,
        metadata.output_tokens,
    )
    return parsed, metadata


def _function_calls(response: object) -> list[object]:
    output = getattr(response, "output", ()) or ()
    return [item for item in output if _optional_tool_field(item, "type") == "function_call"]


def _response_output_history(response: object) -> list[dict[str, Any]]:
    """Serialize every SDK output item for stateless ``store=False`` continuation."""

    output = getattr(response, "output", ()) or ()
    history: list[dict[str, Any]] = []
    for item in output:
        if isinstance(item, BaseModel):
            payload = item.model_dump(mode="json", exclude_none=True)
        elif isinstance(item, Mapping):
            payload = dict(item)
        elif _optional_tool_field(item, "type") == "function_call":
            payload = {
                "type": "function_call",
                "name": _required_tool_field(item, "name"),
                "arguments": _required_tool_field(item, "arguments"),
                "call_id": _required_tool_field(item, "call_id"),
            }
        else:
            raise FunctionToolError("Responses API returned an unserializable output item")
        if not isinstance(payload.get("type"), str):
            raise FunctionToolError("Responses API output item is missing its type")
        history.append(payload)
    return history


def _optional_tool_field(item: object, field: str) -> object | None:
    if isinstance(item, Mapping):
        return item.get(field)
    return getattr(item, field, None)


def _required_tool_field(item: object, field: str) -> str:
    value = _optional_tool_field(item, field)
    if not isinstance(value, str) or not value:
        raise FunctionToolError(f"Responses API tool call is missing {field!r}")
    return value
