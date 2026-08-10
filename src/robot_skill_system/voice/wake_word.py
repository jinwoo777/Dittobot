"""Continuously listen for the fixed ditto wake word without paid API calls."""

from __future__ import annotations

import importlib.util
import io
import math
import threading
import time
import wave
from collections.abc import Callable
from contextlib import suppress
from typing import Any


TranscribeCallback = Callable[[bytes, str, bool], dict[str, Any]]


class WakeWordController:
    """Own one server microphone loop and expose a polling-friendly status.

    The optional ``voice_processing`` package is imported only inside the worker
    thread.  Merely importing the core package therefore never opens a microphone
    or requires the ROS/voice environment used by ``ditto_ws``.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        microphone_device_index: int,
        wake_phrase: str,
        capture_duration_s: float,
        live_mode: bool,
        live_transcription_authorized: bool,
        api_key_configured: bool,
        transcribe: TranscribeCallback,
    ) -> None:
        self._enabled = enabled
        self._microphone_device_index = microphone_device_index
        self._wake_phrase = wake_phrase
        self._capture_duration_s = capture_duration_s
        self._live_mode = live_mode
        self._live_transcription_authorized = live_transcription_authorized
        self._api_key_configured = api_key_configured
        self._transcribe = transcribe
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._stream: Any | None = None
        self._state = "disabled"
        self._last_error: str | None = None
        self._last_result: dict[str, Any] | None = None
        self._updated_at_ns = time.time_ns()
        self._detection_count = 0

    @staticmethod
    def dependency_available() -> bool:
        """Return whether ditto's local wake-word package can be imported."""

        try:
            return importlib.util.find_spec("voice_processing") is not None
        except (ImportError, AttributeError, ValueError):
            return False

    def start(self) -> None:
        """Start listening once; repeated calls are harmless."""

        with self._lock:
            if not self._enabled or (self._thread and self._thread.is_alive()):
                return
            self._stop_event.clear()
            self._state = "starting"
            self._last_error = None
            self._updated_at_ns = time.time_ns()
            self._thread = threading.Thread(
                target=self._run,
                name="ditto-wake-word",
                daemon=True,
            )
            self._thread.start()

    def close(self) -> None:
        """Stop listening and release the microphone stream."""

        self._stop_event.set()
        with self._lock:
            stream = self._stream
        if stream is not None:
            with suppress(Exception):
                stream.stop_stream()
            with suppress(Exception):
                stream.close()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=2.0)
        self._set_state("stopped")

    def status(self) -> dict[str, Any]:
        """Return immutable primitives suitable for the voice status endpoint."""

        with self._lock:
            result = dict(self._last_result) if self._last_result is not None else None
            return {
                "enabled": self._enabled,
                "auto_start": True,
                "state": self._state,
                "wake_phrase": self._wake_phrase,
                "capture_duration_s": self._capture_duration_s,
                "microphone_device_index": self._microphone_device_index,
                "dependency_available": self.dependency_available(),
                "detection_count": self._detection_count,
                "live_transcription_authorized": (
                    self._live_transcription_authorized
                ),
                "last_error": self._last_error,
                "last_result": result,
                "updated_at_ns": self._updated_at_ns,
            }

    def _set_state(self, state: str, *, error: str | None = None) -> None:
        with self._lock:
            self._state = state
            self._last_error = error
            self._updated_at_ns = time.time_ns()

    def _set_result(self, result: dict[str, Any]) -> None:
        with self._lock:
            self._last_result = dict(result)
            self._state = "done"
            self._last_error = None
            self._updated_at_ns = time.time_ns()

    def _run(self) -> None:
        mic_controller: Any | None = None
        try:
            if not self.dependency_available():
                self._set_state(
                    "error",
                    error=(
                        "ditto Wake Word 의존성 voice_processing을 찾을 수 없습니다."
                    ),
                )
                return

            import pyaudio
            from voice_processing.MicController import MicConfig, MicController
            from voice_processing.wakeup_word import WakeupWord

            chunk = 12_000
            rate_hz = 48_000
            channels = 1
            buffer_size = 24_000
            mic_config = MicConfig(
                chunk=chunk,
                rate=rate_hz,
                channels=channels,
                record_seconds=self._capture_duration_s,
                fmt=pyaudio.paInt16,
                device_index=self._microphone_device_index,
                buffer_size=buffer_size,
            )
            mic_controller = MicController(config=mic_config)
            mic_controller.open_stream()
            stream = mic_controller.stream
            wakeup_word = WakeupWord(buffer_size)
            wakeup_word.set_stream(stream)
            with self._lock:
                self._stream = stream

            while not self._stop_event.is_set():
                with self._lock:
                    retained_error = self._last_error
                self._set_state("listening_for_wakeword", error=retained_error)
                while not self._stop_event.is_set() and not wakeup_word.is_wakeup():
                    pass
                if self._stop_event.is_set():
                    break

                with self._lock:
                    self._detection_count += 1
                self._set_state("wakeword_detected")
                self._set_state("recording")
                wav_audio = self._capture_wav(
                    stream,
                    pyaudio_module=pyaudio,
                    chunk=chunk,
                    rate_hz=rate_hz,
                    channels=channels,
                )

                blocked_reason = self._transcription_blocked_reason()
                if blocked_reason is not None:
                    self._set_state("listening_for_wakeword", error=blocked_reason)
                    continue

                self._set_state("processing")
                result = self._transcribe(
                    wav_audio,
                    "audio/wav",
                    self._live_mode and self._live_transcription_authorized,
                )
                self._set_result(result)
        except Exception as exc:
            if not self._stop_event.is_set():
                self._set_state(
                    "error",
                    error=f"Wake Word 자동 감지 시작 실패: {exc}",
                )
        finally:
            with self._lock:
                stream, self._stream = self._stream, None
            if stream is not None:
                with suppress(Exception):
                    stream.stop_stream()
                with suppress(Exception):
                    stream.close()
            if mic_controller is not None:
                close_stream = getattr(mic_controller, "close_stream", None)
                if callable(close_stream):
                    with suppress(Exception):
                        close_stream()

    def _transcription_blocked_reason(self) -> str | None:
        if not self._live_mode:
            return None
        if not self._api_key_configured:
            return (
                "Wake Word는 감지했지만 OPENAI_API_KEY가 없어 "
                "5초 전사를 건너뜁니다."
            )
        if not self._live_transcription_authorized:
            return (
                "Wake Word는 감지했지만 유료 전사가 허용되지 않아 5초 전사를 "
                "건너뜁니다."
            )
        return None

    def _capture_wav(
        self,
        stream: Any,
        *,
        pyaudio_module: Any,
        chunk: int,
        rate_hz: int,
        channels: int,
    ) -> bytes:
        frame_count = math.ceil(rate_hz * self._capture_duration_s / chunk)
        frames: list[bytes] = []
        for _ in range(frame_count):
            if self._stop_event.is_set():
                raise RuntimeError("Wake Word 녹음이 중지되었습니다.")
            try:
                frame = stream.read(chunk, exception_on_overflow=False)
            except TypeError:
                frame = stream.read(chunk)
            frames.append(frame)

        output = io.BytesIO()
        with wave.open(output, "wb") as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(
                pyaudio_module.get_sample_size(pyaudio_module.paInt16)
            )
            wav_file.setframerate(rate_hz)
            wav_file.writeframes(b"".join(frames))
        return output.getvalue()
