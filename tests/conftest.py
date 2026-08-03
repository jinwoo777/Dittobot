"""Suite-wide guard: offline tests must never open an IP network connection."""

from __future__ import annotations

import socket
from collections.abc import Iterator
from typing import Any

import pytest


class _OfflineSocket(socket.socket):
    """Allow local socketpairs while rejecting IPv4/IPv6 connections."""

    def connect(self, address: Any) -> None:
        if self.family in {socket.AF_INET, socket.AF_INET6}:
            raise RuntimeError("network access is disabled in the offline test suite")
        super().connect(address)

    def connect_ex(self, address: Any) -> int:
        if self.family in {socket.AF_INET, socket.AF_INET6}:
            raise RuntimeError("network access is disabled in the offline test suite")
        return super().connect_ex(address)


@pytest.fixture(autouse=True)
def block_external_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Reject DNS and IP connections for every test, including mock OpenAI tests."""

    def blocked(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("network access is disabled in the offline test suite")

    monkeypatch.setattr(socket, "socket", _OfflineSocket)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    yield
