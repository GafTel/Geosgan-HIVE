"""Shared test fixtures.

`no_network` forbids ALL outbound network access except loopback. Loopback must
stay reachable: on Windows CPython emulates `socket.socketpair()` over
127.0.0.1, which asyncio's event loop needs for its self-pipe. No external or
provider endpoint is reachable, so the offline guarantee is preserved (no live
API call is possible). The socket CONSTRUCTOR is never patched.
"""

import os
import socket

# LiteLLM fetches its remote model-cost map at import time unless the local
# map is selected; set it before ANY test module imports litellm (mirrors
# geosgan_hive/__init__.py, which sets it before its own lazy litellm import).
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import pytest

_LOOPBACK_HOSTS = ("127.0.0.1", "::1", "localhost")


def _is_loopback(address) -> bool:
    if address is None:
        return True  # bind()/getsockname()-style calls with no address
    try:
        host = address[0] if isinstance(address, tuple) else address
    except Exception:
        return False
    return str(host).strip().lower() in _LOOPBACK_HOSTS


@pytest.fixture
def no_network(monkeypatch):
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_create_connection = socket.create_connection
    real_getaddrinfo = socket.getaddrinfo

    def _deny(*args, **kwargs):
        raise AssertionError("network access attempted during offline test")

    def connect(self, address, *a, **k):
        if _is_loopback(address):
            return real_connect(self, address, *a, **k)
        raise AssertionError("network access attempted during offline test")

    def connect_ex(self, address, *a, **k):
        if _is_loopback(address):
            return real_connect_ex(self, address, *a, **k)
        raise AssertionError("network access attempted during offline test")

    def create_connection(address, *a, **k):
        if _is_loopback(address):
            return real_create_connection(address, *a, **k)
        raise AssertionError("network access attempted during offline test")

    def getaddrinfo(host, *a, **k):
        if _is_loopback(host):
            return real_getaddrinfo(host, *a, **k)
        raise AssertionError("DNS resolution attempted during offline test")

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
    monkeypatch.setattr(socket, "create_connection", create_connection)
    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
