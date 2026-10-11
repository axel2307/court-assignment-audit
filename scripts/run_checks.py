"""Run the complete suite with external Python socket connections denied.

Dependency installation is a separate setup step. No web service, credential,
downloaded dataset or running browser is needed by these checks.
"""

import os
import socket
import tempfile
from ipaddress import ip_address
from pathlib import Path
from unittest.mock import patch

import pytest


def require_loopback(address):
    try:
        allowed = isinstance(address, tuple) and ip_address(address[0]).is_loopback
    except ValueError:
        allowed = False
    if not allowed:
        raise AssertionError("External network access is disabled during the test suite.")


original_connect = socket.socket.connect
original_connect_ex = socket.socket.connect_ex
original_create_connection = socket.create_connection


def local_connect(sock, address):
    require_loopback(address)
    return original_connect(sock, address)


def local_connect_ex(sock, address):
    require_loopback(address)
    return original_connect_ex(sock, address)


def local_create_connection(address, *args, **kwargs):
    require_loopback(address)
    return original_create_connection(address, *args, **kwargs)


if __name__ == "__main__":
    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    root = Path(__file__).resolve().parents[1]
    temporary_root = root / "artifacts" / "test-tmp"
    temporary_root.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(temporary_root)
    # Windows asyncio implements its internal socketpair over loopback.
    # Allow that local IPC while rejecting every external destination.
    with (
        patch.object(socket.socket, "connect", local_connect),
        patch.object(socket.socket, "connect_ex", local_connect_ex),
        patch.object(socket, "create_connection", local_create_connection),
    ):
        result = pytest.main([str(root / "tests"), "-q"])
    raise SystemExit(result)
