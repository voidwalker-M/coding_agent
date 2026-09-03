"""
CLI registration / login.

Passwords are stored as PBKDF2-HMAC-SHA256 (stdlib only). A login writes a
local session file so later `agent chat` / `agent run --memory` use that
user_id for STM/LTM. This is not a network auth system.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import time
from pathlib import Path
from typing import Any

# OWASP-recommended order of magnitude for PBKDF2-SHA256 on a local CLI.
PBKDF2_ITERATIONS = 210_000
_USERNAME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_-]{1,31}$")
RESERVED_USERNAMES = frozenset({"default", "admin", "agent", "guest"})


class AuthError(ValueError):
    """Registration or login rejected."""


def normalize_username(raw: str) -> str:
    name = (raw or "").strip().lower()
    if not _USERNAME_RE.match(name):
        raise AuthError(
            "username must be 2–32 chars, start with a letter, and use only letters, digits, _ or -"
        )
    return name


def hash_password(password: str, *, iterations: int | None = None) -> str:
    if not password or len(password) < 6:
        raise AuthError("password must be at least 6 characters")
    rounds = int(iterations or PBKDF2_ITERATIONS)
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, rounds)
    return f"pbkdf2_sha256${rounds}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, iter_s, salt_hex, hash_hex = stored.split("$", 3)
        if scheme != "pbkdf2_sha256":
            return False
        iterations = int(iter_s)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(hash_hex)
        got = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
        return secrets.compare_digest(got, expected)
    except Exception:
        return False


def session_path() -> Path:
    override = os.environ.get("AGENT_SESSION_FILE")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".coding-agent" / "session.json"


def save_session(user_id: str, role: str = "user") -> Path:
    path = session_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "user_id": user_id,
        "role": role,
        "logged_in_at": time.time(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return path


def load_session() -> dict[str, Any] | None:
    path = session_path()
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    user_id = str(data.get("user_id") or "").strip()
    if not user_id:
        return None
    return {
        "user_id": user_id,
        "role": str(data.get("role") or "user"),
        "logged_in_at": data.get("logged_in_at"),
    }


def clear_session() -> None:
    path = session_path()
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def register_user(store, username: str, password: str):
    user_id = normalize_username(username)
    if user_id in RESERVED_USERNAMES:
        raise AuthError(f"username '{user_id}' is reserved")
    if store.get_password_hash(user_id):
        raise AuthError(f"username '{user_id}' is already registered")
    store.set_password_hash(user_id, hash_password(password), name=user_id)
    return store.get_user(user_id)


def authenticate(store, username: str, password: str):
    try:
        user_id = normalize_username(username)
    except AuthError:
        raise AuthError("invalid username or password") from None
    stored = store.get_password_hash(user_id)
    if not stored or not verify_password(password, stored):
        raise AuthError("invalid username or password")
    user = store.get_user(user_id)
    if user is None:
        raise AuthError("invalid username or password")
    return user
