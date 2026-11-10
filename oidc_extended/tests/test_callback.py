"""Unit tests for the OIDC callback state/nonce hardening.

frappe is not installed in the test environment, so a stub module is
installed into sys.modules before importing oidc_extended.callback.
The cache, the token exchange (requests.post) and the token decoding
(jwt.decode) are all isolated via monkeypatch.
"""

import json
import sys
import time
import types
from types import SimpleNamespace
from unittest.mock import MagicMock
from urllib.parse import parse_qs, urlparse

import pytest


def _install_frappe_stub():
    frappe = types.ModuleType("frappe")
    utils = types.ModuleType("frappe.utils")

    utils.logger = MagicMock()
    utils.get_url = lambda path="": "https://erp.example" + path

    frappe.utils = utils
    frappe._ = lambda message, *args, **kwargs: message
    frappe.whitelist = lambda *args, **kwargs: (lambda fn: fn)
    frappe.logger = MagicMock(return_value=MagicMock())
    frappe.cache = MagicMock()
    frappe.db = MagicMock()
    frappe.get_doc = MagicMock()
    frappe.get_cached_doc = MagicMock()
    frappe.get_conf = MagicMock()
    frappe.respond_as_web_page = MagicMock()
    frappe.generate_hash = lambda *args, **kwargs: "generated-password"
    frappe.local = SimpleNamespace(response={}, login_manager=MagicMock())
    frappe.session = SimpleNamespace(user="Guest")
    frappe.request = SimpleNamespace(path="/api/method/oidc_extended.callback.custom/test-provider")

    sys.modules["frappe"] = frappe
    sys.modules["frappe.utils"] = utils
    return frappe


frappe = _install_frappe_stub()

from oidc_extended import callback  # noqa: E402


class FakeCache:
    """Minimal dict-backed stand-in for frappe.cache()."""

    def __init__(self):
        self.store = {}

    def set_value(self, key, value, expires_in_sec=None):
        self.store[key] = value

    def get_value(self, key):
        return self.store.get(key)

    def delete_value(self, key):
        self.store.pop(key, None)

    def getdel(self, key):
        return self.store.pop(key, None)


STATE = "state-token"
NONCE = "nonce-token"


def make_record(state=STATE, nonce=NONCE, expires_in=600, redirect_to="/app"):
    return {
        "state": state,
        "nonce": nonce,
        "expires_at": time.time() + expires_in,
        "redirect_to": redirect_to,
    }


def make_claims(**overrides):
    claims = {
        "sub": "oidc-user",
        "email": "oidc-user@example.com",
        "iss": "https://idp.example",
        "aud": "erpnext",
        "exp": time.time() + 600,
        "nonce": NONCE,
        "groups": [],
    }
    claims.update(overrides)
    return claims


@pytest.fixture(autouse=True)
def reset_frappe_state():
    frappe.local.response = {}
    frappe.local.login_manager = MagicMock()
    yield


@pytest.fixture
def cache(monkeypatch):
    fake = FakeCache()
    monkeypatch.setattr(frappe, "cache", lambda: fake)
    return fake


@pytest.fixture
def token_post(monkeypatch):
    """Mocks the token exchange endpoint; returns the mock so tests can assert on it."""
    post = MagicMock(return_value=SimpleNamespace(json=lambda: {"id_token": "raw-id-token"}))
    monkeypatch.setattr(callback.requests, "post", post)
    return post


def configure_provider(monkeypatch):
    provider = SimpleNamespace(
        client_id="erpnext",
        get_password=lambda *args, **kwargs: "client-secret",
        auth_url_data=json.dumps({"scope": "openid profile email"}),
        redirect_url="/api/method/oidc_extended.callback.custom/test-provider",
        base_url="https://idp.example",
        authorize_url="/authorize",
        access_token_url="/token",
        user_id_property="sub",
    )
    config = SimpleNamespace(
        given_name_claim_name=None,
        family_name_claim_name=None,
        email_claim_name=None,
        groups_claim_name=None,
        group_role_mappings=[],
        fallback_role_profiles=[],
        group_module_mappings=[],
        fallback_module_profile=None,
    )
    user = MagicMock()
    user.enabled = True
    user.name = "oidc-user@example.com"
    user.get_social_login_userid.return_value = "oidc-user"
    user.get.return_value = []

    def get_doc(doctype, *args):
        if doctype == "Social Login Key":
            return provider
        return user  # new-user dict and ("User", name) fetches

    monkeypatch.setattr(frappe, "get_doc", get_doc)
    monkeypatch.setattr(frappe, "get_cached_doc", lambda *args, **kwargs: config)
    monkeypatch.setattr(frappe, "get_conf", lambda: {"custom": {"test-provider": "social-login-key"}})
    frappe.db.exists.return_value = None
    return provider, config, user


def mock_token_decoding(monkeypatch, claims):
    monkeypatch.setattr(callback.jwt, "decode", lambda *args, **kwargs: claims)


def assert_failed_login():
    assert frappe.local.response["type"] == "redirect"
    assert frappe.local.response["location"].startswith("/login?error=")
    frappe.local.login_manager.post_login.assert_not_called()


# --- state validation failures -------------------------------------------


def test_missing_state_parameter_redirects(cache, token_post):
    callback.custom("auth-code", "")

    assert_failed_login()
    token_post.assert_not_called()


def test_forged_state_redirects(cache, token_post):
    # A state that was never issued by this server (the cache only holds
    # an unrelated record).
    cache.store[callback._state_cache_key("someone-elses-state")] = make_record(state="someone-elses-state")

    callback.custom("auth-code", "forged-state")

    assert_failed_login()
    token_post.assert_not_called()


def test_state_record_mismatch_redirects(cache, token_post):
    # The record found under the key does not belong to the presented state.
    cache.store[callback._state_cache_key(STATE)] = make_record(state="different-state")

    callback.custom("auth-code", STATE)

    assert_failed_login()
    token_post.assert_not_called()


def test_expired_state_redirects_and_is_consumed(cache, token_post):
    cache.store[callback._state_cache_key(STATE)] = make_record(expires_in=-1)

    callback.custom("auth-code", STATE)

    assert_failed_login()
    token_post.assert_not_called()
    # The record is consumed even though it was expired.
    assert callback._state_cache_key(STATE) not in cache.store


def test_state_replay_fails(cache, token_post, monkeypatch):
    configure_provider(monkeypatch)
    mock_token_decoding(monkeypatch, make_claims())
    cache.store[callback._state_cache_key(STATE)] = make_record()

    # First callback succeeds.
    callback.custom("auth-code", STATE)
    assert frappe.local.response["type"] == "redirect"
    assert frappe.local.response["location"] == "/app"
    assert token_post.call_count == 1
    frappe.local.login_manager.post_login.assert_called_once()

    # Second callback with the same state (replay / refresh / concurrent tab)
    # must fail without touching the token endpoint again.
    frappe.local.response = {}
    frappe.local.login_manager = MagicMock()
    callback.custom("auth-code", STATE)

    assert_failed_login()
    assert token_post.call_count == 1


# --- nonce validation failures --------------------------------------------


def test_missing_nonce_in_id_token_redirects(cache, token_post, monkeypatch):
    configure_provider(monkeypatch)
    claims = make_claims()
    del claims["nonce"]
    mock_token_decoding(monkeypatch, claims)
    cache.store[callback._state_cache_key(STATE)] = make_record()

    callback.custom("auth-code", STATE)

    assert_failed_login()


def test_nonce_mismatch_redirects(cache, token_post, monkeypatch):
    configure_provider(monkeypatch)
    mock_token_decoding(monkeypatch, make_claims(nonce="attacker-nonce"))
    cache.store[callback._state_cache_key(STATE)] = make_record()

    callback.custom("auth-code", STATE)

    assert_failed_login()


# --- login initiation ------------------------------------------------------


def test_login_initiation_stores_state_record(cache, monkeypatch):
    configure_provider(monkeypatch)

    callback.login("test-provider", redirect_to="/app")

    assert frappe.local.response["type"] == "redirect"
    location = frappe.local.response["location"]
    assert location.startswith("https://idp.example/authorize?")

    query = parse_qs(urlparse(location).query)
    state = query["state"][0]
    assert query["response_type"] == ["code"]
    assert query["client_id"] == ["erpnext"]

    record = cache.store[callback._state_cache_key(state)]
    assert record["state"] == state
    assert record["nonce"] == query["nonce"][0]
    assert record["redirect_to"] == "/app"
    assert record["expires_at"] > time.time()


def test_login_initiation_rejects_external_redirect(cache, monkeypatch):
    configure_provider(monkeypatch)

    callback.login("test-provider", redirect_to="https://evil.example/phish")

    record = next(iter(cache.store.values()))
    assert record["redirect_to"] is None
