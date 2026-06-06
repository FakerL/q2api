import importlib.util
from pathlib import Path

import pytest

import app as app_module
from db import SQLiteBackend


async def _use_temp_db(monkeypatch, tmp_path):
    db = SQLiteBackend(tmp_path / "accounts.sqlite3")
    await db.initialize()
    monkeypatch.setattr(app_module, "_db", db)
    app_module._LABEL_UPSERT_LOCKS.clear()
    return db


async def _upsert(**overrides):
    payload = {
        "label": "label-a",
        "client_id": "client-1",
        "client_secret": "secret-1",
        "refresh_token": "refresh-1",
        "access_token": "access-1",
        "other": None,
        "enabled": True,
        "last_refresh_time": None,
        "last_refresh_status": "never",
    }
    payload.update(overrides)
    return await app_module._create_or_update_account_by_label(**payload)


@pytest.mark.asyncio
async def test_same_label_updates_existing_account(monkeypatch, tmp_path):
    db = await _use_temp_db(monkeypatch, tmp_path)

    first = await _upsert(label="label-a", client_id="client-1")
    second = await _upsert(
        label=" label-a ",
        client_id="client-2",
        client_secret="secret-2",
        refresh_token="refresh-2",
        access_token="access-2",
        enabled=False,
        last_refresh_time="2026-06-06T00:00:00",
        last_refresh_status="success",
    )

    rows = await db.fetchall("SELECT * FROM accounts WHERE label=?", ("label-a",))
    assert first["id"] == second["id"]
    assert len(rows) == 1
    assert rows[0]["clientId"] == "client-2"
    assert rows[0]["clientSecret"] == "secret-2"
    assert rows[0]["refreshToken"] == "refresh-2"
    assert rows[0]["accessToken"] == "access-2"
    assert rows[0]["enabled"] == 0
    assert rows[0]["last_refresh_status"] == "success"


@pytest.mark.asyncio
async def test_blank_label_does_not_dedupe(monkeypatch, tmp_path):
    db = await _use_temp_db(monkeypatch, tmp_path)

    first = await _upsert(label="   ", client_id="client-1")
    second = await _upsert(label=None, client_id="client-2")

    rows = await db.fetchall("SELECT * FROM accounts")
    assert first["id"] != second["id"]
    assert len(rows) == 2
    assert all(row["label"] is None for row in rows)


@pytest.mark.asyncio
async def test_dedupe_can_be_disabled_for_generated_labels(monkeypatch, tmp_path):
    db = await _use_temp_db(monkeypatch, tmp_path)

    first = await _upsert(label="批量账号 1", dedupe_by_label=False, client_id="client-1")
    second = await _upsert(label="批量账号 1", dedupe_by_label=False, client_id="client-2")

    rows = await db.fetchall("SELECT * FROM accounts WHERE label=?", ("批量账号 1",))
    assert first["id"] != second["id"]
    assert len(rows) == 2


@pytest.mark.asyncio
async def test_concurrent_same_label_uses_one_account(monkeypatch, tmp_path):
    db = await _use_temp_db(monkeypatch, tmp_path)

    first, second = await app_module.asyncio.gather(
        _upsert(label="label-a", client_id="client-1"),
        _upsert(label="label-a", client_id="client-2"),
    )

    rows = await db.fetchall("SELECT * FROM accounts WHERE label=?", ("label-a",))
    assert first["id"] == second["id"]
    assert len(rows) == 1


def _load_account_feeder():
    module_path = Path(__file__).resolve().parents[1] / "account-feeder" / "app.py"
    spec = importlib.util.spec_from_file_location("account_feeder_app", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
async def test_account_feeder_url_login_preserves_enabled(monkeypatch):
    feeder = _load_account_feeder()
    captured = {}

    feeder.AUTH_SESSIONS["auth-1"] = {
        "clientId": "client-1",
        "clientSecret": "secret-1",
        "deviceCode": "device-1",
        "interval": 1,
        "expiresIn": 300,
        "label": "label-a",
        "enabled": True,
        "status": "pending",
    }

    async def fake_poll_for_tokens(**_kwargs):
        return {"refreshToken": "refresh-1", "accessToken": "access-1"}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"id": "account-1", **captured["json"]}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, json, headers):
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers
            return FakeResponse()

    monkeypatch.setattr(feeder, "poll_for_tokens", fake_poll_for_tokens)
    monkeypatch.setattr(feeder.httpx, "AsyncClient", FakeClient)

    result = await feeder.auth_claim("auth-1")

    assert result["status"] == "completed"
    assert captured["json"]["enabled"] is True
    assert captured["json"]["label"] == "label-a"


@pytest.mark.asyncio
async def test_account_feeder_manual_empty_label_stays_empty(monkeypatch):
    feeder = _load_account_feeder()
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"ok": True}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, json, headers):
            captured["json"] = json
            return FakeResponse()

    monkeypatch.setattr(feeder.httpx, "AsyncClient", FakeClient)

    await feeder.create_account(
        feeder.AccountCreate(
            label=None,
            clientId="client-1",
            clientSecret="secret-1",
            refreshToken="refresh-1",
        )
    )

    assert captured["json"]["accounts"][0]["label"] is None
