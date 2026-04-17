"""Tests for RawVault content-addressed source storage."""
import os

import pytest

from brain_agent.memory.raw_vault import RawVault, source_id_for_sha256


@pytest.fixture
async def vault(tmp_path):
    v = RawVault(
        db_path=str(tmp_path / "raw_vault.db"),
        data_dir=str(tmp_path),
    )
    await v.initialize()
    yield v
    await v.close()


def test_source_id_is_deterministic_from_sha256():
    sha = "a" * 64
    assert source_id_for_sha256(sha) == "src_" + ("a" * 24)


async def test_schema_created(vault, tmp_path):
    import aiosqlite

    async with aiosqlite.connect(str(tmp_path / "raw_vault.db")) as db:
        cur = await db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='sources'"
        )
        row = await cur.fetchone()
    assert row is not None
    assert row[0] == "sources"


async def test_vault_directory_created(vault, tmp_path):
    assert os.path.isdir(str(tmp_path / "vault"))


async def test_indices_created(vault, tmp_path):
    import aiosqlite

    async with aiosqlite.connect(str(tmp_path / "raw_vault.db")) as db:
        cur = await db.execute(
            "SELECT name FROM sqlite_master WHERE type='index' "
            "AND tbl_name='sources'"
        )
        names = {r[0] for r in await cur.fetchall()}
    assert "idx_sources_workspace" in names
    assert "idx_sources_sha256" in names


async def test_ingest_text_inline(vault):
    result = await vault.ingest(
        workspace_id="personal",
        kind="user_utterance",
        extracted_text="hello world",
    )
    assert result["kind"] == "user_utterance"
    assert result["extracted_text"] == "hello world"
    assert result["workspace_id"] == "personal"
    assert result["sha256"]
    assert result["byte_size"] == len("hello world".encode("utf-8"))
    assert result["vault_path"] is not None


async def test_ingest_bytes_below_threshold_copies_to_vault(vault, tmp_path):
    data = b"small-binary-blob"
    result = await vault.ingest(
        workspace_id="personal",
        kind="file",
        data=data,
        mime="application/octet-stream",
        filename="blob.bin",
    )
    assert result["vault_path"] is not None
    assert os.path.exists(result["vault_path"])
    with open(result["vault_path"], "rb") as fh:
        assert fh.read() == data


async def test_ingest_large_bytes_pointer_mode(vault):
    from brain_agent.memory.raw_vault import VAULT_SIZE_THRESHOLD

    big = b"x" * VAULT_SIZE_THRESHOLD
    result = await vault.ingest(
        workspace_id="personal",
        kind="file",
        data=big,
        mime="application/octet-stream",
        filename="big.bin",
    )
    assert result["byte_size"] == VAULT_SIZE_THRESHOLD
    assert result["vault_path"] is None


async def test_ingest_from_path_below_threshold(vault, tmp_path):
    src = tmp_path / "source.txt"
    src.write_text("from-file content")
    result = await vault.ingest(
        workspace_id="personal",
        kind="file",
        path=str(src),
        mime="text/plain",
        filename="source.txt",
    )
    assert result["vault_path"] is not None
    assert os.path.exists(result["vault_path"])


async def test_ingest_from_path_above_threshold_pointer(vault, tmp_path):
    from brain_agent.memory.raw_vault import VAULT_SIZE_THRESHOLD

    src = tmp_path / "big.bin"
    with open(src, "wb") as fh:
        fh.write(b"y" * (VAULT_SIZE_THRESHOLD + 1))
    result = await vault.ingest(
        workspace_id="personal",
        kind="file",
        path=str(src),
        mime="application/octet-stream",
        filename="big.bin",
    )
    assert result["vault_path"] is None
    assert result["uri"] == str(src)


async def test_ingest_same_sha256_dedups(vault):
    a = await vault.ingest(
        workspace_id="personal",
        kind="user_utterance",
        extracted_text="identical text",
    )
    b = await vault.ingest(
        workspace_id="personal",
        kind="user_utterance",
        extracted_text="identical text",
    )
    assert a["id"] == b["id"]
    assert a["sha256"] == b["sha256"]
    all_sources = await vault.list_sources("personal")
    assert len(all_sources) == 1


async def test_ingest_requires_data_path_or_text(vault):
    with pytest.raises(ValueError, match="one of"):
        await vault.ingest(workspace_id="personal", kind="user_utterance")


async def test_get_raw_bytes_from_vault(vault):
    result = await vault.ingest(
        workspace_id="personal",
        kind="file",
        data=b"abc-123",
    )
    got = await vault.get_raw_bytes(result["id"])
    assert got == b"abc-123"


async def test_get_raw_bytes_pointer_mode_reads_original(vault, tmp_path):
    from brain_agent.memory.raw_vault import VAULT_SIZE_THRESHOLD

    big = tmp_path / "big.bin"
    with open(big, "wb") as fh:
        fh.write(b"z" * (VAULT_SIZE_THRESHOLD + 10))
    result = await vault.ingest(
        workspace_id="personal",
        kind="file",
        path=str(big),
    )
    got = await vault.get_raw_bytes(result["id"])
    assert got is not None
    assert len(got) == VAULT_SIZE_THRESHOLD + 10


async def test_get_raw_bytes_missing_returns_none(vault):
    assert await vault.get_raw_bytes("src_doesnotexist") is None


async def test_verify_integrity_valid(vault):
    result = await vault.ingest(
        workspace_id="personal",
        kind="file",
        data=b"integrity-test",
    )
    assert await vault.verify_integrity(result["id"]) is True
    refreshed = await vault.get_source(result["id"])
    assert refreshed["integrity_valid"] == 1


async def test_verify_integrity_tampered_marks_invalid(vault):
    result = await vault.ingest(
        workspace_id="personal",
        kind="file",
        data=b"will-be-tampered",
    )
    with open(result["vault_path"], "wb") as fh:
        fh.write(b"TAMPERED")
    assert await vault.verify_integrity(result["id"]) is False
    refreshed = await vault.get_source(result["id"])
    assert refreshed["integrity_valid"] == 0


async def test_verify_integrity_pointer_missing_marks_invalid(vault, tmp_path):
    from brain_agent.memory.raw_vault import VAULT_SIZE_THRESHOLD

    big = tmp_path / "big.bin"
    with open(big, "wb") as fh:
        fh.write(b"q" * (VAULT_SIZE_THRESHOLD + 1))
    result = await vault.ingest(
        workspace_id="personal",
        kind="file",
        path=str(big),
    )
    os.remove(str(big))
    assert await vault.verify_integrity(result["id"]) is False
    refreshed = await vault.get_source(result["id"])
    assert refreshed["integrity_valid"] == 0


async def test_find_by_sha256(vault):
    result = await vault.ingest(
        workspace_id="personal",
        kind="user_utterance",
        extracted_text="needle",
    )
    found = await vault.find_by_sha256(result["sha256"])
    assert found is not None
    assert found["id"] == result["id"]
