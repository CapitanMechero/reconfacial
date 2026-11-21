"""Almacén CSV: formato, operaciones y detección de escrituras externas (HMAC + manifiesto)."""
import csv
import json
import shutil

import numpy as np
import pytest

from database.csv_store import (
    HEADER, CsvIdentityStore, DatabaseIntegrityError, normalize_name, read_rows, write_rows,
    write_seed,
)
from database.integrity import generate_key, load_or_create_key, parse_key
from recognition.errors import (
    DuplicateIdentityError, IdentityNotFoundError, KeyConfigurationError, TemplateCorruptError,
)
from recognition.template import extract_template, to_base64
from tests.synthetic import make_face


def reg(seed):
    return extract_template(make_face(seed))


@pytest.fixture()
def key():
    return parse_key(generate_key())


@pytest.fixture()
def paths(tmp_path):
    return tmp_path / "db" / "db.csv", tmp_path / "state" / "manifest.json"


@pytest.fixture()
def store(paths, key):
    return CsvIdentityStore(paths[0], key, paths[1])


def test_normalize_name():
    assert normalize_name("  Ana   María ") == ("Ana María", "ana maría")
    for bad in ("", "   ", "x" * 101, "ana\x00"):
        with pytest.raises(ValueError):
            normalize_name(bad)


def test_csv_is_plain_and_has_expected_columns(store, paths):
    store.add_identity("Ana", [reg(1)])
    with open(paths[0], newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == HEADER and rows[1][1] == "Ana" and len(rows[1][4]) == 64
    assert rows[1][2] == to_base64(reg(1))           # template en claro (base64)


def test_register_list_roundtrip(store):
    ident = store.add_identity("Ana", [reg(1), reg(2)])
    assert ident.name == "Ana" and ident.n_templates == 2
    t = store.load_templates()["Ana"]
    assert np.array_equal(t[0].orb.descriptors, reg(1).orb.descriptors)
    X, y, gen = store.load_hog_training_set()
    assert X.shape == (2, 8100) and list(y) == ["Ana", "Ana"] and gen == store.generation == 1
    assert np.allclose(X[0], reg(1).hog)


def test_duplicates_add_and_delete(store):
    store.add_identity("Ana", [reg(1)])
    with pytest.raises(DuplicateIdentityError):
        store.add_identity("  ANA ", [reg(2)])
    assert store.add_templates("ana", [reg(2)]).n_templates == 2
    with pytest.raises(IdentityNotFoundError):
        store.add_templates("Nadie", [reg(3)])
    store.add_identity("Luis", [reg(3)])
    store.delete_identity("ANA")
    assert [i.name for i in store.list_identities()] == ["Luis"]
    with pytest.raises(IdentityNotFoundError):
        store.delete_identity("Ana")


def test_reopen_with_same_key_works(store, paths, key):
    store.add_identity("Ana", [reg(1)])
    again = CsvIdentityStore(paths[0], key, paths[1])
    assert again.exists("ana") and again.generation == 1


# --------------------------- primer arranque (semilla) ---------------------------
def test_seed_is_adopted_and_signed_on_first_start(tmp_path, key):
    seed = tmp_path / "seed.csv"
    write_seed(seed, [("Ana", reg(1)), ("Ana", reg(2)), ("Luis", reg(3))])
    assert all(r.hmac == "" for r in read_rows(seed))
    db, man = tmp_path / "v" / "db.csv", tmp_path / "s" / "manifest.json"
    s = CsvIdentityStore(db, key, man, seed)
    assert s.adopted_on_start and [i.n_templates for i in s.list_identities()] == [2, 1]
    assert all(r.hmac for r in read_rows(db)) and all(r.hmac == "" for r in read_rows(seed))
    assert not CsvIdentityStore(db, key, man, seed).adopted_on_start


def test_corrupt_seed_is_rejected(tmp_path, key):
    seed = tmp_path / "seed.csv"
    write_seed(seed, [("Ana", reg(1))])
    rows = read_rows(seed)
    write_rows(seed, [rows[0].__class__(rows[0].sample_id, "Ana", "no-es-base64", rows[0].created_at)])
    with pytest.raises(TemplateCorruptError):
        CsvIdentityStore(tmp_path / "db.csv", key, tmp_path / "m.json", seed)


# --------------------------- ataques de escritura en la BD ---------------------------
def _edit(path, fn):
    rows = read_rows(path)
    write_rows(path, fn(rows))


def test_renaming_identity_is_detected(store, paths):
    store.add_identity("Ana", [reg(1)])
    store.add_identity("Luis", [reg(2)])
    _edit(paths[0], lambda rows: [r.__class__(r.sample_id, "Luis" if r.identity == "Ana" else r.identity,
                                               r.template, r.created_at, r.hmac) for r in rows])
    with pytest.raises(DatabaseIntegrityError):
        store.load_orb_templates()


def test_template_substitution_is_detected(store, paths):
    store.add_identity("Ana", [reg(1)])
    _edit(paths[0], lambda rows: [r.__class__(r.sample_id, r.identity, to_base64(reg(9)),
                                               r.created_at, r.hmac) for r in rows])
    with pytest.raises(DatabaseIntegrityError):
        store.list_identities()


def test_forged_row_is_detected(store, paths):
    store.add_identity("Ana", [reg(1)])
    _edit(paths[0], lambda rows: rows + [rows[0].__class__("f" * 32, "Intruso", to_base64(reg(5)),
                                                           rows[0].created_at, "0" * 64)])
    with pytest.raises(DatabaseIntegrityError):
        store.list_identities()


def test_deleted_row_is_detected(store, paths):
    store.add_identity("Ana", [reg(1)])
    store.add_identity("Luis", [reg(2)])
    _edit(paths[0], lambda rows: [r for r in rows if r.identity != "Ana"])
    with pytest.raises(DatabaseIntegrityError):
        store.list_identities()


def test_rollback_to_previous_csv_is_detected(store, paths, tmp_path):
    store.add_identity("Ana", [reg(1)])
    backup = tmp_path / "old.csv"
    shutil.copy(paths[0], backup)
    store.delete_identity("Ana")
    shutil.copy(backup, paths[0])                  # «resucitar» a una identidad eliminada
    with pytest.raises(DatabaseIntegrityError):
        store.list_identities()


def test_unsigned_csv_with_existing_state_is_detected(tmp_path, key, paths):
    s = CsvIdentityStore(paths[0], key, paths[1])
    s.add_identity("Ana", [reg(1)])
    write_seed(paths[0], [("Intruso", reg(7))])      # sustituye la BD por una «semilla»
    with pytest.raises(DatabaseIntegrityError):
        s.list_identities()


def test_tampered_manifest_and_wrong_key(store, paths, key):
    store.add_identity("Ana", [reg(1)])
    m = json.loads(paths[1].read_text())
    m["generation"] += 5
    paths[1].write_text(json.dumps(m))
    with pytest.raises(DatabaseIntegrityError):
        store.list_identities()
    with pytest.raises(DatabaseIntegrityError):
        CsvIdentityStore(paths[0], parse_key(generate_key()), paths[1])


def test_signed_rows_without_manifest_are_rejected(store, paths, key):
    store.add_identity("Ana", [reg(1)])
    paths[1].unlink()
    with pytest.raises(DatabaseIntegrityError):
        CsvIdentityStore(paths[0], key, paths[1])


def test_interrupted_write_recovers(store, paths, key, monkeypatch):
    store.add_identity("Ana", [reg(1)])
    original = store._write_manifest
    calls = []

    def crash_on_final(gen, digest, pending=None):
        calls.append(pending)
        if pending is None and len(calls) == 2:
            raise KeyboardInterrupt("corte de luz simulado")
        return original(gen, digest, pending)
    monkeypatch.setattr(store, "_write_manifest", crash_on_final)
    with pytest.raises(KeyboardInterrupt):
        store.add_identity("Luis", [reg(2)])
    monkeypatch.undo()
    reopened = CsvIdentityStore(paths[0], key, paths[1])
    assert reopened.exists("Luis") and reopened.generation == 2


# --------------------------- clave ---------------------------
def test_key_created_once_with_private_permissions(tmp_path):
    k1, created = load_or_create_key(tmp_path / "s" / "k")
    k2, created2 = load_or_create_key(tmp_path / "s" / "k")
    assert created and not created2 and k1 == k2 and len(k1) == 32
    assert oct((tmp_path / "s" / "k").stat().st_mode & 0o777) == "0o600"


@pytest.mark.parametrize("bad", ["", "!!", "YQ=="])
def test_invalid_keys(bad, tmp_path):
    with pytest.raises(KeyConfigurationError):
        parse_key(bad)


def _add_many(db, man, key, prefix, n):
    s = CsvIdentityStore(db, key, man)
    for i in range(n):
        s.add_identity(f"{prefix}{i}", [reg(i)])


def test_concurrent_writes_from_two_processes(tmp_path, key):
    """La app y `python -m sistema` pueden escribir a la vez: no se pierde ni corrompe nada."""
    import multiprocessing as mp
    pytest.importorskip("fcntl")
    db, man = tmp_path / "db.csv", tmp_path / "state" / "manifest.json"
    CsvIdentityStore(db, key, man)
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_add_many, args=(db, man, key, p, 6)) for p in ("a", "b")]
    for p in procs:
        p.start()
    for p in procs:
        p.join(60)
        assert p.exitcode == 0
    final = CsvIdentityStore(db, key, man)
    assert len(final.list_identities()) == 12 and final.generation == 12
