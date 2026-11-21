"""Evaluación de ataques sobre una COPIA del sistema desplegado.

Uso::

    python -m evaluation.attacks [--n 50]

Escenarios (exigidos por la práctica):

A. **Saltarse el detector** (métodos ``_sd``): fotos genuinas sin alinear, entradas que
   no son caras (ruido, superficies lisas, recortes de fondo) y fotos de impostores.
B. **Inyectar templates** (métodos ``_t``): reutilizar un template leído de la BD
   (replay), templates aleatorios y templates malformados.
C. **Leer la BD**: qué obtiene el atacante (el CSV está en claro por decisión del equipo).
D. **Escribir la BD / el volumen**: renombrar, sustituir, falsificar, borrar,
   restaurar, sustituir el modelo SVM; con y sin acceso a la clave HMAC.

Usa los umbrales de ``evaluation/results/deployed.json`` si existe (calibrados) o los de
la configuración. Todo ocurre en un directorio temporal: no toca ``var/``.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import logging
import shutil
import tempfile
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from database.csv_store import ROW_DOMAIN, _Row, read_rows, write_rows
from database.integrity import sign
from evaluation.metrics import Rate
from recognition.bootstrap import build_service
from recognition.config import PROJECT_ROOT, Paths, Settings, Thresholds, load_settings
from recognition.errors import TemplateCorruptError
from recognition.model_lifecycle import ModelStatus
from recognition.orb import OrbTemplate
from recognition.service import FaceRecognitionService
from recognition.template import from_base64, to_base64
from recognition.types import RecognitionStatus, RegisteredFace

log = logging.getLogger(__name__)


def thresholds_for_evaluation(settings: Settings, results: Path) -> tuple[Thresholds, str]:
    rep = results / "deployed.json"
    if rep.is_file():
        d = json.loads(rep.read_text())
        c = d["identification"]["cascade"]
        v = d["verification"]
        return (replace(settings.thresholds, t1=float(c["t1"]), t2=float(c["t2"]),
                        t1_verify=float(v["t1v"]), t2_verify=float(v["t2v"]), calibrated=True),
                "calibrados (evaluation/results/deployed.json)")
    return settings.thresholds, "provisionales (configuración)"


def fresh_system(base: Path, thresholds: Thresholds) -> FaceRecognitionService:
    """Sistema nuevo en ``base`` a partir de la semilla (como un primer arranque)."""
    settings = load_settings()
    paths = Paths(data_dir=base / "data", models_dir=base / "data" / "models",
                  db_path=base / "data" / "db.csv", state_dir=base / "state",
                  seed_db=settings.paths.seed_db)
    s = build_service(Settings(paths=paths, thresholds=thresholds))
    s.ensure_model_current()
    return s


def _rate(flags) -> Rate:
    flags = list(flags)
    return Rate(int(sum(flags)), len(flags))


def _pct(r: Rate) -> str:
    lo, hi = r.wilson()
    return f"{100 * r.value:.1f} % ({r.k}/{r.n}; IC95 {100 * lo:.1f}–{100 * hi:.1f})"


# ------------------------------- A: sin detector -------------------------------
def non_face_images(n: int, photos: list[Path], seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    imgs = []
    for i in range(n):
        kind = i % 4
        if kind == 0:
            imgs.append(rng.integers(0, 256, (128, 128), dtype=np.uint8))            # ruido
        elif kind == 1:
            imgs.append(np.full((128, 128), int(rng.integers(0, 256)), np.uint8))   # liso
        elif kind == 2:
            g = np.linspace(0, 255, 128, dtype=np.float32)
            imgs.append(np.outer(g, np.ones(128)).astype(np.uint8))                 # degradado
        else:                                                                       # esquina de una foto
            img = cv2.imread(str(photos[i % len(photos)]), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(photos[i % len(photos)])
            h, w = img.shape
            imgs.append(img[: max(16, h // 4), : max(16, w // 4)])
    return imgs


def attack_bypass_detector(s, holdout, names, n, photos) -> list[tuple[str, str, str]]:
    rows = []
    genuine = [(i, p) for i, p in holdout]
    det = _rate(s.verify(i, p.read_bytes()).accepted for i, p in genuine)
    sd = _rate(s.verify_without_detector(i, p.read_bytes()).accepted for i, p in genuine)
    rows.append(("A1 Fotos genuinas completas por `verificar_sd` (sin alinear)",
                 f"aceptadas {_pct(sd)}", f"referencia con detector: aceptadas {_pct(det)}"))
    rng = np.random.default_rng(1)
    junk = non_face_images(n, [p for _, p in holdout])
    idf = [s.identify_without_detector(cv2.imencode(".png", j)[1].tobytes()) for j in junk]
    ver = [s.verify_without_detector(str(rng.choice(names)), cv2.imencode(".png", j)[1].tobytes())
           for j in junk]
    via_det = [s.identify(cv2.imencode(".png", j)[1].tobytes()) for j in junk]
    rows.append(("A2 Imágenes que no son caras por `identificar_sd`",
                 f"identificadas como alguien {_pct(_rate(r.is_identified for r in idf))}",
                 f"con detector: rechazadas por «sin cara» "
                 f"{_pct(_rate(r.status is RecognitionStatus.ERROR for r in via_det))}"))
    rows.append(("A3 Imágenes que no son caras por `verificar_sd` (identidad al azar)",
                 f"aceptadas {_pct(_rate(v.accepted for v in ver))}", ""))
    imp_sd, imp_det = [], []
    for k, (true, p) in enumerate(genuine):
        claim = names[(names.index(true) + 1 + k) % len(names)]
        imp_sd.append(s.verify_without_detector(claim, p.read_bytes()).accepted)
        imp_det.append(s.verify(claim, p.read_bytes()).accepted)
    rows.append(("A4 Impostor reclamando otra identidad por `verificar_sd`",
                 f"aceptados {_pct(_rate(imp_sd))}", f"con detector: {_pct(_rate(imp_det))}"))
    return rows


# ------------------------------- B/C: templates -------------------------------
def random_template(rng) -> RegisteredFace:
    hog = rng.uniform(0, 0.4, 8100).astype(np.float32)
    n = 400
    kps = np.column_stack([rng.uniform(10, 118, (n, 2)), np.full(n, 15.0), rng.uniform(0, 360, n),
                           rng.uniform(0, 1e-3, n)]).astype(np.float32)
    return RegisteredFace(hog, OrbTemplate(kps, rng.integers(0, 256, (n, 32), dtype=np.uint8)))


def attack_templates(s, db_path: Path, names, n) -> list[tuple[str, str, str]]:
    rows = []
    stolen: dict[str, str] = {}
    for r in read_rows(db_path):                      # C: el atacante lee el CSV en claro
        stolen.setdefault(r.identity, r.template)
    rep_v = [s.verify_template(name, t).accepted for name, t in stolen.items()]
    rep_i = [s.identify_template(t).identity == name for name, t in stolen.items()]
    rows.append(("B1 Replay: template leído de la BD por `verificar_t` (víctima)",
                 f"aceptados {_pct(_rate(rep_v))}", "el CSV está en claro: cualquier lector lo obtiene"))
    rows.append(("B2 Replay: template leído de la BD por `identificar_t`",
                 f"identificados como la víctima {_pct(_rate(rep_i))}", ""))
    rng = np.random.default_rng(7)
    rand = [random_template(rng) for _ in range(n)]
    rv = [s.verify_template(str(rng.choice(names)), to_base64(t)).accepted for t in rand]
    ri = [s.identify_template(to_base64(t)).is_identified for t in rand]
    rows.append(("B3 Templates aleatorios por `verificar_t` (identidad al azar)",
                 f"aceptados {_pct(_rate(rv))}", ""))
    rows.append(("B4 Templates aleatorios por `identificar_t`",
                 f"identificados {_pct(_rate(ri))}", ""))
    bad = ["", "no-base64!!", to_base64(rand[0])[:100], "RlJUAQ==",
           # bytes fabricados a mano (el atacante no usa nuestro serializador)
           base64.b64encode(b"FRT\x01" + np.full(8100, 1e30, "<f4").tobytes()
                            + rand[0].orb.to_bytes()).decode()]
    handled = 0
    for b in bad:
        try:
            s.verify_template(names[0], b)
        except ValueError:
            handled += 1
    rows.append(("B5 Templates malformados (vacío, truncado, cabecera falsa, valores extremos)",
                 f"rechazados con error de formato {handled}/{len(bad)}", ""))
    return rows


# ------------------------------- D: escritura -------------------------------
def _tamper_cases(attacker_victim: tuple[str, str], intruder_template: str
                  ) -> dict[str, Callable[[list[_Row]], list[_Row]]]:
    a, v = attacker_victim
    return {
        "D1 Renombrar las filas de una identidad a otra":
            lambda rows: [replace(r, identity=v) if r.identity == a else r for r in rows],
        "D2 Sustituir el template de la víctima por el del atacante":
            lambda rows: [replace(r, template=intruder_template) if r.identity == v else r for r in rows],
        "D3 Añadir una fila falsificada (firma inventada)":
            lambda rows: rows + [_Row(uuid.uuid4().hex, v, intruder_template, rows[0].created_at, "0" * 64)],
        "D4 Borrar todas las filas de una identidad":
            lambda rows: [r for r in rows if r.identity != v],
    }


def attack_db_write(base: Path, thresholds, names, intruder: str, intruder_photo: Path
                    ) -> list[tuple[str, str, str]]:
    rows_out = []
    victim = names[0]
    attacker_template = to_base64(fresh_system(base / "ref", thresholds).template_from_photo(
        intruder_photo.read_bytes()))

    def detected(sys_dir: Path) -> bool:
        try:
            s = fresh_system(sys_dir, thresholds)
            s.list_identities()
            return False
        except TemplateCorruptError:
            return True

    for title, fn in _tamper_cases((intruder, victim), attacker_template).items():
        d = base / uuid.uuid4().hex
        fresh_system(d, thresholds)
        db = d / "data" / "db.csv"
        write_rows(db, fn(read_rows(db)))
        rows_out.append((title + " (sin la clave)", "detectado" if detected(d) else "**NO detectado**",
                         "HMAC por fila / manifiesto"))

    d = base / "rollback"
    s = fresh_system(d, thresholds)
    db = d / "data" / "db.csv"
    old = db.read_bytes()
    s.delete(victim)
    db.write_bytes(old)
    rows_out.append(("D5 Restaurar una copia anterior del CSV (resucitar a un eliminado)",
                     "detectado" if detected(d) else "**NO detectado**", "generación en el manifiesto"))

    d = base / "replace_seed"
    fresh_system(d, thresholds)
    shutil.copy(load_settings().paths.seed_db, d / "data" / "db.csv")
    rows_out.append(("D6 Sustituir la BD por un CSV sin firmar (p. ej. la semilla manipulada)",
                     "detectado" if detected(d) else "**NO detectado**", "el manifiesto ya existe"))

    # Con la clave: el atacante también controla el directorio de estado
    d = base / "with_key"
    s = fresh_system(d, thresholds)
    key = (d / "state" / "integrity.key").read_text().strip()
    from database.integrity import parse_key
    k = parse_key(key)
    db = d / "data" / "db.csv"
    rows = read_rows(db)
    forged = _Row(uuid.uuid4().hex, victim, attacker_template, rows[0].created_at)
    forged = replace(forged, hmac=sign(k, forged.sample_id, forged.identity, forged.template,
                                       forged.created_at, domain=ROW_DOMAIN))
    write_rows(db, rows + [forged])
    (d / "state" / "manifest.json").unlink()           # y borra el manifiesto...
    caught = detected(d)
    rows_out.append(("D7 Con la clave: añadir fila firmada y borrar el manifiesto",
                     "detectado" if caught else "**NO detectado**",
                     "filas firmadas sin manifiesto -> se rechaza"))
    d2 = base / "with_key_full"
    s = fresh_system(d2, thresholds)
    k = parse_key((d2 / "state" / "integrity.key").read_text())
    store = s.store
    store._key = k
    store.add_templates(victim, [from_base64(attacker_template)])   # usa la clave como el sistema
    s2 = fresh_system(d2, thresholds)
    accepted = s2.verify(victim, intruder_photo.read_bytes()).accepted
    rows_out.append(("D8 Con la clave: inscribir la cara del atacante como la víctima",
                     f"no detectable; verificación del atacante como víctima: "
                     f"{'ACEPTADA' if accepted else 'rechazada'}",
                     "riesgo residual: quien controla la clave controla el sistema"))

    d = base / "model"
    s = fresh_system(d, thresholds)
    p = s.settings.paths
    p.svm_model.write_bytes(p.svm_model.read_bytes() + b"x")
    meta = json.loads(p.svm_meta.read_text())
    meta["sha256"] = hashlib.sha256(p.svm_model.read_bytes()).hexdigest()
    p.svm_meta.write_text(json.dumps(meta))
    status = s.model_status()
    rows_out.append(("D9 Sustituir el modelo SVM y recalcular su hash (sin la clave)",
                     "detectado (no se carga)" if status is ModelStatus.INVALID else f"**{status.value}**",
                     "meta firmado con HMAC; el pickle no se deserializa"))
    return rows_out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--photos", type=Path, default=PROJECT_ROOT / "data" / "fotos")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "evaluation" / "results")
    args = ap.parse_args()
    logging.basicConfig(level=logging.CRITICAL)
    logging.disable(logging.ERROR)
    settings = load_settings()
    thresholds, origin = thresholds_for_evaluation(settings, args.out)
    with open(settings.paths.holdout, newline="", encoding="utf-8") as f:
        test = [(r["identity"], args.photos / r["photo"]) for r in csv.DictReader(f)
                if r["split"] == "test"]
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        s = fresh_system(base / "main", thresholds)
        names = [i.name for i in s.list_identities()]
        rows = attack_bypass_detector(s, test, names, args.n, args.photos)
        rows += attack_templates(s, s.settings.paths.db_path, names, args.n)
        intruder, intruder_photo = test[-1]
        rows += attack_db_write(base, thresholds, [n for n in names if n != intruder],
                                intruder, intruder_photo)
    lines = ["# Evaluación de ataques", "",
             "Generado por `python -m evaluation.attacks`; no editar a mano.", "",
             f"Umbrales: {origin}. Fotos genuinas: las reservadas de test ({len(test)}).", "",
             "| Ataque | Resultado | Observación |", "|---|---|---|"]
    lines += [f"| {a} | {b} | {c} |" for a, b, c in rows]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "ataques.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
