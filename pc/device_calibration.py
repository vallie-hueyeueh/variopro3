# -*- coding: utf-8 -*-
"""
device_calibration.py
=====================
ФОРМАТ ФАЙЛА КАЛИБРОВКИ ПРИБОРА pc/calibration.json — версия 2 (пакет 14, блок Б.2).

v2 несёт ДВЕ независимые секции магнитометра — по одной на источник поля:
    "mag_raw"     — калибровка СЫРОГО поля (TYPE_MAGNETIC_FIELD_UNCALIBRATED):
                    полный эллипсоид, V ~ железо телефона (~1200 мкТл);
    "mag_android" — «тонкая» калибровка ПОВЕРХ Android-калиброванного поля
                    (TYPE_MAGNETIC_FIELD): ОС уже сняла железо, наша поправка мала.
Каждая секция: model, hard_iron (V), soft_iron (W), target_F_uT, residual_pct
[, created]. Остальные ключи (accel, gyro_bias, declination_deg, location) — как в v1.

Старые файлы v1 (одна секция "mag" + ключ "mag_source") МИГРИРУЮТСЯ на чтении:
mag → mag_raw либо mag_android по mag_source (нет mag_source = сырое, как было
до протокола v4). Ничего на диске само не переписывается — только в памяти;
на диск v2 пишет «Сохранить калибровку прибора».

Источники здесь и всюду на ПК коротко: "raw" | "android".
"""

from __future__ import annotations

import json
import os

import numpy as np

PC_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE_CALIB_PATH = os.path.join(PC_DIR, "calibration.json")

SOURCES = ("raw", "android")                     # короткие имена источников
MAG_KEYS = {"raw": "mag_raw", "android": "mag_android"}
# соответствие длинным именам протокола v4 / старого config
LONG2SHORT = {"raw_uncalibrated": "raw", "android_calibrated": "android"}
SHORT2LONG = {v: k for k, v in LONG2SHORT.items()}


def normalize(d: dict) -> dict:
    """Привести словарь калибровки к v2 (в ПАМЯТИ; вход не меняется).
    v1: "mag" + "mag_source" → "mag_raw" | "mag_android"."""
    if not isinstance(d, dict):
        return {}
    out = dict(d)
    if out.get("version", 1) >= 2 or "mag_raw" in out or "mag_android" in out:
        out["version"] = max(2, int(out.get("version", 2) or 2))
        out.pop("mag", None)
        out.pop("mag_source", None)
        return out
    mag = out.pop("mag", None)
    src_long = out.pop("mag_source", None) or "raw_uncalibrated"
    src = LONG2SHORT.get(src_long, "raw")
    if isinstance(mag, dict):
        out[MAG_KEYS[src]] = mag
    out["version"] = 2
    return out


def load(path: str | None = None) -> dict:
    """Прочитать файл калибровки прибора → нормализованный v2-словарь.
    Файла нет / битый → {} (вызывающий работает по сырым данным).
    path=None → АКТИВНАЯ калибровка (DEVICE_CALIB_PATH читается в момент
    вызова — тесты могут подменять модульный атрибут)."""
    try:
        with open(path or DEVICE_CALIB_PATH, "r", encoding="utf-8") as fh:
            return normalize(json.load(fh))
    except (OSError, ValueError, TypeError):
        return {}


def mag_section(d: dict, source: str) -> dict | None:
    """Секция магнитометра для источника "raw"/"android" в удобном виде:
    {"V": np(3), "M": np(3,3), "F": float|None, "residual_pct": float|None,
     "model": str|None, "created": str|None} или None (секции нет/битая)."""
    sec = d.get(MAG_KEYS.get(source, ""), None)
    if not isinstance(sec, dict):
        return None
    try:
        V = np.asarray(sec["hard_iron"], dtype=float).reshape(3)
        M = np.asarray(sec["soft_iron"], dtype=float).reshape(3, 3)
    except (KeyError, ValueError, TypeError):
        return None
    F = sec.get("target_F_uT")
    res = sec.get("residual_pct")
    return {"V": V, "M": M,
            "F": (float(F) if F else None),
            "residual_pct": (float(res) if res is not None else None),
            "model": sec.get("model"),
            "created": sec.get("created") or d.get("created")}


def migrate_compass_use(cfg: dict) -> str:
    """config.json → compass_use в формате пакета 14: "ransac@raw" |
    "ransac@android" | "live@raw" | "live@android". Старые значения
    ("saved"/"live_ekf" + compass_mag_source) мигрируются."""
    val = cfg.get("compass_use")
    if isinstance(val, str) and "@" in val:
        method, _, src = val.partition("@")
        if method in ("ransac", "live") and src in SOURCES:
            return val
    src = LONG2SHORT.get(cfg.get("compass_mag_source", ""), "android")
    method = "live" if val == "live_ekf" else "ransac"
    return f"{method}@{src}"
