# -*- coding: utf-8 -*-
"""
sensor_priors.py
================
ПРИОРЫ ШУМОВ ПО ДАТАШИТАМ ДАТЧИКОВ (пакет 15, З.2; числа — docs/sensor_datasheets.md).

Телефон с пакета 15 шлёт строки SENSORS (имя/вендор/разрешение каждого датчика).
По ИМЕНИ датчика здесь подбираются паспортные шумы — СТАРТОВЫЕ значения
(приоры) для фильтров; дальше их уточняют замер по статике (config → mekf)
и адаптивный R̂ (vario_app, А.4). Приор НИКОГДА не затирает уже замеренные
значения в config — он даёт разумный старт и строку в лог для сверки.

match(sensors) → dict приоров или {} (датчики не распознаны).
"""

from __future__ import annotations

# Ключи datasheet-таблицы — ПОДСТРОКИ имени датчика (Sensor.name, верхний регистр)
DATASHEETS = {
    # IMU Samsung S23: ST LSM6DSO (аксель + гиро в одном корпусе)
    "LSM6DSO": {
        "chip": "LSM6DSO",
        # гиро: 3.8 mdps/√Hz = 3.8e-3·π/180 ≈ 6.6e-5 рад/с/√Гц
        "sigma_g": 6.6e-5,
        # аксель: 70 µg/√Hz = 70e-6·9.81 ≈ 6.9e-4 (м/с²)/√Гц;
        # на полосе 417/2 Гц → СКО сэмпла ≈ 6.9e-4·√208 ≈ 0.0099 м/с²
        "accel_noise_dens": 6.9e-4,
        "accel_sigma_417": 0.0099,
    },
    # магнитометр S23: AKM AK09918
    "AK09918": {
        "chip": "AK09918",
        "mag_sigma_uT": 0.6,          # ~0.6 мкТл RMS
    },
    # барометр S23: ST LPS22HH
    "LPS22HH": {
        "chip": "LPS22HH",
        # 0.65 Па RMS ≈ 5.5 см высоты (1 Па ≈ 8.4 см на уровне моря);
        # АЦП даёт ~25 Гц — это и есть настоящий темп баро
        "baro_sigma_m": 0.055,
        "R_baro": 0.055 ** 2,         # ≈ 0.0030 м²
        "baro_rate_hz": 25.0,
    },
}


def match(sensors: dict) -> dict:
    """По словарю SENSORS (ключ → {"name", ...}) вернуть найденные приоры:
    {"gyro": {...}, "accel": {...}, "mag": {...}, "baro": {...}} (что нашлось)."""
    out = {}
    if not isinstance(sensors, dict):
        return out
    for key, role in (("gyro", "gyro"), ("acc", "accel"),
                      ("mag", "mag"), ("baro", "baro")):
        info = sensors.get(key) or {}
        name = str(info.get("name", "")).upper()
        for chip, pri in DATASHEETS.items():
            if chip in name:
                out[role] = pri
                break
    return out


def describe(pri: dict) -> str:
    """Однострочное описание найденных приоров для лога/«Связи»."""
    parts = []
    g = pri.get("gyro")
    if g:
        parts.append(f"гиро {g['chip']}: σ_g={g['sigma_g']:.1e} рад/с/√Гц")
    a = pri.get("accel")
    if a:
        parts.append(f"аксель {a['chip']}: σ≈{a['accel_sigma_417']:.4f} м/с² @417 Гц")
    m = pri.get("mag")
    if m:
        parts.append(f"магнитометр {m['chip']}: σ≈{m['mag_sigma_uT']:.1f} мкТл")
    b = pri.get("baro")
    if b:
        parts.append(f"баро {b['chip']}: σ={b['baro_sigma_m']*100:.1f} см "
                     f"(R={b['R_baro']:.4f} м²), {b['baro_rate_hz']:.0f} Гц")
    return "; ".join(parts) if parts else "датчики не распознаны по даташитам"
