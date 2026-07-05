# -*- coding: utf-8 -*-
"""
references.py
=============
ЭТАЛОНЫ для калибровки: локальная гравитация g и магнитное поле Земли (F, D, I).

g — целевой радиус для калибровки акселерометра (|a| → g).
F — целевой радиус для калибровки магнитометра (|B| → F).
D — склонение (понадобится для курса); сохраняем в калибровке прибора.
"""

from __future__ import annotations

import datetime


# ----------------------------------------------------------------------
# ГРАВИТАЦИЯ (формула Сомильяны WGS-84 + поправка на высоту)
# ----------------------------------------------------------------------
def gravity_somigliana(lat_deg: float, alt_m: float) -> float:
    """
    Локальное ускорение свободного падения, м/с².
        g = 9.780327·(1 + 0.0053024·sin²φ − 0.0000058·sin²(2φ)) − 3.086e-6·h
    φ — широта (град), h — высота (м).
    """
    import math
    phi = math.radians(lat_deg)
    s1 = math.sin(phi) ** 2
    s2 = math.sin(2 * phi) ** 2
    g = 9.780327 * (1 + 0.0053024 * s1 - 0.0000058 * s2) - 3.086e-6 * alt_m
    return float(g)


# ----------------------------------------------------------------------
# Десятичный год (нужен моделям геомагнетизма)
# ----------------------------------------------------------------------
def to_decimal_year(d: datetime.date) -> float:
    start = datetime.date(d.year, 1, 1)
    days_in_year = (datetime.date(d.year + 1, 1, 1) - start).days
    day_of_year = (d - start).days
    return d.year + day_of_year / days_in_year


# ----------------------------------------------------------------------
# МАГНИТНОЕ ПОЛЕ — ОФЛАЙН (pygeomag, модель WMM)
# ----------------------------------------------------------------------
def geomag_offline(lat_deg: float, lon_deg: float, alt_m: float,
                   decimal_year: float) -> dict:
    """
    Магнитное поле по офлайн-модели WMM2025 (pygeomag).
    Возвращает {F (мкТл), D (°), I (°), source}.

    WMM2025 действует 2025.0–2030.0. СЕГОДНЯШНЯЯ дата (2026) В ДИАПАЗОНЕ, поэтому
    офлайн ОБЯЗАН работать без NOAA. Чтобы расчёт НИКОГДА не падал по дате: если год
    всё же вне диапазона — считаем у ближайшей границы (поле меняется медленно) и
    помечаем это; pygeomag разрешаем экстраполяцию (allow_date_outside_lifespan).
    """
    try:
        from pygeomag import GeoMag
    except ImportError:
        raise ValueError("Не установлен pygeomag (pip install pygeomag) — выберите «онлайн NOAA».")
    yr = float(decimal_year)
    note = ""
    if not (2025.0 <= yr <= 2030.0):
        clamped = min(2029.999, max(2025.0, yr))
        note = f" (дата {yr:.1f} вне 2025–2030 → взят {clamped:.2f})"
        yr = clamped
    gm = GeoMag(coefficients_file="wmm/WMM_2025.COF")
    r = gm.calculate(glat=lat_deg, glon=lon_deg, alt=alt_m / 1000.0,
                     time=yr, allow_date_outside_lifespan=True)
    return {"F": r.f / 1000.0, "D": r.d, "I": r.i, "source": "WMM2025 (офлайн)" + note}


# ----------------------------------------------------------------------
# МАГНИТНОЕ ПОЛЕ — ОНЛАЙН (NOAA geomag-web calculateIgrfwmm, JSON)
# ----------------------------------------------------------------------
def geomag_online(lat_deg: float, lon_deg: float, alt_m: float,
                  d: datetime.date, api_key: str = "") -> dict:
    """
    Магнитное поле через онлайн-калькулятор NOAA (geomag-web calculateIgrfwmm).
    Возвращает {F (мкТл), D (°), I (°), source}.

    ВАЖНО: NOAA теперь требует бесплатный API-КЛЮЧ (регистрация на
    ngdc.noaa.gov/geomag/calculators/magcalc.shtml). Без ключа сервер отвечает
    HTTP 400. Поэтому ключ обязателен; иначе используйте офлайн WMM2025.
    """
    import json
    import urllib.parse
    import urllib.request
    import urllib.error

    if not api_key:
        raise ValueError(
            "NOAA требует API-ключ (бесплатно: ngdc.noaa.gov/geomag/calculators/"
            "magcalc.shtml). Введите ключ в поле «Ключ NOAA» — или используйте "
            "офлайн WMM2025 (он уже даёт верные F/D/I).")

    # параметры строго по документации, всё URL-кодируется urlencode()
    params = {
        "key": api_key,
        "lat1": lat_deg, "lon1": lon_deg,
        "elevation": alt_m / 1000.0, "elevationUnits": "K",
        "model": "WMM",
        "startYear": d.year, "startMonth": d.month, "startDay": d.day,
        "resultFormat": "json",
    }
    url = ("https://www.ngdc.noaa.gov/geomag-web/calculators/calculateIgrfwmm?"
           + urllib.parse.urlencode(params))
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise ValueError(f"NOAA отклонил запрос (HTTP {e.code}: проверьте ключ/дату). "
                         f"Можно использовать офлайн WMM2025.")
    except Exception as e:
        raise ValueError(f"Нет связи с NOAA (интернет?): {e}")
    try:
        res = data["result"][0]
        total = res.get("totalintensity", res.get("totalIntensity"))
        F = float(total) / 1000.0
        D = float(res["declination"])
        I = float(res["inclination"])
    except (KeyError, IndexError, TypeError, ValueError):
        raise ValueError("NOAA вернул неожиданный ответ. Используйте офлайн WMM2025.")
    return {"F": F, "D": D, "I": I, "source": "NOAA (онлайн)"}


if __name__ == "__main__":
    import datetime as _dt
    # пример точки (центр СПб) — координаты произвольные, для самопроверки формул
    print("g (пример, СПб, 10 м):", round(gravity_somigliana(59.9386, 10.0), 5), "м/с²")
    dy = to_decimal_year(_dt.date(2026, 6, 29))
    print("Магнитное поле (офлайн WMM2025):", geomag_offline(59.9386, 30.3141, 10.0, dy))
