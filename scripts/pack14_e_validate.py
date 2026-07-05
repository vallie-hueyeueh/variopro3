# -*- coding: utf-8 -*-
"""Пакет 14, блок Е — матпроверка на session_2026-07-04_14-51-44.csv через
НАСТОЯЩИЙ пайплайн пульта (vario_app offscreen, пакетный прогон файла):

  • оба режима верт. ускорения (mekf / scalar);
  • покой: |среднее v_фильтра по сегменту| ≤ 0.05 м/с (и за вычетом хода баро);
  • watchdog молчит;
  • RMS(2-й метод − основной) в покое < 0.01 м/с.

Запуск: PYTHONIOENCODING=utf-8 QT_QPA_PLATFORM=offscreen python scripts/pack14_e_validate.py
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pc"))

import numpy as np  # noqa: E402
from PySide6 import QtWidgets  # noqa: E402

import vario_app  # noqa: E402
vario_app.save_config = lambda *a, **k: None          # не трогаем config.json

FILE = os.path.join(ROOT, "data", "session_2026-07-04_14-51-44.csv")


def segments(t, mask, min_len=2.0):
    out, start = [], None
    for i in range(len(t)):
        if mask[i] and start is None:
            start = i
        elif not mask[i] and start is not None:
            if t[i - 1] - t[start] >= min_len:
                out.append((start, i))
            start = None
    if start is not None and t[-1] - t[start] >= min_len:
        out.append((start, len(t)))
    return out


def check(name, cond, detail=""):
    print(f"  {'✓' if cond else '✗'} {name}{('  — ' + detail) if detail else ''}")
    return bool(cond)


def run_mode(win, mode):
    win._va_mode = mode
    assert win._load_file_full(FILE), "файл не загрузился"
    f = win._file
    t = f["t"]
    rest = f["motion"] > 0
    segs = segments(t, rest)
    print(f"\n== режим {mode}: {t[-1]:.1f} с, {len(t)} точек, "
          f"покой {rest.mean() * 100:.0f}% ({len(segs)} сегментов ≥2 с) ==")
    ok = True
    # --- покой: среднее по сегментам. Критерий — как в пакете 13: за вычетом
    # хода САМОГО баро (фильтр обязан следовать за баро; если давление на
    # сегменте реально плывёт, вариометр честно это показывает) ---
    worst = worst_nb = 0.0
    for (i0, i1) in segs:
        vmean = float(np.mean(f["v_filt"][i0:i1]))
        # ход самого баро на сегменте (наклон регрессии)
        vb = float(np.polyfit(t[i0:i1], f["h_raw"][i0:i1], 1)[0])
        worst = max(worst, abs(vmean))
        worst_nb = max(worst_nb, abs(vmean - vb))
    ok &= check("покой: |среднее v − ход баро| ≤ 0.05 м/с по каждому сегменту",
                worst_nb <= 0.05, f"макс {worst_nb:.4f} "
                f"(сырое, вместе с ходом баро: {worst:.4f})")
    # --- watchdog ---
    ok &= check("watchdog молчит", len(win._wd_events) == 0,
                f"событий {len(win._wd_events)}")
    # --- второй метод в покое ---
    v2 = f["v2"]
    m = rest & np.isfinite(v2)
    rms2 = float(np.sqrt(np.mean((v2[m] - f["v_filt"][m]) ** 2)))
    ok &= check("RMS(2-й метод − осн) в покое < 0.01 м/с", rms2 < 0.01,
                f"{rms2:.4f} (точек {m.sum()})")
    # --- курс есть и непрерывен (справочно) ---
    hd = f["head"]
    good = np.isfinite(hd)
    dh = np.abs((np.diff(hd[good]) + 180) % 360 - 180)
    print(f"  (курс: {good.mean() * 100:.0f}% точек, макс шаг между "
          f"соседними {dh.max():.1f}°)")
    return ok


def main():
    app = QtWidgets.QApplication([])
    win = vario_app.VarioApp()
    win._save_config = lambda *a, **k: None
    ok = run_mode(win, "mekf")
    ok &= run_mode(win, "scalar")
    print("\nЕ ПРОЙДЕН ✓" if ok else "\nЕ: ЕСТЬ ПРОВАЛЫ ✗")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
