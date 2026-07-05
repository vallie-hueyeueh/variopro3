# -*- coding: utf-8 -*-
"""Пакет 14, Д.3 — замер среднего времени кадра (_redraw) ДО/ПОСЛЕ оптимизации
отрисовки (setDownsampling(auto, peak) + setClipToView на всех кривых).

Сценарий-максимум: загружен файл 417 Гц (57.7 тыс. точек), идёт проигрывание,
окно «Всё» (все точки в кадре). Один и тот же код, флаги переключаются на лету.
Запуск: PYTHONIOENCODING=utf-8 QT_QPA_PLATFORM=offscreen python scripts/pack14_d3_perf.py
"""
import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pc"))

from PySide6 import QtWidgets  # noqa: E402

import vario_app  # noqa: E402
vario_app.save_config = lambda *a, **k: None

FILE = os.path.join(ROOT, "data", "session_2026-07-04_14-51-44.csv")
FRAMES = 150


def measure(win, label):
    win._play_i = 0
    win._play_t = float(win._file["t"][0])
    win._playing = True
    win._play_wall = None
    win._follow = True
    # прогрев кэшей отрисовки
    for _ in range(10):
        win._file_tick()
        win._dirty = True
        win._redraw()
        QtWidgets.QApplication.processEvents()
    t0 = time.perf_counter()
    worst = 0.0
    for _ in range(FRAMES):
        f0 = time.perf_counter()
        win._play_t += 0.033          # плеер идёт (реальный тик)
        win._file_tick()
        win._dirty = True
        win._redraw()
        QtWidgets.QApplication.processEvents()
        worst = max(worst, time.perf_counter() - f0)
    dt = (time.perf_counter() - t0) / FRAMES * 1000.0
    print(f"  {label:<46} средний кадр {dt:6.2f} мс, худший {worst * 1000:6.1f} мс")
    return dt


def set_ds(win, on: bool):
    for c in (win.curve_h_raw, win.curve_h_filt, win.curve_h_smooth,
              win.curve_h_rts, win.curve_v_raw, win.curve_v_filt,
              win.curve_v_smooth, win.curve_v2, win.curve_v_rts,
              win.curve_v_s1, win.curve_v_s2):
        if on:
            c.setDownsampling(auto=True, method="peak")
            c.setClipToView(True)
        else:
            c.setDownsampling(ds=1, auto=False, method="subsample")
            c.setClipToView(False)
    # заново отдать данные (сбросить кэш прореживания)
    win._file_set_curves()


def main():
    app = QtWidgets.QApplication([])
    win = vario_app.VarioApp()
    win._save_config = lambda *a, **k: None
    win.resize(1280, 860)
    win.show()
    assert win._load_file_full(FILE)
    n = len(win._file["t"])
    print(f"файл: {os.path.basename(FILE)}, {n} точек, окно «Всё», "
          f"{FRAMES} кадров проигрывания")
    # окна «Всё» на обоих графиках (максимум точек в кадре)
    win.combo_window_alt.setCurrentText("Всё")
    win.combo_window_vario.setCurrentText("Всё")
    set_ds(win, False)
    before = measure(win, "ДО (без прореживания, полные кривые)")
    set_ds(win, True)
    after = measure(win, "ПОСЛЕ (downsampling=peak + clipToView)")
    print(f"  ускорение кадра: ×{before / max(after, 1e-9):.1f}; "
          f"бюджет 30 Гц = 33 мс — "
          + ("укладываемся ✓" if after < 33 else "НЕ укладываемся ✗"))
    raise SystemExit(0 if after < 33 else 1)


if __name__ == "__main__":
    main()
