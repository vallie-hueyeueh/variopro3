# -*- coding: utf-8 -*-
"""Автотест Д.4 (пакет 13): «Скачать и воспроизвести» — хвостовые сэмплы
остановленного потока не должны затирать кривые загруженного файла.

QT_QPA_PLATFORM=offscreen python scripts\pack13_d4_test.py
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "pc"))

import numpy as np                     # noqa: E402
from PySide6 import QtWidgets, QtCore  # noqa: E402
import vario_app                       # noqa: E402
from sensor_source import Sample, SimSource  # noqa: E402


def main():
    app = QtWidgets.QApplication([])
    win = vario_app.VarioApp()
    ok = True

    # 1) живой поток: сэмплы принимаются (worker есть, файла нет)
    win.worker = vario_app.SourceWorker(SimSource(dt=0.02, speed=1000, loop=False))
    for i in range(200):
        win._on_sample(Sample(i * 0.02, 0.0, 100.0 + 0.01 * i))
    n_live = len(win.buf_t)
    ok &= n_live > 0
    print(f"1) живой приём: буфер {n_live} точек {'✓' if n_live > 0 else '✗'}")

    # 2) сценарий «Скачать и воспроизвести»: stop → загрузка файла → start,
    #    затем ХВОСТОВЫЕ сэмплы старого потока (как из очереди Qt)
    win.worker = None                # как после stop()
    path = os.path.join("data", "session_2026-07-03_16-12-04.csv")
    assert win._load_file_full(path), "файл не загрузился"
    win.combo_source.setCurrentText("CSV-файл")
    win.start()                      # плеер
    xs0 = win.curve_v_filt.getData()[0]
    n_curve_before = 0 if xs0 is None else len(xs0)
    for i in range(50):              # «хвост» мёртвого потока
        win._on_sample(Sample(1000.0 + i * 0.01, 0.0, 55.0))
    app.processEvents()
    win._redraw()
    xs = win.curve_v_filt.getData()[0]
    n_curve_after = 0 if xs is None else len(xs)
    stray = len(win.buf_t)
    ok &= stray == 0 and n_curve_after == n_curve_before
    print(f"2) после хвостовых сэмплов: буфер {stray} (ждём 0), кривая "
          f"{n_curve_after} точек (было {n_curve_before}) "
          f"{'✓' if stray == 0 and n_curve_after == n_curve_before else '✗'}")
    print("Д.4 АВТОТЕСТ " + ("ПРОЙДЕН ✓" if ok else "ПРОВАЛЕН ✗"))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
