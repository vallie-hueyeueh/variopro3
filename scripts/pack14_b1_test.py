# -*- coding: utf-8 -*-
"""Пакет 14, Б.1 — автотест: переключение «Верт. ускорение: MEKF/скаляр»
НЕ должно прыгать курсом (>90° было до фикса; критерий приёмки: <2° в покое).

Гоняем НАСТОЯЩИЙ vario_app (offscreen) по живому пути (_process_sample) на
покойном участке реальной записи, дёргаем переключатель туда-обратно и меряем
курс. Для сравнения в конце воспроизводим СТАРОЕ поведение (сброс MEKF) —
видно, какой прыжок давал баг.
Запуск: PYTHONIOENCODING=utf-8 QT_QPA_PLATFORM=offscreen python scripts/pack14_b1_test.py
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pc"))

from PySide6 import QtWidgets  # noqa: E402

import vario_app  # noqa: E402
vario_app.save_config = lambda *a, **k: None          # не трогаем config.json

from sensor_source import CsvSource  # noqa: E402

FILE = os.path.join(ROOT, "data", "session_2026-07-04_14-51-44.csv")


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def main():
    app = QtWidgets.QApplication([])
    win = vario_app.VarioApp()
    win._save_config = lambda *a, **k: None
    src = CsvSource(FILE, realtime=False)
    src.open()
    samples = []
    while True:
        s = src.read_sample()
        if s is None or s.t > 10.0:
            break
        samples.append(s)
    src.close()
    print(f"покойный участок: {samples[-1].t:.1f} с, {len(samples)} сэмплов")

    win._load_mag_calibration()
    win._va_mode = "mekf"

    def feed(t0, t1):
        head = None
        for s in samples:
            if t0 <= s.t < t1:
                r = win._process_sample(s, quiet=True)
                if r["heading"] is not None:
                    head = r["heading"]
        return head

    h1 = feed(0.0, 8.0)
    # переключение mekf → scalar (как _on_va_mode_changed, но без GUI-радио)
    win._va_mode = "scalar"
    h2 = feed(8.0, 9.0)
    win._va_mode = "mekf"
    h3 = feed(9.0, 10.0)
    d12 = abs(wrap180(h2 - h1))
    d23 = abs(wrap180(h3 - h2))
    print(f"курс: mekf {h1:.2f}° → scalar {h2:.2f}° → mekf {h3:.2f}°")
    print(f"скачок при переключении: {d12:.2f}° и {d23:.2f}°  (критерий < 2°)")
    ok = d12 < 2.0 and d23 < 2.0

    # СТАРОЕ поведение для сравнения: сброс MEKF (как делал пакет 13)
    win2 = vario_app.VarioApp()
    win2._save_config = lambda *a, **k: None
    win2._load_mag_calibration()
    win2._va_mode = "mekf"
    headA = None
    for s in samples:
        if s.t < 8.0:
            r = win2._process_sample(s, quiet=True)
            if r["heading"] is not None:
                headA = r["heading"]
    win2._mekf = None                     # ← так делал старый переключатель
    win2._head_vec = None
    headB = None
    for s in samples:
        if 8.0 <= s.t < 10.0:
            r = win2._process_sample(s, quiet=True)
            if r["heading"] is not None:
                headB = r["heading"]
    print(f"для сравнения, СТАРОЕ поведение (сброс MEKF): "
          f"{headA:.2f}° → {headB:.2f}° (скачок {abs(wrap180(headB - headA)):.2f}°)")

    print("Б.1 ПРОЙДЕН ✓" if ok else "Б.1 ПРОВАЛЕН ✗")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
