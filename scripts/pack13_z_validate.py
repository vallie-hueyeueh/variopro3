# -*- coding: utf-8 -*-
"""Блок З: валидация адаптивных R/Q (З.1) и RTS-сглаживателя (З.2) на
реальных записях — через НАСТОЯЩИЙ пайплайн пульта (оффскрин GUI).

QT_QPA_PLATFORM=offscreen python scripts\pack13_z_validate.py
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pc"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np                       # noqa: E402
from PySide6 import QtWidgets            # noqa: E402
import vario_app                         # noqa: E402
vario_app.save_config = lambda *a, **k: None      # не трогаем config.json
from pack13_common import baro_speed     # noqa: E402

FILES = {
    "session_2026-07-04_00-38-35.csv": [(118, 125, +1), (128, 140, -1),
                                        (183, 190, +1), (310, 335, -1)],
    "session_2026-07-03_15-34-53.csv": [(80, 92, -1)],
}


def run(path, adaptive, rts=False):
    win = vario_app.VarioApp()
    win.mode = "auto"
    win.radio_auto.setChecked(True)
    win._adaptive_on = adaptive
    win._show_rts = rts
    log = []
    if adaptive:
        orig = win._adapt_rq

        def wrapped(t, h_baro, dt):        # сигнатура пакета 15 (А.4): + h_baro
            orig(t, h_baro, dt)
            log.append((t, win._ad_R_mult, win._ad_Q_mult))
        win._adapt_rq = wrapped
    assert win._load_file_full(path)
    return win._file, log, win


def metrics(f, wins):
    t, vf, h = f["t"], f["v_filt"], f["h_raw"]
    vb = baro_speed(t, h)
    rest = (f["motion"] == 1) & (np.abs(np.nan_to_num(vb, nan=9)) < 0.10)
    rms_rest = float(np.sqrt(np.mean(vf[rest] ** 2))) if rest.any() else np.nan
    peaks = []
    for (a, b, sgn) in wins:
        m = (t >= a) & (t <= b)
        i_b = int(np.argmax(np.nan_to_num(vb[m], nan=0) * sgn))
        tb = t[m][i_b]
        m2 = (t >= tb - 1.5) & (t <= tb + 1.5)
        i_f = int(np.argmax(vf[m2] * sgn))
        peaks.append((float(vf[m2][i_f]), float(t[m2][i_f] - tb)))
    return rms_rest, peaks, rest


def main():
    app = QtWidgets.QApplication([])
    for name, wins in FILES.items():
        path = os.path.join(ROOT, "data", name)
        f_ad, log, w_ad = run(path, adaptive=True)
        f_fx, _, _ = run(path, adaptive=False)
        rms_a, pk_a, rest = metrics(f_ad, wins)
        rms_f, pk_f, _ = metrics(f_fx, wins)
        print(f"\n===== {name} =====")
        print(f"  RMS вариометра в Покое: адаптив {rms_a:.4f} | фикс {rms_f:.4f} м/с "
              f"{'✓ не хуже' if rms_a <= rms_f * 1.05 else '✗ ХУЖЕ'}")
        for i, ((va, la), (vf2, lf)) in enumerate(zip(pk_a, pk_f)):
            print(f"  пик {wins[i][0]}–{wins[i][1]}: адаптив {va:+.2f}@{la:+.2f} с | "
                  f"фикс {vf2:+.2f}@{lf:+.2f} с | Δпик {abs(va - vf2):.2f}, "
                  f"Δлаг {abs(la - lf):.3f} с")
        if log:
            arr = np.array(log)
            print(f"  лог адаптации: R×[{arr[:,1].min():.2f}…{arr[:,1].max():.2f}], "
                  f"Q×[{arr[:,2].min():.2f}…{arr[:,2].max():.2f}]; "
                  f"финальные R̂={0.0225*arr[-1,1]:.4f} м², "
                  f"Q̂={0.30*arr[-1,2]:.3f} м/с²")

    # --- RTS (З.2): снижение шума на статичном участке ---
    print("\n===== RTS-сглаживатель (З.2) =====")
    for name, (t0, t1) in (("session_2026-07-03_15-34-53.csv", (2.0, 21.0)),
                           ("session_2026-07-04_00-38-35.csv", (96.0, 118.0))):
        path = os.path.join(ROOT, "data", name)
        f, _, _ = run(path, adaptive=True, rts=True)
        if f.get("v_rts") is None:
            print(f"  {name}: RTS НЕ ПОСЧИТАЛСЯ ✗")
            continue
        m = (f["t"] >= t0) & (f["t"] <= t1)
        s_f = float(np.std(f["v_filt"][m]))
        s_r = float(np.std(f["v_rts"][m]))
        print(f"  {name} статика {t0}–{t1} с: std фильтра {s_f:.4f} | "
              f"std RTS {s_r:.4f} м/с — шум ниже в {s_f / max(s_r, 1e-9):.2f}×")


if __name__ == "__main__":
    main()
