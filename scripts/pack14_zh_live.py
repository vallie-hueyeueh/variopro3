# -*- coding: utf-8 -*-
"""Пакет 14, блоки Ж + Б.2 (живой тракт): симулятор (v4, оба маг-поля) →
НАСТОЯЩИЙ пульт (MainConsole offscreen) → проверяем:

  Ж: адаптивные R̂/Q̂ на живом потоке — R̂ уходит от паспортного (0.0225 м²)
     к фактическому шуму баро (~0.007); индикатор состояния живёт;
  Б.2: живой сбор ведёт ДВА буфера (raw и android), у каждого свой live-EKF;
     пункты Live-EKF активны только при сборе; выбор live@android работает;
     остановка сбора ЯВНО переключает компас на ransac@android.

Запуск: PYTHONIOENCODING=utf-8 QT_QPA_PLATFORM=offscreen python scripts/pack14_zh_live.py
"""
import os
import subprocess
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pc"))

from PySide6 import QtCore, QtWidgets  # noqa: E402

import vario_app  # noqa: E402
vario_app.save_config = lambda *a, **k: None

PORT = 5557
FILE = os.path.join(ROOT, "data", "session_2026-07-04_14-51-44.csv")
PY = sys.executable
RUN_SEC = 75.0


def check(name, cond, detail=""):
    print(f"  {'✓' if cond else '✗'} {name}{('  — ' + detail) if detail else ''}")
    return bool(cond)


def main():
    sim = subprocess.Popen(
        [PY, os.path.join(ROOT, "pc", "stream_simulator.py"),
         "--file", FILE, "--port", str(PORT)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1.0)
    try:
        run(sim)
    finally:
        sim.kill()


def run(sim):
    import main as main_mod                      # настоящий пульт целиком
    app = QtWidgets.QApplication([])
    win = main_mod.MainConsole()
    win.vario._save_config = lambda *a, **k: None
    v = win.vario
    c = win.calib
    v.combo_source.setCurrentText("Поток (Bluetooth/симулятор)")
    v.edit_stream_url.setText(f"socket://127.0.0.1:{PORT}")
    v.start()
    c.toggle_live_capture()                      # живой сбор (подписка на воркер)

    state = {"t0": time.monotonic(), "phase": 0, "ok": True, "snap": {}}

    def tick():
        el = time.monotonic() - state["t0"]
        if state["phase"] == 0 and el >= RUN_SEC:
            state["phase"] = 1
            s = state["snap"]
            s["R_eff"] = float(v.filter.R[0, 0])
            s["R_mult"] = v._ad_R_mult
            s["Q_mult"] = v._ad_Q_mult
            s["label"] = v.lbl_adaptive.text()
            s["n_raw"] = len(c._live_buf["raw"]["t"])
            s["n_and"] = len(c._live_buf["android"]["t"])
            s["ekf_raw"] = c._live_buf["raw"]["ekf"].metrics()
            s["ekf_and"] = c._live_buf["android"]["ekf"].metrics()
            s["live_enabled"] = c.ref_panel.radio_comp["live@android"].isEnabled()
            # выбрать live@android ПРИ идущем сборе
            c.ref_panel.radio_comp["live@android"].setChecked(True)
            QtCore.QTimer.singleShot(700, tick)
            return
        if state["phase"] == 1:
            state["phase"] = 2
            state["snap"]["use_live"] = v._compass_use
            state["snap"]["ind_live"] = v.lbl_calib_status.text()
            c.toggle_live_capture()              # остановить сбор
            QtCore.QTimer.singleShot(700, tick)
            return
        if state["phase"] == 2:
            state["snap"]["use_after"] = v._compass_use
            app.quit()
            return
        if sim.poll() is not None and el < RUN_SEC - 5:
            print("  (симулятор завершился раньше времени)")
        QtCore.QTimer.singleShot(500, tick)

    QtCore.QTimer.singleShot(500, tick)
    app.exec()
    v.stop()

    s = state["snap"]
    ok = True
    print(f"\n== Ж: адаптив на живом потоке ({RUN_SEC:.0f} с) ==")
    print(f"  R̂ = {s['R_eff']:.4f} м² (паспорт {vario_app.R_DEFAULT}), "
          f"множители R×{s['R_mult']:.2f} Q×{s['Q_mult']:.2f}")
    print(f"  индикатор: «{s['label']}»")
    ok &= check("R̂ ушёл от паспортного вниз (к фактическому шуму баро)",
                s["R_eff"] < 0.6 * vario_app.R_DEFAULT,
                f"{s['R_eff']:.4f} < 0.6·{vario_app.R_DEFAULT}")
    ok &= check("индикатор состояния присутствует",
                ("активен" in s["label"]) or ("заморожен" in s["label"]),
                s["label"])
    print(f"\n== Б.2: живой сбор двух источников ==")
    print(f"  буферы: raw {s['n_raw']} · android {s['n_and']} точек")
    ok &= check("оба буфера наполняются", s["n_raw"] > 1000 and s["n_and"] > 1000)
    er, ea = s["ekf_raw"], s["ekf_and"]
    print(f"  EKF raw: {er}")
    print(f"  EKF android: {ea}")
    ok &= check("пункты Live-EKF активны при сборе", s["live_enabled"])
    ok &= check("выбор live@android применился к компасу",
                s["use_live"] == "live@android", s["use_live"])
    print(f"  индикатор под компасом: «{s['ind_live']}»")
    ok &= check("остановка сбора ЯВНО вернула ransac@android",
                s["use_after"] == "ransac@android", s["use_after"])
    log = os.path.join(ROOT, "data", "logs")
    if os.path.isdir(log):
        files = sorted(os.listdir(log))
        if files:
            tail = open(os.path.join(log, files[-1]), encoding="utf-8").read()
            n_adapt = tail.count("adapt:")
            print(f"  (лог {files[-1]}: событий адаптации {n_adapt})")
            ok &= check("адаптация пишется в лог", n_adapt > 0)
    print("\nЖ+Б.2 (live) ПРОЙДЕН ✓" if ok else "\nЖ+Б.2 (live): ПРОВАЛЫ ✗")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
