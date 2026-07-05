# -*- coding: utf-8 -*-
"""Пакет 15, блок Б.4 — приёмка компаса.

Режимы:
  --files          журнал скачков курса по ФАЙЛАМ (23-21-13 и 14-51-44, оба
                   источника поля): каждый скачок >45°/с с причиной; вердикт —
                   «самопроизвольных» (в покое, без объясняющего события) нет.
  --live N         N секунд ЖИВОГО потока (симулятор socket://127.0.0.1:5555
                   должен уже работать): настоящий VarioApp offscreen, счёт
                   скачков из журнала MEKF и итог.

Запуск: PYTHONIOENCODING=utf-8 QT_QPA_PLATFORM=offscreen \
        python scripts/pack15_b_live.py --files
        python pc/stream_simulator.py --file data/session_2026-07-04_14-51-44.csv &
        python scripts/pack15_b_live.py --live 600
"""
import argparse
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pc"))

import numpy as np  # noqa: E402


def check(name, cond, detail=""):
    print(f"  {'✓' if cond else '✗'} {name}{('  — ' + detail) if detail else ''}")
    return bool(cond)


def run_file(path, source):
    """Прогнать MEKF с магнитометром по файлу, вернуть (mekf, длительность)."""
    import device_calibration as devcal
    import mekf as mekf_mod
    cfg = mekf_mod.load_mekf_config()
    t, F, W, g_ref = mekf_mod.load_session(path)
    arr = np.genfromtxt(path, delimiter=",", names=True)
    names = arr.dtype.names or ()
    cal = devcal.load(mekf_mod.DEVICE_CALIB_PATH)
    F_ref = None
    if source == "android":
        if not all(c in names for c in ("mxa", "mya", "mza")):
            return None, 0.0
        Mc = np.column_stack([arr["mxa"], arr["mya"], arr["mza"]]).astype(float)[:len(t)]
        sec = devcal.mag_section(cal, "android")
        if sec is not None:
            Mc = (Mc - sec["V"]) @ sec["M"].T
            F_ref = sec.get("F")
        if F_ref is None:
            sec_r = devcal.mag_section(cal, "raw")
            F_ref = sec_r.get("F") if sec_r is not None else None
    else:
        Mc = np.column_stack([arr["mx"], arr["my"], arr["mz"]]).astype(float)[:len(t)]
        sec = devcal.mag_section(cal, "raw")
        if sec is None:
            return None, 0.0
        Mc = (Mc - sec["V"]) @ sec["M"].T
        F_ref = sec.get("F")
    m = mekf_mod.make_mekf(cfg, g_ref, mag_F_ref=F_ref)
    for i in range(len(t)):
        m.step(t[i], F[i], W[i], mag_b=Mc[i])
    return m, float(t[-1] - t[0])


def files_mode():
    ok = True
    for name in ("session_2026-07-04_23-21-13.csv",
                 "session_2026-07-04_14-51-44.csv"):
        path = os.path.join(ROOT, "data", name)
        if not os.path.exists(path):
            print(f"({name} нет — пропуск)")
            continue
        for src in ("raw", "android"):
            m, dur = run_file(path, src)
            if m is None:
                print(f"{name} [{src}]: поля нет/нет калибровки — пропуск")
                continue
            jl = m.jump_log
            # «самопроизвольный» = скачок в ПОКОЕ (|ω| < 0.5 рад/с) без
            # объясняющего события (relock после дыры и т.п. видно по причине)
            spont = [j for j in jl if j[3] < 0.5]
            print(f"\n{name} [{src}]: {dur:.0f} с, скачков в журнале {len(jl)}, "
                  f"из них в покое {len(spont)}; маг-коррекций {m.n_mag_upd}, "
                  f"χ²-отказов {m.n_mag_rej_chi2}, |B|-отказов {m.n_mag_rej_field}, "
                  f"relock {m.n_mag_recover}")
            for (tj, dh, dg, wmag, cause) in jl:
                print(f"    t={tj:7.1f} Δкурс {dh:+6.0f}° Δгиро {dg:+6.0f}° "
                      f"|ω|={wmag:4.2f} причина: {cause}")
            ok &= check(f"{src}: самопроизвольных скачков в покое нет",
                        len(spont) == 0, f"{len(spont)}")
    return ok


def live_mode(seconds):
    from PySide6 import QtCore, QtWidgets
    import vario_app
    vario_app.save_config = lambda *a, **k: None
    app = QtWidgets.QApplication([])
    win = vario_app.VarioApp()
    win._save_config = lambda *a, **k: None
    win.mode = "auto"
    win._adaptive_on = True
    win.combo_source.setCurrentText("Поток (Bluetooth/симулятор)")
    win.edit_stream_url.setText("socket://127.0.0.1:5555")
    win.start()
    state = {"received0": None}

    def finish():
        src = win.worker.source if win.worker is not None else None
        rec = getattr(src, "received", 0) if src is not None else 0
        m = win._mekf
        print(f"\n== ЖИВОЙ ПОТОК {seconds} с: пакетов {rec} ==")
        if m is None:
            print("  MEKF не создался (нет IMU в потоке?)")
            app.exit(1)
            return
        jl = list(m.jump_log)
        # «самопроизвольный» = в покое И БЕЗ причины в журнале («?»); скачки с
        # причиной (relock/mag_big/hole) — объяснённая починка курса: на живом
        # симуляторе шов зацикливания файла (телефон «телепортируется») даёт их
        # законно; все печатаются для отчёта
        spont = [j for j in jl if j[3] < 0.5 and j[4] == "?"]
        explained = [j for j in jl if j[4] != "?"]
        print(f"  маг-коррекций {m.n_mag_upd}, χ² {m.n_mag_rej_chi2}, "
              f"|B| {m.n_mag_rej_field}, relock {m.n_mag_recover}, "
              f"скачков {len(jl)} (необъяснённых в покое: {len(spont)}, "
              f"объяснённых: {len(explained)})")
        for (tj, dh, dg, wmag, cause) in jl:
            print(f"    t={tj:7.1f} Δкурс {dh:+6.0f}° Δгиро {dg:+6.0f}° "
                  f"|ω|={wmag:4.2f} причина: {cause}")
        okk = check("живой поток: необъяснённых скачков в покое нет",
                    len(spont) == 0, f"{len(spont)} из {len(jl)}")
        win.stop()
        app.exit(0 if okk else 1)

    QtCore.QTimer.singleShot(int(seconds * 1000), finish)
    return app.exec()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", action="store_true")
    ap.add_argument("--live", type=float, default=0.0)
    args = ap.parse_args()
    if args.files:
        ok = files_mode()
        print("\nБ (файлы) ПРОЙДЕН ✓" if ok else "\nБ (файлы): ПРОВАЛЫ ✗")
        raise SystemExit(0 if ok else 1)
    if args.live > 0:
        raise SystemExit(live_mode(args.live))
    ap.print_help()


if __name__ == "__main__":
    main()
