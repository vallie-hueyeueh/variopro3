# -*- coding: utf-8 -*-
"""Пакет 14, Д.4 — прогон 30 минут: симулятор играет файл по кругу (клиент
переподключается после конца файла — как в жизни), настоящий пульт (MainConsole
offscreen) принимает. Критерии: без падений; память не растёт (первая → вторая
половина прогона); в краш-логе нет CRASH/QT-CRITICAL.

Запуск: PYTHONIOENCODING=utf-8 QT_QPA_PLATFORM=offscreen python scripts/pack14_d4_soak.py [минут]
"""
import ctypes
import ctypes.wintypes as wt
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

PORT = 5558
FILE = os.path.join(ROOT, "data", "session_2026-07-04_14-51-44.csv")
MINUTES = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0


class PMC(ctypes.Structure):
    _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t)]


_K32 = ctypes.WinDLL("kernel32", use_last_error=True)
_GPMI = _K32.K32GetProcessMemoryInfo
_GPMI.argtypes = [wt.HANDLE, ctypes.POINTER(PMC), wt.DWORD]
_GPMI.restype = wt.BOOL


def rss_mb() -> float:
    pmc = PMC()
    pmc.cb = ctypes.sizeof(PMC)
    ok = _GPMI(_K32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
    return (pmc.WorkingSetSize / (1024.0 * 1024.0)) if ok else float("nan")


def main():
    sim = subprocess.Popen(
        [sys.executable, os.path.join(ROOT, "pc", "stream_simulator.py"),
         "--file", FILE, "--port", str(PORT)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1.0)
    import crashlog
    try:
        log_start = os.path.getsize(crashlog.log_path())
    except OSError:
        log_start = 0
    import main as main_mod
    app = QtWidgets.QApplication([])
    win = main_mod.MainConsole()
    win.vario._save_config = lambda *a, **k: None
    v = win.vario
    v.combo_source.setCurrentText("Поток (Bluetooth/симулятор)")
    v.edit_stream_url.setText(f"socket://127.0.0.1:{PORT}")
    v.start()
    # звук на (синтез крутится в колбэке даже без устройства — если аудио нет,
    # честно живём без него; ошибок быть не должно)
    win.sound.btn_mute.setChecked(True)

    # ПРОГРЕСС МЕРИМ ПРИНЯТЫМИ ПАКЕТАМИ (минуты ДАННЫХ), а не настенными
    # часами: если машину/песочницу усыпили — прогон честно продолжится после
    # пробуждения, не насчитав себе «пустых» минут
    RATE_HZ = 84.0                      # факт. темп симулятора для этого файла
    goal_packets = MINUTES * 60.0 * RATE_HZ
    mem = []       # (мин данных, МБ)
    state = {"last_mark": -1}

    def sample():
        src = v.worker.source if v.worker is not None else None
        rec = getattr(src, "received", 0) if src is not None else 0
        data_min = rec / (RATE_HZ * 60.0)
        if int(data_min) > state["last_mark"]:
            state["last_mark"] = int(data_min)
            mem.append((data_min, rss_mb()))
            print(f"  данные {data_min:5.1f} мин  RSS {mem[-1][1]:7.1f} МБ  "
                  f"пакетов {rec}", flush=True)
        if rec >= goal_packets:
            app.quit()
            return
        QtCore.QTimer.singleShot(5_000, sample)

    print(f"прогон {MINUTES:.0f} мин ДАННЫХ (~{goal_packets:.0f} пакетов): "
          f"симулятор {os.path.basename(FILE)} по кругу, порт {PORT}")
    QtCore.QTimer.singleShot(5_000, sample)
    app.exec()
    v.stop()
    sim.kill()

    half = len(mem) // 2
    m1 = sum(m for _, m in mem[1:half]) / max(half - 1, 1)
    m2 = sum(m for _, m in mem[half:]) / max(len(mem) - half, 1)
    grow = m2 - m1
    print(f"\nсредняя память: 1-я половина {m1:.1f} МБ, 2-я {m2:.1f} МБ "
          f"(рост {grow:+.1f} МБ)")
    # краш-лог: смотрим ТОЛЬКО добавленное за время прогона
    bad = 0
    try:
        with open(crashlog.log_path(), encoding="utf-8") as fh:
            fh.seek(log_start)
            txt = fh.read()
        bad = txt.count("CRASH") + txt.count("QT-CRITICAL") + txt.count("QT-FATAL")
    except OSError:
        pass
    ok = grow < 15.0 and bad == 0
    print(f"CRASH/QT-CRITICAL в логе: {bad}")
    print("Д.4 ПРОЙДЕН ✓" if ok else "Д.4 ПРОВАЛЕН ✗")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
