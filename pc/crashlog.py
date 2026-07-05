# -*- coding: utf-8 -*-
"""
crashlog.py
===========
КРАШ-ЛОГ ПУЛЬТА (пакет 14, блок Д.1). Один вызов setup() в начале main.py:

  • faulthandler — «жёсткие» падения интерпретатора (segfault в Qt/OpenGL/аудио)
    пишут трейсбек всех потоков прямо в файл;
  • sys.excepthook — необработанные исключения Python (главный поток);
  • threading.excepthook — необработанные исключения в фоновых потоках
    (читатель потока, аудио и т.п.);
  • Qt message handler — qWarning/qCritical/qFatal от Qt (часто предвестники
    падения: «QObject::connect …», «Timers cannot be stopped from another thread»).

Файл: data\\logs\\variopro_ГГГГММДД.log (дозапись). При следующем вылете причина
будет в файле. Сюда же пишутся служебные события пульта (watchdog, адаптация R/Q):
log_event("watchdog", "...").
"""

from __future__ import annotations

import datetime
import faulthandler
import io
import os
import sys
import threading
import traceback

PC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(PC_DIR)
LOG_DIR = os.path.join(ROOT, "data", "logs")

_fh: io.TextIOBase | None = None          # общий файл лога (append)
_fault_fh = None                          # отдельный дескриптор для faulthandler
_lock = threading.Lock()


def log_path() -> str:
    return os.path.join(LOG_DIR, datetime.datetime.now().strftime("variopro_%Y%m%d.log"))


def _write(kind: str, text: str) -> None:
    """Строка в лог: время + вид события. Никогда не роняет вызывающего."""
    global _fh
    if _fh is None:
        return
    try:
        with _lock:
            stamp = datetime.datetime.now().strftime("%H:%M:%S")
            _fh.write(f"[{stamp}] {kind}: {text}\n")
            _fh.flush()
    except Exception:
        pass


def log_event(kind: str, text: str) -> None:
    """Служебное событие пульта (watchdog, адаптация R/Q, ошибки аудио…)."""
    _write(kind, text)


def setup() -> str | None:
    """Включить все ловушки. Возвращает путь лога (None, если файл не открылся)."""
    global _fh, _fault_fh
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        path = log_path()
        _fh = open(path, "a", encoding="utf-8", buffering=1)
        # отдельный бинарно-совместимый дескриптор для faulthandler (он пишет
        # напрямую в fd и не должен конфликтовать с текстовыми записями)
        _fault_fh = open(path, "a", encoding="utf-8")
        faulthandler.enable(file=_fault_fh, all_threads=True)
    except OSError:
        _fh = None
        return None

    _write("start", f"пульт запущен (python {sys.version.split()[0]}, pid {os.getpid()})")

    # --- необработанные исключения Python (главный поток) ---
    prev_hook = sys.excepthook

    def _excepthook(tp, val, tb):
        _write("CRASH", "необработанное исключение:\n"
               + "".join(traceback.format_exception(tp, val, tb)))
        if prev_hook is not None:
            prev_hook(tp, val, tb)

    sys.excepthook = _excepthook

    # --- необработанные исключения в фоновых потоках ---
    prev_thook = threading.excepthook

    def _thread_hook(args):
        _write("CRASH-THREAD", f"поток {args.thread.name}:\n"
               + "".join(traceback.format_exception(args.exc_type, args.exc_value,
                                                    args.exc_traceback)))
        if prev_thook is not None:
            prev_thook(args)

    threading.excepthook = _thread_hook

    # --- сообщения Qt (qWarning/qCritical/qFatal) ---
    try:
        from PySide6 import QtCore

        def _qt_handler(mode, ctx, message):
            try:
                sev = {QtCore.QtMsgType.QtWarningMsg: "qt-warning",
                       QtCore.QtMsgType.QtCriticalMsg: "QT-CRITICAL",
                       QtCore.QtMsgType.QtFatalMsg: "QT-FATAL"}.get(mode)
                if sev is not None:            # info/debug не пишем — шумно
                    _write(sev, str(message))
            except Exception:
                pass

        QtCore.qInstallMessageHandler(_qt_handler)
    except Exception:
        pass

    return path
