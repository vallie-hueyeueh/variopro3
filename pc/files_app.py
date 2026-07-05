# -*- coding: utf-8 -*-
"""
files_app.py
============
Вкладка «ЗАПИСИ»: менеджер записей телефона и папки data\\ — скачивать,
открывать, удалять и архивировать калибровки, не лазя по папкам.

Верх — «НА ТЕЛЕФОНЕ» (работает при активном источнике «Поток», протокол v3):
    «Обновить список» → LIST; таблица имя/дата/размер;
    «Скачать в data\\» / «Скачать и воспроизвести» → GET (прогресс, сверка CRC32;
    на время передачи телефон приостанавливает поток данных);
    «Удалить на телефоне» → DEL (с подтверждением).

Низ — «НА ПК» (работает всегда), ПЯТЬ списков:
  • Записи датчиков (session_*.csv)  → «Открыть в Вариометре», экспорт с калибровкой;
  • Записи калибровки (calib_*.json) → «Открыть в Калибровке»;
  • Калибровки прибора (архив data\\device_calibrations\\) → «Применить» / «Удалить»,
    видно, какая сейчас активна (= pc\\calibration.json);
  • Демо-файлы (data\\samples\\) — свёрнутая группа с примерами;
  • Скриншоты (data\\screenshots\\) → «Открыть» / «Удалить».

Все таблицы: сортировка кликом по заголовку (со стрелкой направления),
колонки тянутся мышью, «Имя» растягивается под ширину. Двойной клик по строке
открывает файл в нужной вкладке (CSV → Вариометр, calib-JSON → Калибровка).
"""

from __future__ import annotations

import datetime
import json
import os
import re
import shutil

import numpy as np
from PySide6 import QtCore, QtWidgets

PC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(PC_DIR)
DATA_DIR = os.path.join(ROOT, "data")
SAMPLES_DIR = os.path.join(DATA_DIR, "samples")
DEVCAL_DIR = os.path.join(DATA_DIR, "device_calibrations")
SHOTS_DIR = os.path.join(DATA_DIR, "screenshots")
LAYOUTS_DIR = os.path.join(DATA_DIR, "layouts")      # виды пульта (пакет 15, Е)
DEVICE_CALIB_PATH = os.path.join(PC_DIR, "calibration.json")


def _fmt_size(n: int) -> str:
    if n >= 1 << 20:
        return f"{n / (1 << 20):.1f} МБ"
    if n >= 1 << 10:
        return f"{n / (1 << 10):.0f} КБ"
    return f"{n} Б"


def _fmt_mtime(ts: float) -> str:
    return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


# дата СОЗДАНИЯ из имени файла (session_/calib_/calibration_ГГГГ-ММ-ДД_ЧЧ-ММ-СС)
_NAME_DATE_RE = re.compile(
    r"^(?:session|calib|calibration)_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})")


def _name_date_ts(name: str):
    """Unix-время из имени файла или None (пакет 13, Д.1)."""
    m = _NAME_DATE_RE.match(name)
    if not m:
        return None
    try:
        return datetime.datetime.strptime(
            m.group(1), "%Y-%m-%d_%H-%M-%S").timestamp()
    except ValueError:
        return None


def _display_ts(name: str, mtime: float) -> float:
    """Дата для показа (Д.1): mtime, но если имя несёт дату создания и она
    расходится с mtime больше 2 минут — верим ИМЕНИ (например, телефон отдаёт
    lastModified = конец записи, а создание — в имени)."""
    nd = _name_date_ts(name)
    if nd is not None and abs(nd - float(mtime)) > 120.0:
        return nd
    return float(mtime)


class SortItem(QtWidgets.QTableWidgetItem):
    """Ячейка, сортирующаяся по скрытому значению (UserRole), а не по тексту —
    иначе «9 КБ» оказалось бы «больше» «1.2 МБ», а даты путались бы."""

    def __lt__(self, other):
        a = self.data(QtCore.Qt.UserRole)
        b = other.data(QtCore.Qt.UserRole) if isinstance(other, QtWidgets.QTableWidgetItem) else None
        if a is not None and b is not None:
            try:
                return a < b
            except TypeError:
                pass
        return super().__lt__(other)


class FileTable(QtWidgets.QTableWidget):
    """Таблица файлов: сортировка кликом по заголовку (со стрелкой), колонки
    Interactive (тянутся мышью), колонка «Имя» растягивается при изменении
    размера таблицы (ручная ширина остальных колонок сохраняется)."""

    PATH_ROLE = QtCore.Qt.UserRole + 1     # полный путь файла — в ячейке «Имя»

    def __init__(self, columns=("Имя", "Дата", "Размер")):
        super().__init__(0, len(columns))
        self.setHorizontalHeaderLabels(list(columns))
        h = self.horizontalHeader()
        h.setSectionResizeMode(QtWidgets.QHeaderView.Interactive)
        h.setSortIndicatorShown(True)
        h.setStretchLastSection(False)
        self.setSortingEnabled(True)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.setAlternatingRowColors(True)
        self.verticalHeader().setVisible(False)
        self.setColumnWidth(1, 145)
        self.setColumnWidth(2, 80)
        for c in range(3, len(columns)):
            self.setColumnWidth(c, 90)
        self.sortByColumn(1, QtCore.Qt.DescendingOrder)   # свежие сверху

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        others = sum(self.columnWidth(c) for c in range(1, self.columnCount()))
        self.setColumnWidth(0, max(150, self.viewport().width() - others))

    def fill(self, rows):
        """rows: список dict(name, mtime, size, path[, status]). Сортировку на время
        заполнения выключаем (требование Qt), потом возвращаем."""
        sort_col = self.horizontalHeader().sortIndicatorSection()
        sort_ord = self.horizontalHeader().sortIndicatorOrder()
        self.setSortingEnabled(False)
        self.setRowCount(0)
        for f in rows:
            r = self.rowCount()
            self.insertRow(r)
            it0 = SortItem(f["name"])
            it0.setData(QtCore.Qt.UserRole, f["name"].lower())
            it0.setData(self.PATH_ROLE, f.get("path"))
            it0.setTextAlignment(QtCore.Qt.AlignCenter)    # имя тоже по центру (В.1)
            show_ts = _display_ts(f["name"], f["mtime"])   # дата из имени при расхождении (Д.1)
            it1 = SortItem(_fmt_mtime(show_ts))
            it1.setData(QtCore.Qt.UserRole, float(show_ts))
            it1.setTextAlignment(QtCore.Qt.AlignCenter)    # дата/размер — по центру (Д.2)
            it2 = SortItem(_fmt_size(int(f["size"])))
            it2.setData(QtCore.Qt.UserRole, int(f["size"]))
            it2.setTextAlignment(QtCore.Qt.AlignCenter)
            self.setItem(r, 0, it0)
            self.setItem(r, 1, it1)
            self.setItem(r, 2, it2)
            if self.columnCount() > 3:
                it3 = SortItem(f.get("status", ""))
                it3.setData(QtCore.Qt.UserRole, f.get("status", ""))
                it3.setTextAlignment(QtCore.Qt.AlignCenter)
                self.setItem(r, 3, it3)
        self.setSortingEnabled(True)
        self.sortItems(sort_col, sort_ord)

    def autosize(self):
        """«Автоширина» (Д.2): подогнать колонки под содержимое, «Имя» — растянуть."""
        self.resizeColumnsToContents()
        others = sum(self.columnWidth(c) for c in range(1, self.columnCount()))
        self.setColumnWidth(0, max(150, self.viewport().width() - others))

    def selected_path(self):
        r = self.currentRow()
        if r < 0:
            return None
        it = self.item(r, 0)
        return None if it is None else it.data(self.PATH_ROLE)

    def selected_name(self):
        r = self.currentRow()
        if r < 0:
            return None
        it = self.item(r, 0)
        return None if it is None else it.text()


class FilesPanel(QtWidgets.QWidget):
    """source_provider() → активный источник вкладки «Вариометр» (или None);
    play_cb(path)      — открыть CSV в «Вариометре» и запустить;
    open_calib_cb(path) — открыть файл во вкладке «Калибровка»;
    calib_changed_cb() — активная калибровка сменилась («Применить» из архива)."""

    def __init__(self, source_provider, play_cb=None, open_calib_cb=None,
                 calib_changed_cb=None, connect_cb=None, layout_cb=None):
        super().__init__()
        self.provider = source_provider
        self.play_cb = play_cb
        self.open_calib_cb = open_calib_cb
        self.calib_changed_cb = calib_changed_cb
        self.layout_cb = layout_cb   # Е.4: применить вид пульта (path | None)
        self.connect_cb = connect_cb   # Д.3: открыть поток без проигрывания
        self._dl_action = None      # ("save"|"play", имя, mtime) — по завершении GET
        self._waiting_del = None    # имя файла, чьё удаление ждём
        self._blink_timers = {}     # мигание рамок групп (Б: навигация «Файл…»)
        self.after_apply_cb = None  # Д.1 (пакет 15): one-shot возврат на вкладку
                                    # после успешного «Применить» (ставит main.py)

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        outer.addWidget(scroll)
        body = QtWidgets.QWidget()
        scroll.setWidget(body)
        root = QtWidgets.QVBoxLayout(body)

        # ================= НА ТЕЛЕФОНЕ =================
        gb_ph = QtWidgets.QGroupBox("На телефоне")
        vph = QtWidgets.QVBoxLayout(gb_ph)
        rowb = QtWidgets.QHBoxLayout()
        # Д.3: подключение прямо отсюда — БЕЗ запуска проигрывания на «Вариометре»
        self.btn_connect = QtWidgets.QPushButton("🔌 Подключиться к телефону")
        self.btn_connect.setToolTip(
            "Открыть поток по адресу из поля «Поток» вариометра, НЕ запуская\n"
            "проигрывание — только для списка файлов/скачивания. Если поток уже\n"
            "открыт вариометром — используется он.")
        self.btn_connect.clicked.connect(self._on_connect)
        rowb.addWidget(self.btn_connect)
        self.btn_list = QtWidgets.QPushButton("🔄 Обновить список")
        self.btn_list.clicked.connect(self._on_list)
        rowb.addWidget(self.btn_list)
        self.btn_get = QtWidgets.QPushButton("⬇ Скачать в data\\")
        self.btn_get.clicked.connect(lambda: self._on_get("save"))
        rowb.addWidget(self.btn_get)
        self.btn_get_play = QtWidgets.QPushButton("⬇▶ Скачать и воспроизвести")
        self.btn_get_play.clicked.connect(lambda: self._on_get("play"))
        rowb.addWidget(self.btn_get_play)
        self.btn_del_ph = QtWidgets.QPushButton("🗑 Удалить на телефоне")
        self.btn_del_ph.clicked.connect(self._on_del_phone)
        rowb.addWidget(self.btn_del_ph)
        rowb.addStretch(1)
        vph.addLayout(rowb)

        self.tbl_ph = FileTable()
        self.tbl_ph.setMinimumHeight(110)
        vph.addWidget(self.tbl_ph, stretch=1)

        prow = QtWidgets.QHBoxLayout()
        self.progress = QtWidgets.QProgressBar()
        self.progress.setVisible(False)
        prow.addWidget(self.progress, stretch=1)
        self.lbl_ph_status = QtWidgets.QLabel(
            "Запустите источник «Поток» на «Вариометре» и нажмите «Обновить список».")
        self.lbl_ph_status.setWordWrap(True)
        prow.addWidget(self.lbl_ph_status, stretch=2)
        vph.addLayout(prow)
        root.addWidget(gb_ph, stretch=2)

        # ================= НА ПК =================
        toolbar = QtWidgets.QHBoxLayout()
        btn_refresh = QtWidgets.QPushButton("🔄 Обновить")
        btn_refresh.clicked.connect(self.refresh_local)
        toolbar.addWidget(btn_refresh)
        btn_open = QtWidgets.QPushButton("📂 Открыть папку data\\")
        btn_open.setToolTip("Открыть data\\ в проводнике — например, чтобы отправить файл кому-то")
        btn_open.clicked.connect(lambda: os.startfile(DATA_DIR))
        toolbar.addWidget(btn_open)
        btn_fit = QtWidgets.QPushButton("↔ Автоширина")
        btn_fit.setToolTip("Подогнать колонки всех таблиц под содержимое "
                           "(«Имя» растягивается на остаток)")
        btn_fit.clicked.connect(self._autosize_all)
        toolbar.addWidget(btn_fit)
        toolbar.addStretch(1)
        self.lbl_local_status = QtWidgets.QLabel("")
        self.lbl_local_status.setWordWrap(True)
        toolbar.addWidget(self.lbl_local_status, stretch=2)
        root.addLayout(toolbar)

        # --- Записи датчиков (session_*.csv) + Записи калибровки (calib_*.json) ---
        row1 = QtWidgets.QHBoxLayout()

        self.gb_sessions = QtWidgets.QGroupBox("Записи датчиков (session_*.csv)")
        self.gb_sessions.setObjectName("gbSessions")
        vs = QtWidgets.QVBoxLayout(self.gb_sessions)
        rs = QtWidgets.QHBoxLayout()
        self.btn_open_vario = QtWidgets.QPushButton("▶ Открыть в Вариометре")
        self.btn_open_vario.clicked.connect(self._on_open_session)
        rs.addWidget(self.btn_open_vario)
        self.btn_export_cal = QtWidgets.QPushButton("⚗ Экспорт с калибровкой")
        self.btn_export_cal.setToolTip(
            "Применить АКТИВНУЮ калибровку прибора (аксель: смещение+масштаб,\n"
            "магнитометр: V и W) к выбранной записи и сохранить рядом файл\n"
            "*_calibrated.csv. Сырой файл остаётся истиной — конвертация по требованию.")
        self.btn_export_cal.clicked.connect(self._export_calibrated)
        rs.addWidget(self.btn_export_cal)
        self.btn_del_sess = QtWidgets.QPushButton("🗑 Удалить")
        self.btn_del_sess.clicked.connect(lambda: self._delete_selected(self.tbl_sessions))
        rs.addWidget(self.btn_del_sess)
        rs.addStretch(1)
        vs.addLayout(rs)
        self.tbl_sessions = FileTable()
        self.tbl_sessions.setMinimumHeight(120)
        self.tbl_sessions.doubleClicked.connect(lambda *_: self._on_open_session())
        vs.addWidget(self.tbl_sessions, stretch=1)
        row1.addWidget(self.gb_sessions, stretch=3)

        self.gb_calibrec = QtWidgets.QGroupBox(
            "Сырые записи калибровки — с телефона (calib_*.json)")
        self.gb_calibrec.setObjectName("gbCalibRec")
        vc = QtWidgets.QVBoxLayout(self.gb_calibrec)
        lbl_calibrec_note = QtWidgets.QLabel(
            "Точки датчиков, снятые при вращении телефона («сырьё»). Из них "
            "вкладка «Калибровка» СЧИТАЕТ калибровку прибора.")
        lbl_calibrec_note.setWordWrap(True)
        lbl_calibrec_note.setStyleSheet("color:#8a93a0; font-size:11px;")
        vc.addWidget(lbl_calibrec_note)
        rc = QtWidgets.QHBoxLayout()
        self.btn_open_calib = QtWidgets.QPushButton("🧭 Открыть в Калибровке")
        self.btn_open_calib.clicked.connect(self._on_open_calibrec)
        rc.addWidget(self.btn_open_calib)
        self.btn_del_calibrec = QtWidgets.QPushButton("🗑 Удалить")
        self.btn_del_calibrec.clicked.connect(lambda: self._delete_selected(self.tbl_calibrec))
        rc.addWidget(self.btn_del_calibrec)
        rc.addStretch(1)
        vc.addLayout(rc)
        self.tbl_calibrec = FileTable()
        self.tbl_calibrec.setMinimumHeight(120)
        self.tbl_calibrec.doubleClicked.connect(lambda *_: self._on_open_calibrec())
        vc.addWidget(self.tbl_calibrec, stretch=1)
        row1.addWidget(self.gb_calibrec, stretch=2)
        root.addLayout(row1, stretch=3)

        # --- Калибровки прибора (архив) + демо/скриншоты ---
        row2 = QtWidgets.QHBoxLayout()

        self.gb_dev = gb_dev = QtWidgets.QGroupBox(
            "Готовые калибровки прибора — посчитаны на ПК (calibration_*.json)")
        gb_dev.setObjectName("gbDevCal")      # мигание при навигации (Д.1)
        vd = QtWidgets.QVBoxLayout(gb_dev)
        lbl_dev_note = QtWidgets.QLabel(
            "Итоговые V/W/смещения, посчитанные вкладкой «Калибровка» "
            "(архив data\\device_calibrations\\). «Применить» делает выбранную "
            "АКТИВНОЙ (pc\\calibration.json) — ей пользуются компас и вариометр.")
        lbl_dev_note.setWordWrap(True)
        lbl_dev_note.setStyleSheet("color:#8a93a0; font-size:11px;")
        vd.addWidget(lbl_dev_note)
        rd = QtWidgets.QHBoxLayout()
        self.btn_apply_cal = QtWidgets.QPushButton("✔ Применить")
        self.btn_apply_cal.setToolTip(
            "Сделать выбранную калибровку АКТИВНОЙ: скопировать в pc\\calibration.json.\n"
            "Ей сразу начнут пользоваться компас и вариометр (индикатор под компасом обновится).")
        self.btn_apply_cal.clicked.connect(self._apply_device_calib)
        rd.addWidget(self.btn_apply_cal)
        self.btn_del_devcal = QtWidgets.QPushButton("🗑 Удалить")
        self.btn_del_devcal.clicked.connect(lambda: self._delete_selected(self.tbl_devcal))
        rd.addWidget(self.btn_del_devcal)
        rd.addStretch(1)
        vd.addLayout(rd)
        self.tbl_devcal = FileTable(("Имя", "Дата", "Размер", "Статус"))
        self.tbl_devcal.setMinimumHeight(100)
        self.tbl_devcal.doubleClicked.connect(lambda *_: self._apply_device_calib())
        vd.addWidget(self.tbl_devcal, stretch=1)
        self.lbl_active_cal = QtWidgets.QLabel("Активная: —")
        self.lbl_active_cal.setWordWrap(True)
        vd.addWidget(self.lbl_active_cal)
        row2.addWidget(gb_dev, stretch=3)

        rcol = QtWidgets.QVBoxLayout()
        # демо-файлы: свёрнутая группа
        self.gb_samples = QtWidgets.QGroupBox("Демо-файлы (data\\samples\\) — «демо», развернуть")
        self.gb_samples.setCheckable(True)
        self.gb_samples.setChecked(False)
        vsm = QtWidgets.QVBoxLayout(self.gb_samples)
        self.samples_body = QtWidgets.QWidget()
        vsb = QtWidgets.QVBoxLayout(self.samples_body)
        vsb.setContentsMargins(0, 0, 0, 0)
        self.tbl_samples = FileTable()
        self.tbl_samples.setMinimumHeight(90)
        self.tbl_samples.doubleClicked.connect(lambda *_: self._on_open_sample())
        vsb.addWidget(self.tbl_samples)
        rsm = QtWidgets.QHBoxLayout()
        btn_open_sample = QtWidgets.QPushButton("Открыть (по типу файла)")
        btn_open_sample.setToolTip("CSV → Вариометр, JSON → Калибровка")
        btn_open_sample.clicked.connect(self._on_open_sample)
        rsm.addWidget(btn_open_sample)
        rsm.addStretch(1)
        vsb.addLayout(rsm)
        vsm.addWidget(self.samples_body)
        self.samples_body.setVisible(False)
        self.gb_samples.toggled.connect(self.samples_body.setVisible)
        rcol.addWidget(self.gb_samples)

        # виды пульта (пакет 15, Е.4): сохранённые компоновки карточек шапки
        self.gb_layouts = QtWidgets.QGroupBox("Виды пульта (data\\layouts\\)")
        self.gb_layouts.setObjectName("gbLayouts")
        vl = QtWidgets.QVBoxLayout(self.gb_layouts)
        rl = QtWidgets.QHBoxLayout()
        btn_lay_apply = QtWidgets.QPushButton("✔ Применить")
        btn_lay_apply.setToolTip("Применить выбранный вид к шапке «Вариометра»\n"
                                 "(запоминается в config — переживает перезапуск)")
        btn_lay_apply.clicked.connect(self._on_apply_layout)
        rl.addWidget(btn_lay_apply)
        btn_lay_del = QtWidgets.QPushButton("🗑 Удалить")
        btn_lay_del.clicked.connect(lambda: self._delete_selected(self.tbl_layouts))
        rl.addWidget(btn_lay_del)
        btn_lay_factory = QtWidgets.QPushButton("Заводской вид")
        btn_lay_factory.setToolTip("Вернуть карточки шапки в исходную раскладку\n"
                                   "с оригинальными названиями")
        btn_lay_factory.clicked.connect(self._on_factory_layout)
        rl.addWidget(btn_lay_factory)
        rl.addStretch(1)
        vl.addLayout(rl)
        self.tbl_layouts = FileTable()
        self.tbl_layouts.setMinimumHeight(80)
        self.tbl_layouts.doubleClicked.connect(lambda *_: self._on_apply_layout())
        vl.addWidget(self.tbl_layouts, stretch=1)
        rcol.addWidget(self.gb_layouts)

        # скриншоты
        gb_shots = QtWidgets.QGroupBox("Скриншоты (data\\screenshots\\)")
        vsh = QtWidgets.QVBoxLayout(gb_shots)
        rsh = QtWidgets.QHBoxLayout()
        btn_shot_open = QtWidgets.QPushButton("Открыть")
        btn_shot_open.clicked.connect(self._on_open_shot)
        rsh.addWidget(btn_shot_open)
        btn_shot_del = QtWidgets.QPushButton("🗑 Удалить")
        btn_shot_del.clicked.connect(lambda: self._delete_selected(self.tbl_shots))
        rsh.addWidget(btn_shot_del)
        rsh.addStretch(1)
        vsh.addLayout(rsh)
        self.tbl_shots = FileTable()
        self.tbl_shots.setMinimumHeight(90)
        self.tbl_shots.doubleClicked.connect(lambda *_: self._on_open_shot())
        vsh.addWidget(self.tbl_shots, stretch=1)
        rcol.addWidget(gb_shots, stretch=1)

        row2.addLayout(rcol, stretch=2)
        root.addLayout(row2, stretch=2)

        self.refresh_local()

        # поллинг источника: список/прогресс/результаты
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(250)
        self.timer.timeout.connect(self._poll)
        self.timer.start()

    # ------------------------------------------------------------------
    def _src(self):
        src = self.provider() if self.provider else None
        return src if (src is not None and getattr(src, "live", False)) else None

    def _on_connect(self):
        """Д.3: «Подключиться к телефону» — открыть поток без проигрывания."""
        if self._src() is not None:
            self.lbl_ph_status.setText("Поток уже открыт — используется он. "
                                       "Нажмите «Обновить список».")
            return
        if self.connect_cb is None:
            self.lbl_ph_status.setText("Подключение недоступно (нет связки с вариометром).")
            return
        self.lbl_ph_status.setText(self.connect_cb())

    def _autosize_all(self):
        """Д.2: «Автоширина» — по всем таблицам вкладки."""
        for tbl in (self.tbl_ph, self.tbl_sessions, self.tbl_calibrec,
                    self.tbl_devcal, self.tbl_samples, self.tbl_shots):
            tbl.autosize()

    # ---------------- телефонная секция ----------------
    def _on_list(self):
        src = self._src()
        if src is None:
            self.lbl_ph_status.setText(
                "Нет активного потока: на «Вариометре» выберите «Поток» и нажмите «Старт».")
            return
        if src.request_list():
            self.lbl_ph_status.setText("Запросил список…")
        else:
            self.lbl_ph_status.setText("Нет связи — список не запрошен.")

    def _on_get(self, action: str):
        src = self._src()
        name = self.tbl_ph.selected_name()
        if src is None or not name:
            self.lbl_ph_status.setText("Выберите файл в списке (и нужен активный поток).")
            return
        dest = os.path.join(DATA_DIR, name)
        if os.path.exists(dest):
            btn = QtWidgets.QMessageBox.question(
                self, "Файл уже есть",
                f"{name} уже есть в data\\ — перезаписать?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
            if btn != QtWidgets.QMessageBox.Yes:
                return
        # телефонное время файла — чтобы выставить mtime скачанному (Д.1)
        ph_mtime = None
        for f in (src.file_list or []):
            if f.get("name") == name:
                ph_mtime = f.get("mtime")
                break
        self._dl_action = (action, name, ph_mtime)
        if src.request_get(name):
            self.progress.setVisible(True)
            self.progress.setValue(0)
            self.lbl_ph_status.setText(f"Передача файла {name}…")
        else:
            self._dl_action = None
            self.lbl_ph_status.setText("Нет связи — передача не началась.")

    def _on_del_phone(self):
        src = self._src()
        name = self.tbl_ph.selected_name()
        if src is None or not name:
            self.lbl_ph_status.setText("Выберите файл в списке (и нужен активный поток).")
            return
        btn = QtWidgets.QMessageBox.question(
            self, "Удалить на телефоне?",
            f"Удалить {name} С ТЕЛЕФОНА безвозвратно?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if btn != QtWidgets.QMessageBox.Yes:
            return
        self._waiting_del = name
        if not src.request_del(name):
            self._waiting_del = None
            self.lbl_ph_status.setText("Нет связи — удаление не отправлено.")

    def _poll(self):
        src = self._src()
        for w in (self.btn_list, self.btn_get, self.btn_get_play, self.btn_del_ph):
            w.setEnabled(src is not None)
        if src is None:
            return
        # список получен?
        if src.file_list is not None and getattr(self, "_shown_list_wall", None) != src.file_list_wall:
            self._shown_list_wall = src.file_list_wall
            self.tbl_ph.fill([{"name": f["name"], "mtime": f["mtime"],
                               "size": f["size"], "path": f["name"]}
                              for f in src.file_list])
            self.lbl_ph_status.setText(f"На телефоне {len(src.file_list)} файлов.")
        # прогресс передачи
        pr = src.download_progress
        if pr is not None:
            got, total, name = pr
            self.progress.setVisible(True)
            self.progress.setMaximum(max(total, 1))
            self.progress.setValue(got)
            self.lbl_ph_status.setText(
                f"Передача файла {name}… {got * 100 // max(total, 1)}%")
        # передача завершена?
        res = src.download_result
        if res is not None and self._dl_action is not None:
            src.download_result = None
            action, name, ph_mtime = self._dl_action
            self._dl_action = None
            self.progress.setVisible(False)
            if "error" in res:
                self.lbl_ph_status.setText(f"Ошибка передачи: {res['error']}")
            elif not res.get("crc_ok") or not res.get("size_ok"):
                self.lbl_ph_status.setText(
                    f"{name}: КОНТРОЛЬНАЯ СУММА НЕ СОШЛАСЬ — файл не сохранён, повторите.")
            else:
                dest = os.path.join(DATA_DIR, res["name"])
                try:
                    with open(dest, "wb") as fh:
                        fh.write(res["data"])
                except OSError as e:
                    self.lbl_ph_status.setText(f"Не удалось сохранить: {e}")
                    return
                # Д.1: mtime скачанного = времени СОЗДАНИЯ на телефоне: из имени
                # (там штамп начала записи), иначе — телефонное время файла.
                # Так дата в списках и в Проводнике совпадает с телефоном.
                ts = _name_date_ts(res["name"])
                if ts is None and ph_mtime:
                    ts = float(ph_mtime)
                if ts is not None:
                    try:
                        os.utime(dest, (ts, ts))
                    except OSError:
                        pass
                self.lbl_ph_status.setText(
                    f"✓ {res['name']} скачан ({_fmt_size(len(res['data']))}, CRC32 сошёлся) → data\\")
                self.refresh_local()
                if action == "play" and self.play_cb:
                    self.play_cb(dest)
        # удаление завершено?
        if src.del_result is not None and self._waiting_del is not None:
            ok, why = src.del_result
            src.del_result = None
            name = self._waiting_del
            self._waiting_del = None
            if ok == "OK":
                self.lbl_ph_status.setText(f"✓ {name} удалён на телефоне.")
                src.request_list()      # сразу освежить список
            else:
                self.lbl_ph_status.setText(f"Удаление не удалось: {why}")

    # ---------------- локальная секция ----------------
    @staticmethod
    def _scan(directory, keep=None):
        """Список файлов каталога как dict-строки для FileTable."""
        rows = []
        try:
            names = os.listdir(directory)
        except OSError:
            return rows
        for n in names:
            fp = os.path.join(directory, n)
            if not os.path.isfile(fp):
                continue
            if keep is not None and not keep(n):
                continue
            try:
                st = os.stat(fp)
            except OSError:
                continue
            rows.append({"name": n, "mtime": st.st_mtime, "size": st.st_size, "path": fp})
        return rows

    def refresh_local(self):
        """Пересканировать data\\: пять списков + миграция демо-файлов + актив калибровки."""
        for d in (SAMPLES_DIR, DEVCAL_DIR, SHOTS_DIR):
            os.makedirs(d, exist_ok=True)
        # одноразовая миграция: демо-файлы sample_* из data\ → data\samples\
        try:
            for n in os.listdir(DATA_DIR):
                if n.startswith("sample_") and os.path.isfile(os.path.join(DATA_DIR, n)):
                    dst = os.path.join(SAMPLES_DIR, n)
                    if not os.path.exists(dst):
                        shutil.move(os.path.join(DATA_DIR, n), dst)
                    else:
                        os.remove(os.path.join(DATA_DIR, n))
        except OSError:
            pass

        self.tbl_sessions.fill(self._scan(
            DATA_DIR, lambda n: n.lower().endswith(".csv") and not n.startswith("sample_")))
        self.tbl_calibrec.fill(self._scan(
            DATA_DIR, lambda n: n.lower().endswith(".json") and not n.startswith("sample_")))
        self.tbl_samples.fill(self._scan(SAMPLES_DIR))
        self.tbl_shots.fill(self._scan(SHOTS_DIR, lambda n: n.lower().endswith(".png")))
        os.makedirs(LAYOUTS_DIR, exist_ok=True)
        self.tbl_layouts.fill(self._scan(
            LAYOUTS_DIR, lambda n: n.lower().endswith(".json")))
        self._refresh_devcal()

    def _refresh_devcal(self):
        """Архив калибровок прибора + отметка активной (= pc\\calibration.json)."""
        active = None
        try:
            with open(DEVICE_CALIB_PATH, "r", encoding="utf-8") as fh:
                active = json.load(fh)
        except (OSError, ValueError):
            pass
        rows = self._scan(DEVCAL_DIR, lambda n: n.lower().endswith(".json"))
        active_name = None
        for r in rows:
            try:
                with open(r["path"], "r", encoding="utf-8") as fh:
                    d = json.load(fh)
            except (OSError, ValueError):
                d = None
            if active is not None and d == active:
                r["status"] = "✔ активна"
                active_name = r["name"]
            else:
                r["status"] = ""
        # активная калибровка есть, но копии в архиве нет — доложить её в архив,
        # чтобы список был полным (архив применённых калибровок)
        if active is not None and active_name is None:
            stamp = str(active.get("created", "")).replace(" ", "_").replace(":", "-")
            if not stamp:
                stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            dst = os.path.join(DEVCAL_DIR, f"calibration_{stamp}.json")
            try:
                if not os.path.exists(dst):
                    shutil.copyfile(DEVICE_CALIB_PATH, dst)
                    st = os.stat(dst)
                    rows.append({"name": os.path.basename(dst), "mtime": st.st_mtime,
                                 "size": st.st_size, "path": dst, "status": "✔ активна"})
                    active_name = os.path.basename(dst)
            except OSError:
                pass
        self.tbl_devcal.fill(rows)
        if active is None:
            self.lbl_active_cal.setText("Активная: НЕТ (pc\\calibration.json отсутствует)")
            self.lbl_active_cal.setStyleSheet("color:#c0392b; font-weight:bold;")
        else:
            mag = active.get("mag") or {}
            method = {"ellipsoid": "RANSAC", "ekf": "EKF"}.get(mag.get("model"),
                                                               mag.get("model") or "?")
            res = mag.get("residual_pct")
            res_txt = f", остаток {float(res):.1f}%" if res is not None else ""
            name_txt = f" — файл {active_name}" if active_name else ""
            self.lbl_active_cal.setText(
                f"Активная: {method} от {active.get('created', '?')}{res_txt}{name_txt}")
            self.lbl_active_cal.setStyleSheet("")

    # -------- открытие по типу файла --------
    def _on_open_session(self):
        p = self.tbl_sessions.selected_path()
        if p and self.play_cb:
            self.play_cb(p)

    def _on_open_calibrec(self):
        p = self.tbl_calibrec.selected_path()
        if p and self.open_calib_cb:
            self.open_calib_cb(p)

    def _on_open_sample(self):
        p = self.tbl_samples.selected_path()
        if not p:
            return
        if p.lower().endswith(".csv") and self.play_cb:
            self.play_cb(p)
        elif p.lower().endswith(".json") and self.open_calib_cb:
            self.open_calib_cb(p)

    def _on_apply_layout(self):
        """Е.4: применить выбранный вид пульта к шапке «Вариометра»."""
        p = self.tbl_layouts.selected_path()
        if not p:
            self.lbl_local_status.setText("Выберите вид в списке «Виды пульта».")
            return
        if self.layout_cb is None:
            self.lbl_local_status.setText("Применение недоступно (нет связки).")
            return
        self.layout_cb(p)
        self.lbl_local_status.setText(f"✓ Вид пульта применён: {os.path.basename(p)}")

    def _on_factory_layout(self):
        """Е.4: вернуть заводскую раскладку карточек."""
        if self.layout_cb is None:
            return
        self.layout_cb(None)
        self.lbl_local_status.setText("✓ Вид пульта: заводской.")

    def _on_open_shot(self):
        p = self.tbl_shots.selected_path()
        if p:
            try:
                os.startfile(p)
            except OSError as e:
                self.lbl_local_status.setText(f"Не удалось открыть: {e}")

    def _delete_selected(self, tbl: FileTable):
        p = tbl.selected_path()
        if not p:
            return
        btn = QtWidgets.QMessageBox.question(
            self, "Удалить файл?",
            f"Удалить {os.path.basename(p)} безвозвратно?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if btn == QtWidgets.QMessageBox.Yes:
            try:
                os.remove(p)
            except OSError as e:
                QtWidgets.QMessageBox.warning(self, "Ошибка", str(e))
            self.refresh_local()

    # -------- архив калибровок прибора --------
    def _apply_device_calib(self):
        """«Применить»: выбранный файл архива становится активной калибровкой.
        Д.6 + пакет 14 (Б.2): файлы v1 мигрируются в v2 (mag → mag_raw/
        mag_android); если в файле НЕТ каких-то секций (accel / mag_raw /
        mag_android / gyro_bias) — они НЕ теряются молча: берутся из прежней
        активной, с предупреждением и пометкой смешанного источника."""
        p = self.tbl_devcal.selected_path()
        if not p:
            self.lbl_local_status.setText("Выберите калибровку в списке архива.")
            return
        try:
            with open(p, "r", encoding="utf-8") as fh:
                new = json.load(fh)
            if not isinstance(new, dict):
                raise ValueError("это не файл калибровки")
        except (OSError, ValueError) as e:
            QtWidgets.QMessageBox.warning(self, "Ошибка", f"Не удалось применить: {e}")
            return
        import device_calibration as devcal
        new = devcal.normalize(new)               # v1 → v2 (в памяти)
        cur = devcal.load(DEVICE_CALIB_PATH)      # прежняя активная (v2)
        kept = []      # что взяли из прежней активной
        for key, label in (("accel", "аксель"),
                           ("mag_raw", "магнитометр raw"),
                           ("mag_android", "магнитометр android")):
            if not new.get(key) and cur.get(key):
                new[key] = cur[key]
                kept.append(label)
        if new.get("gyro_bias") is None and cur.get("gyro_bias") is not None:
            new["gyro_bias"] = cur["gyro_bias"]
            kept.append("гироскоп")
        try:
            with open(DEVICE_CALIB_PATH, "w", encoding="utf-8") as fh:
                json.dump(new, fh, ensure_ascii=False, indent=2)
        except OSError as e:
            QtWidgets.QMessageBox.warning(self, "Ошибка", f"Не удалось применить: {e}")
            return
        self._refresh_devcal()
        if self.calib_changed_cb:
            self.calib_changed_cb()               # обновить индикатор под компасом
        # Д.1 (пакет 15): пришли сюда кнопкой «Применить готовую калибровку…»
        # с «Калибровки» → после успешного применения вернуться туда (one-shot)
        cb = self.after_apply_cb
        self.after_apply_cb = None
        if cb is not None:
            try:
                cb()
            except Exception:
                pass
        if kept:
            QtWidgets.QMessageBox.warning(
                self, "Смешанная калибровка",
                f"В файле {os.path.basename(p)} нет секций: {', '.join(kept)}.\n\n"
                f"Эти секции ОСТАВЛЕНЫ из прежней активной калибровки —\n"
                f"иначе вариометр/компас потеряли бы рабочие настройки.")
            self.lbl_local_status.setText(
                f"✓ Применена {os.path.basename(p)} — СМЕШАННЫЙ источник: "
                f"{', '.join(kept)} из прежней активной.")
        else:
            self.lbl_local_status.setText(
                f"✓ Применена калибровка: {os.path.basename(p)}")

    # -------- экспорт калиброванного CSV --------
    def _export_calibrated(self):
        """Применить АКТИВНУЮ калибровку прибора к выбранной session-записи и
        сохранить *_calibrated.csv рядом. Сырой файл не меняется."""
        p = self.tbl_sessions.selected_path()
        if not p or not p.lower().endswith(".csv"):
            self.lbl_local_status.setText("Выберите session_*.csv в списке записей датчиков.")
            return
        import device_calibration as devcal
        cal = devcal.load(DEVICE_CALIB_PATH)      # v1 мигрируется на чтении
        if not cal:
            QtWidgets.QMessageBox.warning(
                self, "Нет калибровки",
                "Нет активной калибровки прибора (pc\\calibration.json).\n"
                "Сначала сохраните её во вкладке «Калибровка».")
            return
        acc = cal.get("accel") or {}
        # экспорт калибрует СЫРЫЕ колонки mx..mz → секция mag_raw (v2)
        mag = cal.get("mag_raw") or {}
        try:
            arr = np.genfromtxt(p, delimiter=",", names=True)
            names = list(arr.dtype.names or [])
            need = ("ax", "ay", "az", "mx", "my", "mz")
            if not all(c in names for c in need):
                raise ValueError("в файле нет колонок IMU (ax..az, mx..mz)")
            cols = {n: np.asarray(arr[n], dtype=float) for n in names}
            # аксель: (сырое − offset) · scales (диагональная модель)
            rep = []
            if "offset" in acc and "scales" in acc:
                off = np.asarray(acc["offset"], float)
                scl = np.asarray(acc["scales"], float)
                A = np.column_stack([cols["ax"], cols["ay"], cols["az"]])
                A = (A - off) * scl
                cols["ax"], cols["ay"], cols["az"] = A[:, 0], A[:, 1], A[:, 2]
                rep.append("аксель")
            # магнитометр: m = W·(сырое − V)
            if "hard_iron" in mag and "soft_iron" in mag:
                V = np.asarray(mag["hard_iron"], float)
                W = np.asarray(mag["soft_iron"], float)
                M = np.column_stack([cols["mx"], cols["my"], cols["mz"]])
                M = (M - V) @ W.T
                cols["mx"], cols["my"], cols["mz"] = M[:, 0], M[:, 1], M[:, 2]
                rep.append("магнитометр")
            if not rep:
                raise ValueError("в активной калибровке нет ни акселя, ни магнитометра")
            out = np.column_stack([cols[n] for n in names])
            base, ext = os.path.splitext(p)
            dst = base + "_calibrated" + ext
            fmt = ["%.6f" if n in ("ax", "ay", "az", "gx", "gy", "gz") else "%.4f"
                   for n in names]
            np.savetxt(dst, out, delimiter=",", header=",".join(names),
                       comments="", fmt=fmt)
        except (OSError, ValueError) as e:
            QtWidgets.QMessageBox.warning(self, "Ошибка экспорта", str(e))
            return
        # контроль качества: |a| должен лечь на g, |B| — на F
        a_norm = float(np.mean(np.linalg.norm(
            np.column_stack([cols["ax"], cols["ay"], cols["az"]]), axis=1)))
        b_norm = float(np.mean(np.linalg.norm(
            np.column_stack([cols["mx"], cols["my"], cols["mz"]]), axis=1)))
        g_ref = acc.get("target_g")
        f_ref = mag.get("target_F_uT")
        g_txt = f" (цель g = {float(g_ref):.3f})" if g_ref else ""
        f_txt = f" (цель F = {float(f_ref):.1f})" if f_ref else ""
        self.refresh_local()
        QtWidgets.QMessageBox.information(
            self, "Экспорт с калибровкой",
            f"Сохранено: {os.path.basename(dst)}\n"
            f"Применено: {', '.join(rep)}.\n\n"
            f"Средний |a| = {a_norm:.3f} м/с²{g_txt}\n"
            f"Средний |B| = {b_norm:.1f} мкТл{f_txt}\n\n"
            "Если запись сделана в чистом месте, |a| ≈ g и |B| ≈ F. Большое\n"
            "расхождение |B| — магнитная обстановка записи ≠ обстановке калибровки.")
        self.lbl_local_status.setText(f"✓ Экспортировано: {os.path.basename(dst)}")

    # -------- мигающая рамка (навигация «Файл…» / «Загрузить файл…») --------
    def highlight(self, kind: str):
        """Мигнуть зелёной рамкой нужного списка 3 раза:
        kind = 'session' | 'calib' | 'devcal' (Д.1) | 'layouts' (Е.4)."""
        gb = {"session": self.gb_sessions, "calib": self.gb_calibrec,
              "devcal": self.gb_dev,
              "layouts": getattr(self, "gb_layouts", None)}.get(kind)
        if gb is None:
            return
        old = self._blink_timers.pop(kind, None)
        if old is not None:
            old.stop()
            gb.setStyleSheet("")
        name = gb.objectName()
        state = {"n": 0}
        timer = QtCore.QTimer(self)

        def tick():
            on = state["n"] % 2 == 0
            gb.setStyleSheet(
                f"QGroupBox#{name} {{ border: 2px solid #2cae2c; margin-top: 8px; }}"
                if on else "")
            state["n"] += 1
            if state["n"] >= 6:                  # 3 полных мигания
                timer.stop()
                gb.setStyleSheet("")
                self._blink_timers.pop(kind, None)

        timer.timeout.connect(tick)
        self._blink_timers[kind] = timer
        timer.start(300)
        tick()

    def apply_theme(self, pal: dict):
        pass  # таблицы красятся общей темой окна (QSS в main.py)
