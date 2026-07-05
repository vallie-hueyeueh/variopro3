# -*- coding: utf-8 -*-
"""
calib_app.py
============
Окно «КАЛИБРОВКА» (Фаза 2, ПК-пульт).

Что показывает (тёмная «диспетчерская»):
  • два интерактивных 3D-вида рядом — отдельно МАГНИТОМЕТР, отдельно АКСЕЛЕРОМЕТР;
  • на каждом: КРАСНЫЕ точки = сырые (не калиброванные), ЗЕЛЁНЫЕ = калиброванные,
    полупрозрачная сфера-эталон (со слайдером прозрачности);
  • мышь: вращение — зажать и тянуть, зум — колесо;
  • рядом цифры: смещение V (hard-iron), масштабы/матрица (soft-iron),
    остаточная ошибка, число точек N.

Источник точек — session CSV (как у телефона): колонки mx,my,mz (магнитометр)
и ax,ay,az (акселерометр). По кнопке можно загрузить свой файл; «Демо» берёт
синтетику data/sample_calib.csv.

Математика калибровки — в модуле calibration.py (подгонка эллипсоида).
Запуск:
    python pc/calib_app.py                 # откроется окно
    python pc/calib_app.py --selftest      # загрузит демо, снимет скриншоты, выйдет
"""

from __future__ import annotations

import os
import sys
import html
import json
import math
import datetime
import argparse
import webbrowser

import numpy as np

from PySide6 import QtCore, QtGui, QtNetwork, QtWidgets
import pyqtgraph as pg
import pyqtgraph.opengl as gl

from calibration import calibrate, calibrate_diagonal, calibrate_robust, CalibResult
import mag_ekf            # EKF-калибровка магнитометра (рекурсивный + живой LiveMagEKF)
import references
import device_calibration as devcal       # формат калибровки прибора v2 (пакет 14)
from sensor_source import StreamSource   # живой сбор (Фаза 5, шаг 3)
from widgets import StepSpinBox           # единый спинбокс (пакет 15, Г)


class _LiveReader(QtCore.QThread):
    """Мини-читатель живого потока ДЛЯ КАЛИБРОВКИ (когда вариометр поток не держит):
    крутит StreamSource.read_sample в своём потоке и шлёт сэмплы сигналом.
    Если поток уже открыт вариометром — этот класс НЕ используется (подписка на
    его воркер: телефон принимает только ОДНО Bluetooth-подключение)."""

    sampleReady = QtCore.Signal(object)

    def __init__(self, url: str):
        super().__init__()
        self._src = StreamSource(url)
        self._running = True

    def run(self):
        try:
            self._src.open()
            while self._running:
                s = self._src.read_sample()
                if s is not None:
                    self.sampleReady.emit(s)
        except Exception:
            pass
        finally:
            try:
                self._src.close()
            except Exception:
                pass

    def stop(self):
        self._running = False


# «семена» телесных секторов (покрытие сферы) — общий генератор с LiveMagEKF
_fibonacci_sphere = mag_ekf.fibonacci_directions

NOAA_URL = "https://www.ngdc.noaa.gov/geomag/calculators/magcalc.shtml"

# пути проекта
PC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(PC_DIR)
CONFIG_PATH = os.path.join(ROOT, "config.json")
DATA_DIR = os.path.join(ROOT, "data")
SAMPLES_DIR = os.path.join(DATA_DIR, "samples")              # демо-файлы sample_*
DEVCAL_DIR = os.path.join(DATA_DIR, "device_calibrations")   # архив применённых калибровок
APP_ICON = os.path.join(ROOT, "assets", "logo.png")
DOCS_DIR = os.path.join(ROOT, "docs")

# цвета точек (R,G,B,A)
COL_RAW = (1.0, 0.30, 0.30, 1.0)   # красный  — сырые
COL_CAL = (0.30, 1.0, 0.45, 1.0)   # зелёный  — калиброванные
COL_SPHERE = (0.42, 0.68, 1.0)     # голубая сфера-эталон


def calib_summary_lines(r: "CalibResult", unit: str) -> list[str]:
    """
    ЕДИНЫЙ набор чисел про калибровку — ОДИНАКОВЫЙ и под сферой (подпись), и в
    панели «Качество калибровки». Названия и значения совпадают, чтобы не путать.
    """
    v = float(np.linalg.norm(r.V))
    lines = [
        f"Эталон (целевой радиус): {r.radius:.2f} {unit}",
        f"Восстановленный радиус:  {r.mean_raw_radius:.2f} {unit}  (должен ≈ эталону)",
    ]
    if getattr(r, "robust", False) and r.n_total:
        lines.append(f"Остаток ДО отбраковки:   {r.residual_before_rel * 100:.2f}%")
        lines.append(f"Остаток ПОСЛЕ:           {r.residual_rel * 100:.2f}%")
        lines.append(f"Отброшено выбросов:      {r.n_dropped} из {r.n_total}")
    else:
        lines.append(f"Остаток (разброс длины): {r.residual_rel * 100:.2f}%")
    lines.append(f"Смещение нуля |V| (железо телефона): {v:.2f} {unit}")
    lines.append("Полуось 1/2/3 (масштаб по осям): "
                 f"{r.axes[0]:.2f}  {r.axes[1]:.2f}  {r.axes[2]:.2f} {unit}")
    return lines


# ----------------------------------------------------------------------
# Загрузка точек из session CSV
# ----------------------------------------------------------------------
def load_session_csv(path: str):
    """Прочитать CSV с заголовком, вернуть структурированный массив numpy."""
    return np.genfromtxt(path, delimiter=",", names=True)


def session_stream(arr):
    """
    Поток для EKF из session CSV: (t, mag Nx3, gyro Nx3), строки ВЫРОВНЕНЫ.
    Нужны колонки t, mx,my,mz, gx,gy,gz. Возвращает None, если их нет/мало.
    """
    names = arr.dtype.names
    need = ("t", "mx", "my", "mz", "gx", "gy", "gz")
    if names is None or not all(c in names for c in need):
        return None
    M = np.column_stack([np.asarray(arr[c], dtype=float) for c in need])
    M = M[np.isfinite(M).all(axis=1)]
    if M.shape[0] < 50:
        return None
    return M[:, 0], M[:, 1:4], M[:, 4:7]


def extract_xyz(arr, cols):
    """
    Достать три колонки (например mx,my,mz) как массив Nx3.
    Пропускаем строки с пропусками и нулевые (датчик ещё не дал данных).
    Возвращает None, если таких колонок в файле нет.
    """
    names = arr.dtype.names
    if names is None or not all(c in names for c in cols):
        return None
    pts = np.column_stack([np.asarray(arr[c], dtype=float) for c in cols])
    pts = pts[np.isfinite(pts).all(axis=1)]
    nz = np.linalg.norm(pts, axis=1) > 1e-9
    return pts[nz]


def load_calib_json(path):
    """
    Прочитать файл калибровки телефона calib_*.json (см. docs/calib_format.md).
    Возвращает (gyro_bias|None, accel_pts Nx3|None, mag_pts Nx3|None):
      • accel_points → точки акселерометра;
      • mag_stream  → берём колонки mx,my,mz (индексы 1,2,3);
      • gyro_bias   → как есть.
    """
    with open(path, "r", encoding="utf-8") as fh:
        obj = json.load(fh)

    gb = obj.get("gyro_bias")
    gyro_bias = (np.asarray(gb, dtype=float)
                 if isinstance(gb, (list, tuple)) and len(gb) >= 3 else None)

    ap = obj.get("accel_points")
    accel = None
    if ap:
        a = np.asarray(ap, dtype=float)
        if a.ndim == 2 and a.shape[1] >= 3:
            accel = a[:, :3]

    ms = obj.get("mag_stream")
    mag = None
    if ms:
        m = np.asarray(ms, dtype=float)
        if m.ndim == 2 and m.shape[1] >= 4:
            mag = m[:, 1:4]   # t,mx,my,mz,... → mx,my,mz

    # GPS-координаты с телефона (для варианта «из GPS»)
    gps = obj.get("gps")
    gps_d = None
    if isinstance(gps, dict) and gps.get("lat") is not None and gps.get("lon") is not None:
        gps_d = {"lat": float(gps["lat"]), "lon": float(gps["lon"]),
                 "alt": (float(gps["alt"]) if gps.get("alt") is not None else None)}

    return gyro_bias, accel, mag, gps_d


# ----------------------------------------------------------------------
# Одна панель: 3D-вид одного датчика + цифры
# ----------------------------------------------------------------------
class CalibPanel(QtWidgets.QWidget):

    def __init__(self, title: str, unit: str):
        super().__init__()
        self.title = title
        self.unit = unit
        self.result: CalibResult | None = None

        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(6, 6, 6, 6)

        # цвета темы (по умолчанию тёмная; меняются через apply_theme)
        self._info_fg = "#d6dbe1"
        self._accent = "#cfe3ff"
        self._leg_raw = "#ff5555"
        self._leg_cal = "#55ff77"

        # заголовок + кнопки зума (пакет 13, блок В.1): «−» / «100%» / «+» и
        # индикатор процента; колесо мыши по 3D-виду тоже меняет индикатор
        hrow = QtWidgets.QHBoxLayout()
        self.header = QtWidgets.QLabel(title)
        self.header.setStyleSheet(f"font-weight:bold; font-size:15px; color:{self._accent};")
        hrow.addWidget(self.header)
        hrow.addStretch(1)
        self._zoom = 1.0            # 1.0 = 100% (данные вписаны в кадр)
        self._cam_base = 144.0      # дистанция камеры при 100% (ставит set_cam_auto)
        for txt, tip, cb in (("−", "Отдалить", lambda: self.zoom_by(1 / 1.25)),
                             ("100%", "Вписать данные в кадр", self.zoom_reset),
                             ("+", "Приблизить", lambda: self.zoom_by(1.25))):
            b = QtWidgets.QPushButton(txt)
            # Б.4 (пакет 14): «100%» была обрезана фикс-шириной 48 — даём запас
            b.setFixedWidth(30 if txt != "100%" else 64)
            b.setToolTip(tip)
            b.clicked.connect(cb)
            hrow.addWidget(b)
        self.lbl_zoom = QtWidgets.QLabel("зум 100%")
        self.lbl_zoom.setStyleSheet("color:#8a93a0;")
        hrow.addWidget(self.lbl_zoom)
        v.addLayout(hrow)

        # 3D-вид. Б.4 (пакет 14): у ОБЕИХ панелей ГОЛУБОЕ 3D-поле строго
        # одинаковой высоты при любом окне — вся «переменная» часть панели
        # (текст цифр) ниже зажата фикс-высотой, поэтому stretch=1 достаётся
        # только 3D-виду, одинаково в обеих колонках грида.
        self.glview = gl.GLViewWidget()
        self.glview.setBackgroundColor("#0b0d10")
        self.glview.setMinimumHeight(300)
        self.glview.installEventFilter(self)      # колесо мыши → обновить «зум %»
        v.addWidget(self.glview, stretch=1)

        # вспомогательные элементы сцены
        self.grid = gl.GLGridItem()
        self.grid.setColor((80, 90, 105, 120))
        self.glview.addItem(self.grid)
        self.axis = gl.GLAxisItem()
        self.glview.addItem(self.axis)

        # точки (создаём пустыми, наполним в set_data)
        self.scatter_raw = gl.GLScatterPlotItem(pos=np.zeros((0, 3)), color=COL_RAW,
                                                 size=6.0, pxMode=True)
        self.scatter_cal = gl.GLScatterPlotItem(pos=np.zeros((0, 3)), color=COL_CAL,
                                                 size=6.0, pxMode=True)
        # БАГ светлой темы: по умолчанию точки рисуются блендом 'additive' (цвет
        # СКЛАДЫВАЕТСЯ с фоном) — на светлом фоне сложение уводит их в белый.
        # Переключаем на ОБЫЧНОЕ смешивание (translucent): цвет виден на любом фоне,
        # красные/зелёные различимы и в тёмной, и в светлой теме.
        self.scatter_raw.setGLOptions("translucent")
        self.scatter_cal.setGLOptions("translucent")
        self.glview.addItem(self.scatter_raw)
        self.glview.addItem(self.scatter_cal)
        self.sphere = None  # появится в set_data (нужен радиус)

        # ползунки: прозрачность сферы + яркость/насыщенность/размер точек
        ctrl = QtWidgets.QGridLayout()
        ctrl.addWidget(QtWidgets.QLabel("Прозрачность сферы:"), 0, 0)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(22)
        self.slider.valueChanged.connect(self._update_sphere_alpha)
        ctrl.addWidget(self.slider, 0, 1)

        ctrl.addWidget(QtWidgets.QLabel("Яркость точек:"), 1, 0)
        self.sl_bright = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sl_bright.setRange(20, 100)
        self.sl_bright.setValue(100)            # по умолчанию ярко
        self.sl_bright.valueChanged.connect(self._restyle_points)
        ctrl.addWidget(self.sl_bright, 1, 1)

        ctrl.addWidget(QtWidgets.QLabel("Насыщенность:"), 2, 0)
        self.sl_sat = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sl_sat.setRange(0, 100)
        self.sl_sat.setValue(100)               # по умолчанию насыщенно
        self.sl_sat.valueChanged.connect(self._restyle_points)
        ctrl.addWidget(self.sl_sat, 2, 1)

        ctrl.addWidget(QtWidgets.QLabel("Размер точки:"), 3, 0)
        self.sl_size = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sl_size.setRange(3, 18)
        self.sl_size.setValue(10)               # по умолчанию крупнее
        self.sl_size.valueChanged.connect(self._restyle_points)
        ctrl.addWidget(self.sl_size, 3, 1)

        # совмещение центров: у реального телефона красное (сырое) облако смещено на
        # ~1230 мкТл (внутреннее железо) и улетает в сторону от зелёного. Галочка сдвигает
        # красное к центру (вычитает среднее) — ТОЛЬКО ОТОБРАЖЕНИЕ, на расчёт не влияет.
        self.chk_center = QtWidgets.QCheckBox("Совместить центры (показ)")
        self.chk_center.setChecked(True)
        self.chk_center.setToolTip("Сдвинуть красное (сырое) облако к центру, чтобы оно было видно "
                                   "рядом с зелёным. Только отображение — на калибровку не влияет.")
        self.chk_center.toggled.connect(self._on_center_toggled)
        ctrl.addWidget(self.chk_center, 4, 0, 1, 2)
        self.lbl_center_note = QtWidgets.QLabel("")
        self.lbl_center_note.setWordWrap(True)
        self.lbl_center_note.setStyleSheet("color:#c0a060; font-size:11px;")
        self.lbl_center_note.setFixedHeight(30)    # Б.4: высота не «плавает»
        ctrl.addWidget(self.lbl_center_note, 5, 0, 1, 2)
        v.addLayout(ctrl)

        # цифры (легенда + значения) — в скролле ФИКСИРОВАННОЙ высоты (Б.4):
        # разное число строк текста больше не отъедает высоту у 3D-поля
        self.info = QtWidgets.QLabel()
        self.info.setTextFormat(QtCore.Qt.RichText)
        self.info.setWordWrap(True)
        self.info.setAlignment(QtCore.Qt.AlignTop)
        info_scroll = QtWidgets.QScrollArea()
        info_scroll.setWidget(self.info)
        info_scroll.setWidgetResizable(True)
        info_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        info_scroll.setFixedHeight(190)
        v.addWidget(info_scroll)

        self._set_info_html(
            '<span style="color:#888">Нет данных. Загрузите CSV или нажмите «Демо».</span>')

    # ---- зум 3D-сцены (блок В.1) ----
    def set_cam_auto(self, dist: float, **kw):
        """Автоматическая постановка камеры под данные: dist = дистанция «100%».
        Пользовательский зум СОХРАНЯЕТСЯ (живое облако обновляет камеру 2.5 раз/с —
        без этого зум сбрасывался бы каждый тик)."""
        self._cam_base = float(dist)
        self.glview.setCameraPosition(distance=self._cam_base / self._zoom, **kw)
        self._update_zoom_label()

    def zoom_by(self, factor: float):
        self._zoom = min(16.0, max(0.1, self._zoom * float(factor)))
        self.glview.setCameraPosition(distance=self._cam_base / self._zoom)
        self._update_zoom_label()

    def zoom_reset(self):
        self._zoom = 1.0
        self.glview.setCameraPosition(distance=self._cam_base)
        self._update_zoom_label()

    def _update_zoom_label(self):
        try:
            d = float(self.glview.opts["distance"])
            self._zoom = self._cam_base / max(d, 1e-6)
        except (KeyError, TypeError):
            pass
        self.lbl_zoom.setText(f"зум {self._zoom * 100:.0f}%")

    def eventFilter(self, obj, ev):
        if obj is self.glview and ev.type() == QtCore.QEvent.Wheel:
            QtCore.QTimer.singleShot(0, self._update_zoom_label)
        return False

    def _alpha(self) -> float:
        return self.slider.value() / 100.0

    def _update_sphere_alpha(self):
        if self.sphere is not None:
            self.sphere.setColor((*COL_SPHERE, self._alpha()))

    def _point_color(self, hue_deg):
        """Цвет точки (RGBA 0..1) по ползункам: яркость и насыщенность; hue 0=красный, 120=зелёный."""
        s = self.sl_sat.value() / 100.0
        val = self.sl_bright.value() / 100.0
        c = QtGui.QColor.fromHsvF((hue_deg % 360) / 360.0, s, val)
        return (c.redF(), c.greenF(), c.blueF(), 1.0)

    def _point_size(self):
        return float(self.sl_size.value())

    def _raw_display_offset(self):
        """Сдвиг для ОТОБРАЖЕНИЯ сырых точек: если «Совместить центры» включено — вычесть
        среднее (красное к центру). На расчёт калибровки НЕ влияет."""
        r = self.result
        if r is None or not self.chk_center.isChecked():
            return np.zeros(3)
        return r.raw.mean(axis=0)

    def _on_center_toggled(self):
        """Галочка «Совместить центры»: перерисовать точки/камеру (расчёт не трогаем)."""
        if self.result is not None:
            self.set_data(self.result)

    def _restyle_points(self):
        """Применить цвет/размер к уже показанным точкам (без пересчёта калибровки)."""
        r = self.result
        if r is None:
            return
        off = self._raw_display_offset()
        col_raw = np.tile(np.array(self._point_color(0), dtype=np.float32), (r.raw.shape[0], 1))
        col_cal = np.tile(np.array(self._point_color(120), dtype=np.float32), (r.calibrated.shape[0], 1))
        sz = self._point_size()
        self.scatter_raw.setData(pos=r.raw - off, color=col_raw, size=sz, pxMode=True)
        self.scatter_cal.setData(pos=r.calibrated, color=col_cal, size=sz, pxMode=True)

    def _set_info_html(self, body: str):
        self.info.setText(body)

    def set_message(self, msg: str):
        """Показать сообщение (например, что датчика нет в файле) и очистить точки."""
        self.result = None
        self.scatter_raw.setData(pos=np.zeros((0, 3)))
        self.scatter_cal.setData(pos=np.zeros((0, 3)))
        if self.sphere is not None:
            self.glview.removeItem(self.sphere)
            self.sphere = None
        self._set_info_html(f'<span style="color:#e0a060">{html.escape(msg)}</span>')

    def set_data(self, result: CalibResult):
        """Показать сырые/калиброванные точки, сферу-эталон и цифры."""
        self.result = result
        # цвет/размер точек берём с ползунков (красные=сырые, зелёные=калиброванные);
        # цвет ВСЕГДА массивом (N,4), иначе pyqtgraph рисует точки белыми
        self._restyle_points()

        # сфера-эталон радиуса result.radius (пересоздаём — радиус мог измениться)
        if self.sphere is not None:
            self.glview.removeItem(self.sphere)
        md = gl.MeshData.sphere(rows=24, cols=48, radius=result.radius)
        self.sphere = gl.GLMeshItem(meshdata=md, smooth=True, shader="balloon",
                                    glOptions="translucent",
                                    color=(*COL_SPHERE, self._alpha()))
        self.glview.addItem(self.sphere)  # добавляем последней — для корректной прозрачности

        # сетка/оси под масштаб
        r = float(result.radius)
        self.grid.setSize(x=2 * r, y=2 * r, z=1)
        self.grid.setSpacing(x=r / 2, y=r / 2, z=1)
        self.axis.setSize(x=1.3 * r, y=1.3 * r, z=1.3 * r)

        # камера: под ОТОБРАЖАЕМЫЕ сырые точки (с учётом сдвига «Совместить центры»)
        off = self._raw_display_offset()
        ext = max(float(np.abs(result.raw - off).max()), r) * 1.2
        self.set_cam_auto(ext * 2.4, elevation=22, azimuth=40)

        # подпись про смещение красного облака (внутреннее железо телефона)
        off_mag = float(np.linalg.norm(result.raw.mean(axis=0)))
        if off_mag > 2.0 * r:
            mode = "совмещено для показа" if self.chk_center.isChecked() else "показано как есть"
            self.lbl_center_note.setText(
                f"красное смещено на ~{off_mag:.0f} {self.unit} — "
                f"внутреннее железо телефона ({mode})")
        else:
            self.lbl_center_note.setText("")

        self._set_info_html(self._info_html(result))

    def _info_html(self, r: CalibResult) -> str:
        M = r.M_corr
        mat = "\n".join(
            "  [" + "  ".join(f"{M[i, j]:+.4f}" for j in range(3)) + "]"
            for i in range(3)
        )
        u = self.unit
        # ЕДИНЫЙ набор чисел — те же названия и значения, что в панели «Качество»
        core = "\n".join(calib_summary_lines(r, u))
        txt = (
            f"N = {r.n_points} точек\n"
            + core + "\n"
            f"  (компоненты смещения V: {r.V[0]:+.2f}  {r.V[1]:+.2f}  {r.V[2]:+.2f} {u})\n"
            f"Матрица soft-iron M_corr:\n{mat}"
        )
        legend = (f'<span style="color:{self._leg_raw}">■ сырые</span> &nbsp;&nbsp;'
                  f'<span style="color:{self._leg_cal}">■ калиброванные</span>')
        return (f'{legend}<pre style="font-family:Consolas,monospace;'
                f'font-size:11pt;margin:4px 0;color:{self._info_fg}">{html.escape(txt)}</pre>')

    def apply_theme(self, pal: dict):
        """Перекрасить 3D-вид и панель цифр под тему (фон сцены, цвет текста/легенды/заголовка)."""
        self.glview.setBackgroundColor(pal["gl_bg"])
        self._info_fg = pal["info_fg"]
        self._accent = pal["accent"]
        self._leg_raw = pal["leg_raw"]
        self._leg_cal = pal["leg_cal"]
        self.header.setStyleSheet(f"font-weight:bold; font-size:15px; color:{self._accent};")
        if self.result is not None:
            self._set_info_html(self._info_html(self.result))
        else:
            self._set_info_html(
                f'<span style="color:{self._info_fg}">Нет данных. '
                f'Загрузите CSV или нажмите «Демо».</span>')


# ----------------------------------------------------------------------
# Карта на тайлах OpenStreetMap (QGraphicsView, БЕЗ QtWebEngine).
# Перетаскивание — двигать, колесо — зум, клик — выбрать точку.
# ----------------------------------------------------------------------
def _lonlat_to_px(lon, lat, z):
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n * 256.0
    lat = max(min(lat, 85.05112878), -85.05112878)
    s = math.sin(math.radians(lat))
    y = (0.5 - math.log((1 + s) / (1 - s)) / (4 * math.pi)) * n * 256.0
    return x, y


def _px_to_lonlat(x, y, z):
    n = 2 ** z
    lon = x / (n * 256.0) * 360.0 - 180.0
    yy = 0.5 - y / (n * 256.0)
    lat = math.degrees(math.atan(math.sinh(2 * math.pi * yy)))
    return lon, lat


class _MapView(QtWidgets.QGraphicsView):
    clicked = QtCore.Signal(float, float)   # координаты сцены x,y
    zoomed = QtCore.Signal(int)

    def __init__(self, scene):
        super().__init__(scene)
        self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
        self.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
        self._press = None

    def mousePressEvent(self, e):
        self._press = e.position().toPoint()
        super().mousePressEvent(e)

    def mouseReleaseEvent(self, e):
        super().mouseReleaseEvent(e)
        p = e.position().toPoint()
        if self._press is not None and (p - self._press).manhattanLength() < 5:
            sp = self.mapToScene(p)
            self.clicked.emit(sp.x(), sp.y())
        self._press = None

    def wheelEvent(self, e):
        self.zoomed.emit(1 if e.angleDelta().y() > 0 else -1)


class MapDialog(QtWidgets.QDialog):

    TILE = 256
    RADIUS = 4   # тайлов в каждую сторону от центра

    def __init__(self, lat, lon, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Карта — тяните мышью, колесо — зум, клик — точка")
        self.resize(820, 640)
        self._z = 11
        self._lat = float(lat)
        self._lon = float(lon)
        self.picked = (self._lat, self._lon)
        self._items = {}
        self._marker = None

        self._net = QtNetwork.QNetworkAccessManager(self)
        self._net.finished.connect(self._on_tile)

        v = QtWidgets.QVBoxLayout(self)
        self.scene = QtWidgets.QGraphicsScene(self)
        self.view = _MapView(self.scene)
        self.view.clicked.connect(self._on_click)
        self.view.zoomed.connect(self._on_zoom)
        v.addWidget(self.view, 1)

        row = QtWidgets.QHBoxLayout()
        self.lbl = QtWidgets.QLabel(f"φ={self._lat:.5f}  λ={self._lon:.5f}")
        row.addWidget(self.lbl, 1)
        btn_ok = QtWidgets.QPushButton("Использовать точку")
        btn_ok.clicked.connect(self.accept)
        btn_cancel = QtWidgets.QPushButton("Отмена")
        btn_cancel.clicked.connect(self.reject)
        row.addWidget(btn_ok)
        row.addWidget(btn_cancel)
        v.addLayout(row)

        QtCore.QTimer.singleShot(0, self._rebuild)  # построить после показа

    def _rebuild(self):
        self.scene.clear()
        self._items.clear()
        self._marker = None
        z = self._z
        cx, cy = _lonlat_to_px(self._lon, self._lat, z)
        ctx, cty = int(cx // self.TILE), int(cy // self.TILE)
        nmax = 2 ** z
        for dx in range(-self.RADIUS, self.RADIUS + 1):
            for dy in range(-self.RADIUS, self.RADIUS + 1):
                tx, ty = ctx + dx, cty + dy
                if tx < 0 or ty < 0 or tx >= nmax or ty >= nmax:
                    continue
                item = QtWidgets.QGraphicsPixmapItem()
                item.setOffset(tx * self.TILE, ty * self.TILE)
                pm = QtGui.QPixmap(self.TILE, self.TILE)
                pm.fill(QtGui.QColor("#dfe6ee"))
                item.setPixmap(pm)
                self.scene.addItem(item)
                self._items[(tx, ty)] = item
                self._request(tx, ty, z)
        x0 = (ctx - self.RADIUS) * self.TILE
        y0 = (cty - self.RADIUS) * self.TILE
        side = (2 * self.RADIUS + 1) * self.TILE
        self.scene.setSceneRect(x0, y0, side, side)
        self.view.centerOn(cx, cy)
        self._place_marker(cx, cy)

    def _request(self, tx, ty, z):
        url = f"https://tile.openstreetmap.org/{z}/{tx}/{ty}.png"
        req = QtNetwork.QNetworkRequest(QtCore.QUrl(url))
        req.setHeader(QtNetwork.QNetworkRequest.UserAgentHeader, "VarioPro/1.0 (calibration map)")
        req.setAttribute(QtNetwork.QNetworkRequest.User, f"{tx},{ty},{z}")
        self._net.get(req)

    def _on_tile(self, reply):
        try:
            tag = reply.request().attribute(QtNetwork.QNetworkRequest.User)
            data = reply.readAll()
            reply.deleteLater()
            if not tag:
                return
            tx, ty, z = (int(s) for s in str(tag).split(","))
            if z != self._z:
                return
            item = self._items.get((tx, ty))
            if item is None:
                return
            pm = QtGui.QPixmap()
            if pm.loadFromData(data):
                item.setPixmap(pm)
        except Exception:
            pass  # битый тайл — просто пропускаем, не роняем приложение

    def _on_zoom(self, d):
        try:
            c = self.view.mapToScene(self.view.viewport().rect().center())
            self._lon, self._lat = _px_to_lonlat(c.x(), c.y(), self._z)
            self._z = max(3, min(18, self._z + d))
            self._rebuild()
        except Exception:
            pass

    def _on_click(self, sx, sy):
        try:
            lon, lat = _px_to_lonlat(sx, sy, self._z)
            self.picked = (lat, lon)
            self.lbl.setText(f"φ={lat:.5f}  λ={lon:.5f}")
            self._place_marker(sx, sy)
        except Exception:
            pass

    def _place_marker(self, sx, sy):
        if self._marker is not None:
            self.scene.removeItem(self._marker)
        self._marker = self.scene.addEllipse(
            sx - 6, sy - 6, 12, 12, QtGui.QPen(QtGui.QColor("#d00000"), 3))


# ----------------------------------------------------------------------
# Окно «Инструкция» — берём текст из docs/manual.md (единый источник правды).
# ----------------------------------------------------------------------
MANUAL_PATH = os.path.join(DOCS_DIR, "manual.md")


class HelpDialog(QtWidgets.QDialog):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Инструкция — VarioPro3")
        self.resize(720, 640)
        v = QtWidgets.QVBoxLayout(self)
        browser = QtWidgets.QTextBrowser()
        browser.setOpenExternalLinks(True)
        # текст инструкции — из docs/manual.md (markdown)
        try:
            with open(MANUAL_PATH, "r", encoding="utf-8") as fh:
                browser.setMarkdown(fh.read())
        except OSError as e:
            browser.setPlainText(f"Не удалось открыть docs/manual.md:\n{e}")
        v.addWidget(browser, 1)
        btn = QtWidgets.QPushButton("Закрыть")
        btn.clicked.connect(self.accept)
        v.addWidget(btn)


# ----------------------------------------------------------------------
# Панель «Эталоны»: местоположение, гравитация g, магнитное поле F/D/I,
# качество калибровки и сохранение калибровки прибора.
# ----------------------------------------------------------------------
class ReferencePanel(QtWidgets.QWidget):

    # барометр даёт ТОЛЬКО высоту, поэтому у широты/долготы его нет
    LATLON_SOURCES = ["вручную", "из GPS", "по карте"]
    ALT_SOURCES = ["вручную", "из барометра", "из GPS"]

    def __init__(self, window):
        super().__init__()
        self.win = window
        self.baro_provider = None     # функция, отдающая высоту из барометра (если есть)
        self.declination = None       # склонение D (после «Получить поле»)
        self._gps = None              # GPS-координаты из загруженного файла телефона

        v = QtWidgets.QVBoxLayout(self)

        # ПАКЕТ 15 (Д.4): панель выстроена МАСТЕРОМ — нумерованные шаги
        # 1. Данные → 2. Эталоны места → 3. Результаты → 4. Выбор → 5. Сохранить.
        # Существующие блоки перетитулованы и переставлены; логика не менялась.

        # === ШАГ 1 — ДАННЫЕ (файл или живой сбор) ===
        gb_data = QtWidgets.QGroupBox("Шаг 1 — Данные: файл или живой сбор")
        ld = QtWidgets.QVBoxLayout(gb_data)
        lbl_d = QtWidgets.QLabel(
            "Файл: кнопка «Загрузить файл…» сверху (запись calib_*.json или "
            "session CSV). Живой сбор — прямо отсюда:")
        lbl_d.setWordWrap(True)
        lbl_d.setStyleSheet("color:#8a93a0; font-size:11px;")
        ld.addWidget(lbl_d)
        self.btn_live = QtWidgets.QPushButton("▶ Слушать поток")
        self.btn_live.setToolTip(
            "Собирать точки магнитометра прямо из живого потока (телефон/симулятор).\n"
            "Если поток уже открыт вариометром — подпишемся на него (второе\n"
            "подключение НЕ открывается: телефон принимает только одно). Иначе —\n"
            "подключимся сами по адресу из поля «Поток» вариометра.")
        self.btn_live.clicked.connect(self.win.toggle_live_capture)
        ld.addWidget(self.btn_live)
        self.lbl_live_status = QtWidgets.QLabel("поток не слушается")
        self.lbl_live_status.setWordWrap(True)
        self.lbl_live_status.setStyleSheet("color:#8a93a0;")
        ld.addWidget(self.lbl_live_status)
        self.lbl_live_stats = QtWidgets.QLabel(
            "Точек: raw 0 · android 0 · Покрытие сферы: 0% (цель ≥80%)")
        self.lbl_live_stats.setStyleSheet("font-weight:bold;")
        ld.addWidget(self.lbl_live_stats)
        hint = QtWidgets.QLabel(
            "Вращайте телефон во ВСЕХ плоскостях («восьмёрки») до покрытия ≥80%. "
            "Для остатка <3% собирайте ПОД ОТКРЫТЫМ НЕБОМ, вдали от стали "
            "(в помещении остаток будет 10%+ — это свойство места, не метода).")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#8a93a0; font-size:11px;")
        ld.addWidget(hint)
        lrow = QtWidgets.QHBoxLayout()
        self.btn_live_ransac = QtWidgets.QPushButton(
            "Посчитать RANSAC по собранному (только магнитометр)")
        self.btn_live_ransac.setToolTip(
            "Пакетный RANSAC-эллипсоид по СОБРАННЫМ маг-буферам — отдельно для\n"
            "сырого и Android-поля. Кандидаты попадают в таблицу шага 3\n"
            "и уходят в файл кнопкой «Сохранить калибровку прибора» (шаг 5).\n"
            "Аксель/гироскоп/загруженный файл НЕ трогаются.")
        self.btn_live_ransac.clicked.connect(self.win.live_compute_ransac)
        self.btn_live_ransac.setEnabled(False)
        lrow.addWidget(self.btn_live_ransac)
        self.btn_live_save = QtWidgets.QPushButton("Сохранить сырьё…")
        self.btn_live_save.setFlat(True)
        self.btn_live_save.setStyleSheet("font-size:11px; color:#8a93a0;")
        self.btn_live_save.setToolTip(
            "Вторичная кнопка: собранный маг-буфер → data\\calib_*.json («сырьё для\n"
            "истории», только магнитометр): открывается штатной загрузкой, виден в\n"
            "«Записях», можно пересчитать позже. Оба поля сохраняются (raw — основным\n"
            "потоком mag_stream, android — рядом отдельной секцией).")
        self.btn_live_save.clicked.connect(self.win.live_save_json)
        self.btn_live_save.setEnabled(False)
        lrow.addWidget(self.btn_live_save)
        lrow.addStretch(1)
        ld.addLayout(lrow)
        v.addWidget(gb_data)

        # === ШАГ 2 — ЭТАЛОНЫ МЕСТА ===
        gb_loc = QtWidgets.QGroupBox("Шаг 2 — Эталоны места: местоположение")
        grid = QtWidgets.QGridLayout(gb_loc)
        # координаты по умолчанию — ПРИМЕР (центр Санкт-Петербурга);
        # пользователь задаёт своё место полями/картой/из GPS
        self.spin_lat = self._spin(-90, 90, 4, 59.9386)
        self.spin_lon = self._spin(-180, 180, 4, 30.3141)
        self.spin_alt = self._spin(-500, 9000, 1, 10.0)
        self.src_lat = self._mk_combo(self.LATLON_SOURCES)
        self.src_lon = self._mk_combo(self.LATLON_SOURCES)
        self.src_alt = self._mk_combo(self.ALT_SOURCES)
        rows = [("Широта φ, °:", self.spin_lat, self.src_lat),
                ("Долгота λ, °:", self.spin_lon, self.src_lon),
                ("Высота h, м:", self.spin_alt, self.src_alt)]
        for i, (lbl, sp, sc) in enumerate(rows):
            grid.addWidget(QtWidgets.QLabel(lbl), i, 0)
            grid.addWidget(sp, i, 1)
            grid.addWidget(sc, i, 2)
        # ТОЛЬКО ДЛЯ ЧТЕНИЯ: исходная высота, пришедшая с барометра/GPS — видна,
        # даже если пользователь потом правит редактируемое поле «Высота h» слева.
        self.alt_src_value = QtWidgets.QLineEdit()
        self.alt_src_value.setReadOnly(True)
        self.alt_src_value.setPlaceholderText("пришло: —")
        self.alt_src_value.setMaximumWidth(150)
        self.alt_src_value.setToolTip(
            "Исходная высота, полученная с барометра или GPS (только чтение).\n"
            "Поле «Высота h» слева можно править вручную — это значение остаётся для справки.")
        grid.addWidget(self.alt_src_value, 2, 3)
        self.btn_map = QtWidgets.QPushButton("Открыть карту…")
        self.btn_map.setToolTip("Выбрать широту/долготу кликом по карте (OpenStreetMap, нужен интернет)")
        self.btn_map.clicked.connect(self._open_map)
        grid.addWidget(self.btn_map, 3, 0, 1, 3)
        self.lbl_loc_msg = QtWidgets.QLabel("")
        self.lbl_loc_msg.setWordWrap(True)
        grid.addWidget(self.lbl_loc_msg, 4, 0, 1, 3)
        v.addWidget(gb_loc)

        # --- Гравитация ---
        gb_g = QtWidgets.QGroupBox("Шаг 2 — Гравитация (эталон для акселерометра)")
        lg = QtWidgets.QVBoxLayout(gb_g)
        btn_g = QtWidgets.QPushButton("Вычислить g")
        btn_g.clicked.connect(self._compute_g)
        self.lbl_g = QtWidgets.QLabel("g = —  м/с²")
        self.lbl_g.setStyleSheet("font-weight:bold;")
        lg.addWidget(btn_g)
        lg.addWidget(self.lbl_g)
        v.addWidget(gb_g)

        # --- Магнитное поле ---
        gb_m = QtWidgets.QGroupBox("Шаг 2 — Магнитное поле (эталон для магнитометра)")
        lm = QtWidgets.QFormLayout(gb_m)
        self.combo_model = QtWidgets.QComboBox()
        self.combo_model.addItems(["офлайн WMM (pygeomag)", "онлайн NOAA"])
        self.date_edit = QtWidgets.QDateEdit()
        self.date_edit.setCalendarPopup(True)
        self.date_edit.setDisplayFormat("yyyy-MM-dd")
        self.date_edit.setDate(QtCore.QDate.currentDate())
        self.edit_key = QtWidgets.QLineEdit()
        self.edit_key.setPlaceholderText("нужен только для онлайн-NOAA")
        self.edit_key.setToolTip("Бесплатный ключ NOAA: ngdc.noaa.gov/geomag/calculators/magcalc.shtml")
        btn_m = QtWidgets.QPushButton("Получить поле")
        btn_m.clicked.connect(self._compute_field)
        self.lbl_field = QtWidgets.QLabel("F = —   D = —   I = —")
        self.lbl_field.setStyleSheet("font-weight:bold;")
        self.lbl_field.setWordWrap(True)
        # подпись: ДЛЯ КАКОЙ точки/даты посчитан эталон (видно, что берутся
        # координаты с карты/GPS и высота из поля h, а не что-то зашитое)
        self.lbl_field_where = QtWidgets.QLabel("")
        self.lbl_field_where.setWordWrap(True)
        self.lbl_field_where.setStyleSheet("color:#8a93a0; font-size:11px;")
        lm.addRow("Модель:", self.combo_model)
        lm.addRow("Дата:", self.date_edit)
        lm.addRow("Ключ NOAA:", self.edit_key)
        lm.addRow(btn_m)
        lm.addRow(self.lbl_field)
        lm.addRow(self.lbl_field_where)
        # копируемый текст про ключ NOAA + кликабельная ссылка
        self.lbl_noaa = QtWidgets.QLabel(
            'Онлайн-NOAA требует бесплатный ключ. Получить: '
            f'<a href="{NOAA_URL}">ngdc.noaa.gov/geomag</a>. '
            'Офлайн-WMM2025 работает без ключа.')
        self.lbl_noaa.setWordWrap(True)
        self.lbl_noaa.setTextFormat(QtCore.Qt.RichText)
        self.lbl_noaa.setTextInteractionFlags(
            QtCore.Qt.TextSelectableByMouse | QtCore.Qt.LinksAccessibleByMouse)
        self.lbl_noaa.setOpenExternalLinks(True)
        lm.addRow(self.lbl_noaa)
        btn_noaa = QtWidgets.QPushButton("Открыть сайт NOAA")
        btn_noaa.clicked.connect(lambda: webbrowser.open(NOAA_URL))
        lm.addRow(btn_noaa)
        v.addWidget(gb_m)

        # --- Обновить эталоны для этой точки (g + поле + качество одной кнопкой) ---
        btn_recalc = QtWidgets.QPushButton("Обновить эталоны для этой точки")
        btn_recalc.setToolTip(
            "Считает СРАЗУ и тяжесть g, и магнитное поле F (WMM) по текущим\n"
            "координатам/дате, затем обновляет «Качество». Это «Вычислить g» +\n"
            "«Получить поле» одной кнопкой.")
        btn_recalc.clicked.connect(self._recalc)
        v.addWidget(btn_recalc)

        # === ШАГ 3 — РЕЗУЛЬТАТЫ ПО ИСТОЧНИКАМ И МЕТОДАМ (пакет 15, Д.3) ===
        gb_q = QtWidgets.QGroupBox("Шаг 3 — Результаты по источникам и методам")
        lq = QtWidgets.QVBoxLayout(gb_q)
        self.lbl_q_acc = QtWidgets.QLabel("Акселерометр: —")
        self.lbl_q_acc.setWordWrap(True)
        self.lbl_q_acc.setToolTip(
            "Остаток акселерометра = разброс длины вектора |a| после калибровки\n"
            "относительно эталона g, %. Оценки: отлично < 1%, хорошо < 3%,\n"
            "плохо ≥ 3% (переснимите 6+ статичных поз).")
        lq.addWidget(self.lbl_q_acc)
        # таблица «источник × метод» с цветовой оценкой; выбранная комбинация
        # помечена ▶ (обновляется при переключении селектора шага 4)
        self.lbl_matrix = QtWidgets.QLabel("")
        self.lbl_matrix.setTextFormat(QtCore.Qt.RichText)
        self.lbl_matrix.setToolTip(
            "«Остаток» = разброс длины вектора поля ПОСЛЕ калибровки, % — чем\n"
            "меньше, тем ровнее сфера. Оценки: отлично < 3%, хорошо < 10%,\n"
            "плохо ≥ 10% (обычно значит «снято в помещении»).\n"
            "Строка ▶ — комбинация, ВЫБРАННАЯ для компаса (шаг 4).\n"
            "RANSAC: свежий кандидат живого сбора, иначе загруженный файл,\n"
            "иначе сохранённая секция calibration.json. Live-EKF: живые числа\n"
            "при идущем сборе.")
        lq.addWidget(self.lbl_matrix)
        lbl_res_note = QtWidgets.QLabel(
            "«Остаток» = разброс длины откалиброванного вектора поля, % — чем меньше, "
            "тем ровнее сфера. У RANSAC под сферой: «ДО/ПОСЛЕ отбраковки» — по всем "
            "точкам / после удаления выбросов (магнитных помех).")
        lbl_res_note.setWordWrap(True)
        lbl_res_note.setStyleSheet("color:#8a93a0; font-size:11px;")
        lq.addWidget(lbl_res_note)
        # LIVE-EKF (рантайм-подстройка; в файл НЕ сохраняется)
        self.lbl_live_ekf = QtWidgets.QLabel("EKF: —")
        self.lbl_live_ekf.setWordWrap(True)
        lq.addWidget(self.lbl_live_ekf)
        self.live_plot = pg.PlotWidget()
        self.live_plot.setMinimumHeight(110)
        self.live_plot.setMaximumHeight(130)
        self.live_plot.setBackground("#101418")
        self.live_plot.setLabel("left", "остаток, %")
        self.live_plot.setLabel("bottom", "точка")
        self.live_plot.showGrid(x=True, y=True, alpha=0.3)
        self.live_plot.getAxis("left").enableAutoSIPrefix(False)
        self.live_plot.getAxis("bottom").enableAutoSIPrefix(False)
        self.live_curve = self.live_plot.plot(pen=pg.mkPen("#5aa0ff", width=2))
        lq.addWidget(self.live_plot)
        # Сравнение методов по загруженному файлу (анимация сходимости EKF)
        self.btn_run_ekf = QtWidgets.QPushButton("Прогнать EKF по загруженному файлу")
        self.btn_run_ekf.setToolTip(
            "Проверка математики: воспроизводит загруженный файл по точкам и\n"
            "показывает, как EKF сходится к RANSAC (расхождение — в %).")
        self.btn_run_ekf.clicked.connect(self.win.run_ekf_compare)
        lq.addWidget(self.btn_run_ekf)
        self.lbl_cmp_live = QtWidgets.QLabel(
            "Загрузите запись с гироскопом и нажмите «Прогнать EKF по загруженному файлу».")
        self.lbl_cmp_live.setWordWrap(True)
        self.lbl_cmp_live.setTextFormat(QtCore.Qt.RichText)
        lq.addWidget(self.lbl_cmp_live)
        self.cmp_plot = pg.PlotWidget()
        self.cmp_plot.setMinimumHeight(150)
        self.cmp_plot.setMaximumHeight(180)
        self.cmp_plot.setBackground("#101418")
        self.cmp_plot.setLabel("left", "остаток, %")
        self.cmp_plot.setLabel("bottom", "обработано точек")
        self.cmp_plot.showGrid(x=True, y=True, alpha=0.3)
        self.cmp_plot.getAxis("left").enableAutoSIPrefix(False)
        self.cmp_plot.getAxis("bottom").enableAutoSIPrefix(False)
        self.cmp_curve_ekf = self.cmp_plot.plot(pen=pg.mkPen("#5aa0ff", width=2), name="EKF")
        self.cmp_line_ransac = self.cmp_plot.addLine(
            y=0, pen=pg.mkPen("#e0a060", width=2, style=QtCore.Qt.DashLine))
        lq.addWidget(self.cmp_plot)
        self.lbl_cmp_table = QtWidgets.QLabel("")
        self.lbl_cmp_table.setWordWrap(True)
        self.lbl_cmp_table.setTextFormat(QtCore.Qt.RichText)
        lq.addWidget(self.lbl_cmp_table)
        self.lbl_cmp_honest = QtWidgets.QLabel(
            "RANSAC точнее на готовой записи — его результат и сохраняется. Live-EKF — "
            "рантайм-подстройка компаса (когда магнитная обстановка меняется) и "
            "независимая проверка математики.")
        self.lbl_cmp_honest.setWordWrap(True)
        self.lbl_cmp_honest.setStyleSheet("color:#8a93a0; font-size:11px;")
        lq.addWidget(self.lbl_cmp_honest)
        v.addWidget(gb_q)

        # === ШАГ 4 — ВЫБОР ДЛЯ КОМПАСА (Б.2): метод × источник, 4 варианта ===
        gb_sel = QtWidgets.QGroupBox("Шаг 4 — Компас использует")
        ll = QtWidgets.QVBoxLayout(gb_sel)
        comp_grid = QtWidgets.QGridLayout()
        self.radio_comp = {}
        for (r, c), key, title in (
                ((0, 0), "ransac@raw", "RANSAC @ сырое поле"),
                ((0, 1), "ransac@android", "RANSAC @ Android-поле"),
                ((1, 0), "live@raw", "Live-EKF @ сырое поле"),
                ((1, 1), "live@android", "Live-EKF @ Android-поле")):
            rb = QtWidgets.QRadioButton(title)
            self.radio_comp[key] = rb
            comp_grid.addWidget(rb, r, c)
        self.radio_comp["ransac@raw"].setToolTip(
            "V/W секции mag_raw из pc/calibration.json применяются к СЫРОМУ полю.")
        self.radio_comp["ransac@android"].setToolTip(
            "«Тонкая» калибровка mag_android поверх Android-поля (ОС уже сняла железо);\n"
            "секции нет — Android-поле используется как есть.")
        for key in ("live@raw", "live@android"):
            self.radio_comp[key].setToolTip(
                "Живой EKF по потоку — подстраивается, пока идёт сбор.\n"
                "Доступно только при запущенном «Слушать поток» (в файл не пишется;\n"
                "при остановке сбора компас явно переключится на RANSAC).")
            self.radio_comp[key].setEnabled(False)
        self._comp_group = QtWidgets.QButtonGroup(self)
        for rb in self.radio_comp.values():
            self._comp_group.addButton(rb)
            rb.toggled.connect(self.win._on_compass_use_changed)
        ll.addLayout(comp_grid)
        v.addWidget(gb_sel)

        # === ШАГ 5 — СОХРАНИТЬ (gyro + сохранение + применить из архива) ===
        gb_s = QtWidgets.QGroupBox("Шаг 5 — Сохранить")
        ls = QtWidgets.QVBoxLayout(gb_s)
        self.lbl_gyro = QtWidgets.QLabel("Гироскоп bias: —")
        self.lbl_gyro.setWordWrap(True)
        btn_save = QtWidgets.QPushButton("Сохранить калибровку прибора")
        btn_save.setToolTip(
            "Записать АКТИВНУЮ калибровку pc/calibration.json (формат v2: ОБЕ\n"
            "секции магнитометра — свежий RANSAC-кандидат источника, иначе\n"
            "прежняя секция; Live-EKF в файл не пишется) + копию в архив\n"
            "data\\device_calibrations\\.")
        btn_save.clicked.connect(self.win.save_device_calibration)
        self.lbl_save = QtWidgets.QLabel("")
        self.lbl_save.setWordWrap(True)
        ls.addWidget(self.lbl_gyro)
        ls.addWidget(btn_save)
        # Д.1 (пакет 15): применить ГОТОВУЮ калибровку из архива — навигация в
        # «Записи» (секция мигнёт), после «Применить» пульт вернётся сюда
        self.btn_apply_ready = QtWidgets.QPushButton(
            "Применить готовую калибровку… (из архива)")
        self.btn_apply_ready.setToolTip(
            "Перейти на вкладку «Записи» к списку «Готовые калибровки прибора»\n"
            "(он мигнёт зелёной рамкой): выберите файл → «Применить» → пульт\n"
            "вернётся сюда, индикатор под компасом обновится.")
        self.btn_apply_ready.clicked.connect(self.win._on_apply_ready_button)
        ls.addWidget(self.btn_apply_ready)
        ls.addWidget(self.lbl_save)
        v.addWidget(gb_s)
        v.addStretch(1)

        self.src_lat.currentIndexChanged.connect(self._apply_sources)
        self.src_lon.currentIndexChanged.connect(self._apply_sources)
        self.src_alt.currentIndexChanged.connect(self._apply_sources)

    def _spin(self, lo, hi, dec, val):
        s = StepSpinBox()          # пакет 15 (Г): центр, ПКМ «Шаг…», ширина по содержимому
        s.setRange(lo, hi)
        s.setDecimals(dec)
        s.setValue(val)
        return s

    def _mk_combo(self, items):
        c = QtWidgets.QComboBox()
        c.addItems(items)
        return c

    def apply_theme(self, pal):
        # текст берёт цвета из общей темы окна; перекрашиваем графики сравнения и живого сбора
        try:
            fg = pg.mkColor(pal["plot_fg"])
            for plot in (self.cmp_plot, self.live_plot):
                plot.setBackground(pal["plot_bg"])
                for ax in ("left", "bottom"):
                    a = plot.getAxis(ax)
                    a.setTextPen(fg)
                    a.setPen(fg)
                    if a.label is not None:
                        a.label.setDefaultTextColor(fg)
        except Exception:
            pass

    def _open_map(self):
        """Открыть карту OpenStreetMap и взять координаты по клику. Не должна ронять приложение."""
        try:
            dlg = MapDialog(self.spin_lat.value(), self.spin_lon.value(), self)
            ok = dlg.exec()
            if ok and dlg.picked is not None:
                lat, lon = dlg.picked
                self.spin_lat.setValue(lat)
                self.spin_lon.setValue(lon)
                self.src_lat.setCurrentText("по карте")
                self.src_lon.setCurrentText("по карте")
                self.lbl_loc_msg.setText(f"Координаты с карты: φ={lat:.5f}, λ={lon:.5f}")
        except Exception as e:
            self.lbl_loc_msg.setText(f"Карта недоступна: {e}. Введите координаты вручную.")

    def set_gps(self, gps):
        """Запомнить GPS-координаты из файла телефона и обновить поля при источнике «из GPS»."""
        self._gps = gps
        self._apply_sources()

    def _apply_sources(self):
        """Источник: широта/долгота — вручную/GPS/по карте; высота — вручную/барометр/GPS."""
        msgs = []
        gps = self._gps
        self.spin_lat.setReadOnly(False)
        self.spin_lon.setReadOnly(False)
        # широта
        if self.src_lat.currentText() == "из GPS":
            if gps:
                self.spin_lat.setValue(gps["lat"]); msgs.append("Широта из GPS телефона.")
            else:
                msgs.append("GPS телефон пока не передаёт — широта вручную.")
        # долгота
        if self.src_lon.currentText() == "из GPS":
            if gps:
                self.spin_lon.setValue(gps["lon"]); msgs.append("Долгота из GPS телефона.")
            else:
                msgs.append("GPS телефон пока не передаёт — долгота вручную.")
        if "по карте" in (self.src_lat.currentText(), self.src_lon.currentText()):
            msgs.append("Нажмите «Открыть карту…» и выберите точку.")
        # высота: всегда можно поправить ВРУЧНУЮ; исходник с барометра/GPS — в
        # отдельном поле «пришло: …» справа (только чтение)
        src = self.src_alt.currentText()
        self.spin_alt.setReadOnly(False)
        self.alt_src_value.clear()
        if src == "из барометра":
            alt = self.baro_provider() if self.baro_provider else None
            if alt is not None:
                self.spin_alt.setValue(float(alt))
                self.alt_src_value.setText(f"барометр: {alt:.1f} м")
                msgs.append("Высота взята с барометра (можно поправить вручную; "
                            "исходное — в поле «пришло» справа).")
            else:
                self.alt_src_value.setText("барометр: нет")
                msgs.append("Барометр не подключён — высота вручную.")
        elif src == "из GPS":
            if gps and gps.get("alt") is not None:
                self.spin_alt.setValue(gps["alt"])
                self.alt_src_value.setText(f"GPS: {gps['alt']:.1f} м")
                msgs.append("Высота взята с GPS (можно поправить вручную; "
                            "исходное — в поле «пришло» справа).")
            else:
                self.alt_src_value.setText("GPS: нет")
                msgs.append("GPS не дал высоту — вручную.")
        self.lbl_loc_msg.setText("\n".join(msgs))

    def _recalc(self):
        """Кнопка «Пересчитать»: заново посчитать g, поле и блок «Качество», видимо обновить."""
        self._compute_g()
        self._compute_field()

    def _compute_g(self):
        lat, alt = self.spin_lat.value(), self.spin_alt.value()
        g = references.gravity_somigliana(lat, alt)
        self.lbl_g.setText(f"g = {g:.4f}  м/с²   (для φ={lat:.4f}°, h={alt:.0f} м)")
        self.win.set_target_g(g)

    def _compute_field(self):
        # эталон считается ИМЕННО для координат из полей (вручную/карта/GPS),
        # высоты из поля h и выбранной даты — это подписывается под результатом
        qd = self.date_edit.date()
        d = datetime.date(qd.year(), qd.month(), qd.day())
        lat, lon, alt = self.spin_lat.value(), self.spin_lon.value(), self.spin_alt.value()
        try:
            if self.combo_model.currentIndex() == 0:
                res = references.geomag_offline(lat, lon, alt, references.to_decimal_year(d))
            else:
                res = references.geomag_online(lat, lon, alt, d, self.edit_key.text().strip())
        except Exception as e:
            self.lbl_field.setStyleSheet("font-weight:bold; color:#c0392b;")
            self.lbl_field.setText(str(e))
            self.lbl_field_where.setText("")
            return
        self.declination = res["D"]
        self.lbl_field.setStyleSheet("font-weight:bold;")
        self.lbl_field.setText(
            f"F = {res['F']:.2f} мкТл   D = {res['D']:+.2f}°   I = {res['I']:+.2f}°\n({res['source']})")
        self.lbl_field_where.setText(
            f"эталон для φ={lat:.4f}°, λ={lon:.4f}°, h={alt:.0f} м, "
            f"дата {d.strftime('%d.%m.%Y')}")
        self.win.set_target_F(res["F"])

    def show_gyro_bias(self, gb):
        if gb is None:
            self.lbl_gyro.setText("Гироскоп bias: нет в файле")
        else:
            self.lbl_gyro.setText(
                f"Гироскоп bias:\n  [{gb[0]:+.5f} {gb[1]:+.5f} {gb[2]:+.5f}] рад/с")

    def current_location(self):
        return {"lat": self.spin_lat.value(), "lon": self.spin_lon.value(),
                "alt": self.spin_alt.value()}

    def update_quality(self, acc_res, target_g, mag_res, target_F):
        """Шаг 3 (пакет 15, Д.3): строка акселерометра с оценкой словом +
        обновление таблицы «источник × метод» (её строит окно)."""
        if acc_res is None:
            self.lbl_q_acc.setText("Акселерометр: нет данных "
                                   "(нужен файл с 6+ статичными позами)")
            self.lbl_q_acc.setStyleSheet("")
        else:
            resid = acc_res.residual_rel * 100
            word, color = (("отлично", "#2c7a2c") if resid < 1.0 else
                           (("хорошо", "#c09010") if resid < 3.0 else
                            ("плохо — переснимите", "#c0392b")))
            dev_txt = ""
            if target_g:
                dev = abs(acc_res.mean_raw_radius - target_g) / target_g * 100
                dev_txt = f", |a| отличается от g на {dev:.2f}%"
            self.lbl_q_acc.setText(
                f"Акселерометр: остаток {resid:.2f}% — {word}{dev_txt}")
            self.lbl_q_acc.setStyleSheet(f"font-weight:bold; color:{color};")
        upd = getattr(self.win, "_update_matrix", None)
        if upd is not None and getattr(self, "lbl_matrix", None) is not None:
            upd()


# ----------------------------------------------------------------------
# Главное окно
# ----------------------------------------------------------------------
class CalibWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("VarioPro3 — Калибровка датчиков")
        self.resize(1500, 860)
        if os.path.exists(APP_ICON):
            self.setWindowIcon(QtGui.QIcon(APP_ICON))

        # данные и результаты
        self._mag_pts = None
        self._acc_pts = None
        self._gyro_bias = None
        self._acc_result = None
        self._mag_result = None
        # целевые радиусы: g для акселерометра, F (мкТл) для магнитометра
        self.target_g = 9.81
        self.target_F = None     # None → натуральный радиус, пока не вычислено поле
        # EKF-сравнение: поток (t,mag,gyro), результат для сохранения, анимация
        self._mag_stream = None
        self._ekf_save = None
        self._ekf_best = None
        self._ekf_curve = None
        self._ekf_i = 0
        self._ekf_xs = []
        self._ekf_ys = []
        self._ekf_timer = None

        # --- ЖИВОЙ СБОР по потоку (Фаза 5-3; пакет 14 Б.2 — ДВА буфера) ---
        self._worker_provider = None      # main.py: () → SourceWorker вариометра | None
        self._url_provider = None         # main.py: () → URL из поля «Поток» вариометра
        self._live_on = False             # идёт сбор
        self._live_own = None             # наш _LiveReader (если вариометр поток не держит)
        self._live_sub = None             # воркер вариометра, на который подписаны
        # ДВА параллельных буфера — по одному на источник поля (Б.2):
        #   raw     — сырое поле mx..mz (эллипсоид, железо телефона);
        #   android — Android-калиброванное mxa..mza («тонкая» калибровка).
        self._live_buf = {s: {"t": [], "mag": [], "gyro": [], "ekf": None}
                          for s in devcal.SOURCES}
        self._live_ransac = {s: None for s in devcal.SOURCES}  # (CalibResult, stamp)
        self._live_seeds = _fibonacci_sphere(50)   # сектора покрытия сферы
        self._live_xs = []                # график остатка (raw): номер точки
        self._live_ys = []                #   остаток, %
        self._live_warned_notraw = False  # источник отдаёт НЕ сырое поле
        self._data_mag_source = "raw"     # источник mag-данных ЗАГРУЖЕННОГО файла
        self._live_cal_cache = {s: None for s in devcal.SOURCES}  # live-EKF для компаса
        self.compass_use_changed_cb = None   # main.py: сообщить вариометру о смене режима
        self._ui_loading = True              # не реагировать на программную установку радио
        cfg0 = {}
        try:
            with open(CONFIG_PATH, encoding="utf-8") as fh:
                cfg0 = json.load(fh)
        except (OSError, ValueError):
            cfg0 = {}
        self._compass_use0 = devcal.migrate_compass_use(cfg0)
        self._live_timer = QtCore.QTimer(self)
        self._live_timer.setInterval(400)
        self._live_timer.timeout.connect(self._live_tick)
        self._live_timer.start()

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)

        # ---- верхняя панель управления ----
        # «Загрузить файл…» ведёт на вкладку «Записи» (нужный список мигнёт);
        # маленькая «Обзор…» рядом — классический диалог для файлов вне data\
        self.files_nav_cb = None            # ставит main.py: перейти на «Записи»
        self.calibration_saved_cb = None    # ставит main.py: калибровка сохранена
        bar = QtWidgets.QHBoxLayout()
        btn_load = QtWidgets.QPushButton("Загрузить файл…")
        btn_load.setToolTip("Выбрать запись калибровки на вкладке «Записи» (список подсветится).\n"
                            "Для файла вне data\\ — кнопка «Обзор…» рядом.")
        btn_load.clicked.connect(self._on_load_button)
        bar.addWidget(btn_load)
        btn_browse = QtWidgets.QPushButton("Обзор…")
        btn_browse.setToolTip("Классический диалог выбора файла (для файлов вне data\\)")
        # Д.2 (пакет 15): фикс-ширина 64 обрезала «Обзор…» — ширина по содержимому
        btn_browse.clicked.connect(self.on_load)
        bar.addWidget(btn_browse)
        btn_demo = QtWidgets.QPushButton("Демо (синтетика)")
        btn_demo.clicked.connect(self.load_demo)
        bar.addWidget(btn_demo)
        btn_help = QtWidgets.QPushButton("Инструкция")
        btn_help.clicked.connect(self._show_help)
        bar.addWidget(btn_help)
        bar.addStretch(1)
        self.lbl_status = QtWidgets.QLabel("Загрузите calib_*.json с телефона или «Демо».")
        bar.addWidget(self.lbl_status)
        root.addLayout(bar)

        # ---- слева ГРИД 2 РАВНЫЕ колонки (обе 3D-сцены одинакового размера,
        # блок В.1), справа — панель эталонов ----
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.mag_panel = CalibPanel("МАГНИТОМЕТР  (mx, my, mz)", "мкТл")
        self.acc_panel = CalibPanel("АКСЕЛЕРОМЕТР  (ax, ay, az)", "м/с²")
        panels = QtWidgets.QWidget()
        pgrid = QtWidgets.QGridLayout(panels)
        pgrid.setContentsMargins(0, 0, 0, 0)
        pgrid.addWidget(self.mag_panel, 0, 0)
        pgrid.addWidget(self.acc_panel, 0, 1)
        pgrid.setColumnStretch(0, 1)
        pgrid.setColumnStretch(1, 1)       # строго поровну, независимо от контента
        self.ref_panel = ReferencePanel(self)
        ref_scroll = QtWidgets.QScrollArea()
        ref_scroll.setWidget(self.ref_panel)
        ref_scroll.setWidgetResizable(True)
        ref_scroll.setMinimumWidth(360)
        split.addWidget(panels)
        split.addWidget(ref_scroll)
        split.setSizes([1040, 380])
        root.addWidget(split, stretch=1)

        # режим компаса из config.json (v2: метод@источник); live-режим без
        # идущего сбора невозможен — ЯВНО понижаем до RANSAC того же источника
        # (никакого молчаливого fallback, Б.2)
        use0 = self._compass_use0
        if use0.startswith("live@"):
            use0 = "ransac@" + use0.split("@", 1)[1]
            self._write_compass_use(use0)
            self.lbl_status.setText(
                "Компас: Live-EKF был выбран, но сбора нет → переключено на "
                + use0 + " (запустите «Слушать поток», чтобы вернуть Live-EKF).")
        self.ref_panel.radio_comp.get(use0, self.ref_panel.
                                      radio_comp["ransac@android"]).setChecked(True)
        self._ui_loading = False
        self._update_matrix()

    def set_baro_provider(self, fn):
        """Подключить источник высоты из барометра (вызывается из main.py)."""
        self.ref_panel.baro_provider = fn

    # ------------------------------------------------------------------
    # «КОМПАС ИСПОЛЬЗУЕТ» (пакет 14, Б.2): метод × источник, 4 варианта
    # ------------------------------------------------------------------
    def _compass_use_now(self) -> str:
        for key, rb in self.ref_panel.radio_comp.items():
            if rb.isChecked():
                return key
        return "ransac@android"

    def _write_compass_use(self, mode: str):
        try:
            with open(CONFIG_PATH, encoding="utf-8") as fh:
                cfg = json.load(fh)
        except (OSError, ValueError):
            cfg = {}
        cfg["compass_use"] = mode
        cfg.pop("compass_mag_source", None)   # старый ключ больше не нужен
        try:
            with open(CONFIG_PATH, "w", encoding="utf-8") as fh:
                json.dump(cfg, fh, ensure_ascii=False, indent=2)
        except OSError:
            pass

    def _on_compass_use_changed(self, checked: bool = False):
        if self._ui_loading or not checked:
            return
        mode = self._compass_use_now()
        self._write_compass_use(mode)
        self._update_matrix()          # Д.3: отметка ▶ следует за селектором
        if self.compass_use_changed_cb:
            self.compass_use_changed_cb(mode)
        method, _, src = mode.partition("@")
        src_txt = "сырое поле" if src == "raw" else "Android-поле"
        self.lbl_status.setText(
            "Компас использует: "
            + (f"RANSAC ({src_txt}, секция калибровки mag_{src})."
               if method == "ransac" else
               f"Live-EKF ({src_txt}) — подстройка, пока идёт сбор; в файл не пишется."))

    def live_mag_cal(self, source: str):
        """Для компаса вариометра: текущая live-EKF калибровка источника
        ("raw"/"android") — {V, M, residual_pct} или None (сбор не идёт /
        EKF не готов). Обновляется тиком 0.4 с — по сэмплам дёшево."""
        return self._live_cal_cache.get(source)

    def _show_help(self):
        HelpDialog(self).exec()

    def apply_theme(self, pal: dict):
        """Перекрасить оба 3D-вида, панели цифр и панель эталонов под тему."""
        self.mag_panel.apply_theme(pal)
        self.acc_panel.apply_theme(pal)
        self.ref_panel.apply_theme(pal)

    # ------------------------------------------------------------------
    def _on_load_button(self):
        """«Загрузить файл…»: перейти на вкладку «Записи» (мигнёт список калибровок);
        если окно запущено отдельно (без вкладок) — классический диалог."""
        if self.files_nav_cb is not None:
            self.files_nav_cb("calib")
        else:
            self.on_load()

    def on_load(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Выберите файл калибровки", DATA_DIR,
            "Калибровка/запись (*.json *.csv);;Все файлы (*)")
        if path:
            self._apply(path)

    def load_demo(self):
        """Загрузить синтетику data/samples/sample_calib.csv (создать, если её нет)."""
        path = os.path.join(SAMPLES_DIR, "sample_calib.csv")
        if not os.path.exists(path):
            try:
                import make_calib_sample
                make_calib_sample.main()
            except Exception as e:
                self.lbl_status.setText(f"Не удалось создать демо: {e}")
                return
        self._apply(path)

    def _apply(self, path: str):
        """Загрузить файл (calib_*.json ИЛИ session CSV) и посчитать калибровку."""
        gps = None
        self._stop_ekf_timer()          # прервать прошлую анимацию EKF
        self._mag_stream = None
        self._ekf_save = None
        self._ekf_best = None
        try:
            if path.lower().endswith(".json"):
                self._gyro_bias, self._acc_pts, self._mag_pts, gps = load_calib_json(path)
                try:
                    self._mag_stream = mag_ekf.load_mag_stream(path)   # t,mag,gyro
                except Exception:
                    self._mag_stream = None
            else:
                arr = load_session_csv(path)
                self._acc_pts = extract_xyz(arr, ("ax", "ay", "az"))
                self._mag_pts = extract_xyz(arr, ("mx", "my", "mz"))
                self._gyro_bias = None
                self._mag_stream = session_stream(arr)                 # t,mag,gyro (если есть гиро)
        except Exception as e:
            self.lbl_status.setText(f"Ошибка чтения: {e}")
            return

        self.ref_panel.show_gyro_bias(self._gyro_bias)
        self.ref_panel.set_gps(gps)
        # источник mag-данных файла: session-CSV и телефонные calib_*.json несут
        # СЫРОЕ поле; сырьё живого сбора могло быть снято с Android-поля
        src_long = None
        if path.lower().endswith(".json"):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    src_long = json.load(fh).get("mag_source")
            except (OSError, ValueError):
                src_long = None
        self._data_mag_source = devcal.LONG2SHORT.get(src_long or "", "raw")
        self._recompute()
        self._reset_ekf_compare_ui()
        self._update_matrix()
        self.lbl_status.setText(f"Загружено: {os.path.basename(path)} "
                                f"(поле: {self._data_mag_source})")

    def _recompute(self):
        """Пересчитать обе калибровки: акселерометр (диагональ, цель g), магнитометр (эллипсоид, цель F)."""
        # акселерометр — модель «смещение + масштаб по осям» (диагональная)
        self._acc_result = self._calc(self.acc_panel, self._acc_pts, self.target_g,
                                      diagonal=True, min_pts=6,
                                      missing="Нет точек акселерометра (ax,ay,az)")
        # магнитометр — полный эллипсоид (hard-iron + soft-iron с перекрёстными)
        self._mag_result = self._calc(self.mag_panel, self._mag_pts, self.target_F,
                                      diagonal=False, min_pts=20, robust=True,
                                      missing="Нет точек магнитометра (mx,my,mz)")
        self.ref_panel.update_quality(self._acc_result, self.target_g,
                                      self._mag_result, self.target_F)

    def _calc(self, panel, pts, target, diagonal, min_pts, missing, robust=False):
        if pts is None:
            panel.set_message(missing)
            return None
        if len(pts) < min_pts:
            panel.set_message(f"Мало точек ({len(pts)}). Нужно ≥ {min_pts}.")
            return None
        try:
            if diagonal:
                res = calibrate_diagonal(pts, target_radius=target)
            elif robust:
                # магнитометр: эллипсоид + отбраковка выбросов (RANSAC)
                res = calibrate_robust(pts, target_radius=target, model="ellipsoid")
            else:
                res = calibrate(pts, target_radius=target)
            panel.set_data(res)
            return res
        except Exception as e:
            panel.set_message(f"Не удалось откалибровать: {e}")
            return None

    def set_target_g(self, g: float):
        """Эталон g (из «Вычислить g») → целевой радиус акселерометра."""
        self.target_g = float(g)
        self._acc_result = self._calc(self.acc_panel, self._acc_pts, self.target_g,
                                      diagonal=True, min_pts=6,
                                      missing="Нет точек акселерометра (ax,ay,az)")
        self.ref_panel.update_quality(self._acc_result, self.target_g,
                                      self._mag_result, self.target_F)

    def set_target_F(self, F: float):
        """Эталон F (из «Получить поле») → целевой радиус магнитометра."""
        self.target_F = float(F)
        self._mag_result = self._calc(self.mag_panel, self._mag_pts, self.target_F,
                                      diagonal=False, min_pts=20, robust=True,
                                      missing="Нет точек магнитометра (mx,my,mz)")
        self.ref_panel.update_quality(self._acc_result, self.target_g,
                                      self._mag_result, self.target_F)

    # ------------------------------------------------------------------
    # СРАВНЕНИЕ МЕТОДОВ: прогон EKF по записи с анимацией сходимости к RANSAC
    # ------------------------------------------------------------------
    def _stop_ekf_timer(self):
        t = getattr(self, "_ekf_timer", None)
        if t is not None:
            t.stop()
            self._ekf_timer = None

    def _reset_ekf_compare_ui(self):
        """Сбросить блок сравнения при загрузке нового файла."""
        rp = self.ref_panel
        rp.cmp_curve_ekf.setData([], [])
        rp.cmp_line_ransac.setValue(0)
        rp.lbl_cmp_table.setText("")
        if self._mag_stream is None:
            rp.btn_run_ekf.setEnabled(False)
            rp.lbl_cmp_live.setText(
                "<span style='color:#c0a000'>Нет гироскопа в файле — EKF недоступен. "
                "Нужен calib_*.json или session CSV с колонками gx,gy,gz.</span>")
        else:
            rp.btn_run_ekf.setEnabled(True)
            n = len(self._mag_stream[0])
            rp.lbl_cmp_live.setText(
                f"Готово к прогону EKF ({n} точек). Нажмите «Прогнать EKF по загруженному файлу».")

    def run_ekf_compare(self):
        """Кнопка «Прогнать EKF по записи»: считаем EKF и анимируем сходимость к RANSAC."""
        rp = self.ref_panel
        if self._mag_stream is None:
            rp.lbl_cmp_live.setText("<span style='color:#c0392b'>Нет потока с гироскопом.</span>")
            return
        if self._mag_result is None:
            rp.lbl_cmp_live.setText(
                "<span style='color:#c0392b'>Сначала должен посчитаться RANSAC (магнитометр).</span>")
            return
        self._stop_ekf_timer()
        t, mag, gyro = self._mag_stream
        F_ref = self._mag_result.radius
        rp.lbl_cmp_live.setText("Считаю EKF…")
        rp.btn_run_ekf.setEnabled(False)
        QtWidgets.QApplication.processEvents()
        try:
            best, _p, _n = mag_ekf.run_mag_ekf_autosign(
                t, mag, gyro, F_ref=F_ref, record_checkpoints=60)
        except Exception as e:
            rp.lbl_cmp_live.setText(f"<span style='color:#c0392b'>Ошибка EKF: {e}</span>")
            rp.btn_run_ekf.setEnabled(True)
            return
        self._ekf_best = best
        self._ekf_curve = best.get("curve", [])
        if not self._ekf_curve:
            rp.lbl_cmp_live.setText("<span style='color:#c0392b'>EKF не дал кривую (мало точек?).</span>")
            rp.btn_run_ekf.setEnabled(True)
            return
        # горизонтальная линия остатка RANSAC + рамки графика
        res_r = self._mag_result.residual_rel * 100.0
        rp.cmp_line_ransac.setValue(res_r)
        kmax = self._ekf_curve[-1]["k"]
        ymax = max(res_r, max(c["residual"] * 100 for c in self._ekf_curve)) * 1.15 + 1
        rp.cmp_plot.setXRange(0, kmax, padding=0.05)
        rp.cmp_plot.setYRange(0, ymax, padding=0)
        rp.cmp_curve_ekf.setData([], [])
        self._ekf_xs, self._ekf_ys = [], []
        self._ekf_i = 0
        self._ekf_timer = QtCore.QTimer(self)
        self._ekf_timer.setInterval(70)
        self._ekf_timer.timeout.connect(self._ekf_tick)
        self._ekf_timer.start()

    def _ekf_tick(self):
        if not self._ekf_curve or self._ekf_i >= len(self._ekf_curve):
            self._ekf_finish()
            return
        rp = self.ref_panel
        c = self._ekf_curve[self._ekf_i]
        self._ekf_xs.append(c["k"])
        self._ekf_ys.append(c["residual"] * 100.0)
        rp.cmp_curve_ekf.setData(self._ekf_xs, self._ekf_ys)
        self._update_cmp_labels(c)
        self._ekf_i += 1

    def _cmp_divergences(self, absVk, Vk, res_k, field_k):
        """Расхождения EKF↔RANSAC: по |V|, центру V (вектор), остатку, полю (в %)."""
        r = self._mag_result
        Vr = np.asarray(r.V, float)
        absVr = float(np.linalg.norm(Vr))
        res_r = r.residual_rel * 100.0
        field_r = float(r.radius)
        d_absV = abs(absVk - absVr) / absVr * 100 if absVr > 0 else 0.0
        d_center = float(np.linalg.norm(np.asarray(Vk, float) - Vr)) / absVr * 100 if absVr > 0 else 0.0
        d_res = abs(res_k - res_r) / res_r * 100 if res_r > 0 else 0.0
        d_field = abs(field_k - field_r) / field_r * 100 if field_r > 0 else 0.0
        return absVr, res_r, field_r, d_absV, d_center, d_res, d_field

    def _update_cmp_labels(self, c):
        absVk = c["absV"]; Vk = c["V_uT"]; res_k = c["residual"] * 100.0; field_k = c["field"]
        absVr, res_r, field_r, d_absV, d_center, d_res, d_field = self._cmp_divergences(
            absVk, Vk, res_k, field_k)
        self.ref_panel.lbl_cmp_live.setText(
            f"<b>обработано точек: {c['k']}</b><br>"
            f"|V| (железо): EKF <b>{absVk:.0f}</b> / RANSAC <b>{absVr:.0f}</b> мкТл — расх. {d_absV:.1f}%<br>"
            f"центр V (вектор): расх. {d_center:.1f}%<br>"
            f"остаток: EKF <b>{res_k:.1f}%</b> / RANSAC <b>{res_r:.1f}%</b> — расх. {d_res:.0f}%<br>"
            f"поле: EKF {field_k:.1f} / RANSAC {field_r:.1f} мкТл — расх. {d_field:.1f}%")

    def _ekf_finish(self):
        self._stop_ekf_timer()
        rp = self.ref_panel
        rp.btn_run_ekf.setEnabled(True)
        best = self._ekf_best
        if best is None:
            return
        Vk = np.asarray(best["V_uT"], float)
        absVk = float(np.linalg.norm(Vk))
        res_k = best["residual_rel"] * 100.0
        field_k = float(best["field_uT"])
        self._update_cmp_labels({"k": best["n"], "absV": absVk, "V_uT": Vk,
                                 "residual": best["residual_rel"], "field": field_k})
        absVr, res_r, field_r, d_absV, d_center, d_res, d_field = self._cmp_divergences(
            absVk, Vk, res_k, field_k)
        rows = [
            ("|V|, мкТл", f"{absVk:.0f}", f"{absVr:.0f}", f"{d_absV:.1f}%"),
            ("центр V", "—", "—", f"{d_center:.1f}%"),
            ("остаток, %", f"{res_k:.1f}", f"{res_r:.1f}", f"{d_res:.0f}%"),
            ("поле, мкТл", f"{field_k:.1f}", f"{field_r:.1f}", f"{d_field:.1f}%"),
        ]
        head = f"{'величина':<12}{'EKF':>8}{'RANSAC':>9}{'расх.':>8}\n"
        body = "\n".join(f"{a:<12}{b:>8}{cc:>9}{d:>8}" for (a, b, cc, d) in rows)
        self.ref_panel.lbl_cmp_table.setText(
            f"<b>Итоговая таблица</b> (знак ΔR {best['sign']:+.0f}):"
            f"<pre style='font-family:Consolas,monospace;font-size:11pt'>{head}{body}</pre>")
        # запомнить результат EKF для сохранения (если выбран метод EKF)
        self._ekf_save = {
            "V": Vk,
            "M_corr": np.asarray(best["Winv"], float),  # b = W⁻¹(z−V), уже привязан к F
            "radius": field_k,
            "residual_rel": float(best["residual_rel"]),
        }

    # ------------------------------------------------------------------
    # ЖИВОЙ СБОР КАЛИБРОВКИ ПО ПОТОКУ (Фаза 5, шаг 3)
    # ------------------------------------------------------------------
    def set_stream_providers(self, worker_provider, url_provider):
        """main.py: доступ к живому потоку вариометра (воркер + URL поля «Поток»)."""
        self._worker_provider = worker_provider
        self._url_provider = url_provider

    def _vario_worker_live(self):
        """Воркер вариометра, если он сейчас держит ЖИВОЙ поток (иначе None)."""
        w = self._worker_provider() if self._worker_provider else None
        try:
            if w is not None and getattr(w.source, "live", False) and w.isRunning():
                return w
        except RuntimeError:
            pass                       # воркер уже удалён Qt — считаем, что его нет
        return None

    def toggle_live_capture(self):
        """Кнопка «Слушать поток» / «Остановить сбор»."""
        if self._live_on:
            self._live_stop("сбор остановлен")
        else:
            self._live_start()

    def _live_start(self):
        # новый сбор — с чистого листа (оба буфера, оба EKF, график)
        for src in devcal.SOURCES:
            self._live_buf[src] = {"t": [], "mag": [], "gyro": [],
                                   "ekf": mag_ekf.LiveMagEKF(sigma_m_uT=0.8)}
            self._live_cal_cache[src] = None
        self._live_xs, self._live_ys = [], []
        self._live_warned_notraw = False
        self.ref_panel.live_curve.setData([], [])
        w = self._vario_worker_live()
        if w is not None:
            # поток уже открыт вариометром → ПОДПИСКА (второе подключение не открываем)
            w.sampleReady.connect(self._on_live_sample)
            self._live_sub = w
            src_txt = "подписка на поток ВАРИОМЕТРА (общее подключение)"
        else:
            url = (self._url_provider() if self._url_provider else "") or \
                "socket://127.0.0.1:5555"
            self._live_own = _LiveReader(url)
            self._live_own.sampleReady.connect(self._on_live_sample)
            self._live_own.start()
            src_txt = f"своё подключение: {url}"
        self._live_on = True
        # пункты Live-EKF селектора оживают на время сбора (Б.2)
        for key in ("live@raw", "live@android"):
            self.ref_panel.radio_comp[key].setEnabled(True)
        self.ref_panel.btn_live.setText("⏹ Остановить сбор")
        self.ref_panel.lbl_live_status.setText(
            src_txt + " · собираются ОБА поля (raw и android)")
        self.mag_panel.set_message("Живой сбор: вращайте телефон во всех плоскостях…")
        self.lbl_status.setText("Живой сбор калибровки: точки копятся (оба источника).")

    def _live_stop(self, why: str):
        self._live_on = False
        for src in devcal.SOURCES:
            self._live_cal_cache[src] = None
        # live-EKF недоступен без сбора: пункты гаснут; если он был ВЫБРАН —
        # ЯВНО переключаемся на RANSAC того же источника (не молча, Б.2)
        use = self._compass_use_now()
        if use.startswith("live@"):
            new = "ransac@" + use.split("@", 1)[1]
            self.ref_panel.radio_comp[new].setChecked(True)   # штатный путь
            self.lbl_status.setText(
                f"Сбор остановлен → компас переключён на {new} "
                f"(Live-EKF без сбора невозможен).")
        for key in ("live@raw", "live@android"):
            self.ref_panel.radio_comp[key].setEnabled(False)
        if self._live_sub is not None:
            try:
                self._live_sub.sampleReady.disconnect(self._on_live_sample)
            except (TypeError, RuntimeError):
                pass
            self._live_sub = None
        if self._live_own is not None:
            self._live_own.stop()
            self._live_own.wait(2000)
            self._live_own.deleteLater()
            self._live_own = None
        self.ref_panel.btn_live.setText("▶ Слушать поток")
        n_raw = len(self._live_buf["raw"]["t"])
        n_and = len(self._live_buf["android"]["t"])
        self.ref_panel.lbl_live_status.setText(
            f"{why}; собрано raw {n_raw} · android {n_and} точек (буферы сохранены)")

    @QtCore.Slot(object)
    def _on_live_sample(self, s):
        """Каждая точка потока → в ОБА буфера (Б.2): сырое поле mx..mz в raw,
        Android-калиброванное mxa..mza в android; у каждого свой живой EKF.
        Поток v3 без mxa — наполняется только raw."""
        if not self._live_on or s.gyro3 is None:
            return
        fed = False
        if s.mag3 is not None and getattr(s, "mag_raw", False):
            b = self._live_buf["raw"]
            b["t"].append(float(s.t))
            b["mag"].append([float(v) for v in s.mag3])
            b["gyro"].append([float(v) for v in s.gyro3])
            try:
                b["ekf"].add(s.t, s.mag3, s.gyro3)
            except Exception:
                pass                    # одна битая точка не должна ронять сбор
            fed = True
        ma = getattr(s, "mag3a", None)
        if ma is not None:
            b = self._live_buf["android"]
            b["t"].append(float(s.t))
            b["mag"].append([float(v) for v in ma])
            b["gyro"].append([float(v) for v in s.gyro3])
            try:
                b["ekf"].add(s.t, ma, s.gyro3)
            except Exception:
                pass
            fed = True
        if (not fed and s.mag3 is not None and not getattr(s, "mag_raw", False)
                and not self._live_warned_notraw):
            self._live_warned_notraw = True
            self.ref_panel.lbl_live_status.setText(
                "⚠ источник отдаёт НЕ сырое поле (синтетика?) — калибровка по нему "
                "некорректна (нужен телефонный поток/симулятор)")

    def _live_coverage_pct(self, mags: np.ndarray) -> float:
        """Покрытие сферы: доля из 50 телесных секторов (Фибоначчи-семена),
        в которых есть хоть одна точка. Направления — от текущего центроида,
        и ТОЛЬКО от точек с заметным радиусом (шум изотропен и «покрыл» бы
        все сектора даже на неподвижном телефоне)."""
        if len(mags) < 10:
            return 0.0
        d = mags - mags.mean(axis=0)
        n = np.linalg.norm(d, axis=1)
        keep = n > mag_ekf.LiveMagEKF.MIN_SPREAD_UT
        if not keep.any():
            return 0.0
        d = d[keep] / n[keep, None]
        owner = np.argmax(d @ self._live_seeds.T, axis=1)
        return 100.0 * len(np.unique(owner)) / len(self._live_seeds)

    def _live_tick(self):
        """Таймер 0.4 с: живые метки/облако/график/таблица; конфликт подключений."""
        rp = self.ref_panel
        n_raw = len(self._live_buf["raw"]["t"])
        n_and = len(self._live_buf["android"]["t"])
        have = max(n_raw, n_and)
        rp.btn_live_ransac.setEnabled(have >= 100)
        rp.btn_live_save.setEnabled(have >= 100)
        if not self._live_on:
            return
        # конфликт подключений: вариометр запустил поток, пока мы держали своё →
        # отдаём канал ему и подписываемся (телефон принимает ОДНО подключение)
        w = self._vario_worker_live()
        if self._live_own is not None and w is not None:
            self._live_own.stop()
            self._live_own.wait(2000)
            self._live_own.deleteLater()
            self._live_own = None
            w.sampleReady.connect(self._on_live_sample)
            self._live_sub = w
            rp.lbl_live_status.setText(
                "поток перехвачен вариометром → подписка на его подключение")
        # подписка умерла (вариометр остановил поток) — сбор на паузу
        if self._live_sub is not None and self._vario_worker_live() is None:
            self._live_stop("поток вариометра остановлен — нажмите «Слушать» заново")
            return
        if have == 0:
            return
        # покрытие считаем по буферу с бОльшим числом точек (вращение общее)
        rich = "raw" if n_raw >= n_and else "android"
        mags = np.asarray(self._live_buf[rich]["mag"], float)
        cov = self._live_coverage_pct(mags)
        rp.lbl_live_stats.setText(
            f"Точек: raw {n_raw} · android {n_and} · "
            f"Покрытие сферы: {cov:.0f}% (цель ≥80%)")
        # живое облако сырого буфера на большой сфере магнитометра (центрируем —
        # иначе железо ~1230 мкТл уносит облако далеко от начала координат)
        if n_raw:
            p = self.mag_panel
            raw_m = np.asarray(self._live_buf["raw"]["mag"], float)
            pts = raw_m - raw_m.mean(axis=0)
            col = np.tile(np.array(p._point_color(0), dtype=np.float32),
                          (len(pts), 1))
            p.scatter_raw.setData(pos=pts, color=col, size=p._point_size(),
                                  pxMode=True)
            ext = float(np.abs(pts).max()) if len(pts) else 60.0
            p.set_cam_auto(max(ext, 30.0) * 2.4)
            p._set_info_html(
                f'<span style="color:{p._leg_raw}">■ сырые (живой сбор, '
                f'центрировано)</span><br>N = {n_raw} точек · покрытие {cov:.0f}%')
        # живые EKF обоих источников: строка + кэш для компаса; график — raw
        lines = []
        for src in devcal.SOURCES:
            ekf = self._live_buf[src]["ekf"]
            m = ekf.metrics() if ekf else {"ready": False}
            if m.get("ready"):
                lines.append(
                    f"EKF@{src} (знак {m['sign']:+.0f}): |V| {m['absV_uT']:.0f} мкТл · "
                    f"поле {m['field_uT']:.1f} мкТл · остаток "
                    f"{m['residual_rel'] * 100:.1f}%")
                if src == "raw":
                    self._live_xs.append(m["n"])
                    self._live_ys.append(m["residual_rel"] * 100.0)
                    rp.live_curve.setData(self._live_xs, self._live_ys)
                try:
                    res = ekf.result(F_ref=self.target_F)
                except Exception:
                    res = None
                if res is not None:
                    self._live_cal_cache[src] = {
                        "V": np.asarray(res["V_uT"], float),
                        "M": np.asarray(res["Winv"], float),
                        "residual_pct": float(res["residual_rel"]) * 100.0,
                        "source": src,
                    }
            else:
                lines.append(
                    f"EKF@{src}: жду вращения — секторов {m.get('pre_sectors', 0)}/"
                    f"{m.get('need_sectors', 15)} (точек {m.get('n', 0)})")
        rp.lbl_live_ekf.setText("\n".join(lines) if lines else "EKF: —")
        self._update_matrix()

    def live_compute_ransac(self):
        """«Посчитать RANSAC по собранному (только магнитометр)» — Б.3: пакетный
        RANSAC по КАЖДОМУ собранному маг-буферу (raw и android) в ОТДЕЛЬНЫЕ
        кандидаты. Аксель/гироскоп/загруженный файл НЕ трогаются (это и был
        старый баг «calibration без accel»)."""
        done = []
        for src in devcal.SOURCES:
            buf = self._live_buf[src]
            if len(buf["t"]) < 100:
                continue
            try:
                res = calibrate_robust(np.asarray(buf["mag"], float),
                                       target_radius=self.target_F,
                                       model="ellipsoid")
            except Exception as e:
                self.lbl_status.setText(f"RANSAC@{src} не посчитался: {e}")
                continue
            stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._live_ransac[src] = (res, stamp)
            done.append(f"{src}: остаток {res.residual_rel * 100:.2f}% "
                        f"({len(buf['t'])} точек)")
        if not done:
            self.lbl_status.setText(
                "Мало точек для RANSAC (нужно ≥100 хотя бы в одном буфере) — "
                "продолжайте сбор.")
            return
        # показать СЫРОЙ кандидат на 3D-сфере магнитометра (только показ:
        # состояние загруженного файла не меняется)
        show = self._live_ransac["raw"] or self._live_ransac["android"]
        if show is not None:
            self.mag_panel.set_data(show[0])
        self._update_matrix()
        self.lbl_status.setText(
            "RANSAC по собранному (только магнитометр): " + " · ".join(done)
            + ". Дальше — «Сохранить калибровку прибора» (запишет обе секции).")

    def live_save_json(self):
        """«Сохранить сырьё…» (Б.3, только магнитометр): собранные буферы →
        data\\calib_*.json. Сырое поле — основным mag_stream (как у телефона),
        Android-поле — рядом в mag_stream_android (читатели незнакомые ключи
        игнорируют). Аксель/гироскоп/загруженный файл не трогаются."""
        n_raw = len(self._live_buf["raw"]["t"])
        n_and = len(self._live_buf["android"]["t"])
        if max(n_raw, n_and) < 100:
            self.lbl_status.setText("Мало точек для сохранения (нужно ≥100).")
            return
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = os.path.join(DATA_DIR, f"calib_{stamp}.json")

        def stream(src):
            b = self._live_buf[src]
            return [[round(b["t"][i], 4)] + [round(v, 4) for v in b["mag"][i]]
                    + [round(v, 6) for v in b["gyro"][i]]
                    for i in range(len(b["t"]))]

        main_src = "raw" if n_raw >= 100 else "android"
        obj = {
            "format": "variopro_calib", "version": 1,
            "created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "device": "живой поток (ПК, Фаза 5-3)",
            "mag_source": devcal.SHORT2LONG[main_src],
            "mag_stream_columns": ["t", "mx", "my", "mz", "gx", "gy", "gz"],
            "mag_stream": stream(main_src),
        }
        other = "android" if main_src == "raw" else "raw"
        if len(self._live_buf[other]["t"]) >= 100:
            obj["mag_stream_android" if other == "android"
                else "mag_stream_raw"] = stream(other)
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(obj, fh, ensure_ascii=False)
        except OSError as e:
            self.lbl_status.setText(f"Не удалось сохранить: {e}")
            return
        self.lbl_status.setText(
            f"✓ Сырьё сохранено: {os.path.basename(path)} (raw {n_raw} · "
            f"android {n_and} точек) — открывается штатной загрузкой.")

    # ------------------------------------------------------------------
    # СВОДНАЯ ТАБЛИЦА «источник × метод» (пакет 14, Б.2)
    # ------------------------------------------------------------------
    def _update_matrix(self):
        """Таблица «источник × метод» (пакет 15, Д.3): остаток % каждой из 4
        комбинаций С ЦВЕТОВОЙ ОЦЕНКОЙ (отлично <3% / хорошо <10% / плохо ≥10%);
        ВЫБРАННАЯ для компаса комбинация помечена ▶ и жирным. RANSAC — свежий
        кандидат живого сбора, иначе результат загруженного файла, иначе
        сохранённая секция calibration.json; Live-EKF — живые числа при сборе."""
        saved = devcal.load()
        rows = []                     # (ключ, имя, остаток|None, происхождение)
        for src in devcal.SOURCES:
            cand = self._live_ransac.get(src)
            if cand is not None:
                res, stamp = cand
                rows.append((f"ransac@{src}", f"RANSAC@{src}",
                             res.residual_rel * 100, f"живой сбор {stamp[11:16]}"))
            elif (self._mag_result is not None
                  and self._data_mag_source == src):
                rows.append((f"ransac@{src}", f"RANSAC@{src}",
                             self._mag_result.residual_rel * 100,
                             "загруженный файл"))
            else:
                sec = devcal.mag_section(saved, src)
                if sec is not None and sec.get("residual_pct") is not None:
                    rows.append((f"ransac@{src}", f"RANSAC@{src}",
                                 float(sec["residual_pct"]),
                                 f"сохранённая ({(sec.get('created') or '?')[:16]})"))
                else:
                    rows.append((f"ransac@{src}", f"RANSAC@{src}", None,
                                 "нет данных"))
        for src in devcal.SOURCES:
            lc = self._live_cal_cache.get(src)
            if lc is not None:
                rows.append((f"live@{src}", f"Live-EKF@{src}",
                             float(lc["residual_pct"]), "идёт сбор"))
            else:
                rows.append((f"live@{src}", f"Live-EKF@{src}", None,
                             "сбор не идёт" if not self._live_on else "EKF греется"))
        use_now = self._compass_use_now()
        lines = []
        for (key, name, resid, origin) in rows:
            sel = key == use_now
            mark = "▶ " if sel else "  "
            if resid is None:
                res_txt, verdict, color = "     —", "", "#8a93a0"
            else:
                verdict, color = (("отлично", "#2c7a2c") if resid < 3.0 else
                                  (("хорошо", "#c09010") if resid < 10.0 else
                                   ("плохо", "#c0392b")))
                res_txt = f"{resid:6.2f}%"
            row_txt = (f"{mark}{name:<18}{res_txt:>9}  {verdict:<8} {origin}")
            row_html = html.escape(row_txt)
            if resid is not None:
                row_html = row_html.replace(html.escape(verdict),
                                            f'<span style="color:{color};">'
                                            f'{html.escape(verdict)}</span>', 1)
            if sel:
                row_html = f"<b>{row_html}</b>"
            lines.append(row_html)
        self.ref_panel.lbl_matrix.setText(
            "<b>Источник × метод (остаток = разброс длины вектора после "
            "калибровки, %):</b>"
            f"<pre style='font-family:Consolas,monospace;font-size:10pt;"
            f"margin:2px 0'>" + "\n".join(lines) + "</pre>")

    def _on_apply_ready_button(self):
        """Д.1 (пакет 15): «Применить готовую калибровку…» → вкладка «Записи»,
        секция готовых калибровок мигнёт; после «Применить» пульт вернётся сюда
        (возврат настраивает main.py). Отдельного окна — классический диалог нет:
        без вкладок кнопка просто подсказывает путь."""
        if self.files_nav_cb is not None:
            self.files_nav_cb("devcal")
        else:
            self.lbl_status.setText(
                "Готовые калибровки лежат в data\\device_calibrations\\ — "
                "применить можно из вкладки «Записи» полного пульта.")

    def save_device_calibration(self):
        """Сохранить калибровку прибора в pc/calibration.json — ФОРМАТ v2
        (пакет 14, Б.2): ОБЕ секции магнитометра (mag_raw и mag_android).
        Приоритет секции источника: свежий RANSAC-кандидат живого сбора →
        результат загруженного файла (если его источник совпадает) → прежняя
        секция из активной калибровки (НЕ теряется). Live-EKF в файл не пишется."""
        prev = devcal.load()               # прежняя активная (уже нормализована v2)
        has_live = any(self._live_ransac[s] is not None for s in devcal.SOURCES)
        if (self._acc_result is None and self._mag_result is None
                and self._gyro_bias is None and not has_live):
            self.ref_panel.lbl_save.setStyleSheet("color:#c0392b;")
            self.ref_panel.lbl_save.setText(
                "Нечего сохранять — загрузите файл калибровки или после живого "
                "сбора нажмите «Посчитать RANSAC по собранному».")
            return
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        out = {
            "format": "variopro_device_calibration", "version": 2,
            "created": now,
            "gyro_bias": (self._gyro_bias.tolist()
                          if self._gyro_bias is not None
                          else prev.get("gyro_bias")),
            "declination_deg": (self.ref_panel.declination
                                if self.ref_panel.declination is not None
                                else prev.get("declination_deg")),
            "location": self.ref_panel.current_location(),
        }
        kept = []                          # что унаследовано от прежней активной
        if self._gyro_bias is None and prev.get("gyro_bias") is not None:
            kept.append("гироскоп")
        if self._acc_result is not None:
            r = self._acc_result
            out["accel"] = {
                "model": "diagonal",
                "offset": r.V.tolist(),
                "scales": r.scales.tolist(),
                "target_g": r.radius,
                "residual_pct": r.residual_rel * 100,
            }
        elif prev.get("accel"):
            out["accel"] = prev["accel"]
            kept.append("аксель")
        # секции магнитометра — по источникам (метод в файле ВСЕГДА RANSAC)
        def mag_dict(res: CalibResult, created: str) -> dict:
            return {"model": "ellipsoid",
                    "hard_iron": res.V.tolist(),
                    "soft_iron": res.M_corr.tolist(),
                    "target_F_uT": res.radius,
                    "residual_pct": res.residual_rel * 100,
                    "created": created}
        fresh = []
        for src in devcal.SOURCES:
            key = devcal.MAG_KEYS[src]
            cand = self._live_ransac.get(src)
            if cand is not None:
                out[key] = mag_dict(cand[0], cand[1])
                fresh.append(f"{src}: живой сбор")
            elif self._mag_result is not None and self._data_mag_source == src:
                out[key] = mag_dict(self._mag_result, now)
                fresh.append(f"{src}: загруженный файл")
            elif prev.get(key):
                out[key] = prev[key]
                kept.append(f"магнитометр {src}")
        path = os.path.join(PC_DIR, "calibration.json")
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(out, fh, ensure_ascii=False, indent=2)
            # копия в АРХИВ применённых калибровок (вкладка «Записи» → «Применить»)
            arch_note = ""
            try:
                os.makedirs(DEVCAL_DIR, exist_ok=True)
                stamp = now.replace(" ", "_").replace(":", "-")
                arch = os.path.join(DEVCAL_DIR, f"calibration_{stamp}.json")
                with open(arch, "w", encoding="utf-8") as fh:
                    json.dump(out, fh, ensure_ascii=False, indent=2)
                arch_note = f"\n+ копия в архив: {os.path.basename(arch)}"
            except OSError:
                arch_note = "\n(копия в архив не записалась)"
            self.ref_panel.lbl_save.setStyleSheet("color:#2c7a2c; font-weight:bold;")
            note = ("  (" + "; ".join(fresh) + ")") if fresh else ""
            kept_note = (f"\nиз прежней активной: {', '.join(kept)}" if kept else "")
            self.ref_panel.lbl_save.setText(
                f"✓ Сохранено (v2): {path}{note}{kept_note}{arch_note}")
            self._update_matrix()
            if self.calibration_saved_cb:
                self.calibration_saved_cb()   # обновить индикатор под компасом и «Записи»
        except OSError as e:
            self.ref_panel.lbl_save.setStyleSheet("color:#c0392b;")
            self.ref_panel.lbl_save.setText(f"Ошибка сохранения: {e}")


# ----------------------------------------------------------------------
def apply_dark_theme(app: QtWidgets.QApplication):
    app.setStyleSheet("""
        QWidget { background-color:#14161a; color:#d6dbe1; }
        QMainWindow, QSplitter::handle { background-color:#14161a; }
        QLabel { color:#d6dbe1; }
        QPushButton { background:#222a35; border:1px solid #3a4554;
                      padding:6px 12px; border-radius:4px; }
        QPushButton:hover { background:#2c3744; }
        QDoubleSpinBox { background:#1b212a; border:1px solid #3a4554; padding:3px; }
        QSlider::groove:horizontal { height:6px; background:#2a323d; border-radius:3px; }
        QSlider::handle:horizontal { background:#5aa0ff; width:14px;
                                     margin:-5px 0; border-radius:7px; }
    """)


def main():
    parser = argparse.ArgumentParser(description="VarioPro3 — калибровка датчиков (Фаза 2)")
    parser.add_argument("--selftest", action="store_true",
                        help="загрузить демо, снять скриншоты в docs/ и выйти")
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    if os.path.exists(APP_ICON):
        app.setWindowIcon(QtGui.QIcon(APP_ICON))
    apply_dark_theme(app)
    win = CalibWindow()
    win.show()
    win.load_demo()   # сразу показываем синтетику

    if args.selftest:
        def capture_and_quit():
            os.makedirs(DOCS_DIR, exist_ok=True)
            # 3D-содержимое снимаем из самих GL-виджетов (надёжнее, чем окно целиком)
            for name, panel in (("calib_mag", win.mag_panel), ("calib_accel", win.acc_panel)):
                try:
                    panel.glview.grabFramebuffer().save(os.path.join(DOCS_DIR, f"{name}.png"))
                except Exception as e:
                    print(f"grab {name}: {e}")
            try:
                win.grab().save(os.path.join(DOCS_DIR, "screenshot_calib.png"))
            except Exception:
                pass
            mr = win.mag_panel.result
            ar = win.acc_panel.result
            if mr is not None:
                print(f"Магнитометр  V найден: [{mr.V[0]:+.3f} {mr.V[1]:+.3f} {mr.V[2]:+.3f}]"
                      f"  (истина 14, -9, 7),  остаток {mr.residual_rel*100:.2f}%")
            if ar is not None:
                print(f"Акселерометр V найден: [{ar.V[0]:+.3f} {ar.V[1]:+.3f} {ar.V[2]:+.3f}]"
                      f"  (истина 0.2, -0.15, 0.3),  остаток {ar.residual_rel*100:.2f}%")
            print("Скриншоты сохранены в docs/: calib_mag.png, calib_accel.png, screenshot_calib.png")
            app.quit()
        QtCore.QTimer.singleShot(2500, capture_and_quit)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
