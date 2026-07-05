# -*- coding: utf-8 -*-
"""
vario_app.py
============
ДЕСКТОП-ПРИЛОЖЕНИЕ ВАРИОМЕТРА (Фаза 0, без телефона).

Что показывает окно:
    • два живых графика: ВЫСОТА (м) и ВАРИОМЕТР (м/с), ось X — секунды;
    • крупные цифры текущей высоты и вертикальной скорости;
    • кнопки Старт / Стоп / Сброс;
    • выбор источника данных: Симуляция или CSV-файл;
    • галочки «показать фильтрованное» и «показать сырое»;
    • панель «Параметры фильтра»: режим Авто/Ручной, поля R (шум баро, σ²),
      Q (через sigma_accel) и время калибровки, кнопка «Применить»; настройки
      сохраняются в config.json и подхватываются при следующем запуске;
    • число текущего оценённого смещения акселерометра;
    • выбор «Окно просмотра» (последние 5/10/20/30/60 с или «Всё») и «Шаг сетки
      по времени» — обе настройки тоже сохраняются в config.json;
    • кнопку «Экспорт PNG» (картинка строится через matplotlib с backend Agg,
      чтобы окно не зависало).

Масштабирование графиков мышью (колесо — зум, перетаскивание — сдвиг, правый
клик — «View All») работает всегда; во время чтения вид сам следует за данными,
на паузе — полностью в вашем распоряжении.

КАК ЭТО УСТРОЕНО (простыми словами)
-----------------------------------
1) Источник данных (sensor_source.py) выдаёт замеры: время, ускорение, высота баро.
2) Каждый замер прогоняется через ФИЛЬТР Калмана (baro_inertial_vario.py) —
   получаем СГЛАЖЕННУЮ высоту и вертикальную скорость (это и есть вариометр).
3) Для сравнения считаем «сырой» вариометр — простую производную баро
   (как делают «в лоб»; он сильно шумит — это видно на графике).
4) Чтение данных идёт в ОТДЕЛЬНОМ потоке (SourceWorker), чтобы окно не
   подвисало. Поток присылает замеры в окно через сигнал Qt.
5) Перерисовка графиков — по таймеру ~30 кадров/с, отдельно от прихода данных
   (так плавно и не нагружает процессор).

Запуск:
    python pc/vario_app.py                  # обычный режим (откроется окно)
    python pc/vario_app.py --selftest       # самопроверка: запустит симуляцию,
                                            # сохранит скриншот в docs/ и закроется
"""

from __future__ import annotations

import os
import sys
import json
import argparse
from collections import deque
from datetime import datetime
from math import sin, cos, atan2, radians, degrees, sqrt, exp

import numpy as np

from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

# наши модули (лежат рядом, в папке pc/)
from baro_inertial_vario import BaroInertialVario
from sensor_source import SimSource, CsvSource, StreamSource, Sample
from mekf import make_mekf, load_mekf_config   # ориентация MEKF (Фаза 5, шаг 2)
import device_calibration as devcal            # калибровка прибора v2 (пакет 14, Б.2)
import crashlog                                # лог событий (watchdog, адаптация)
from widgets import (StepSpinBox, make_delta_field,   # единый спинбокс (пакет 15, Г)
                     CardsPanel)                      # карточки шапки (пакет 15, Е)

# ----------------------------------------------------------------------
# Пути проекта: корень = папка над pc/
# ----------------------------------------------------------------------
PC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(PC_DIR)
DATA_DIR = os.path.join(ROOT, "data")
SAMPLES_DIR = os.path.join(DATA_DIR, "samples")            # демо-файлы sample_*
LAYOUTS_DIR = os.path.join(DATA_DIR, "layouts")            # виды пульта (пакет 15, Е)
DOCS_DIR = os.path.join(ROOT, "docs")
APP_ICON = os.path.join(ROOT, "assets", "logo.png")        # логотип для иконок окон
DEVICE_CALIB_PATH = os.path.join(PC_DIR, "calibration.json")  # калибровка прибора

# ----------------------------------------------------------------------
# Настройки отображения
# ----------------------------------------------------------------------
DT = 0.02                 # шаг фильтра, с (данные Фазы 0 идут с частотой 50 Гц)
BUFFER_POINTS = 6000      # сколько последних точек храним (~120 c при 50 Гц)
WARMUP_SEC = 1.5          # прогрев фильтра после старта: точки не выводим, числа «—»
                          # (вместе с посевом первым баро убирает ложный пик на старте)

# ----------------------------------------------------------------------
# Параметры фильтра: значения ПО УМОЛЧАНИЮ (используются в авто-режиме)
# ----------------------------------------------------------------------
# Фильтру нужны два «шумовых» числа:
#   R = шум барометра (дисперсия, м²)         — насколько НЕ доверяем баро;
#   sigma_accel = шум ускорения (СКО, м/с²)   — через него задаётся Q (шум процесса).
# В фильтре R задаётся как sigma_baro², поэтому при создании пересчитываем:
#   sigma_baro = sqrt(R).
SIGMA_BARO_DEFAULT = 0.15                       # СКО баро по умолчанию, м
R_DEFAULT = round(SIGMA_BARO_DEFAULT ** 2, 6)   # = 0.0225 м² (дисперсия)
SIGMA_ACCEL_DEFAULT = 0.30                      # СКО ускорения по умолчанию, м/с²
CALIB_TIME_DEFAULT = 5.0                        # время усреднения для калибровки, с

# параметры просмотра по умолчанию
GRID_STEP_DEFAULT = 10.0                        # шаг сетки по времени по умолчанию, с

# цвета слотов ручной настройки (пакет 14, А.5): кривые, подписи, курсор
SLOT_COLORS = {"s1": "#17a2b8", "s2": "#d63fb0"}   # бирюзовый / пурпурный

# файл настроек в корне проекта (рядом с README.md)
CONFIG_PATH = os.path.join(ROOT, "config.json")

# светлая «научная» тема для графиков
pg.setConfigOptions(antialias=True, background="w", foreground="k")


def _norm_view(view) -> dict:
    """
    Привести настройки просмотра к виду {alt:{…}, vario:{…}} — у КАЖДОГО графика
    своё окно, шаг сетки и режим оси Y (авто по окну / ручной min-max).
    Понимает и СТАРЫЙ общий формат {window_sec, grid_step}: раскладывает его
    в оба графика (обратная совместимость).
    """
    out = {
        "alt":   {"window_sec": None, "grid_step": GRID_STEP_DEFAULT,
                  "y_mode": "auto", "y_min": 0.0, "y_max": 300.0},
        "vario": {"window_sec": None, "grid_step": GRID_STEP_DEFAULT,
                  "y_mode": "auto", "y_min": -3.0, "y_max": 3.0},
    }
    if not isinstance(view, dict):
        return out
    # старый общий формат → в оба графика
    if "window_sec" in view or "grid_step" in view:
        ws = view.get("window_sec", None)
        for k in ("alt", "vario"):
            out[k]["window_sec"] = None if ws is None else float(ws)
            try:
                out[k]["grid_step"] = float(view.get("grid_step", GRID_STEP_DEFAULT))
            except (TypeError, ValueError):
                pass
    # новый формат (отдельные блоки) — имеет приоритет
    for k in ("alt", "vario"):
        sub = view.get(k)
        if isinstance(sub, dict):
            if "window_sec" in sub:
                v = sub["window_sec"]
                out[k]["window_sec"] = None if v is None else float(v)
            if "grid_step" in sub:
                try:
                    out[k]["grid_step"] = float(sub["grid_step"])
                except (TypeError, ValueError):
                    pass
            if sub.get("y_mode") in ("auto", "manual"):
                out[k]["y_mode"] = sub["y_mode"]
            for kk in ("y_min", "y_max"):
                if kk in sub:
                    try:
                        out[k][kk] = float(sub[kk])
                    except (TypeError, ValueError):
                        pass
    return out


def load_config() -> dict:
    """
    Прочитать настройки из config.json. Если файла нет или он повреждён —
    вернуть значения по умолчанию. Никогда не падаем из-за плохого файла.
    """
    cfg = {
        "mode": "auto",  # "auto" или "manual"
        "manual": {
            "R": R_DEFAULT,
            "sigma_accel": SIGMA_ACCEL_DEFAULT,
            "calib_time": CALIB_TIME_DEFAULT,
        },
        "view": _norm_view(None),  # по графику: окно просмотра + шаг сетки
        "smooth_sec": 0.0,         # гауссово сглаживание вариометра (третье число), с; 0 = выкл
        "show_smooth": False,      # галочка «показать сглаженное» (зелёные кривые)
        "compass_tau_sec": 0.15,   # сглаживание стрелки компаса (вектор), с
                                   # (AHRS гладкий сам — оставлен лишь короткий хвост)
        # что компас использует (пакет 14, Б.2): метод@источник —
        # "ransac@android" (по умолчанию) | "ransac@raw" | "live@android" | "live@raw".
        # Старые значения ("saved"/"live_ekf" + compass_mag_source) мигрируются.
        "compass_use": "ransac@android",
        "show_second": False,      # пунктир 2-го метода верт. ускорения (Г.1)
        # ДВА СЛОТА ручной настройки R/Q (пакет 14, А.5; заменили «теневой фильтр»)
        "manual_slots": {
            "s1": {"R": R_DEFAULT, "sigma_accel": SIGMA_ACCEL_DEFAULT, "show": False},
            "s2": {"R": R_DEFAULT, "sigma_accel": SIGMA_ACCEL_DEFAULT, "show": False},
        },
        "sound_source": "vario",   # звук от: "vario" (фильтр) | "smooth" (Сглаж. N с)
        "panel_layout": None,      # активный вид пульта: имя файла в data\layouts
                                   # или None = заводской (пакет 15, Е.3)
        # адаптивные R/Q в режиме «Авто» (Фаза 5 / блок З.1, Sage-Husa-класс);
        # False = прежний Авто с фиксированными параметрами
        "adaptive_rq": True,
        "show_rts": False,         # кривая RTS-сглаживателя в файловом режиме (З.2)
        # источник вертикального ускорения из сырого IMU:
        #   "mekf"   — проекция через ориентацию MEKF (pc/mekf.py), по умолчанию;
        #   "scalar" — старое скалярное приближение |a_калибр| − g
        "vertical_accel_mode": "mekf",
        # детектор покоя/движения (переменное доверие акселерометру); все пороги здесь
        "motion": {
            "gyro_rest": 0.15,   # ПОКОЙ: |гироскоп| ниже, рад/с …
            "acc_rest": 0.3,     # … И ||a|−g| ниже, м/с² …
            "hold_sec": 0.3,     # … устойчиво столько секунд
            "gyro_dyn": 0.5,     # ДИНАМИКА: |гироскоп| выше, рад/с …
            "acc_dyn": 0.8,      # … ИЛИ ||a|−g| выше, м/с²
            "k_dyn": 5.0,        # множитель на sigma_accel в динамике (недоверие акселю)
            "trans_sec": 0.3,    # плавный переход множителя, с
        },
        # ZUPT — «прибить» вариометр к нулю при СТРОГОМ покое (пакет 15, А.1)
        "zupt": {
            "enabled": True,
            "gyro_max": 0.05,    # сглаж. |гироскоп| ниже, рад/с …
            "acc_max": 0.15,     # … И сглаж. ||f|−g| ниже, м/с² …
            "hold_sec": 0.3,     # … устойчиво столько секунд → измерение v=0
            "sigma_v": 0.02,     # СКО ZUPT-измерения, м/с (R_zupt = σ²)
        },
        # робастный баро (Хьюбер, А.2): |e| > k·√S → R×(|e|/(k√S))²; 0 = выкл
        "huber_k": 2.0,
        # детектор КАЧАНИЯ (А.3): знакопеременное вращение (размахивание) —
        # баро «дышит» портом, на время качания его дисперсия R × k_baro
        "osc": {
            "enabled": True,
            "gz_std": 0.8,       # скользящее std(gz) за окно выше, рад/с …
            "mean_ratio": 0.5,   # … при |среднее(gz)| < ratio·std (знакопеременно)
            "win_sec": 1.0,      # окно статистики, с
            # удержание состояния. ТЗ предлагало 0.5 с (не мигать в нулях
            # фазы); поднято до 2.0 с — перекрывает ПАУЗЫ между сериями
            # взмахов (на 23-21-13 окно 258–273: |v| 4.0 → 0.10). Безопасно:
            # строгий покой (ZUPT-условие 0.3 с) снимает качание ДОСРОЧНО
            "hold_sec": 2.0,
            "k_baro": 8.0,       # множитель R баро в качании
        },
    }
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
            loaded = json.load(fh)
        if isinstance(loaded, dict):
            if loaded.get("mode") in ("auto", "manual"):
                cfg["mode"] = loaded["mode"]
            man = loaded.get("manual", {})
            if isinstance(man, dict):
                for key in ("R", "sigma_accel", "calib_time"):
                    if key in man:
                        cfg["manual"][key] = float(man[key])
            cfg["view"] = _norm_view(loaded.get("view"))
            if "smooth_sec" in loaded:
                try:
                    cfg["smooth_sec"] = float(loaded["smooth_sec"])
                except (TypeError, ValueError):
                    pass
            mo = loaded.get("motion", {})
            if isinstance(mo, dict):
                for key in cfg["motion"]:
                    if key in mo:
                        try:
                            cfg["motion"][key] = float(mo[key])
                        except (TypeError, ValueError):
                            pass
            # ZUPT / Хьюбер / качание (пакет 15, блок А)
            for grp in ("zupt", "osc"):
                sub = loaded.get(grp, {})
                if isinstance(sub, dict):
                    for key in cfg[grp]:
                        if key in sub:
                            try:
                                cfg[grp][key] = (bool(sub[key]) if key == "enabled"
                                                 else float(sub[key]))
                            except (TypeError, ValueError):
                                pass
            if "huber_k" in loaded:
                try:
                    cfg["huber_k"] = float(loaded["huber_k"])
                except (TypeError, ValueError):
                    pass
            if "show_smooth" in loaded:
                cfg["show_smooth"] = bool(loaded["show_smooth"])
            if "compass_tau_sec" in loaded:
                try:
                    cfg["compass_tau_sec"] = float(loaded["compass_tau_sec"])
                except (TypeError, ValueError):
                    pass
            if loaded.get("vertical_accel_mode") in ("mekf", "scalar"):
                cfg["vertical_accel_mode"] = loaded["vertical_accel_mode"]
            # compass_use: новый формат метод@источник; старые значения мигрируются
            cfg["compass_use"] = devcal.migrate_compass_use(loaded)
            if "show_second" in loaded:
                cfg["show_second"] = bool(loaded["show_second"])
            if loaded.get("sound_source") in ("vario", "smooth"):
                cfg["sound_source"] = loaded["sound_source"]
            if isinstance(loaded.get("panel_layout"), str):
                cfg["panel_layout"] = loaded["panel_layout"]
            # слоты ручной настройки (А.5); старый shadow_filter мигрирует в слот 1
            slots = loaded.get("manual_slots")
            if isinstance(slots, dict):
                for sk in ("s1", "s2"):
                    sub = slots.get(sk)
                    if isinstance(sub, dict):
                        for kk in ("R", "sigma_accel"):
                            if kk in sub:
                                try:
                                    cfg["manual_slots"][sk][kk] = float(sub[kk])
                                except (TypeError, ValueError):
                                    pass
                        if "show" in sub:
                            cfg["manual_slots"][sk]["show"] = bool(sub["show"])
            else:
                sfl = loaded.get("shadow_filter")
                if isinstance(sfl, dict):
                    try:
                        cfg["manual_slots"]["s1"] = {
                            "R": float(sfl.get("R", R_DEFAULT)),
                            "sigma_accel": float(sfl.get("sigma_accel",
                                                         SIGMA_ACCEL_DEFAULT)),
                            "show": bool(sfl.get("on", False))}
                    except (TypeError, ValueError):
                        pass
            if "adaptive_rq" in loaded:
                cfg["adaptive_rq"] = bool(loaded["adaptive_rq"])
            if "show_rts" in loaded:
                cfg["show_rts"] = bool(loaded["show_rts"])
            # блок «sound» просто ПРОБРАСЫВАЕТСЯ (его читает sound_app).
            # БАГ пакета 13: load_config его не возвращал вовсе — настройки
            # звука писались в config.json, но никогда не читались обратно
            if isinstance(loaded.get("sound"), dict):
                cfg["sound"] = loaded["sound"]
    except (FileNotFoundError, ValueError, OSError, TypeError):
        pass  # любой сбой → остаёмся на значениях по умолчанию
    return cfg


def save_config(cfg: dict) -> None:
    """
    Сохранить настройки в config.json (без падения при ошибке записи).
    Сначала читаем уже записанное и ОБЪЕДИНЯЕМ, чтобы не затереть ключи, которые
    пишет кто-то другой (например, выбор темы из главного окна main.py).
    """
    existing = {}
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
            loaded = json.load(fh)
        if isinstance(loaded, dict):
            existing = loaded
    except (FileNotFoundError, ValueError, OSError):
        existing = {}
    existing.update(cfg)
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as fh:
            json.dump(existing, fh, ensure_ascii=False, indent=2)
    except OSError:
        pass


def nice_vario_range(v: np.ndarray):
    """
    Подобрать «красивую» рамку по оси Y для графика вариометра.

    Масштаб подстраиваем под ФИЛЬТРОВАННЫЙ сигнал (по нему мы и летаем),
    но делаем рамку не меньше ±2 м/с и добавляем небольшой запас. Благодаря
    этому полезная линия всегда хорошо видна, а редкие выбросы «сырого»
    сигнала просто уходят за край графика и не мешают.
    """
    if v.size == 0:
        return -2.0, 2.0
    lo, hi = float(np.min(v)), float(np.max(v))
    center = 0.5 * (lo + hi)
    half = max(0.5 * (hi - lo) * 1.25, 2.0)  # минимум ±2 м/с
    return center - half, center + half


# ======================================================================
# КОМПАС: курс с наклон-компенсацией (ОСИ ANDROID)
# ======================================================================
def tilt_compensated_heading(ax, ay, az, mx, my, mz, decl_deg=0.0):
    """
    Курс 0..360° в ОСЯХ ANDROID: x — вправо, y — к верхнему краю экрана,
    z — из экрана; в покое az ≈ +g (реакция опоры вверх).

    Метод — тот же, что в Android SensorManager.getRotationMatrix/getOrientation:
        E = m × a  — направление «восток» в осях телефона,
        N = a × E  — направление «север» в осях телефона,
        курс верхнего края телефона ψ = atan2(E_y, N_y) + склонение D.
    Наклон-компенсация получается автоматически: двойное векторное произведение
    и есть проекция поля в горизонтальную плоскость, заданную вектором a.

    ВАЖНО: прежняя формула (roll/pitch + проекция, AN4248) была написана для
    «самолётных» осей (x вперёд, z вниз) и на данных Android давала ЗЕРКАЛЬНЫЙ
    курс (отклик −1 на вращение) — это и показал полевой тест.

    accel — м/с², mag — мкТл (ПОСЛЕ калибровки прибора). Возвращает None в
    вырожденном случае (|a| ≈ 0 — свободное падение, или поле вдоль вертикали).

    Считается скалярно (без numpy): функция зовётся на каждом IMU-сэмпле
    (417 Гц), а нужны от векторов только y-компоненты E и N и их нормы.
    """
    # E = m × a (восток в осях телефона)
    ex = my * az - mz * ay
    ey = mz * ax - mx * az
    ez = mx * ay - my * ax
    nE = sqrt(ex * ex + ey * ey + ez * ez)
    if nE < 1e-6:
        return None
    # N = a × E (север в осях телефона)
    nx = ay * ez - az * ey
    ny = az * ex - ax * ez
    nz = ax * ey - ay * ex
    nN = sqrt(nx * nx + ny * ny + nz * nz)
    if nN < 1e-6:
        return None
    # нормировка обязательна: |E| ≠ |N|, atan2 по ненормированным компонентам врал бы
    heading = degrees(atan2(ey / nE, ny / nN)) + decl_deg
    return heading % 360.0


def cardinal_ru(deg: float) -> str:
    """Сторона света по курсу (8 румбов, по-русски)."""
    dirs = ["С", "СВ", "В", "ЮВ", "Ю", "ЮЗ", "З", "СЗ"]
    return dirs[int((deg % 360.0) / 45.0 + 0.5) % 8]


class DotIndicator(QtWidgets.QWidget):
    """Круглый индикатор приёма пакетов: зелёный (пульсирует), красный, серый."""

    def __init__(self, diameter: int = 14):
        super().__init__()
        self._color = QtGui.QColor("#777777")
        self.setFixedSize(diameter + 4, diameter + 4)
        self._d = diameter

    def set_color(self, color):
        c = QtGui.QColor(color)
        if c != self._color:
            self._color = c
            self.update()

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(self._color)
        p.drawEllipse(QtCore.QPointF(self.width() / 2, self.height() / 2),
                      self._d / 2, self._d / 2)


class PlotInteractGuard(QtCore.QObject):
    """
    Ловит РЕАЛЬНЫЕ действия мыши на графике (колесо / перетаскивание левой
    кнопкой) через фильтр событий вьюпорта — независимо от того, какие сигналы
    шлёт конкретная версия pyqtgraph. По действию вызывает колбэк приложения:
    отключить слежение за временем и (если пользователь менял ось Y) перевести
    ЭТОТ график в ручной режим Y. Клик для курсора-инспектора (без движения)
    ничего не трогает.
    """

    DRAG_PX = 8   # порог: меньше — это клик (курсор), больше — панорама

    def __init__(self, app, key: str, viewport):
        super().__init__(viewport)
        self.app = app
        self.key = key
        self._press = None

    def eventFilter(self, obj, ev):
        t = ev.type()
        if t == QtCore.QEvent.Wheel:
            self.app._on_user_plot_interact(self.key)
        elif t == QtCore.QEvent.MouseButtonPress and ev.buttons() & QtCore.Qt.LeftButton:
            self._press = ev.position().toPoint()
        elif t == QtCore.QEvent.MouseMove and (ev.buttons() & QtCore.Qt.LeftButton):
            if (self._press is not None and
                    (ev.position().toPoint() - self._press).manhattanLength() > self.DRAG_PX):
                self._press = None
                self.app._on_user_plot_interact(self.key)
        elif t == QtCore.QEvent.MouseButtonRelease:
            self._press = None
        return False    # событие НЕ съедаем — pyqtgraph работает как обычно


def make_value_label(sample: str, color: str, px: int = 0) -> QtWidgets.QLabel:
    """
    QLabel под ЖИВОЕ число: моноширинный шрифт + ФИКСИРОВАННАЯ ширина (по образцу
    самой длинной строки sample) + выравнивание влево. Цифры меняются «на месте»,
    соседние подписи не дёргаются. px=0 — размер шрифта по умолчанию.
    """
    f = QtGui.QFont("Consolas")
    f.setStyleHint(QtGui.QFont.Monospace)   # нет Consolas → любой моноширинный
    if px > 0:
        f.setPixelSize(px)
    f.setBold(True)
    lbl = QtWidgets.QLabel("—")
    lbl.setFont(f)
    lbl.setStyleSheet(f"color: {color};")   # только цвет: шрифт задан через QFont
    fm = QtGui.QFontMetrics(f)
    lbl.setFixedWidth(fm.horizontalAdvance(sample) + 6)  # запас под самое длинное
    # пакет 15 (Е.1): цифры по центру поля (ширина фиксирована — не прыгает)
    lbl.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
    return lbl


class CompassWidget(QtWidgets.QWidget):
    """Круглый компас со стрелкой курса. setHeading(градусы)."""

    def __init__(self, size=120):
        super().__init__()
        self._heading = None
        self.setFixedSize(size, size)

    def setHeading(self, deg):
        self._heading = deg
        self.update()

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        w = self.width(); h = self.height()
        cx, cy = w / 2.0, h / 2.0
        r = min(w, h) / 2.0 - 4
        # циферблат
        p.setPen(QtGui.QPen(QtGui.QColor("#888"), 2))
        p.setBrush(QtGui.QColor(255, 255, 255, 20))
        p.drawEllipse(QtCore.QPointF(cx, cy), r, r)
        # буквы N/E/S/W
        p.setPen(QtGui.QColor("#aaa"))
        f = p.font(); f.setPointSize(8); f.setBold(True); p.setFont(f)
        for txt, ang in (("N", 0), ("E", 90), ("S", 180), ("W", 270)):
            a = radians(ang)
            x = cx + (r - 9) * sin(a)
            y = cy - (r - 9) * cos(a)
            p.drawText(QtCore.QRectF(x - 8, y - 8, 16, 16),
                       QtCore.Qt.AlignCenter, txt)
        if self._heading is None:
            return
        # стрелка курса (красная — вперёд/на курс, серая — назад)
        a = radians(self._heading)
        dx, dy = sin(a), -cos(a)
        tipx, tipy = cx + dx * (r - 12), cy + dy * (r - 12)
        tailx, taily = cx - dx * (r - 22), cy - dy * (r - 22)
        p.setPen(QtGui.QPen(QtGui.QColor("#888"), 3))
        p.drawLine(QtCore.QPointF(cx, cy), QtCore.QPointF(tailx, taily))
        p.setPen(QtGui.QPen(QtGui.QColor("#d83b3b"), 3))
        p.drawLine(QtCore.QPointF(cx, cy), QtCore.QPointF(tipx, tipy))
        p.setBrush(QtGui.QColor("#d83b3b"))
        p.drawEllipse(QtCore.QPointF(tipx, tipy), 4, 4)


# ======================================================================
# ПОТОК ЧТЕНИЯ ИСТОЧНИКА
# ======================================================================
class SourceWorker(QtCore.QThread):
    """
    Отдельный поток: крутит цикл чтения источника и присылает каждый замер
    в окно через сигнал sampleReady. Так окно не «замерзает», пока мы ждём
    данные (особенно важно для будущего Bluetooth, где чтение блокирующее).
    """

    sampleReady = QtCore.Signal(object)               # объект Sample (t, a, h, accel3, mag3)
    errorOccurred = QtCore.Signal(str)                # текст ошибки
    finished = QtCore.Signal()                        # источник закончился/остановлен

    def __init__(self, source):
        super().__init__()
        self.source = source
        self._running = True

    def run(self):
        # 1) открыть источник
        try:
            self.source.open()
        except Exception as e:
            self.errorOccurred.emit(f"Не удалось открыть источник: {e}")
            self.finished.emit()
            return
        # 2) читать замеры, пока не остановят или не кончатся данные.
        #    Для ЖИВОГО потока (source.live) None означает «данных сейчас нет»
        #    (тишина/переподключение) — продолжаем крутить цикл, а не выходим.
        try:
            live = bool(getattr(self.source, "live", False))
            while self._running:
                s = self.source.read_sample()
                if s is None:
                    if live:
                        continue
                    break
                self.sampleReady.emit(s)
        except Exception as e:
            self.errorOccurred.emit(f"Ошибка чтения данных: {e}")
        finally:
            # 3) аккуратно закрыть источник
            try:
                self.source.close()
            except Exception:
                pass
            self.finished.emit()

    def stop(self):
        """Попросить поток остановиться (он сам выйдет из цикла)."""
        self._running = False


# ======================================================================
# ГЛАВНОЕ ОКНО
# ======================================================================
class VarioApp(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("VarioPro3 — Фаза 0 (ПК)")
        self.resize(1100, 760)
        if os.path.exists(APP_ICON):
            self.setWindowIcon(QtGui.QIcon(APP_ICON))

        # --- настройки (читаются из config.json при запуске) ---
        cfg = load_config()
        self.mode = cfg["mode"]                   # "auto" / "manual"
        self.manual_params = dict(cfg["manual"])  # пользовательские R / sigma_accel / calib_time
        self._init_view = dict(cfg["view"])       # окно просмотра и шаг сетки (применим после сборки)
        self._init_smooth = float(cfg.get("smooth_sec", 0.0))  # гауссово сглаживание (с)
        self._loading = False                     # защита от лишних сигналов при настройке

        # --- состояние обработки ---
        self.filter = None                       # ядро-фильтр (создадим в _sync_filter_panel)
        self.worker: SourceWorker | None = None  # поток чтения (когда запущен)
        self.csv_path: str | None = None         # выбранный CSV-файл
        self._prev_t = None                      # для «сырого» вариометра (производная)
        self._prev_h = None
        self._dirty = False                      # есть ли новые данные для перерисовки
        self._view_refresh = False               # нужно один раз обновить рамки графиков
        self._cursor_t = None                    # инспектор по клику: время курсора t_c (None = нет)
        self._t_first = None                     # время первого сэмпла (посев + прогрев фильтра)
        self._warming = False                    # идёт прогрев (числа «—», точки не выводим)
        self._dot_seen = 0                       # для пульса индикатора приёма
        self._dot_flash = 0.0
        self._h_ref = None                       # «Ноль высоты»: опорная высота (None = абсолютная)
        self._init_show_smooth = bool(cfg.get("show_smooth", False))
        self._compass_tau = float(cfg.get("compass_tau_sec", 0.5))  # сглаживание стрелки, с
        self._head_vec = None                    # сглаженный единичный вектор курса (cos, sin)
        self._neutral_fg = "#d6dbe1"             # нейтральный цвет чисел (из темы, apply_theme)
        self._cal_meta = None                    # метаданные активной калибровки (метод/дата/остаток)
        # авто-масштаб Y: когда пересчитывать в следующий раз (по каждому графику)
        self._y_next = {"alt": 0.0, "vario": 0.0}

        # --- ФАЙЛОВЫЙ РЕЖИМ (CSV): весь файл прогнан пакетно, проигрывание = плеер ---
        self._file = None                        # dict полных серий или None (live-режим)
        self._playing = False                    # идёт проигрывание по сериям
        self._play_i = 0                         # индекс текущей точки
        self._play_t = 0.0                       # текущее время проигрывания
        self._play_wall = None                   # wall-часы прошлого тика плеера
        self._follow = True                      # вид следует за проигрыванием

        # --- компас (курс) ---
        # К СЫРОМУ полю (session CSV / будущий Bluetooth-raw, Sample.mag_raw=True) перед
        # расчётом курса применяем калибровку прибора: m = M·(сырое − V) (hard/soft-iron из
        # calibration.json) — иначе из-за железа телефона (~1230 мкТл) курс пляшет при наклоне.
        # Для Android-калиброванного/синтетического поля (mag_raw=False) калибровку НЕ трогаем.
        self._heading = None                     # последний курс, °
        self._decl = 0.0                         # склонение D, ° (из calibration.json)
        # калибровка магнитометра v2 (пакет 14, Б.2): по СЕКЦИИ на источник поля
        self._mag_cal = {"raw": None, "android": None}   # devcal.mag_section(...)
        self._warned_no_cal = False              # предупреждали про сырое поле без калибровки?
        self._noted_no_android = False           # в данных нет android-поля (старый файл)
        self._mag_F_ref = None                   # эталон |B| (мкТл) для гейта AHRS
        # «Компас использует»: метод@источник (ransac|live @ raw|android)
        self._compass_use = cfg.get("compass_use", "ransac@android")
        self.live_mag_provider = None            # main.py: (source) → live-EKF V/M или None
        self._calib_ind_live_txt = None          # кэш живой строки индикатора
        self.sound_cb = None                     # main.py: звук вариометра (блок Ж)
        self.sound_source = cfg.get("sound_source", "vario")  # звук от: vario | smooth (А.2)

        # --- установка нуля акселерометра (старт-усреднение a за первые N секунд) ---
        self._zero_N = 0.0                        # окно усреднения, с (из поля «Установка нуля»)
        self._zero_done = False                   # уже установили ноль?
        self._zero_accum = 0.0                    # сумма a за окно
        self._zero_count = 0                      # число отсчётов в окне
        self._zero_elapsed = 0.0                  # накоплено секунд ПОКОЯ (усредняем только в покое)

        # --- детектор ПОКОЙ/ДВИЖЕНИЕ + переменное доверие акселерометру ---
        # В движении настоящие ускорения вращения (рычаг датчика ω²r + ошибка
        # ориентации) портят вертикальное ускорение → фильтр должен меньше верить
        # акселю (множитель k_dyn на sigma_accel) и не обновлять его ноль (bias).
        self._motion_cfg = dict(cfg.get("motion", {}))  # пороги из config.json
        self._motion_state = "rest"               # "rest" / "dyn" / "osc" (качание)
        self._rest_timer = 0.0                    # сколько подряд секунд «тихо» (для гистерезиса)
        self._trust = 1.0                         # текущий множитель (плавно к цели)
        self._last_t = None                       # для реального dt между сэмплами

        # --- ПАКЕТ 15, блок А: ZUPT + Хьюбер + детектор КАЧАНИЯ ---
        self._zupt_cfg = dict(cfg.get("zupt", {}))
        self._huber_k = float(cfg.get("huber_k", 2.0))
        self._osc_cfg = dict(cfg.get("osc", {}))
        # качание: скользящее окно gz (сумма/сумма квадратов — O(1) на сэмпл)
        self._osc_win = deque()                   # (t, gz)
        self._osc_sum = 0.0
        self._osc_sum2 = 0.0
        self._osc_until = -1.0                    # состояние держится до этого t
        # ZUPT: сглаженные |ω| и ||f|−g| (EMA ~0.1 с) + накопленный покой
        self._zupt_w = None
        self._zupt_a = None
        self._zupt_timer = 0.0
        self._zupt_active = False
        # сколько строк приходится на один НАСТОЯЩИЙ отсчёт баро (баро ~25 Гц,
        # строки 104/417 Гц повторяют значение) — живой замер для k_baro
        self._baro_rows = 1.0
        self._baro_row_count = 0
        self._baro_prev_val = None
        self.accel_input_enabled = True           # использовать аксель как вход фильтра
        self.motion_enabled = True                # включён ли детектор (False — поведение как раньше)
        # калибровка акселерометра (диагональная) и эталон g из calibration.json
        self._acc_off = np.zeros(3)
        self._acc_scl = np.ones(3)
        self._g_ref = 9.81
        self._gyro_bias = np.zeros(3)             # bias гироскопа (для MEKF)

        # --- ВЕРТИКАЛЬНОЕ УСКОРЕНИЕ из сырого IMU: MEKF-проекция или скаляр ---
        # "mekf" (по умолчанию): a_vert = (R_wb·f_b)_z − g через ориентацию (pc/mekf.py) —
        #   честно при наклонах/вращении; "scalar": прежнее |a_калибр| − g.
        # Детектор Покой/Движение и accel_trust работают ПОВЕРХ в обоих режимах.
        self._va_mode = cfg.get("vertical_accel_mode", "mekf")
        self._mekf = None                         # создаётся лениво на первом IMU-сэмпле
        self._mekf_cfg = load_mekf_config()       # параметры из config.json → mekf
        self._jump_seen = 0                       # журнал скачков курса (Б.1)
        self._sensors_logged = False              # SENSORS уже в логе (З.2)
        self._temp_logged = False                 # TEMP уже в логе (З.3)

        # --- ТЕНЕВОЙ ПРОГОН ВТОРОГО МЕТОДА (Г.1): вариометр ДРУГИМ вертикальным
        # ускорением на тех же входах и с теми же R/Q. С пакета 14 (Б.1) MEKF
        # ОДИН и считается всегда — «второй метод» просто берёт другой выход
        # (mekf-проекция ↔ скаляр), отдельный MEKF тени больше не нужен.
        self._show_second = bool(cfg.get("show_second", False))
        self.filter2 = None               # теневой фильтр высоты (строится с основным)
        self.buf_v2 = deque(maxlen=BUFFER_POINTS)

        # --- ДВА СЛОТА РУЧНОЙ НАСТРОЙКИ R/Q (пакет 14, А.5; вместо «теневого
        # фильтра»): параллельные фильтры на тех же входах + эталонный фильтр
        # «Авто» (паспортные R/Q) для метрики отличия. Работают в режиме «Ручной».
        ms = cfg.get("manual_slots", {})
        self._slots = {
            "s1": dict(ms.get("s1", {"R": R_DEFAULT, "sigma_accel": SIGMA_ACCEL_DEFAULT,
                                     "show": False})),
            "s2": dict(ms.get("s2", {"R": R_DEFAULT, "sigma_accel": SIGMA_ACCEL_DEFAULT,
                                     "show": False})),
        }
        self.filter_s1 = None             # фильтры слотов (строятся с основным)
        self.filter_s2 = None
        self.filter_auto = None           # эталон «Авто» для метрики RMS-отличия
        self.buf_v_s1 = deque(maxlen=BUFFER_POINTS)
        self.buf_v_s2 = deque(maxlen=BUFFER_POINTS)
        self.buf_v_auto = deque(maxlen=BUFFER_POINTS)
        self._slot_metric_next = 0.0      # метрика слотов пересчитывается ~1 раз/с

        # --- АДАПТИВНЫЕ R/Q в режиме «Авто» (Фаза 5; ПЕРЕСМОТР в пакете 15, А.4):
        # R̂ — НЕПРЕРЫВНО по ВТОРЫМ РАЗНОСТЯМ сырой высоты на настоящих
        # 25-Гц отсчётах баро (d = z_i − 2z_{i−1} + z_{i−2}; при белом шуме
        # Var(d) = 6σ²; вклад реального ускорения a·Δt² ≈ 0.005 м — пренебрежим),
        # робастно: σ̂² = (1.4826·MAD)²/6. Работает и в ПОЛЁТЕ (условие «только
        # Покой» убрано); замирает лишь после watchdog. Q̂ — по NIS, как раньше.
        # Оба ×[0.3…3] от паспортных, плавно (R τ~10 с, Q τ~5 с).
        self._adaptive_on = bool(cfg.get("adaptive_rq", True))
        self._adapt_r_mode = "d2"         # "d2" (пакет 15) | "legacy" (пакет 14)
        self._ad_innov = deque()          # legacy: (t, y²) в ПОКОЕ, окно ~10 с
        self._ad_d2 = deque()             # (t, d) вторые разности 25-Гц баро, окно ~10 с
        self._ad_prev_alt = None          # последнее СМЕНИВШЕЕСЯ значение баро
        self._ad_prev_alt2 = None         # предпоследнее
        self._ad_nis = 1.0                # EMA нормированного квадрата инновации
        self._ad_R_mult = 1.0             # эффективный множитель R
        self._ad_Q_mult = 1.0             # эффективный множитель Q
        self._ad_hold_until = 0.0         # замирание после watchdog, до t
        self._ad_next_R = 0.0             # когда пересчитывать MAD (раз в 0.5 с)
        self._ad_log_last = (1.0, 1.0)    # последние залогированные множители (Ж)

        # --- RTS-СГЛАЖИВАТЕЛЬ для файлового режима (блок З.2): обратный проход
        # после пакетного прогона → кривая «Сглаженный (RTS, анализ)»; в реальном
        # времени не участвует (галочка, config show_rts).
        self._show_rts = bool(cfg.get("show_rts", False))

        # --- WATCHDOG фильтра (пакет 13, блок А.4): если фильтр разошёлся с баро
        # (|v_фильтра − v_баро(окно 1 с)| > 1.5 м/с дольше 3 с) — мягкий пересев
        # h и v по барометру, событие в статус-строку и в консоль (лог).
        self._wd_win = deque()                    # (t, h_baro) за последнюю ~1 с
        self._wd_bad = 0.0                        # сколько секунд подряд расходимся
        self._wd_events = []                      # (t, расхождение) — лог срабатываний

        # --- буферы данных (кольцевые, хранят последние BUFFER_POINTS точек) ---
        self.buf_t = deque(maxlen=BUFFER_POINTS)        # время, с
        self.buf_h_raw = deque(maxlen=BUFFER_POINTS)    # высота баро (сырая)
        self.buf_h_filt = deque(maxlen=BUFFER_POINTS)   # высота после фильтра
        self.buf_v_raw = deque(maxlen=BUFFER_POINTS)    # вариометр сырой (производная баро)
        self.buf_v_filt = deque(maxlen=BUFFER_POINTS)   # вариометр после фильтра
        self.buf_h_smooth = deque(maxlen=BUFFER_POINTS) # высота, сглаженная окном N с
        self.buf_v_smooth = deque(maxlen=BUFFER_POINTS) # вариометр, сглаженный (та же серия, что число)

        self._panel_layout = cfg.get("panel_layout")   # активный вид пульта (Е.3)

        self._build_ui()            # собрать интерфейс (включая панель параметров и просмотра)
        self._build_plots()         # собрать графики
        self._sync_filter_panel()   # выставить поля по режиму и создать фильтр
        self._sync_view()           # выставить окно просмотра и шаг сетки
        self._on_source_changed()   # видимость файловых кнопок и галочки RTS (А.6)
        self.refresh_calib_indicator()   # индикатор активной калибровки под компасом
        self._apply_saved_panel_layout()  # сохранённый вид пульта (Е.3)

        # таймер перерисовки графиков (~30 кадров/с)
        self.redraw_timer = QtCore.QTimer(self)
        self.redraw_timer.setInterval(33)
        self.redraw_timer.timeout.connect(self._redraw)
        self.redraw_timer.start()

    # ------------------------------------------------------------------
    # ИНТЕРФЕЙС: панель управления сверху
    # ------------------------------------------------------------------
    def _build_ui(self):
        # центральная область — в QScrollArea: при сжатии окна (до ~1000 px и уже)
        # появляется горизонтальная прокрутка, ничего не обрезается и не ломается
        central = QtWidgets.QWidget()
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(central)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.setCentralWidget(scroll)
        root = QtWidgets.QVBoxLayout(central)

        # ---- строка кнопок и настроек ----
        bar = QtWidgets.QHBoxLayout()

        # выбор источника
        bar.addWidget(QtWidgets.QLabel("Источник:"))
        self.combo_source = QtWidgets.QComboBox()
        self.combo_source.addItems(["Симуляция", "CSV-файл", "Поток (Bluetooth/симулятор)"])
        self.combo_source.currentIndexChanged.connect(self._on_source_changed)
        bar.addWidget(self.combo_source)

        # кнопка выбора CSV-файла (активна только для CSV): ведёт на вкладку
        # «Записи» (список записей мигнёт зелёной рамкой); рядом — маленький
        # «Обзор…» с классическим диалогом для файлов вне data\
        self.files_nav_cb = None            # ставит main.py: перейти на «Записи»
        self.btn_file = QtWidgets.QPushButton("Файл…")
        self.btn_file.setToolTip("Выбрать запись на вкладке «Записи» (список подсветится).\n"
                                 "Для файла вне data\\ — кнопка «Обзор…» рядом.")
        self.btn_file.clicked.connect(self._on_file_button)
        self.btn_file.setEnabled(False)
        bar.addWidget(self.btn_file)
        self.btn_file_browse = QtWidgets.QPushButton("Обзор…")
        self.btn_file_browse.setToolTip("Классический диалог выбора файла (для файлов вне data\\)")
        # Д.2 (пакет 15): фикс-ширина 64 обрезала «Обзор…» — ширина по содержимому
        self.btn_file_browse.clicked.connect(self._choose_file)
        self.btn_file_browse.setEnabled(False)
        bar.addWidget(self.btn_file_browse)

        # URL живого потока (активно только для «Поток»)
        self.edit_stream_url = QtWidgets.QLineEdit("socket://127.0.0.1:5555")
        self.edit_stream_url.setFixedWidth(210)
        self.edit_stream_url.setToolTip(
            "Куда подключаться за живым потоком.\n"
            "Симулятор: socket://127.0.0.1:5555 (запустите python pc\\stream_simulator.py)\n"
            "Для Bluetooth: COM5 (виртуальный COM-порт после сопряжения)")
        self.edit_stream_url.setEnabled(False)
        bar.addWidget(self.edit_stream_url)

        # индикатор приёма + мини-качество: с пакета 15 (Е.3) живут в КАРТОЧКЕ
        # «Качество связи» (создаются здесь, добавляются в карточку ниже)
        self.link_dot = DotIndicator()
        self.link_dot.setToolTip("Приём пакетов потока: зелёный пульс — данные идут,\n"
                                 "красный — нет данных (идёт переподключение)")
        self.lbl_link_q = QtWidgets.QLabel("")
        self.lbl_link_q.setFixedWidth(120)
        self.lbl_link_q.setToolTip("Качество связи: насколько фактический темп и потери\n"
                                   "дотягивают до номинала отправителя (вкладка «Связь/Задержка»)")

        bar.addSpacing(20)

        # Старт / Стоп / Сброс
        self.btn_start = QtWidgets.QPushButton("▶ Старт")
        self.btn_start.clicked.connect(self.start)
        bar.addWidget(self.btn_start)

        self.btn_stop = QtWidgets.QPushButton("⏸ Стоп")
        self.btn_stop.clicked.connect(self.stop)
        self.btn_stop.setEnabled(False)
        bar.addWidget(self.btn_stop)

        self.btn_reset = QtWidgets.QPushButton("⟲ Сброс")
        self.btn_reset.clicked.connect(self.reset)
        bar.addWidget(self.btn_reset)

        self.btn_rerun = QtWidgets.QPushButton("↻ Запустить повторно")
        self.btn_rerun.setToolTip("Перезапустить текущий источник с начала (без выбора файла заново)")
        self.btn_rerun.clicked.connect(self.rerun)
        bar.addWidget(self.btn_rerun)

        bar.addStretch(1)
        root.addLayout(bar)
        # галочки видимости и экспорт живут на строке «Просмотр» (вторая строка) —
        # так верхняя панель влезает в окно шириной ~1000 px

        # ---- КАРТОЧКИ ШАПКИ (пакет 15, Е.3) ----
        # Прежняя жёсткая строка цифр разложена на КАРТОЧКИ (Высота, Абс.
        # высота, Вариометр, Сглаж., 2-й метод, Компас, Курсор, Качество
        # связи, Громкость): в режиме «Компоновка» их можно таскать по сетке
        # 8 px и переименовывать; вид сохраняется в data\layouts\*.json.
        # Все виджеты и их имена — ПРЕЖНИЕ (логика обновления не менялась).
        self.cards_panel = CardsPanel()

        def card(key, title, build):
            w = QtWidgets.QWidget()
            build(w)
            return self.cards_panel.add_card(key, title, w)

        # --- карточка «Высота» (с кнопкой «Ноль высоты») ---
        def _b_alt(w):
            lay = QtWidgets.QHBoxLayout(w)
            lay.setContentsMargins(0, 0, 0, 0)
            cap_alt = QtWidgets.QLabel("Высота:")
            cap_alt.setStyleSheet("font-size: 22px; font-weight: bold; color: #1f5fd0;")
            lay.addWidget(cap_alt)
            self.lbl_alt = make_value_label("+99999.9 м", "#1f5fd0", px=22)
            lay.addWidget(self.lbl_alt)
            zbox = QtWidgets.QVBoxLayout()
            zbox.setSpacing(0)
            self.btn_zero_alt = QtWidgets.QPushButton("Ноль высоты")
            self.btn_zero_alt.setToolTip(
                "Запомнить текущую высоту как 0 и показывать ОТНОСИТЕЛЬНУЮ высоту\n"
                "(большое число, график, строка «Курсор»). Повторное нажатие — вернуть\n"
                "абсолютную. Фильтр/вариометр/экспорт не меняются.")
            self.btn_zero_alt.clicked.connect(self._toggle_zero_alt)
            zbox.addWidget(self.btn_zero_alt)
            self.lbl_zero_note = QtWidgets.QLabel("")
            self.lbl_zero_note.setStyleSheet("font-size: 10px; color: #7a7f87;")
            zbox.addWidget(self.lbl_zero_note)
            lay.addLayout(zbox)
        card("alt", "Высота", _b_alt)

        # --- карточка «Абс. высота» (пакет 15, Е.2): всегda НАСТОЯЩАЯ
        # абсолютная высота, крупно, чернильным цветом темы ---
        def _b_alt_abs(w):
            lay = QtWidgets.QHBoxLayout(w)
            lay.setContentsMargins(0, 0, 0, 0)
            self.cap_alt_abs = QtWidgets.QLabel("Абс. высота:")
            self.cap_alt_abs.setStyleSheet(
                "font-size: 22px; font-weight: bold; color: #111111;")
            lay.addWidget(self.cap_alt_abs)
            self.lbl_alt_abs = make_value_label("+99999.9 м", "#111111", px=22)
            self.lbl_alt_abs.setToolTip(
                "Реальная АБСОЛЮТНАЯ высота фильтра (QNH 1013.25) — не зависит\n"
                "от «Ноля высоты». У относительной высоты остаётся мелкая\n"
                "подпись «0 = X м абс.» под кнопкой.")
            lay.addWidget(self.lbl_alt_abs)
        card("alt_abs", "Абс. высота", _b_alt_abs)

        # --- карточка «Вариометр» ---
        def _b_vario(w):
            lay = QtWidgets.QHBoxLayout(w)
            lay.setContentsMargins(0, 0, 0, 0)
            cap_vario = QtWidgets.QLabel("Вариометр:")
            cap_vario.setStyleSheet("font-size: 22px; font-weight: bold; color: #c0392b;")
            lay.addWidget(cap_vario)
            # цвет ЗНАЧЕНИЯ меняется по знаку (>+0.05 зелёный, <−0.05 красный,
            # около нуля — цвет темы); ширина/шрифт фиксированы — не прыгает
            self.lbl_vario = make_value_label("+999.99 м/с", self._neutral_fg, px=22)
            lay.addWidget(self.lbl_vario)
        card("vario", "Вариометр", _b_vario)

        # --- карточка «Сглаж.» (гауссово сглаживание; графики не меняет) ---
        def _b_smooth(w):
            lay = QtWidgets.QHBoxLayout(w)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.addWidget(QtWidgets.QLabel("Сглаж.(Гаусс), с:"))
            self.spin_smooth = StepSpinBox()
            self.spin_smooth.setDecimals(1)
            self.spin_smooth.setRange(0.0, 60.0)
            self.spin_smooth.setSingleStep(0.5)
            self.spin_smooth.setSpecialValueText("—")   # 0 показывается как «—»
            self.spin_smooth.setToolTip(
                "Каузальное (только прошлое) гауссово среднее вариометра за последние N секунд.\n"
                "Показывается ТРЕТЬИМ числом рядом — на графики НЕ влияет. 0 / «—» = выключено.")
            self.spin_smooth.setValue(self._init_smooth)   # из config (до сигнала)
            self.spin_smooth.valueChanged.connect(self._on_smooth_changed)
            lay.addWidget(self.spin_smooth)
            lay.addSpacing(12)
            self.lbl_smooth_cap = QtWidgets.QLabel("Сглаж.:")
            self.lbl_smooth_cap.setStyleSheet(
                "font-size: 22px; font-weight: bold; color: #888;")
            lay.addWidget(self.lbl_smooth_cap)
            self.lbl_vario_smooth = make_value_label("+999.99 м/с", "#888", px=22)
            lay.addWidget(self.lbl_vario_smooth)
        card("smooth", "Сглаж.", _b_smooth)

        # --- карточка «Компас» ---
        def _b_compass(w):
            lay = QtWidgets.QHBoxLayout(w)
            lay.setContentsMargins(0, 0, 0, 0)
            self.compass = CompassWidget(110)
            lay.addWidget(self.compass)
            cbox = QtWidgets.QVBoxLayout()
            hrow = QtWidgets.QHBoxLayout()
            cap_head = QtWidgets.QLabel("Курс:")
            cap_head.setStyleSheet("font-size: 20px; font-weight: bold; color: #2c7a2c;")
            hrow.addWidget(cap_head)
            self.lbl_heading = make_value_label("359° СЗ", "#2c7a2c", px=20)
            hrow.addWidget(self.lbl_heading)
            hrow.addStretch(1)
            cbox.addLayout(hrow)
            self.lbl_compass_note = QtWidgets.QLabel(
                f"курс: AHRS (MEKF+магнитометр)\nстрелка сглажена {self._compass_tau:g} с")
            self.lbl_compass_note.setStyleSheet("font-size: 11px; color: #7a7f87;")
            cbox.addWidget(self.lbl_compass_note)
            self.lbl_calib_status = QtWidgets.QLabel("")
            self.lbl_calib_status.setStyleSheet("font-size: 11px; color: #7a7f87;")
            self.lbl_calib_status.setToolTip(
                "Какая калибровка прибора сейчас применяется к сырому потоку/файлам\n"
                "(из pc/calibration.json). Обновляется при «Сохранить калибровку прибора»\n"
                "и «Применить» из архива на вкладке «Записи».")
            cbox.addWidget(self.lbl_calib_status)
            cbox.addStretch(1)
            lay.addLayout(cbox)
        card("compass", "Компас", _b_compass)

        # --- карточка «2-й метод» (пакет 14, А.1) ---
        def _b_second(w):
            lay = QtWidgets.QVBoxLayout(w)
            lay.setContentsMargins(0, 0, 0, 0)
            self.frame2 = QtWidgets.QWidget()
            f2v = QtWidgets.QVBoxLayout(self.frame2)
            f2v.setContentsMargins(0, 2, 0, 0)
            f2v.setSpacing(2)
            f2 = QtWidgets.QHBoxLayout()
            f2v.addLayout(f2)
            self.lbl2_cap = QtWidgets.QLabel("2-й метод (|a|−g):")
            self.lbl2_cap.setStyleSheet(
                "font-size: 12px; font-weight: bold; color: #8e6bb5;")
            self.lbl2_cap.setToolTip(
                "Теневой прогон ВТОРОГО метода вертикального ускорения (Г.1):\n"
                "тот же фильтр и входы, но a_vert другим способом (MEKF ↔ |a|−g).\n"
                "Для сравнения — на фильтр, большие числа и звук не влияет.")
            f2.addWidget(self.lbl2_cap)
            f2.addSpacing(10)
            cap2v = QtWidgets.QLabel("Вариометр:")
            cap2v.setStyleSheet("font-size: 12px; color: #8e6bb5;")
            f2.addWidget(cap2v)
            self.lbl2_vario = make_value_label("+999.99 м/с", self._neutral_fg, px=22)
            f2.addWidget(self.lbl2_vario)
            f2.addSpacing(25)
            cap2s = QtWidgets.QLabel("Сглаж.:")
            cap2s.setStyleSheet("font-size: 12px; color: #8e6bb5;")
            f2.addWidget(cap2s)
            self.lbl2_smooth = make_value_label("+999.99 м/с", self._neutral_fg, px=22)
            f2.addWidget(self.lbl2_smooth)
            f2.addSpacing(15)
            self.chk_second = QtWidgets.QCheckBox("показать 2-й метод")
            self.chk_second.setChecked(self._show_second)
            self.chk_second.setToolTip(
                "Пунктирная кривая второго метода на графике вариометра")
            self.chk_second.stateChanged.connect(self._on_second_toggled)
            f2.addWidget(self.chk_second)
            f2.addStretch(1)
            self.frame2.setVisible(False)   # появится, когда пойдут данные IMU
            lay.addWidget(self.frame2)
        card("second", "2-й метод", _b_second)

        # --- карточка «Курсор» (инспектор по клику) ---
        def _b_cursor(w):
            crow = QtWidgets.QHBoxLayout(w)
            crow.setContentsMargins(0, 0, 0, 0)
            cap_cur = QtWidgets.QLabel("Курсор:")
            cap_cur.setStyleSheet("font-weight: bold; color: #2cae2c;")
            cap_cur.setToolTip(
                "Клик ЛЕВОЙ кнопкой по графику ставит курсор времени.\n"
                "Значения — фильтрованные, в ближайшей точке данных. Esc/✕ — убрать.")
            crow.addWidget(cap_cur)
            crow.addWidget(QtWidgets.QLabel("t ="))
            self.lbl_cur_t = make_value_label("99999.99 с", "#2cae2c")
            crow.addWidget(self.lbl_cur_t)
            crow.addWidget(QtWidgets.QLabel("Высота:"))
            self.lbl_cur_alt = make_value_label("+99999.9 м", "#2cae2c")
            crow.addWidget(self.lbl_cur_alt)
            crow.addWidget(QtWidgets.QLabel("Вариометр:"))
            self.lbl_cur_vario = make_value_label("+999.99 м/с", "#2cae2c")
            crow.addWidget(self.lbl_cur_vario)
            crow.addWidget(QtWidgets.QLabel("Сглаж.:"))
            self.lbl_cur_smooth = make_value_label("+999.99 м/с", "#2cae2c")
            crow.addWidget(self.lbl_cur_smooth)
            self.lbl_cur_slots = QtWidgets.QLabel("")
            f_cs = QtGui.QFont("Consolas")
            f_cs.setStyleHint(QtGui.QFont.Monospace)
            self.lbl_cur_slots.setFont(f_cs)
            crow.addWidget(self.lbl_cur_slots)
            self.btn_cur_clear = QtWidgets.QPushButton("✕")
            self.btn_cur_clear.setFixedSize(22, 22)
            self.btn_cur_clear.setToolTip("Убрать курсор и линии (или клавиша Esc)")
            self.btn_cur_clear.clicked.connect(self._clear_cursor)
            crow.addWidget(self.btn_cur_clear)
        card("cursor", "Курсор", _b_cursor)

        # --- карточка «Качество связи» (индикатор создан в верхней панели) ---
        def _b_link(w):
            lay = QtWidgets.QHBoxLayout(w)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.addWidget(self.link_dot)
            lay.addWidget(self.lbl_link_q)
        card("link", "Качество связи", _b_link)

        # --- карточка «Громкость» (сюда main.py вставит компактный дубль) ---
        def _b_volume(w):
            self._sound_slot = QtWidgets.QHBoxLayout(w)
            self._sound_slot.setContentsMargins(0, 0, 0, 0)
        card("volume", "Громкость", _b_volume)

        root.addWidget(self.cards_panel)
        self.cards_panel.relayout()

        # Esc убирает курсор (работает по всему окну; безвредно, если курсора нет)
        QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Escape), self,
                        activated=self._clear_cursor)

        # сюда (root) дальше добавятся панель параметров и графики
        self._root_layout = root

        # ---- панель «Параметры фильтра» ----
        self._build_filter_panel()

        # ---- строка «Просмотр»: окно и шаг сетки ----
        self._build_view_bar()

        # строка состояния снизу
        self.statusBar().showMessage("Готово. Выберите источник и нажмите «Старт».")

    # ------------------------------------------------------------------
    # ГРАФИКИ
    # ------------------------------------------------------------------
    def _build_plots(self):
        # график высоты
        self.plot_alt = pg.PlotWidget()
        # ВАЖНО: единицы пишем прямо в подписи и НЕ передаём units=, иначе
        # pyqtgraph сам подставляет SI-приставку и на больших значениях рисует
        # «км», деля подписи на 1000 (из-за этого высота выглядела «нулевой»).
        self.plot_alt.setLabel("left", "Высота, м")
        self.plot_alt.setLabel("bottom", "Время, с")
        self.plot_alt.getAxis("left").enableAutoSIPrefix(False)    # никаких авто-«км»
        self.plot_alt.getAxis("bottom").enableAutoSIPrefix(False)
        self.plot_alt.showGrid(x=True, y=True, alpha=0.3)
        self.curve_h_raw = self.plot_alt.plot(pen=pg.mkPen("#9aa0a6", width=1))
        self.curve_h_filt = self.plot_alt.plot(pen=pg.mkPen("#1f5fd0", width=2))
        self.curve_h_smooth = self.plot_alt.plot(pen=pg.mkPen("#1e9e46", width=2))
        self.curve_h_rts = self.plot_alt.plot(
            pen=pg.mkPen("#e67e22", width=1.5, style=QtCore.Qt.DashDotLine))

        # график вариометра
        self.plot_vario = pg.PlotWidget()
        self.plot_vario.setLabel("left", "Вариометр, м/с")
        self.plot_vario.setLabel("bottom", "Время, с")
        self.plot_vario.getAxis("left").enableAutoSIPrefix(False)
        self.plot_vario.getAxis("bottom").enableAutoSIPrefix(False)
        self.plot_vario.showGrid(x=True, y=True, alpha=0.3)
        # горизонтальная линия нуля (граница «вверх/вниз»)
        self.plot_vario.addLine(y=0, pen=pg.mkPen("#444444", width=1, style=QtCore.Qt.DashLine))
        self.curve_v_raw = self.plot_vario.plot(pen=pg.mkPen("#9aa0a6", width=1))
        self.curve_v_filt = self.plot_vario.plot(pen=pg.mkPen("#c0392b", width=2))
        self.curve_v_smooth = self.plot_vario.plot(pen=pg.mkPen("#1e9e46", width=2))
        # пунктир ВТОРОГО метода верт. ускорения (Г.1; галочка «показать 2-й метод»)
        self.curve_v2 = self.plot_vario.plot(
            pen=pg.mkPen("#8e6bb5", width=1.5, style=QtCore.Qt.DashLine))
        # RTS-сглаживатель (З.2, только файловый режим)
        self.curve_v_rts = self.plot_vario.plot(
            pen=pg.mkPen("#e67e22", width=1.5, style=QtCore.Qt.DashDotLine))
        # кривые СЛОТОВ ручной настройки (пакет 14, А.5)
        self.curve_v_s1 = self.plot_vario.plot(
            pen=pg.mkPen(SLOT_COLORS["s1"], width=1.5))
        self.curve_v_s2 = self.plot_vario.plot(
            pen=pg.mkPen(SLOT_COLORS["s2"], width=1.5))
        # ПРОИЗВОДИТЕЛЬНОСТЬ (пакет 14, Д.3): на длинных сериях (файл 417 Гц —
        # сотни тысяч точек) рисуем прореженно (peak сохраняет выбросы) и только
        # видимую часть; на данные не влияет — только на отрисовку
        for c in (self.curve_h_raw, self.curve_h_filt, self.curve_h_smooth,
                  self.curve_h_rts, self.curve_v_raw, self.curve_v_filt,
                  self.curve_v_smooth, self.curve_v2, self.curve_v_rts,
                  self.curve_v_s1, self.curve_v_s2):
            c.setDownsampling(auto=True, method="peak")
            c.setClipToView(True)

        # ЛЕГЕНДЫ — компактной строкой НАД каждым графиком (цветные метки),
        # чтобы не закрывать кривые в области рисования; там же — переключатель
        # масштаба оси Y («Авто (по окну)» / «Ручной» + поля min/max)
        hdr_alt, self.ycombo_alt, self.yspin_min_alt, self.yspin_max_alt = \
            self._plot_header_row(
                [("#9aa0a6", "Баро (сырое)"), ("#1f5fd0", "Фильтр Калмана"),
                 ("#1e9e46", "Сглаженное (Гаусс N с)"),
                 ("#e67e22", "RTS (анализ)")], "alt", decimals=1)
        hdr_var, self.ycombo_vario, self.yspin_min_vario, self.yspin_max_vario = \
            self._plot_header_row(
                [("#9aa0a6", "Сырой (производная баро)"), ("#c0392b", "Фильтр Калмана"),
                 ("#1e9e46", "Сглаженное (Гаусс N с)"),
                 ("#8e6bb5", "2-й метод (пунктир)"),
                 ("#e67e22", "RTS (анализ)"),
                 (SLOT_COLORS["s1"], "Ручной 1"),
                 (SLOT_COLORS["s2"], "Ручной 2")], "vario", decimals=2)

        # оси X НЕзависимы: у каждого графика своё окно просмотра и свой шаг сетки
        # (поэтому setXLink НЕ ставим — графики прокручиваются раздельно)

        # --- инспектор по клику: одна и та же координата t_c на ОБОИХ графиках ---
        # (оси не связываем; линия просто существует в t_c и видна там, где окно её содержит)
        cur_pen = pg.mkPen("#2cae2c", width=1.5)
        self.cursor_line_alt = pg.InfiniteLine(angle=90, movable=False, pen=cur_pen)
        self.cursor_line_vario = pg.InfiniteLine(angle=90, movable=False, pen=cur_pen)
        for line, plot in ((self.cursor_line_alt, self.plot_alt),
                           (self.cursor_line_vario, self.plot_vario)):
            line.setVisible(False)
            plot.addItem(line)
        # пассивный слушатель кликов (sigMouseClicked): штатные жесты pyqtgraph —
        # панораму (drag), зум (колесо), меню (правая кнопка) — НЕ перехватываем
        self.plot_alt.scene().sigMouseClicked.connect(
            lambda ev: self._on_plot_clicked(ev, self.plot_alt))
        self.plot_vario.scene().sigMouseClicked.connect(
            lambda ev: self._on_plot_clicked(ev, self.plot_vario))
        # ручная панорама во время проигрывания файла отключает слежение за временем
        self.plot_alt.getViewBox().sigRangeChangedManually.connect(self._user_panned)
        self.plot_vario.getViewBox().sigRangeChangedManually.connect(self._user_panned)
        # СТРАХОВКА паритета мыши (баг «на вариометре не работает зум при
        # проигрывании»): ловим колесо/драг прямо на вьюпорте КАЖДОГО графика —
        # слежение отключается и Y уходит в «Ручной» одинаково на обоих графиках,
        # независимо от того, какие сигналы шлёт установленная версия pyqtgraph
        self._guard_alt = PlotInteractGuard(self, "alt", self.plot_alt.viewport())
        self.plot_alt.viewport().installEventFilter(self._guard_alt)
        self._guard_vario = PlotInteractGuard(self, "vario", self.plot_vario.viewport())
        self.plot_vario.viewport().installEventFilter(self._guard_vario)
        # авто-загущение сетки времени: при широком окне подписи не слипаются
        self.plot_alt.getViewBox().sigXRangeChanged.connect(
            lambda *a: self._update_time_ticks("alt"))
        self.plot_vario.getViewBox().sigXRangeChanged.connect(
            lambda *a: self._update_time_ticks("vario"))
        # А.4 (пакет 14): кнопка «A» pyqtgraph и «View All» из меню включают
        # внутренний авто-масштаб вьюбокса — ловим смену его состояния и
        # переводим НАШ комбо Y в «Авто (по окну)» (единый источник истины)
        self.plot_alt.getViewBox().sigStateChanged.connect(
            lambda *_: self._on_vb_autorange("alt"))
        self.plot_vario.getViewBox().sigStateChanged.connect(
            lambda *_: self._on_vb_autorange("vario"))

        # два графика друг под другом, над каждым — его строка легенды/масштаба
        self._root_layout.addLayout(hdr_alt)
        self._root_layout.addWidget(self.plot_alt, stretch=1)
        self._root_layout.addLayout(hdr_var)
        self._root_layout.addWidget(self.plot_vario, stretch=1)

        self._apply_visibility()

    def _plot_header_row(self, legend_items, key: str, decimals: int):
        """Строка над графиком: слева легенда (цветные метки), справа — режим оси Y.
        Возвращает (layout, combo_режима, spin_min, spin_max)."""
        row = QtWidgets.QHBoxLayout()
        html_txt = " &nbsp; ".join(
            f'<span style="color:{c}; font-weight:bold;">▬</span> {name}'
            for c, name in legend_items)
        lbl = QtWidgets.QLabel(html_txt)
        lbl.setTextFormat(QtCore.Qt.RichText)
        row.addWidget(lbl)
        row.addStretch(1)
        row.addWidget(QtWidgets.QLabel("Y:"))
        combo = QtWidgets.QComboBox()
        combo.addItems(["Авто (по окну)", "Ручной"])
        combo.setToolTip(
            "Масштаб оси Y этого графика.\n"
            "Авто (по окну): раз в длину окна пересчитывается min/max по данным\n"
            "в видимом окне (+10% запаса, с гистерезисом — не дёргается каждый кадр).\n"
            "Ручной: пределы из полей min/max. Зум/панорама мышью тоже переводят в «Ручной».")
        combo.currentIndexChanged.connect(lambda *_: self._on_y_mode_changed(key))
        row.addWidget(combo)
        s_min = StepSpinBox()
        s_max = StepSpinBox()
        for s, cap in ((s_min, "min"), (s_max, "max")):
            s.setDecimals(decimals)
            s.setRange(-100000.0, 100000.0)
            s.setSingleStep(0.5 if decimals >= 2 else 5.0)
            s.setToolTip(cap + " оси Y (в режиме «Ручной»)")
            s.setPrefix(cap + " ")
            s.setFixedWidth(92)
            s.set_fit_width(False)   # живёт в строке зума: ширина стабильная
            # без keyboardTracking: значение применяется по Enter/уходу из поля,
            # а не на каждую нажатую цифру (иначе «1» из «118» уже дёргает ось)
            s.setKeyboardTracking(False)
            # передаём, КАКОЕ поле правили: при min ≥ max второе поле подтянется
            # (раньше правка молча игнорировалась — «min не применяется», блок Г.2)
            s.valueChanged.connect(
                lambda *_, which=cap: self._on_y_manual_changed(key, which))
            row.addWidget(s)
        return row, combo, s_min, s_max

    # ------------------------------------------------------------------
    # ИНСПЕКТОР ПО КЛИКУ: курсор времени t_c на обоих графиках + строка значений
    # ------------------------------------------------------------------
    def _series_t(self):
        """Активная ось времени: полные серии файла или кольцевой буфер (live)."""
        if self._file is not None:
            return self._file["t"]
        return np.fromiter(self.buf_t, dtype=float)

    def _on_plot_clicked(self, ev, plot):
        """Левый клик (без модификаторов) по графику → поставить курсор на ближайшую
        точку данных. Двойной клик — вернуть слежение за временем (файловый плеер).
        Правая кнопка/drag/колесо не трогаются (стандарт pyqtgraph)."""
        if ev.button() != QtCore.Qt.LeftButton or ev.modifiers() != QtCore.Qt.NoModifier:
            return
        if ev.double():
            self._follow = True         # двойной клик = снова следить за временем
            return
        vb = plot.getViewBox()
        if not vb.sceneBoundingRect().contains(ev.scenePos()):
            return                      # клик мимо поля данных (по осям/подписям)
        t = self._series_t()
        if len(t) == 0:
            return                      # данных ещё нет — ставить не на что
        t_click = float(vb.mapSceneToView(ev.scenePos()).x())
        self._set_cursor(t_click, t)

    def _set_cursor(self, t_click: float, t=None):
        """Прищёлкнуть t_click к ближайшей точке данных и показать курсор."""
        if t is None:
            t = self._series_t()
        idx = int(np.searchsorted(t, t_click))
        if idx >= len(t):
            idx = len(t) - 1
        elif idx > 0 and abs(t[idx - 1] - t_click) <= abs(t[idx] - t_click):
            idx -= 1
        self._cursor_t = float(t[idx])
        self.cursor_line_alt.setPos(self._cursor_t)
        self.cursor_line_vario.setPos(self._cursor_t)
        self.cursor_line_alt.setVisible(True)
        self.cursor_line_vario.setVisible(True)
        self._update_cursor_row(idx)

    def _update_cursor_row(self, idx: int):
        """Заполнить строку «Курсор» значениями ближайшей точки (фильтрованные
        данные; высота — с учётом «Ноля высоты»). Плюс значения слотов ручной
        настройки (А.5), если их галочки включены."""
        ref = self._h_ref or 0.0
        self.lbl_cur_t.setText(f"{self._cursor_t:8.2f} с")
        s1 = s2 = None
        if self._file is not None:
            f = self._file
            self.lbl_cur_alt.setText(f"{f['h_filt'][idx] - ref:+7.1f} м")
            self.lbl_cur_vario.setText(f"{f['v_filt'][idx]:+6.2f} м/с")
            sm = None if f.get("v_smooth") is None else float(f["v_smooth"][idx])
            if f.get("v_s1") is not None and idx < len(f["v_s1"]):
                s1, s2 = float(f["v_s1"][idx]), float(f["v_s2"][idx])
        else:
            self.lbl_cur_alt.setText(f"{self.buf_h_filt[idx] - ref:+7.1f} м")
            self.lbl_cur_vario.setText(f"{self.buf_v_filt[idx]:+6.2f} м/с")
            sm = self._vario_smooth_value(end_idx=idx)
            if idx < len(self.buf_v_s1):
                s1 = self.buf_v_s1[idx]
                s2 = self.buf_v_s2[idx]
        self.lbl_cur_smooth.setText("—" if sm is None else f"{sm:+6.2f} м/с")
        parts = []
        if self.mode == "manual":
            if self._slots["s1"]["show"] and s1 is not None and np.isfinite(s1):
                parts.append(f'<span style="color:{SLOT_COLORS["s1"]};">'
                             f'Р1: {s1:+6.2f}</span>')
            if self._slots["s2"]["show"] and s2 is not None and np.isfinite(s2):
                parts.append(f'<span style="color:{SLOT_COLORS["s2"]};">'
                             f'Р2: {s2:+6.2f}</span>')
        self.lbl_cur_slots.setText(" | ".join(parts))

    def _refresh_cursor(self):
        """Пересчитать строку «Курсор» после ЛЮБОГО пересчёта серий (А.3):
        смена N сглаживания, перепрогон файла, «Ноль высоты» и т.п. Если точка
        курсора пережила пересчёт — прищёлкиваем заново к ближайшей."""
        if self._cursor_t is None:
            return
        t = self._series_t()
        if len(t) == 0 or self._cursor_t < t[0] - 1e-9 or self._cursor_t > t[-1] + 1e-9:
            self._clear_cursor()
            return
        self._set_cursor(self._cursor_t, t)

    def _clear_cursor(self):
        """Убрать курсор: спрятать линии, строка «Курсор» → прочерки."""
        self._cursor_t = None
        self.cursor_line_alt.setVisible(False)
        self.cursor_line_vario.setVisible(False)
        for lbl in (self.lbl_cur_t, self.lbl_cur_alt,
                    self.lbl_cur_vario, self.lbl_cur_smooth):
            lbl.setText("—")
        self.lbl_cur_slots.setText("")

    def _sign_color(self, v) -> str:
        """Цвет значения вариометра по знаку: > +0.05 зелёный (набор),
        < −0.05 красный (снижение), около нуля — нейтральный цвет темы."""
        try:
            v = float(v)
        except (TypeError, ValueError):
            return self._neutral_fg
        if not np.isfinite(v):
            return self._neutral_fg
        if v > 0.05:
            return "#1e9e46"
        if v < -0.05:
            return "#c0392b"
        return self._neutral_fg

    def apply_theme(self, pal: dict):
        """
        Перекрасить графики под тему (вызывается из главного окна main.py).
        pal — словарь цветов: plot_bg (фон), plot_fg (оси/подписи/легенда).
        """
        bg = pal["plot_bg"]
        pen = pg.mkPen(pal["plot_fg"])
        col = pg.mkColor(pal["plot_fg"])
        for plot in (self.plot_alt, self.plot_vario):
            plot.setBackground(bg)
            for name in ("left", "bottom"):
                ax = plot.getAxis(name)
                ax.setTextPen(pen)   # цвет чисел на осях
                ax.setPen(pen)       # цвет линий осей и сетки
                if ax.label is not None:
                    ax.label.setDefaultTextColor(col)  # цвет подписи оси
        # нейтральный цвет живых чисел = цвет текста темы; перекрасить сразу
        self._neutral_fg = pal.get("plot_fg", "#d6dbe1")
        ink = "#111111" if pal.get("plot_bg") == "w" else "#f2f2f2"   # Е.2
        self.cap_alt_abs.setStyleSheet(
            f"font-size: 22px; font-weight: bold; color: {ink};")
        self.lbl_alt_abs.setStyleSheet(f"color: {ink};")
        if self._file is not None:
            self._file_show_numbers()
        elif self.buf_v_filt:
            self.lbl_vario.setStyleSheet(f"color: {self._sign_color(self.buf_v_filt[-1])};")
            self._update_smooth_label()
        else:
            self.lbl_vario.setStyleSheet(f"color: {self._neutral_fg};")
            self.lbl_vario_smooth.setStyleSheet(f"color: {self._neutral_fg};")

    # ------------------------------------------------------------------
    # ПАНЕЛЬ «ПАРАМЕТРЫ ФИЛЬТРА»
    # ------------------------------------------------------------------
    def _build_filter_panel(self):
        """Собрать панель с режимом Авто/Ручной, полями R, Q, калибровки и числом смещения."""
        box = QtWidgets.QGroupBox("Параметры фильтра")
        vbox = QtWidgets.QVBoxLayout(box)
        lay = QtWidgets.QHBoxLayout()      # строка 1: режим и параметры
        lay2 = QtWidgets.QHBoxLayout()     # строка 2: смещение акселя + детектор
        vbox.addLayout(lay)
        vbox.addLayout(lay2)

        # --- переключатель Авто (адаптивный) / Ручной ---
        self.radio_auto = QtWidgets.QRadioButton(
            "Авто (адаптивный)" if self._adaptive_on else "Авто")
        self.radio_manual = QtWidgets.QRadioButton("Ручной")
        self.radio_auto.setToolTip(
            "Параметры по умолчанию + АДАПТАЦИЯ (Фаза 5): R подстраивается по\n"
            "инновациям баро в Покое (робастно, окно ~10 с), Q — по NIS (плавно\n"
            "~5 с), оба в пределах ×[0.3…3] от паспортных. Отключить адаптацию:\n"
            "config.json → adaptive_rq: false (останется прежний Авто).")
        self.radio_manual.setToolTip("Задавать параметры вручную и применять по кнопке")
        lay.addWidget(self.radio_auto)
        lay.addWidget(self.radio_manual)
        # живые ЭФФЕКТИВНЫЕ R и Q адаптации + состояние (моноширинно; «—», когда не Авто)
        self.lbl_adaptive = make_value_label(
            "R̂ 0.0675 м²  Q̂ 0.900 м/с² · заморожен (Движение)", "#2c7a2c")
        self.lbl_adaptive.setToolTip(
            "Текущие эффективные R и Q адаптивного авто-режима и его состояние:\n"
            "активен / заморожен (Движение — R не обновляется; watchdog — 5 с\n"
            "паузы после пересева; прогрев — первые секунды после старта).")
        lay.addWidget(self.lbl_adaptive)
        lay.addSpacing(15)

        # --- поле R (шум барометра, дисперсия σ², м²) ---
        lay.addWidget(QtWidgets.QLabel("R (шум баро, σ²):"))
        self.spin_R = StepSpinBox()
        self.spin_R.setDecimals(4)
        self.spin_R.setRange(0.0001, 100.0)
        self.spin_R.setSingleStep(0.005)
        self.spin_R.setSuffix(" м²")
        self.spin_R.setToolTip("Дисперсия шума барометра. Больше R → сильнее сглаживание (но больше задержка).")
        lay.addWidget(self.spin_R)

        # --- поле Q через sigma_accel (СКО ускорения, м/с²) ---
        lay.addWidget(QtWidgets.QLabel("Q (sigma_accel):"))
        self.spin_Q = StepSpinBox()
        self.spin_Q.setDecimals(3)
        self.spin_Q.setRange(0.001, 10.0)
        self.spin_Q.setSingleStep(0.05)
        self.spin_Q.setSuffix(" м/с²")
        self.spin_Q.setToolTip("СКО ускорения (задаёт шум процесса Q). Больше Q → быстрее реакция (но больше шум).")
        lay.addWidget(self.spin_Q)

        # --- поле «Установка нуля акселер.» (старт-усреднение; сейчас только хранится) ---
        lay.addWidget(QtWidgets.QLabel("Установка нуля акселер.:"))
        self.spin_calib = StepSpinBox()
        self.spin_calib.setDecimals(1)
        self.spin_calib.setRange(0.1, 120.0)
        self.spin_calib.setSingleStep(0.5)
        self.spin_calib.setSuffix(" с")
        self.spin_calib.setToolTip(
            "Время в начале для оценки нуля акселерометра.\n"
            "Сам ноль фильтр Калмана оценивает НЕПРЕРЫВНО (см. «Смещение акселер.»);\n"
            "никакого Гауссова сглаживания НЕТ — сглаживает фильтр через R/Q.\n"
            "Значение пока сохраняется (старт-усреднение, Фаза 1).")
        lay.addWidget(self.spin_calib)

        # --- кнопка применить ---
        self.btn_apply = QtWidgets.QPushButton("Применить")
        self.btn_apply.setToolTip("Пересоздать фильтр с этими параметрами")
        self.btn_apply.clicked.connect(self._apply_manual_params)
        lay.addWidget(self.btn_apply)

        lay.addStretch(1)

        # --- строка 2: смещение акселерометра + индикатор детектора ---
        cap_bias = QtWidgets.QLabel("Смещение акселерометра:")
        cap_bias.setStyleSheet("font-weight: bold; color: #2c7a2c;")
        cap_bias.setToolTip("Оценка фильтром смещения нуля акселерометра (переменная b).")
        lay2.addWidget(cap_bias)
        self.lbl_bias = make_value_label("+99.999 м/с²", "#2c7a2c")
        self.lbl_bias.setToolTip("Оценка фильтром смещения нуля акселерометра (переменная b).")
        lay2.addWidget(self.lbl_bias)
        lay2.addSpacing(20)

        # индикатор детектора: Покой (зелёный) / Движение (оранжевый)
        self.lbl_motion = QtWidgets.QLabel("Покой")
        self.lbl_motion.setStyleSheet("font-weight: bold; color: #2c7a2c;")
        self.lbl_motion.setToolTip(
            "Детектор по гироскопу и |a|−g (пороги в config.json → motion).\n"
            "В движении фильтр меньше верит акселерометру (множитель k_dyn на\n"
            "sigma_accel) и НЕ обновляет его ноль — вариометр не врёт при вращении.")
        lay2.addWidget(self.lbl_motion)
        lay2.addSpacing(25)

        # --- источник вертикального ускорения: MEKF (ориентация) / скалярное ---
        # ВАЖНО: своя QButtonGroup — иначе эти радио слиплись бы в одну
        # взаимоисключающую группу с парой Авто/Ручной (общий родитель)
        lay2.addWidget(QtWidgets.QLabel("Верт. ускорение:"))
        self.radio_va_mekf = QtWidgets.QRadioButton("MEKF (ориентация)")
        self.radio_va_scalar = QtWidgets.QRadioButton("скалярное |a|−g")
        self.radio_va_mekf.setToolTip(
            "Проекция ускорения на вертикаль через ориентацию (кватернион, pc/mekf.py):\n"
            "честно при наклонах и вращении — нет «фантомного подъёма» от ω²r.\n"
            "Инициализация ~1.5 с покоя внутри общего прогрева (на это время — скаляр).")
        self.radio_va_scalar.setToolTip(
            "Прежнее приближение |a_калибр| − g: честно в покое/наклоне,\n"
            "при вращении подмешивает центростремительное (для сравнения).")
        self._va_group = QtWidgets.QButtonGroup(self)
        self._va_group.addButton(self.radio_va_mekf)
        self._va_group.addButton(self.radio_va_scalar)
        lay2.addWidget(self.radio_va_mekf)
        lay2.addWidget(self.radio_va_scalar)
        self.radio_va_mekf.toggled.connect(self._on_va_mode_changed)
        lay2.addStretch(1)

        # реагируем на смену режима (хватает одного радио — они взаимоисключающие)
        self.radio_auto.toggled.connect(self._on_mode_changed)

        # --- ДВА СЛОТА РУЧНОЙ НАСТРОЙКИ (пакет 14, А.5; вместо «теневого
        # фильтра»): в режиме «Ручной» видны значения Авто (только чтение) и два
        # слота R/Q со своими кривыми на графике вариометра, живой метрикой
        # отличия от Авто и кнопкой «Сделать основным».
        self.gb_slots = QtWidgets.QGroupBox("Подбор R/Q — два пробных слота")
        gs = QtWidgets.QGridLayout(self.gb_slots)
        # строка эталона — ЖИВАЯ (пакет 15, Ж.1): эталонный фильтр «Авто»
        # адаптируется и в Ручном режиме, здесь его текущие R̂/Q̂ и метод
        self.lbl_auto_ref = QtWidgets.QLabel(
            f"Авто (эталон, живые): R̂ {R_DEFAULT:.4f} м² · "
            f"Q̂ {SIGMA_ACCEL_DEFAULT:.3f} м/с²")
        self.lbl_auto_ref.setToolTip(
            "Параллельный фильтр «Авто» с ЖИВЫМИ адаптивными R̂/Q̂ (как если бы\n"
            "стоял режим «Авто (адаптивный)») — эталон для сравнения слотов.\n"
            "Подпись метода — текущий способ вертикального ускорения.")
        self.lbl_auto_ref.setStyleSheet("color:#7a7f87;")
        gs.addWidget(self.lbl_auto_ref, 0, 0, 1, 8)
        self._slot_ui = {}
        for row, sk, title in ((1, "s1", "Ручной 1"), (2, "s2", "Ручной 2")):
            color = SLOT_COLORS[sk]
            chk = QtWidgets.QCheckBox(title)
            chk.setChecked(bool(self._slots[sk]["show"]))
            chk.setStyleSheet(f"color:{color}; font-weight:bold;")
            chk.setToolTip("Показать кривую этого слота на графике вариометра\n"
                           "и его значение в строке «Курсор».")
            chk.stateChanged.connect(lambda *_, k=sk: self._on_slot_toggled(k))
            gs.addWidget(chk, row, 0)
            # отметка «● основной» (Ж.3): значения слота сейчас в главных полях
            mark = QtWidgets.QLabel("")
            mark.setStyleSheet(f"color:{color}; font-weight:bold;")
            mark.setToolTip("Этот слот сейчас — ОСНОВНОЙ ручной фильтр\n"
                            "(его R и Q стоят в главных полях). Отметка живёт,\n"
                            "пока значения совпадают.")
            mark.setFixedWidth(88)
            gs.addWidget(mark, row, 1)
            gs.addWidget(QtWidgets.QLabel("R:"), row, 2)
            spR = StepSpinBox()
            spR.setDecimals(4)
            spR.setRange(0.0001, 100.0)
            spR.setSingleStep(0.005)
            spR.setSuffix(" м²")
            spR.setValue(float(self._slots[sk]["R"]))
            spR.setKeyboardTracking(False)
            spR.setToolTip("Шум барометра R этого слота (дисперсия, м²)")
            spR.valueChanged.connect(lambda *_, k=sk: self._on_slot_params(k))
            gs.addWidget(spR, row, 3)
            gs.addWidget(QtWidgets.QLabel("Q:"), row, 4)
            spQ = StepSpinBox()
            spQ.setDecimals(3)
            spQ.setRange(0.001, 10.0)
            spQ.setSingleStep(0.05)
            spQ.setSuffix(" м/с²")
            spQ.setValue(float(self._slots[sk]["sigma_accel"]))
            spQ.setKeyboardTracking(False)
            spQ.setToolTip("Шум ускорения sigma_accel этого слота (СКО, м/с²)")
            spQ.valueChanged.connect(lambda *_, k=sk: self._on_slot_params(k))
            gs.addWidget(spQ, row, 5)
            # метрика отличия (Ж.2): «Δ от Авто: 12%», при тихом Авто — абсолют
            met = make_value_label("Δ от Авто: 0.144 м/с", color)
            met.setText("Δ от Авто: —")
            met.setToolTip(
                "Δ от Авто = RMS(v_слота − v_авто) / RMS(v_авто) по видимому окну\n"
                "графика вариометра, в процентах. Если RMS(v_авто) < 0.05 м/с\n"
                "(эталон почти молчит — проценты врали бы), показывается абсолют:\n"
                "«Δ: 0.14 м/с» = RMS(v_слота − v_авто).")
            gs.addWidget(met, row, 6)
            btn = QtWidgets.QPushButton("→ Сделать основным")
            btn.setToolTip("Перенести R и Q этого слота в ОСНОВНОЙ ручной фильтр\n"
                           "(большие числа и звук всегда идут от основного).")
            btn.clicked.connect(lambda *_, k=sk: self._slot_make_main(k))
            gs.addWidget(btn, row, 7)
            self._slot_ui[sk] = {"chk": chk, "R": spR, "Q": spQ, "met": met,
                                 "mark": mark}
        gs.setColumnStretch(6, 1)
        # видимые поля «Δ» (пакет 15, Г.3): шаг колеса/стрелок для полей R и Q
        # (главных и обоих слотов) — правится тут же, без ПКМ
        drow = QtWidgets.QHBoxLayout()
        drow.addWidget(QtWidgets.QLabel("Δ шага:  R"))
        self.delta_R = make_delta_field(
            [self.spin_R, self._slot_ui["s1"]["R"], self._slot_ui["s2"]["R"]],
            0.005, "Δ для полей R (главного и слотов): на сколько меняется R\n"
                   "за один щелчок колеса/стрелок.")
        drow.addWidget(self.delta_R)
        drow.addWidget(QtWidgets.QLabel("Q"))
        self.delta_Q = make_delta_field(
            [self.spin_Q, self._slot_ui["s1"]["Q"], self._slot_ui["s2"]["Q"]],
            0.05, "Δ для полей Q (главного и слотов): на сколько меняется Q\n"
                  "за один щелчок колеса/стрелок.")
        drow.addWidget(self.delta_Q)
        drow.addStretch(1)
        gs.addLayout(drow, 3, 0, 1, 8)
        vbox.addWidget(self.gb_slots)

        self._root_layout.addWidget(box)

    def _build_filter(self, R: float, sigma_accel: float, h0: float = 0.0):
        """Создать фильтр из R и sigma_accel. В фильтре R = sigma_baro², значит sigma_baro = sqrt(R)."""
        sigma_baro = max(float(R), 1e-9) ** 0.5
        flt = BaroInertialVario(dt=DT, sigma_accel=float(sigma_accel),
                                sigma_baro=sigma_baro, h0=float(h0))
        flt.huber_k = self._huber_k          # робастный баро (пакет 15, А.2)
        return flt

    def _rebuild_all_filters(self, R: float, Q: float, h0: float = 0.0):
        """Пересоздать ВСЕ фильтры одной командой: основной, теневой второго
        метода (те же R/Q, Г.1), эталон «Авто» и два слота ручной настройки
        (пакет 14, А.5)."""
        self.filter = self._build_filter(R, Q, h0=h0)
        self.filter2 = self._build_filter(R, Q, h0=h0)
        self.filter_auto = self._build_filter(R_DEFAULT, SIGMA_ACCEL_DEFAULT, h0=h0)
        self.filter_s1 = self._build_filter(self._slots["s1"]["R"],
                                            self._slots["s1"]["sigma_accel"], h0=h0)
        self.filter_s2 = self._build_filter(self._slots["s2"]["R"],
                                            self._slots["s2"]["sigma_accel"], h0=h0)

    def _params_for_mode(self):
        """Вернуть (R, sigma_accel) для текущего режима: defaults в авто, пользовательские в ручном."""
        if self.mode == "manual":
            return self.manual_params["R"], self.manual_params["sigma_accel"]
        return R_DEFAULT, SIGMA_ACCEL_DEFAULT

    def _refresh_panel_enabled(self):
        """Поля и кнопка активны только в ручном режиме; панель слотов (А.5)
        в режиме Авто скрывается."""
        manual = self.radio_manual.isChecked()
        for w in (self.spin_R, self.spin_Q, self.spin_calib, self.btn_apply):
            w.setEnabled(manual)
        self.gb_slots.setVisible(manual)
        self._apply_slot_visibility()

    # ------------------------------------------------------------------
    # СЛОТЫ РУЧНОЙ НАСТРОЙКИ (пакет 14, А.5)
    # ------------------------------------------------------------------
    def _apply_slot_visibility(self):
        """Кривые слотов видны только в Ручном режиме и по своим галочкам."""
        manual = self.mode == "manual"
        self.curve_v_s1.setVisible(manual and bool(self._slots["s1"]["show"]))
        self.curve_v_s2.setVisible(manual and bool(self._slots["s2"]["show"]))

    def _on_slot_toggled(self, sk: str):
        """Галочка слота: видимость кривой + значение в строке «Курсор»."""
        if self._loading:
            return
        self._slots[sk]["show"] = bool(self._slot_ui[sk]["chk"].isChecked())
        self._apply_slot_visibility()
        self._save_config()
        if self._file is not None:
            self._file_set_curves()
        else:
            self._dirty = True
        self._refresh_cursor()

    def _on_slot_params(self, sk: str):
        """Поля R/Q слота: пересоздать фильтр слота (основной не трогаем);
        файл — перепрогнать (слотам нужны серии)."""
        if self._loading:
            return
        ui = self._slot_ui[sk]
        self._slots[sk]["R"] = float(ui["R"].value())
        self._slots[sk]["sigma_accel"] = float(ui["Q"].value())
        flt = self._build_filter(self._slots[sk]["R"],
                                 self._slots[sk]["sigma_accel"],
                                 h0=self._current_alt())
        if sk == "s1":
            self.filter_s1 = flt
        else:
            self.filter_s2 = flt
        self._save_config()
        if self._file is not None and self.worker is None:
            path = self._file["path"]
            self._playing = False
            if self._load_file_full(path):
                self.statusBar().showMessage(
                    f"Слот «{'Ручной 1' if sk == 's1' else 'Ручной 2'}»: "
                    f"R = {self._slots[sk]['R']:.4f}, "
                    f"Q = {self._slots[sk]['sigma_accel']:.3f} — файл перепрогнан.")

    def _slot_make_main(self, sk: str):
        """«Сделать основным»: перенести R/Q слота в основной ручной фильтр
        (большие числа и звук всегда идут от основного)."""
        if self.mode != "manual":
            return
        self.spin_R.setValue(float(self._slots[sk]["R"]))
        self.spin_Q.setValue(float(self._slots[sk]["sigma_accel"]))
        self._apply_manual_params()
        if self._file is not None and self.worker is None:
            path = self._file["path"]
            self._playing = False
            self._load_file_full(path)
        self.statusBar().showMessage(
            f"Слот «{'Ручной 1' if sk == 's1' else 'Ручной 2'}» стал основным: "
            f"R = {self._slots[sk]['R']:.4f}, "
            f"Q = {self._slots[sk]['sigma_accel']:.3f}.")

    def _slot_marks_update(self):
        """Ж.3: отметка «● основной» у слота, чьи R/Q сейчас стоят в главных
        полях ручного фильтра (живёт, пока значения совпадают)."""
        for sk in ("s1", "s2"):
            is_main = (self.mode == "manual"
                       and abs(self._slots[sk]["R"]
                               - float(self.manual_params["R"])) < 1e-9
                       and abs(self._slots[sk]["sigma_accel"]
                               - float(self.manual_params["sigma_accel"])) < 1e-9)
            mark = self._slot_ui[sk]["mark"]
            txt = "● основной" if is_main else ""
            if mark.text() != txt:
                mark.setText(txt)

    def _slot_metrics_update(self):
        """Живая метрика слотов (~1 раз/с), пакет 15 Ж.2: «Δ от Авто: N%» =
        RMS(v_слота − v_авто)/RMS(v_авто) по ВИДИМОМУ окну графика вариометра;
        при RMS(v_авто) < 0.05 м/с — абсолют «Δ: X м/с» (проценты от ноль-шума
        врали бы). Плюс живая строка эталона (Ж.1) и отметки «● основной» (Ж.3)."""
        if self.mode != "manual":
            return
        import time as _t
        now = _t.monotonic()
        if now < self._slot_metric_next:
            return
        self._slot_metric_next = now + 1.0
        self._slot_marks_update()
        # живой эталон «Авто» (Ж.1): текущие адаптивные R̂/Q̂ + метод
        if self.filter_auto is not None:
            va_txt = ("MEKF" if self._va_mode == "mekf" else "скаляр")
            self.lbl_auto_ref.setText(
                f"Авто (эталон, живые, метод {va_txt}): "
                f"R̂ {float(self.filter_auto.R[0, 0]):.4f} м² · "
                f"Q̂ {float(self.filter_auto.sigma_accel):.3f} м/с²"
                + ("" if self._adaptive_on else " · адаптация выключена"))
        # серии: файл или кольцевые буферы
        if self._file is not None:
            f = self._file
            t = f["t"]
            va = f.get("v_auto")
            vs = {"s1": f.get("v_s1"), "s2": f.get("v_s2")}
        elif self.buf_t:
            t = np.fromiter(self.buf_t, dtype=float)
            va = np.fromiter(self.buf_v_auto, dtype=float)
            vs = {"s1": np.fromiter(self.buf_v_s1, dtype=float),
                  "s2": np.fromiter(self.buf_v_s2, dtype=float)}
        else:
            return
        if va is None:
            return
        try:
            x0, x1 = self.plot_vario.getViewBox().viewRange()[0]
        except Exception:
            return
        m = (t >= x0) & (t <= x1) & np.isfinite(va)
        if m.sum() < 10:
            for sk in ("s1", "s2"):
                self._slot_ui[sk]["met"].setText("Δ от Авто: —")
            return
        rms_auto = float(np.sqrt(np.mean(va[m] ** 2)))
        for sk in ("s1", "s2"):
            v = vs.get(sk)
            if v is None or len(v) != len(t):
                self._slot_ui[sk]["met"].setText("Δ от Авто: —")
                continue
            d = v[m] - va[m]
            d = d[np.isfinite(d)]
            if d.size < 10:
                self._slot_ui[sk]["met"].setText("Δ от Авто: —")
                continue
            rms = float(np.sqrt(np.mean(d ** 2)))
            if rms_auto >= 0.05:
                self._slot_ui[sk]["met"].setText(
                    f"Δ от Авто: {100.0 * rms / rms_auto:3.0f}%")
            else:
                self._slot_ui[sk]["met"].setText(f"Δ: {rms:5.2f} м/с")

    def _set_motion_label(self, state: str):
        """Индикатор Покой (зелёный) / Движение (оранжевый) / Качание (синий).
        «Качание» (пакет 15, А.3) — знакопеременное размахивание: баро «дышит»
        портом, его дисперсия временно ×k_baro; ZUPT в качании выключен."""
        txt, color = {"rest": ("Покой", "#2c7a2c"),
                      "osc": ("Качание", "#2f6fd0")}.get(state,
                                                         ("Движение", "#e67e22"))
        if self.lbl_motion.text() != txt:
            self.lbl_motion.setText(txt)
            self.lbl_motion.setStyleSheet(f"font-weight: bold; color: {color};")

    def _mirror_filter_state(self, flt):
        """Скопировать в параллельный фильтр (тень/слоты/эталон) текущее
        состояние детекторов основного: доверие акселю, заморозку bias,
        множитель R баро (качание) и ZUPT (пакет 15, блок А)."""
        flt.accel_trust = self._trust
        flt.bias_frozen = self.filter.bias_frozen
        flt.R_baro_mult = self.filter.R_baro_mult
        flt.zupt_r = self.filter.zupt_r

    def _current_alt(self) -> float:
        """Текущая оценка высоты (чтобы при пересоздании фильтра график не прыгал)."""
        return self.buf_h_filt[-1] if self.buf_h_filt else 0.0

    def last_baro_altitude(self):
        """Последняя высота по барометру из текущего источника (для панели «Эталоны»). None, если данных нет."""
        return float(self.buf_h_raw[-1]) if self.buf_h_raw else None

    def _sync_filter_panel(self):
        """Стартовая настройка панели: выставить значения и режим, создать фильтр. Без сигналов."""
        self._loading = True
        if self.mode == "manual":
            self.radio_manual.setChecked(True)
            self.spin_R.setValue(self.manual_params["R"])
            self.spin_Q.setValue(self.manual_params["sigma_accel"])
            self.spin_calib.setValue(self.manual_params["calib_time"])
        else:
            self.radio_auto.setChecked(True)
            # в авто-режиме в полях показываем именно значения по умолчанию
            self.spin_R.setValue(R_DEFAULT)
            self.spin_Q.setValue(SIGMA_ACCEL_DEFAULT)
            self.spin_calib.setValue(CALIB_TIME_DEFAULT)
        # источник вертикального ускорения — из config.json
        (self.radio_va_mekf if self._va_mode == "mekf"
         else self.radio_va_scalar).setChecked(True)
        self.lbl2_cap.setText("2-й метод (|a|−g):" if self._va_mode == "mekf"
                              else "2-й метод (MEKF):")
        self._refresh_panel_enabled()
        self._loading = False

        # создать фильтр под текущий режим
        R, Q = self._params_for_mode()
        self._rebuild_all_filters(R, Q)

    def _on_va_mode_changed(self):
        """Переключатель «Верт. ускорение»: MEKF (ориентация) / скалярное.
        Меняется ТОЛЬКО вход фильтра высоты (пакет 14, Б.1): ориентация (AHRS)
        считается всегда и НЕ переинициализируется — курс при переключении не
        прыгает. Файл перепрогоняется сразу; живой поток подхватывает на лету."""
        if self._loading:
            return
        self._va_mode = "mekf" if self.radio_va_mekf.isChecked() else "scalar"
        self.lbl2_cap.setText("2-й метод (|a|−g):" if self._va_mode == "mekf"
                              else "2-й метод (MEKF):")
        self._save_config()
        mode_txt = ("MEKF (проекция через ориентацию)" if self._va_mode == "mekf"
                    else "скалярное |a|−g")
        if self._file is not None and self.worker is None:
            # файл прогнан пакетно старым режимом → перепрогнать целиком новым
            path = self._file["path"]
            self._playing = False
            if self._load_file_full(path):
                self.statusBar().showMessage(
                    f"Верт. ускорение: {mode_txt} — файл перепрогнан. «Старт» — проиграть.")
        else:
            self.statusBar().showMessage(f"Верт. ускорение: {mode_txt}.")

    def _on_mode_changed(self):
        """Пользователь переключил Авто/Ручной."""
        if self._loading:
            return  # это программная настройка при запуске — ничего не делаем
        self.mode = "manual" if self.radio_manual.isChecked() else "auto"
        self._refresh_panel_enabled()
        if self.mode == "auto":
            # авто: показываем значения по умолчанию и СРАЗУ применяем их к фильтру
            self.spin_R.setValue(R_DEFAULT)
            self.spin_Q.setValue(SIGMA_ACCEL_DEFAULT)
            self.spin_calib.setValue(CALIB_TIME_DEFAULT)
            self._rebuild_all_filters(R_DEFAULT, SIGMA_ACCEL_DEFAULT, h0=self._current_alt())
            self.statusBar().showMessage("Авто-режим: параметры по умолчанию применены.")
        else:
            # ручной: вернуть сохранённые значения; применятся по кнопке «Применить»
            self.spin_R.setValue(self.manual_params["R"])
            self.spin_Q.setValue(self.manual_params["sigma_accel"])
            self.spin_calib.setValue(self.manual_params["calib_time"])
            self.statusBar().showMessage("Ручной режим: измените параметры и нажмите «Применить».")
        self._save_config()
        # загруженный файл перепрогоняется под новый режим (в Ручном появляются
        # серии слотов и эталона Авто — А.5; в Авто главный фильтр — паспортный)
        if self._file is not None and self.worker is None:
            path = self._file["path"]
            self._playing = False
            self._load_file_full(path)

    def _apply_manual_params(self):
        """Кнопка «Применить»: взять значения из полей и пересоздать фильтр (только в ручном режиме)."""
        if not self.radio_manual.isChecked():
            return
        self.manual_params = {
            "R": float(self.spin_R.value()),
            "sigma_accel": float(self.spin_Q.value()),
            "calib_time": float(self.spin_calib.value()),
        }
        # пересоздаём фильтры; текущую высоту сохраняем (h0), чтобы график не прыгнул
        self._rebuild_all_filters(self.manual_params["R"],
                                  self.manual_params["sigma_accel"],
                                  h0=self._current_alt())
        self._save_config()
        self.statusBar().showMessage(
            f"Применено: R = {self.manual_params['R']:.4f} м², "
            f"sigma_accel = {self.manual_params['sigma_accel']:.3f} м/с², "
            f"установка нуля = {self.manual_params['calib_time']:.1f} с.")

    def _save_config(self):
        """Сохранить режим, параметры фильтра и настройки просмотра в config.json."""
        save_config({
            "mode": self.mode,
            "manual": self.manual_params,
            "view": self._view_settings(),
            "smooth_sec": float(self.spin_smooth.value()),
            "motion": self._motion_cfg,   # пороги детектора покоя/движения
            "zupt": self._zupt_cfg,       # строгий покой → v=0 (пакет 15, А.1)
            "osc": self._osc_cfg,         # детектор качания (А.3)
            "huber_k": self._huber_k,     # робастный баро (А.2)
            "vertical_accel_mode": self._va_mode,
            "show_second": bool(self._show_second),
            "manual_slots": {k: {"R": float(v["R"]),
                                 "sigma_accel": float(v["sigma_accel"]),
                                 "show": bool(v["show"])}
                             for k, v in self._slots.items()},
            "sound_source": self.sound_source,
            "panel_layout": self._panel_layout,
        })

    # ------------------------------------------------------------------
    # ПАНЕЛЬ «ПРОСМОТР»: окно по времени и шаг сетки
    # ------------------------------------------------------------------
    def _build_view_bar(self):
        """ДВЕ независимые пары контролов: своё «окно просмотра» и свой «шаг сетки»
        у графика ВЫСОТЫ и отдельно у графика ВАРИОМЕТРА (оси X не связаны)."""
        vbar = QtWidgets.QHBoxLayout()
        items = ["5 с", "10 с", "20 с", "30 с", "60 с", "Всё"]

        def make_pair(title):
            vbar.addWidget(QtWidgets.QLabel(title))
            combo = QtWidgets.QComboBox()
            combo.addItems(items)
            combo.setToolTip(
                "Сколько последних секунд показывать на ЭТОМ графике\n"
                "(своё, независимое скользящее окно). «Всё» — вся запись.")
            combo.currentIndexChanged.connect(self._on_window_changed)
            vbar.addWidget(combo)
            vbar.addWidget(QtWidgets.QLabel("сетка, с:"))
            spin = StepSpinBox()
            spin.setDecimals(1)
            spin.setRange(0.5, 120.0)
            spin.setSingleStep(0.5)
            spin.setSuffix(" с")
            spin.setToolTip(
                "Густота линий сетки по времени для ЭТОГО графика.\n"
                "Это НЕ разрешение данных: данные всегда 50 Гц, рисуются ВСЕ точки.")
            spin.valueChanged.connect(self._on_grid_changed)
            vbar.addWidget(spin)
            return combo, spin

        self.combo_window_alt, self.spin_grid_alt = make_pair("Высота — окно:")
        vbar.addSpacing(25)
        self.combo_window_vario, self.spin_grid_vario = make_pair("Вариометр — окно:")
        vbar.addSpacing(25)

        # галочки видимости кривых (сюда с верхней панели — для узких окон)
        self.chk_filt = QtWidgets.QCheckBox("показать фильтрованное")
        self.chk_filt.setChecked(True)
        self.chk_filt.stateChanged.connect(self._apply_visibility)
        vbar.addWidget(self.chk_filt)
        self.chk_raw = QtWidgets.QCheckBox("показать сырое")
        self.chk_raw.setChecked(True)
        self.chk_raw.stateChanged.connect(self._apply_visibility)
        vbar.addWidget(self.chk_raw)
        self.chk_smooth = QtWidgets.QCheckBox("показать сглаженное")
        self.chk_smooth.setChecked(self._init_show_smooth)
        self.chk_smooth.setToolTip(
            "Зелёные кривые: вариометр и высота, сглаженные каузальным гауссом за N с\n"
            "(ровно та серия, что даёт число «Сглаж. N с»). N — поле «Сглаж.(Гаусс), с».")
        self.chk_smooth.stateChanged.connect(self._apply_visibility)
        vbar.addWidget(self.chk_smooth)
        # RTS-сглаживатель (З.2): ТОЛЬКО файловый режим (А.6: в live галочка скрыта)
        self.chk_rts = QtWidgets.QCheckBox("RTS (анализ)")
        self.chk_rts.setChecked(self._show_rts)
        self.chk_rts.setToolTip(
            "RTS — сглаживание ЗАДНИМ ЧИСЛОМ по всему файлу (обратный проход\n"
            "Раух-Тунг-Стрибела): использует и прошлое, и БУДУЩЕЕ каждой точки,\n"
            "поэтому глаже фильтра. Только для анализа записей — в реальном\n"
            "времени будущего нет, в живом потоке кривая не участвует.\n"
            "Включение перепрогоняет файл.")
        self.chk_rts.stateChanged.connect(self._on_rts_toggled)
        vbar.addWidget(self.chk_rts)

        vbar.addStretch(1)
        # компактный дубль звука с пакета 15 живёт в КАРТОЧКЕ «Громкость» (Е.3)
        # --- КОМПОНОВКА карточек шапки (пакет 15, Е.3) ---
        self.btn_layout_mode = QtWidgets.QPushButton("Компоновка")
        self.btn_layout_mode.setCheckable(True)
        self.btn_layout_mode.setToolTip(
            "Режим компоновки карточек шапки: видна сетка 8 px, карточки\n"
            "таскаются мышью (прилипают к узлам), заголовки редактируются\n"
            "(очистить — вернётся серый оригинал). Графики и панель фильтра\n"
            "не двигаются (v1).")
        self.btn_layout_mode.toggled.connect(self._on_layout_mode)
        vbar.addWidget(self.btn_layout_mode)
        self.btn_layout_save = QtWidgets.QPushButton("Зафиксировать и сохранить вид…")
        self.btn_layout_save.setToolTip(
            "Сохранить текущую компоновку карточек в data\\layouts\\*.json\n"
            "и сделать её активной. Применить/удалить/вернуть заводскую —\n"
            "на вкладке «Записи», секция «Виды пульта».")
        self.btn_layout_save.clicked.connect(self._save_layout_file)
        self.btn_layout_save.setVisible(False)     # виден в режиме компоновки
        vbar.addWidget(self.btn_layout_save)
        self.btn_layouts_nav = QtWidgets.QPushButton("Виды…")
        self.btn_layouts_nav.setToolTip(
            "Сохранённые виды пульта: вкладка «Записи» → «Виды пульта»\n"
            "(список мигнёт зелёной рамкой).")
        self.btn_layouts_nav.clicked.connect(
            lambda: self.files_nav_cb("layouts") if self.files_nav_cb else None)
        vbar.addWidget(self.btn_layouts_nav)
        self.btn_export = QtWidgets.QPushButton("💾 Экспорт PNG")
        self.btn_export.clicked.connect(self.export_png)
        vbar.addWidget(self.btn_export)
        self._root_layout.addLayout(vbar)
        # строка-факт про RTS по загруженному файлу (А.6): что это и фактический K
        self.lbl_rts_info = QtWidgets.QLabel("")
        self.lbl_rts_info.setStyleSheet("font-size: 11px; color: #b8741a;")
        self.lbl_rts_info.setWordWrap(True)
        self._root_layout.addWidget(self.lbl_rts_info)

    def _window_seconds(self, combo):
        """Окно в секундах для данного выпадающего списка или None для «Всё»."""
        txt = combo.currentText()
        if txt.startswith("Всё"):
            return None
        try:
            return float(txt.split()[0].replace(",", "."))
        except ValueError:
            return None

    def _view_settings(self):
        """Настройки просмотра (ПО ГРАФИКУ) для сохранения в config.json."""
        return {
            "alt":   {"window_sec": self._window_seconds(self.combo_window_alt),
                      "grid_step": float(self.spin_grid_alt.value()),
                      "y_mode": self._y_mode("alt"),
                      "y_min": float(self.yspin_min_alt.value()),
                      "y_max": float(self.yspin_max_alt.value())},
            "vario": {"window_sec": self._window_seconds(self.combo_window_vario),
                      "grid_step": float(self.spin_grid_vario.value()),
                      "y_mode": self._y_mode("vario"),
                      "y_min": float(self.yspin_min_vario.value()),
                      "y_max": float(self.yspin_max_vario.value())},
        }

    def _apply_grid_step(self):
        """Свой шаг сетки по времени КАЖДОМУ графику отдельно (с авто-загущением)."""
        self._update_time_ticks("alt")
        self._update_time_ticks("vario")

    @staticmethod
    def _nice_step(target: float) -> float:
        """Ближайший «красивый» шаг ≥ target из ряда 1-2-5·10^k."""
        if target <= 0:
            return 1.0
        import math as _m
        k = _m.floor(_m.log10(target))
        for mult in (1.0, 2.0, 5.0, 10.0):
            step = mult * (10.0 ** k)
            if step >= target - 1e-12:
                return step
        return 10.0 ** (k + 1)

    def _update_time_ticks(self, key: str):
        """Шаг сетки времени: из поля «сетка, с», но если окно ШИРОКОЕ и подписей
        вышло бы больше ~14 — автоматически увеличиваем шаг до «красивого»
        (1-2-5·10^k), чтобы подписи оси не слипались."""
        plot, spin = ((self.plot_alt, self.spin_grid_alt) if key == "alt"
                      else (self.plot_vario, self.spin_grid_vario))
        step = float(spin.value())
        try:
            x0, x1 = plot.getViewBox().viewRange()[0]
            span = max(0.0, x1 - x0)
        except Exception:
            span = 0.0
        if span > 0 and span / step > 14:
            step = self._nice_step(span / 10.0)
        # переустанавливаем, только если шаг реально изменился (вызов идёт на
        # каждое изменение X-диапазона — при слежении это каждый кадр)
        if not hasattr(self, "_tick_step"):
            self._tick_step = {}
        if self._tick_step.get(key) == step:
            return
        self._tick_step[key] = step
        ax = plot.getAxis("bottom")
        ax.setTickSpacing(major=step, minor=step)

    # ------------------------------------------------------------------
    # МАСШТАБ ОСИ Y: «Авто (по окну)» с гистерезисом / «Ручной» (поля min-max)
    # ------------------------------------------------------------------
    def _y_widgets(self, key: str):
        if key == "alt":
            return (self.plot_alt, self.ycombo_alt,
                    self.yspin_min_alt, self.yspin_max_alt, self.combo_window_alt)
        return (self.plot_vario, self.ycombo_vario,
                self.yspin_min_vario, self.yspin_max_vario, self.combo_window_vario)

    def _y_mode(self, key: str) -> str:
        combo = self.ycombo_alt if key == "alt" else self.ycombo_vario
        return "manual" if combo.currentIndex() == 1 else "auto"

    def _on_vb_autorange(self, key: str):
        """А.4 (пакет 14): кнопка «A» pyqtgraph / «View All» включили внутренний
        авто-масштаб Y вьюбокса, а комбо стояло в «Ручной» — рассинхрон. Комбо —
        единый источник истины: переводим его в «Авто (по окну)» штатным путём
        (наш авто-режим выставит рамку и сам погасит внутренний авто-масштаб)."""
        if self._loading:
            return
        plot, combo, _s1, _s2, _w = self._y_widgets(key)
        try:
            y_auto = bool(plot.getViewBox().autoRangeEnabled()[1])
        except Exception:
            return
        if y_auto and self._y_mode(key) == "manual":
            combo.setCurrentIndex(0)          # → _on_y_mode_changed("auto")

    def _on_y_mode_changed(self, key: str):
        if self._loading:
            return
        plot, combo, s_min, s_max, _w = self._y_widgets(key)
        manual = self._y_mode(key) == "manual"
        if manual:
            # старт ручного режима — с текущих видимых пределов (без прыжка)
            lo, hi = plot.getViewBox().viewRange()[1]
            self._loading = True
            s_min.setValue(lo)
            s_max.setValue(hi)
            self._loading = False
            plot.setYRange(lo, hi, padding=0)
        else:
            self._y_next[key] = 0.0     # авто: пересчитать немедленно
            self._auto_y_tick(force_key=key)
        self._save_config()

    def _on_y_manual_changed(self, key: str, which: str | None = None):
        """Поля min/max. Ось обязана показывать РОВНО [min,max]. Если после
        правки min ≥ max — редактируемое поле главное, второе подтягивается.
        А.4: правка поля в режиме «Авто» переводит график в «Ручной» (поля
        всегда активны — единый источник истины без рассинхрона)."""
        if self._loading:
            return
        plot, combo, s_min, s_max, _w = self._y_widgets(key)
        if self._y_mode(key) != "manual":
            self._loading = True
            combo.setCurrentIndex(1)          # «Ручной» — без пересчёта полей
            self._loading = False
        lo, hi = float(s_min.value()), float(s_max.value())
        if hi <= lo:
            gap = 0.1 if key == "vario" else 1.0
            self._loading = True
            if which == "max":
                lo = hi - gap
                s_min.setValue(lo)
            else:                      # правили min (или неизвестно) — тянем max
                hi = lo + gap
                s_max.setValue(hi)
            self._loading = False
        plot.setYRange(lo, hi, padding=0)
        self._save_config()

    def _y_series_in_window(self, key: str, x0: float, x1: float):
        """Данные для авто-масштаба Y в окне [x0, x1]: фильтрованная серия
        (+сглаженная, если включена). Сырую производную баро НЕ берём — её
        выбросы раздували бы масштаб. Возвращает np-массив или None."""
        ref = self._h_ref or 0.0
        if self._file is not None:
            f = self._file
            t = f["t"]
            series = [f["h_filt"] - ref] if key == "alt" else [f["v_filt"]]
            if self.chk_smooth.isChecked():
                sm = f.get("h_smooth" if key == "alt" else "v_smooth")
                if sm is not None:
                    series.append(sm - ref if key == "alt" else sm)
        else:
            if not self.buf_t:
                return None
            t = np.fromiter(self.buf_t, dtype=float)
            if key == "alt":
                series = [np.fromiter(self.buf_h_filt, dtype=float) - ref]
            else:
                series = [np.fromiter(self.buf_v_filt, dtype=float)]
        m = (t >= x0) & (t <= x1)
        if not m.any():
            return None
        vals = np.concatenate([s[m] for s in series])
        vals = vals[np.isfinite(vals)]
        return vals if vals.size else None

    def _auto_y_tick(self, force_key: str | None = None):
        """Автомасштаб Y: раз в ДЛИНУ ОКНА пересчитать min/max по данным в видимом
        окне (+10% запаса). Гистерезис: если новая рамка почти та же (<5% размаха)
        — не трогаем, чтобы график не дёргался каждый кадр. Если данные вылезли
        за рамку — пересчитываем сразу, не дожидаясь периода."""
        import time as _t
        now = _t.monotonic()
        for key in ("alt", "vario"):
            if self._y_mode(key) != "auto":
                continue
            plot, _c, _s1, _s2, combo_win = self._y_widgets(key)
            vb = plot.getViewBox()
            try:
                (x0, x1), (ylo, yhi) = vb.viewRange()
            except Exception:
                continue
            vals = self._y_series_in_window(key, x0, x1)
            if vals is None:
                continue
            vmin, vmax = float(vals.min()), float(vals.max())
            escaped = vmin < ylo or vmax > yhi
            due = (now >= self._y_next[key]) or (force_key == key) or escaped
            if not due:
                continue
            win = self._window_seconds(combo_win)
            self._y_next[key] = now + (win if win else 5.0)
            pad = 0.10 * max(vmax - vmin, 0.2 if key == "vario" else 1.0)
            lo, hi = vmin - pad, vmax + pad
            span = max(yhi - ylo, 1e-9)
            # гистерезис: рамка сдвигается, только если изменение заметное
            if (not escaped and force_key != key
                    and abs(lo - ylo) < 0.05 * span and abs(hi - yhi) < 0.05 * span):
                continue
            plot.setYRange(lo, hi, padding=0)

    def _on_user_plot_interact(self, key: str):
        """РЕАЛЬНОЕ колесо/перетаскивание на графике (из PlotInteractGuard):
        отключить слежение за временем; если пользователь изменил ось Y —
        перевести ЭТОТ график в «Ручной» (поля min/max подхватят текущий вид)."""
        self._user_panned()
        plot, combo, s_min, s_max, _w = self._y_widgets(key)
        before = tuple(plot.getViewBox().viewRange()[1])

        def check_after():
            lo, hi = plot.getViewBox().viewRange()[1]
            span = max(abs(before[1] - before[0]), 1e-9)
            if (abs(lo - before[0]) > 1e-3 * span or abs(hi - before[1]) > 1e-3 * span):
                if self._y_mode(key) != "manual":
                    self._loading = True
                    combo.setCurrentIndex(1)      # «Ручной»
                    self._loading = False
                self._loading = True
                s_min.setValue(lo)
                s_max.setValue(hi)
                self._loading = False
                self._save_config()
        QtCore.QTimer.singleShot(0, check_after)

    def _on_window_changed(self):
        if self._loading:
            return
        self._view_refresh = True   # применить новое окно сразу (даже на паузе)
        self._y_next = {"alt": 0.0, "vario": 0.0}   # авто-Y пересчитать сразу
        self._save_config()

    def _on_grid_changed(self):
        if self._loading:
            return
        self._apply_grid_step()
        self._save_config()

    def _sync_view(self):
        """Стартовая настройка просмотра из config.json (без срабатывания сигналов)."""
        self._loading = True
        for combo, spin, key in (
            (self.combo_window_alt, self.spin_grid_alt, "alt"),
            (self.combo_window_vario, self.spin_grid_vario, "vario"),
        ):
            sub = self._init_view.get(key, {})
            win = sub.get("window_sec", None)
            if win is None:
                combo.setCurrentText("Всё")
            else:
                idx = combo.findText(f"{int(win)} с")
                combo.setCurrentIndex(idx if idx >= 0 else combo.count() - 1)
            spin.setValue(float(sub.get("grid_step", GRID_STEP_DEFAULT)))
            # режим оси Y и ручные пределы — из config.json
            _plot, ycombo, s_min, s_max, _w = self._y_widgets(key)
            manual = sub.get("y_mode") == "manual"
            ycombo.setCurrentIndex(1 if manual else 0)
            s_min.setValue(float(sub.get("y_min", -3.0 if key == "vario" else 0.0)))
            s_max.setValue(float(sub.get("y_max", 3.0 if key == "vario" else 300.0)))
            if manual and s_max.value() > s_min.value():
                _plot.setYRange(float(s_min.value()), float(s_max.value()), padding=0)
        self._loading = False
        self._apply_grid_step()
        self._view_refresh = True   # один раз подогнать рамки при запуске

    # ------------------------------------------------------------------
    # ОБРАБОТЧИКИ ИНТЕРФЕЙСА
    # ------------------------------------------------------------------
    def _on_source_changed(self):
        """Файловые кнопки включаем только для CSV, поле URL — только для потока.
        Галочка RTS (А.6) видна только в файловом режиме — в живом потоке
        сглаживание задним числом невозможно по построению."""
        kind = self.combo_source.currentText()
        self.btn_file.setEnabled(kind == "CSV-файл")
        self.btn_file_browse.setEnabled(kind == "CSV-файл")
        self.edit_stream_url.setEnabled(kind.startswith("Поток"))
        file_mode = kind == "CSV-файл"
        self.chk_rts.setVisible(file_mode)
        self.lbl_rts_info.setVisible(file_mode)

    def _on_file_button(self):
        """«Файл…»: перейти на вкладку «Записи» (мигнёт нужный список); если пульт
        запущен отдельным окном (без вкладок) — классический диалог."""
        if self.files_nav_cb is not None:
            self.files_nav_cb("session")
        else:
            self._choose_file()

    def _choose_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Выберите CSV-запись", DATA_DIR, "CSV-файлы (*.csv);;Все файлы (*)")
        if path:
            self.csv_path = path
            # файловый режим: сразу прогоняем ВЕСЬ файл и показываем кривые целиком
            self._load_file_full(path)

    def _apply_visibility(self):
        """Показать/спрятать кривые по галочкам (галка «сглаженное» — в config.json)."""
        show_filt = self.chk_filt.isChecked()
        show_raw = self.chk_raw.isChecked()
        show_sm = self.chk_smooth.isChecked()
        self.curve_h_filt.setVisible(show_filt)
        self.curve_v_filt.setVisible(show_filt)
        self.curve_h_raw.setVisible(show_raw)
        self.curve_v_raw.setVisible(show_raw)
        self.curve_h_smooth.setVisible(show_sm)
        self.curve_v_smooth.setVisible(show_sm)
        self.curve_v2.setVisible(self.chk_second.isChecked())
        if not self._loading:
            save_config({"show_smooth": show_sm})

    def _on_second_toggled(self):
        """Галочка «показать 2-й метод» (Г.1): пунктир на графике вариометра."""
        if self._loading:
            return
        self._show_second = self.chk_second.isChecked()
        self.curve_v2.setVisible(self._show_second)
        self._save_config()
        if self._file is not None:
            self._file_set_curves()
        else:
            self._dirty = True

    # ------------------------------------------------------------------
    # ФАЙЛОВЫЙ РЕЖИМ: весь файл сразу (пакетный прогон) + проигрывание-плеер
    # ------------------------------------------------------------------
    def _load_file_full(self, path: str) -> bool:
        """Прогнать ВЕСЬ CSV через ТОТ ЖЕ пайплайн (_process_sample: тот же фильтр,
        посев первым баро, детектор, ноль) пакетно и показать кривые целиком.
        История не обрезается; плеер потом просто ходит по готовым сериям —
        поэтому числа при проигрывании ТОЖДЕСТВЕННО совпадают с пакетным прогоном."""
        try:
            src = CsvSource(path, realtime=False)
            src.open()
        except Exception as e:
            self._show_error(str(e))
            return False
        cur_t = self._cursor_t            # курсор переживает перепрогон файла (А.3)
        self._clear_data()
        self._load_mag_calibration()
        self._zero_N = float(self.spin_calib.value())
        if self._show_rts:
            self.filter.record = []       # запись шагов для RTS (З.2)
        T, HR, HF, VR, VF, HD, MO = [], [], [], [], [], [], []
        V2, VA, VS1, VS2 = [], [], [], []
        n_warm = 0                        # шагов фильтра до конца прогрева
        while True:
            s = src.read_sample()
            if s is None:
                break
            r = self._process_sample(s, quiet=True)
            if r["warming"]:
                n_warm += 1
                continue
            T.append(r["t"]); HR.append(r["h_baro"]); HF.append(r["h_filt"])
            VR.append(r["v_raw"]); VF.append(r["v_filt"])
            HD.append(r["heading"] if r["heading"] is not None else float("nan"))
            # 1 = Покой, 0 = Движение, −1 = Качание (пакет 15, А.3);
            # проверки «rest = motion > 0» работают как раньше
            MO.append({"rest": 1, "osc": -1}.get(r["motion"], 0))
            V2.append(r["v2"] if r["v2"] is not None else float("nan"))
            VA.append(r["v_auto"] if r["v_auto"] is not None else float("nan"))
            VS1.append(r["v_s1"] if r["v_s1"] is not None else float("nan"))
            VS2.append(r["v_s2"] if r["v_s2"] is not None else float("nan"))
        src.close()
        if len(T) < 2:
            self._show_error("В файле слишком мало данных.")
            return False
        f = {"path": path,
             "t": np.array(T), "h_raw": np.array(HR), "h_filt": np.array(HF),
             "v_raw": np.array(VR), "v_filt": np.array(VF),
             "head": np.array(HD), "motion": np.array(MO),
             "v2": np.array(V2), "v_auto": np.array(VA),
             "v_s1": np.array(VS1), "v_s2": np.array(VS2)}
        # RTS-сглаживатель (З.2): обратный проход по записанным шагам фильтра;
        # первые n_warm шагов — прогрев (в сериях их нет) — отрезаем
        if self._show_rts and self.filter.record:
            try:
                xs = self._rts_smooth(self.filter.record)
                f["h_rts"] = xs[n_warm:, 0]
                f["v_rts"] = xs[n_warm:, 1]
            except Exception as e:
                self.statusBar().showMessage(f"RTS не посчитался: {e}")
            self.filter.record = None     # запись больше не нужна (память)
        self._file = f
        self._update_rts_info()           # факт по файлу: «шум ×K ниже» (А.6)
        self._recalc_file_smooth()          # зелёные серии (тем же окном N)
        self._file_set_curves()             # все кривые целиком
        self._play_i = len(f["t"]) - 1      # «стоим» в конце файла
        self._play_t = float(f["t"][-1])
        self._playing = False
        self._follow = True
        # вид: весь файл целиком, Y — по выбранному режиму (авто пересчитается сразу)
        self.plot_alt.setXRange(f["t"][0], f["t"][-1], padding=0.02)
        self.plot_vario.setXRange(f["t"][0], f["t"][-1], padding=0.02)
        self._y_next = {"alt": 0.0, "vario": 0.0}
        self._auto_y_tick(force_key="alt")
        self._auto_y_tick(force_key="vario")
        self._file_show_numbers()
        if cur_t is not None:             # вернуть курсор на то же время (А.3):
            self._set_cursor(cur_t)       # значения пересчитаются по новым сериям
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._slot_metric_next = 0.0      # метрику слотов пересчитать сразу
        self.statusBar().showMessage(
            f"Файл загружен целиком: {os.path.basename(path)} — "
            f"{f['t'][-1]:.1f} с, {len(f['t'])} точек. «Старт» — проиграть "
            f"(с курсора, если поставлен).")
        return True

    def _update_rts_info(self):
        """А.6: строка-факт про RTS по ЗАГРУЖЕННОМУ файлу: во сколько раз шум
        вариометра на статике (Покой по детектору) ниже у RTS, чем у фильтра."""
        f = self._file
        if f is None or f.get("v_rts") is None:
            self.lbl_rts_info.setText("")
            return
        rest = f["motion"] > 0
        n = min(len(f["v_filt"]), len(f["v_rts"]))
        rest = rest[:n]
        if rest.sum() < 100:
            self.lbl_rts_info.setText("RTS: статики в файле мало — K не посчитать")
            return
        s_f = float(np.std(f["v_filt"][:n][rest]))
        s_r = float(np.std(f["v_rts"][:n][rest]))
        if s_r <= 1e-9 or s_f <= 1e-9:
            self.lbl_rts_info.setText("")
            return
        if s_f >= s_r:
            fact = f"шум на статике ×{s_f / s_r:.2f} ниже фильтра (этот файл)"
        else:
            # с пакета 15 ZUPT прижимает фильтр на строгом покое почти к нулю —
            # на таких файлах статика у фильтра уже тише RTS (это не поломка RTS)
            fact = (f"на статике фильтр уже прижат ZUPT (тише RTS ×{s_r / s_f:.2f}) — "
                    f"RTS полезен на движении")
        self.lbl_rts_info.setText(
            f"RTS — сглаживание задним числом по всему файлу (использует будущее): "
            f"только для анализа записей, в реальном времени невозможно; {fact}.")

    @staticmethod
    def _rts_smooth(rec):
        """Сглаживатель Раух-Тунг-Стрибела по записи шагов фильтра
        (dt, x_pred, P_pred, x_post, P_post) → массив (n,3) сглаженных состояний.
        Обратный проход: C_k = P_k|k·F_{k+1}ᵀ·P_{k+1|k}⁻¹;
        x_k|N = x_k|k + C_k·(x_{k+1|N} − x_{k+1|k})."""
        n = len(rec)
        dts = np.array([r[0] for r in rec])
        xp = np.stack([r[1][:, 0] for r in rec])          # (n,3) прогнозы
        Pp = np.stack([r[2] for r in rec])                # (n,3,3)
        xu = np.stack([r[3][:, 0] for r in rec])          # (n,3) после коррекции
        Pu = np.stack([r[4] for r in rec])
        # F для каждого шага (зависит только от dt) — пачкой
        F = np.zeros((n, 3, 3))
        F[:, 0, 0] = 1.0; F[:, 1, 1] = 1.0; F[:, 2, 2] = 1.0
        F[:, 0, 1] = dts
        F[:, 0, 2] = -0.5 * dts * dts
        F[:, 1, 2] = -dts
        Pp_inv = np.linalg.inv(Pp)                        # батч-обращение (n,3,3)
        C = Pu[:-1] @ F[1:].transpose(0, 2, 1) @ Pp_inv[1:]
        xs = xu.copy()
        for k in range(n - 2, -1, -1):                    # рекурсия — по цепочке
            xs[k] = xu[k] + C[k] @ (xs[k + 1] - xp[k + 1])
        return xs

    def _on_rts_toggled(self):
        """Галочка «RTS (анализ)»: включение перепрогоняет файл (нужна запись
        шагов); выключение просто прячет кривые."""
        if self._loading:
            return
        self._show_rts = self.chk_rts.isChecked()
        save_config({"show_rts": self._show_rts})
        if self._file is not None and self.worker is None:
            if self._show_rts and self._file.get("v_rts") is None:
                path = self._file["path"]
                self._playing = False
                self._load_file_full(path)
            else:
                self._file_set_curves()

    def _recalc_file_smooth(self):
        """Пересчитать зелёные (сглаженные) серии файла под текущее окно N."""
        f = self._file
        if f is None:
            return
        N = float(self.spin_smooth.value())
        f["v_smooth"] = self._causal_gauss_series(f["v_filt"], f["t"], N)
        f["h_smooth"] = self._causal_gauss_series(f["h_filt"], f["t"], N)
        # сглаженная серия ВТОРОГО метода — для числа «сглаж.» в рамке (Г.1)
        v2 = f.get("v2")
        f["v2_smooth"] = (self._causal_gauss_series(v2, f["t"], N)
                          if v2 is not None and np.isfinite(v2).any() else None)

    def _file_set_curves(self):
        """Выставить кривые файла целиком (высота — с учётом «Ноля высоты»)."""
        f = self._file
        ref = self._h_ref or 0.0
        self.curve_h_raw.setData(f["t"], f["h_raw"] - ref)
        self.curve_h_filt.setData(f["t"], f["h_filt"] - ref)
        self.curve_v_raw.setData(f["t"], f["v_raw"])
        self.curve_v_filt.setData(f["t"], f["v_filt"])
        if f.get("v_smooth") is not None:
            self.curve_v_smooth.setData(f["t"], f["v_smooth"])
            self.curve_h_smooth.setData(f["t"], f["h_smooth"] - ref)
        else:
            self.curve_v_smooth.setData([], [])
            self.curve_h_smooth.setData([], [])
        v2 = f.get("v2")
        if v2 is not None and np.isfinite(v2).any() and self.chk_second.isChecked():
            self.curve_v2.setData(f["t"], v2, connect="finite")
        else:
            self.curve_v2.setData([], [])
        # RTS (З.2): кривые анализа
        if self._show_rts and f.get("v_rts") is not None:
            self.curve_v_rts.setData(f["t"], f["v_rts"])
            self.curve_h_rts.setData(f["t"], f["h_rts"] - ref)
        else:
            self.curve_v_rts.setData([], [])
            self.curve_h_rts.setData([], [])
        # слоты ручной настройки (А.5)
        for sk, curve in (("s1", self.curve_v_s1), ("s2", self.curve_v_s2)):
            vsk = f.get("v_" + sk)
            if (self.mode == "manual" and self._slots[sk]["show"]
                    and vsk is not None and np.isfinite(vsk).any()):
                curve.setData(f["t"], vsk, connect="finite")
            else:
                curve.setData([], [])

    def _file_show_numbers(self):
        """Большие числа/компас/индикатор из серий файла в точке _play_i."""
        f = self._file
        i = max(0, min(self._play_i, len(f["t"]) - 1))
        ref = self._h_ref or 0.0
        self.lbl_alt.setText(f"{f['h_filt'][i] - ref:+7.1f} м")
        self.lbl_alt_abs.setText(f"{f['h_filt'][i]:+7.1f} м")   # Е.2: всегда абсолютная
        self.lbl_vario.setText(f"{f['v_filt'][i]:+6.2f} м/с")
        self.lbl_vario.setStyleSheet(f"color: {self._sign_color(f['v_filt'][i])};")
        if f.get("v_smooth") is not None:
            N = float(self.spin_smooth.value())
            self.lbl_smooth_cap.setText(f"Сглаж. {N:g} с:")
            val = f["v_smooth"][i]
            self.lbl_vario_smooth.setText(f"{val:+6.2f} м/с")
            self.lbl_vario_smooth.setStyleSheet(f"color: {self._sign_color(val)};")
        else:
            self.lbl_smooth_cap.setText("Сглаж.:")
            self.lbl_vario_smooth.setText("—")
            self.lbl_vario_smooth.setStyleSheet(f"color: {self._neutral_fg};")
        # рамка второго метода (Г.1)
        v2 = f.get("v2")
        if v2 is not None and np.isfinite(v2[i]):
            self.frame2.setVisible(True)
            self.lbl2_vario.setText(f"{v2[i]:+6.2f} м/с")
            self.lbl2_vario.setStyleSheet(f"color: {self._sign_color(v2[i])};")
            s2 = f.get("v2_smooth")
            if s2 is not None and np.isfinite(s2[i]):
                self.lbl2_smooth.setText(f"{s2[i]:+6.2f} м/с")
                self.lbl2_smooth.setStyleSheet(f"color: {self._sign_color(s2[i])};")
            else:
                self.lbl2_smooth.setText("—")
        hd = f["head"][i]
        if np.isfinite(hd):
            self.compass.setHeading(float(hd))
            self.lbl_heading.setText(f"{hd:3.0f}° {cardinal_ru(float(hd))}")
        mo = f["motion"][i]
        self._set_motion_label("rest" if mo > 0 else ("osc" if mo < 0 else "dyn"))

    def _file_play_start(self):
        """Старт плеера: с курсора, с паузы или с начала."""
        f = self._file
        n = len(f["t"])
        if self._cursor_t is not None:
            self._play_i = int(np.argmin(np.abs(f["t"] - self._cursor_t)))
        elif self._play_i >= n - 1:
            self._play_i = 0                 # стояли в конце → с начала
        self._play_t = float(f["t"][self._play_i])
        self._playing = True
        self._play_wall = None
        self._follow = True
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.combo_source.setEnabled(False)
        self.btn_file.setEnabled(False)
        self.statusBar().showMessage(
            f"Проигрывание с {self._play_t:.1f} с (двойной клик по графику — "
            f"снова следить за временем).")

    def _file_pause(self):
        """Стоп = ПАУЗА: остаёмся на месте, «Старт» продолжит отсюда."""
        self._playing = False
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.combo_source.setEnabled(True)
        self._on_source_changed()
        self.statusBar().showMessage(
            f"Пауза на {self._play_t:.1f} с. «Старт» — продолжить, "
            f"«Запустить повторно» — с начала.")

    def _file_tick(self):
        """Тик плеера (из _redraw): двигаем время по wall-часам, обновляем числа и вид."""
        f = self._file
        if f is None or not self._playing:
            return
        import time as _time
        now = _time.perf_counter()
        if self._play_wall is not None:
            self._play_t += now - self._play_wall
        self._play_wall = now
        n = len(f["t"])
        while self._play_i < n - 1 and f["t"][self._play_i + 1] <= self._play_t:
            self._play_i += 1
        if self._play_i >= n - 1:
            self._playing = False
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self.combo_source.setEnabled(True)
            self._on_source_changed()
            self.statusBar().showMessage("Файл проигран до конца. «Старт» — с начала.")
        self._file_show_numbers()
        if self._follow:
            # вид следует за временем (окна просмотра у графиков свои)
            for plot, combo in ((self.plot_alt, self.combo_window_alt),
                                (self.plot_vario, self.combo_window_vario)):
                win = self._window_seconds(combo)
                x0 = f["t"][0] if win is None else max(f["t"][0], self._play_t - win)
                plot.setXRange(x0, self._play_t, padding=0.0)

    def _user_panned(self, *args):
        """Пользователь панорамирует мышью во время проигрывания — не дёргаем вид
        назад (слежение вернёт «Запустить повторно» или двойной клик)."""
        if self._file is not None and self._playing:
            self._follow = False

    # ------------------------------------------------------------------
    # СТАРТ / СТОП / СБРОС
    # ------------------------------------------------------------------
    def _make_source(self):
        """Создать объект источника по выбору пользователя."""
        kind = self.combo_source.currentText()
        if kind == "Симуляция":
            # реальное время (speed=1); сценарий проигрывается один раз и
            # аккуратно останавливается (loop=False) — без артефакта «склейки»
            return SimSource(dt=DT, speed=1.0, loop=False)
        elif kind.startswith("Поток"):
            url = self.edit_stream_url.text().strip() or "socket://127.0.0.1:5555"
            return StreamSource(url)
        else:
            # CSV: если файл не выбран — берём пример из data/samples/
            path = self.csv_path or os.path.join(SAMPLES_DIR, "sample_flight.csv")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"CSV-файл не найден: {path}\n"
                    f"Создайте пример командой:  python pc/make_sample_data.py")
            return CsvSource(path, speed=1.0, realtime=True)

    def _clear_data(self):
        """Полностью очистить данные и ЛИНИИ графиков (перед каждым запуском/повтором)."""
        R, Q = self._params_for_mode()
        self._rebuild_all_filters(R, Q)
        self._prev_t = None
        self._prev_h = None
        self._heading = None
        self._warned_no_cal = False
        self._noted_no_android = False
        self._clear_cursor()          # курсор-инспектор указывает в стёртую историю
        # сброс установки нуля акселерометра
        self._zero_done = False
        self._zero_accum = 0.0
        self._zero_count = 0
        self._zero_elapsed = 0.0
        # сброс детектора движения (новый фильтр создаётся с trust=1, bias живой)
        self._motion_state = "rest"
        self._rest_timer = 0.0
        self._trust = 1.0
        self._last_t = None
        self._mekf = None             # MEKF заново (инициализация в общем прогреве)
        self._jump_seen = 0           # прочитанных записей журнала скачков (Б.1)
        self._sensors_logged = False  # SENSORS/приоры — заново на новом подключении
        self._temp_logged = False
        # адаптация R/Q — с чистого листа (З.1 + пакет 15 А.4)
        self._ad_innov.clear()
        self._ad_d2.clear()
        self._ad_prev_alt = None
        self._ad_prev_alt2 = None
        self._ad_nis = 1.0
        self._ad_R_mult = 1.0
        self._ad_Q_mult = 1.0
        self._ad_hold_until = 0.0
        self._ad_next_R = 0.0
        self._ad_log_last = (1.0, 1.0)
        # детектор качания и ZUPT — заново (пакет 15, блок А)
        self._osc_win.clear()
        self._osc_sum = 0.0
        self._osc_sum2 = 0.0
        self._osc_until = -1.0
        self._zupt_w = None
        self._zupt_a = None
        self._zupt_timer = 0.0
        self._zupt_active = False
        self._baro_rows = 1.0
        self._baro_row_count = 0
        self._baro_prev_val = None
        # watchdog фильтра — заново
        self._wd_win.clear()
        self._wd_bad = 0.0
        self._wd_events = []
        # сброс посева/прогрева фильтра и пульса индикатора
        self._t_first = None
        self._warming = False
        self._dot_seen = 0
        self._dot_flash = 0.0
        self._head_vec = None         # сглаживание компаса заново
        for b in (self.buf_t, self.buf_h_raw, self.buf_h_filt,
                  self.buf_v_raw, self.buf_v_filt,
                  self.buf_h_smooth, self.buf_v_smooth, self.buf_v2,
                  self.buf_v_auto, self.buf_v_s1, self.buf_v_s2):
            b.clear()
        for c in (self.curve_h_raw, self.curve_h_filt,
                  self.curve_v_raw, self.curve_v_filt,
                  self.curve_h_smooth, self.curve_v_smooth, self.curve_v2,
                  self.curve_h_rts, self.curve_v_rts,
                  self.curve_v_s1, self.curve_v_s2):
            c.setData([], [])
        self._dirty = False

    def _load_mag_calibration(self):
        """Из pc/calibration.json (v2, пакет 14 Б.2): склонение D, ОБЕ секции
        магнитометра (mag_raw / mag_android — по источнику поля) и калибровка
        АКСЕЛЕРОМЕТРА (offset/scales + эталон g). Старые файлы v1 мигрируются
        на чтении (device_calibration.normalize)."""
        self._decl = 0.0
        self._mag_cal = {"raw": None, "android": None}
        self._acc_off = np.zeros(3)
        self._acc_scl = np.ones(3)
        self._g_ref = 9.81
        self._gyro_bias = np.zeros(3)
        self._mag_F_ref = None
        d = devcal.load(DEVICE_CALIB_PATH)
        if not d:
            return
        if d.get("declination_deg") is not None:
            try:
                self._decl = float(d["declination_deg"])
            except (TypeError, ValueError):
                pass
        for src in ("raw", "android"):
            self._mag_cal[src] = devcal.mag_section(d, src)
        # эталон |B| для гейта AHRS: секция активного источника, иначе любая
        src_now = self._compass_source()
        for sec in (self._mag_cal.get(src_now), self._mag_cal.get("raw"),
                    self._mag_cal.get("android")):
            if sec is not None and sec.get("F"):
                self._mag_F_ref = sec["F"]
                break
        acc = d.get("accel") or {}
        if "offset" in acc and "scales" in acc:
            self._acc_off = np.asarray(acc["offset"], dtype=float)
            self._acc_scl = np.asarray(acc["scales"], dtype=float)
        if acc.get("target_g"):
            self._g_ref = float(acc["target_g"])
        if d.get("gyro_bias") is not None:
            self._gyro_bias = np.asarray(d["gyro_bias"], dtype=float)

    def _compass_source(self) -> str:
        """Источник поля компаса из compass_use: "raw" | "android"."""
        return "android" if self._compass_use.endswith("@android") else "raw"

    def _compass_method(self) -> str:
        """Метод компаса из compass_use: "ransac" | "live"."""
        return "live" if self._compass_use.startswith("live@") else "ransac"

    def set_compass_use(self, mode: str):
        """Б.2: «Компас использует» = метод@источник (переключает вкладка
        «Калибровка», выбор хранится в config.json → compass_use)."""
        method, _, src = mode.partition("@")
        if method not in ("ransac", "live") or src not in ("raw", "android"):
            return
        self._compass_use = mode
        # эталон |B| гейта AHRS следует за источником (MEKF не пересоздаём —
        # ориентация непрерывна, это только параметр гейта)
        sec = self._mag_cal.get(src) or self._mag_cal.get(
            "raw" if src == "android" else "android")
        if sec is not None and sec.get("F"):
            self._mag_F_ref = sec["F"]
            if self._mekf is not None:
                self._mekf.mag_F_ref = self._mag_F_ref
        self._noted_no_android = False
        self._update_calib_indicator_live(force=True)

    def _live_cal_for(self, source: str):
        """Живая калибровка live-EKF для компаса: только если ВЫБРАН метод live
        с этим источником и сбор на «Калибровке» реально идёт. Иначе None
        (выбор live без сбора невозможен по построению — пункты серые)."""
        if (self._compass_method() != "live" or self._compass_source() != source
                or self.live_mag_provider is None):
            return None
        return self.live_mag_provider(source)

    def _update_calib_indicator_live(self, force: bool = False):
        """Строка под компасом при методе Live-EKF: живой остаток подстройки.
        При методе RANSAC строку ведёт refresh_calib_indicator."""
        if self._compass_method() != "live":
            if self._calib_ind_live_txt is not None or force:
                self._calib_ind_live_txt = None
                self.refresh_calib_indicator()
            return
        src = self._compass_source()
        lc = self.live_mag_provider(src) if self.live_mag_provider else None
        if lc is not None:
            txt = (f"Компас: Live-EKF@{src} · остаток {lc['residual_pct']:.1f}%")
            color = "#2c7a2c" if lc["residual_pct"] < 5.0 else "#c09010"
        else:
            txt = f"Компас: Live-EKF@{src} — сбор остановлен (переключите метод)"
            color = "#c09010"
        if txt != self._calib_ind_live_txt:
            self._calib_ind_live_txt = txt
            self.lbl_calib_status.setText(txt)
            self.lbl_calib_status.setStyleSheet(f"font-size: 11px; color: {color};")

    def refresh_calib_indicator(self):
        """Перечитать активную калибровку прибора и обновить индикатор под
        компасом: «Компас: RANSAC@android · остаток 3.3%» (пакет 14, Б.2).
        Вызывается при запуске и из main.py после «Сохранить»/«Применить»."""
        self._load_mag_calibration()
        src = self._compass_source()
        sec = self._mag_cal.get(src)
        if sec is None:
            if src == "android":
                # Android-поле пригодно и БЕЗ тонкой калибровки (ОС уже сняла железо)
                self.lbl_calib_status.setText(
                    "Компас: RANSAC@android — тонкой калибровки нет (поле ОС как есть)")
                self.lbl_calib_status.setStyleSheet("font-size: 11px; color: #c09010;")
            else:
                self.lbl_calib_status.setText(
                    "Компас: RANSAC@raw — НЕТ калибровки (сырые данные)")
                self.lbl_calib_status.setStyleSheet(
                    "font-size: 11px; color: #c0392b; font-weight: bold;")
            return
        method = {"ellipsoid": "RANSAC", "ekf": "EKF"}.get(sec.get("model"),
                                                           sec.get("model") or "RANSAC")
        created = sec.get("created") or "?"
        try:
            created = datetime.strptime(created, "%Y-%m-%d %H:%M:%S").strftime("%d.%m %H:%M")
        except (ValueError, TypeError):
            pass
        res = sec.get("residual_pct")
        res_txt = f" · остаток {res:.1f}%" if res is not None else ""
        self.lbl_calib_status.setText(
            f"Компас: {method}@{src} от {created}{res_txt}")
        color = "#2c7a2c" if (res is not None and res < 5.0) else "#c09010"
        self.lbl_calib_status.setStyleSheet(f"font-size: 11px; color: {color};")

    def rerun(self):
        """«Запустить повторно» — с начала. Файл: плеер на 0 + слежение; live —
        перезапуск источника. При сбое НЕ роняем приложение."""
        try:
            if (self.combo_source.currentText() == "CSV-файл"
                    and self._file is not None and self.worker is None):
                self._playing = False
                self._clear_cursor()
                self._play_i = 0
                self._follow = True
                self._file_play_start()
                return
            self.stop()      # остановить и отцепить старый поток (см. stop())
            self.start()     # запустить заново (start сам чистит буферы и фильтр)
        except Exception as e:
            self._show_error(f"Не удалось перезапустить: {e}")

    def add_sound_compact(self, widget: QtWidgets.QWidget):
        """Вставить компактный дубль звука (иконка+бар, блок Ж.5) в строку
        «Просмотр» — рядом с «Экспорт PNG». Вызывает main.py."""
        self._sound_slot.addWidget(widget)

    def set_sound_source(self, src: str):
        """А.2 (пакет 14): источник звука — "vario" (фильтр) | "smooth"
        (Сглаж. N с). Переключает компактный дубль на «Вариометре»
        (с пакета 15 В.1 селектора на вкладке «Звук» больше нет)."""
        if src in ("vario", "smooth") and src != self.sound_source:
            self.sound_source = src
            save_config({"sound_source": src})

    def sound_source_desc(self) -> str:
        """В.1 (пакет 15): человеческое описание ФАКТИЧЕСКОГО источника звука
        для подписи на вкладке «Звук»: «Ручной 2 · сглаж. 0.7 с». Звук всегда
        идёт от ОСНОВНОГО фильтра; если его R/Q совпадают со слотом — называем
        слот по имени."""
        if self.mode == "auto":
            base = "Авто (адаптивный)" if self._adaptive_on else "Авто"
        else:
            base = "Ручной"
            for sk, title in (("s1", "Ручной 1"), ("s2", "Ручной 2")):
                if (abs(self._slots[sk]["R"]
                        - float(self.manual_params["R"])) < 1e-9
                        and abs(self._slots[sk]["sigma_accel"]
                                - float(self.manual_params["sigma_accel"])) < 1e-9):
                    base = title
                    break
        n = float(self.spin_smooth.value())
        ser = (f"сглаж. {n:g} с" if (self.sound_source == "smooth" and n > 0)
               else "фильтр")
        return f"{base} · {ser}"

    def connect_stream_standby(self) -> str:
        """Д.3: открыть поток для «Записей» БЕЗ проигрывания (менеджеру записей
        нужен живой канал для LIST/GET/DEL). Если поток уже открыт — используется
        он. «Старт» на «Вариометре» подключит проигрывание к этому же каналу."""
        if self.worker is not None and getattr(self.worker.source, "live", False):
            return "Поток уже открыт — используется он."
        if self.worker is not None:
            return "Источник занят (идёт чтение) — остановите его на «Вариометре»."
        url = self.edit_stream_url.text().strip() or "socket://127.0.0.1:5555"
        w = SourceWorker(StreamSource(url))
        w._standby = True                 # данные не играем (sampleReady не подключён)
        w.errorOccurred.connect(self._show_error)
        w.finished.connect(lambda wk=w: self._worker_done(wk))
        self.worker = w
        w.start()
        self.combo_source.setCurrentText("Поток (Bluetooth/симулятор)")
        self._on_source_changed()
        self.btn_stop.setEnabled(True)
        self.statusBar().showMessage(
            f"Подключено к телефону БЕЗ проигрывания: {url}. "
            f"«Старт» — начать проигрывание по этому же каналу.")
        return f"Подключаюсь: {url} (без проигрывания). Нажмите «Обновить список»."

    def start(self):
        if self.worker is not None:
            # standby-подключение из «Записей» (Д.3): подключить проигрывание
            # к УЖЕ открытому каналу (второе подключение телефон не примет)
            if getattr(self.worker, "_standby", False):
                self._file = None
                self._playing = False
                self._clear_data()
                self._load_mag_calibration()
                self._zero_N = float(self.spin_calib.value())
                self.worker._standby = False
                self.worker.sampleReady.connect(self._on_sample)
                self._view_refresh = True
                self.btn_start.setEnabled(False)
                self.btn_stop.setEnabled(True)
                self.combo_source.setEnabled(False)
                self.btn_file.setEnabled(False)
                self.statusBar().showMessage(
                    "Проигрывание подключено к открытому каналу телефона.")
            return  # уже запущено

        # ФАЙЛОВЫЙ РЕЖИМ (CSV): весь файл уже прогнан пакетно → «Старт» = плеер
        # (с курсора / с паузы / с начала). Live-воркер не нужен.
        if self.combo_source.currentText() == "CSV-файл":
            if self._file is None:
                path = self.csv_path or os.path.join(SAMPLES_DIR, "sample_flight.csv")
                if not os.path.exists(path):
                    self._show_error(f"CSV-файл не найден: {path}\n"
                                     f"Создайте пример:  python pc/make_sample_data.py")
                    return
                if not self._load_file_full(path):
                    return
            if self._playing:
                return
            self._file_play_start()
            return

        try:
            source = self._make_source()
        except Exception as e:
            self._show_error(str(e))
            return

        self._file = None          # уходим из файлового режима в live
        self._playing = False
        # БАГ нескольких кривых: перед новым запуском полностью чистим данные и линии
        self._clear_data()
        # подгрузить склонение D для компаса (если калибровка прибора сохранена)
        self._load_mag_calibration()
        # окно установки нуля акселерометра берём из поля «Установка нуля акселер.»
        self._zero_N = float(self.spin_calib.value())

        # запустить поток чтения
        self.worker = SourceWorker(source)
        self.worker.sampleReady.connect(self._on_sample)
        self.worker.errorOccurred.connect(self._show_error)
        # привязываем сигнал к КОНКРЕТНОМУ потоку: если он завершится уже ПОСЛЕ
        # перезапуска, его «хвостовой» finished не обнулит НОВЫЙ поток (это и роняло
        # «Запустить повторно»).
        self.worker.finished.connect(lambda wk=self.worker: self._worker_done(wk))
        self.worker.start()
        self._view_refresh = True  # при старте вернуть авто-масштаб и применить окно

        # обновить кнопки
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.combo_source.setEnabled(False)
        self.btn_file.setEnabled(False)
        self.statusBar().showMessage(f"Идёт чтение: {source.name}")

    def stop(self):
        """Файловый режим: Стоп = ПАУЗА (не сброс). Live: остановить чтение.
        Сразу снимаем ссылку на поток и ОТЦЕПЛЯЕМ его сигналы, чтобы запоздавший
        finished старого потока не сбросил новый."""
        if self._file is not None and self.worker is None:
            self._file_pause()
            return
        w = self.worker
        self.worker = None
        if w is not None:
            try:
                w.sampleReady.disconnect()
                w.errorOccurred.disconnect()
                w.finished.disconnect()
            except (TypeError, RuntimeError):
                pass            # уже отцеплены — не страшно
            try:
                w.stop()
                w.wait(2000)    # дать потоку выйти из цикла чтения (до 2 c)
            except RuntimeError:
                pass
            w.deleteLater()     # удалить QThread безопасно, после завершения
        self._reset_after_run()

    def _worker_done(self, wk):
        """Слот finished КОНКРЕТНОГО потока. Если он уже не текущий (был перезапуск) —
        игнорируем, иначе сбросили бы только что запущенный новый поток."""
        if wk is not self.worker:
            return
        self.worker = None
        self._reset_after_run()

    def _on_worker_finished(self):
        """Совместимый обработчик (используется в --selftest): погасить поток и сбросить UI."""
        self.worker = None
        self._reset_after_run()

    def _reset_after_run(self):
        """Вернуть кнопки/вид в исходное состояние (БЕЗ обнуления ссылки на поток)."""
        self._view_refresh = True  # один раз подогнать рамки под итоговые данные
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.combo_source.setEnabled(True)
        self._on_source_changed()
        if self.statusBar().currentMessage().startswith("Идёт чтение"):
            self.statusBar().showMessage("Готово (источник завершён или остановлен). «Старт» — повторить.")

    def reset(self):
        """Сбросить всё: остановить чтение/плеер, очистить графики, пересоздать фильтр."""
        self._playing = False
        self._file = None
        self.stop()
        self._clear_data()
        self._view_refresh = False
        self._redraw()
        self.lbl_alt.setText("—")
        self.lbl_alt_abs.setText("—")
        self.lbl_vario.setText("—")
        self.lbl_vario.setStyleSheet(f"color: {self._neutral_fg};")
        self.lbl_heading.setText("—")
        self.compass.setHeading(None)
        # сброс «Ноля высоты»
        self._h_ref = None
        self.btn_zero_alt.setText("Ноль высоты")
        self.lbl_zero_note.setText("")
        self.statusBar().showMessage("Сброшено.")

    # ------------------------------------------------------------------
    # ДЕТЕКТОР ПОКОЙ/ДВИЖЕНИЕ/КАЧАНИЕ (гистерезис) + доверие акселю + ZUPT
    # ------------------------------------------------------------------
    def _update_motion(self, gyro_mag: float, acc_dev: float, gz: float,
                       t: float, dt: float):
        """
        ПОКОЙ:    |гироскоп| < gyro_rest И ||a|−g| < acc_rest устойчиво hold_sec.
        ДИНАМИКА: |гироскоп| > gyro_dyn ИЛИ ||a|−g| > acc_dyn (сразу).
        КАЧАНИЕ (пакет 15, А.3): скользящее за osc.win_sec std(gz) > gz_std при
        |среднее| < mean_ratio·std (знакопеременное размахивание) — с удержанием
        hold_sec, чтобы не мигать в нулях фазы. В качании: R баро × k_baro
        (порт «дышит»), доверие акселю НЕ ослабляется (MEKF вращение держит),
        bias заморожен, ZUPT выключен.
        ZUPT (А.1): сглаженные (EMA ~0.1 с) |ω| < gyro_max И ||f|−g| < acc_max
        устойчиво hold_sec → фильтру ставится zupt_r (измерение v=0).
        """
        c = self._motion_cfg
        # --- СТРОГИЙ ПОКОЙ (для ZUPT и досрочного снятия «Качания»): EMA ~0.1 с
        # по |ω| и ||f|−g| + накопление устойчивости ---
        z = self._zupt_cfg
        a_ema = 1.0 - exp(-dt / 0.1)
        self._zupt_w = (gyro_mag if self._zupt_w is None
                        else self._zupt_w + (gyro_mag - self._zupt_w) * a_ema)
        self._zupt_a = (acc_dev if self._zupt_a is None
                        else self._zupt_a + (acc_dev - self._zupt_a) * a_ema)
        strict = (self._zupt_w < float(z.get("gyro_max", 0.05))
                  and self._zupt_a < float(z.get("acc_max", 0.15)))
        self._zupt_timer = self._zupt_timer + dt if strict else 0.0
        strict_rest = self._zupt_timer >= float(z.get("hold_sec", 0.3))
        # --- качание: окно gz с суммами (O(1) на сэмпл) ---
        osc = self._osc_cfg
        osc_now = False
        if osc.get("enabled", True):
            self._osc_win.append((t, gz))
            self._osc_sum += gz
            self._osc_sum2 += gz * gz
            win = float(osc.get("win_sec", 1.0))
            while self._osc_win and t - self._osc_win[0][0] > win:
                _t0, g0 = self._osc_win.popleft()
                self._osc_sum -= g0
                self._osc_sum2 -= g0 * g0
            n = len(self._osc_win)
            if n >= 50:
                mean = self._osc_sum / n
                var = max(0.0, self._osc_sum2 / n - mean * mean)
                std = sqrt(var)
                if (std > float(osc.get("gz_std", 0.8))
                        and abs(mean) < float(osc.get("mean_ratio", 0.5)) * std):
                    self._osc_until = t + float(osc.get("hold_sec", 0.5))
            if strict_rest:
                # телефон строго неподвижен ≥0.3 с — качание закончилось,
                # хвост окна/удержания не должен держать R×k_baro и глушить ZUPT
                self._osc_until = -1.0
            osc_now = t <= self._osc_until
        # --- покой/движение (гистерезис, как раньше) ---
        if gyro_mag > c["gyro_dyn"] or acc_dev > c["acc_dyn"]:
            base_state = "dyn"
            self._rest_timer = 0.0
        elif gyro_mag < c["gyro_rest"] and acc_dev < c["acc_rest"]:
            self._rest_timer += dt
            base_state = ("rest" if self._rest_timer >= c["hold_sec"]
                          else ("dyn" if self._motion_state != "rest" else "rest"))
        else:
            self._rest_timer = 0.0        # между порогами: держим текущее состояние
            base_state = "rest" if self._motion_state == "rest" else "dyn"
        self._motion_state = "osc" if osc_now else base_state
        # --- доверие акселю: на КАЧАНИИ НЕ ослабляем (ТЗ А.3) — маленький Q
        # делает v МЕДЛЕННЫМ трекером демпфированного (×k_baro) баро; k_dyn
        # здесь накачал бы Q и свёл демпфирование на нет (замерено: с k_dyn
        # v ходил до ±1.45, с trust=1 — в планке). Работает ТОЛЬКО в связке с
        # занулённым входом акселя (см. _process_sample: a_vert на 18.9 рад/с
        # лжёт до ±10 м/с² в обоих режимах — интегрировать его нельзя). ---
        target = c["k_dyn"] if self._motion_state == "dyn" else 1.0
        alpha = min(1.0, dt / max(c["trans_sec"], 1e-3))
        self._trust += (target - self._trust) * alpha
        self.filter.accel_trust = self._trust
        self.filter.bias_frozen = (self._motion_state != "rest")
        # множитель R баро на качании. ВАЖНО: строки идут в ~fs/25 раз чаще
        # настоящих 25-Гц отсчётов баро (значения повторяются) — статистически
        # это завышает вес баро в fs/25 раз, и «×k_baro на строку» почти не
        # демпфирует. Поэтому k_baro масштабируется на фактическое число строк
        # на один отсчёт баро (живой замер _baro_rows) — получается честное
        # «R×k_baro на 25-Гц отсчёт», как в ТЗ.
        if self._motion_state == "osc":
            self.filter.R_baro_mult = (float(osc.get("k_baro", 8.0))
                                       * max(1.0, self._baro_rows))
        else:
            self.filter.R_baro_mult = 1.0
        # --- ZUPT: строгий покой (посчитан в начале; «Качание» при строгом
        # покое уже снято досрочно, в живом качании EMA |ω| за 0.1 с в нулях
        # фазы не проседает — ложных срабатываний нет) ---
        self._zupt_active = (bool(z.get("enabled", True)) and strict_rest
                             and self._motion_state != "osc")
        self.filter.zupt_r = (float(z.get("sigma_v", 0.02)) ** 2
                              if self._zupt_active else None)

    # ------------------------------------------------------------------
    # АДАПТИВНЫЕ R/Q (Фаза 5 / блок З.1; R переписан в пакете 15, А.4)
    # ------------------------------------------------------------------
    def _adapt_targets(self):
        """Кому применять адаптивные R̂/Q̂ и по кому мерить NIS (пакет 15, Ж.1):
        в Авто — основной фильтр (+тень 2-го метода); эталонный фильтр «Авто»
        адаптируется ВСЕГДА (в Ручном режиме он шагает и даёт живую строку
        эталона). Возвращает (nis_источник, [фильтры])."""
        if self.mode == "auto":
            targets = [f for f in (self.filter, self.filter2, self.filter_auto)
                       if f is not None]
            return self.filter, targets
        # Ручной: адаптируется только эталон «Авто» (основной — руками)
        return self.filter_auto, ([self.filter_auto]
                                  if self.filter_auto is not None else [])

    def _adapt_rq(self, t: float, h_baro: float, dt: float):
        """R̂ (пакет 15, А.4) — НЕПРЕРЫВНО по вторым разностям сырой высоты на
        настоящих 25-Гц отсчётах баро: d = z_i − 2z_{i−1} + z_{i−2} (только по
        СМЕНИВШИМСЯ значениям — строки 417 Гц повторяют баро ~17 раз!);
        при белом шуме Var(d) = 6σ²; робастно σ̂² = (1.4826·MAD)²/6; вклад
        реального ускорения a·Δt² ≈ 0.005 м пренебрежим → работает и в полёте.
        Q̂ — множитель по NIS (EMA, τ~5 с). Оба ×[0.3…3] от паспортных.
        Замирает только после watchdog (5 с)."""
        src, targets = self._adapt_targets()
        if src is None or not targets:
            return
        # --- сбор вторых разностей (по сменившимся значениям баро) ---
        if self._adapt_r_mode == "d2":
            if self._ad_prev_alt is None or h_baro != self._ad_prev_alt:
                if self._ad_prev_alt is not None and self._ad_prev_alt2 is not None:
                    d = h_baro - 2.0 * self._ad_prev_alt + self._ad_prev_alt2
                    self._ad_d2.append((t, d))
                self._ad_prev_alt2 = self._ad_prev_alt
                self._ad_prev_alt = h_baro
            while self._ad_d2 and t - self._ad_d2[0][0] > 10.0:
                self._ad_d2.popleft()
        if t < self._ad_hold_until:
            return                                 # watchdog: адаптация замерла
        y, S = src.last_innov, src.last_S
        # --- Q по NIS: фильтр стабильно «удивляется» (NIS>1) → поднять Q.
        # На КАЧАНИИ вход NIS замирает: инновации там большие ПО ПОСТРОЕНИЮ
        # (баро специально демпфирован ×k_baro) — накачка Q вернула бы фильтру
        # скорость и свела демпфирование на нет (замер: |v| на качании
        # 1.14→0.4 после заморозки) ---
        a5 = min(1.0, dt / 5.0)
        if self._motion_state != "osc":
            nis = (y * y) / max(S, 1e-9)
            self._ad_nis += (nis - self._ad_nis) * a5
        q_target = min(3.0, max(0.3, sqrt(max(self._ad_nis, 1e-6))))
        self._ad_Q_mult += (q_target - self._ad_Q_mult) * a5
        # --- R: v2 по вторым разностям (или legacy пакета 14 по инновациям) ---
        if self._adapt_r_mode == "d2":
            if t >= self._ad_next_R and len(self._ad_d2) >= 25:
                self._ad_next_R = t + 0.5          # MAD — раз в полсекунды
                dd = np.fromiter((d for _, d in self._ad_d2), dtype=float)
                mad = float(np.median(np.abs(dd - np.median(dd))))
                sigma2 = (1.4826 * mad) ** 2 / 6.0
                r_target = min(3.0, max(0.3, sigma2 / R_DEFAULT))
                # τ~10 с при пересчёте раз в 0.5 с
                self._ad_R_mult += (r_target - self._ad_R_mult) * min(1.0, 0.5 / 10.0)
        else:                                      # legacy (пакет 14): только Покой
            if self._motion_state == "rest" or not self.motion_enabled:
                self._ad_innov.append((t, y * y))
            while self._ad_innov and t - self._ad_innov[0][0] > 10.0:
                self._ad_innov.popleft()
            if t >= self._ad_next_R and len(self._ad_innov) >= 50:
                self._ad_next_R = t + 0.5
                med = float(np.median([q for _, q in self._ad_innov]))
                hph = max(0.0, S - float(src.R[0, 0]))
                R_hat = med / 0.4549 - hph
                r_target = min(3.0, max(0.3, R_hat / R_DEFAULT))
                self._ad_R_mult += (r_target - self._ad_R_mult) * min(1.0, 0.5 / 5.0)
        for flt in targets:
            flt.sigma_accel = SIGMA_ACCEL_DEFAULT * self._ad_Q_mult
            flt.R[0, 0] = R_DEFAULT * self._ad_R_mult
        # ЛОГ адаптации (пакет 14, Ж): пишем, когда множитель заметно уехал
        lr, lq = self._ad_log_last
        if abs(self._ad_R_mult - lr) > 0.15 or abs(self._ad_Q_mult - lq) > 0.15:
            self._ad_log_last = (self._ad_R_mult, self._ad_Q_mult)
            crashlog.log_event(
                "adapt", f"t={t:.1f} с: R×{self._ad_R_mult:.2f} "
                f"(R̂={R_DEFAULT * self._ad_R_mult:.4f} м²), Q×{self._ad_Q_mult:.2f} "
                f"(Q̂={SIGMA_ACCEL_DEFAULT * self._ad_Q_mult:.3f} м/с²)")

    # ------------------------------------------------------------------
    # ОБЩИЙ ПАЙПЛАЙН ОДНОГО ЗАМЕРА (live-поток И пакетный прогон файла)
    # ------------------------------------------------------------------
    def _process_sample(self, s, quiet: bool = False):
        """Один замер через ВЕСЬ пайплайн (dt → IMU → детектор → ноль → посев →
        фильтр → сырой варио → компас со сглаживанием). Возвращает dict; ключ
        'warming' = True, пока идёт прогрев (точки не выводить). quiet=True —
        пакетный прогон файла: без сообщений в statusBar."""
        t, a, h_baro = s.t, s.a_world_vertical, s.h_baro

        # РЕАЛЬНЫЙ шаг времени (записи телефона ~400+ Гц, симуляция 50 Гц)
        dt = None
        if self._last_t is not None:
            dt = t - self._last_t
            if not (0.0 < dt < 0.5):
                dt = None            # дыра/склейка в метках — берём прежний шаг
        self._last_t = t
        dt_eff = dt if dt is not None else DT

        # ВЕЛИЧИНЫ IMU для детектора: |гироскоп| и отклонение |a_калибр| от g
        gyro_mag = 0.0
        gz_now = 0.0
        if s.gyro3 is not None:
            gx, gy, gz = s.gyro3
            gz_now = gz
            gyro_mag = sqrt(gx * gx + gy * gy + gz * gz)
        acc_dev = 0.0
        a_cal_norm = None
        if s.accel3 is not None:
            ac = (np.asarray(s.accel3, dtype=float) - self._acc_off) * self._acc_scl
            a_cal_norm = float(np.linalg.norm(ac))
            acc_dev = abs(a_cal_norm - self._g_ref)

        # ЧИСЛО СТРОК на один настоящий отсчёт баро (живой замер, EMA):
        # нужно детектору качания, чтобы «R×k_baro» был честным на 25-Гц
        # отсчёт, а не на повторяющуюся строку (пакет 15, А.3)
        self._baro_row_count += 1
        if self._baro_prev_val is None or h_baro != self._baro_prev_val:
            if self._baro_prev_val is not None:
                self._baro_rows += (self._baro_row_count - self._baro_rows) * 0.05
            self._baro_prev_val = h_baro
            self._baro_row_count = 0

        # ДЕТЕКТОР ПОКОЙ/ДВИЖЕНИЕ/КАЧАНИЕ → доверие акселю, bias, R баро, ZUPT
        # (обновляем ДО вычисления вертикального ускорения: его состояние
        # масштабирует R аксель-коррекции MEKF на этом же сэмпле)
        if self.motion_enabled:
            self._update_motion(gyro_mag, acc_dev, gz_now, t, dt_eff)

        # ПОЛЕ ДЛЯ КОМПАСА/AHRS (пакет 14, Б.2): источник и метод — из селектора
        # «Компас использует» (метод@источник). К полю применяется калибровка
        # ЕГО ЖЕ источника (секции mag_raw / mag_android калибровки v2) —
        # несовпадение источников исключено ПО ПОСТРОЕНИЮ.
        m_comp = None
        comp_src = self._compass_source()
        ma = getattr(s, "mag3a", None)
        if (comp_src == "android" and ma is None and s.mag3 is not None
                and getattr(s, "mag_raw", False) and not self._noted_no_android):
            # в данных нет Android-поля (старый файл / поток v3) →
            # честно падаем на сырое, с пометкой (не молча)
            self._noted_no_android = True
            if not quiet:
                self.statusBar().showMessage(
                    "Компас: в данных нет Android-поля (запись v3) — "
                    "используется сырое поле + его калибровка.")
        if comp_src == "android" and ma is not None:
            v = np.asarray(ma, dtype=float)
            lc = self._live_cal_for("android")
            sec = self._mag_cal.get("android")
            if lc is not None:
                v = lc["M"] @ (v - lc["V"])      # live-EKF подстройка
            elif sec is not None:
                v = sec["M"] @ (v - sec["V"])    # «тонкая» калибровка поверх ОС
            m_comp = (float(v[0]), float(v[1]), float(v[2]))
        elif s.mag3 is not None:
            mcx, mcy, mcz = s.mag3
            if getattr(s, "mag_raw", False):
                lc = self._live_cal_for("raw")
                sec = self._mag_cal.get("raw")
                if lc is not None:               # live-EKF подстройка
                    vv = lc["M"] @ (np.array([mcx, mcy, mcz], dtype=float)
                                    - lc["V"])
                    mcx, mcy, mcz = float(vv[0]), float(vv[1]), float(vv[2])
                elif sec is not None:
                    vv = sec["M"] @ (np.array([mcx, mcy, mcz], dtype=float)
                                     - sec["V"])
                    mcx, mcy, mcz = float(vv[0]), float(vv[1]), float(vv[2])
                elif not self._warned_no_cal:
                    self._warned_no_cal = True
                    if not quiet:
                        self.statusBar().showMessage(
                            "Нет калибровки СЫРОГО магнитометра (секция mag_raw) — курс по "
                            "сырому полю (возможен дрейф). Сохраните калибровку прибора.")
            m_comp = (mcx, mcy, mcz)

        # ВЕРТИКАЛЬНОЕ УСКОРЕНИЕ для фильтра. ОРИЕНТАЦИЯ (MEKF/AHRS) СЧИТАЕТСЯ
        # ВСЕГДА (пакет 14, Б.1): один MEKF на оба режима, переключатель «Верт.
        # ускорение» выбирает лишь, ЧТО идёт входом фильтра высоты:
        #   "mekf"   — проекция через ориентацию a = (R_wb·f_b)_z − g;
        #   "scalar" — прежнее приближение a = |a_калибр| − g.
        # Курс (AHRS) от переключателя больше НЕ зависит и не прыгает; второй
        # метод — просто другой выход того же расчёта (отдельный MEKF не нужен).
        # Пока MEKF инициализируется (~1.5 с покоя, внутри общего прогрева) —
        # он возвращает None, на это время подставляется скаляр.
        a2 = None                       # вертикальное ускорение ВТОРОГО метода (Г.1)
        if getattr(s, "imu_raw", False) and self.accel_input_enabled and a_cal_norm is not None:
            a_scalar = a_cal_norm - self._g_ref
            gb = self._gyro_bias
            w_cal = ((s.gyro3[0] - gb[0], s.gyro3[1] - gb[1], s.gyro3[2] - gb[2])
                     if s.gyro3 is not None else (0.0, 0.0, 0.0))
            in_motion = (self._motion_state != "rest") if self.motion_enabled else None
            if self._mekf is None:
                self._mekf = make_mekf(self._mekf_cfg, self._g_ref,
                                       mag_F_ref=self._mag_F_ref)
            a_mekf = self._mekf.step(t, ac, w_cal, in_motion=in_motion,
                                     mag_b=m_comp)
            # ЖУРНАЛ СКАЧКОВ КУРСА (пакет 15, Б.1) → краш-лог с причиной
            jl = self._mekf.jump_log
            if len(jl) > self._jump_seen:
                for (tj, dh, dg, wmag, cause) in jl[self._jump_seen:]:
                    crashlog.log_event(
                        "compass", f"скачок курса t={tj:.1f} с: Δкурс {dh:+.0f}° "
                        f"при Δгиро {dg:+.0f}° за окно 1 с (|ω|={wmag:.2f} рад/с), "
                        f"причина: {cause}")
                self._jump_seen = len(jl)
            if self._va_mode == "mekf":
                a = a_mekf if a_mekf is not None else a_scalar
                a2 = a_scalar                       # тень = скалярное |a|−g
            else:
                a = a_scalar
                # тень = MEKF; пока он греется — скаляр (иначе фильтр тени
                # пропускал бы шаги и его переходный процесс портил сравнение)
                a2 = a_mekf if a_mekf is not None else a_scalar
            # КАЧАНИЕ (пакет 15, А.3): вертикальное ускорение на размахивании
            # ЛЖЁТ в обоих режимах (замер на 23-21-13: MEKF до ±10 м/с² —
            # рассинхрон выборок гиро/акселя на 18.9 рад/с; скаляр до ±37 —
            # ω²r по построению), а вход a интегрируется в v НАПРЯМУЮ, мимо
            # весов Q/R. Правило проекта: «детектор глушит МУСОР акселя, а не
            # реальный ход баро» — на качании вход акселя зануляется, реальный
            # ход телефона приходит через демпфированный (×k_baro) барометр.
            if self._motion_state == "osc":
                a = 0.0
                a2 = 0.0

        # УСТАНОВКА НУЛЯ АКСЕЛЕРОМЕТРА (только в покое)
        if not self._zero_done and self._zero_N > 0:
            if (not self.motion_enabled) or self._motion_state == "rest":
                self._zero_accum += a
                self._zero_count += 1
                self._zero_elapsed += dt_eff
            if self._zero_elapsed >= self._zero_N and self._zero_count > 0:
                b0 = self._zero_accum / self._zero_count
                try:
                    for flt in (self.filter, self.filter2, self.filter_auto,
                                self.filter_s1, self.filter_s2):
                        if flt is not None:
                            flt.x[2, 0] = b0
                            flt.P[2, 2] = min(flt.P[2, 2], 0.05)
                except Exception:
                    pass
                self._zero_done = True
                if not quiet:
                    self.statusBar().showMessage(
                        f"Ноль акселерометра установлен: {b0:+.3f} м/с² "
                        f"(усреднено за {self._zero_N:.1f} с покоя). Дальше уточняет фильтр.")

        # ПОСЕВ ФИЛЬТРА первым показанием баро (общий для всех источников)
        if self._t_first is None:
            self._t_first = t
            self._warming = True
            try:
                for flt in (self.filter, self.filter2, self.filter_auto,
                            self.filter_s1, self.filter_s2):
                    if flt is not None:
                        flt.x[0, 0] = float(h_baro)
                        flt.x[1, 0] = 0.0
                        flt.P[0, 0] = 0.25
                        flt.P[1, 1] = 0.25
            except Exception:
                pass
            if not quiet:
                self.lbl_alt.setText("—")
                self.lbl_vario.setText("—")
                self.statusBar().showMessage(
                    f"Инициализация фильтра ({WARMUP_SEC:.1f} с)…")

        # фильтр → сглаженные высота и скорость
        h_filt, v_filt = self.filter.step(a, h_baro, dt)

        # ТЕНЬ ВТОРОГО МЕТОДА (Г.1): тот же фильтр (R/Q), другое верт. ускорение
        v2 = None
        if a2 is not None and self.filter2 is not None:
            self._mirror_filter_state(self.filter2)
            _h2, v2 = self.filter2.step(a2, h_baro, dt)

        # СЛОТЫ РУЧНОЙ НАСТРОЙКИ + эталон «Авто» (пакет 14, А.5): тот же вход a,
        # свои R/Q. Считаются только в режиме «Ручной» (в Авто панель скрыта).
        v_auto = v_s1 = v_s2 = None
        if self.mode == "manual" and self.filter_auto is not None:
            for flt in (self.filter_auto, self.filter_s1, self.filter_s2):
                self._mirror_filter_state(flt)
            _ha, v_auto = self.filter_auto.step(a, h_baro, dt)
            _h1, v_s1 = self.filter_s1.step(a, h_baro, dt)
            _h2s, v_s2 = self.filter_s2.step(a, h_baro, dt)

        # WATCHDOG (блок А.4): скорость по баро за окно ~1 с против скорости фильтра.
        # Расхождение > 1.5 м/с дольше 3 с = фильтр «уехал» (что бы ни было причиной)
        # → мягкий пересев h и v по барометру (bias b не трогаем), событие в лог.
        self._wd_win.append((t, h_baro))
        while self._wd_win and t - self._wd_win[0][0] > 1.0:
            self._wd_win.popleft()
        if len(self._wd_win) >= 5 and t - self._wd_win[0][0] >= 0.8:
            v_baro_wd = ((h_baro - self._wd_win[0][1])
                         / (t - self._wd_win[0][0]))
            if self._motion_state == "osc":
                # КАЧАНИЕ (пакет 15, А.3): расхождение с баро там ЗАДУМАНО
                # (R×k_baro глушит «дыхание» порта) — пересев по шумному баро
                # разрушил бы демпфирование
                self._wd_bad = 0.0
            elif abs(v_filt - v_baro_wd) > 1.5:
                self._wd_bad += dt_eff
            else:
                self._wd_bad = 0.0
            if self._wd_bad > 3.0:
                diff = v_filt - v_baro_wd
                try:
                    self.filter.x[0, 0] = float(h_baro)
                    self.filter.x[1, 0] = float(v_baro_wd)
                    self.filter.P[0, 0] = 0.25
                    self.filter.P[1, 1] = 0.25
                except Exception:
                    pass
                self._wd_bad = 0.0
                self._wd_events.append((t, diff))
                self._ad_hold_until = t + 5.0   # адаптация R/Q замирает (З.1)
                h_filt, v_filt = self.filter.altitude, self.filter.vario
                print(f"[watchdog] t={t:.1f} с: фильтр пересеян по баро "
                      f"(расхождение {diff:+.2f} м/с)")
                crashlog.log_event("watchdog", f"t={t:.1f} с: фильтр пересеян "
                                   f"по баро (расхождение {diff:+.2f} м/с)")
                if not quiet:
                    self.statusBar().showMessage(
                        f"Фильтр пересеян по баро: расхождение с баро-скоростью "
                        f"{diff:+.1f} м/с дольше 3 с (t={t:.1f} с).")

        # АДАПТИВНЫЕ R/Q (блок З.1 + пакет 15 А.4/Ж.1): в Авто ведут основной
        # фильтр; эталон «Авто» адаптируется и в Ручном (живая строка эталона)
        if self._adaptive_on:
            self._adapt_rq(t, h_baro, dt_eff)

        # «сырой» вариометр = производная высоты баро
        if self._prev_t is not None and t > self._prev_t:
            v_raw = (h_baro - self._prev_h) / (t - self._prev_t)
        else:
            v_raw = 0.0
        self._prev_t, self._prev_h = t, h_baro

        # ПРОГРЕВ: первые WARMUP_SEC точки не выводим
        warming = False
        if self._warming:
            if t - self._t_first < WARMUP_SEC:
                warming = True
            else:
                self._warming = False
                if not quiet:
                    self.statusBar().showMessage("Фильтр инициализирован — идёт чтение.")
                    self._last_stream_status = None

        # КОМПАС (блок Б): в mekf-режиме курс даёт AHRS (MEKF + магнитометр —
        # yaw(q) + склонение), это гладкий фильтрованный курс; иначе (scalar,
        # синтетика, MEKF ещё не инициализирован) — прежняя формула Android
        # SensorManager. Затем СГЛАЖИВАНИЕ ПО КРУГУ: экспоненциально усредняем
        # единичный ВЕКТОР (cos, sin) с короткой постоянной compass_tau (~0.15 с;
        # AHRS гладкий сам — длинное сглаживание только добавляло бы лаг).
        heading = None
        if s.accel3 is not None and m_comp is not None:
            # Б.1 (пакет 14): курс — от AHRS в ОБОИХ режимах верт. ускорения
            # (ориентация считается всегда); прежняя формула — только прогрев
            # MEKF, синтетика и данные без IMU
            if (getattr(s, "imu_raw", False)
                    and self._mekf is not None and self._mekf.initialized):
                raw_head = (self._mekf.heading_deg + self._decl) % 360.0
            else:
                # аксель для наклон-компенсации: у РЕАЛЬНОГО IMU (imu_raw) берём
                # уже откалиброванный вектор ac — та же калибровка, что в детекторе
                if getattr(s, "imu_raw", False) and a_cal_norm is not None:
                    ax, ay, az = float(ac[0]), float(ac[1]), float(ac[2])
                else:
                    ax, ay, az = s.accel3
                raw_head = tilt_compensated_heading(
                    ax, ay, az, m_comp[0], m_comp[1], m_comp[2], self._decl)
            if raw_head is not None:
                hr = radians(raw_head)
                alpha = 1.0 - exp(-dt_eff / max(self._compass_tau, 1e-3))
                if self._head_vec is None:
                    self._head_vec = [cos(hr), sin(hr)]
                else:
                    self._head_vec[0] += (cos(hr) - self._head_vec[0]) * alpha
                    self._head_vec[1] += (sin(hr) - self._head_vec[1]) * alpha
                heading = degrees(atan2(self._head_vec[1], self._head_vec[0])) % 360.0

        return {"t": t, "h_baro": h_baro, "h_filt": h_filt, "v_filt": v_filt,
                "v_raw": v_raw, "heading": heading, "warming": warming,
                "motion": self._motion_state, "v2": v2,
                "v_auto": v_auto, "v_s1": v_s1, "v_s2": v_s2}

    # ------------------------------------------------------------------
    # ПРИХОД ЗАМЕРА (из потока чтения — live-источники)
    # ------------------------------------------------------------------
    @QtCore.Slot(object)
    def _on_sample(self, s):
        # СТРАЖ (пакет 13, Д.4): «хвостовые» сэмплы остановленного потока ещё
        # лежат в очереди Qt и прилетают ПОСЛЕ загрузки файла — без стража они
        # наполняли кольцевой буфер, и _redraw живой веткой затирал кривые файла
        # («поля движутся, а графики пустые» после «Скачать и воспроизвести»).
        if self._file is not None or self.worker is None:
            return
        r = self._process_sample(s)
        if r["warming"]:
            return
        # буферы (кольцевые); сглаженные серии = ровно то же число, что «Сглаж. N с»
        self.buf_t.append(r["t"])
        self.buf_h_raw.append(r["h_baro"])
        self.buf_h_filt.append(r["h_filt"])
        self.buf_v_raw.append(r["v_raw"])
        self.buf_v_filt.append(r["v_filt"])
        self.buf_v2.append(r["v2"] if r["v2"] is not None else float("nan"))
        self.buf_v_auto.append(r["v_auto"] if r["v_auto"] is not None else float("nan"))
        self.buf_v_s1.append(r["v_s1"] if r["v_s1"] is not None else float("nan"))
        self.buf_v_s2.append(r["v_s2"] if r["v_s2"] is not None else float("nan"))
        vs = self._vario_smooth_value()
        hs = self._height_smooth_value()
        self.buf_v_smooth.append(vs if vs is not None else float("nan"))
        self.buf_h_smooth.append(hs if hs is not None else float("nan"))
        if r["heading"] is not None:
            self._heading = r["heading"]
        self._dirty = True  # появились новые данные → таймер перерисует

    # ------------------------------------------------------------------
    # ГАУССОВО СГЛАЖИВАНИЕ ВАРИОМЕТРА (третье число; графики не трогает)
    # ------------------------------------------------------------------
    def _vario_smooth_value(self, end_idx: int | None = None):
        """
        Каузальное (только ПРОШЛОЕ) гауссово среднее вариометра за последние N секунд.
        Вес отсчёта на момент t: exp(−0.5·((now−t)/σ)²), σ = N/2; веса нормируем на сумму 1.
        Берём фильтрованный вариометр из буфера. Возвращает None, если N≤0 или нет данных.

        end_idx=None — живое значение «сейчас» (конец буфера, быстрый путь по deque);
        end_idx=i    — историческое значение в точке i (для курсора-инспектора).
        """
        N = float(self.spin_smooth.value())
        if N <= 0 or not self.buf_t:
            return None
        sigma = N / 2.0
        ws = 0.0
        acc = 0.0
        if end_idx is None:
            now = self.buf_t[-1]
            # идём от конца буфера в прошлое, пока не выйдем за окно последних N секунд
            for tt, vv in zip(reversed(self.buf_t), reversed(self.buf_v_filt)):
                dt = now - tt
                if dt > N:
                    break
                w = exp(-0.5 * (dt / sigma) ** 2)
                ws += w
                acc += w * vv
        else:
            ts = list(self.buf_t)
            vs = list(self.buf_v_filt)
            end_idx = max(0, min(int(end_idx), len(ts) - 1))
            now = ts[end_idx]
            for i in range(end_idx, -1, -1):   # только прошлое относительно точки курсора
                dt = now - ts[i]
                if dt > N:
                    break
                w = exp(-0.5 * (dt / sigma) ** 2)
                ws += w
                acc += w * vs[i]
        return acc / ws if ws > 0 else None

    def _height_smooth_value(self):
        """Высота, сглаженная тем же каузальным гауссовым окном N с (для зелёной
        кривой на графике высоты). None, если N≤0 или нет данных."""
        N = float(self.spin_smooth.value())
        if N <= 0 or not self.buf_t:
            return None
        sigma = N / 2.0
        now = self.buf_t[-1]
        ws = acc = 0.0
        for tt, hh in zip(reversed(self.buf_t), reversed(self.buf_h_filt)):
            d = now - tt
            if d > N:
                break
            w = exp(-0.5 * (d / sigma) ** 2)
            ws += w
            acc += w * hh
        return acc / ws if ws > 0 else None

    def _smooth_last_of(self, t_arr, v_arr):
        """Каузальное гауссово среднее ПОСЛЕДНЕЙ точки произвольной серии
        (то же окно N, что и «Сглаж.»; для рамки 2-го метода, Г.1)."""
        N = float(self.spin_smooth.value())
        if N <= 0 or len(t_arr) == 0:
            return None
        sigma = N / 2.0
        now = t_arr[-1]
        ws = acc = 0.0
        for i in range(len(t_arr) - 1, -1, -1):
            d = now - t_arr[i]
            if d > N:
                break
            if not np.isfinite(v_arr[i]):
                continue
            w = exp(-0.5 * (d / sigma) ** 2)
            ws += w
            acc += w * v_arr[i]
        return acc / ws if ws > 0 else None

    @staticmethod
    def _causal_gauss_series(arr, t_arr, N):
        """ПАКЕТНОЕ каузальное гауссово сглаживание серии (для файлового режима):
        свёртка с полуядром назад по времени (сетка записей телефона равномерна).
        Возвращает np-массив или None (N≤0)."""
        if N <= 0 or len(arr) < 2:
            return None
        dt = float(np.median(np.diff(t_arr)))
        if dt <= 0:
            return None
        K = max(1, int(round(N / dt)))
        w = np.exp(-0.5 * ((np.arange(K + 1) * dt) / (N / 2.0)) ** 2)
        num = np.convolve(arr, w, mode="full")[:len(arr)]
        den = np.convolve(np.ones(len(arr)), w, mode="full")[:len(arr)]
        return num / den

    def _update_smooth_label(self):
        """Обновить третье число: подпись «Сглаж. N с:» отдельно, значение — в
        value-лейбле фиксированной ширины; цвет по знаку меняем ТОЛЬКО цветом
        (ширина и шрифт не трогаются — ничего не прыгает)."""
        val = self._vario_smooth_value()
        if val is None:
            self.lbl_smooth_cap.setText("Сглаж.:")
            self.lbl_vario_smooth.setText("—")
            self.lbl_vario_smooth.setStyleSheet(f"color: {self._neutral_fg};")
            return
        N = float(self.spin_smooth.value())
        self.lbl_smooth_cap.setText(f"Сглаж. {N:g} с:")   # меняется лишь при смене N
        self.lbl_vario_smooth.setText(f"{val:+6.2f} м/с")
        # цвет по знаку: >+0.05 зелёный, <−0.05 красный, около нуля — цвет темы
        self.lbl_vario_smooth.setStyleSheet(f"color: {self._sign_color(val)};")

    def _on_smooth_changed(self):
        """Пользователь изменил окно сглаживания: пересчитать число (и зелёные
        серии файла, если он загружен) и сохранить."""
        if self._loading:
            return
        if self._file is not None:
            self._recalc_file_smooth()
            self._file_set_curves()
            self._file_show_numbers()
        else:
            self._update_smooth_label()
        self._refresh_cursor()    # А.3: строка «Курсор» пересчитывается сразу
        self._save_config()

    # ------------------------------------------------------------------
    # ВИДЫ ПУЛЬТА (пакет 15, Е.3): компоновка карточек шапки
    # ------------------------------------------------------------------
    def _on_layout_mode(self, on: bool):
        """Кнопка «Компоновка»: сетка 8 px + перетаскивание + правка заголовков."""
        self.cards_panel.set_layout_mode(on)
        self.btn_layout_save.setVisible(on)
        if on:
            self.statusBar().showMessage(
                "Компоновка: таскайте карточки мышью (сетка 8 px), заголовки "
                "редактируются. «Зафиксировать и сохранить вид…» — записать.")

    def _save_layout_file(self):
        """«Зафиксировать и сохранить вид…» → data\\layouts\\*.json + активный."""
        os.makedirs(LAYOUTS_DIR, exist_ok=True)
        name = datetime.now().strftime("layout_%Y-%m-%d_%H-%M-%S.json")
        path = os.path.join(LAYOUTS_DIR, name)
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(self.cards_panel.layout_dict(), fh,
                          ensure_ascii=False, indent=2)
        except OSError as e:
            self._show_error(f"Не удалось сохранить вид: {e}")
            return
        self._panel_layout = name
        self._save_config()
        self.btn_layout_mode.setChecked(False)
        self.statusBar().showMessage(
            f"Вид пульта сохранён и применён: {name} (список — «Записи» → "
            f"«Виды пульта»).")

    def apply_panel_layout(self, path: str | None):
        """«Применить»/«Заводской вид» из «Записей» (None = заводской)."""
        if path is None:
            self.cards_panel.apply_layout(None)
            self._panel_layout = None
            self.statusBar().showMessage("Вид пульта: заводской.")
        else:
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    obj = json.load(fh)
            except (OSError, ValueError) as e:
                self._show_error(f"Не удалось применить вид: {e}")
                return
            self.cards_panel.apply_layout(obj)
            self._panel_layout = os.path.basename(path)
            self.statusBar().showMessage(
                f"Вид пульта применён: {self._panel_layout}.")
        self._save_config()

    def _apply_saved_panel_layout(self):
        """При старте: применить вид из config (panel_layout), если файл жив."""
        if not self._panel_layout:
            return
        path = os.path.join(LAYOUTS_DIR, self._panel_layout)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                self.cards_panel.apply_layout(json.load(fh))
        except (OSError, ValueError):
            self._panel_layout = None      # файл удалили — заводской, честно

    # ------------------------------------------------------------------
    # «НОЛЬ ВЫСОТЫ»: показывать высоту относительно запомненной (отображение)
    # ------------------------------------------------------------------
    def _current_shown_alt(self):
        """Текущая фильтрованная высота (файл: точка плеера; live: конец буфера)."""
        if self._file is not None:
            return float(self._file["h_filt"][min(self._play_i,
                                                  len(self._file["t"]) - 1)])
        if self.buf_h_filt:
            return float(self.buf_h_filt[-1])
        return None

    def _toggle_zero_alt(self):
        """Кнопка «Ноль высоты»: запомнить опорную / вернуть абсолютную."""
        if self._h_ref is None:
            h = self._current_shown_alt()
            if h is None:
                self.statusBar().showMessage("Нет данных — нечего брать за ноль высоты.")
                return
            self._h_ref = h
            self.btn_zero_alt.setText("Абс. высота")
            self.lbl_zero_note.setText(f"0 = {h:.1f} м абс.")
        else:
            self._h_ref = None
            self.btn_zero_alt.setText("Ноль высоты")
            self.lbl_zero_note.setText("")
        # перерисовать под новую опору
        if self._file is not None:
            self._file_set_curves()
            self._file_show_numbers()
            if self._cursor_t is not None:
                self._set_cursor(self._cursor_t)
        else:
            self._dirty = True
            if self._cursor_t is not None:
                self._set_cursor(self._cursor_t)

    # ------------------------------------------------------------------
    # ПЕРЕРИСОВКА (по таймеру)
    # ------------------------------------------------------------------
    def _redraw(self):
        # ФАЙЛОВЫЙ РЕЖИМ: тик плеера (числа/вид из готовых серий); live-часть ниже
        # не работает (кольцевые буферы пусты — кривые файла стоят целиком)
        self._file_tick()
        # авто-масштаб Y обоих графиков (сам решает, пора ли пересчитывать)
        if self._file is not None or self.buf_t:
            self._auto_y_tick()
        # смещение акселерометра показываем всегда (даже до старта чтения)
        if self.filter is not None:
            self.lbl_bias.setText(f"{self.filter.accel_bias:+.3f} м/с²")
        # живые эффективные R̂/Q̂ адаптации + индикатор состояния (З.1 + пакет 14 Ж;
        # с пакета 15 А.4 R адаптируется и в полёте — «заморожен (Движение)»
        # остался только у legacy-режима)
        if self.mode == "auto" and self._adaptive_on and self.filter is not None:
            if self._warming or self._t_first is None:
                state = " · прогрев"
            elif self._last_t is not None and self._last_t < self._ad_hold_until:
                state = " · заморожен (watchdog)"
            elif (self._adapt_r_mode != "d2" and self.motion_enabled
                    and self._motion_state != "rest"):
                state = " · заморожен (Движение)"   # legacy: R замирает в движении
            else:
                state = " · активен"
            self.lbl_adaptive.setText(
                f"R̂ {float(self.filter.R[0, 0]):.4f} м²  "
                f"Q̂ {float(self.filter.sigma_accel):.3f} м/с²{state}")
        elif self.lbl_adaptive.text() != "—":
            self.lbl_adaptive.setText("—")
        # живая метрика слотов ручной настройки (А.5, ~1 раз/с)
        self._slot_metrics_update()
        # индикатор под компасом в режиме Live-EKF (остаток обновляется живьём)
        self._update_calib_indicator_live()
        # ЗВУК (блок Ж): выбранная серия → бипы (live и проигрывание файла;
        # ~30 Гц — темп перерисовки). None = тишина (пауза/нет данных).
        # А.2 (пакет 14): источник звука — «Вариометр (фильтр)» или «Сглаж. N с»
        # (переключатель у громкости); сглаженной серии нет (N=0) → фильтр.
        if self.sound_cb is not None:
            v_now = None
            want_smooth = self.sound_source == "smooth"
            if self._file is not None:
                if self._playing:
                    i = min(self._play_i, len(self._file["t"]) - 1)
                    sm = self._file.get("v_smooth")
                    if want_smooth and sm is not None and np.isfinite(sm[i]):
                        v_now = float(sm[i])
                    else:
                        v_now = float(self._file["v_filt"][i])
            elif self.worker is not None and self.buf_v_filt:
                if (want_smooth and self.buf_v_smooth
                        and np.isfinite(self.buf_v_smooth[-1])):
                    v_now = float(self.buf_v_smooth[-1])
                else:
                    v_now = float(self.buf_v_filt[-1])
            self.sound_cb(v_now)
        # индикатор детектора движения (в файловом режиме его ведёт плеер по серии)
        if self._file is None:
            self._set_motion_label(self._motion_state)
        # курсор-инспектор: точка t_c выпала из кольцевого буфера → «—» и убрать линии
        if (self._file is None and self._cursor_t is not None
                and self.buf_t and self.buf_t[0] > self._cursor_t):
            self._clear_cursor()
        # живой поток: статус связи + круглый индикатор приёма + мини-качество
        if self.worker is not None and getattr(self.worker.source, "live", False):
            src = self.worker.source
            # SENSORS → приоры по даташитам (пакет 15, З.2): один раз на
            # подключение — в лог; старт адаптации R сажается по паспорту баро
            if not self._sensors_logged and getattr(src, "sensors", None):
                if len(src.sensors) >= 4:
                    self._sensors_logged = True
                    import sensor_priors
                    pri = sensor_priors.match(src.sensors)
                    names = ", ".join(f"{k}={v.get('name', '?')}"
                                      for k, v in src.sensors.items()
                                      if v.get("name") not in (None, "-"))
                    crashlog.log_event("sensors", f"датчики телефона: {names}")
                    crashlog.log_event("sensors",
                                       "приоры: " + sensor_priors.describe(pri))
                    b = pri.get("baro")
                    if b and self._adaptive_on and len(self._ad_d2) < 25:
                        self._ad_R_mult = min(3.0, max(0.3,
                                                       b["R_baro"] / R_DEFAULT))
                        crashlog.log_event(
                            "sensors", f"стартовый R̂ по приору {b['chip']}: "
                            f"{R_DEFAULT * self._ad_R_mult:.4f} м² "
                            f"(дальше уточнит адаптация)")
            if (not self._temp_logged
                    and getattr(src, "temp_last", None) is not None):
                self._temp_logged = True
                crashlog.log_event("temp", f"телефон шлёт температуру: "
                                   f"{src.temp_last['c']:.1f} °C (только показ)")
            stat = f"Поток: {src.status}"
            if stat != getattr(self, "_last_stream_status", None):
                self._last_stream_status = stat
                self.statusBar().showMessage(stat)
            m = src.link_metrics()
            sil = m["silence_s"]
            if sil is not None and sil <= 1.0:
                # данные идут: зелёный, вспышка на каждый новый пакет
                if src.received > self._dot_seen:
                    self._dot_seen = src.received
                    self._dot_flash = 1.0
                self._dot_flash *= 0.80
                g = 120 + int(120 * self._dot_flash)
                self.link_dot.set_color(QtGui.QColor(30, min(240, g), 60))
                q = m["quality_pct"]
                if q is None:
                    self.lbl_link_q.setText("кач. —")
                    self.lbl_link_q.setStyleSheet("color:#888;")
                else:
                    col = "#2c7a2c" if q >= 95 else ("#c09010" if q >= 70 else "#c0392b")
                    self.lbl_link_q.setText(f"кач. {q:.0f}%")
                    self.lbl_link_q.setStyleSheet(f"color:{col}; font-weight:bold;")
            else:
                # тишина > 1 c: красный + сколько секунд нет данных
                self.link_dot.set_color(QtGui.QColor(200, 60, 50))
                self.lbl_link_q.setText("нет данных" if sil is None
                                        else f"нет данных {sil:.0f} с")
                self.lbl_link_q.setStyleSheet("color:#c0392b; font-weight:bold;")
        else:
            self.link_dot.set_color(QtGui.QColor("#777777"))
            if self.lbl_link_q.text():
                self.lbl_link_q.setText("")
        if not self.buf_t:
            return
        # Перерисовываем только когда есть новые данные или попросили обновить вид.
        # Если идёт чтение — двигаем рамки за данными; если стоим на паузе —
        # рамки НЕ трогаем, чтобы работало масштабирование/сдвиг мышью.
        if not (self._dirty or self._view_refresh):
            return
        apply_ranges = (self.worker is not None) or self._view_refresh
        refresh = self._view_refresh
        self._dirty = False

        t = np.fromiter(self.buf_t, dtype=float)
        v_filt = np.fromiter(self.buf_v_filt, dtype=float)
        ref = self._h_ref or 0.0     # «Ноль высоты»: чисто отображение

        # кривые обновляем всегда; видимость управляется галочками отдельно
        self.curve_h_raw.setData(t, np.fromiter(self.buf_h_raw, dtype=float) - ref)
        self.curve_h_filt.setData(t, np.fromiter(self.buf_h_filt, dtype=float) - ref)
        self.curve_v_raw.setData(t, np.fromiter(self.buf_v_raw, dtype=float))
        self.curve_v_filt.setData(t, v_filt)
        if self.chk_smooth.isChecked() and float(self.spin_smooth.value()) > 0:
            self.curve_v_smooth.setData(t, np.fromiter(self.buf_v_smooth, dtype=float))
            self.curve_h_smooth.setData(
                t, np.fromiter(self.buf_h_smooth, dtype=float) - ref)
        # 2-й метод (Г.1): пунктир + рамка чисел (живой поток)
        if self.buf_v2:
            v2arr = np.fromiter(self.buf_v2, dtype=float)
            if np.isfinite(v2arr[-1]):
                if self.chk_second.isChecked():
                    self.curve_v2.setData(t, v2arr, connect="finite")
                self.frame2.setVisible(True)
                self.lbl2_vario.setText(f"{v2arr[-1]:+6.2f} м/с")
                self.lbl2_vario.setStyleSheet(f"color: {self._sign_color(v2arr[-1])};")
                s2 = self._smooth_last_of(t, v2arr)
                if s2 is None:
                    self.lbl2_smooth.setText("—")
                else:
                    self.lbl2_smooth.setText(f"{s2:+6.2f} м/с")
                    self.lbl2_smooth.setStyleSheet(f"color: {self._sign_color(s2)};")
        # слоты ручной настройки (А.5), живой поток
        if self.mode == "manual" and self.buf_v_s1:
            for sk, curve, buf in (("s1", self.curve_v_s1, self.buf_v_s1),
                                   ("s2", self.curve_v_s2, self.buf_v_s2)):
                if self._slots[sk]["show"]:
                    curve.setData(t, np.fromiter(buf, dtype=float),
                                  connect="finite")

        if apply_ranges:
            # КАЖДЫЙ график — своё окно по времени (оси X независимы);
            # ось Y ведёт _auto_y_tick (режимы Авто/Ручной у каждого графика)
            win_a = self._window_seconds(self.combo_window_alt)
            x0a = t[0] if win_a is None else max(t[0], t[-1] - win_a)
            self.plot_alt.setXRange(x0a, t[-1], padding=0.0)
            win_v = self._window_seconds(self.combo_window_vario)
            x0v = t[0] if win_v is None else max(t[0], t[-1] - win_v)
            self.plot_vario.setXRange(x0v, t[-1], padding=0.0)
            if refresh:
                self._y_next = {"alt": 0.0, "vario": 0.0}   # авто-Y пересчитать сразу
            self._view_refresh = False

        # крупные цифры — последние значения (постоянная ширина, знак всегда;
        # цвет вариометра по знаку: набор зелёный / снижение красный)
        self.lbl_alt.setText(f"{self.buf_h_filt[-1] - ref:+7.1f} м")
        self.lbl_alt_abs.setText(f"{self.buf_h_filt[-1]:+7.1f} м")   # Е.2: всегда абсолютная
        self.lbl_vario.setText(f"{self.buf_v_filt[-1]:+6.2f} м/с")
        self.lbl_vario.setStyleSheet(f"color: {self._sign_color(self.buf_v_filt[-1])};")
        self._update_smooth_label()   # третье число — гауссово сглаживание (графики не трогаем)
        # компас курса
        if self._heading is not None:
            self.compass.setHeading(self._heading)
            # ширина постоянная: градусы в поле из 3 знаков + румб
            self.lbl_heading.setText(f"{self._heading:3.0f}° {cardinal_ru(self._heading)}")

    # ------------------------------------------------------------------
    # ЭКСПОРТ ГРАФИКА В PNG (matplotlib, backend Agg — без окна, не зависает)
    # ------------------------------------------------------------------
    def export_png(self):
        # активные серии: полный файл или кольцевые буферы (абсолютная высота)
        if self._file is not None:
            f = self._file
            t, h_raw, h_filt = f["t"], f["h_raw"], f["h_filt"]
            v_raw, v_filt = f["v_raw"], f["v_filt"]
        elif self.buf_t:
            t = np.fromiter(self.buf_t, dtype=float)
            h_raw = np.fromiter(self.buf_h_raw, dtype=float)
            h_filt = np.fromiter(self.buf_h_filt, dtype=float)
            v_raw = np.fromiter(self.buf_v_raw, dtype=float)
            v_filt = np.fromiter(self.buf_v_filt, dtype=float)
        else:
            self._show_error("Нет данных для экспорта. Сначала запустите чтение.")
            return

        # ВАЖНО: backend Agg включаем ДО импорта pyplot — рисуем в файл без окна
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})
        fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

        ax[0].plot(t, h_raw, color="0.7", lw=0.9, label="Баро (сырое)")
        ax[0].plot(t, h_filt, "C0", lw=1.8, label="Фильтр Калмана")
        ax[0].set_ylabel("Высота, м")
        ax[0].set_title("VarioPro3 — экспорт графика")
        ax[0].legend(loc="upper left")

        ax[1].plot(t, v_raw, color="0.7", lw=0.8, label="Сырой (производная баро)")
        ax[1].plot(t, v_filt, "C3", lw=1.8, label="Фильтр Калмана")
        ax[1].axhline(0, color="0.4", lw=1, ls="--")
        ax[1].set_ylim(*nice_vario_range(v_filt))  # рамку — по фильтрованному сигналу
        ax[1].set_ylabel("Вариометр, м/с")
        ax[1].set_xlabel("Время, с")
        ax[1].legend(loc="upper left")

        fig.tight_layout()
        # скриншоты складываются в data\screenshots\ (их список — на вкладке «Записи»)
        shots_dir = os.path.join(DATA_DIR, "screenshots")
        os.makedirs(shots_dir, exist_ok=True)
        fname = datetime.now().strftime("variopro_вариометр_%Y-%m-%d_%H-%M-%S.png")
        path = os.path.join(shots_dir, fname)
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        self.statusBar().showMessage(f"График сохранён: {path}")
        if not getattr(self, "_selftest_quiet", False):
            QtWidgets.QMessageBox.information(
                self, "Экспорт PNG",
                f"График сохранён:\n{path}\n\nСписок скриншотов — на вкладке «Записи».")
        return path

    # ------------------------------------------------------------------
    # СЛУЖЕБНОЕ
    # ------------------------------------------------------------------
    def _show_error(self, text: str):
        self.statusBar().showMessage("Ошибка: " + text)
        QtWidgets.QMessageBox.warning(self, "Ошибка", text)

    def closeEvent(self, event):
        """При закрытии окна аккуратно останавливаем поток чтения и сохраняем настройки."""
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait(2000)
        self._save_config()
        super().closeEvent(event)


# ======================================================================
# ТОЧКА ВХОДА
# ======================================================================
def main():
    parser = argparse.ArgumentParser(description="VarioPro3 — десктоп-вариометр (Фаза 0)")
    parser.add_argument("--selftest", action="store_true",
                        help="самопроверка: запустить симуляцию, сохранить скриншот и выйти")
    parser.add_argument("--source", choices=["sim", "csv"], default="sim",
                        help="источник для самопроверки")
    parser.add_argument("--seconds", type=float, default=5.0,
                        help="сколько секунд снимать перед скриншотом (самопроверка)")
    parser.add_argument("--speed", type=float, default=20.0,
                        help="ускорение симуляции в режиме самопроверки")
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    if os.path.exists(APP_ICON):
        app.setWindowIcon(QtGui.QIcon(APP_ICON))
    win = VarioApp()
    win.show()

    if args.selftest:
        # --- режим самопроверки: всё делаем автоматически и закрываемся ---
        win._selftest_quiet = True      # без модальных окон (экспорт PNG)
        if args.source == "sim":
            win.combo_source.setCurrentText("Симуляция")
            # ускоренная симуляция, чтобы за пару секунд показать весь сценарий
            win.worker = SourceWorker(SimSource(dt=DT, speed=args.speed, loop=False))
            win.worker.sampleReady.connect(win._on_sample)
            win.worker.errorOccurred.connect(win._show_error)
            win.worker.finished.connect(win._on_worker_finished)
            win.worker.start()
            win._view_refresh = True
            win.btn_start.setEnabled(False)
            win.btn_stop.setEnabled(True)
        else:
            win.combo_source.setCurrentText("CSV-файл")
            win.start()

        def capture_and_quit():
            os.makedirs(DOCS_DIR, exist_ok=True)
            shot = os.path.join(DOCS_DIR, "screenshot_app.png")
            win.grab().save(shot)          # снимок окна
            try:
                exported = win.export_png()  # заодно проверяем экспорт через Agg
            except Exception as e:
                exported = f"(ошибка экспорта: {e})"
            print(f"Самопроверка завершена.")
            print(f"  скриншот окна:   {shot}")
            print(f"  экспорт графика: {exported}")
            win.stop()
            app.quit()

        QtCore.QTimer.singleShot(int(args.seconds * 1000), capture_and_quit)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
