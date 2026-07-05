# -*- coding: utf-8 -*-
"""
sound_app.py
============
ЗВУК ВАРИОМЕТРА (Фаза 4; пакет 14 — блоки А.2, В.2, Г): вкладка «Звук» + бипы.

Что умеет:
  • библиотека профилей в data\\sound_profiles (список с датами, «Выбрать
    активным», «Удалить», «Импорт» из файла или вставкой текста, «Автоширина»,
    кнопка «Открыть сайт профилей» → windeckfalken.de);
  • формат — конфиг XC Tracer: строки  tone=<варио м/с>,<частота Гц>,<цикл мс>,
    <скважность %>  (вариации формата допускаются); незнакомые строки файла
    ХРАНЯТСЯ КАК ЕСТЬ (это служебные параметры конфига — файл не портится);
  • ЭТАЛОННАЯ СЕМАНТИКА СИНТЕЗА (пакет 14, Г.2): между точками таблицы —
    линейная интерполяция; частота и цикл ФИКСИРУЮТСЯ НА НАЧАЛО каждого бипа
    (внутри бипа тон постоянный — начатый бип доигрывается); синус с атакой/
    релизом ~8 мс (без щелчков); скважность ≥100 → непрерывный тон с плавной
    подстройкой частоты (τ≈50 мс); между sink_on и climb_on — тишина;
  • ПОРОГИ В UI (Г.1): «подъём от» / «спуск от» — по умолчанию из профиля
    (Climb/SinkToneOnThreshold), гистерезис — ширина из Off-порогов профиля;
    действуют и на живой звук, и на тест-полигон; хранятся в config → sound;
  • переключатель «Звук от: Вариометр (фильтр) / Сглаж. N с» (А.2) — здесь и
    в компактном дубле на «Вариометре»;
  • тест-полигон в стиле пульта (Г.3): крупный ползунок, живая строка
    «v → частота/цикл/скважность», «Свип min→max за 5 с»;
  • АУДИО-КОЛБЭК (Д.2): работает в потоке PortAudio, НЕ трогает GUI и НЕ
    аллоцирует память (предвыделенные буферы + out= у numpy-операций).
Если аудио на машине недоступно — честная надпись в статусе, пульт не падает.
"""

from __future__ import annotations

import datetime
import math
import os
import re
import threading
import webbrowser

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from vario_app import save_config, load_config  # merge-сохранение config.json
from widgets import StepSpinBox, make_delta_field  # единый спинбокс (пакет 15, Г)

PC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(PC_DIR)
PROFILES_DIR = os.path.join(ROOT, "data", "sound_profiles")
PROFILES_URL = "https://www.windeckfalken.de/special/xctracer/handson/main.html"

DEFAULT_PROFILE_NAME = "default_xctracer.txt"
DEFAULT_PROFILE_TEXT = """\
# Профиль звука VarioPro3 по умолчанию (формат XC Tracer).
# tone=<варио м/с>,<частота Гц>,<длина цикла мс>,<скважность %>
# Скважность 100 = непрерывный тон. Другие профили: windeckfalken.de
climbToneOnThreshold=0.2
climbToneOffThreshold=0.15
sinkToneOnThreshold=-2.5
sinkToneOffThreshold=-2.3
tone=-10.0,200,200,100
tone=-3.0,280,200,100
tone=-0.5,300,350,100
tone=0.1,400,600,50
tone=1.2,550,552,52
tone=2.7,763,483,55
tone=4.2,985,412,58
tone=6.0,1234,332,62
tone=8.0,1517,241,66
tone=10.0,1800,150,70
"""

_TONE_RE = re.compile(
    r"^\s*tone\s*=\s*(-?\d+(?:[.,]\d+)?)\s*[,;]\s*(\d+(?:[.,]\d+)?)\s*[,;]\s*"
    r"(\d+(?:[.,]\d+)?)\s*[,;]\s*(\d+(?:[.,]\d+)?)\s*$", re.IGNORECASE)
_THRESH_RE = re.compile(
    r"^\s*(climbToneOnThreshold|climbToneOffThreshold|"
    r"sinkToneOnThreshold|sinkToneOffThreshold)\s*=\s*(-?\d+(?:[.,]\d+)?)\s*$",
    re.IGNORECASE)


def _num(s: str) -> float:
    return float(s.replace(",", "."))


class ToneProfile:
    """Таблица тонов XC Tracer + пороги вкл/выкл. Незнакомые строки хранятся
    как есть (это служебные параметры конфига прибора — не звук)."""

    def __init__(self, text: str, name: str = "?"):
        self.name = name
        self.text = text                     # исходник целиком (ничего не теряем)
        self.points = []                     # [(vario, freq, cycle_ms, duty_pct)]
        self.thresholds = {}                 # пороги, если есть в файле
        self.unknown_lines = 0               # служебные строки (сохранены в text)
        for line in text.splitlines():
            if not line.strip() or line.lstrip().startswith(("#", ";", "//")):
                continue
            m = _TONE_RE.match(line)
            if m:
                self.points.append(tuple(_num(g) for g in m.groups()))
                continue
            m = _THRESH_RE.match(line)
            if m:
                self.thresholds[m.group(1)[0].lower() + m.group(1)[1:]] = _num(m.group(2))
                continue
            self.unknown_lines += 1          # храним как есть (self.text)
        self.points.sort(key=lambda p: p[0])

    @property
    def valid(self) -> bool:
        return len(self.points) >= 2

    def lookup(self, v: float):
        """(частота Гц, цикл мс, скважность %) для вариометра v — линейная
        интерполяция; вне диапазона — крайние точки."""
        p = self.points
        if not p:
            return (0.0, 0.0, 0.0)
        if v <= p[0][0]:
            return p[0][1:]
        if v >= p[-1][0]:
            return p[-1][1:]
        for i in range(1, len(p)):
            if v <= p[i][0]:
                a, b = p[i - 1], p[i]
                k = (v - a[0]) / max(b[0] - a[0], 1e-9)
                return tuple(a[j] + k * (b[j] - a[j]) for j in (1, 2, 3))
        return p[-1][1:]

    def default_thresholds(self):
        """Пороги профиля (c_on, c_off, s_on, s_off); отсутствующие — None."""
        th = self.thresholds
        c_on = th.get("climbToneOnThreshold")
        c_off = th.get("climbToneOffThreshold", c_on)
        s_on = th.get("sinkToneOnThreshold")
        s_off = th.get("sinkToneOffThreshold", s_on)
        return c_on, c_off, s_on, s_off


class VarioTonePlayer:
    """Синтезатор бипов по эталонной семантике XC Tracer (пакет 14, Г.2).

    GUI-поток зовёт set_vario() ~20–30 Гц → под локом обновляются ЦЕЛЕВЫЕ
    (частота, цикл, скважность). Аудио-поток (колбэк PortAudio) фиксирует
    целевые значения В НАЧАЛЕ КАЖДОГО БИПА (внутри бипа тон постоянный),
    рисует огибающую с атакой/релизом ~8 мс, непрерывный тон (скважность
    ≥99.5) плавно подстраивает частоту (τ≈50 мс). Колбэк не трогает GUI и
    не аллоцирует память (Д.2): все буферы предвыделены, numpy с out=.
    Аудио недоступно → self.error (строка), пульт живёт дальше."""

    FS = 44100
    # ПАКЕТ 15 (В.3): диагностика показала, что СИНТЕЗ верный (WAV из этого же
    # колбэка: пачки 674.7 Гц ровно 200/200 мс, непрерывный 440.0 Гц), а «вой»
    # из колонок — underrun потока: блок 512 (~12 мс) душится GIL при
    # перерисовке графиков (~26 мс на длинном файле). Лечение по ТЗ:
    # блок больше + latency='high' (запас буфера у PortAudio).
    BLOCK = 1024                   # ~23 мс
    EDGE = int(0.008 * FS)         # атака/релиз 8 мс, отсчётов
    CONT_TAU = 0.05                # τ подстройки частоты непрерывного тона, с

    def __init__(self):
        self.error = None
        self._sd = None
        self._stream = None
        self._lock = threading.Lock()
        self._profile: ToneProfile | None = None
        self._enabled = False
        self._volume = 0.6
        # форма волны (пакет 15, В.4): «меандр» — как настоящий прибор
        # (жёсткий писк), «синус» — мягкий; по умолчанию меандр
        self.waveform = "square"
        # эффективные пороги (Г.1): (c_on, c_off, s_on, s_off); None = нет порога
        self._th = (None, None, None, None)
        # целевые параметры от GUI (под локом)
        self._freq = 0.0
        self._cycle = 0.0
        self._duty = 0.0
        self._gate_on = False      # звучит ли сейчас (гистерезис порогов)
        # ---- состояние аудио-потока (только колбэк) ----
        self._mode = "silence"     # silence | beep | cont
        self._phase = 0.0          # фаза синуса, рад
        self._env = 0.0            # текущая огибающая 0..1
        self._beep_f = 0.0         # параметры ТЕКУЩЕГО бипа (латч на старте)
        self._beep_len = 1         # длина цикла, отсчётов
        self._beep_on = 0          # звучащая часть, отсчётов
        self._beep_pos = 0         # позиция в цикле, отсчётов
        self._cont_f = 0.0         # текущая частота непрерывного тона
        # ---- предвыделенные буферы (Д.2: без аллокаций в колбэке) ----
        n = self.BLOCK
        self._idx = np.arange(n + 1, dtype=np.float64)
        self._w1 = np.empty(n, dtype=np.float64)
        self._w2 = np.empty(n, dtype=np.float64)
        self._w3 = np.empty(n, dtype=np.float64)
        self._w4 = np.empty(n, dtype=np.float64)
        try:
            import sounddevice as sd
            self._sd = sd
        except Exception as e:
            self.error = f"sounddevice недоступен: {e}"

    # ---- управление (GUI-поток) ----
    def set_profile(self, profile: ToneProfile | None):
        with self._lock:
            self._profile = profile

    def set_thresholds(self, c_on, c_off, s_on, s_off):
        """Эффективные пороги зоны тишины (Г.1); None = порога нет."""
        with self._lock:
            self._th = (c_on, c_off, s_on, s_off)

    def set_enabled(self, on: bool):
        self._enabled = bool(on)
        if on:
            self._ensure_stream()
        else:
            with self._lock:
                self._freq = 0.0

    def set_volume(self, v01: float):
        self._volume = min(1.0, max(0.0, float(v01)))

    def set_waveform(self, w: str):
        """«square» (меандр, как прибор) | «sine» (мягкий синус). В.4."""
        if w in ("square", "sine"):
            self.waveform = w

    def _audible(self, v: float) -> bool:
        """Зона тишины между sink_on и climb_on, с гистерезисом Off-порогов."""
        c_on, c_off, s_on, s_off = self._th
        if c_on is None and s_on is None:
            return True
        climb = sink = False
        if c_on is not None:
            climb = v >= ((c_off if c_off is not None else c_on)
                          if self._gate_on else c_on)
        if s_on is not None:
            sink = v <= ((s_off if s_off is not None else s_on)
                         if self._gate_on else s_on)
        return climb or sink

    def set_vario(self, v):
        """Обновить целевой тон от вариометра (None = тишина). ~20–30 Гц."""
        if not self._enabled:
            return
        with self._lock:
            prof = self._profile
            if v is None or prof is None or not prof.valid:
                self._freq = 0.0
                self._gate_on = False
                return
            v = float(v)
            if not self._audible(v):
                self._freq = 0.0
                self._gate_on = False
                return
            self._gate_on = True
            f, cyc, duty = prof.lookup(v)
            self._freq, self._cycle, self._duty = float(f), float(cyc), float(duty)

    # ---- поток ----
    def _ensure_stream(self):
        if self._stream is not None or self._sd is None:
            return
        try:
            # latency='high' (В.3): PortAudio берёт буфер с запасом — пропуски
            # GIL на перерисовке больше не рвут звук («вой»/треск underrun'ов)
            self._stream = self._sd.OutputStream(
                samplerate=self.FS, channels=1, blocksize=self.BLOCK,
                dtype="float32", latency="high", callback=self._callback)
            self._stream.start()
            self.error = None
        except Exception as e:
            self._stream = None
            self.error = f"аудио недоступно: {e}"

    def close(self):
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    # ---- сегмент синуса с огибающей (helpers колбэка; без аллокаций) ----
    def _emit(self, out, i0: int, n: int, f: float, env_from_pos) -> None:
        """out[i0:i0+n] = волна(фаза..) × env × volume. env_from_pos(буфер w3, n)
        обязан заполнить w3[:n] значениями огибающей 0..1. Волна — синус или
        меандр (В.4: sign(sin), как настоящий прибор); фаза непрерывна."""
        w1 = self._w1[:n]
        w2 = self._w2[:n]
        w3 = self._w3[:n]
        np.multiply(self._idx[1:n + 1], 2.0 * math.pi * f / self.FS, out=w1)
        np.add(w1, self._phase, out=w1)
        self._phase = float(w1[-1] % (2.0 * math.pi))
        np.sin(w1, out=w2)
        if self.waveform == "square":
            np.sign(w2, out=w2)
            np.multiply(w2, 0.6, out=w2)   # меандр громче синуса той же амплитуды
        env_from_pos(w3, n)
        self._env = float(w3[-1])
        np.multiply(w2, w3, out=w2)
        np.multiply(w2, self._volume, out=w2)
        out[i0:i0 + n, 0] = w2

    def _env_beep(self, w3, n):
        """Огибающая бипа: атака/релиз EDGE внутри звучащей части цикла."""
        pos0 = self._beep_pos
        on = self._beep_on
        w4 = self._w4[:n]
        np.add(self._idx[:n], float(pos0), out=w4)          # позиции в цикле
        np.subtract(float(on), w4, out=w3)                  # (on − pos)
        np.minimum(w4, w3, out=w3)                          # min(pos, on−pos)
        np.divide(w3, float(self.EDGE), out=w3)
        np.clip(w3, 0.0, 1.0, out=w3)                       # pos≥on → отрицательное → 0

    def _env_att(self, w3, n):
        """Линейный подъём огибающей к 1 со скоростью 1/EDGE (атака)."""
        np.multiply(self._idx[1:n + 1], 1.0 / float(self.EDGE), out=w3)
        np.add(w3, self._env, out=w3)
        np.clip(w3, 0.0, 1.0, out=w3)

    def _env_rel(self, w3, n):
        """Линейный спад огибающей к 0 со скоростью 1/EDGE (релиз)."""
        np.multiply(self._idx[1:n + 1], -1.0 / float(self.EDGE), out=w3)
        np.add(w3, self._env, out=w3)
        np.clip(w3, 0.0, 1.0, out=w3)

    def _callback(self, out, frames, _time, _status):
        # целевые параметры — один раз на блок (под локом)
        with self._lock:
            f_t, cyc_t, duty_t = self._freq, self._cycle, self._duty
        if not self._enabled:
            f_t = 0.0
        if frames > self.BLOCK:                 # нестандартный блок — не рискуем
            out[:, 0] = 0.0
            return
        i = 0
        while i < frames:
            n_left = frames - i
            if self._mode == "beep":
                # доигрываем текущий бип до конца цикла (параметры залатчены)
                n = min(n_left, self._beep_len - self._beep_pos)
                if n > 0:
                    self._emit(out, i, n, self._beep_f, self._env_beep)
                    self._beep_pos += n
                    i += n
                if self._beep_pos >= self._beep_len:
                    # ГРАНИЦА ЦИКЛА: здесь (и только здесь) латчим новые
                    # параметры из целевых (Г.2: тон фиксируется на начало бипа)
                    self._beep_pos = 0
                    if f_t <= 0.0:
                        self._mode = "silence"
                    elif duty_t >= 99.5:
                        self._mode = "cont"
                        self._cont_f = f_t
                    else:
                        self._latch_beep(f_t, cyc_t, duty_t)
                continue
            if self._mode == "cont":
                if f_t <= 0.0 or duty_t < 99.5:
                    self._mode = "silence"      # релиз сделает ветка silence
                    continue
                # плавная подстройка частоты (τ≈50 мс), фаза непрерывна
                alpha = 1.0 - math.exp(-(n_left / self.FS) / self.CONT_TAU)
                self._cont_f += (f_t - self._cont_f) * alpha
                self._emit(out, i, n_left, self._cont_f, self._env_att)
                i = frames
                continue
            # silence: релиз к нулю (на последней частоте), затем тишина/новый тон
            if self._env > 1e-4:
                n = min(n_left, int(self._env * self.EDGE) + 1)
                f_rel = self._cont_f if self._cont_f > 0 else max(self._beep_f, 200.0)
                self._emit(out, i, n, f_rel, self._env_rel)
                i += n
                continue
            self._env = 0.0
            if f_t > 0.0:
                if duty_t >= 99.5:
                    self._mode = "cont"
                    self._cont_f = f_t
                else:
                    self._mode = "beep"
                    self._beep_pos = 0
                    self._latch_beep(f_t, cyc_t, duty_t)
                continue
            out[i:frames, 0] = 0.0              # честная тишина
            i = frames

    def _latch_beep(self, f: float, cyc_ms: float, duty: float):
        """Зафиксировать параметры НОВОГО бипа (вызывается на границе цикла)."""
        self._beep_f = float(f)
        self._beep_len = max(int(cyc_ms * self.FS / 1000.0), 2 * self.EDGE)
        self._beep_on = max(int(self._beep_len * duty / 100.0), 2)


class _ImportTextDialog(QtWidgets.QDialog):
    """«Импорт» вставкой текста: поле + имя файла."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Импорт профиля — вставьте текст")
        self.resize(520, 420)
        v = QtWidgets.QVBoxLayout(self)
        v.addWidget(QtWidgets.QLabel(
            "Вставьте текст профиля XC Tracer (строки tone=...). "
            "Служебные строки сохранятся как есть."))
        self.edit = QtWidgets.QPlainTextEdit()
        v.addWidget(self.edit, stretch=1)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Имя файла:"))
        self.name = QtWidgets.QLineEdit(
            datetime.datetime.now().strftime("profile_%Y-%m-%d_%H-%M-%S.txt"))
        row.addWidget(self.name, stretch=1)
        v.addLayout(row)
        bb = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)


class SoundPanel(QtWidgets.QWidget):
    """Вкладка «Звук»: профили + пороги + тест-полигон + громкость."""

    SRC_ITEMS = ["Вариометр (фильтр)", "Сглаж. N с"]
    SRC_KEYS = ["vario", "smooth"]

    def __init__(self):
        super().__init__()
        os.makedirs(PROFILES_DIR, exist_ok=True)
        self._write_default_profile()
        cfg_all = load_config()
        cfg = cfg_all.get("sound", {}) if isinstance(cfg_all.get("sound"), dict) else {}
        self.player = VarioTonePlayer()
        self.profile: ToneProfile | None = None
        self._active_name = cfg.get("profile", DEFAULT_PROFILE_NAME)
        # пользовательские пороги (Г.1): None = брать из профиля
        self._th_climb_on = cfg.get("climb_on")
        self._th_sink_on = cfg.get("sink_on")
        self._sweep_timer = None
        self._compacts = []          # компактные дубли (Вариометр): (btn, sl, combo)
        self._loading = True
        # источник звука (А.2): владелец состояния — вариометр (vario_app);
        # main.py подключает эти колбэки
        self.sound_source_get = lambda: cfg_all.get("sound_source", "vario")
        self.sound_source_set = lambda src: None

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        outer.addWidget(scroll)
        body = QtWidgets.QWidget()
        scroll.setWidget(body)
        root = QtWidgets.QVBoxLayout(body)

        # ---- громкость, вкл/выкл, источник, пороги ----
        gb_vol = QtWidgets.QGroupBox("Звук вариометра")
        lvbox = QtWidgets.QVBoxLayout(gb_vol)
        lv = QtWidgets.QHBoxLayout()
        lvbox.addLayout(lv)
        self.btn_mute = QtWidgets.QPushButton()
        self.btn_mute.setCheckable(True)
        self.btn_mute.setChecked(bool(cfg.get("enabled", False)))
        self.btn_mute.setFixedWidth(120)
        self.btn_mute.toggled.connect(self._on_enabled)
        lv.addWidget(self.btn_mute)
        lv.addWidget(QtWidgets.QLabel("Громкость:"))
        self.sl_vol = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sl_vol.setRange(0, 100)
        self.sl_vol.setValue(int(float(cfg.get("volume", 0.6)) * 100))
        self.sl_vol.valueChanged.connect(self._on_volume)
        lv.addWidget(self.sl_vol, stretch=1)
        # форма волны (пакет 15, В.4): меандр (как прибор) / синус
        lv.addWidget(QtWidgets.QLabel("Форма:"))
        self.combo_wave = QtWidgets.QComboBox()
        self.combo_wave.addItems(["меандр (как прибор)", "синус"])
        self.combo_wave.setToolTip(
            "Форма волны бипов. «Меандр» — жёсткий писк, как настоящий XC Tracer\n"
            "(по умолчанию); «синус» — мягкий тон. Хранится в config → sound.")
        self.combo_wave.setCurrentIndex(
            0 if cfg.get("waveform", "square") == "square" else 1)
        self.player.set_waveform(cfg.get("waveform", "square")
                                 if cfg.get("waveform") in ("square", "sine")
                                 else "square")
        self.combo_wave.currentIndexChanged.connect(self._on_waveform)
        lv.addWidget(self.combo_wave)
        # селектор «Звук от…» С ВКЛАДКИ УБРАН (пакет 15, В.1): звук ВСЕГДА идёт
        # от ОСНОВНОГО фильтра, переключатель «фильтр/сглаженное» остаётся на
        # «Вариометре» (компактный дубль). Комбо живёт СКРЫТЫМ — оно источник
        # истины для синхронизации компактов (bind_source/get/set).
        self.combo_src = QtWidgets.QComboBox()
        self.combo_src.addItems(self.SRC_ITEMS)
        self.combo_src.currentIndexChanged.connect(self._on_source_combo)
        self.combo_src.setVisible(False)
        # подпись ФАКТИЧЕСКОГО источника звука: «звук: Ручной 2 · сглаж. 0.7 с»
        self.lbl_src = QtWidgets.QLabel("звук: —")
        self.lbl_src.setToolTip(
            "Что сейчас озвучивается: ОСНОВНОЙ фильтр (Авто адаптивный или\n"
            "Ручной/слот, чьи R и Q стоят в главных полях) и какая серия —\n"
            "мгновенный фильтр или сглаженное за N с (переключатель — на\n"
            "«Вариометре», рядом с громкостью).")
        lv.addWidget(self.lbl_src)
        self.lbl_status = QtWidgets.QLabel("")
        self.lbl_status.setWordWrap(True)
        lv.addWidget(self.lbl_status, stretch=1)
        self.source_desc_provider = None      # main.py: () → «Ручной 2 · сглаж. 0.7 с»
        self._src_timer = QtCore.QTimer(self)
        self._src_timer.setInterval(1000)
        self._src_timer.timeout.connect(self._refresh_src_label)
        self._src_timer.start()
        # --- пороги срабатывания (Г.1) ---
        lth = QtWidgets.QHBoxLayout()
        lvbox.addLayout(lth)
        lth.addWidget(QtWidgets.QLabel("Пороги:  подъём от"))
        self.spin_climb = StepSpinBox()
        self.spin_climb.setDecimals(2)
        self.spin_climb.setRange(-5.0, 5.0)
        self.spin_climb.setSingleStep(0.05)
        self.spin_climb.setSuffix(" м/с")
        self.spin_climb.setKeyboardTracking(False)
        self.spin_climb.setToolTip(
            "Порог ПОДЪЁМА: бипы набора звучат при вариометре выше этого значения.\n"
            "По умолчанию — ClimbToneOnThreshold активного профиля; гистерезис\n"
            "выключения (Off) — ширина из профиля. Действует на живой звук и тест.")
        self.spin_climb.valueChanged.connect(self._on_thresholds_edited)
        lth.addWidget(self.spin_climb)
        lth.addWidget(QtWidgets.QLabel("· спуск от"))
        self.spin_sink = StepSpinBox()
        self.spin_sink.setDecimals(2)
        self.spin_sink.setRange(-10.0, 2.0)
        self.spin_sink.setSingleStep(0.05)
        self.spin_sink.setSuffix(" м/с")
        self.spin_sink.setKeyboardTracking(False)
        self.spin_sink.setToolTip(
            "Порог СПУСКА: тон снижения звучит при вариометре ниже этого значения\n"
            "(обычно отрицательное). По умолчанию — SinkToneOnThreshold профиля;\n"
            "гистерезис — из Off-порога. Между порогами — тишина.")
        self.spin_sink.valueChanged.connect(self._on_thresholds_edited)
        lth.addWidget(self.spin_sink)
        btn_th_reset = QtWidgets.QPushButton("↺ из профиля")
        btn_th_reset.setToolTip("Вернуть пороги к значениям активного профиля")
        btn_th_reset.clicked.connect(self._reset_thresholds)
        lth.addWidget(btn_th_reset)
        lth.addWidget(QtWidgets.QLabel("Δ"))
        self.delta_th = make_delta_field(
            [self.spin_climb, self.spin_sink], 0.05,
            "Δ для полей порогов: на сколько меняется порог за один щелчок\n"
            "колеса/стрелок (Г.3).")
        lth.addWidget(self.delta_th)
        lth.addStretch(1)
        # человеческая подпись порогов (пакет 15, В.2): отдельной строкой,
        # значения КЛИКАБЕЛЬНЫ (клик ставит фокус в соответствующее поле)
        self.lbl_th = QtWidgets.QLabel("")
        self.lbl_th.setTextFormat(QtCore.Qt.RichText)
        self.lbl_th.setStyleSheet("color:#8a93a0;")
        self.lbl_th.setWordWrap(True)
        self.lbl_th.setToolTip(
            "Зона тишины: пока вариометр между порогом спуска и порогом подъёма,\n"
            "бипов нет. Гистерезис: звук ВКЛЮЧАЕТСЯ от порога «подъём от», а\n"
            "ВЫКЛЮЧАЕТСЯ чуть ниже (ширина — из Off-порогов профиля) — иначе у\n"
            "самого порога звук дребезжал бы вкл/выкл на каждом колебании.\n"
            "Клик по числу — фокус в поле порога.")
        self.lbl_th.linkActivated.connect(self._on_th_link)
        lvbox.addWidget(self.lbl_th)
        root.addWidget(gb_vol)

        # ---- библиотека профилей ----
        gb_lib = QtWidgets.QGroupBox("Профили звука (data\\sound_profiles)")
        ll = QtWidgets.QVBoxLayout(gb_lib)
        rowb = QtWidgets.QHBoxLayout()
        btn_act = QtWidgets.QPushButton("✔ Выбрать активным")
        btn_act.clicked.connect(self._activate_selected)
        rowb.addWidget(btn_act)
        btn_del = QtWidgets.QPushButton("🗑 Удалить")
        btn_del.clicked.connect(self._delete_selected)
        rowb.addWidget(btn_del)
        btn_imp_file = QtWidgets.QPushButton("Импорт (файл)…")
        btn_imp_file.clicked.connect(self._import_file)
        rowb.addWidget(btn_imp_file)
        btn_imp_text = QtWidgets.QPushButton("Импорт (вставить текст)…")
        btn_imp_text.clicked.connect(self._import_text)
        rowb.addWidget(btn_imp_text)
        btn_fit = QtWidgets.QPushButton("↔ Автоширина")
        btn_fit.setToolTip("Подогнать колонки таблицы под содержимое (как в «Записях»)")
        btn_fit.clicked.connect(lambda: self.tbl.autosize())
        rowb.addWidget(btn_fit)
        btn_site = QtWidgets.QPushButton("🌐 Открыть сайт профилей")
        btn_site.setToolTip(PROFILES_URL)
        btn_site.clicked.connect(lambda: webbrowser.open(PROFILES_URL))
        rowb.addWidget(btn_site)
        rowb.addStretch(1)
        ll.addLayout(rowb)
        from files_app import FileTable      # та же таблица, что в «Записях»
        self.tbl = FileTable(("Имя", "Дата", "Размер", "Статус"))
        self.tbl.setMinimumHeight(140)
        self.tbl.doubleClicked.connect(lambda *_: self._activate_selected())
        ll.addWidget(self.tbl)
        self.lbl_prof = QtWidgets.QLabel("")
        self.lbl_prof.setWordWrap(True)
        ll.addWidget(self.lbl_prof)
        root.addWidget(gb_lib, stretch=1)

        # ---- тест-полигон (Г.3: стиль пульта, крупный ползунок) ----
        gb_t = QtWidgets.QGroupBox("Тест-полигон — послушать профиль без данных")
        lt = QtWidgets.QVBoxLayout(gb_t)
        row_val = QtWidgets.QHBoxLayout()
        f_big = QtGui.QFont("Consolas")
        f_big.setStyleHint(QtGui.QFont.Monospace)
        f_big.setPixelSize(22)
        f_big.setBold(True)
        self.lbl_test_v = QtWidgets.QLabel("+0.00 м/с")
        self.lbl_test_v.setFont(f_big)
        fm = QtGui.QFontMetrics(f_big)
        self.lbl_test_v.setFixedWidth(fm.horizontalAdvance("+99.99 м/с") + 8)
        self.lbl_test_v.setAlignment(QtCore.Qt.AlignCenter)   # В.5: цифры по центру
        row_val.addWidget(self.lbl_test_v)
        f_mid = QtGui.QFont("Consolas")
        f_mid.setStyleHint(QtGui.QFont.Monospace)
        f_mid.setPixelSize(14)
        self.lbl_test = QtWidgets.QLabel("→ —")
        self.lbl_test.setFont(f_mid)
        row_val.addWidget(self.lbl_test, stretch=1)
        self.btn_sweep = QtWidgets.QPushButton("▶ Свип min→max (5 с)")
        self.btn_sweep.setToolTip("Плавно проиграть весь диапазон за 5 секунд "
                                  "(повторное нажатие — стоп)")
        self.btn_sweep.clicked.connect(self._sweep)
        row_val.addWidget(self.btn_sweep)
        lt.addLayout(row_val)
        self.sl_test = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sl_test.setRange(0, 1000)
        self.sl_test.setValue(500)
        self.sl_test.setMinimumHeight(34)
        self.sl_test.setToolTip("Потяните ползунок — тон звучит, пока он зажат")
        self.sl_test.setStyleSheet(
            "QSlider::groove:horizontal{height:10px;background:#55606d;"
            "border-radius:5px;}"
            "QSlider::handle:horizontal{background:#5aa0ff;width:24px;"
            "margin:-8px 0;border-radius:12px;}")
        self.sl_test.valueChanged.connect(self._on_test_slider)
        self.sl_test.sliderPressed.connect(lambda: setattr(self, "_testing", True))
        self.sl_test.sliderReleased.connect(self._test_release)
        lt.addWidget(self.sl_test)
        row_mm = QtWidgets.QHBoxLayout()
        row_mm.addWidget(QtWidgets.QLabel("Диапазон:  min"))
        self.spin_min = StepSpinBox()
        self.spin_min.setRange(-20, 20)
        self.spin_min.setDecimals(1)
        self.spin_min.setValue(-5.0)
        self.spin_min.setSuffix(" м/с")
        self.spin_min.valueChanged.connect(self._on_test_slider)
        row_mm.addWidget(self.spin_min)
        row_mm.addWidget(QtWidgets.QLabel("max"))
        self.spin_max = StepSpinBox()
        self.spin_max.setRange(-20, 20)
        self.spin_max.setDecimals(1)
        self.spin_max.setValue(5.0)
        self.spin_max.setSuffix(" м/с")
        self.spin_max.valueChanged.connect(self._on_test_slider)
        row_mm.addWidget(self.spin_max)
        row_mm.addStretch(1)
        lt.addLayout(row_mm)
        root.addWidget(gb_t)
        root.addStretch(1)

        self._testing = False
        self._live_v = None           # последнее значение от вариометра
        self.refresh()
        self._load_active()
        self._loading = False
        self._on_enabled(self.btn_mute.isChecked())
        self._on_volume(self.sl_vol.value())
        self._on_test_slider()

    # ------------------------------------------------------------------
    def bind_source(self, getter, setter):
        """main.py: связка с vario_app (источник звука живёт там, А.2)."""
        self.sound_source_get = getter
        self.sound_source_set = setter
        self._loading = True
        try:
            idx = self.SRC_KEYS.index(getter())
        except ValueError:
            idx = 0
        self.combo_src.setCurrentIndex(idx)
        self._loading = False
        self._sync_compacts()

    def _on_source_combo(self):
        if self._loading:
            return
        self.sound_source_set(self.SRC_KEYS[self.combo_src.currentIndex()])
        self._sync_compacts()

    def _write_default_profile(self):
        p = os.path.join(PROFILES_DIR, DEFAULT_PROFILE_NAME)
        if not os.path.exists(p):
            try:
                with open(p, "w", encoding="utf-8") as fh:
                    fh.write(DEFAULT_PROFILE_TEXT)
            except OSError:
                pass

    def _save_cfg(self):
        if self._loading:
            return
        save_config({"sound": {"enabled": self.btn_mute.isChecked(),
                               "volume": self.sl_vol.value() / 100.0,
                               "profile": self._active_name,
                               "climb_on": self._th_climb_on,
                               "sink_on": self._th_sink_on,
                               "waveform": ("square"
                                            if self.combo_wave.currentIndex() == 0
                                            else "sine")}})

    def refresh(self):
        rows = []
        try:
            for n in os.listdir(PROFILES_DIR):
                fp = os.path.join(PROFILES_DIR, n)
                if not os.path.isfile(fp):
                    continue
                st = os.stat(fp)
                rows.append({"name": n, "mtime": st.st_mtime, "size": st.st_size,
                             "path": fp,
                             "status": "✔ активен" if n == self._active_name else ""})
        except OSError:
            pass
        self.tbl.fill(rows)

    # ---- пороги (Г.1) ----
    def _effective_thresholds(self):
        """(c_on, c_off, s_on, s_off): On — поля UI (или профиль), ширина
        гистерезиса Off — из профиля (по умолчанию 0.05 / 0.10)."""
        p_c_on = p_c_off = p_s_on = p_s_off = None
        if self.profile is not None:
            p_c_on, p_c_off, p_s_on, p_s_off = self.profile.default_thresholds()
        c_on = self._th_climb_on if self._th_climb_on is not None else p_c_on
        s_on = self._th_sink_on if self._th_sink_on is not None else p_s_on
        c_gap = (p_c_on - p_c_off) if (p_c_on is not None and p_c_off is not None) else 0.05
        s_gap = (p_s_off - p_s_on) if (p_s_on is not None and p_s_off is not None) else 0.10
        c_off = (c_on - max(c_gap, 0.0)) if c_on is not None else None
        s_off = (s_on + max(s_gap, 0.0)) if s_on is not None else None
        return c_on, c_off, s_on, s_off

    def _apply_thresholds(self):
        c_on, c_off, s_on, s_off = self._effective_thresholds()
        self.player.set_thresholds(c_on, c_off, s_on, s_off)
        self._loading = True
        self.spin_climb.setValue(c_on if c_on is not None else 0.0)
        self.spin_sink.setValue(s_on if s_on is not None else 0.0)
        self._loading = False
        # человеческая подпись (В.2); числа — ссылки на свои поля
        if c_on is None and s_on is None:
            self.lbl_th.setText("Порогов нет — звучит вся шкала.")
            return
        def _lnk(v, which):
            return (f'<a href="{which}" style="text-decoration:none;">'
                    f'{v:+.2f}</a>')
        parts = []
        if s_on is not None and c_on is not None:
            parts.append(f"Звук молчит между {_lnk(s_on, 'sink')} и "
                         f"{_lnk(c_on, 'climb')} м/с.")
        if c_on is not None and c_off is not None:
            parts.append(f"Гистерезис подъёма: включение от {_lnk(c_on, 'climb')}, "
                         f"выключение ниже {c_off:+.2f}")
        if s_on is not None and s_off is not None:
            parts.append(f"спуска: включение от {_lnk(s_on, 'sink')}, "
                         f"выключение выше {s_off:+.2f}")
        src = ("пороги из профиля" if self._th_climb_on is None
               and self._th_sink_on is None else "настроено вручную")
        self.lbl_th.setText("; ".join(parts)
                            + f" (чтобы не дребезжал у порога; {src}).")

    def _on_th_link(self, which: str):
        """Клик по числу в подписи порогов → фокус в соответствующее поле (В.2)."""
        w = self.spin_climb if which == "climb" else self.spin_sink
        w.setFocus()
        w.selectAll()

    def _on_waveform(self):
        """В.4: форма волны — меандр/синус."""
        if self._loading:
            return
        self.player.set_waveform("square" if self.combo_wave.currentIndex() == 0
                                 else "sine")
        self._save_cfg()

    def _refresh_src_label(self):
        """В.1: подпись фактического источника звука («звук: Ручной 2 · сглаж. 0.7 с»)."""
        if self.source_desc_provider is None:
            return
        try:
            txt = "звук: " + self.source_desc_provider()
        except Exception:
            return
        if self.lbl_src.text() != txt:
            self.lbl_src.setText(txt)

    def _on_thresholds_edited(self):
        if self._loading:
            return
        self._th_climb_on = float(self.spin_climb.value())
        self._th_sink_on = float(self.spin_sink.value())
        self._apply_thresholds()
        self._save_cfg()

    def _reset_thresholds(self):
        self._th_climb_on = None
        self._th_sink_on = None
        self._apply_thresholds()
        self._save_cfg()

    def _load_active(self):
        p = os.path.join(PROFILES_DIR, self._active_name)
        try:
            text = open(p, "r", encoding="utf-8").read()
        except OSError:
            self.profile = None
            self.lbl_prof.setText(f"Активный профиль {self._active_name} не читается.")
            self.player.set_profile(None)
            return
        self.profile = ToneProfile(text, self._active_name)
        self.player.set_profile(self.profile)
        self._apply_thresholds()
        pts = self.profile.points
        th = self.profile.thresholds
        th_txt = (f"; пороги профиля: подъём от {th.get('climbToneOnThreshold')}, "
                  f"спуск от {th.get('sinkToneOnThreshold')}"
                  if th else "; порогов в профиле нет")
        extra = (f"; служебные параметры сохранены ({self.profile.unknown_lines} строк)"
                 if self.profile.unknown_lines else "")
        self.lbl_prof.setText(
            f"Активен: {self._active_name} — точек {len(pts)}, диапазон "
            f"{pts[0][0]:+.1f}…{pts[-1][0]:+.1f} м/с{th_txt}{extra}"
            if self.profile.valid else
            f"Профиль {self._active_name} без таблицы tone= — звук не сработает.")

    # ---- обработчики ----
    def _on_enabled(self, on: bool):
        self.btn_mute.setText("🔊 Звук ВКЛ" if on else "🔇 Звук выкл")
        self.player.set_enabled(on)
        self._update_status()
        self._save_cfg()
        self._sync_compacts()

    def _on_volume(self, val: int):
        self.player.set_volume(val / 100.0)
        self._save_cfg()
        self._sync_compacts()

    def _update_status(self):
        if self.player.error:
            self.lbl_status.setText("⚠ " + self.player.error +
                                    " — пульт работает без звука.")
            self.lbl_status.setStyleSheet("color:#c09010;")
        else:
            self.lbl_status.setText("аудио готово" if self.btn_mute.isChecked()
                                    else "звук выключен")
            self.lbl_status.setStyleSheet("color:#8a93a0;")

    def _activate_selected(self):
        n = self.tbl.selected_name()
        if not n:
            return
        self._active_name = n
        self._load_active()
        self._save_cfg()
        self.refresh()

    def _delete_selected(self):
        p = self.tbl.selected_path()
        if not p:
            return
        if QtWidgets.QMessageBox.question(
                self, "Удалить профиль?",
                f"Удалить {os.path.basename(p)}?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No) != \
                QtWidgets.QMessageBox.Yes:
            return
        try:
            os.remove(p)
        except OSError as e:
            QtWidgets.QMessageBox.warning(self, "Ошибка", str(e))
        self.refresh()

    def _import_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Импорт профиля XC Tracer", "",
            "Профили (*.txt *.cfg *.ini);;Все файлы (*)")
        if not path:
            return
        try:
            text = open(path, "r", encoding="utf-8", errors="replace").read()
        except OSError as e:
            QtWidgets.QMessageBox.warning(self, "Ошибка", str(e))
            return
        self._import_common(text, os.path.basename(path))

    def _import_text(self):
        dlg = _ImportTextDialog(self)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        self._import_common(dlg.edit.toPlainText(), dlg.name.text().strip()
                            or "profile.txt")

    def _import_common(self, text: str, name: str):
        prof = ToneProfile(text, name)
        if not prof.valid:
            QtWidgets.QMessageBox.warning(
                self, "Не профиль",
                "В тексте не нашлось ≥2 строк tone=… — это не профиль XC Tracer.")
            return
        dst = os.path.join(PROFILES_DIR, name)
        if os.path.exists(dst) and QtWidgets.QMessageBox.question(
                self, "Файл есть", f"{name} уже есть — перезаписать?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No) != \
                QtWidgets.QMessageBox.Yes:
            return
        try:
            with open(dst, "w", encoding="utf-8") as fh:
                fh.write(text)
        except OSError as e:
            QtWidgets.QMessageBox.warning(self, "Ошибка", str(e))
            return
        self.refresh()
        self.lbl_prof.setText(f"Импортирован: {name} (точек {len(prof.points)}). "
                              f"«Выбрать активным» — включить его.")

    # ---- тест-полигон ----
    def _test_value(self) -> float:
        lo, hi = self.spin_min.value(), self.spin_max.value()
        return lo + (hi - lo) * self.sl_test.value() / 1000.0

    def _on_test_slider(self):
        v = self._test_value()
        self.lbl_test_v.setText(f"{v:+.2f} м/с")
        if self.profile is not None and self.profile.valid:
            c_on, c_off, s_on, s_off = self._effective_thresholds()
            silent = not ((c_on is None and s_on is None)
                          or (c_on is not None and v >= c_on)
                          or (s_on is not None and v <= s_on))
            if silent:
                self.lbl_test.setText("→ тишина (между порогами спуска и подъёма)")
            else:
                f, cyc, duty = self.profile.lookup(v)
                if duty >= 99.5:
                    self.lbl_test.setText(f"→ {f:.0f} Гц · непрерывный тон")
                else:
                    on_ms = cyc * duty / 100.0
                    self.lbl_test.setText(
                        f"→ {f:.0f} Гц · цикл {cyc:.0f} мс · скважность "
                        f"{duty:.0f}% (звучит {on_ms:.0f} мс)")
        else:
            self.lbl_test.setText("→ нет активного профиля")
        if self._testing:
            self.player.set_vario(v)

    def _test_release(self):
        self._testing = False
        self.player.set_vario(self._live_v)

    def _sweep(self):
        """Свип min→max за 5 с (повторное нажатие — стоп)."""
        if self._sweep_timer is not None:
            self._sweep_timer.stop()
            self._sweep_timer = None
            self.btn_sweep.setText("▶ Свип min→max (5 с)")
            self._testing = False
            self.player.set_vario(self._live_v)
            return
        self._testing = True
        self.btn_sweep.setText("⏹ Стоп")
        self.sl_test.setValue(0)
        t = QtCore.QTimer(self)
        t.setInterval(50)
        state = {"k": 0}

        def tick():
            state["k"] += 1
            self.sl_test.setValue(int(1000 * state["k"] * 0.05 / 5.0))
            self._on_test_slider()
            if state["k"] * 0.05 >= 5.0:
                self._sweep()          # штатная остановка

        t.timeout.connect(tick)
        t.start()
        self._sweep_timer = t

    # ---- вход от вариометра (live и файл; зовётся ~20–30 Гц) ----
    def feed_vario(self, v):
        self._live_v = v
        if not self._testing:
            self.player.set_vario(v)

    # ---- компактный дубль для «Вариометра» ----
    def compact_widget(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        btn = QtWidgets.QToolButton()
        btn.setCheckable(True)
        btn.setChecked(self.btn_mute.isChecked())
        btn.setToolTip("Звук вариометра (дубль вкладки «Звук»)")
        btn.toggled.connect(self.btn_mute.setChecked)
        h.addWidget(btn)
        sl = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        sl.setRange(0, 100)
        sl.setValue(self.sl_vol.value())
        sl.setFixedWidth(70)
        sl.setToolTip("Громкость")
        sl.valueChanged.connect(self.sl_vol.setValue)
        h.addWidget(sl)
        combo = QtWidgets.QComboBox()
        combo.addItems(["Варио", "Сглаж."])
        combo.setToolTip("Звук от: Вариометр (фильтр) / Сглаж. N с (А.2;\n"
                         "дубль переключателя на вкладке «Звук»)")
        combo.currentIndexChanged.connect(
            lambda i: (None if self._loading else
                       self.combo_src.setCurrentIndex(i)))
        h.addWidget(combo)
        self._compacts.append((btn, sl, combo))
        self._sync_compacts()
        return w

    def _sync_compacts(self):
        on = self.btn_mute.isChecked()
        src_idx = self.combo_src.currentIndex()
        for (btn, sl, combo) in self._compacts:
            btn.blockSignals(True)
            btn.setChecked(on)
            btn.setText("🔊" if on else "🔇")
            btn.blockSignals(False)
            sl.blockSignals(True)
            sl.setValue(self.sl_vol.value())
            sl.blockSignals(False)
            combo.blockSignals(True)
            combo.setCurrentIndex(src_idx)
            combo.blockSignals(False)

    def apply_theme(self, pal: dict):
        pass                                  # общая тема окна красит сама

    def close_audio(self):
        self.player.close()
