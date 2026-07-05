# -*- coding: utf-8 -*-
"""
link_app.py
===========
Вкладка «СВЯЗЬ/ЗАДЕРЖКА»: качество живого потока в «диспетчерском» виде.

Показывает по активному StreamSource (его создаёт вкладка «Вариометр», когда
источник = «Поток»):
  • статус подключения и БОЛЬШОЕ цветное «Качество связи» — в шапке;
  • группа «Канал»: частота пакетов (факт/номинал), джиттер, потери по seq;
  • группа «Часы (PING/PONG)»: RTT, смещение часов, ЧЕСТНАЯ задержка данных;
  • группа «GPS телефона»: последняя GPS-строка потока (только показ);
  • график честной задержки за ~30 с.

Оформление как на «Вариометре»: подпись — отдельно, значение — моноширинный
QLabel ФИКСИРОВАННОЙ ширины (по самому длинному образцу, включая слово
«синхронизация…») — цифры меняются на месте, ничего не прыгает и не режется.
Цвет по смыслу ТОЛЬКО у статуса, потерь и качества; остальные значения —
нейтральным цветом темы (обновляется в apply_theme).
"""

from __future__ import annotations

import time

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from vario_app import make_value_label


class LinkPanel(QtWidgets.QWidget):
    """Панель качества связи. source_provider() должен вернуть активный источник
    вкладки «Вариометр» (или None) — берём из него метрики, если это живой поток."""

    def __init__(self, source_provider):
        super().__init__()
        self.provider = source_provider
        self._fg = "#d6dbe1"          # нейтральный цвет значений (из темы)

        root = QtWidgets.QVBoxLayout(self)

        # --- шапка: статус слева, БОЛЬШОЕ качество справа ---
        srow = QtWidgets.QHBoxLayout()
        cap = QtWidgets.QLabel("Статус потока:")
        cap.setStyleSheet("font-size: 20px; font-weight: bold;")
        srow.addWidget(cap)
        self.lbl_status = QtWidgets.QLabel("нет активного потока")
        self.lbl_status.setStyleSheet("font-size: 20px; font-weight: bold; color: #888;")
        srow.addWidget(self.lbl_status)
        srow.addStretch(1)
        qcap = QtWidgets.QLabel("Качество связи:")
        qcap.setStyleSheet("font-size: 26px; font-weight: bold;")
        qcap.setToolTip("min(1, факт.частота/номинал) × (1 − потери) × 100.\n"
                        "Отходите с телефоном — частота падает/потери растут — процент падает.")
        srow.addWidget(qcap)
        self.val_quality = QtWidgets.QLabel("—")
        f = QtGui.QFont("Consolas")
        f.setStyleHint(QtGui.QFont.Monospace)
        f.setPixelSize(44)
        f.setBold(True)
        self.val_quality.setFont(f)
        self.val_quality.setStyleSheet("color:#888;")
        self.val_quality.setFixedWidth(
            QtGui.QFontMetrics(f).horizontalAdvance("100 %") + 8)
        self.val_quality.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        srow.addWidget(self.val_quality)
        root.addLayout(srow)

        hint = QtWidgets.QLabel(
            "Метрики появляются, когда на вкладке «Вариометр» выбран источник "
            "«Поток (Bluetooth/симулятор)» и нажат «Старт». Симулятор: "
            "python pc\\stream_simulator.py")
        hint.setStyleSheet("color: #7a7f87;")
        hint.setWordWrap(True)
        root.addWidget(hint)

        # --- значения: моноширинные, фиксированная ширина по образцу ---
        # (образец включает и числа, и слово «синхронизация…» — что длиннее)
        PX = 20
        self.val_rate = make_value_label("999.9 Гц (ном. 999)", self._fg, px=PX)
        self.val_jitter = make_value_label("9999.9 мс", self._fg, px=PX)
        self.val_loss = make_value_label("синхронизация…", "#888", px=PX)
        self.val_rtt = make_value_label("синхронизация…", self._fg, px=PX)
        self.val_offset = make_value_label("синхронизация…", self._fg, px=PX)
        self.val_delay = make_value_label("синхронизация…", self._fg, px=PX)
        self.val_gps = make_value_label("h=9999.9 м ±99.9 м, 999 с назад", self._fg, px=PX)
        self._neutral_vals = (self.val_rate, self.val_jitter, self.val_rtt,
                              self.val_offset, self.val_delay, self.val_gps)
        # val_temp/val_sensors создаются ниже (группа «Датчики телефона»);
        # они тоже нейтрального цвета — перекрашиваются в apply_theme отдельно

        def add_group(title, rows):
            gb = QtWidgets.QGroupBox(title)
            grid = QtWidgets.QGridLayout(gb)
            for i, (txt, w, tip) in enumerate(rows):
                lab = QtWidgets.QLabel(txt)
                lab.setStyleSheet("font-size: 15px;")
                lab.setToolTip(tip)
                w.setToolTip(tip)
                grid.addWidget(lab, i, 0)
                grid.addWidget(w, i, 1)
            grid.setColumnMinimumWidth(0, 250)
            grid.setColumnStretch(2, 1)
            root.addWidget(gb)
            return gb

        add_group("Канал", [
            ("Частота пакетов (окно 2 с):", self.val_rate,
             "Сколько пакетов в секунду реально доходит (в скобках — номинал из HELLO)"),
            ("Джиттер (СКО интервала):", self.val_jitter,
             "Насколько неровно приходят пакеты"),
            ("Потери (по seq, окно ~10 с):", self.val_loss,
             "Доля пакетов, не дошедших вовсе (дырки в счётчике seq)"),
        ])
        add_group("Часы (PING/PONG)", [
            ("RTT (пинг туда-обратно):", self.val_rtt,
             "Полный оборот PING→PONG по каналу (без синхронизации часов)"),
            ("Смещение часов:", self.val_offset,
             "Насколько часы отправителя впереди часов ПК (NTP-схема по 4+ обменам, медиана)"),
            ("ЧЕСТНАЯ задержка данных:", self.val_delay,
             "приём − (t_send − смещение часов): задержка без вранья от разницы часов"),
        ])
        add_group("GPS телефона", [
            ("Последний фикс:", self.val_gps,
             "Последняя GPS-строка потока: высота, точность и сколько секунд назад пришла.\n"
             "В фильтр GPS пока не подключён — только показ (задел для Фазы 5)."),
        ])

        # --- Датчики телефона (SENSORS, пакет 15 З.1) + температура (TEMP) ---
        gb_sens = QtWidgets.QGroupBox("Датчики телефона (строки SENSORS после HELLO)")
        vs = QtWidgets.QVBoxLayout(gb_sens)
        self.val_sensors = QtWidgets.QLabel("—")
        self.val_sensors.setWordWrap(True)
        self.val_sensors.setToolTip(
            "Метаданные датчиков, присланные телефоном после HELLO: имя, вендор,\n"
            "разрешение, диапазон, минимальный период. По имени ПК подбирает\n"
            "паспортные шумы (приоры) — docs/sensor_datasheets.md; старые прошивки\n"
            "строк SENSORS не шлют (нужен свежий APK) — тогда здесь прочерк.")
        f_s = QtGui.QFont("Consolas")
        f_s.setStyleHint(QtGui.QFont.Monospace)
        self.val_sensors.setFont(f_s)
        vs.addWidget(self.val_sensors)
        trow = QtWidgets.QHBoxLayout()
        tcap = QtWidgets.QLabel("Температура телефона (TEMP):")
        tcap.setStyleSheet("font-size: 15px;")
        tcap.setToolTip(
            "Строка TEMP раз в ~5 с — только если у телефона ЕСТЬ датчик\n"
            "температуры среды (на S23 через Android API его обычно нет —\n"
            "тогда честный прочерк). В фильтр НЕ вводится: температурные\n"
            "дрейфы компенсируются оценкой смещений (b_g, b, адаптивный R̂).")
        trow.addWidget(tcap)
        self.val_temp = make_value_label("— (датчика нет / строки не идут)",
                                         self._fg, px=PX)
        self.val_temp.setToolTip(tcap.toolTip())
        trow.addWidget(self.val_temp)
        trow.addStretch(1)
        vs.addLayout(trow)
        root.addWidget(gb_sens)

        # --- график задержки (последние ~30 с) ---
        self.plot = pg.PlotWidget()
        self.plot.setLabel("left", "Задержка, мс")
        self.plot.setLabel("bottom", "Секунды назад")
        self.plot.getAxis("left").enableAutoSIPrefix(False)
        self.plot.getAxis("bottom").enableAutoSIPrefix(False)
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.curve = self.plot.plot(pen=pg.mkPen("#2c7a2c", width=2))
        root.addWidget(self.plot, stretch=1)

        # опрос активного источника 2 раза в секунду
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(500)
        self.timer.timeout.connect(self.refresh)
        self.timer.start()

    # ------------------------------------------------------------------
    def _set_neutral(self, lbl, text):
        lbl.setText(text)
        lbl.setStyleSheet(f"color: {self._fg};")

    def _dash(self, status_txt, color="#888"):
        self.lbl_status.setText(status_txt)
        self.lbl_status.setStyleSheet(
            f"font-size: 20px; font-weight: bold; color: {color};")
        for w in self._neutral_vals:
            self._set_neutral(w, "—")
        self.val_loss.setText("—")
        self.val_loss.setStyleSheet("color:#888;")
        self.val_quality.setText("—")
        self.val_quality.setStyleSheet("color:#888;")
        self.val_sensors.setText("—")
        self._set_neutral(self.val_temp, "—")
        self.curve.setData([], [])

    def refresh(self):
        src = self.provider() if self.provider else None
        if src is None or not getattr(src, "live", False):
            self._dash("нет активного потока")
            return
        # статус с цветом: зелёный подключено, оранжевый переподключение
        st = src.status
        color = "#2c7a2c" if st == "подключено" else "#e67e22"
        self.lbl_status.setText(st)
        self.lbl_status.setStyleSheet(
            f"font-size: 20px; font-weight: bold; color: {color};")

        m = src.link_metrics()               # ЕДИНЫЕ числа (те же, что у мини-индикатора)
        now = time.time()
        events = list(src.recv_events)
        win = [(w, ts, sq) for (w, ts, sq) in events if now - w <= 2.0]

        nom = f" (ном. {src.nominal_hz:.0f})" if src.nominal_hz else ""
        self._set_neutral(self.val_rate, f"{m['fact_hz']:5.1f} Гц{nom}")
        # джиттер: СКО межпакетного интервала в окне
        if len(win) >= 3:
            arr = np.array([w for (w, _, _) in win])
            jit = float(np.diff(arr).std()) * 1000.0
            self._set_neutral(self.val_jitter, f"{jit:6.1f} мс")
        else:
            self._set_neutral(self.val_jitter, "—")
        # потери: цвет по смыслу (зелёный ~0, жёлтый < 5%, красный дальше)
        if m["loss_pct"] is None:
            self.val_loss.setText("—")
            self.val_loss.setStyleSheet("color:#888;")
        else:
            lp = m["loss_pct"]
            lcol = "#2c7a2c" if lp < 0.5 else ("#c09010" if lp < 5.0 else "#c0392b")
            self.val_loss.setText(f"{lp:5.2f} %")
            self.val_loss.setStyleSheet(f"color:{lcol}; font-weight:bold;")
        # часы: пока обменов PING/PONG < 4 — «синхронизация…»
        if m["rtt_ms"] is None:
            self._set_neutral(self.val_rtt, "синхронизация…")
            self._set_neutral(self.val_offset, "синхронизация…")
            self._set_neutral(self.val_delay, "синхронизация…")
        else:
            self._set_neutral(self.val_rtt, f"{m['rtt_ms']:6.1f} мс")
            self._set_neutral(self.val_offset, f"{m['offset_s']:+10.3f} с")
            if m["delay_ms"] is not None:
                self._set_neutral(self.val_delay,
                                  f"{m['delay_ms']:4.0f} ± {m['delay_std_ms']:3.0f} мс")
            else:
                self._set_neutral(self.val_delay, "—")
        # GPS телефона: высота, точность, возраст строки (честно стареет)
        g = getattr(src, "gps_last", None)
        if g is None:
            self._set_neutral(self.val_gps, "—")
        else:
            age = now - g["wall"]
            self._set_neutral(self.val_gps,
                              f"h={g['alt']:6.1f} м ±{g['acc']:4.1f} м, {age:3.0f} с назад")
        # датчики телефона (SENSORS) + температура (TEMP) — пакет 15, З
        sens = getattr(src, "sensors", None) or {}
        rows_s = [f"{k:<5} {v.get('name', '?')} ({v.get('vendor', '?')}); "
                  f"разр. {v.get('resolution', '?')}, диапазон "
                  f"{v.get('max_range', '?')}, minDelay {v.get('min_delay_us', '?')} мкс"
                  for k, v in sens.items() if v.get("name") not in (None, "-")]
        txt_s = "\n".join(rows_s) if rows_s else (
            "— (телефон не прислал SENSORS: старый APK или поток не открыт)")
        if self.val_sensors.text() != txt_s:
            self.val_sensors.setText(txt_s)
        tl = getattr(src, "temp_last", None)
        if tl is None:
            self._set_neutral(self.val_temp, "— (датчика нет / строки не идут)")
        else:
            self._set_neutral(self.val_temp,
                              f"{tl['c']:+5.1f} °C, {now - tl['wall']:3.0f} с назад")
        # качество связи: большое цветное число
        q = m["quality_pct"]
        if q is None:
            self.val_quality.setText("—")
            self.val_quality.setStyleSheet("color:#888;")
        else:
            col = "#2c7a2c" if q >= 95 else ("#c09010" if q >= 70 else "#c0392b")
            self.val_quality.setText(f"{q:3.0f} %")
            self.val_quality.setStyleSheet(f"color:{col};")
        # график: ЧЕСТНАЯ задержка за последние 30 с (пока нет offset — сырая)
        off = m["offset_s"] or 0.0
        tail = [(w, ts) for (w, ts, _) in events if now - w <= 30.0]
        if tail:
            xs = np.array([w - now for (w, _) in tail])
            ys = np.array([(w - (ts - off)) * 1000.0 for (w, ts) in tail])
            self.curve.setData(xs, ys)
        else:
            self.curve.setData([], [])

    # ------------------------------------------------------------------
    def apply_theme(self, pal: dict):
        """Перекрасить график и нейтральные значения под тему (вызывает main.py)."""
        self._fg = pal.get("plot_fg", "#d6dbe1")
        for w in self._neutral_vals:
            w.setStyleSheet(f"color: {self._fg};")
        self.val_temp.setStyleSheet(f"color: {self._fg};")
        try:
            self.plot.setBackground(pal["plot_bg"])
            fg = pg.mkColor(pal["plot_fg"])
            for ax in ("left", "bottom"):
                a = self.plot.getAxis(ax)
                a.setTextPen(fg)
                a.setPen(fg)
                if a.label is not None:
                    a.label.setDefaultTextColor(fg)
        except Exception:
            pass
