# -*- coding: utf-8 -*-
"""
widgets.py
==========
ОБЩИЕ ВИДЖЕТЫ ПУЛЬТА (пакет 15, блок Г).

StepSpinBox — единый числовой ввод всех вкладок:
  • цифры ПО ЦЕНТРУ;
  • колесо мыши при наведении меняет значение на шаг (штатно у Qt, оставлено);
  • ПКМ → пункт «Шаг…» — ввести шаг (дельту) с клавиатуры;
  • текущий шаг дописывается в тултип;
  • ширина — ПО СОДЕРЖИМОМУ (префикс + число + суффикс), не растянутая.

make_delta_field — маленькое видимое поле «Δ»: правит шаг сразу у НЕСКОЛЬКИХ
StepSpinBox (панели «Подбор R/Q» и «Пороги звука», блок Г.3).
"""

from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets


class StepSpinBox(QtWidgets.QDoubleSpinBox):
    """QDoubleSpinBox пульта: центр, «Шаг…» по ПКМ, шаг в тултипе, ширина по
    содержимому. Сигналы/поведение стандартные — прямая замена."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self._base_tip = ""
        self._fit_width_on = True
        # ширину пересчитываем при изменении значения (число могло удлиниться)
        self.valueChanged.connect(self._fit_width)

    # ---- тултип: базовый текст + строка про шаг ----
    def setToolTip(self, text: str) -> None:                 # noqa: N802 (Qt)
        self._base_tip = text or ""
        self._apply_tip()

    def _apply_tip(self):
        step = self.singleStep()
        extra = (f"Шаг: {step:g} (колесо мыши на поле; ПКМ → «Шаг…» — задать "
                 f"свой)")
        tip = (self._base_tip + "\n" + extra) if self._base_tip else extra
        super().setToolTip(tip)

    def setSingleStep(self, val: float) -> None:             # noqa: N802 (Qt)
        super().setSingleStep(val)
        self._apply_tip()

    # ---- ПКМ: штатное меню + «Шаг…» ----
    def contextMenuEvent(self, ev: QtGui.QContextMenuEvent) -> None:
        menu = self.lineEdit().createStandardContextMenu()
        menu.addSeparator()
        act = menu.addAction(f"Шаг… (сейчас {self.singleStep():g})")
        act.triggered.connect(self._ask_step)
        menu.exec(ev.globalPos())

    def _ask_step(self):
        val, ok = QtWidgets.QInputDialog.getDouble(
            self, "Шаг изменения", "Насколько менять значение за один щелчок "
            "(колесо/стрелки):", self.singleStep(), 1e-6, 1e6, 6)
        if ok and val > 0:
            self.setSingleStep(val)

    # ---- ширина по содержимому ----
    def set_fit_width(self, on: bool):
        """Выключить подгонку (если поле живёт в форме с общей шириной)."""
        self._fit_width_on = bool(on)
        if on:
            self._fit_width()

    def _fit_width(self):
        if not self._fit_width_on:
            return
        fm = QtGui.QFontMetrics(self.font())
        txt = self.prefix() + self.textFromValue(self.value()) + self.suffix()
        # запас: 2 символа на редактирование + стрелки спинбокса + рамки
        w = fm.horizontalAdvance(txt + "00") + 28
        self.setFixedWidth(max(64, w))

    def setSuffix(self, s: str) -> None:                      # noqa: N802 (Qt)
        super().setSuffix(s)
        self._fit_width()

    def setPrefix(self, s: str) -> None:                      # noqa: N802 (Qt)
        super().setPrefix(s)
        self._fit_width()

    def setDecimals(self, n: int) -> None:                    # noqa: N802 (Qt)
        super().setDecimals(n)
        self._fit_width()


class HeaderCard(QtWidgets.QFrame):
    """Карточка шапки вариометра (пакет 15, Е.3): тонкая строка-заголовок
    (редактируется в режиме «Компоновка»; пустой ввод → серый плейсхолдер с
    оригинальным названием) + произвольное содержимое. В режиме компоновки
    таскается мышью по сетке 8 px родительской CardsPanel."""

    def __init__(self, key: str, default_title: str, content: QtWidgets.QWidget,
                 panel: "CardsPanel"):
        super().__init__(panel)
        self.key = key
        self.default_title = default_title
        self.panel = panel
        self._drag_from = None
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(6, 2, 6, 4)
        v.setSpacing(1)
        self.title_edit = QtWidgets.QLineEdit()
        self.title_edit.setPlaceholderText(default_title)   # серый оригинал
        self.title_edit.setFrame(False)
        self.title_edit.setReadOnly(True)
        self.title_edit.setStyleSheet(
            "QLineEdit { background: transparent; border: none; "
            "color: #8a93a0; font-size: 10px; }")
        self.title_edit.setText("")                          # по умолчанию — плейсхолдер
        self.title_edit.textEdited.connect(lambda *_: panel.notify_changed())
        v.addWidget(self.title_edit)
        v.addWidget(content)

    # ---- перетаскивание в режиме компоновки ----
    def mousePressEvent(self, ev):
        if self.panel.layout_mode and ev.button() == QtCore.Qt.LeftButton:
            self._drag_from = ev.position().toPoint()
            self.raise_()
            ev.accept()
            return
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._drag_from is not None and self.panel.layout_mode:
            new = self.pos() + (ev.position().toPoint() - self._drag_from)
            g = CardsPanel.GRID
            x = max(0, round(new.x() / g) * g)
            y = max(0, round(new.y() / g) * g)
            self.move(x, y)
            ev.accept()
            return
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if self._drag_from is not None:
            self._drag_from = None
            self.panel.card_moved(self)
            ev.accept()
            return
        super().mouseReleaseEvent(ev)


class CardsPanel(QtWidgets.QWidget):
    """Контейнер карточек шапки с АБСОЛЮТНЫМ позиционированием по сетке 8 px
    (Е.3). Вне режима компоновки карточки стоят по сохранённой раскладке (или
    по автопотоку слева направо). В режиме компоновки видна «миллиметровка»,
    карточки таскаются с прилипанием, заголовки редактируются.

    layout_dict() → {"cards": {key: {"x","y","title"}}} (для data\\layouts\\);
    apply_layout(None) — заводской автопоток."""

    GRID = 8

    def __init__(self, on_change=None):
        super().__init__()
        self.cards: dict[str, HeaderCard] = {}
        self.order: list[str] = []
        self.layout_mode = False
        self._pos: dict[str, tuple] = {}     # key → (x, y); пусто = автопоток
        self._on_change = on_change
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                           QtWidgets.QSizePolicy.Fixed)

    # ---- сборка ----
    def add_card(self, key: str, title: str, content: QtWidgets.QWidget):
        card = HeaderCard(key, title, content, self)
        card.show()
        self.cards[key] = card
        self.order.append(key)
        return card

    def notify_changed(self):
        if self._on_change:
            self._on_change()

    # ---- раскладка ----
    def _flow_positions(self) -> dict:
        """Заводской автопоток: слева направо, перенос по ширине панели."""
        g = self.GRID
        width = max(self.width(), 640)
        x = y = 0
        row_h = 0
        out = {}
        for key in self.order:
            card = self.cards[key]
            if not card.isVisibleTo(self) and not self.layout_mode:
                pass                          # скрытые тоже размещаем (место стабильно)
            sz = card.sizeHint()
            w = sz.width()
            if x > 0 and x + w > width:
                x = 0
                y += ((row_h + g - 1) // g) * g + g
                row_h = 0
            out[key] = (x, y)
            x += ((w + g - 1) // g) * g + g
            row_h = max(row_h, sz.height())
        return out

    def relayout(self):
        flow = None
        bottom = 0
        for key in self.order:
            card = self.cards[key]
            if key in self._pos:
                x, y = self._pos[key]
            else:
                if flow is None:
                    flow = self._flow_positions()
                x, y = flow[key]
            card.move(int(x), int(y))
            card.resize(card.sizeHint())
            bottom = max(bottom, card.geometry().bottom())
        self.setMinimumHeight(bottom + self.GRID)
        self.setMaximumHeight(bottom + self.GRID)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if not self._pos:
            self.relayout()                   # автопоток следует за шириной

    def card_moved(self, card: HeaderCard):
        # первый же перенос фиксирует ЯВНЫЕ позиции всех карточек (иначе
        # автопоток «уедет» под ноги при следующем relayout)
        if not self._pos:
            for key in self.order:
                c = self.cards[key]
                self._pos[key] = (c.x(), c.y())
        self._pos[card.key] = (card.x(), card.y())
        self.relayout()
        self.notify_changed()

    # ---- режим компоновки ----
    def set_layout_mode(self, on: bool):
        self.layout_mode = bool(on)
        for card in self.cards.values():
            card.title_edit.setReadOnly(not on)
            card.setCursor(QtCore.Qt.SizeAllCursor if on
                           else QtCore.Qt.ArrowCursor)
            card.setStyleSheet(
                "HeaderCard { border: 1px dashed #5aa0ff; }" if on else "")
        self.update()

    def paintEvent(self, ev):
        super().paintEvent(ev)
        if not self.layout_mode:
            return
        p = QtGui.QPainter(self)               # «миллиметровка» 8 px
        pen = QtGui.QPen(QtGui.QColor(120, 140, 170, 70))
        p.setPen(pen)
        g = self.GRID
        for x in range(0, self.width(), g):
            for y in range(0, self.height(), g):
                p.drawPoint(x, y)

    # ---- сохранение/применение ----
    def layout_dict(self) -> dict:
        # позиции: явные, а если их нет — текущий автопоток (фиксируем как есть)
        pos = dict(self._pos) or {k: (self.cards[k].x(), self.cards[k].y())
                                  for k in self.order}
        return {"format": "variopro_layout", "version": 1,
                "cards": {k: {"x": int(pos[k][0]), "y": int(pos[k][1]),
                              "title": self.cards[k].title_edit.text() or None}
                          for k in self.order}}

    def apply_layout(self, obj: dict | None):
        """None = заводской вид (автопоток, оригинальные названия)."""
        self._pos = {}
        for key in self.order:
            self.cards[key].title_edit.setText("")
        if isinstance(obj, dict):
            cards = obj.get("cards", {})
            for key, sub in cards.items():
                card = self.cards.get(key)
                if card is None or not isinstance(sub, dict):
                    continue
                try:
                    g = self.GRID
                    self._pos[key] = (max(0, round(float(sub["x"]) / g) * g),
                                      max(0, round(float(sub["y"]) / g) * g))
                except (KeyError, TypeError, ValueError):
                    pass
                if sub.get("title"):
                    card.title_edit.setText(str(sub["title"]))
            # карточкам без позиции — место из автопотока, зафиксированное
            if self._pos:
                flow = self._flow_positions()
                for key in self.order:
                    self._pos.setdefault(key, flow[key])
        self.relayout()


def make_delta_field(targets, initial: float, tip: str = "") -> StepSpinBox:
    """Маленькое видимое поле «Δ» (Г.3): его значение = ШАГ всех targets.
    targets — список StepSpinBox; правка Δ сразу меняет их singleStep."""
    d = StepSpinBox()
    d.setDecimals(4)
    d.setRange(1e-6, 1e6)
    d.setValue(float(initial))
    d.setKeyboardTracking(False)
    d.setToolTip(tip or "Δ — на сколько меняется значение соседних полей за "
                        "один щелчок (колесо/стрелки).")

    def apply():
        for t in targets:
            t.setSingleStep(float(d.value()))
    d.valueChanged.connect(lambda *_: apply())
    apply()
    return d
