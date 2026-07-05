# -*- coding: utf-8 -*-
"""
main.py
=======
ГЛАВНОЕ ОКНО-ПУЛЬТ VarioPro3 (единая «научная диспетчерская»).

Объединяет два уже готовых раздела в одном окне с вкладками:
  • «Вариометр»  — живые графики высоты/вариометра + панель параметров фильтра
                   (класс VarioApp из vario_app.py);
  • «Калибровка» — две 3D-сферы магнитометра/акселерометра + панель смещений
                   (класс CalibWindow из calib_app.py).

ВАЖНО: существующая логика НЕ переписана. Мы переиспользуем готовые окна как
панели внутри вкладок. Переключение вкладок — одним кликом, без перезапуска.

Тема (Светлая/Тёмная) переключается кнопкой в правом верхнем углу и меняет
оформление ВСЕГО окна, включая фон графиков вариометра и фон 3D-сфер калибровки.
Выбор сохраняется в config.json и восстанавливается при запуске.

Навигация расширяемая: новые разделы добавляются одной строкой через
self.add_section(...). Пустые вкладки пока не добавляем.

Запуск (один на всё):
    python pc/main.py
    python pc/main.py --selftest    # снять скриншоты обеих тем и выйти
"""

from __future__ import annotations

import os
import sys
import json
import argparse

from PySide6 import QtCore, QtGui, QtWidgets

# КРАШ-ЛОГ (пакет 14, Д.1) — включаем ДО импорта тяжёлых модулей, чтобы
# поймать и падение на импорте; лог: data\logs\variopro_ГГГГММДД.log
import crashlog
_LOG_PATH = crashlog.setup()

# переиспользуем готовые модули как есть
import vario_app
import calib_app
import link_app
import files_app
import sound_app

PC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(PC_DIR)
DOCS_DIR = os.path.join(ROOT, "docs")
CONFIG_PATH = vario_app.CONFIG_PATH   # тот же config.json, что и у вариометра
APP_ICON = os.path.join(ROOT, "assets", "logo.png")


# ----------------------------------------------------------------------
# ТЕМЫ ОФОРМЛЕНИЯ
# Каждая тема — это:
#   qss      — таблица стилей всего окна (рамка, кнопки, вкладки);
#   plot_bg/plot_fg — фон и цвет осей/подписей графиков вариометра;
#   gl_bg    — фон 3D-сцен калибровки;
#   info_fg/accent/leg_raw/leg_cal — цвета текста/заголовков/легенды калибровки.
# ----------------------------------------------------------------------
_DARK_QSS = """
    QWidget { background-color:#14161a; color:#d6dbe1; }
    QMainWindow, QSplitter::handle { background-color:#14161a; }
    QLabel { color:#d6dbe1; }
    QPushButton { background:#222a35; border:1px solid #3a4554; padding:6px 12px; border-radius:4px; color:#d6dbe1; }
    QPushButton:hover { background:#2c3744; }
    QPushButton:disabled { color:#6b7480; border-color:#2a313b; }
    QComboBox, QDoubleSpinBox, QSpinBox, QLineEdit { background:#1b212a; border:1px solid #3a4554; padding:3px; color:#d6dbe1; }
    QCheckBox, QRadioButton { color:#d6dbe1; }
    QGroupBox { border:1px solid #2a313b; margin-top:8px; }
    QGroupBox::title { subcontrol-origin: margin; left:8px; padding:0 4px; }
    QSlider::groove:horizontal { height:6px; background:#2a323d; border-radius:3px; }
    QSlider::handle:horizontal { background:#5aa0ff; width:14px; margin:-5px 0; border-radius:7px; }
    QStatusBar { color:#aeb6c0; }
    QTabWidget::pane { border-top:1px solid #2a3340; }
    QTabBar::tab { background:#1b212a; color:#a8b2bd; padding:8px 20px; margin-right:2px;
                   border:1px solid #2a3340; border-bottom:none;
                   border-top-left-radius:5px; border-top-right-radius:5px; }
    QTabBar::tab:selected { background:#2c3744; color:#eaf2ff; }
    QTabBar::tab:hover { background:#27313c; }
    /* таблицы (вкладка «Записи»): тёмный фон + выделение ЦЕЛОЙ строки */
    QTableWidget { background:#10141a; alternate-background-color:#161b23;
                   color:#d6dbe1; gridline-color:#2a313b;
                   selection-background-color:#2f5faa; selection-color:#eaf2ff; }
    QTableWidget::item:selected { background:#2f5faa; color:#eaf2ff; }
    QHeaderView::section { background:#222a35; color:#d6dbe1; padding:4px 6px;
                           border:1px solid #2a313b; }
    QTableCornerButton::section { background:#222a35; border:1px solid #2a313b; }
    QProgressBar { background:#1b212a; border:1px solid #3a4554; border-radius:3px;
                   text-align:center; color:#d6dbe1; }
    QProgressBar::chunk { background:#2f5faa; }
    QScrollArea { border:none; }
"""

_LIGHT_QSS = """
    QWidget { background-color:#f3f4f6; color:#1a1d22; }
    QMainWindow, QSplitter::handle { background-color:#f3f4f6; }
    QLabel { color:#1a1d22; }
    QPushButton { background:#e6e9ee; border:1px solid #b9c0cb; padding:6px 12px; border-radius:4px; color:#1a1d22; }
    QPushButton:hover { background:#dde2e9; }
    QPushButton:disabled { color:#9aa1ab; border-color:#d2d7de; }
    QComboBox, QDoubleSpinBox, QSpinBox, QLineEdit { background:#ffffff; border:1px solid #b9c0cb; padding:3px; color:#1a1d22; }
    QCheckBox, QRadioButton { color:#1a1d22; }
    QGroupBox { border:1px solid #c7ccd4; margin-top:8px; }
    QGroupBox::title { subcontrol-origin: margin; left:8px; padding:0 4px; }
    QSlider::groove:horizontal { height:6px; background:#c7ccd4; border-radius:3px; }
    QSlider::handle:horizontal { background:#2f6fd0; width:14px; margin:-5px 0; border-radius:7px; }
    QStatusBar { color:#41474f; }
    QTabWidget::pane { border-top:1px solid #c7ccd4; }
    QTabBar::tab { background:#e6e9ee; color:#454b54; padding:8px 20px; margin-right:2px;
                   border:1px solid #c7ccd4; border-bottom:none;
                   border-top-left-radius:5px; border-top-right-radius:5px; }
    QTabBar::tab:selected { background:#ffffff; color:#10131a; }
    QTabBar::tab:hover { background:#eef1f5; }
    /* таблицы (вкладка «Записи»): светлый фон + выделение ЦЕЛОЙ строки */
    QTableWidget { background:#ffffff; alternate-background-color:#f2f4f7;
                   color:#1a1d22; gridline-color:#d5dae1;
                   selection-background-color:#cfe0f7; selection-color:#10131a; }
    QTableWidget::item:selected { background:#cfe0f7; color:#10131a; }
    QHeaderView::section { background:#e6e9ee; color:#1a1d22; padding:4px 6px;
                           border:1px solid #c7ccd4; }
    QTableCornerButton::section { background:#e6e9ee; border:1px solid #c7ccd4; }
    QProgressBar { background:#ffffff; border:1px solid #b9c0cb; border-radius:3px;
                   text-align:center; color:#1a1d22; }
    QProgressBar::chunk { background:#2f6fd0; }
    QScrollArea { border:none; }
"""

THEMES = {
    "dark": {
        "qss": _DARK_QSS,
        "plot_bg": "#0b0d10", "plot_fg": "#d6dbe1",
        "gl_bg": "#0b0d10",
        "info_fg": "#d6dbe1", "accent": "#cfe3ff",
        "leg_raw": "#ff5555", "leg_cal": "#55ff77",
    },
    "light": {
        "qss": _LIGHT_QSS,
        "plot_bg": "w", "plot_fg": "#202020",
        "gl_bg": "#e9edf2",
        "info_fg": "#1a1d22", "accent": "#15457f",
        "leg_raw": "#cc2b2b", "leg_cal": "#1f9d4d",
    },
}


def read_theme() -> str:
    """Прочитать выбранную тему из config.json (по умолчанию тёмная)."""
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
        t = cfg.get("theme")
        return t if t in ("light", "dark") else "dark"
    except (FileNotFoundError, ValueError, OSError, AttributeError):
        return "dark"


def write_theme(theme: str):
    """Сохранить тему в config.json (merge не затирает остальные настройки)."""
    vario_app.save_config({"theme": theme})


class MainConsole(QtWidgets.QMainWindow):
    """Главное окно: вкладки + переключатель темы."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("VarioPro3 — пульт (диспетчерская)")
        self.resize(1320, 900)
        if os.path.exists(APP_ICON):
            self.setWindowIcon(QtGui.QIcon(APP_ICON))

        # вкладочная навигация
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setMovable(False)
        self.setCentralWidget(self.tabs)

        # тумблер темы в правом верхнем углу (рядом с вкладками)
        self.theme = read_theme()
        self.btn_theme = QtWidgets.QPushButton()
        self.btn_theme.setToolTip("Переключить светлую/тёмную тему всего окна")
        self.btn_theme.clicked.connect(self.toggle_theme)
        self.tabs.setCornerWidget(self.btn_theme, QtCore.Qt.TopRightCorner)

        # --- раздел «Вариометр» ---
        self.vario = vario_app.VarioApp()
        self.add_section(self.vario, "Вариометр")

        # --- раздел «Калибровка» ---
        self.calib = calib_app.CalibWindow()
        self.add_section(self.calib, "Калибровка")
        # «высота из барометра» в панели эталонов берётся из вкладки «Вариометр»
        self.calib.set_baro_provider(self.vario.last_baro_altitude)
        # живой сбор калибровки: доступ к потоку вариометра (подписка вместо
        # второго подключения — телефон принимает только одно) и к его URL
        self.calib.set_stream_providers(
            lambda: self.vario.worker,
            lambda: self.vario.edit_stream_url.text().strip())
        # Б.5: «Компас использует» — live-EKF калибровка для компаса вариометра
        # и обратная связь при переключении режима на вкладке «Калибровка»
        self.vario.live_mag_provider = self.calib.live_mag_cal
        self.calib.compass_use_changed_cb = self.vario.set_compass_use
        self.calib.load_demo()   # сразу показать синтетику

        # --- раздел «Связь/Задержка» (Фаза 3): качество живого потока ---
        self.link = link_app.LinkPanel(
            lambda: (self.vario.worker.source if self.vario.worker is not None else None))
        self.add_section(self.link, "Связь/Задержка")

        # --- раздел «Записи» (Фаза 3D): менеджер записей телефона и data\ ---
        self.files = files_app.FilesPanel(
            lambda: (self.vario.worker.source if self.vario.worker is not None else None),
            play_cb=self._play_csv,
            open_calib_cb=self._open_calib,
            calib_changed_cb=self._calib_changed,
            connect_cb=self.vario.connect_stream_standby,
            layout_cb=self.vario.apply_panel_layout)   # виды пульта (пакет 15, Е)
        self.add_section(self.files, "Записи")

        # --- раздел «Звук» (Фаза 4, пакет 13 блок Ж): профили + живые бипы ---
        self.sound = sound_app.SoundPanel()
        self.add_section(self.sound, "Звук")
        self.vario.sound_cb = self.sound.feed_vario          # вариометр → бипы
        # А.2 (пакет 14): источник звука (фильтр/сглаженное) живёт в vario_app
        self.sound.bind_source(lambda: self.vario.sound_source,
                               self.vario.set_sound_source)
        # В.1 (пакет 15): подпись фактического источника на вкладке «Звук»
        self.sound.source_desc_provider = self.vario.sound_source_desc
        self.vario.add_sound_compact(self.sound.compact_widget())  # дубль громкости

        # НАВИГАЦИЯ ВЫБОРА ФАЙЛОВ: «Файл…» (Вариометр) и «Загрузить файл…»
        # (Калибровка) ведут на «Записи», где нужный список мигает зелёной рамкой
        self.vario.files_nav_cb = self._nav_to_files
        self.calib.files_nav_cb = self._nav_to_files
        # «Сохранить калибровку прибора» → обновить индикатор под компасом и архив
        self.calib.calibration_saved_cb = self._calib_changed
        # при заходе на «Записи» освежаем списки (скриншоты/экспорт могли добавиться)
        self.tabs.currentChanged.connect(self._on_tab_changed)

        # применить сохранённую тему КО ВСЕМУ окну (рамка + графики + 3D)
        self.apply_theme(self.theme)

        # ЗАДЕЛ на будущее (НЕ добавляем пустые вкладки сейчас):
        #   self.add_section(SoundPanel(),   "Звук")
        #   self.add_section(MetricsPanel(), "Метрики обучения")

    def add_section(self, widget: QtWidgets.QWidget, title: str) -> int:
        """Добавить раздел-вкладку (используется и для будущих разделов)."""
        return self.tabs.addTab(widget, title)

    def _play_csv(self, path: str):
        """Открыть CSV во вкладке «Вариометр» и запустить воспроизведение
        (вызывается из «Записей»: «Открыть в Вариометре» / «Скачать и воспроизвести»)."""
        v = self.vario
        try:
            v.stop()
        except Exception:
            pass
        v.combo_source.setCurrentText("CSV-файл")
        v.csv_path = path
        if v._load_file_full(path):
            self.tabs.setCurrentWidget(v)
            v.start()

    def _open_calib(self, path: str):
        """Открыть файл во вкладке «Калибровка» (из «Записей»: «Открыть в Калибровке»)."""
        self.calib._apply(path)
        self.tabs.setCurrentWidget(self.calib)

    def _nav_to_files(self, kind: str):
        """«Файл…»/«Загрузить файл…»/«Применить готовую…»/«Виды…» → перейти на
        «Записи» и мигнуть нужным списком. Для kind='devcal' (Д.1) после
        успешного «Применить» пульт вернётся на «Калибровку» (one-shot)."""
        self.tabs.setCurrentWidget(self.files)
        self.files.refresh_local()
        self.files.highlight(kind)
        if kind == "devcal":
            self.files.after_apply_cb = (
                lambda: self.tabs.setCurrentWidget(self.calib))

    def _calib_changed(self):
        """Активная калибровка прибора сменилась (Сохранить/Применить):
        обновить индикатор под компасом и списки «Записей»."""
        self.vario.refresh_calib_indicator()
        self.files.refresh_local()

    def _on_tab_changed(self, idx: int):
        if self.tabs.widget(idx) is self.files:
            self.files.refresh_local()

    def apply_theme(self, theme: str):
        """Применить тему ко всему: рамка окна, графики вариометра, 3D-сферы калибровки."""
        pal = THEMES.get(theme, THEMES["dark"])
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.setStyleSheet(pal["qss"])     # рамка, кнопки, вкладки
        self.vario.apply_theme(pal)           # фон и оси графиков вариометра
        self.calib.apply_theme(pal)           # фон 3D-сцен и цвета панели
        self.link.apply_theme(pal)            # график задержки
        self.files.apply_theme(pal)           # вкладка «Записи»
        self.sound.apply_theme(pal)           # вкладка «Звук»
        self.theme = theme
        self.btn_theme.setText("Тема: Тёмная  ☾" if theme == "dark" else "Тема: Светлая  ☀")

    def toggle_theme(self):
        """Кнопка: переключить тему и сохранить выбор."""
        self.apply_theme("light" if self.theme == "dark" else "dark")
        write_theme(self.theme)

    def closeEvent(self, event):
        """При закрытии пульта корректно гасим раздел «Вариометр» (поток + настройки)."""
        try:
            self.vario.stop()
            self.vario._save_config()
        except Exception:
            pass
        try:
            self.sound.close_audio()   # закрыть аудио-поток (блок Ж)
        except Exception:
            pass
        super().closeEvent(event)


def main():
    parser = argparse.ArgumentParser(description="VarioPro3 — главный пульт")
    parser.add_argument("--selftest", action="store_true",
                        help="снять скриншоты обеих тем (обе вкладки) в docs/ и выйти")
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    if os.path.exists(APP_ICON):
        app.setWindowIcon(QtGui.QIcon(APP_ICON))
    win = MainConsole()       # тема применяется внутри (читается из config.json)
    win.show()

    if args.selftest:
        # Запускаем быструю симуляцию вариометра, чтобы графики наполнились,
        # затем снимаем обе вкладки в ТЁМНОЙ и в СВЕТЛОЙ теме (4 кадра).
        from sensor_source import SimSource
        original_theme = win.theme
        win.vario.combo_source.setCurrentText("Симуляция")
        win.vario.worker = vario_app.SourceWorker(
            SimSource(dt=vario_app.DT, speed=25, loop=False))
        win.vario.worker.sampleReady.connect(win.vario._on_sample)
        win.vario.worker.errorOccurred.connect(win.vario._show_error)
        win.vario.worker.finished.connect(win.vario._on_worker_finished)
        win.vario.worker.start()
        win.vario._view_refresh = True
        win.vario.btn_start.setEnabled(False)
        win.vario.btn_stop.setEnabled(True)
        os.makedirs(DOCS_DIR, exist_ok=True)

        def cap(name):
            win.grab().save(os.path.join(DOCS_DIR, name))

        def step1():
            win.apply_theme("dark")
            win.tabs.setCurrentWidget(win.vario)
            QtCore.QTimer.singleShot(900, step2)

        def step2():
            cap("theme_dark_vario.png")
            win.tabs.setCurrentWidget(win.calib)
            QtCore.QTimer.singleShot(1300, step3)

        def step3():
            cap("theme_dark_calib.png")
            win.apply_theme("light")
            win.tabs.setCurrentWidget(win.vario)
            QtCore.QTimer.singleShot(1000, step4)

        def step4():
            cap("theme_light_vario.png")
            win.tabs.setCurrentWidget(win.calib)
            QtCore.QTimer.singleShot(1300, step5)

        def step5():
            cap("theme_light_calib.png")
            win.apply_theme(original_theme)
            write_theme(original_theme)
            print("Сняты 4 скриншота в docs/:")
            print("  theme_dark_vario.png, theme_dark_calib.png")
            print("  theme_light_vario.png, theme_light_calib.png")
            win.vario.stop()
            app.quit()

        QtCore.QTimer.singleShot(3500, step1)   # дать данным вариометра накопиться

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
