# -*- coding: utf-8 -*-
"""Пакет 15, блок И — регресс-тест порядка инициализации.

Краш из data\\logs\\variopro_20260704.log (17:18:41): CalibWindow.__init__ →
_update_matrix → self.lbl_matrix (виджет ещё не создан) — промежуточная сборка
пакета 14. Тест конструирует ПОЛНЫЙ MainConsole при пустом / частичном /
полном config.json и calibration.json — исключений быть не должно.

Файлы НИКА НЕ ТРОГАЮТСЯ: пути config/calibration подменяются на временные
через атрибуты модулей.

Запуск: PYTHONIOENCODING=utf-8 QT_QPA_PLATFORM=offscreen \
        python scripts/pack15_i_init.py
"""
import json
import os
import sys
import tempfile
import traceback

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pc"))

CASES = {
    "пустые файлы отсутствуют": (None, None),
    "пустой JSON {}": ({}, {}),
    "частичный config (только mode)": ({"mode": "manual"}, None),
    "частичный config (view-старый формат)": (
        {"view": {"window_sec": 10, "grid_step": 1}, "sound": {"volume": 0.2}},
        {"format": "variopro_device_calibration", "version": 1,
         "mag": {"model": "ellipsoid", "hard_iron": [1, 2, 3],
                 "soft_iron": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                 "target_F_uT": 53.0, "residual_pct": 5.0},
         "mag_source": "raw_uncalibrated"}),
    "битый JSON": ("{оборванный", "не json вовсе"),
    "полные (копии текущих)": ("COPY", "COPY"),
}


def write_case(path, content, real_path):
    if content is None:
        if os.path.exists(path):
            os.remove(path)
        return
    if content == "COPY":
        import shutil
        if os.path.exists(real_path):
            shutil.copyfile(real_path, path)
        return
    with open(path, "w", encoding="utf-8") as fh:
        if isinstance(content, str):
            fh.write(content)
        else:
            json.dump(content, fh, ensure_ascii=False)


def main():
    from PySide6 import QtWidgets
    import vario_app
    import calib_app
    import sound_app
    import files_app
    import device_calibration as devcal
    import main as main_mod

    real_cfg = vario_app.CONFIG_PATH
    real_cal = vario_app.DEVICE_CALIB_PATH
    tmp = tempfile.mkdtemp(prefix="variopro_init_")
    cfg_p = os.path.join(tmp, "config.json")
    cal_p = os.path.join(tmp, "calibration.json")

    # подменить пути ВО ВСЕХ модулях, где они скопированы константой
    vario_app.CONFIG_PATH = cfg_p
    vario_app.DEVICE_CALIB_PATH = cal_p
    calib_app.CONFIG_PATH = cfg_p
    devcal.DEVICE_CALIB_PATH = cal_p
    files_app.DEVICE_CALIB_PATH = cal_p
    main_mod.CONFIG_PATH = cfg_p
    vario_app.save_config = lambda *a, **k: None      # ничего не пишем
    sound_app.save_config = lambda *a, **k: None

    app = QtWidgets.QApplication([])
    ok = True
    for name, (cfg_c, cal_c) in CASES.items():
        write_case(cfg_p, cfg_c, real_cfg)
        write_case(cal_p, cal_c, real_cal)
        try:
            win = main_mod.MainConsole()
            # лёгкая встряска пост-инициализации: перерисовка/матрица/качество
            win.vario._redraw()
            win.calib._update_matrix()
            win.files.refresh_local()
            win.deleteLater()
            QtWidgets.QApplication.processEvents()
            print(f"  ✓ {name}")
        except Exception:
            ok = False
            print(f"  ✗ {name}:")
            traceback.print_exc()
    print("\nИ ПРОЙДЕН ✓" if ok else "\nИ: ЕСТЬ ПРОВАЛЫ ✗")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
