# -*- coding: utf-8 -*-
"""Пакет 14 — офскрин-тесты UI-логики и формата калибровки v2:

  А.3  курсор пересчитывается при смене N сглаживания и перепрогоне файла;
  А.4  кнопка «A» (autoRange вьюбокса) ↔ комбо Y; правка min/max в Авто → Ручной;
  А.5  слоты: серии в файле, «Сделать основным» переносит R/Q в основной;
  А.6  галочка RTS видна только в файловом режиме;
  Б.2  device_calibration: миграция v1→v2, migrate_compass_use;
  Б.3  «Посчитать RANSAC по собранному» НЕ трогает аксель/гиро/загруженный файл;
       «Сохранить калибровку прибора» пишет обе секции и наследует прежние.

Запуск: PYTHONIOENCODING=utf-8 QT_QPA_PLATFORM=offscreen python scripts/pack14_ui_test.py
"""
import json
import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pc"))

import numpy as np  # noqa: E402
from PySide6 import QtWidgets  # noqa: E402

import vario_app  # noqa: E402
vario_app.save_config = lambda *a, **k: None
import device_calibration as devcal  # noqa: E402
import calib_app  # noqa: E402
import mag_ekf  # noqa: E402

FILE = os.path.join(ROOT, "data", "session_2026-07-04_14-51-44.csv")
OK = True


def check(name, cond, detail=""):
    global OK
    print(f"  {'✓' if cond else '✗'} {name}{('  — ' + detail) if detail else ''}")
    OK &= bool(cond)


def test_devcal():
    print("== Б.2: device_calibration (миграция v1→v2) ==")
    v1 = {"format": "variopro_device_calibration", "version": 1,
          "created": "2026-07-04 14:42:55", "gyro_bias": [0, 0, 0],
          "accel": {"offset": [0, 0, 0], "scales": [1, 1, 1]},
          "mag": {"model": "ellipsoid", "hard_iron": [1, 2, 3],
                  "soft_iron": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                  "target_F_uT": 53.0, "residual_pct": 5.5},
          "mag_source": "raw_uncalibrated"}
    d = devcal.normalize(v1)
    check("v1(mag,raw) → v2.mag_raw", "mag_raw" in d and "mag" not in d)
    v1a = dict(v1)
    v1a["mag_source"] = "android_calibrated"
    d2 = devcal.normalize(v1a)
    check("v1(mag,android) → v2.mag_android", "mag_android" in d2)
    sec = devcal.mag_section(d, "raw")
    check("mag_section: V/M/F читаются", sec is not None
          and sec["F"] == 53.0 and sec["V"].shape == (3,))
    check("mag_section отсутствующего источника → None",
          devcal.mag_section(d, "android") is None)
    check("migrate_compass_use: saved+нет источника → ransac@android",
          devcal.migrate_compass_use({"compass_use": "saved"}) == "ransac@android")
    check("migrate: live_ekf + raw → live@raw",
          devcal.migrate_compass_use({"compass_use": "live_ekf",
                                      "compass_mag_source": "raw_uncalibrated"})
          == "live@raw")
    check("migrate: новый формат проходит как есть",
          devcal.migrate_compass_use({"compass_use": "ransac@raw"}) == "ransac@raw")


def synth_mag_cloud(V, n=1200, F=53.0, seed=5):
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return d * F + np.asarray(V, float) + rng.normal(0, 0.4, (n, 3))


def test_calib_b3(tmp):
    print("\n== Б.3: живой RANSAC не трогает аксель/гиро/файл; сохранение v2 ==")
    win = calib_app.CalibWindow()
    win.load_demo()
    QtWidgets.QApplication.processEvents()
    acc_before = win._acc_result
    gyro_before = win._gyro_bias
    magpts_before = win._mag_pts
    check("демо загрузилось (аксель есть)", acc_before is not None)
    # наполняем живые буферы синтетикой (два источника с разным железом)
    t = np.arange(1200) / 100.0
    for src, V in (("raw", [1200, -40, 25]), ("android", [0.5, -0.2, 0.1])):
        cloud = synth_mag_cloud(V)
        b = win._live_buf[src]
        b["t"] = list(t)
        b["mag"] = cloud.tolist()
        b["gyro"] = [[0.0, 0.0, 0.0]] * len(t)
    win.live_compute_ransac()
    check("аксель НЕ сброшен", win._acc_result is acc_before)
    check("гироскоп НЕ сброшен", win._gyro_bias is gyro_before)
    check("mag-точки загруженного файла НЕ подменены",
          win._mag_pts is magpts_before)
    ok_raw = win._live_ransac["raw"] is not None
    ok_and = win._live_ransac["android"] is not None
    check("кандидаты RANSAC посчитаны для ОБОИХ источников", ok_raw and ok_and)
    if ok_raw:
        res = win._live_ransac["raw"][0]
        errV = float(np.linalg.norm(res.V - [1200, -40, 25]))
        check("RANSAC@raw нашёл железо", errV < 2.0, f"|ΔV| = {errV:.2f} мкТл")
    # --- сохранение v2 в ПЕСОЧНИЦУ (реальный calibration.json не трогаем) ---
    calib_app.PC_DIR = tmp
    calib_app.DEVCAL_DIR = os.path.join(tmp, "arch")
    devcal.DEVICE_CALIB_PATH = os.path.join(tmp, "calibration.json")
    # «прежняя активная» с аксель-секцией — проверим наследование
    prev = {"format": "variopro_device_calibration", "version": 2,
            "created": "2026-07-01 10:00:00",
            "gyro_bias": [0.1, 0.2, 0.3],
            "accel": {"offset": [9, 9, 9], "scales": [1, 1, 1], "target_g": 9.8}}
    json.dump(prev, open(devcal.DEVICE_CALIB_PATH, "w", encoding="utf-8"))
    win2 = calib_app.CalibWindow()          # без загруженного файла
    win2._live_ransac = win._live_ransac    # кандидаты живого сбора
    win2.save_device_calibration()
    saved = json.load(open(devcal.DEVICE_CALIB_PATH, encoding="utf-8"))
    check("v2: записаны ОБЕ секции", "mag_raw" in saved and "mag_android" in saved)
    check("v2: version = 2", saved.get("version") == 2)
    check("аксель унаследован из прежней активной",
          saved.get("accel", {}).get("offset") == [9, 9, 9])
    check("гироскоп унаследован", saved.get("gyro_bias") == [0.1, 0.2, 0.3])
    win.deleteLater()
    win2.deleteLater()


def test_vario_ui():
    print("\n== А.3/А.4/А.5/А.6: вариометр (offscreen) ==")
    win = vario_app.VarioApp()
    win._save_config = lambda *a, **k: None
    win.combo_source.setCurrentText("CSV-файл")
    assert win._load_file_full(FILE)
    QtWidgets.QApplication.processEvents()
    # --- А.6: видимость RTS ---
    check("А.6: галочка RTS видна в файловом режиме", not win.chk_rts.isHidden())
    win.combo_source.setCurrentText("Поток (Bluetooth/симулятор)")
    check("А.6: в потоке галочка RTS скрыта", win.chk_rts.isHidden())
    win.combo_source.setCurrentText("CSV-файл")
    # --- А.3: курсор пересчитывается при смене N ---
    t_mid = float(win._file["t"][len(win._file["t"]) // 2])
    win._set_cursor(t_mid)
    before = win.lbl_cur_smooth.text()
    win.spin_smooth.setValue(8.0)          # смена N → пересчёт серий и курсора
    after = win.lbl_cur_smooth.text()
    check("А.3: «Сглаж.» в строке курсора пересчиталось при смене N",
          before != after and after != "—", f"{before} → {after}")
    win.spin_smooth.setValue(1.0)
    # --- А.4: правка min в режиме Авто → комбо переходит в Ручной ---
    win.ycombo_vario.setCurrentIndex(0)     # Авто
    QtWidgets.QApplication.processEvents()
    win.yspin_min_vario.setValue(-7.5)      # ручная правка поля
    check("А.4: правка min переводит комбо в «Ручной»",
          win._y_mode("vario") == "manual")
    # --- А.4: кнопка «A» (autoRange вьюбокса) → комбо в «Авто» ---
    win.plot_vario.getViewBox().enableAutoRange()   # то, что делает кнопка «A»
    QtWidgets.QApplication.processEvents()
    check("А.4: autoRange возвращает комбо в «Авто (по окну)»",
          win._y_mode("vario") == "auto")
    # --- А.5: слоты ---
    win.radio_manual.setChecked(True)       # → перепрогон файла в Ручном
    QtWidgets.QApplication.processEvents()
    f = win._file
    check("А.5: серии слотов и эталона Авто посчитаны",
          f.get("v_s1") is not None and np.isfinite(f["v_s1"]).any()
          and f.get("v_auto") is not None and np.isfinite(f["v_auto"]).any())
    win._slot_ui["s1"]["R"].setValue(0.01)
    QtWidgets.QApplication.processEvents()
    win._slot_metric_next = 0.0
    win._slot_metrics_update()
    met = win._slot_ui["s1"]["met"].text()
    # пакет 15 (Ж.2): подпись метрики теперь «Δ от Авто: N%» / «Δ: X м/с»
    check("А.5: живая метрика слота считается", "Δ" in met and "—" not in met,
          met)
    win._slot_make_main("s1")
    check("А.5: «Сделать основным» перенёс R в основной",
          abs(win.manual_params["R"] - 0.01) < 1e-9,
          f"R = {win.manual_params['R']}")
    # --- курсор пережил перепрогоны и показывает слоты ---
    win._set_cursor(t_mid)
    win._slots["s1"]["show"] = True
    win._refresh_cursor()
    check("А.5: строка курсора дополняется значением слота",
          "Р1:" in win.lbl_cur_slots.text(), win.lbl_cur_slots.text())
    win.deleteLater()


def main():
    app = QtWidgets.QApplication([])
    tmp = tempfile.mkdtemp(prefix="pack14_")
    test_devcal()
    test_calib_b3(tmp)
    test_vario_ui()
    print("\nUI-ТЕСТЫ ПРОЙДЕНЫ ✓" if OK else "\nUI-ТЕСТЫ: ПРОВАЛЫ ✗")
    raise SystemExit(0 if OK else 1)


if __name__ == "__main__":
    main()
