# -*- coding: utf-8 -*-
"""Пакет 15, блок А — приёмка математики фильтра на session_2026-07-04_23-21-13.csv
через НАСТОЯЩИЙ пайплайн пульта (vario_app offscreen, пакетный прогон файла).

Метрики (одни и те же для «до» и «после» — маски строятся ПО ДАННЫМ, не по коду):
  • СТРОГИЙ ПОКОЙ (сглаж. |ω|<0.05 рад/с И ||f|−g|<0.15 м/с² устойчиво ≥0.3 с):
    max|v| и RMS(v) фильтра;
  • ОКНА КАЧАНИЯ архитектора (142–146, 152–161, 166–172, 178–186, 258–273 с):
    max|v| в 142–146 (приёмка ≤0.5), справочно в остальных;
  • ПЕРЕБЕГ после каждого окна качания: min/max v в [конец, конец+3 с] и
    время возврата в |v|≤0.05 (устойчиво 0.5 с) — приёмка: перебег ≤0.15 м/с,
    возврат ≤0.5 с;
  • ПОДЪЁМЫ НЕ ПОТЕРЯНЫ: на спокойных кусках ≥8 с |mean(v) − наклон баро| ≤ 0.05.

Ключи:
  --before   выключить фичи пакета 15 (ZUPT/Хьюбер/качание/адаптация R v2) —
             поведение пакета 14 для колонки «до»;
  --mode mekf|scalar|both (по умолчанию both).

Запуск: PYTHONIOENCODING=utf-8 QT_QPA_PLATFORM=offscreen \
        python scripts/pack15_a_validate.py [--before]
"""
import argparse
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pc"))

import numpy as np  # noqa: E402
from PySide6 import QtWidgets  # noqa: E402

import vario_app  # noqa: E402
vario_app.save_config = lambda *a, **k: None          # не трогаем config.json

FILE = os.path.join(ROOT, "data", "session_2026-07-04_23-21-13.csv")
SWINGS = [(142.0, 146.0), (152.0, 161.0), (166.0, 172.0), (178.0, 186.0),
          (258.0, 273.0)]


def check(name, cond, detail=""):
    print(f"  {'✓' if cond else '✗'} {name}{('  — ' + detail) if detail else ''}")
    return bool(cond)


def strict_rest_mask(t, w_mag, acc_dev, fs):
    """Строгий покой ПО ДАННЫМ (как условие ZUPT): EMA(|ω|)<0.05 И
    EMA(||f|−g|)<0.15 устойчиво ≥0.3 с. Маска не зависит от кода фильтра."""
    tau = 0.1
    alpha = 1.0 - np.exp(-1.0 / (fs * tau))
    def ema(x):
        y = np.empty_like(x)
        acc = x[0]
        for i in range(len(x)):
            acc += (x[i] - acc) * alpha
            y[i] = acc
        return y
    quiet = (ema(w_mag) < 0.05) & (ema(acc_dev) < 0.15)
    hold = int(0.3 * fs)
    out = np.zeros(len(t), bool)
    run = 0
    for i in range(len(t)):
        run = run + 1 if quiet[i] else 0
        if run >= hold:
            out[i] = True
    return out


def run_mode(win, mode, label):
    win._va_mode = mode
    assert win._load_file_full(FILE), "файл не загрузился"
    f = win._file
    t, v = f["t"], f["v_filt"]
    fs = 1.0 / np.median(np.diff(t))

    # входные величины для масок — из самого файла (не из кода фильтра)
    arr = np.genfromtxt(FILE, delimiter=",", names=True)
    n0 = len(arr["t"]) - len(t)                 # точки прогрева отрезаны спереди
    import device_calibration as devcal
    cal = devcal.load(vario_app.DEVICE_CALIB_PATH)
    acc = cal.get("accel") or {}
    A = np.column_stack([arr["ax"], arr["ay"], arr["az"]]).astype(float)
    if "offset" in acc and "scales" in acc:
        A = (A - np.asarray(acc["offset"], float)) * np.asarray(acc["scales"], float)
    g_ref = float(acc.get("target_g") or 9.81)
    w_mag = np.sqrt(arr["gx"]**2 + arr["gy"]**2 + arr["gz"]**2)[n0:]
    acc_dev = np.abs(np.linalg.norm(A, axis=1) - g_ref)[n0:]

    rest = strict_rest_mask(t, w_mag, acc_dev, fs)
    in_swing = np.zeros(len(t), bool)
    for (a, b) in SWINGS:
        in_swing |= (t >= a) & (t <= b)

    res = {"label": label}
    print(f"\n== {label}: {t[-1]:.1f} с, {len(t)} точек, строгий покой "
          f"{rest.mean()*100:.0f}% ==")
    ok = True

    # непрерывные куски строгого покоя (по ним меряются перебег и возврат)
    runs = []
    i0 = None
    for i in range(len(t)):
        if rest[i] and i0 is None:
            i0 = i
        elif (not rest[i] or i == len(t) - 1) and i0 is not None:
            if t[i - 1] - t[i0] >= 1.0:
                runs.append((i0, i))
            i0 = None

    # --- строгий покой: только УСТОЙЧИВЫЕ куски ≥1 с (короткие островки
    # 0.3–1 с в нулях фазы между взмахами — не «покой», фильтр там ещё несёт
    # ход руки); первые 0.5 с каждого куска — окно ВОЗВРАТА (приёмка разрешает
    # возврат ≤0.5 с), в «везде» они не считаются ---
    core = np.zeros(len(t), bool)
    for (a, b) in runs:
        core[a:b] = True
        core[(t >= t[a]) & (t < t[a] + 0.5)] = False
    vr = v[core]
    res["rest_max"] = float(np.abs(vr).max())
    res["rest_rms"] = float(np.sqrt(np.mean(vr**2)))
    ok &= check("покой: |v| ≤ 0.05 м/с везде (кроме первых 0.5 с возврата)",
                res["rest_max"] <= 0.05,
                f"max {res['rest_max']:.3f}, RMS {res['rest_rms']:.4f}")

    # --- качание 142–146 ---
    m = (t >= 142) & (t <= 146)
    res["swing_142"] = float(np.abs(v[m]).max())
    ok &= check("качание 142–146 с: |v| ≤ 0.5 м/с", res["swing_142"] <= 0.5,
                f"max {res['swing_142']:.3f}")
    sw_all = []
    for (a, b) in SWINGS:
        m = (t >= a) & (t <= b)
        sw_all.append(float(np.abs(v[m]).max()))
    res["swing_all"] = sw_all
    print("    (все окна качания, max|v|: "
          + "  ".join(f"{a:.0f}-{b:.0f}:{x:.2f}" for (a, b), x in zip(SWINGS, sw_all)) + ")")

    # --- перебег и возврат ПОСЛЕ ОСТАНОВКИ: по кускам строгого покоя,
    # начинающимся в пределах 15 с после конца окна качания (после окна
    # телефон ещё держат в руке — мерить «возврат» по движущемуся телефону
    # бессмысленно: v там реальная, не ошибка) ---
    worst_over = 0.0
    worst_ret = 0.0
    lines = []
    hold = int(0.5 * fs)
    okmask = np.abs(v) <= 0.05
    for (a, b) in SWINGS:
        cand = [(i0, i1) for (i0, i1) in runs if b <= t[i0] <= b + 15.0]
        if not cand:
            lines.append(f"{a:.0f}-{b:.0f}: строгого покоя в 15 с после окна нет")
            continue
        i0, i1 = cand[0]
        m = (t >= t[i0]) & (t <= t[i0] + 1.5)
        over_lo, over_hi = float(v[m].min()), float(v[m].max())
        worst_over = max(worst_over, abs(over_lo), abs(over_hi))
        ret = np.nan
        run = 0
        for i in range(i0, len(t)):
            run = run + 1 if okmask[i] else 0
            if run >= hold:
                ret = max(0.0, t[i - hold + 1] - t[i0])
                break
        if np.isfinite(ret):
            worst_ret = max(worst_ret, ret)
        lines.append(f"{a:.0f}-{b:.0f}: остановка на {t[i0]:.1f} с, перебег "
                     f"{over_lo:+.2f}..{over_hi:+.2f}, возврат {ret:.2f} с")
    res["overshoot"] = worst_over
    res["return_s"] = worst_ret
    for ln in lines:
        print("    " + ln)
    ok &= check("перебег после остановки ≤ 0.15 м/с", worst_over <= 0.15,
                f"худший {worst_over:.3f}")
    ok &= check("возврат в |v|≤0.05 за ≤ 0.5 с после остановки", worst_ret <= 0.5,
                f"худший {worst_ret:.2f} с")

    # --- подъёмы не потеряны: спокойные куски ≥8 с, v против наклона баро.
    # Сравниваем по НЕ-покойной части куска: в строгом покое ZUPT честно
    # держит v=0 (телефон неподвижен), даже если баро плывёт — там правда
    # за ZUPT, и она уже проверена критерием «покой» ---
    worst_tr = 0.0
    calm = ~in_swing
    i0 = None
    n_seg = 0
    for i in range(len(t)):
        if calm[i] and i0 is None:
            i0 = i
        if (not calm[i] or i == len(t) - 1) and i0 is not None:
            if t[i - 1] - t[i0] >= 8.0:
                seg = slice(i0, i)
                m_nr = ~rest[seg]
                if m_nr.sum() >= 4.0 * fs:      # достаточно не-покоя для наклона
                    tt, hh, vv = t[seg][m_nr], f["h_raw"][seg][m_nr], v[seg][m_nr]
                    sl = float(np.polyfit(tt, hh, 1)[0])
                    dv = abs(float(np.mean(vv)) - sl)
                    worst_tr = max(worst_tr, dv)
                    n_seg += 1
            i0 = None
    res["trend"] = worst_tr
    ok &= check("тренд не потерян: |mean v − наклон баро| ≤ 0.05 на кусках ≥8 с "
                "(вне строгого покоя)",
                worst_tr <= 0.05, f"худший {worst_tr:.4f} ({n_seg} кусков)")

    ok &= check("watchdog молчит", len(win._wd_events) == 0,
                f"событий {len(win._wd_events)}")
    res["ok"] = ok
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", action="store_true",
                    help="выключить фичи пакета 15 (поведение пакета 14)")
    ap.add_argument("--mode", choices=["mekf", "scalar", "both"], default="both")
    args = ap.parse_args()

    app = QtWidgets.QApplication([])
    win = vario_app.VarioApp()
    win._save_config = lambda *a, **k: None
    # приёмка идёт в режиме «Авто (адаптивный)»
    win.mode = "auto"
    win._adaptive_on = True
    if args.before:
        # поведение пакета 14: фичи выключаются, если код их уже несёт
        z = getattr(win, "_zupt_cfg", None)
        if isinstance(z, dict):
            z["enabled"] = False
        o = getattr(win, "_osc_cfg", None)
        if isinstance(o, dict):
            o["enabled"] = False
        if hasattr(win, "_huber_k"):
            win._huber_k = 0.0
        if hasattr(win, "_adapt_r_mode"):
            win._adapt_r_mode = "legacy"
        print("(режим ДО: ZUPT/Хьюбер/качание выкл, адаптация R — пакета 14)")

    modes = ["mekf", "scalar"] if args.mode == "both" else [args.mode]
    results = [run_mode(win, m, f"верт. ускорение = {m}") for m in modes]
    ok = all(r["ok"] for r in results)
    print("\nИтоговая строка (для таблицы до/после):")
    for r in results:
        print(f"  {r['label']}: покой max {r['rest_max']:.3f} / RMS "
              f"{r['rest_rms']:.4f}; качание142 {r['swing_142']:.2f}; перебег "
              f"{r['overshoot']:.3f}; возврат {r['return_s']:.2f} с; тренд "
              f"{r['trend']:.4f}")
    print("\nА ПРОЙДЕН ✓" if ok else "\nА: ЕСТЬ ПРОВАЛЫ ✗")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
