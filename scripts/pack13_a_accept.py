# -*- coding: utf-8 -*-
"""Блок А, приёмка фикса: по каждому файлу — покой (|vario|, RMS mekf−scalar),
возврат после вращений, сохранность реальных пиков по баро; дыры разных длин.

python scripts\pack13_a_accept.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pack13_common import (ROOT, load_cfg, read_session, run_arrays,  # noqa: E402
                           baro_speed, episodes)

FILES = ["session_2026-07-04_00-38-35.csv",
         "session_2026-07-03_16-12-04.csv",
         "session_2026-07-03_15-34-53.csv"]


def rest_segments(t, rest, vb, skip_head=1.0, min_len=3.0):
    """Сегменты покоя ПО ДЕТЕКТОРУ (min_len с), в которых медиана |v_баро| < 0.1
    (телефон реально неподвижен), без первых skip_head секунд (переходные)."""
    segs = []
    for (a, b) in episodes(t, rest == 1, min_len=min_len):
        m = (t >= a) & (t <= b)
        if np.nanmedian(np.abs(vb[m])) < 0.10 and b - a > skip_head:
            segs.append((a + skip_head, b))
    return segs


def seg_mask(t, segs):
    m = np.zeros(len(t), bool)
    for (a, b) in segs:
        m |= (t >= a) & (t <= b)
    return m


def recovery_times(t, vf, rest, vb, wmask):
    """Для каждого перехода Движение→Покой после вращения: время до
    |vario_сглаж(0.5 с)| ≤ 0.05 устойчиво 1 с."""
    # сглаженный вариометр 0.5 с (скользящее среднее)
    dtm = np.median(np.diff(t))
    k = max(1, int(round(0.25 / dtm)))
    vs = np.convolve(vf, np.ones(2 * k + 1) / (2 * k + 1), mode="same")
    m = (rest == 1) & (np.abs(np.nan_to_num(vb, nan=9)) < 0.10)
    out = []
    for (a, b) in episodes(t, m, min_len=2.0):
        i0 = np.searchsorted(t, a)
        # было ли вращение перед сегментом (окно 5 с до входа)?
        pre = (t >= a - 5.0) & (t < a)
        if not wmask[pre].any():
            continue
        rec = None
        i = i0
        while t[i] <= b:
            w = (t >= t[i]) & (t <= min(t[i] + 1.0, b))
            if np.abs(vs[w]).max() <= 0.05:
                rec = t[i] - a
                break
            i += 1
            if i >= len(t):
                break
        out.append((a, rec))
    return out


def peaks_report(t, vf, vb, windows):
    """Пик по баро заданного знака в окне; пик фильтра ТОГО ЖЕ знака в ±1.5 с."""
    rep = []
    for (a, b, sgn) in windows:
        m = (t >= a) & (t <= b)
        vbs = np.nan_to_num(vb[m], nan=0.0) * sgn
        i_b = int(np.argmax(vbs))
        tb, vbp = t[m][i_b], vb[m][i_b]
        m2 = (t >= tb - 1.5) & (t <= tb + 1.5)
        i_f = int(np.argmax(vf[m2] * sgn))
        tf, vfp = t[m2][i_f], vf[m2][i_f]
        rep.append((a, b, vbp, tb, vfp, tf, tf - tb))
    return rep


def main():
    cfg = load_cfg()
    for name in FILES:
        path = os.path.join(ROOT, "data", name)
        if not os.path.exists(path):
            print(f"{name}: НЕТ файла")
            continue
        t, A, G, H = read_session(path)
        rm = run_arrays(t, A, G, H, "mekf", cfg)
        rs = run_arrays(t, A, G, H, "scalar", cfg)
        vb = baro_speed(rm["t"], rm["h"])
        wmag = np.linalg.norm(G, axis=1)
        # маска «было вращение»: |ω| > 3 рад/с (на оси t серий после прогрева)
        wmask = np.interp(rm["t"], t, wmag) > 3.0
        segs = rest_segments(rm["t"], rm["rest"], vb)
        m = seg_mask(rm["t"], segs)
        print(f"\n===== {name} ({t[-1]:.0f} с) =====")
        print(f"покой: {len(segs)} сегментов, {sum(b-a for a,b in segs):.0f} с "
              f"(без первой 1 с каждого)")
        if segs:
            vf, vf2 = rm["vf"][m], rs["vf"][m]
            means, dmeans = [], []
            for a, b in segs:
                mm = (rm["t"] >= a) & (rm["t"] <= b)
                means.append(rm["vf"][mm].mean())
                dmeans.append(rm["vf"][mm].mean() - np.nanmean(vb[mm]))
            print(f"  mekf:   |среднее по сегменту| max {max(abs(x) for x in means):.3f} м/с "
                  f"(за вычетом хода САМОГО баро: {max(abs(x) for x in dmeans):.3f}); "
                  f"мгновенно p95 |v| {np.percentile(np.abs(vf), 95):.3f}, "
                  f"max |v| {np.abs(vf).max():.3f}")
            rmsd = float(np.sqrt(np.mean((vf - vf2) ** 2)))
            print(f"  RMS(mekf − scalar) в покое: {rmsd:.4f} м/с "
                  f"{'✓' if rmsd < 0.01 else '✗ (>0.01)'}")
        recs = recovery_times(rm["t"], rm["vf"], rm["rest"], vb, wmask)
        if recs:
            worst = max((r for _, r in recs if r is not None), default=None)
            fails = [a for a, r in recs if r is None]
            print(f"  возврат ≤0.05 после вращений: {len(recs)} переходов, "
                  f"худший {worst if worst is None else f'{worst:.1f} с'}"
                  f"{'; НЕ вернулся: ' + str(fails) if fails else ''}")
        mk = rm["pipeline"].mekf
        print(f"  MEKF: упд {mk.n_upd}, χ² {mk.n_rej_chi2}, ‖f‖ {mk.n_rej_gate}, "
              f"восстановлений {mk.n_recover}, дыр {mk.n_hole_infl}")
        # пики по баро (реальные подъёмы/спуски; знак — из отчёта встройки MEKF)
        if "16-12-04" in name:
            wins = [(25, 33, +1), (84, 90, -1), (95, 100, -1)]
        elif "15-34-53" in name:
            wins = [(80, 92, -1)]
        else:
            wins = [(118, 125, +1), (128, 140, -1), (183, 190, +1), (310, 335, -1)]
        print("  пики (окно, v_баро пик @t, v_mekf пик @t, лаг):")
        for (a, b, sgn) in wins:
            (aa, bb, vbp, tb, vfp, tf, lag) = peaks_report(
                rm["t"], rm["vf"], vb, [(a, b, sgn)])[0]
            i2 = peaks_report(rs["t"], rs["vf"], vb, [(a, b, sgn)])[0]
            print(f"    {a:>3.0f}–{b:<3.0f} баро {vbp:+.2f}@{tb:6.1f} | "
                  f"mekf {vfp:+.2f}@{tf:6.1f} (лаг {lag:+.2f} с) | "
                  f"scalar {i2[4]:+.2f}@{i2[5]:6.1f} (Δt мод {tf - i2[5]:+.3f} с)")

    # --- дыры разных длин на 00-38-35 (078 c BT-дроп, 3 c GET, 20 c) ---
    print("\n===== дыры (00-38-35, децимация ×4 ≈ поток 104 Гц) =====")
    t, A, G, H = read_session(os.path.join(ROOT, "data", FILES[0]))
    t, A, G, H = t[::4], A[::4], G[::4], H[::4]
    for (g0, g1) in ((60.0, 60.8), (62.0, 65.0), (75.0, 95.0), (185.0, 200.0)):
        keep = ~((t >= g0) & (t <= g1))
        r = run_arrays(t[keep], A[keep], G[keep], H[keep], "mekf", cfg)
        vb = baro_speed(r["t"], r["h"])
        rest = (r["rest"] == 1) & (np.abs(np.nan_to_num(vb, nan=9)) < 0.10)
        bad = rest & (np.abs(r["vf"]) > 1.0)
        eps = episodes(r["t"], bad)
        after = rest & (r["t"] > g1)
        mk = r["pipeline"].mekf
        print(f"  дыра {g0}–{g1} с: эпизоды {len(eps)}; a_vert в покое после "
              f"{r['a'][after].mean():+.3f}±{r['a'][after].std():.3f}; "
              f"восст. {mk.n_recover}, дыр {mk.n_hole_infl}, χ² {mk.n_rej_chi2}")


if __name__ == "__main__":
    main()
