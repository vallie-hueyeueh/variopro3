# -*- coding: utf-8 -*-
"""Блок А: контролируемая репликация живого тракта — ДЫРА в данных во время
вращения (Bluetooth-потери / пауза при GET). Телефон вращался, строк нет →
гироскоп «не видел» поворот → ориентация MEKF врёт после дыры. Смотрим,
восстанавливается ли MEKF и что делает вариометр.

python scripts\pack13_a_gap.py data\session_2026-07-04_00-38-35.csv --gap 75 95
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pack13_common import (load_cfg, read_session, run_arrays, baro_speed,  # noqa: E402
                           episodes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--gap", nargs=2, type=float, default=(75.0, 95.0),
                    help="вырезать строки t∈[a,b] (дыра как при потере связи)")
    ap.add_argument("--decimate", type=int, default=4, help="имитация потока ~104 Гц")
    ap.add_argument("--plot", default="")
    a = ap.parse_args()
    cfg = load_cfg()
    t, A, G, H = read_session(a.file)
    if a.decimate > 1:
        t, A, G, H = t[::a.decimate], A[::a.decimate], G[::a.decimate], H[::a.decimate]
    g0, g1 = a.gap
    keep = ~((t >= g0) & (t <= g1))
    print(f"дыра {g0}–{g1} с: вырезано {int((~keep).sum())} строк из {len(t)}")
    t2, A2, G2, H2 = t[keep], A[keep], G[keep], H[keep]

    for mode in ("mekf", "scalar"):
        r = run_arrays(t2, A2, G2, H2, mode, cfg)
        vb = baro_speed(r["t"], r["h"])
        rest = (r["rest"] == 1) & (np.abs(np.nan_to_num(vb, nan=9)) < 0.10)
        after = r["t"] > g1
        bad = rest & (np.abs(r["vf"]) > 1.0)
        eps = episodes(r["t"], bad)
        print(f"\n== {mode} ==")
        print(f"  эпизоды |vario|>1 в Покое: {eps}")
        m = rest & after
        if m.any():
            print(f"  vario в Покое ПОСЛЕ дыры: mean {r['vf'][m].mean():+.3f}, "
                  f"min {r['vf'][m].min():+.3f}, max {r['vf'][m].max():+.3f}")
            print(f"  a_vert в Покое ПОСЛЕ дыры: mean {r['a'][m].mean():+.3f} "
                  f"± {r['a'][m].std():.3f} м/с²")
        if mode == "mekf":
            mk = r["pipeline"].mekf
            print(f"  MEKF: наклон û в конце {r['tilt'][-1]:.1f}°; счётчики: "
                  f"принято {mk.n_upd}, χ²-отказов {mk.n_rej_chi2}, "
                  f"‖f‖-пропусков {mk.n_rej_gate}")
            # восстановление: первый момент после дыры, когда |vario| ≤ 0.05 устойчиво 1 с
            tt, vf = r["t"], r["vf"]
            rec = None
            for i in range(len(tt)):
                if tt[i] <= g1:
                    continue
                w = (tt >= tt[i]) & (tt <= tt[i] + 1.0)
                if w.any() and np.abs(vf[w]).max() <= 0.05:
                    rec = tt[i]
                    break
            print(f"  возврат |vario|≤0.05 (устойчиво 1 с) после дыры: "
                  f"{'нет до конца файла' if rec is None else f'{rec - g1:.1f} с'}")
            # хвост χ²-отказов после дыры: сколько отказов подряд
            ch = r["chi2"]
            i1 = np.searchsorted(tt, g1)
            dch = np.diff(ch[i1:])
            print(f"  χ²-отказы после дыры: {int(ch[-1] - ch[i1])} шт за "
                  f"{tt[-1] - g1:.0f} с (доля сэмплов {np.mean(dch > 0) * 100:.0f}%)")
        if a.plot and mode == "mekf":
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
            ax[0].plot(r["t"], r["vf"], "C3", lw=0.8, label="vario mekf")
            ax[0].plot(r["t"], vb, "k", lw=0.6, alpha=0.5, label="v баро")
            ax[0].axvspan(g0, g1, color="0.85", label="дыра")
            ax[0].legend(); ax[0].set_ylabel("м/с"); ax[0].set_ylim(-6, 3)
            ax[1].plot(r["t"], r["a"], "C3", lw=0.5, label="a_vert mekf")
            ax[1].axvspan(g0, g1, color="0.85")
            ax[1].legend(); ax[1].set_ylabel("м/с²"); ax[1].set_ylim(-8, 8)
            ax[2].plot(r["t"], r["tilt"], "C2", lw=0.8, label="наклон û, °")
            axb = ax[2].twinx()
            axb.plot(r["t"], r["chi2"], "C1", lw=0.8, label="χ²-отказы (накопл.)")
            ax[2].axvspan(g0, g1, color="0.85")
            ax[2].legend(loc="upper left"); axb.legend(loc="lower right")
            ax[2].set_xlabel("t, с")
            fig.tight_layout()
            fig.savefig(a.plot, dpi=110)
            print(f"  график: {a.plot}")


if __name__ == "__main__":
    main()
