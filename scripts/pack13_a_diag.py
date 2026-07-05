# -*- coding: utf-8 -*-
"""Блок А, диагностика: точная реплика пайплайна vario_app._process_sample
(без GUI) + эталон Mahony. Прогоняет session-CSV в режимах mekf/scalar,
находит эпизоды |vario|>1 в Покое, печатает внутренние состояния.

python scripts\pack13_a_diag.py data\session_2026-07-04_00-38-35.csv
"""
import argparse
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pc"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baro_inertial_vario import BaroInertialVario          # noqa: E402
from mekf import make_mekf, load_mekf_config               # noqa: E402
from ref_attitude import load_calibrated, run_reference, rest_mask  # noqa: E402

DT = 0.02
WARMUP_SEC = 1.5


def load_cfg():
    with open(os.path.join(ROOT, "config.json"), encoding="utf-8") as fh:
        return json.load(fh)


def load_device_calib():
    d = json.load(open(os.path.join(ROOT, "pc", "calibration.json"), encoding="utf-8"))
    acc = d.get("accel") or {}
    return (np.asarray(acc.get("offset", [0, 0, 0]), float),
            np.asarray(acc.get("scales", [1, 1, 1]), float),
            float(acc.get("target_g", 9.81)),
            np.asarray(d.get("gyro_bias") or [0, 0, 0], float))


def read_session(path):
    arr = np.genfromtxt(path, delimiter=",", names=True)
    t = np.asarray(arr["t"], float)
    t = t - t[0]                                   # CsvSource сдвигает t к нулю
    A = np.column_stack([arr["ax"], arr["ay"], arr["az"]]).astype(float)
    G = np.column_stack([arr["gx"], arr["gy"], arr["gz"]]).astype(float)
    H = np.asarray(arr["altitude"], float)
    m = np.isfinite(t) & np.isfinite(A).all(1) & np.isfinite(G).all(1) & np.isfinite(H)
    return t[m], A[m], G[m], H[m]


def run_pipeline(t, A_raw, G_raw, H, mode, cfg, decimate=1, collect_mekf=False):
    """Точная реплика _process_sample. Возвращает dict серий (после прогрева)."""
    if decimate > 1:
        t, A_raw, G_raw, H = t[::decimate], A_raw[::decimate], G_raw[::decimate], H[::decimate]
    mo = cfg.get("motion", {})
    man = cfg.get("manual", {})
    R = float(man.get("R", 0.0225)) if cfg.get("mode") == "manual" else 0.0225
    sigma_accel = float(man.get("sigma_accel", 0.30)) if cfg.get("mode") == "manual" else 0.30
    zero_N = float(man.get("calib_time", 5.0)) if cfg.get("mode") == "manual" else 5.0
    acc_off, acc_scl, g_ref, gyro_bias = load_device_calib()
    flt = BaroInertialVario(dt=DT, sigma_accel=sigma_accel, sigma_baro=math.sqrt(R))
    mekf = make_mekf(load_mekf_config(), g_ref) if mode == "mekf" else None

    # состояние реплики (как поля VarioApp)
    last_t = None
    motion_state, rest_timer, trust = "rest", 0.0, 1.0
    zero_done, zero_accum, zero_count, zero_elapsed = False, 0.0, 0, 0.0
    zero_t = None
    t_first, warming = None, False
    out = {k: [] for k in ("t", "h", "hf", "vf", "a", "motion", "bias", "frozen",
                           "tilt", "p_th", "nrej_chi2", "nrej_gate", "nupd", "avm")}
    for i in range(len(t)):
        ti = t[i]
        h_baro = H[i]
        dt = None
        if last_t is not None:
            d = ti - last_t
            if 0.0 < d < 0.5:
                dt = d
        last_t = ti
        dt_eff = dt if dt is not None else DT
        gx, gy, gz = G_raw[i]
        gyro_mag = math.sqrt(gx * gx + gy * gy + gz * gz)
        ac = (A_raw[i] - acc_off) * acc_scl
        a_cal_norm = float(np.linalg.norm(ac))
        acc_dev = abs(a_cal_norm - g_ref)
        # детектор (гистерезис, как _update_motion)
        if gyro_mag > mo["gyro_dyn"] or acc_dev > mo["acc_dyn"]:
            motion_state, rest_timer = "dyn", 0.0
        elif gyro_mag < mo["gyro_rest"] and acc_dev < mo["acc_rest"]:
            rest_timer += dt_eff
            if rest_timer >= mo["hold_sec"]:
                motion_state = "rest"
        else:
            rest_timer = 0.0
        target = 1.0 if motion_state == "rest" else mo["k_dyn"]
        trust += (target - trust) * min(1.0, dt_eff / max(mo["trans_sec"], 1e-3))
        flt.accel_trust = trust
        flt.bias_frozen = (motion_state != "rest")
        # вертикальное ускорение
        a = a_cal_norm - g_ref
        av = None
        if mode == "mekf":
            w_cal = (gx - gyro_bias[0], gy - gyro_bias[1], gz - gyro_bias[2])
            av = mekf.step(ti, ac, w_cal, in_motion=(motion_state != "rest"))
            if av is not None:
                a = av
        # установка нуля (только в покое)
        if not zero_done and zero_N > 0:
            if motion_state == "rest":
                zero_accum += a
                zero_count += 1
                zero_elapsed += dt_eff
            if zero_elapsed >= zero_N and zero_count > 0:
                flt.x[2, 0] = zero_accum / zero_count
                flt.P[2, 2] = min(flt.P[2, 2], 0.05)
                zero_done, zero_t = True, ti
        # посев первым баро
        if t_first is None:
            t_first, warming = ti, True
            flt.x[0, 0] = float(h_baro)
            flt.x[1, 0] = 0.0
            flt.P[0, 0] = 0.25
            flt.P[1, 1] = 0.25
        h_filt, v_filt = flt.step(a, h_baro, dt)
        if warming:
            if ti - t_first < WARMUP_SEC:
                continue
            warming = False
        out["t"].append(ti); out["h"].append(h_baro)
        out["hf"].append(h_filt); out["vf"].append(v_filt); out["a"].append(a)
        out["motion"].append(1 if motion_state == "rest" else 0)
        out["bias"].append(flt.accel_bias); out["frozen"].append(flt.bias_frozen)
        if collect_mekf and mekf is not None and mekf.initialized:
            out["tilt"].append(math.degrees(math.acos(max(-1.0, min(1.0, mekf._u2)))))
            out["p_th"].append(math.sqrt(max(mekf.P[0, 0], mekf.P[1, 1])))
            out["nrej_chi2"].append(mekf.n_rej_chi2)
            out["nrej_gate"].append(mekf.n_rej_gate)
            out["nupd"].append(mekf.n_upd)
            out["avm"].append(a)
    res = {k: np.asarray(v) for k, v in out.items() if v}
    res["zero_t"] = zero_t
    res["mekf"] = mekf
    return res


def baro_speed(t, h, win=1.0):
    """Скорость по баро: наклон регрессии высоты в окне win (центрированное)."""
    v = np.full(len(t), np.nan)
    j0 = 0
    for i in range(len(t)):
        while t[i] - t[j0] > win:
            j0 += 1
        if i - j0 >= 5:
            tt = t[j0:i + 1] - t[i]
            hh = h[j0:i + 1]
            v[i] = np.polyfit(tt, hh, 1)[0]
    return v


def episodes(t, mask, min_len=0.5):
    """Непрерывные куски True длиннее min_len → список (t0, t1)."""
    out, start = [], None
    for i in range(len(t)):
        if mask[i] and start is None:
            start = t[i]
        elif not mask[i] and start is not None:
            if t[i - 1] - start >= min_len:
                out.append((start, t[i - 1]))
            start = None
    if start is not None and t[-1] - start >= min_len:
        out.append((start, t[-1]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--plot", default="")
    a = ap.parse_args()
    cfg = load_cfg()
    print(f"config: mode={cfg.get('mode')}, manual={cfg.get('manual')}")
    t, A, G, H = read_session(a.file)
    fs = 1.0 / np.median(np.diff(t))
    print(f"{os.path.basename(a.file)}: {t[-1]:.1f} с, {len(t)} сэмплов, fs≈{fs:.0f} Гц, "
          f"max dt={np.max(np.diff(t))*1e3:.1f} мс, NaN баро: {np.isnan(H).sum()}")

    r_m = run_pipeline(t, A, G, H, "mekf", cfg, collect_mekf=True)
    r_s = run_pipeline(t, A, G, H, "scalar", cfg)
    vb = baro_speed(r_m["t"], r_m["h"])

    # покой пайплайна И баро-скорость ~0
    rest = (r_m["motion"] == 1) & (np.abs(np.nan_to_num(vb, nan=9)) < 0.10)
    bad = rest & (np.abs(r_m["vf"]) > 1.0)
    print(f"\nПокой (детектор + |v_баро|<0.1): {rest.mean()*100:.0f}% времени")
    print(f"эпизоды |vario|>1 м/с В ПОКОЕ (mekf): {episodes(r_m['t'], bad)}")
    print(f"  vario mekf  в покое: mean {r_m['vf'][rest].mean():+.3f}, "
          f"min {r_m['vf'][rest].min():+.3f}, max {r_m['vf'][rest].max():+.3f}")
    rest_s = (r_s["motion"] == 1) & (np.abs(np.nan_to_num(vb, nan=9)) < 0.10)
    print(f"  vario scalar в покое: mean {r_s['vf'][rest_s].mean():+.3f}, "
          f"min {r_s['vf'][rest_s].min():+.3f}, max {r_s['vf'][rest_s].max():+.3f}")
    print(f"  a_vert mekf в покое: mean {r_m['a'][rest].mean():+.3f} ± {r_m['a'][rest].std():.3f}"
          f" | scalar: {r_s['a'][rest_s].mean():+.3f} ± {r_s['a'][rest_s].std():.3f}")
    print(f"  установка нуля: mekf t={r_m['zero_t']}, scalar t={r_s['zero_t']}")
    mk = r_m["mekf"]
    print(f"  MEKF счётчики: принято {mk.n_upd}, отбраковано χ² {mk.n_rej_chi2}, "
          f"пропущено по ‖f‖ {mk.n_rej_gate}")

    # --- эталон Mahony: посэмплово, 417 Гц и децимация ×4 ---
    tc, Fc, Wc, g = load_calibrated(a.file)
    tc = tc - tc[0]
    for dec, label in ((1, "417 Гц"), (4, "104 Гц (каждый 4-й)")):
        td, Fd, Wd = tc[::dec], Fc[::dec], Wc[::dec]
        av_ref, ub_ref = run_reference(td, Fd, Wd, g)
        q = rest_mask(td, Fd, Wd, g) & (td > 3.0)
        print(f"\nЭТАЛОН Mahony {label}: a_vert в покое {av_ref[q].mean():+.4f} ± "
              f"{av_ref[q].std():.4f} м/с²")
        # наш MEKF standalone на тех же данных
        m2 = make_mekf(load_mekf_config(), g)
        av_our = np.full(len(td), np.nan)
        tilt_our = np.zeros(len(td))
        for i in range(len(td)):
            rr = m2.step(td[i], Fd[i], Wd[i])
            if rr is not None:
                av_our[i] = rr
            tilt_our[i] = math.degrees(math.acos(max(-1.0, min(1.0,
                float(np.dot(m2.up_body, ub_ref[i]))))))
        dq = q & np.isfinite(av_our)
        diff = av_our[dq] - av_ref[dq]
        print(f"  наш MEKF − эталон в покое: mean {diff.mean():+.4f}, max|Δ| "
              f"{np.abs(diff).max():.4f} м/с²; расхождение вертикалей: "
              f"медиана {np.median(tilt_our[dq]):.2f}°, max {tilt_our[dq].max():.2f}°")
        print(f"  наш MEKF a_vert в покое: {np.nanmean(av_our[dq]):+.4f} ± "
              f"{np.nanstd(av_our[dq]):.4f}; счётчики: упд {m2.n_upd}, χ²-отказов "
              f"{m2.n_rej_chi2}, ‖f‖-пропусков {m2.n_rej_gate}")

    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
        ax[0].plot(r_m["t"], r_m["vf"], "C3", lw=0.8, label="vario mekf")
        ax[0].plot(r_s["t"], r_s["vf"], "C0", lw=0.8, alpha=0.7, label="vario scalar")
        ax[0].plot(r_m["t"], vb, "k", lw=0.6, alpha=0.5, label="v баро (окно 1 с)")
        ax[0].set_ylabel("вариометр, м/с"); ax[0].legend(); ax[0].set_ylim(-6, 6)
        ax[1].plot(r_m["t"], r_m["a"], "C3", lw=0.5, label="a_vert mekf")
        ax[1].plot(r_s["t"], r_s["a"], "C0", lw=0.5, alpha=0.6, label="a scalar")
        ax[1].set_ylabel("a_vert, м/с²"); ax[1].legend(); ax[1].set_ylim(-8, 8)
        if len(r_m["tilt"]):
            n = len(r_m["tilt"]); toff = r_m["t"][len(r_m["t"]) - n:]
            ax[2].plot(toff, r_m["tilt"], "C2", lw=0.7, label="наклон MEKF (угол û от e_z), °")
            ax[2].plot(toff, np.array(r_m["p_th"]) * 180 / math.pi, "C4", lw=0.7,
                       label="σθ MEKF, °")
            ax2b = ax[2].twinx()
            ax2b.plot(toff, r_m["nrej_chi2"], "C1", lw=0.7, label="χ²-отказы (накопл.)")
            ax2b.legend(loc="lower right")
            ax[2].legend(loc="upper left"); ax[2].set_ylabel("градусы")
        ax[3].plot(r_m["t"], r_m["motion"], "C2", lw=0.7, label="Покой=1")
        ax[3].plot(r_m["t"], r_m["bias"], "C5", lw=0.7, label="bias фильтра b, м/с²")
        ax[3].legend(); ax[3].set_xlabel("t, с")
        fig.tight_layout()
        fig.savefig(a.plot, dpi=110)
        print(f"\nграфик: {a.plot}")


if __name__ == "__main__":
    main()
