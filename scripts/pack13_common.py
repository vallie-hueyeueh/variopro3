# -*- coding: utf-8 -*-
"""Общее для диагностики пакета 13: пошаговая реплика vario_app._process_sample
(без GUI), чтение session-CSV, баро-скорость, поиск эпизодов."""
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pc"))

from baro_inertial_vario import BaroInertialVario      # noqa: E402
from mekf import make_mekf, load_mekf_config           # noqa: E402

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
    t = t - t[0]
    A = np.column_stack([arr["ax"], arr["ay"], arr["az"]]).astype(float)
    G = np.column_stack([arr["gx"], arr["gy"], arr["gz"]]).astype(float)
    H = np.asarray(arr["altitude"], float)
    m = np.isfinite(t) & np.isfinite(A).all(1) & np.isfinite(G).all(1) & np.isfinite(H)
    return t[m], A[m], G[m], H[m]


class PipelineReplica:
    """Точная реплика _process_sample: детектор → MEKF/скаляр → ноль → посев →
    фильтр. step(t, accel3_raw, gyro3_raw, h_baro) → dict или None (прогрев)."""

    def __init__(self, mode: str, cfg: dict, mekf_factory=None):
        self.mode = mode
        self.mo = cfg.get("motion", {})
        man = cfg.get("manual", {})
        manual = cfg.get("mode") == "manual"
        R = float(man.get("R", 0.0225)) if manual else 0.0225
        self.sigma_accel = float(man.get("sigma_accel", 0.30)) if manual else 0.30
        self.zero_N = float(man.get("calib_time", 5.0)) if manual else 5.0
        self.acc_off, self.acc_scl, self.g_ref, self.gyro_bias = load_device_calib()
        self.flt = BaroInertialVario(dt=DT, sigma_accel=self.sigma_accel,
                                     sigma_baro=math.sqrt(R))
        if mode != "mekf":
            self.mekf = None
        elif mekf_factory is not None:
            self.mekf = mekf_factory(self.g_ref)
        else:
            self.mekf = make_mekf(load_mekf_config(), self.g_ref)
        self.last_t = None
        self.motion_state, self.rest_timer, self.trust = "rest", 0.0, 1.0
        self.zero_done, self.zero_accum, self.zero_count = False, 0.0, 0
        self.zero_elapsed, self.zero_t = 0.0, None
        self.t_first, self.warming = None, False
        # watchdog (как в vario_app, блок А.4)
        self.wd_win = []
        self.wd_bad = 0.0
        self.wd_events = []

    def step(self, ti, a3, g3, h_baro):
        dt = None
        if self.last_t is not None:
            d = ti - self.last_t
            if 0.0 < d < 0.5:
                dt = d
        self.last_t = ti
        dt_eff = dt if dt is not None else DT
        gx, gy, gz = float(g3[0]), float(g3[1]), float(g3[2])
        gyro_mag = math.sqrt(gx * gx + gy * gy + gz * gz)
        ac = (np.asarray(a3, float) - self.acc_off) * self.acc_scl
        a_cal_norm = float(np.linalg.norm(ac))
        acc_dev = abs(a_cal_norm - self.g_ref)
        mo = self.mo
        if gyro_mag > mo["gyro_dyn"] or acc_dev > mo["acc_dyn"]:
            self.motion_state, self.rest_timer = "dyn", 0.0
        elif gyro_mag < mo["gyro_rest"] and acc_dev < mo["acc_rest"]:
            self.rest_timer += dt_eff
            if self.rest_timer >= mo["hold_sec"]:
                self.motion_state = "rest"
        else:
            self.rest_timer = 0.0
        target = 1.0 if self.motion_state == "rest" else mo["k_dyn"]
        self.trust += (target - self.trust) * min(1.0, dt_eff / max(mo["trans_sec"], 1e-3))
        self.flt.accel_trust = self.trust
        self.flt.bias_frozen = (self.motion_state != "rest")
        a = a_cal_norm - self.g_ref
        if self.mode == "mekf":
            gb = self.gyro_bias
            av = self.mekf.step(ti, ac, (gx - gb[0], gy - gb[1], gz - gb[2]),
                                in_motion=(self.motion_state != "rest"))
            if av is not None:
                a = av
        if not self.zero_done and self.zero_N > 0:
            if self.motion_state == "rest":
                self.zero_accum += a
                self.zero_count += 1
                self.zero_elapsed += dt_eff
            if self.zero_elapsed >= self.zero_N and self.zero_count > 0:
                self.flt.x[2, 0] = self.zero_accum / self.zero_count
                self.flt.P[2, 2] = min(self.flt.P[2, 2], 0.05)
                self.zero_done, self.zero_t = True, ti
        if self.t_first is None:
            self.t_first, self.warming = ti, True
            self.flt.x[0, 0] = float(h_baro)
            self.flt.x[1, 0] = 0.0
            self.flt.P[0, 0] = 0.25
            self.flt.P[1, 1] = 0.25
        h_filt, v_filt = self.flt.step(a, float(h_baro), dt)
        # watchdog — зеркально vario_app._process_sample
        self.wd_win.append((ti, float(h_baro)))
        while self.wd_win and ti - self.wd_win[0][0] > 1.0:
            self.wd_win.pop(0)
        if len(self.wd_win) >= 5 and ti - self.wd_win[0][0] >= 0.8:
            v_baro_wd = (float(h_baro) - self.wd_win[0][1]) / (ti - self.wd_win[0][0])
            if abs(v_filt - v_baro_wd) > 1.5:
                self.wd_bad += dt_eff
            else:
                self.wd_bad = 0.0
            if self.wd_bad > 3.0:
                diff = v_filt - v_baro_wd
                self.flt.x[0, 0] = float(h_baro)
                self.flt.x[1, 0] = float(v_baro_wd)
                self.flt.P[0, 0] = 0.25
                self.flt.P[1, 1] = 0.25
                self.wd_bad = 0.0
                self.wd_events.append((ti, diff))
                h_filt, v_filt = self.flt.altitude, self.flt.vario
        if self.warming:
            if ti - self.t_first < WARMUP_SEC:
                return None
            self.warming = False
        out = {"t": ti, "h": float(h_baro), "hf": h_filt, "vf": v_filt, "a": a,
               "rest": self.motion_state == "rest", "bias": self.flt.accel_bias}
        if self.mekf is not None and self.mekf.initialized:
            out["tilt"] = math.degrees(math.acos(max(-1.0, min(1.0, self.mekf._u2))))
            out["chi2"] = self.mekf.n_rej_chi2
        return out


def run_arrays(t, A, G, H, mode, cfg):
    """Прогнать массивы через реплику; вернуть dict np-серий."""
    p = PipelineReplica(mode, cfg)
    keys = ("t", "h", "hf", "vf", "a", "rest", "bias", "tilt", "chi2")
    out = {k: [] for k in keys}
    for i in range(len(t)):
        r = p.step(t[i], A[i], G[i], H[i])
        if r is None:
            continue
        for k in keys:
            out[k].append(r.get(k, np.nan))
    res = {k: np.asarray(v, float) for k, v in out.items()}
    res["pipeline"] = p
    return res


def baro_speed(t, h, win=1.0):
    """Скорость по баро: наклон регрессии высоты в скользящем окне win с."""
    v = np.full(len(t), np.nan)
    j0 = 0
    for i in range(len(t)):
        while t[i] - t[j0] > win:
            j0 += 1
        if i - j0 >= 5:
            v[i] = np.polyfit(t[j0:i + 1] - t[i], h[j0:i + 1], 1)[0]
    return v


def episodes(t, mask, min_len=0.5):
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
