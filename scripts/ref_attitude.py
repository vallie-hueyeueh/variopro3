# -*- coding: utf-8 -*-
"""Эталонная ориентация для сверки MEKF (пакет 13, блок А).
Mahony без хитростей: интеграция кватерниона по гироскопу + коррекция Kp=2
по направлению гравитации, ТОЛЬКО когда | |f|-g | < 0.8 м/с². Выход a_vert.

python scripts\ref_attitude.py data\session_*.csv [--decimate 4] [--save out.npz]
"""
import argparse
import json
import math
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_calibrated(path):
    """CSV → t, аксель (калиброванный), гиро (без bias), g_ref."""
    arr = np.genfromtxt(path, delimiter=",", names=True)
    t = np.asarray(arr["t"], float)
    F = np.column_stack([arr["ax"], arr["ay"], arr["az"]]).astype(float)
    W = np.column_stack([arr["gx"], arr["gy"], arr["gz"]]).astype(float)
    g_ref = 9.81
    try:
        with open(os.path.join(ROOT, "pc", "calibration.json"), encoding="utf-8") as fh:
            cal = json.load(fh)
        acc = cal.get("accel") or {}
        if "offset" in acc and "scales" in acc:
            F = (F - np.asarray(acc["offset"], float)) * np.asarray(acc["scales"], float)
        if acc.get("target_g"):
            g_ref = float(acc["target_g"])
        if cal.get("gyro_bias") is not None:
            W = W - np.asarray(cal["gyro_bias"], float)
    except (OSError, ValueError):
        pass
    m = np.isfinite(t) & np.isfinite(F).all(axis=1) & np.isfinite(W).all(axis=1)
    return t[m], F[m], W[m], g_ref


def run_reference(t, F, W, g, kp=2.0, gate=0.8):
    """Mahony: q_wb (Гамильтон, скаляр первым), û = третья строка R_wb.
    Возвращает (a_vert (N,), û (N,3)). Инициализация — среднее первых 0.5 с."""
    n = len(t)
    f0 = F[t <= t[0] + 0.5].mean(axis=0)
    u = f0 / np.linalg.norm(f0)                     # вертикаль в осях тела
    # кратчайший поворот u → e_z (рыскание 0)
    c = np.cross(u, [0.0, 0.0, 1.0])
    d = float(u[2])
    q = np.array([1.0 + d, c[0], c[1], c[2]])
    if np.linalg.norm(q) < 1e-9:                    # u ≈ −e_z
        q = np.array([0.0, 1.0, 0.0, 0.0])
    q /= np.linalg.norm(q)
    av = np.zeros(n)
    ub = np.zeros((n, 3))
    prev_t = t[0]
    dt_med = float(np.median(np.diff(t)))
    for i in range(n):
        dt = t[i] - prev_t
        prev_t = t[i]
        if not (0.0 < dt < 0.5):
            dt = dt_med
        w0, x0, y0, z0 = q
        # û = R_bw·e_z (третья строка R_wb)
        u0 = 2.0 * (x0 * z0 - w0 * y0)
        u1 = 2.0 * (y0 * z0 + w0 * x0)
        u2 = 1.0 - 2.0 * (x0 * x0 + y0 * y0)
        fx, fy, fz = F[i]
        fn = math.sqrt(fx * fx + fy * fy + fz * fz)
        wx, wy, wz = W[i]
        if abs(fn - g) < gate and fn > 1e-6:
            mx, my, mz = fx / fn, fy / fn, fz / fn  # измеренная вертикаль
            ex = my * u2 - mz * u1                   # e = u_изм × û
            ey = mz * u0 - mx * u2
            ez = mx * u1 - my * u0
            wx += kp * ex
            wy += kp * ey
            wz += kp * ez
        # q ← q ⊗ δq(ω·dt)
        vx, vy, vz = wx * dt, wy * dt, wz * dt
        a = math.sqrt(vx * vx + vy * vy + vz * vz)
        if a > 1e-12:
            s = math.sin(a / 2.0) / a
            dw, dx, dy, dz = math.cos(a / 2.0), vx * s, vy * s, vz * s
            q = np.array([w0 * dw - x0 * dx - y0 * dy - z0 * dz,
                          w0 * dx + x0 * dw + y0 * dz - z0 * dy,
                          w0 * dy - x0 * dz + y0 * dw + z0 * dx,
                          w0 * dz + x0 * dy - y0 * dx + z0 * dw])
            q /= np.linalg.norm(q)
        av[i] = u0 * fx + u1 * fy + u2 * fz - g
        ub[i] = (u0, u1, u2)
    return av, ub


def rest_mask(t, F, W, g, w_thr=0.10, a_thr=0.20, win=1.0):
    """Спокойные окна 1 с: |ω| и ||f|−g| малы во всём окне."""
    wmag = np.linalg.norm(W, axis=1)
    adev = np.abs(np.linalg.norm(F, axis=1) - g)
    quiet = np.zeros(len(t), bool)
    for s in np.arange(t[0], t[-1], win):
        m = (t >= s) & (t < s + win)
        if m.any() and wmag[m].max() < w_thr and adev[m].max() < a_thr:
            quiet |= m
    return quiet


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--decimate", type=int, default=0, help="брать каждый N-й сэмпл")
    ap.add_argument("--save", help="сохранить a_vert/û в .npz")
    a = ap.parse_args()
    t, F, W, g = load_calibrated(a.file)
    if a.decimate > 1:
        t, F, W = t[::a.decimate], F[::a.decimate], W[::a.decimate]
    av, ub = run_reference(t, F, W, g)
    q = rest_mask(t, F, W, g) & (t > t[0] + 3.0)
    print(f"{os.path.basename(a.file)}: {len(t)} сэмплов, fs≈{1/np.median(np.diff(t)):.0f} Гц")
    print(f"  a_vert в покое ({q.sum()/ (1/np.median(np.diff(t))):.0f} с): "
          f"среднее {av[q].mean():+.4f} ± {av[q].std():.4f} м/с² (эталон ~0.00±0.08)")
    if a.save:
        np.savez(a.save, t=t, a_vert=av, up_body=ub)
        print(f"  сохранено: {a.save}")
