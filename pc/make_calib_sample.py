# -*- coding: utf-8 -*-
"""
make_calib_sample.py
====================
Создаёт синтетический session-CSV для проверки калибровки: data/sample_calib.csv.

Формат — как у записей телефона (12 колонок):
    t,ax,ay,az,gx,gy,gz,mx,my,mz,pressure,altitude

В нём:
  • магнитометр (mx,my,mz) — идеальный шар |B|≈48 мкТл, искажён ИЗВЕСТНЫМИ
    смещением V и матрицей soft-iron + шум;
  • акселерометр (ax,ay,az) — идеальный шар |a|≈g=9.81, искажён слабее.

Так в окне «Калибровка» видно, как красные (сырые) точки лежат криво,
а зелёные (калиброванные) — на шаре. Истинные смещения печатаются ниже,
чтобы можно было сверить с найденными.

Запуск:  python pc/make_calib_sample.py
"""

import os
import csv
import numpy as np

from calibration import make_synthetic
from mag_ekf import rot_from_omega

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
SAMPLES_DIR = os.path.join(DATA_DIR, "samples")   # демо-файлы живут здесь


def gen_mag_stream(n, dt, radius, V, W, noise, seed):
    """
    Согласованный поток магнитометра: телефон РЕАЛЬНО вращается по гироскопу
    (b = R·B_внеш), поэтому годится и для эллипсоида (точки покрывают сферу), и
    для EKF (гиро соответствует вращению). Возвращает (t, mag Nx3, gyro Nx3).
    """
    rng = np.random.default_rng(seed)
    d = np.array([0.45, -0.20, 0.87])
    Bext = radius * d / np.linalg.norm(d)          # поле Земли длиной radius
    t = np.arange(n) * dt
    gyro = np.column_stack([1.2 * np.sin(0.70 * t + 0.1),
                            1.0 * np.sin(0.50 * t + 1.0),
                            0.8 * np.sin(0.31 * t + 2.0)])   # «кувыркание» — покрывает сферу
    mag = np.zeros((n, 3))
    R = np.eye(3)
    Wm = np.asarray(W, float)
    Vv = np.asarray(V, float)
    for k in range(n):
        if k > 0:
            R = rot_from_omega(gyro[k], dt) @ R
        b = R @ Bext
        mag[k] = Wm @ b + Vv + rng.normal(0.0, noise, 3)
    return t, mag, gyro

# Известные искажения (их потом должна восстановить калибровка)
W_MAG = np.array([[1.15, 0.06, -0.04],
                  [0.06, 0.82, 0.05],
                  [-0.04, 0.05, 1.22]])
V_MAG = (14.0, -9.0, 7.0)
R_MAG = 48.0

W_ACC = np.array([[1.02, 0.01, 0.00],
                  [0.01, 0.98, 0.015],
                  [0.00, 0.015, 1.04]])
V_ACC = (0.20, -0.15, 0.30)
R_ACC = 9.81


def main():
    n = 1500   # длиннее → телефон лучше покрывает сферу → EKF сходится к RANSAC в ~1%
    # магнитометр: вращение по гироскопу (согласовано) — годится и эллипсоиду, и EKF
    _t, mag, gyro = gen_mag_stream(n=n, dt=0.02, radius=R_MAG, V=V_MAG, W=W_MAG, noise=0.4, seed=1)
    acc, _, _, _ = make_synthetic(n=n, radius=R_ACC, V_true=V_ACC, W=W_ACC, noise=0.03, seed=2)

    os.makedirs(SAMPLES_DIR, exist_ok=True)
    path = os.path.join(SAMPLES_DIR, "sample_calib.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["t", "ax", "ay", "az", "gx", "gy", "gz",
                    "mx", "my", "mz", "pressure", "altitude"])
        for i in range(n):
            t = i * 0.02
            w.writerow([
                f"{t:.3f}",
                f"{acc[i,0]:.5f}", f"{acc[i,1]:.5f}", f"{acc[i,2]:.5f}",
                f"{gyro[i,0]:.5f}", f"{gyro[i,1]:.5f}", f"{gyro[i,2]:.5f}",
                f"{mag[i,0]:.4f}", f"{mag[i,1]:.4f}", f"{mag[i,2]:.4f}",
                "1013.25", "0.000",
            ])

    print(f"Создан {path}  ({n} строк)")

    # --- демо-файл В ФОРМАТЕ ТЕЛЕФОНА: data/sample_calib.json ---
    import json
    rng2 = np.random.default_rng(11)
    da = rng2.normal(size=(12, 3)); da /= np.linalg.norm(da, axis=1, keepdims=True)
    accel_pts = (R_ACC * da) @ W_ACC.T + np.array(V_ACC) + rng2.normal(0, 0.02, (12, 3))
    # поток магнитометра для JSON — тоже согласован с гироскопом (для EKF)
    nm = 1500
    tm, mstream, gstream = gen_mag_stream(n=nm, dt=0.02, radius=R_MAG, V=V_MAG, W=W_MAG, noise=0.4, seed=5)
    obj = {
        "format": "variopro_calib", "version": 1,
        "created": "2026-06-29 16:00:00", "device": "Samsung S23 (демо)",
        "accel_g": R_ACC,
        "gyro_bias": [0.012, -0.020, 0.006],
        "gps": {"lat": 59.9386, "lon": 30.3141, "alt": 10.0},  # пример (центр СПб)
        "accel_points": [[round(float(c), 5) for c in p] for p in accel_pts],
        "mag_stream_columns": ["t", "mx", "my", "mz", "gx", "gy", "gz"],
        "mag_stream": [[round(float(tm[i]), 4),
                        round(float(mstream[i, 0]), 4), round(float(mstream[i, 1]), 4),
                        round(float(mstream[i, 2]), 4),
                        round(float(gstream[i, 0]), 6), round(float(gstream[i, 1]), 6),
                        round(float(gstream[i, 2]), 6)] for i in range(nm)],
    }
    jpath = os.path.join(SAMPLES_DIR, "sample_calib.json")
    with open(jpath, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1)
    print(f"Создан {jpath}  (формат телефона: 12 поз акс., {nm} точек маг., gyro_bias)")

    print("Истинные искажения, заложенные в файлы (для сверки с калибровкой):")
    print(f"  Магнитометр: V = {V_MAG}, радиус |B| = {R_MAG} мкТл")
    print(f"  Акселерометр: V = {V_ACC}, радиус g = {R_ACC} м/с²")


if __name__ == "__main__":
    main()
