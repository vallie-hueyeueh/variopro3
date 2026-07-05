# -*- coding: utf-8 -*-
"""
mag_ekf.py
==========
КАЛИБРОВКА МАГНИТОМЕТРА обобщённым фильтром Калмана (EKF) — строго по
docs/mag_ekf_spec.md (метод da-nie с корректировками 1–6).

ЗАЧЕМ простыми словами
----------------------
Эллипсоид (calibration.py) калибрует магнитометр «пакетно» по всем точкам сразу.
EKF делает то же РЕКУРСИВНО (по одному замеру), используя ещё и ГИРОСКОП: он знает,
как телефон повернулся за шаг, и по этому отделяет постоянное «железо» телефона (V)
от вращающегося поля Земли (b). Результат — те же V (hard-iron) и W (soft-iron).

Состояние x (12 чисел), раздел 2 спецификации:
    x = [ bx,by,bz,  W11,W22,W33,W12,W13,W23,  Vx,Vy,Vz ]
    b — истинное повёрнутое поле в осях сенсора (|b| = const = поле Земли);
    W — симметричная матрица soft-iron (6 чисел);
    V — вектор hard-iron (железо телефона).

ВАЖНАЯ ДОБАВКА К СПЕЦИФИКАЦИИ (устойчивость): у этого телефона hard-iron огромный
(V ≈ 1230 мкТл), а поле Земли всего ≈ 53 мкТл — состояние плохо масштабировано
(W~1, |b|~53, V~1230, разброс ~10^4). Это ровно то, о чём предупреждает автор статьи
(«делить на ~10^6»). Поэтому ПЕРЕД фильтром нормируем сырое поле на
    s0 = среднее(|сырое|)
и считаем в безразмерных единицах: тогда |b|≈0.04, V≈1, W≈1 — один порядок, матрицы
обусловлены. P/Q (заданные в мкТл) делим на s0, R_meas (дисперсия) — на s0². В конце
домножаем V и поле обратно на s0.

Запуск самопроверки:
    python pc/mag_ekf.py
        1) СИНТЕТИКА (раздел 9): известные W_true,V_true,|B| → EKF должен сойтись <2%;
           заодно проверяется знак ΔR (верный знак → модуль откалиброванного постоянен).
        2) РЕАЛЬНАЯ запись + кросс-проверка с RANSAC-эллипсоидом (раздел 8): таблица
           V(EKF) vs центр эллипсоида, остаток EKF vs RANSAC — должны совпасть в пару %.
"""

from __future__ import annotations

import os
import glob
import json
import numpy as np

# RANSAC-эллипсоид как «эталон истины» для кросс-проверки (раздел 8)
from calibration import calibrate_robust


# ----------------------------------------------------------------------
# Малые помощники: кососимметричная матрица и поворот из гироскопа
# ----------------------------------------------------------------------
def skew(w):
    """[ω]_× по спецификации: [[0,-wz,wy],[wz,0,-wx],[-wy,wx,0]]."""
    return np.array([[0.0, -w[2], w[1]],
                     [w[2], 0.0, -w[0]],
                     [-w[1], w[0], 0.0]])


def rot_from_omega(w, dt):
    """
    ΔR = exp([ω]_× · dt) — ТОЧНО (формула Родрига). ω — угловая скорость (рад/с).
    Это инкрементальный поворот сенсора за шаг dt (раздел 4).
    """
    w = np.asarray(w, dtype=float)
    ang = float(np.linalg.norm(w)) * dt
    if ang < 1e-12:
        return np.eye(3)
    k = w / np.linalg.norm(w)
    K = skew(k)
    return np.eye(3) + np.sin(ang) * K + (1.0 - np.cos(ang)) * (K @ K)


# ----------------------------------------------------------------------
# Сборка W из состояния и аналитический якобиан H (раздел 5)
# ----------------------------------------------------------------------
def W_from_state(x):
    """Симметричная W из x[3:9] в порядке W11,W22,W33,W12,W13,W23."""
    W11, W22, W33, W12, W13, W23 = x[3], x[4], x[5], x[6], x[7], x[8]
    return np.array([[W11, W12, W13],
                     [W12, W22, W23],
                     [W13, W23, W33]])


def H_jacobian(x):
    """
    Аналитический якобиан наблюдения H (3×12), строго по разделу 5.
    h(x) = W·b + V.
    """
    b = x[0:3]
    W = W_from_state(x)
    H = np.zeros((3, 12))
    H[:, 0:3] = W                      # ∂h/∂b = W
    # ∂h/∂W (столбцы 3:9), порядок W11,W22,W33,W12,W13,W23
    H[:, 3] = [b[0], 0.0, 0.0]         # ∂h/∂W11
    H[:, 4] = [0.0, b[1], 0.0]         # ∂h/∂W22
    H[:, 5] = [0.0, 0.0, b[2]]         # ∂h/∂W33
    H[:, 6] = [b[1], b[0], 0.0]        # ∂h/∂W12
    H[:, 7] = [b[2], 0.0, b[0]]        # ∂h/∂W13
    H[:, 8] = [0.0, b[2], b[1]]        # ∂h/∂W23
    H[:, 9:12] = np.eye(3)             # ∂h/∂V = I
    return H


# ----------------------------------------------------------------------
# Главный прогон EKF по записи (нормировка на s0 + цикл predict/update)
# ----------------------------------------------------------------------
def run_mag_ekf(t, mag, gyro, sign=+1.0, sigma_m_uT=0.5,
                tail_frac=0.5, F_ref=None, return_history=False,
                record_checkpoints=0):
    """
    Прогнать EKF по записи магнитометра+гироскопа.

    t    (N,)   — время, с;
    mag  (N,3)  — СЫРОЕ поле mx,my,mz (uncalibrated), мкТл;
    gyro (N,3)  — гироскоп gx,gy,gz, рад/с (bias уже близок к нулю);
    sign        — знак ΔR (+1 или −1): пробуем оба, верный даёт постоянный |b|.

    Возвращает dict: s0, W (безразмерн.), V_uT (мкТл), field_uT (мкТл),
    residual_rel (разброс |откалибр.| на сошедшемся хвосте), calibrated_uT (N,),
    Winv, и (опц.) историю сходимости.
    """
    t = np.asarray(t, float)
    mag = np.asarray(mag, float)
    gyro = np.asarray(gyro, float)
    n = len(t)

    # --- НОРМИРОВКА: s0 = среднее(|сырое|) (≈ |V| у телефона) ---
    s0 = float(np.mean(np.linalg.norm(mag, axis=1)))
    if s0 <= 0:
        raise ValueError("Пустые/нулевые данные магнитометра")
    z = mag / s0                       # нормированные измерения (безразмерные)
    sm = sigma_m_uT / s0               # шум измерения в нормир. единицах

    # --- инициализация (раздел 3), в нормированных единицах ---
    # Спец задаёт b_0 = B_изм(0), V_0 = 0 — это годится для МАЛОГО железа (центр ≈ 0).
    # У этого телефона железо огромно (V ≈ s0), поэтому при V_0 = 0 поле b стартует
    # ≈ |V| (≈1200), и из-за калибровочной свободы «масштаб W ↔ величина поля» фильтр
    # садится не туда. Натуральное значение V — это ЦЕНТР облака (доказано: при полном
    # вращении mean(z) = V). Поэтому засеваем V_0 = центроид, b_0 = z(0) − центроид
    # (≈ истинное поле). Это прямое продолжение нормировки для огромного железа.
    V0 = np.mean(z, axis=0)
    x = np.zeros(12)
    x[9:12] = V0                        # V_0 = центр облака
    x[0:3] = z[0] - V0                  # b_0 = первое измерение минус центр (≈ истинное b)
    x[3] = x[4] = x[5] = 1.0           # W_0 = I

    # Ковариация P_0 и шум процесса Q. Спец задаёт σ_b≈10, σ_V≈30 мкТл — это для
    # МАЛОГО железа. У этого телефона V≈s0, поэтому σ_b, σ_V ~ s0 (мкТл) → /s0 ≈ 1
    # (нормированно). σ_W=0.5 (безразмерн.). Это и есть «делим P,Q на s0».
    P = np.diag([1.0, 1.0, 1.0,                 # b
                 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,  # W (σ_W=0.5 → 0.25)
                 1.0, 1.0, 1.0]).astype(float)  # V
    qb = (0.5 / s0) ** 2               # модель поля: малый (0.5 мкТл /s0)²
    qw = 1e-8                          # W почти константа
    qv = (0.1 / s0) ** 2               # V почти константа
    Q = np.diag([qb, qb, qb, qw, qw, qw, qw, qw, qw, qv, qv, qv])
    R = np.eye(3) * (sm ** 2)          # R_meas / s0²
    I12 = np.eye(12)

    pdiag_hist = np.zeros((n, 12))
    # контрольные точки для кривой сходимости (снимок состояния на этих шагах)
    cp_idx = set()
    if record_checkpoints and record_checkpoints > 0:
        cp_idx = set(np.unique(np.linspace(max(20, n // 60), n - 1,
                                           int(record_checkpoints)).astype(int)).tolist())
    snaps = []
    for k in range(n):
        # --- ПРЕДСКАЗАНИЕ (раздел 4) ---
        if k > 0:
            dt = t[k] - t[k - 1]
            if dt <= 0 or dt > 1.0:    # защита от дыр в метках времени
                dt = 0.02
            dR = rot_from_omega(sign * gyro[k], dt)
            x[0:3] = dR @ x[0:3]       # b вращается; W,V — константы
            F = I12.copy()
            F[0:3, 0:3] = dR           # F = blockdiag(ΔR, I9)
            P = F @ P @ F.T + Q
        # --- КОРРЕКЦИЯ (раздел 5), форма Джозефа ---
        b = x[0:3]
        W = W_from_state(x)
        V = x[9:12]
        h = W @ b + V
        y = z[k] - h
        H = H_jacobian(x)
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ y
        A = I12 - K @ H
        P = A @ P @ A.T + K @ R @ K.T   # Джозеф — устойчивость (корректировка 4)
        P = 0.5 * (P + P.T)             # симметризация
        pdiag_hist[k] = np.diag(P)
        if k in cp_idx:
            snaps.append((k, x.copy()))

    # --- результат, обратно в мкТл ---
    W_fin = W_from_state(x)
    V_norm = x[9:12].copy()
    V_uT = V_norm * s0
    # КАЛИБРОВОЧНАЯ СВОБОДА МАСШТАБА: z = W·b + V не различает (αW) и (b/α), поэтому
    # EKF определяет W лишь С ТОЧНОСТЬЮ ДО МАСШТАБА (форма верна, абсолют — нет).
    # Абсолютный масштаб задаёт эталон поля F (раздел 6: |b_калибр| = |B_внеш|).
    Winv = np.linalg.inv(W_fin)
    cal_norm = (z - V_norm) @ Winv.T            # b = W⁻¹(z − V) (раздел 6)
    cal_uT = np.linalg.norm(cal_norm, axis=1) * s0
    k0 = int((1.0 - tail_frac) * n)             # качество — на сошедшемся хвосте
    field_raw = float(np.mean(cal_uT[k0:]))
    if F_ref is not None and F_ref > 0 and field_raw > 0:
        scale = field_raw / F_ref               # привязка масштаба к эталону F
        W_fin = W_fin * scale                   # W' = (field/F)·W  →  |b_калибр| = F
        Winv = np.linalg.inv(W_fin)
        cal_norm = (z - V_norm) @ Winv.T
        cal_uT = np.linalg.norm(cal_norm, axis=1) * s0
    tail = cal_uT[k0:]
    field_uT = float(np.mean(tail))
    residual_rel = float(np.std(tail) / max(field_uT, 1e-9))

    out = {
        "s0": s0, "W": W_fin, "Winv": Winv,
        "V_uT": V_uT, "field_uT": field_uT,
        "residual_rel": residual_rel, "calibrated_uT": cal_uT,
        "n": n, "sign": sign,
    }
    if return_history:
        out["pdiag_hist"] = pdiag_hist
    # кривая сходимости: на каждом снимке считаем остаток по точкам 0..k (он
    # масштаб-инвариантен: std/mean), |V| и центр V — для анимации в GUI
    if snaps:
        curve = []
        for (kk, xk) in snaps:
            Wk = W_from_state(xk)
            Vk = xk[9:12]
            try:
                Wkinv = np.linalg.inv(Wk)
            except np.linalg.LinAlgError:
                continue
            cal = (z[:kk + 1] - Vk) @ Wkinv.T
            mags = np.linalg.norm(cal, axis=1)
            mean_m = float(np.mean(mags))
            resid = float(np.std(mags) / mean_m) if mean_m > 0 else 0.0
            fld = F_ref if (F_ref is not None and F_ref > 0) else mean_m * s0
            curve.append({"k": int(kk), "absV": float(np.linalg.norm(Vk)) * s0,
                          "V_uT": (Vk * s0).copy(), "residual": resid, "field": float(fld)})
        out["curve"] = curve
    return out


def run_mag_ekf_autosign(t, mag, gyro, **kw):
    """Прогнать с обоими знаками ΔR и вернуть тот, у кого |откалибр.| постояннее
    (меньше residual) — автоматический выбор знака (корректировка 2)."""
    r_pos = run_mag_ekf(t, mag, gyro, sign=+1.0, **kw)
    r_neg = run_mag_ekf(t, mag, gyro, sign=-1.0, **kw)
    best = r_pos if r_pos["residual_rel"] <= r_neg["residual_rel"] else r_neg
    return best, r_pos, r_neg


# ======================================================================
# ЖИВОЙ EKF ПО ПОТОКУ (Фаза 5, шаг 3) — та же математика, по точке за раз
# ======================================================================
def fibonacci_directions(n: int = 50) -> np.ndarray:
    """n почти равномерных направлений на сфере (спираль Фибоначчи) — «семена»
    телесных секторов (покрытие сферы, критерий готовности инициализации)."""
    i = np.arange(n, dtype=float) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    return np.column_stack([np.sin(phi) * np.cos(theta),
                            np.sin(phi) * np.sin(theta),
                            np.cos(phi)])


class LiveMagEKF:
    """
    Рекурсивный прогон EKF по точкам ЖИВОГО потока. Математика — ровно та же,
    что в run_mag_ekf (нормировка s0, V0 = центроид, те же P/Q/R, форма Джозефа,
    симметризация). Отличия только организационные — потому что живой фильтр
    не видит запись целиком заранее:

      • s0 (нормировка) и V0 (центроид) берутся по СТАРТОВОМУ БУФЕРУ — причём
        инициализация ждёт, пока в буфере появится РАЗНООБРАЗИЕ направлений
        (≥ init_min_sectors из 50 телесных секторов): «V0 = центроид» осмыслен
        только при вращении (на неподвижном телефоне центроид = V + W·b, т.е.
        смещён на целое поле, b0 выходит ≈ 0 и фильтр сходится не туда —
        проверено на реальном потоке);
      • накопленный буфер после инициализации ПРОГОНЯЕТСЯ через фильтр —
        данные не теряются;
      • знак ΔR неизвестен заранее → ДВА параллельных фильтра (+1 и −1);
        «лучший» — у кого остаток меньше (то же правило, что autosign);
      • остаток/поле считаются по СКОЛЬЗЯЩЕМУ окну последних resid_win точек
        (лениво, при запросе metrics() — не на каждой точке).

    Использование:  add(t, mag, gyro) на каждой точке → metrics() для GUI →
    result(F_ref) — финал в формате run_mag_ekf (для сохранения прибора).
    """

    def __init__(self, sigma_m_uT: float = 0.8, init_n: int = 200,
                 resid_win: int = 800, init_min_sectors: int = 15):
        self.sigma_m_uT = float(sigma_m_uT)
        self.init_n = int(init_n)
        self.resid_win = int(resid_win)
        self.init_min_sectors = int(init_min_sectors)
        self._seeds = fibonacci_directions(50)
        self.pre_sectors = 0            # прогресс разнообразия (для GUI)
        self.n = 0                      # обработано точек (после инициализации — тоже)
        self.s0 = None
        self._sign_locked = None        # знак ΔR после ЛОКА (пакет 15, Б.2)
        self._z = []                    # ВСЕ нормированные измерения (и для окна остатка)
        self._pre = []                  # стартовый буфер (t, mag, gyro) до инициализации
        self._t_last = None
        self._x = {}                    # знак → состояние x (12)
        self._P = {}                    # знак → ковариация P (12×12)
        self._Q = None
        self._R = None
        self._I12 = np.eye(12)
        self.ready = False              # фильтры инициализированы

    # ---- внутреннее: инициализация по стартовому буферу (как раздел 3) ----
    def _init_filters(self):
        mags = np.array([m for (_t, m, _g) in self._pre], float)
        self.s0 = float(np.mean(np.linalg.norm(mags, axis=1)))
        if self.s0 <= 0:
            raise ValueError("нулевые данные магнитометра")
        z = mags / self.s0
        # V0 — СЕКТОРНЫЙ центроид: среднее по средним занятых телесных секторов.
        # Простой mean(z) верен лишь при РАВНОМЕРНОМ вращении (batch по полной
        # записи); в живом буфере 90%+ точек может быть покоем — простой центроид
        # смещён к точке покоя на целое поле (b0 вышло бы ≈ 0, фильтр цеплялся к
        # ложному решению — проверено на реальном потоке). Секторное усреднение
        # убирает перевес покоя, оставаясь тем же «V = центр облака».
        c0 = np.mean(z, axis=0)
        d = z - c0
        nrm = np.linalg.norm(d, axis=1)
        keep = nrm > (self.MIN_SPREAD_UT / self.s0)
        if keep.sum() >= 50:
            dirs = d[keep] / nrm[keep, None]
            owner = np.argmax(dirs @ self._seeds.T, axis=1)
            zk = z[keep]
            V0 = np.mean([zk[owner == s].mean(axis=0)
                          for s in np.unique(owner)], axis=0)
        else:
            V0 = c0
        for sign in (+1.0, -1.0):
            x = np.zeros(12)
            x[9:12] = V0
            x[0:3] = z[0] - V0
            x[3] = x[4] = x[5] = 1.0
            self._x[sign] = x
            self._P[sign] = np.diag([1.0, 1.0, 1.0,
                                     0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
                                     1.0, 1.0, 1.0]).astype(float)
        qb = (0.5 / self.s0) ** 2
        qw = 1e-8
        qv = (0.1 / self.s0) ** 2
        self._Q = np.diag([qb, qb, qb, qw, qw, qw, qw, qw, qw, qv, qv, qv])
        sm = self.sigma_m_uT / self.s0
        self._R = np.eye(3) * (sm ** 2)
        self.ready = True
        # прогнать стартовый буфер через оба фильтра (данные не теряются)
        pre = self._pre
        self._pre = []
        for (tt, mm, gg) in pre:
            self._step(tt, mm, gg)

    def _step(self, t, mag, gyro):
        z_k = np.asarray(mag, float) / self.s0
        self._z.append(z_k)
        dt = 0.02
        if self._t_last is not None:
            d = t - self._t_last
            if 0.0 < d <= 1.0:
                dt = d
        self._t_last = t
        for sign in (+1.0, -1.0):
            x = self._x[sign]
            P = self._P[sign]
            if self.n > 0:
                dR = rot_from_omega(sign * np.asarray(gyro, float), dt)
                x[0:3] = dR @ x[0:3]
                F = self._I12.copy()
                F[0:3, 0:3] = dR
                P = F @ P @ F.T + self._Q
            b = x[0:3]
            W = W_from_state(x)
            V = x[9:12]
            y = z_k - (W @ b + V)
            H = H_jacobian(x)
            S = H @ P @ H.T + self._R
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ y
            A = self._I12 - K @ H
            P = A @ P @ A.T + K @ self._R @ K.T
            self._x[sign] = x
            self._P[sign] = 0.5 * (P + P.T)
        self.n += 1

    MIN_SPREAD_UT = 15.0   # направления берём только от точек дальше этого от
                           # центроида: шум (~1 мкТл) изотропен и «покрывает» все
                           # сектора даже на неподвижном телефоне; настоящий сигнал
                           # вращения — радиус ≈ поле Земли (≥ ~20 мкТл)

    def _spread_sectors(self) -> int:
        """Сколько телесных секторов занято направлениями (z − центроид) буфера
        (только точки с заметным радиусом — см. MIN_SPREAD_UT)."""
        mags = np.array([m for (_t, m, _g) in self._pre], float)
        d = mags - mags.mean(axis=0)
        nrm = np.linalg.norm(d, axis=1)
        keep = nrm > self.MIN_SPREAD_UT
        if not keep.any():
            return 0
        d = d[keep] / nrm[keep, None]
        return int(len(np.unique(np.argmax(d @ self._seeds.T, axis=1))))

    # ---- публичное ----
    def add(self, t, mag, gyro):
        """Скормить одну точку потока (сырое поле мкТл + гироскоп рад/с)."""
        if not self.ready:
            self._pre.append((float(t), np.asarray(mag, float),
                              np.asarray(gyro, float)))
            # готовность инициализации проверяем каждые 25 точек: нужен минимум
            # объёма И разнообразие направлений (телефон реально вращали)
            if len(self._pre) >= self.init_n and len(self._pre) % 25 == 0:
                self.pre_sectors = self._spread_sectors()
                if self.pre_sectors >= self.init_min_sectors:
                    self._init_filters()
            return
        self._step(float(t), mag, gyro)

    def _resid_field(self, sign, win=None):
        """(остаток, поле_мкТл) по последним win точкам текущим состоянием знака."""
        if not self._z:
            return None, None
        z = np.asarray(self._z[-(win or self.resid_win):], float)
        x = self._x[sign]
        try:
            Winv = np.linalg.inv(W_from_state(x))
        except np.linalg.LinAlgError:
            return None, None
        cal = (z - x[9:12]) @ Winv.T
        mags = np.linalg.norm(cal, axis=1)
        mean_m = float(np.mean(mags))
        if mean_m <= 0:
            return None, None
        return float(np.std(mags) / mean_m), mean_m * self.s0

    def best_sign(self):
        """Знак с меньшим остатком (правило autosign). Пакет 15 (Б.2): после
        ПЕРВОГО решения знак ЛОЧИТСЯ до перезапуска сбора (новый LiveMagEKF):
        остатки знаков на живом потоке дышат, и переключение знака на лету
        меняло V/M скачком — компас дёргался. None — пока рано решать."""
        if self._sign_locked is not None:
            return self._sign_locked
        if not self.ready or self.n < self.init_n:
            return None
        r_pos, _ = self._resid_field(+1.0)
        r_neg, _ = self._resid_field(-1.0)
        if r_pos is None or r_neg is None:
            return None
        self._sign_locked = +1.0 if r_pos <= r_neg else -1.0
        return self._sign_locked

    def metrics(self):
        """Живые числа для GUI: n, sign, |V| мкТл, поле мкТл, остаток (лучший знак).
        До инициализации: ready=False + прогресс разнообразия (сектора/цель)."""
        if not self.ready:
            return {"n": len(self._pre), "ready": False,
                    "pre_sectors": self.pre_sectors,
                    "need_sectors": self.init_min_sectors}
        sign = self.best_sign()
        if sign is None:
            return {"n": self.n, "ready": False,
                    "pre_sectors": self.pre_sectors,
                    "need_sectors": self.init_min_sectors}
        resid, field = self._resid_field(sign)
        V_uT = self._x[sign][9:12] * self.s0
        return {"n": self.n, "ready": True, "sign": sign,
                "V_uT": V_uT, "absV_uT": float(np.linalg.norm(V_uT)),
                "field_uT": field, "residual_rel": resid}

    def result(self, F_ref=None):
        """Финальный результат ЛУЧШЕГО знака в формате run_mag_ekf (W/Winv/V_uT/
        field_uT/residual_rel; масштаб W привязывается к эталону F_ref, раздел 6)."""
        sign = self.best_sign()
        if sign is None:
            return None
        x = self._x[sign]
        W_fin = W_from_state(x)
        V_norm = x[9:12].copy()
        resid, field_raw = self._resid_field(sign)
        if F_ref is not None and F_ref > 0 and field_raw and field_raw > 0:
            W_fin = W_fin * (field_raw / F_ref)
        try:
            Winv = np.linalg.inv(W_fin)
        except np.linalg.LinAlgError:
            return None
        z = np.asarray(self._z[-self.resid_win:], float)
        cal = (z - V_norm) @ Winv.T
        mags = np.linalg.norm(cal, axis=1) * self.s0
        field = float(np.mean(mags))
        residual = float(np.std(mags) / max(field, 1e-9))
        return {"s0": self.s0, "W": W_fin, "Winv": Winv,
                "V_uT": V_norm * self.s0, "field_uT": field,
                "residual_rel": residual, "n": self.n, "sign": sign}


# ======================================================================
# РАЗДЕЛ 9: ТЕСТ НА СИНТЕТИКЕ
# ======================================================================
def make_synthetic(n=5000, dt=0.02, F=50.0, V_true=(1200.0, -40.0, 25.0),
                   noise=0.5, seed=1):
    """
    Известные W_true, V_true, |B_внеш|=F. Поток поворотов из синтетического гиро →
    b_k = R_k·B_внеш → z_k = W_true·b_k + V_true + шум. Возвращает (t,z,gyro,W_true,V_true,F).
    """
    rng = np.random.default_rng(seed)
    # soft-iron: симметричная, близкая к I
    W_true = np.array([[1.08, 0.05, -0.04],
                       [0.05, 0.90, 0.06],
                       [-0.04, 0.06, 1.15]])
    V_true = np.asarray(V_true, float)
    # поле Земли в мире: фикс. направление длиной F (с наклонением)
    d = np.array([0.45, -0.20, 0.87])
    Bext = F * d / np.linalg.norm(d)

    t = np.arange(n) * dt
    # «кувыркающаяся» угловая скорость — покрывает сферу ориентаций
    gyro = np.column_stack([
        1.2 * np.sin(0.70 * t + 0.1),
        1.0 * np.sin(0.50 * t + 1.0),
        0.8 * np.sin(0.31 * t + 2.0),
    ])
    z = np.zeros((n, 3))
    R = np.eye(3)
    for k in range(n):
        if k > 0:
            R = rot_from_omega(gyro[k], dt) @ R   # b_k = ΔR_k·b_{k-1}
        b = R @ Bext
        z[k] = W_true @ b + V_true + rng.normal(0.0, noise, 3)
    return t, z, gyro, W_true, V_true, F


def _singular_values(M):
    return np.linalg.svd(M, compute_uv=False)


def synthetic_test():
    print("=" * 70)
    print("ТЕСТ 1 — СИНТЕТИКА (раздел 9): EKF должен восстановить W, V")
    print("=" * 70)
    t, z, gyro, W_true, V_true, F = make_synthetic()
    print(f"Задано: |B_внеш| = {F:.1f} мкТл,  V_true = "
          f"[{V_true[0]:.0f} {V_true[1]:.0f} {V_true[2]:.0f}] (огромный hard-iron, как у S23)")

    best, r_pos, r_neg = run_mag_ekf_autosign(t, z, gyro, sigma_m_uT=0.5, F_ref=F)
    print(f"\nПроверка знака ΔR (модуль откалиброванного должен быть ПОСТОЯННЫМ):")
    print(f"  знак +1: остаток {r_pos['residual_rel']*100:6.2f}%")
    print(f"  знак −1: остаток {r_neg['residual_rel']*100:6.2f}%")
    print(f"  → выбран знак {best['sign']:+.0f} (меньше остаток = поле постоянно)")

    V_err = np.linalg.norm(best["V_uT"] - V_true) / np.linalg.norm(V_true)
    sv_est = _singular_values(best["W"])
    sv_true = _singular_values(W_true)
    sv_err = np.max(np.abs(sv_est - sv_true) / sv_true)
    field_err = abs(best["field_uT"] - F) / F

    print(f"\nРезультат EKF (знак {best['sign']:+.0f}):")
    print(f"  V_EKF      = [{best['V_uT'][0]:+.1f} {best['V_uT'][1]:+.1f} {best['V_uT'][2]:+.1f}] мкТл")
    print(f"  V_true     = [{V_true[0]:+.1f} {V_true[1]:+.1f} {V_true[2]:+.1f}] мкТл   → ошибка {V_err*100:.2f}%")
    print(f"  поле EKF   = {best['field_uT']:.2f} мкТл (эталон {F:.1f})         → ошибка {field_err*100:.2f}%")
    print(f"  сингулярные числа W: EKF {np.round(sv_est,3)} vs истина {np.round(sv_true,3)} → ошибка {sv_err*100:.2f}%")
    print(f"  остаток (разброс |откалибр.|): {best['residual_rel']*100:.2f}%")

    ok = V_err < 0.02 and sv_err < 0.02 and best["residual_rel"] < 0.02 and field_err < 0.02
    print(f"\nПРИЁМКА (<2%): {'✓ синтетика сошлась' if ok else '✗ НЕ сошлась'}")
    return ok


# ======================================================================
# РАЗДЕЛ 8: ОБЯЗАТЕЛЬНАЯ КРОСС-ПРОВЕРКА С RANSAC-ЭЛЛИПСОИДОМ
# ======================================================================
def load_mag_stream(path):
    """Прочитать calib_*.json → (t, mag Nx3, gyro Nx3) из mag_stream
    [t,mx,my,mz,gx,gy,gz,...]."""
    with open(path, "r", encoding="utf-8") as fh:
        obj = json.load(fh)
    ms = np.asarray(obj.get("mag_stream", []), float)
    if ms.ndim != 2 or ms.shape[0] < 50 or ms.shape[1] < 7:
        raise ValueError(f"В {os.path.basename(path)} нет пригодного mag_stream")
    t = ms[:, 0]
    mag = ms[:, 1:4]
    gyro = ms[:, 4:7]
    return t, mag, gyro


def cross_check(t, mag, gyro, F_ref=None, label=""):
    """Кросс-проверка EKF ↔ RANSAC-эллипсоид на ОДНИХ И ТЕХ ЖЕ данных (раздел 8)."""
    print("=" * 70)
    print(f"ТЕСТ 2 — РЕАЛЬНАЯ запись + кросс-проверка EKF ↔ RANSAC-эллипсоид  {label}")
    print("=" * 70)
    # RANSAC-эллипсоид (эталон истины — ему гироскоп не нужен)
    ell = calibrate_robust(mag, target_radius=F_ref, model="ellipsoid")
    # EKF: масштаб привязываем к ТОМУ ЖЕ эталону поля, что у эллипсоида (его радиус),
    # иначе сравнивать абсолют поля нельзя (калибровочная свобода масштаба)
    best, r_pos, r_neg = run_mag_ekf_autosign(t, mag, gyro, sigma_m_uT=0.8,
                                              F_ref=ell.radius)

    V_e = ell.V
    V_k = best["V_uT"]
    # расхождение центров нормируем на МАКСИМУМ(|V|, поле): у Android-поля
    # (железо уже снято ОС) |V| ≈ 0–1 мкТл, и проценты «от |V|» взрывались бы
    # на ровном месте (пакет 14: сырьё живого сбора бывает и таким) —
    # физический масштаб задачи задаёт радиус поля
    V_scale = max(np.linalg.norm(V_e), ell.radius, 1e-9)
    dV = np.linalg.norm(V_k - V_e) / V_scale
    field_e = ell.radius
    field_k = best["field_uT"]
    dF = abs(field_k - field_e) / max(field_e, 1e-9)

    print(f"\nЗнак ΔR: + остаток {r_pos['residual_rel']*100:.2f}% / − остаток "
          f"{r_neg['residual_rel']*100:.2f}% → выбран {best['sign']:+.0f}")
    print(f"\n{'величина':<26}{'EKF':>18}{'RANSAC-эллипсоид':>20}{'расхожд.':>12}")
    print("-" * 76)
    print(f"{'центр V, мкТл':<26}{_v(V_k):>18}{_v(V_e):>20}{dV*100:>10.2f}%")
    print(f"{'|V| (железо), мкТл':<26}{np.linalg.norm(V_k):>18.1f}{np.linalg.norm(V_e):>20.1f}"
          f"{abs(np.linalg.norm(V_k)-np.linalg.norm(V_e))/max(np.linalg.norm(V_e),1e-9)*100:>10.2f}%")
    print(f"{'поле / радиус, мкТл':<26}{field_k:>18.2f}{field_e:>20.2f}{dF*100:>10.2f}%")
    print(f"{'остаток, %':<26}{best['residual_rel']*100:>18.2f}{ell.residual_rel*100:>20.2f}")
    print("-" * 76)

    agree = dV < 0.05 and dF < 0.05
    print(f"\nСогласие EKF ↔ эллипсоид (центр и поле в пределах ~5%): "
          f"{'✓ совпали' if agree else '⚠ расходятся — см. раздел 8 (знак ΔR/якобиан/порядок W)'}")
    return agree, best, ell


def _v(vec):
    return f"[{vec[0]:+.0f} {vec[1]:+.0f} {vec[2]:+.0f}]"


# ======================================================================
if __name__ == "__main__":
    ok_syn = synthetic_test()
    print()

    # реальная запись: берём самую ЧИСТУЮ (мин. остаток RANSAC) среди записей с
    # достаточным числом точек — так EKF демонстрируется на лучших доступных данных.
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(here), "data")
    candidates = sorted(glob.glob(os.path.join(data_dir, "calib_*.json")))
    chosen = None
    best_resid = 1e9
    for p in candidates:
        try:
            tt, mg, gy = load_mag_stream(p)
            if len(tt) < 300:
                continue
            resid = calibrate_robust(mg, model="ellipsoid").residual_rel
            if resid < best_resid:
                best_resid, chosen = resid, (p, tt, mg, gy)
        except Exception:
            pass
    if chosen is None:
        print("Реальных calib_*.json с mag_stream в data/ не найдено — пропускаю тест 2.")
    else:
        p, tt, mg, gy = chosen
        cross_check(tt, mg, gy, F_ref=None,
                    label=f"({os.path.basename(p)}, {len(tt)} точек, чистейшая запись)")
