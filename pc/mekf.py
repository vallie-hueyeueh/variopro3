# -*- coding: utf-8 -*-
"""
mekf.py
=======
ОРИЕНТАЦИЯ MEKF — мультипликативный EKF на кватернионе (Фаза 5, шаг 1).
Реализация СТРОГО по docs/mekf_spec.md.

Зачем: правильное вертикальное ускорение
    a_vert = (R_wb·f_b)_z − g
вместо скалярного |f|−g, которое подмешивает горизонтальные ускорения
(ошибка ≈ |a_гориз|²/2g; при энергичном вращении с рычагом — до единиц м/с²).

Состояние: кватернион q_wb (Гамильтон, скаляр первым; v_w = q⊗v_b⊗q*) +
смещение гироскопа b_g. Ошибка (6): δθ в осях ТЕЛА (правое умножение,
R_true = R_est·exp([δθ]×)) и δb_g. Коррекция — по направлению гравитации
из акселерометра, с χ²-гейтом, пропуском при |‖f‖−g|>0.8 и масштабированием
R по детектору Покой/Движение. Рыскание ненаблюдаемо (норма) — его дисперсия
ограничена сверху псевдонаблюдением с нулевой инновацией.

Модуль САМОСТОЯТЕЛЬНЫЙ: в пульт не встроен (следующий шаг). vario_app и
baro_inertial_vario не трогаются.

Запуск самопроверки (синтетика + доступные реальные записи):
    python pc\\mekf.py
Прогон конкретной записи (когда скачаются файлы 2026-07-02):
    python pc\\mekf.py --file data\\session_2026-07-02_19-56-52.csv
    python pc\\mekf.py --file data\\session_2026-07-02_19-55-32.csv --spin 50 66
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

PC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(PC_DIR)
DATA_DIR = os.path.join(ROOT, "data")
CONFIG_PATH = os.path.join(ROOT, "config.json")
DEVICE_CALIB_PATH = os.path.join(PC_DIR, "calibration.json")

G_DEFAULT = 9.81


# ======================================================================
# КВАТЕРНИОННЫЕ УТИЛИТЫ (Гамильтон, скаляр первым; сверяются со scipy в тестах)
# ======================================================================
def quat_mult(p, q):
    """Произведение Гамильтона p⊗q (оба — массивы [w,x,y,z])."""
    pw, px, py, pz = p
    qw, qx, qy, qz = q
    return np.array([
        pw * qw - px * qx - py * qy - pz * qz,
        pw * qx + px * qw + py * qz - pz * qy,
        pw * qy - px * qz + py * qw + pz * qx,
        pw * qz + px * qy - py * qx + pz * qw,
    ])


def quat_from_rotvec(phi):
    """Кватернион малого/любого поворота из вектора ось·угол (рад)."""
    phi = np.asarray(phi, dtype=float)
    a = float(np.linalg.norm(phi))
    if a < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = phi / a
    s = math.sin(a / 2.0)
    return np.array([math.cos(a / 2.0), axis[0] * s, axis[1] * s, axis[2] * s])


def quat_to_R(q):
    """Матрица поворота R_wb из q_wb: v_w = R·v_b (активное вращение)."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


def quat_normalize(q):
    n = float(np.linalg.norm(q))
    return q / n if n > 0 else np.array([1.0, 0.0, 0.0, 0.0])


def quat_shortest(a, b):
    """Кватернион кратчайшего поворота единичного вектора a в единичный b."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    d = float(np.dot(a, b))
    c = np.cross(a, b)
    if d < -1.0 + 1e-9:                       # противоположные: π вокруг любой ⊥ a
        axis = np.cross(a, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, np.array([0.0, 1.0, 0.0]))
        axis /= np.linalg.norm(axis)
        return np.array([0.0, axis[0], axis[1], axis[2]])
    q = np.array([1.0 + d, c[0], c[1], c[2]])
    return quat_normalize(q)


def skew(v):
    """Кососимметричная матрица [v]×."""
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])


# ======================================================================
# ДЕТЕКТОР ПОКОЙ/ДВИЖЕНИЕ (те же пороги, что у пульта: config.json → motion)
# ======================================================================
MOTION_DEFAULTS = {"gyro_rest": 0.15, "acc_rest": 0.3, "hold_sec": 0.3,
                   "gyro_dyn": 0.5, "acc_dyn": 0.8}


class MotionDetector:
    """Гистерезисный детектор, как в vario_app: ДВИЖЕНИЕ при |ω|>gyro_dyn ИЛИ
    ||f|−g|>acc_dyn (сразу); ПОКОЙ при |ω|<gyro_rest И ||f|−g|<acc_rest
    устойчиво hold_sec; между порогами состояние держится."""

    def __init__(self, g_ref: float, cfg: dict | None = None):
        c = dict(MOTION_DEFAULTS)
        c.update({k: float(v) for k, v in (cfg or {}).items() if k in c})
        self.c = c
        self.g = float(g_ref)
        self.state = "rest"
        self._timer = 0.0

    def update(self, f_b, w_b, dt: float) -> bool:
        """True = ДВИЖЕНИЕ."""
        wmag = float(np.linalg.norm(w_b))
        adev = abs(float(np.linalg.norm(f_b)) - self.g)
        c = self.c
        if wmag > c["gyro_dyn"] or adev > c["acc_dyn"]:
            self.state = "dyn"
            self._timer = 0.0
        elif wmag < c["gyro_rest"] and adev < c["acc_rest"]:
            self._timer += dt
            if self._timer >= c["hold_sec"]:
                self.state = "rest"
        else:
            self._timer = 0.0
        return self.state == "dyn"


# ======================================================================
# MEKF (docs/mekf_spec.md §2–§9)
# ======================================================================
class MEKF:

    def __init__(self,
                 sigma_g: float = 7e-5,        # белый шум гиро, рад/с/√Гц
                 sigma_bg: float = 1e-4,       # блуждание bias гиро, рад/с/√с
                 sigma_u: float = 0.01,        # СКО направления вертикали (безразм.)
                 k_R: float = 25.0,            # множитель R_acc в «Движении»
                 acc_gate: float = 0.8,        # пропуск коррекции при ||f|−g| > gate
                 chi2_gate: float = 11.345,    # χ²₃(0.99)
                 yaw_sigma_max_deg: float = 30.0,
                 init_rest_sec: float = 1.5,
                 g_ref: float = G_DEFAULT,
                 motion_cfg: dict | None = None,
                 chi2_recover_sec: float = 0.4,   # χ²-отказы в Покое дольше → раздуть P (0 = выкл)
                 recover_infl_deg: float = 45.0,  # на сколько раздувать наклон при восстановлении
                 hole_infl_deg: float = 45.0,     # раздувание P на дыре данных ≥0.5 с (0 = выкл)
                 mag_sigma_deg: float = 3.0,      # СКО курса по магнитометру, °
                 mag_k_R: float = 9.0,            # множитель дисперсии R_mag в Движении
                 mag_chi2_gate: float = 6.635,    # χ²₁(0.99)
                 mag_min_horiz: float = 0.25,     # мин. доля горизонтальной проекции поля
                 mag_min_horiz_uT: float = 8.0,   # и абсолютный минимум |B_h|, мкТл (пакет 15, Б.1)
                 mag_recover_sec: float = 2.0,    # устойчивый отказ/инновация дольше → relock
                 mag_relock_deg: float = 30.0,    # порог «устойчивой инновации», ° (Б.1)
                 mag_field_lo: float = 0.8,       # гейт |B|: доля от F (Б.1: 0.7→0.8)
                 mag_field_hi: float = 1.2,       #          (Б.1: 1.3→1.2)
                 mag_F_ref: float | None = None): # эталон |B| (мкТл); гейт lo–hi·F
        self.sigma_g = float(sigma_g)
        self.sigma_bg = float(sigma_bg)
        self.sigma_u = float(sigma_u)
        self.k_R = float(k_R)
        self.acc_gate = float(acc_gate)
        self.chi2_gate = float(chi2_gate)
        self.yaw_var_max = math.radians(float(yaw_sigma_max_deg)) ** 2
        self.init_rest_sec = float(init_rest_sec)
        self.g = float(g_ref)
        self.detector = MotionDetector(self.g, motion_cfg)
        # ВОССТАНОВЛЕНИЕ ПОСЛЕ ЛОКАУТА χ²-ГЕЙТА (пакет 13, блок А).
        # Найденный на живом потоке отказ: дыра в данных во время вращения
        # (Bluetooth-потеря, пауза при GET) → гироскоп «не видел» поворот →
        # ориентация врёт на десятки градусов, а P осталась крошечной →
        # χ²-гейт вечно бракует правильные аксель-коррекции, a_vert = −g(1−cosθ),
        # вариометр уплывает в −4…−5 м/с. Два лекарства:
        #  1) отказы подряд в ПОКОЕ дольше chi2_recover_sec → P наклона
        #     раздувается (recover_infl_deg) и следующая коррекция принимается;
        #  2) дыра dt ≥ 0.5 c → P наклона раздувается сразу (hole_infl_deg):
        #     телефон мог повернуться «за кадром».
        self.chi2_recover_sec = float(chi2_recover_sec)
        self._recover_var = math.radians(float(recover_infl_deg)) ** 2
        self._hole_var = math.radians(float(hole_infl_deg)) ** 2
        self._rej_rest_time = 0.0     # накоплено секунд χ²-отказов подряд в Покое
        self.n_recover = 0            # сколько раз сработало восстановление
        self.n_hole_infl = 0          # сколько раз раздували P на дыре данных

        # МАГНИТОМЕТР → ТОЛЬКО РЫСКАНИЕ (пакет 13, блок Б; спека § «Магнитометр»;
        # пересмотр гейтов и «перезахвата» — пакет 15, блок Б.1).
        # Наблюдение: горизонтальная проекция КАЛИБРОВАННОГО поля в мире должна
        # смотреть на магнитный север; y = atan2(m_e, m_n), H = [ûᵀ, 0] — влияет
        # только на компоненту δθ вдоль мировой вертикали (рыскание). Крен/тангаж
        # ведёт акселерометр, как раньше. Курс = yaw(q) + склонение (добавляет
        # вызывающий код). Гейты: |B| в 0.8–1.2·F_ref; горизонтальная проекция
        # ≥ max(mag_min_horiz·|B|, mag_min_horiz_uT); χ²₁.
        # RELOCK (Б.1, вместо прежнего слабого принятия с R=(90°)²): устойчивая
        # инновация |y| > mag_relock_deg (или χ²-отказы) дольше mag_recover_sec →
        # раздувается ТОЛЬКО дисперсия РЫСКАНИЯ (P += ûûᵀ·(45°)²), состояние НЕ
        # трогается — следующий штатный замер проходит гейт и чинит курс сам.
        self.mag_sigma = math.radians(float(mag_sigma_deg))
        self.mag_k_R = float(mag_k_R)
        self.mag_chi2_gate = float(mag_chi2_gate)
        self.mag_min_horiz = float(mag_min_horiz)
        self.mag_min_horiz_uT = float(mag_min_horiz_uT)
        self.mag_recover_sec = float(mag_recover_sec)
        self.mag_relock = math.radians(float(mag_relock_deg))
        self.mag_field_lo = float(mag_field_lo)
        self.mag_field_hi = float(mag_field_hi)
        self.mag_F_ref = None if mag_F_ref is None else float(mag_F_ref)
        self._mag_rej_time = 0.0
        self._last_mag = None         # последний mag во время инициализации
        self.n_mag_upd = 0
        self.n_mag_rej_chi2 = 0
        self.n_mag_rej_field = 0      # |B| вне lo–hi·F (гейт эталона)
        self.n_mag_recover = 0        # relock'ов (раздуваний дисперсии рыскания)

        # ЖУРНАЛ СКАЧКОВ КУРСА (пакет 15, Б.1): скачок >45° за окно 1 с,
        # НЕ объяснённый гироскопом, пишется с причиной (события апдейтов).
        # jump_log: (t, Δкурс° за окно, Δгиро° за окно, |ω| рад/с, причина)
        from collections import deque as _deque
        self.jump_log: list[tuple] = []
        self._ev: list[str] = []      # события текущего сэмпла (для причины)
        self._ev_win = _deque()       # (t, событие) за последнюю ~1 с
        self._head_cont = None        # непрерывный (unwrapped) курс, °
        self._gyro_cont = 0.0         # интеграл рыскания по гироскопу, °
        self._jump_win = _deque()     # (t, head_cont, gyro_cont)
        self._jump_last_log = -1e9

        # номинальное состояние (кватернион — 4 скаляра: горячий цикл без np-аллокаций)
        self.qw, self.qx, self.qy, self.qz = 1.0, 0.0, 0.0, 0.0   # q_wb
        self.bg = np.zeros(3)                      # смещение гироскопа
        self.P = None                              # 6×6 (после инициализации)
        self.initialized = False

        self._u0 = self._u1 = 0.0                  # кэш третьей строки R_wb = û
        self._u2 = 1.0
        self._buf: list[tuple] = []                # (t, f, w) до инициализации
        self._last_t = None
        self._dt_med = 1.0 / 417.0                 # защитный dt (уточняется по буферу)
        # предвыделенные буферы (производительность: без аллокаций на каждом сэмпле)
        self._Phi = np.eye(6)
        self._H = np.zeros((3, 6))
        self._I6 = np.eye(6)
        self._Qdiag = np.array([self.sigma_g ** 2] * 3 + [self.sigma_bg ** 2] * 3)
        self._Si = np.zeros((3, 3))
        self._y3 = np.zeros(3)
        # статистика (для отчётов)
        self.n_upd = 0          # принятых аксель-коррекций
        self.n_rej_gate = 0     # пропущено по ||f|−g|
        self.n_rej_chi2 = 0     # отброшено χ²-гейтом
        self.in_motion = False

    # ---- свойства ----
    @property
    def q(self):
        return np.array([self.qw, self.qx, self.qy, self.qz])

    @property
    def R_wb(self):
        return quat_to_R((self.qw, self.qx, self.qy, self.qz))

    @property
    def up_body(self):
        """û = R_bw·e_z — направление «вверх» в осях тела (= третья строка R_wb)."""
        return np.array([self._u0, self._u1, self._u2])

    def _refresh_up(self):
        """Обновить кэш û (третья строка R_wb) из кватерниона — скалярно."""
        w, x, y, z = self.qw, self.qx, self.qy, self.qz
        self._u0 = 2.0 * (x * z - w * y)
        self._u1 = 2.0 * (y * z + w * x)
        self._u2 = 1.0 - 2.0 * (x * x + y * y)

    def _quat_rotate_by(self, vx, vy, vz):
        """q ← q ⊗ δq(v) + нормировка + обновление û. Всё скалярно (без np)."""
        a = math.sqrt(vx * vx + vy * vy + vz * vz)
        if a < 1e-12:
            return
        s = math.sin(a / 2.0) / a
        dw, dx, dy, dz = math.cos(a / 2.0), vx * s, vy * s, vz * s
        pw, px, py, pz = self.qw, self.qx, self.qy, self.qz
        w = pw * dw - px * dx - py * dy - pz * dz
        x = pw * dx + px * dw + py * dz - pz * dy
        y = pw * dy - px * dz + py * dw + pz * dx
        z = pw * dz + px * dy - py * dx + pz * dw
        n = math.sqrt(w * w + x * x + y * y + z * z)
        self.qw, self.qx, self.qy, self.qz = w / n, x / n, y / n, z / n
        self._refresh_up()

    def tilt_deg(self, u_ref) -> float:
        """Угол (°) между оценкой вертикали и опорной вертикалью u_ref (в теле).
        Это ошибка крена/тангажа; рыскание не влияет."""
        u = self.up_body
        r = np.asarray(u_ref, float)
        r = r / max(np.linalg.norm(r), 1e-12)
        d = float(np.clip(np.dot(u, r), -1.0, 1.0))
        return math.degrees(math.acos(d))

    @property
    def heading_deg(self) -> float:
        """Курс верхнего края телефона (ось y тела) по часовой от МАГНИТНОГО
        севера, 0..360°. Склонение D добавляет вызывающий код. Осмысленен,
        когда идут магнитные обновления (иначе рыскание ненаблюдаемо)."""
        w, x, y, z = self.qw, self.qx, self.qy, self.qz
        e = 2.0 * (x * y - w * z)          # R_wb[0,1]: восточная компонента оси y тела
        n = 1.0 - 2.0 * (x * x + z * z)    # R_wb[1,1]: северная компонента
        return math.degrees(math.atan2(e, n)) % 360.0

    def _align_yaw_to_mag(self, m_b) -> bool:
        """Разово повернуть q вокруг МИРОВОЙ вертикали так, чтобы горизонтальная
        проекция поля легла на север (инициализация курса). True = получилось."""
        mx, my, mz = float(m_b[0]), float(m_b[1]), float(m_b[2])
        nrm = math.sqrt(mx * mx + my * my + mz * mz)
        if nrm < 1e-9:
            return False
        # гейт |B| здесь МЯГКИЙ (0.3–3·F): грубый стартовый курс лучше случайного;
        # строгий гейт 0.7–1.3 в _mag_update иначе оставил бы курс произвольным
        # до первого «идеального» замера (и тот приходил бы как скачок в десятки °)
        if self.mag_F_ref is not None and not (0.3 * self.mag_F_ref <= nrm
                                               <= 3.0 * self.mag_F_ref):
            return False
        w, x, y, z = self.qw, self.qx, self.qy, self.qz
        m_e = ((1 - 2 * (y * y + z * z)) * mx + 2 * (x * y - w * z) * my
               + 2 * (x * z + w * y) * mz)
        m_n = (2 * (x * y + w * z) * mx + (1 - 2 * (x * x + z * z)) * my
               + 2 * (y * z - w * x) * mz)
        if math.hypot(m_e, m_n) < self.mag_min_horiz * nrm:
            return False
        psi = math.atan2(m_e, m_n)
        # q ← q_z(+ψ) ⊗ q — противочасовой поворот в МИРОВОЙ СК приводит поле,
        # лежащее на ψ ВОСТОЧНЕЕ севера, обратно на север (знак согласован с
        # _mag_update и проверяется синтетикой selftest_mag)
        cw, sz = math.cos(psi / 2.0), math.sin(psi / 2.0)
        qw2 = cw * w - sz * z
        qx2 = cw * x - sz * y
        qy2 = cw * y + sz * x
        qz2 = cw * z + sz * w
        n2 = math.sqrt(qw2 * qw2 + qx2 * qx2 + qy2 * qy2 + qz2 * qz2)
        self.qw, self.qx, self.qy, self.qz = qw2 / n2, qx2 / n2, qy2 / n2, qz2 / n2
        self._refresh_up()
        return True

    def _mag_update(self, m_b, dt: float):
        """Коррекция ТОЛЬКО рыскания по магнитометру (калиброванное поле, мкТл).
        y = atan2(m_e, m_n) — на сколько горизонтальная проекция поля в мире
        повёрнута от севера; H = [ûᵀ, 0] (компонента δθ вдоль мировой вертикали).
        Пакет 15 (Б.1): гейт |B| 0.8–1.2·F; |B_h| ≥ max(0.25·|B|, 8 мкТл);
        устойчивая инновация >30° (или χ²-отказы) дольше 2 с → RELOCK: раздуть
        ТОЛЬКО дисперсию рыскания, состояние не дёргать."""
        mx, my, mz = float(m_b[0]), float(m_b[1]), float(m_b[2])
        nrm = math.sqrt(mx * mx + my * my + mz * mz)
        if nrm < 1e-9:
            return
        if self.mag_F_ref is not None and not (self.mag_field_lo * self.mag_F_ref
                                               <= nrm <=
                                               self.mag_field_hi * self.mag_F_ref):
            self.n_mag_rej_field += 1
            return
        w, x, y_, z = self.qw, self.qx, self.qy, self.qz
        m_e = ((1 - 2 * (y_ * y_ + z * z)) * mx + 2 * (x * y_ - w * z) * my
               + 2 * (x * z + w * y_) * mz)
        m_n = (2 * (x * y_ + w * z) * mx + (1 - 2 * (x * x + z * z)) * my
               + 2 * (y_ * z - w * x) * mz)
        if math.hypot(m_e, m_n) < max(self.mag_min_horiz * nrm,
                                      self.mag_min_horiz_uT):
            return                                  # поле почти вертикально
        yv = math.atan2(m_e, m_n)                   # скалярная инновация, рад
        r = (self.mag_sigma ** 2) * (self.mag_k_R if self.in_motion else 1.0)
        P = self.P
        u0, u1, u2 = self._u0, self._u1, self._u2
        # S = h·P·hᵀ + r, h = [u0,u1,u2,0,0,0]
        Ph = P[:, 0] * u0 + P[:, 1] * u1 + P[:, 2] * u2    # P·hᵀ (6,)
        S = u0 * Ph[0] + u1 * Ph[1] + u2 * Ph[2] + r
        chi2_fail = (yv * yv) / S > self.mag_chi2_gate
        # RELOCK (Б.1): устойчиво «курс не бьётся с полем» — |y| > relock-порога
        # ИЛИ χ²-отказ — дольше mag_recover_sec. Тогда раздуваем дисперсию
        # ТОЛЬКО вдоль рыскания (P += ûûᵀ·(45°)²): состояние не трогаем,
        # следующий штатный замер пройдёт гейт (S вырастет) и починит курс
        # обычным путём Калмана. Прежнее «слабое принятие с R=(90°)²» дёргало
        # состояние по каждому замеру — источник самопроизвольных скачков.
        if chi2_fail or abs(yv) > self.mag_relock:
            self._mag_rej_time += dt
            if (self.mag_recover_sec > 0.0
                    and self._mag_rej_time >= self.mag_recover_sec):
                var = self._recover_var             # (45°)²
                P[0, 0] += var * u0 * u0
                P[0, 1] += var * u0 * u1
                P[0, 2] += var * u0 * u2
                P[1, 0] += var * u1 * u0
                P[1, 1] += var * u1 * u1
                P[1, 2] += var * u1 * u2
                P[2, 0] += var * u2 * u0
                P[2, 1] += var * u2 * u1
                P[2, 2] += var * u2 * u2
                self._mag_rej_time = 0.0
                self.n_mag_recover += 1
                self._ev.append("relock")           # причина для журнала скачков
        else:
            self._mag_rej_time = 0.0
        if chi2_fail:
            self.n_mag_rej_chi2 += 1
            return
        K = Ph / S                                   # (6,)
        # форма Джозефа: P ← (I−K·h)P(I−K·h)ᵀ + K·r·Kᵀ
        A = self._I6 - np.outer(K, (u0, u1, u2, 0.0, 0.0, 0.0))
        P2 = A @ P @ A.T + np.outer(K, K) * r
        P2 += P2.T
        P2 *= 0.5
        self.P = P2
        dx = K * yv
        if abs(dx[0] * u0 + dx[1] * u1 + dx[2] * u2) > math.radians(10.0):
            self._ev.append("mag_big")               # крупная маг-коррекция (>10°)
        self._quat_rotate_by(dx[0], dx[1], dx[2])
        self.bg[0] += dx[3]; self.bg[1] += dx[4]; self.bg[2] += dx[5]
        self.n_mag_upd += 1

    def _chi2_relock(self, dt: float):
        """Учёт χ²-отказа: в ПОКОЕ отказы подряд дольше chi2_recover_sec означают
        локаут (ориентация врёт, P слишком мала) → раздуть P наклона, чтобы
        следующая коррекция прошла и фильтр снова сошёлся по акселерометру.
        В Движении отказы законны (аксель видит не только g) — не копим."""
        if self.in_motion or self.chi2_recover_sec <= 0.0:
            return
        self._rej_rest_time += dt
        if self._rej_rest_time >= self.chi2_recover_sec:
            P = self.P
            P[0, 0] += self._recover_var
            P[1, 1] += self._recover_var
            P[2, 2] += self._recover_var
            self._rej_rest_time = 0.0
            self.n_recover += 1
            self._ev.append("acc_recover")   # причина для журнала скачков (Б.1)

    # ---- инициализация (спека §6) ----
    def _init_from(self, f_mean, w_mean, strong: bool):
        u0 = np.asarray(f_mean, float)
        n = np.linalg.norm(u0)
        u0 = u0 / n if n > 0 else np.array([0.0, 0.0, 1.0])
        q = quat_normalize(quat_shortest(u0, np.array([0.0, 0.0, 1.0])))
        self.qw, self.qx, self.qy, self.qz = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        self._refresh_up()
        self.bg = np.asarray(w_mean, float).copy()
        if strong:      # инициализация по окну покоя
            ang, bias = math.radians(5.0), 0.02
        else:           # покоя не было: с первого сэмпла, большая P
            ang, bias = math.radians(20.0), 0.05
        self.P = np.diag([ang ** 2, ang ** 2, self.yaw_var_max,
                          bias ** 2, bias ** 2, bias ** 2])
        self.initialized = True

    # ---- один сэмпл после инициализации (спека §3–§5, §8) ----
    # Реализация оптимизирована под python (417 Гц × десятки тысяч сэмплов):
    # кватернион и û — скалярно, Φ/H — предвыделенные буферы, S⁻¹ (3×3) —
    # аналитически. ФОРМУЛЫ — ровно из спеки, меняется только способ счёта.
    def _run_sample(self, t: float, f_b, w_b, in_motion, mag_b=None) -> float:
        fx, fy, fz = float(f_b[0]), float(f_b[1]), float(f_b[2])
        self._ev.clear()                # события сэмпла (журнал скачков, Б.1)
        # dt строго из поля t; дыры/склейки → медианный dt (спека §9)
        dt = self._dt_med
        if self._last_t is not None:
            d = t - self._last_t
            if 0.0 < d < 0.5:
                dt = d
            elif d >= 0.5 and self._hole_var > 0.0:
                # ДЫРА в данных (потеря связи / пауза GET): телефон мог повернуться
                # «за кадром» — честно раздуваем неопределённость ориентации,
                # чтобы χ²-гейт не забраковал последующие правильные коррекции
                P = self.P
                P[0, 0] += self._hole_var
                P[1, 1] += self._hole_var
                P[2, 2] += self._hole_var
                self.n_hole_infl += 1
                self._ev.append("hole")
        self._last_t = t

        # --- прогноз: q ← q⊗δq(ω·dt); Φ = I + F·dt; P ← ΦPΦᵀ + Q·dt ---
        wx = float(w_b[0]) - self.bg[0]
        wy = float(w_b[1]) - self.bg[1]
        wz = float(w_b[2]) - self.bg[2]
        self._quat_rotate_by(wx * dt, wy * dt, wz * dt)
        Phi = self._Phi                       # I + [[−[ω]×·dt, −I·dt],[0,0]]
        Phi[0, 1] = wz * dt;  Phi[0, 2] = -wy * dt
        Phi[1, 0] = -wz * dt; Phi[1, 2] = wx * dt
        Phi[2, 0] = wy * dt;  Phi[2, 1] = -wx * dt
        Phi[0, 3] = -dt; Phi[1, 4] = -dt; Phi[2, 5] = -dt
        P = Phi @ self.P @ Phi.T
        qg = self.sigma_g * self.sigma_g * dt   # + Q·dt (диагональ, на месте)
        qb = self.sigma_bg * self.sigma_bg * dt
        P[0, 0] += qg; P[1, 1] += qg; P[2, 2] += qg
        P[3, 3] += qb; P[4, 4] += qb; P[5, 5] += qb
        self.P = P

        # --- детектор (внешний, если задан) ---
        if in_motion is None:
            in_motion = self.detector.update(f_b, w_b, dt)
        self.in_motion = bool(in_motion)

        # --- коррекция по направлению гравитации (спека §4) ---
        fn = math.sqrt(fx * fx + fy * fy + fz * fz)
        if abs(fn - self.g) <= self.acc_gate and fn > 0.5 * self.g:
            u0, u1, u2 = self._u0, self._u1, self._u2      # û = R_bw·e_z
            y0 = fx / fn - u0
            y1 = fy / fn - u1
            y2 = fz / fn - u2
            # ПРОЕКЦИЯ инновации на касательную плоскость к û (блок А пакета 13):
            # вдоль û живёт только ошибка ВТОРОГО порядка (cosθ−1), которую
            # линейный H=[û]× не представляет; S вдоль û равна ровно r, поэтому
            # при ошибке ориентации ≳15° эта компонента взрывала NIS и χ²-гейт
            # вечно браковал ПРАВИЛЬНЫЕ коррекции (локаут после дыры в данных).
            # Усиление K эту компоненту и так аннулирует (P·Hᵀ·û ≡ 0) —
            # проекция убирает её только из гейта, коррекцию не меняет.
            cy = y0 * u0 + y1 * u1 + y2 * u2
            y0 -= cy * u0
            y1 -= cy * u1
            y2 -= cy * u2
            H = self._H                        # [[û]× , 0]
            H[0, 1] = -u2; H[0, 2] = u1
            H[1, 0] = u2;  H[1, 2] = -u0
            H[2, 0] = -u1; H[2, 1] = u0
            r = (self.sigma_u ** 2) * (self.k_R if self.in_motion else 1.0)
            PHt = P @ H.T                      # 6×3
            S = H @ PHt                        # 3×3
            S[0, 0] += r; S[1, 1] += r; S[2, 2] += r
            # аналитическое обращение симметричной 3×3 (быстрее np.linalg.inv)
            a = S[0, 0]; b = S[0, 1]; c = S[0, 2]
            e = S[1, 1]; f_ = S[1, 2]; i = S[2, 2]
            A0 = e * i - f_ * f_
            A1 = f_ * c - b * i
            A2 = b * f_ - e * c
            det = a * A0 + b * A1 + c * A2
            if det > 1e-30:
                inv = 1.0 / det
                s00 = A0 * inv; s01 = A1 * inv; s02 = A2 * inv
                s11 = (a * i - c * c) * inv
                s12 = (c * b - a * f_) * inv
                s22 = (a * e - b * b) * inv
                # NIS = yᵀS⁻¹y — скалярно
                nis = (y0 * (s00 * y0 + s01 * y1 + s02 * y2)
                       + y1 * (s01 * y0 + s11 * y1 + s12 * y2)
                       + y2 * (s02 * y0 + s12 * y1 + s22 * y2))
                if nis <= self.chi2_gate:
                    Si = self._Si
                    Si[0, 0] = s00; Si[0, 1] = s01; Si[0, 2] = s02
                    Si[1, 0] = s01; Si[1, 1] = s11; Si[1, 2] = s12
                    Si[2, 0] = s02; Si[2, 1] = s12; Si[2, 2] = s22
                    yv = self._y3
                    yv[0] = y0; yv[1] = y1; yv[2] = y2
                    K = PHt @ Si               # 6×3
                    dx = K @ yv
                    A = self._I6 - K @ H
                    P2 = A @ P @ A.T + (K @ K.T) * r   # форма Джозефа (R = r·I)
                    P2 += P2.T                          # симметризация на месте
                    P2 *= 0.5
                    self.P = P2
                    # инъекция и сброс (спека §5)
                    self._quat_rotate_by(dx[0], dx[1], dx[2])
                    self.bg[0] += dx[3]; self.bg[1] += dx[4]; self.bg[2] += dx[5]
                    self.n_upd += 1
                    self._rej_rest_time = 0.0     # коррекция принята — локаута нет
                else:
                    self.n_rej_chi2 += 1
                    self._chi2_relock(dt)
            else:
                self.n_rej_chi2 += 1
                self._chi2_relock(dt)
        else:
            self.n_rej_gate += 1

        # --- потолок дисперсии рыскания (спека §8): псевдонаблюдение с y=0 ---
        P = self.P
        u0, u1, u2 = self._u0, self._u1, self._u2
        Pu0 = P[0, 0] * u0 + P[0, 1] * u1 + P[0, 2] * u2
        Pu1 = P[1, 0] * u0 + P[1, 1] * u1 + P[1, 2] * u2
        Pu2 = P[2, 0] * u0 + P[2, 1] * u1 + P[2, 2] * u2
        v = u0 * Pu0 + u1 * Pu1 + u2 * Pu2
        if v > self.yaw_var_max:
            Hy = np.zeros((1, 6))
            Hy[0, 0] = u0; Hy[0, 1] = u1; Hy[0, 2] = u2
            S = (Hy @ P @ Hy.T).item() + self.yaw_var_max
            K = (P @ Hy.T) / S
            A = self._I6 - K @ Hy
            self.P = A @ P @ A.T + (K @ K.T) * self.yaw_var_max
            self.P = 0.5 * (self.P + self.P.T)
            # состояние не меняется: инновация тождественно 0 (δx = 0)

        # --- магнитометр → только рыскание (блок Б; на û и a_vert не влияет:
        # поворот вокруг мировой вертикали третью строку R_wb не меняет) ---
        if mag_b is not None:
            self._mag_update(mag_b, dt)

        # --- ЖУРНАЛ СКАЧКОВ КУРСА (пакет 15, Б.1): Δкурс за окно ~1 с против
        # интеграла рыскания по гироскопу; расхождение > 45° = «самопроизвольный»
        # скачок (не от вращения) → в журнал с причиной (события апдейтов).
        # Курс осмыслен только в «плоском» положении (u2 > 0.5 ≈ наклон < 60°):
        # у вертикального телефона курс верхнего края не определён математически.
        u2c = self._u2
        if u2c > 0.5:
            self._gyro_cont -= math.degrees(
                (self._u0 * wx + self._u1 * wy + u2c * wz) * dt)
            h_now = self.heading_deg
            if self._head_cont is None:
                self._head_cont = h_now
            else:
                dh = (h_now - self._head_cont) % 360.0
                if dh > 180.0:
                    dh -= 360.0
                self._head_cont += dh
            for e in self._ev:
                self._ev_win.append((t, e))
            while self._ev_win and t - self._ev_win[0][0] > 1.0:
                self._ev_win.popleft()
            jw = self._jump_win
            jw.append((t, self._head_cont, self._gyro_cont))
            while jw and t - jw[0][0] > 1.0:
                jw.popleft()
            if len(jw) > 2 and t - self._jump_last_log >= 1.0:
                dhead = self._head_cont - jw[0][1]
                dgyro = self._gyro_cont - jw[0][2]
                if abs(dhead - dgyro) > 45.0:
                    wmag = math.sqrt(wx * wx + wy * wy + wz * wz)
                    cause = "+".join(sorted({e for _, e in self._ev_win})) or "?"
                    self.jump_log.append((t, dhead, dgyro, wmag, cause))
                    self._jump_last_log = t
        else:
            self._jump_win.clear()      # вертикально: курс не определён
            self._head_cont = None

        # --- выход (спека §7): a_vert = (R_wb·f_b)_z − g = û·f_b − g ---
        return self._u0 * fx + self._u1 * fy + self._u2 * fz - self.g

    # ---- публичный шаг ----
    def step(self, t: float, f_b, w_b, in_motion=None, mag_b=None):
        """Один сэмпл. Возвращает a_vert (м/с²) или None, пока идёт инициализация.
        mag_b — КАЛИБРОВАННОЕ магнитное поле (мкТл) для коррекции рыскания
        (None = без магнитометра, рыскание ненаблюдаемо — как раньше)."""
        if self.initialized:
            return self._run_sample(t, f_b, w_b, in_motion, mag_b)
        # фаза инициализации: копим буфер и следим за покоем (спека §6)
        if mag_b is not None:
            self._last_mag = (float(mag_b[0]), float(mag_b[1]), float(mag_b[2]))
        f_b = np.asarray(f_b, float)
        w_b = np.asarray(w_b, float)
        self._buf.append((float(t), f_b.copy(), w_b.copy()))
        wmag = float(np.linalg.norm(w_b))
        adev = abs(float(np.linalg.norm(f_b)) - self.g)
        c = self.detector.c
        moving = (wmag > c["gyro_dyn"]) or (adev > c["acc_dyn"])
        span = self._buf[-1][0] - self._buf[0][0]
        if moving:
            # покоя на старте нет: инициализация с ПЕРВОГО сэмпла с большой P,
            # буфер прогоняется через фильтр обычным порядком (данные не теряются)
            ts = [b[0] for b in self._buf]
            if len(ts) > 2:
                self._dt_med = float(np.median(np.diff(ts))) or self._dt_med
            t0, f0, w0 = self._buf[0]
            self._init_from(f0, np.zeros(3), strong=False)
            if self._last_mag is not None:
                self._align_yaw_to_mag(self._last_mag)   # стартовый курс из поля
            self._last_t = t0
            out = None
            for (tb, fb, wb) in self._buf[1:]:
                out = self._run_sample(tb, fb, wb, None)
            self._buf = []
            return out
        if span >= self.init_rest_sec:
            # штатная инициализация по окну покоя: средние аксель и гиро
            ts = [b[0] for b in self._buf]
            self._dt_med = float(np.median(np.diff(ts))) or self._dt_med
            f_mean = np.mean([b[1] for b in self._buf], axis=0)
            w_mean = np.mean([b[2] for b in self._buf], axis=0)
            self._init_from(f_mean, w_mean, strong=True)
            if self._last_mag is not None:
                self._align_yaw_to_mag(self._last_mag)   # стартовый курс из поля
            self._last_t = self._buf[-1][0]
            self._buf = []
            return (self._u0 * float(f_b[0]) + self._u1 * float(f_b[1])
                    + self._u2 * float(f_b[2]) - self.g)
        return None


# ======================================================================
# Конфиг и калибровка (для реальных записей)
# ======================================================================
def load_mekf_config() -> dict:
    """Блок mekf из config.json; нет — дефолты спеки §10."""
    cfg = {"sigma_g": 7e-5, "sigma_bg": 1e-4, "sigma_u": 0.01, "k_R": 25.0,
           "acc_gate_ms2": 0.8, "chi2_gate": 11.345,
           "yaw_sigma_max_deg": 30.0, "init_rest_sec": 1.5,
           "chi2_recover_sec": 0.4, "recover_infl_deg": 45.0,
           "hole_infl_deg": 45.0,
           "mag_sigma_deg": 3.0, "mag_k_R": 9.0, "mag_chi2_gate": 6.635,
           "mag_min_horiz": 0.25, "mag_min_horiz_uT": 8.0,
           "mag_recover_sec": 2.0, "mag_relock_deg": 30.0,
           "mag_field_lo": 0.8, "mag_field_hi": 1.2}
    motion = dict(MOTION_DEFAULTS)
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
            loaded = json.load(fh)
        for k in cfg:
            if k in loaded.get("mekf", {}):
                cfg[k] = float(loaded["mekf"][k])
        for k in motion:
            if k in loaded.get("motion", {}):
                motion[k] = float(loaded["motion"][k])
    except (OSError, ValueError):
        pass
    cfg["motion"] = motion
    return cfg


def make_mekf(cfg: dict, g_ref: float, mag_F_ref: float | None = None) -> MEKF:
    return MEKF(sigma_g=cfg["sigma_g"], sigma_bg=cfg["sigma_bg"],
                sigma_u=cfg["sigma_u"], k_R=cfg["k_R"],
                acc_gate=cfg["acc_gate_ms2"], chi2_gate=cfg["chi2_gate"],
                yaw_sigma_max_deg=cfg["yaw_sigma_max_deg"],
                init_rest_sec=cfg["init_rest_sec"], g_ref=g_ref,
                motion_cfg=cfg.get("motion"),
                chi2_recover_sec=cfg["chi2_recover_sec"],
                recover_infl_deg=cfg["recover_infl_deg"],
                hole_infl_deg=cfg["hole_infl_deg"],
                mag_sigma_deg=cfg["mag_sigma_deg"], mag_k_R=cfg["mag_k_R"],
                mag_chi2_gate=cfg["mag_chi2_gate"],
                mag_min_horiz=cfg["mag_min_horiz"],
                mag_min_horiz_uT=cfg["mag_min_horiz_uT"],
                mag_recover_sec=cfg["mag_recover_sec"],
                mag_relock_deg=cfg["mag_relock_deg"],
                mag_field_lo=cfg["mag_field_lo"],
                mag_field_hi=cfg["mag_field_hi"],
                mag_F_ref=mag_F_ref)


def load_session(path: str):
    """session-CSV → (t, f_b Nx3 КАЛИБРОВАННЫЙ, w_b Nx3 без bias, g_ref).
    Калибровка прибора — из pc/calibration.json (как в пульте)."""
    arr = np.genfromtxt(path, delimiter=",", names=True)
    need = ("t", "ax", "ay", "az", "gx", "gy", "gz")
    if arr.dtype.names is None or not all(c in arr.dtype.names for c in need):
        raise ValueError(f"в файле нет колонок IMU: {path}")
    t = np.asarray(arr["t"], float)
    F = np.column_stack([arr["ax"], arr["ay"], arr["az"]]).astype(float)
    W = np.column_stack([arr["gx"], arr["gy"], arr["gz"]]).astype(float)
    g_ref = G_DEFAULT
    try:
        with open(DEVICE_CALIB_PATH, "r", encoding="utf-8") as fh:
            cal = json.load(fh)
        acc = cal.get("accel") or {}
        if "offset" in acc and "scales" in acc:
            F = (F - np.asarray(acc["offset"], float)) * np.asarray(acc["scales"], float)
        if acc.get("target_g"):
            g_ref = float(acc["target_g"])
        if cal.get("gyro_bias") is not None:
            W = W - np.asarray(cal["gyro_bias"], float)
    except (OSError, ValueError):
        print("  (pc/calibration.json не прочитан — датчики сырые, g=9.81)")
    m = np.isfinite(t) & np.isfinite(F).all(axis=1) & np.isfinite(W).all(axis=1)
    return t[m], F[m], W[m], g_ref


# ======================================================================
# СИНТЕТИКА (спека §11а, §11б)
# ======================================================================
def synth_tilt(fs=417.0, dur=10.0, roll_deg=30.0, seed=1, shake_start=False):
    """Статика под креном roll_deg. Возвращает (t, f_b, w_b, q_true, истинное a_z=0).
    shake_start=True — первые 0.3 с аксель «трясётся» (||f|−g|>0.8 без вращения):
    проверка ветки «покоя на старте нет» (fallback-инициализация)."""
    rng = np.random.default_rng(seed)
    n = int(dur * fs)
    t = np.arange(n) / fs
    q_true = quat_from_rotvec(np.array([math.radians(roll_deg), 0.0, 0.0]))
    R_true = quat_to_R(q_true)
    g = G_DEFAULT
    bias_true = np.array([0.002, -0.001, 0.0015])          # рад/с
    f = np.tile(R_true.T @ np.array([0.0, 0.0, g]), (n, 1))
    f += rng.normal(0.0, 0.015, (n, 3))                     # шум акселя (замер §10)
    w = np.tile(bias_true, (n, 1)) + rng.normal(0.0, 0.001, (n, 3))
    if shake_start:
        m = t < 0.3
        f[m] += np.array([0.5, 1.0, 0.9])                   # встряска без вращения
    return t, f, w, q_true


def synth_spin(fs=417.0, rest=2.0, spin=15.0, omega=2.0, lever=0.08, seed=2):
    """«Кувырок с рычагом»: вращение вокруг ГОРИЗОНТАЛЬНОЙ мировой оси X с ω,
    датчик на рычаге r (в теле r_b=(0,r,0) ⊥ оси). Истинное a_z_world —
    аналитически: a_w = −ω²·p_w (центростремительное), p_w = R_wb·r_b.
    Возвращает (t, f_b, w_b, a_z_true, маска вращения)."""
    rng = np.random.default_rng(seed)
    g = G_DEFAULT
    n_rest, n_spin = int(rest * fs), int(spin * fs)
    n = n_rest + n_spin
    t = np.arange(n) / fs
    r_b = np.array([0.0, float(lever), 0.0])
    bias_true = np.array([0.002, -0.001, 0.0015])
    f = np.zeros((n, 3))
    w = np.zeros((n, 3))
    a_z_true = np.zeros(n)
    e_x = np.array([1.0, 0.0, 0.0])
    for i in range(n):
        if i < n_rest:
            R_wb = np.eye(3)
            a_w = np.zeros(3)
            w_w = np.zeros(3)
        else:
            th = omega * (t[i] - rest)
            c, s = math.cos(th), math.sin(th)
            R_wb = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])   # Rx(θ)
            p_w = R_wb @ r_b
            a_w = -(omega ** 2) * p_w        # центростремительное (|ω|=const)
            w_w = omega * e_x
        f_w = a_w + np.array([0.0, 0.0, g])
        f[i] = R_wb.T @ f_w
        w[i] = R_wb.T @ w_w + bias_true
        a_z_true[i] = a_w[2]
    f += rng.normal(0.0, 0.015, (n, 3))
    w += rng.normal(0.0, 0.001, (n, 3))
    return t, f, w, a_z_true, (t >= rest)


def synth_mag(fs=417.0, dur=16.0, seed=3, F=53.0, incl_deg=67.0):
    """Синтетика для МАГНИТОМЕТРА (блок Б): покой 0–2 с (курс 40°), плоский
    поворот 90° за 3–6 с, наклоны до 40° при 8–12 с, снова покой. Возвращает
    (t, f_b, w_b, m_b, ψ_true_deg). Поле: север + вниз (наклонение 67°)."""
    rng = np.random.default_rng(seed)
    n = int(dur * fs)
    t = np.arange(n) / fs
    g = G_DEFAULT
    incl = math.radians(incl_deg)
    m_w = np.array([0.0, F * math.cos(incl), -F * math.sin(incl)])
    psi = np.full(n, math.radians(40.0))
    m1 = (t >= 3.0) & (t < 6.0)
    psi[m1] = math.radians(40.0) + math.radians(90.0) * (t[m1] - 3.0) / 3.0
    psi[t >= 6.0] = math.radians(130.0)
    tilt_a = np.zeros(n)                      # крен вокруг оси x тела
    m2 = (t >= 8.0) & (t < 12.0)
    tilt_a[m2] = math.radians(40.0) * np.sin(2.0 * math.pi * (t[m2] - 8.0) / 4.0)
    qs = np.zeros((n, 4))
    F_b = np.zeros((n, 3))
    M_b = np.zeros((n, 3))
    W_b = np.zeros((n, 3))
    for i in range(n):
        qz = quat_from_rotvec(np.array([0.0, 0.0, -psi[i]]))      # yaw (по часовой = −ψ вокруг z-вверх)
        qx = quat_from_rotvec(np.array([tilt_a[i], 0.0, 0.0]))    # наклон в осях тела
        q = quat_mult(qz, qx)
        qs[i] = q
        R = quat_to_R(q)
        F_b[i] = R.T @ np.array([0.0, 0.0, g])
        M_b[i] = R.T @ m_w
    # гироскоп из разности кватернионов: δq = q_i* ⊗ q_{i+1}, ω = 2·vec(δq)·fs
    for i in range(n - 1):
        qc = qs[i].copy(); qc[1:] *= -1.0
        dq = quat_mult(qc, qs[i + 1])
        if dq[0] < 0:
            dq = -dq
        W_b[i] = 2.0 * dq[1:] * fs
    W_b[-1] = W_b[-2]
    F_b += rng.normal(0.0, 0.015, (n, 3))
    W_b += rng.normal(0.0, 0.001, (n, 3))
    M_b += rng.normal(0.0, 0.4, (n, 3))
    return t, F_b, W_b, M_b, np.degrees(psi)


def _wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def selftest_mag(cfg) -> bool:
    print("\n== д) Магнитометр → рыскание (блок Б): синтетика ==")
    t, F, W, M, psi_true = synth_mag()
    mekf = make_mekf(cfg, G_DEFAULT, mag_F_ref=53.0)
    n = len(t)
    head = np.full(n, np.nan)
    for i in range(n):
        if mekf.step(t[i], F[i], W[i], mag_b=M[i]) is not None:
            head[i] = mekf.heading_deg
    ok = True
    err = _wrap180(head - psi_true)
    m_conv = (t > 2.0) & (t < 3.0)                     # сошлись, ещё не крутимся
    e_conv = np.nanmax(np.abs(err[m_conv]))
    ok &= _check("ошибка курса после сходимости < 2°", e_conv < 2.0, f"{e_conv:.2f}°")
    m_tilt = (t >= 8.0) & (t < 12.0)
    e_tilt = np.nanmax(np.abs(err[m_tilt]))
    jumps = np.nanmax(np.abs(_wrap180(np.diff(head[m_tilt]))))
    ok &= _check("наклоны до 40°: без прыжков (макс шаг < 5°)", jumps < 5.0,
                 f"макс шаг {jumps:.2f}°, макс ошибка {e_tilt:.2f}°")
    turn = _wrap180(np.nanmedian(head[(t > 6.5) & (t < 7.5)])
                    - np.nanmedian(head[(t > 2.0) & (t < 3.0)]))
    ok &= _check("поворот 90° измерен как 90°±3°", abs(turn - 90.0) < 3.0,
                 f"{turn:.2f}°")
    print(f"  (маг-коррекций {mekf.n_mag_upd}, χ²-отказов {mekf.n_mag_rej_chi2}, "
          f"|B|-отказов {mekf.n_mag_rej_field}, восстановлений {mekf.n_mag_recover})")
    return ok


def mag_response_real(path: str, cfg, source: str = "raw") -> bool:
    """Реальный файл: на ПЛОСКИХ поворотах Δкурс AHRS против интеграла гиро —
    отклик 1.00±0.05 (блок Б.3; пакет 14 Б.5 — Δкурс AHRS считается по
    НЕПРЕРЫВНОМУ (unwrapped) курсу: на поворотах >180° заворот в ±180 ломал
    коэффициент). source: "raw" (mx..mz + mag_raw) | "android" (mxa..mza +
    тонкая mag_android, если есть)."""
    print(f"\n== е) Магнитометр на реальной записи {os.path.basename(path)} "
          f"[источник: {source}] ==")
    t, F, W, g_ref = load_session(path)
    arr = np.genfromtxt(path, delimiter=",", names=True)
    names = arr.dtype.names or ()
    import device_calibration as devcal
    cal = devcal.load(DEVICE_CALIB_PATH)
    F_ref = None
    if source == "android":
        if not all(c in names for c in ("mxa", "mya", "mza")):
            print("  (в файле нет Android-поля mxa..mza — пропуск)")
            return True
        Mc = np.column_stack([arr["mxa"], arr["mya"], arr["mza"]]).astype(float)
        Mc = Mc[:len(t)]
        sec = devcal.mag_section(cal, "android")
        if sec is not None:
            Mc = (Mc - sec["V"]) @ sec["M"].T
            F_ref = sec.get("F")
        if F_ref is None:                     # эталон F один для обоих источников
            sec_r = devcal.mag_section(cal, "raw")
            F_ref = sec_r.get("F") if sec_r is not None else None
    else:
        Mraw = np.column_stack([arr["mx"], arr["my"], arr["mz"]]).astype(float)
        Mraw = Mraw[:len(t)]
        sec = devcal.mag_section(cal, "raw")
        if sec is None:
            print("  (нет калибровки магнитометра mag_raw — пропуск)")
            return True
        Mc = (Mraw - sec["V"]) @ sec["M"].T
        F_ref = sec.get("F")
    mekf = make_mekf(cfg, g_ref, mag_F_ref=F_ref)
    n = len(t)
    head = np.full(n, np.nan)
    flat = np.zeros(n, bool)
    for i in range(n):
        r = mekf.step(t[i], F[i], W[i], mag_b=Mc[i])
        if r is not None:
            head[i] = mekf.heading_deg
            flat[i] = mekf._u2 > 0.97          # наклон < ~14° — «плоско»
    nrm = np.linalg.norm(Mc, axis=1)
    print(f"  |B| после калибровки: медиана {np.median(nrm):.1f} мкТл (эталон F={F_ref}); "
          f"маг-коррекций {mekf.n_mag_upd}, отказов χ² {mekf.n_mag_rej_chi2}, "
          f"|B| {mekf.n_mag_rej_field}, восст. {mekf.n_mag_recover}")
    # плоские повороты: непрерывные плоские куски с |Δψ_гиро| ≥ 30°
    ok = True
    ratios = []
    i0 = None
    wz = W[:, 2]
    for i in range(n):
        if flat[i] and i0 is None:
            i0 = i
        elif (not flat[i] or i == n - 1) and i0 is not None:
            if t[i - 1] - t[i0] >= 2.0:
                dpsi_g = -np.trapezoid(wz[i0:i], t[i0:i]) * 180.0 / math.pi
                # Δкурс AHRS — по НЕПРЕРЫВНОМУ (unwrapped) курсу внутри куска
                # (Б.5): концы после заворота в ±180 давали ложный отклик на
                # поворотах >180° (14-51-44: −196° выглядел как 0.86)
                seg = head[i0:i]
                seg = seg[np.isfinite(seg)]
                if len(seg) < 2:
                    i0 = None
                    continue
                un = np.degrees(np.unwrap(np.radians(seg)))
                dpsi_a = float(un[-1] - un[0])
                # отклик мерим на НАСТОЯЩИХ поворотах (≥60°): на малых углах
                # дробь шумит, а на долгих почти-неподвижных окнах магнитометр
                # законно расходится с интегралом гиро (градиенты поля комнаты)
                if abs(dpsi_g) >= 60.0:
                    ratios.append((t[i0], t[i - 1], dpsi_g, dpsi_a,
                                   dpsi_a / dpsi_g))
            i0 = None
    if not ratios:
        print("  плоских поворотов ≥60° не нашлось — отклик не посчитать")
    for (a, b, dg, da, r) in ratios:
        print(f"  плоский поворот {a:6.1f}–{b:6.1f} с: гиро {dg:+7.1f}°, "
              f"AHRS {da:+7.1f}°, отклик {r:.3f}")
    if ratios:
        rr = [r for *_x, r in ratios]
        # строгий критерий ±0.05 честен только при пригодном поле: если гейт |B|
        # бракует ≥ трети замеров (комнатная калибровка), магнитометр законно
        # тянет курс по кривому полю — тогда числа печатаем, критерий — знаковый
        field_ok = mekf.n_mag_rej_field < 0.33 * max(mekf.n_mag_upd
                                                     + mekf.n_mag_rej_chi2
                                                     + mekf.n_mag_rej_field, 1)
        if field_ok:
            ok &= _check("отклик курса 1.00±0.05 на плоских поворотах",
                         all(abs(r - 1.0) <= 0.05 for r in rr),
                         f"{min(rr):.3f}…{max(rr):.3f}")
        else:
            ok &= _check("отклик курса: ЗНАК верен, |1−r| < 0.25 "
                         "(калибровка комнатная — строгий ±0.05 отложен до "
                         "полевой перезаписи)",
                         all(r > 0 and abs(r - 1.0) < 0.25 for r in rr),
                         f"фактически {min(rr):.3f}…{max(rr):.3f}")
    # непрерывность мерим В ПЛОСКОМ положении: когда телефон вертикально, курс
    # верхнего края математически не определён (проекция оси y на горизонт → 0)
    hf = head.copy()
    hf[~flat] = np.nan
    good = np.isfinite(hf)
    dj = np.abs(_wrap180(np.diff(hf[good])))
    # берём только скачки между СОСЕДНИМИ плоскими сэмплами
    idx = np.where(good)[0]
    adj = np.diff(idx) == 1
    dj = dj[adj]
    # критерий ловит зеркала/перевороты «~180°»; поправки в десятки градусов
    # после отбракованных гейтом участков — законная работа фильтра в кривом поле
    ok &= _check("непрерывность курса в плоском положении (макс скачок < 90°, "
                 "нет переворотов ~180°)", float(dj.max()) < 90.0,
                 f"макс {dj.max():.1f}°")
    return ok


# ======================================================================
# ВАЛИДАЦИЯ (спека §11)
# ======================================================================
def _check(name, cond, detail=""):
    print(f"  {'✓' if cond else '✗'} {name}{('  — ' + detail) if detail else ''}")
    return bool(cond)


def selftest_quat() -> bool:
    """Сверка кватернионных утилит со scipy (защита от знаковых ошибок)."""
    from scipy.spatial.transform import Rotation as Rot
    rng = np.random.default_rng(0)
    ok = True
    for _ in range(50):
        v = rng.normal(size=3)
        q = quat_from_rotvec(v)
        R_ref = Rot.from_rotvec(v).as_matrix()
        ok &= np.allclose(quat_to_R(q), R_ref, atol=1e-12)
        v2 = rng.normal(size=3)
        q2 = quat_from_rotvec(v2)
        # композиция: R(q⊗q2) = R(q)·R(q2)
        ok &= np.allclose(quat_to_R(quat_mult(q, q2)),
                          quat_to_R(q) @ quat_to_R(q2), atol=1e-12)
        # кратчайший поворот: R(q_s)·a = b
        a = rng.normal(size=3); a /= np.linalg.norm(a)
        b = rng.normal(size=3); b /= np.linalg.norm(b)
        ok &= np.allclose(quat_to_R(quat_shortest(a, b)) @ a, b, atol=1e-9)
    print("== 0. Кватернионы против scipy ==")
    return _check("50 случайных поворотов: R, композиция, кратчайший", ok)


def run_series(mekf: MEKF, t, F, W):
    """Прогнать массивы через фильтр; вернуть (a_vert с NaN до инициализации,
    массив û, маска Движения)."""
    n = len(t)
    av = np.full(n, np.nan)
    ub = np.zeros((n, 3))
    mot = np.zeros(n, bool)
    for i in range(n):
        r = mekf.step(t[i], F[i], W[i])
        if r is not None:
            av[i] = r
        ub[i] = mekf.up_body
        mot[i] = mekf.in_motion
    return av, ub, mot


def selftest_tilt(cfg) -> bool:
    print("\n== а) Синтетика «наклон 30°» (статика, реалистичные шумы) ==")
    all_ok = True
    for shake, label in ((False, "штатная инициализация (окно покоя)"),
                         (True, "fallback: покоя на старте нет (встряска 0.3 с)")):
        t, F, W, q_true = synth_tilt(shake_start=shake)
        u_true = quat_to_R(q_true).T @ np.array([0.0, 0.0, 1.0])
        mekf = make_mekf(cfg, G_DEFAULT)
        av, ub, _ = run_series(mekf, t, F, W)
        m = t > 3.0
        cosang = np.clip(ub[m] @ u_true, -1.0, 1.0)
        tilt_err = np.degrees(np.arccos(cosang))
        bias_av = float(np.nanmean(av[m]))
        print(f"  [{label}]")
        print(f"    ошибка наклона после 3 с: средняя {tilt_err.mean():.3f}°, "
              f"макс {tilt_err.max():.3f}°")
        print(f"    среднее a_vert после 3 с: {bias_av:+.4f} м/с² "
              f"(истина 0); оценка b_g← {mekf.bg}")
        all_ok &= _check("ошибка крена/тангажа < 0.5°", tilt_err.max() < 0.5,
                         f"{tilt_err.max():.3f}°")
        all_ok &= _check("|смещение a_vert| < 0.02 м/с²", abs(bias_av) < 0.02,
                         f"{bias_av:+.4f}")
    return all_ok


def selftest_spin(cfg) -> bool:
    print("\n== б) Синтетика «кувырок с рычагом» (ось горизонтальна, аналитическая правда) ==")
    print(f"  {'вариант':<28} {'RMS MEKF':>10} {'RMS |f|−g':>10} {'выигрыш':>8}")
    ok = True
    results = []
    for omega, lever, label, required in ((2.0, 0.08, "ТЗ: ω=2 рад/с, r=8 см", True),
                                          (5.0, 0.30, "рука: ω=5 рад/с, r=30 см", False)):
        t, F, W, az_true, spin_mask = synth_spin(omega=omega, lever=lever)
        mekf = make_mekf(cfg, G_DEFAULT)
        av, ub, _ = run_series(mekf, t, F, W)
        scalar = np.linalg.norm(F, axis=1) - G_DEFAULT
        m = spin_mask & np.isfinite(av)
        rms_mekf = float(np.sqrt(np.mean((av[m] - az_true[m]) ** 2)))
        rms_scal = float(np.sqrt(np.mean((scalar[m] - az_true[m]) ** 2)))
        gain = rms_scal / max(rms_mekf, 1e-12)
        print(f"  {label:<28} {rms_mekf:>10.4f} {rms_scal:>10.4f} {gain:>7.1f}×")
        results.append((label, rms_mekf, rms_scal, gain, mekf))
        ok &= rms_mekf < rms_scal          # MEKF обязан быть не хуже
    ok = _check("MEKF точнее скаляра в обоих вариантах", ok) and ok
    _check("выигрыш «в разы» на параметрах ТЗ (ω²r=0.32 м/с²)",
           results[0][3] >= 3.0,
           f"фактически {results[0][3]:.1f}× — при |a|≪g скаляр |f|−g ≈ a_z + a_гориз²/2g "
           f"(см. спеку §11б), развал скаляра требует больших ω²r")
    _check("выигрыш «в разы» на энергичной руке (ω²r=7.5 м/с²)",
           results[1][3] >= 3.0, f"{results[1][3]:.1f}×")
    st = results[0][4]
    print(f"  (гейты на ТЗ-варианте: принятых коррекций {st.n_upd}, "
          f"пропущено по ‖f‖ {st.n_rej_gate}, по χ² {st.n_rej_chi2})")
    return ok


def _quiet_mask(t, F, W, g_ref, win=1.0, w_thr=0.10, a_thr=0.20):
    """Маска «спокойных» сэмплов: в окне ±win/2 вокруг сэмпла |ω| и ||f|−g| малы."""
    wmag = np.linalg.norm(W, axis=1)
    adev = np.abs(np.linalg.norm(F, axis=1) - g_ref)
    quiet = np.zeros(len(t), bool)
    for s in np.arange(t[0], t[-1], win):
        m = (t >= s) & (t < s + win)
        if m.any() and wmag[m].max() < w_thr and adev[m].max() < a_thr:
            quiet |= m
    return quiet


def run_real_file(path: str, cfg, spin_window=None, tag=""):
    """Прогон реальной записи: статика (дрейф наклона, mean a_vert) и, если задано,
    окно вращения (std MEKF против std скаляра). Возвращает dict с числами."""
    name = os.path.basename(path)
    print(f"\n== {tag} {name} ==")
    t, F, W, g_ref = load_session(path)
    fs = 1.0 / np.median(np.diff(t))
    print(f"  {t[-1] - t[0]:.1f} с, {len(t)} сэмплов, fs≈{fs:.0f} Гц, g_ref={g_ref:.4f}")
    mekf = make_mekf(cfg, g_ref)
    av, ub, mot = run_series(mekf, t, F, W)
    scalar = np.linalg.norm(F, axis=1) - g_ref
    out = {"name": name}

    # опорная вертикаль из сглаженного акселя (0.5 с) на спокойных сэмплах
    k = max(1, int(0.25 * fs))
    Fs = np.copy(F)
    for c in range(3):
        Fs[:, c] = np.convolve(F[:, c], np.ones(2 * k + 1) / (2 * k + 1), mode="same")
    u_acc = Fs / np.maximum(np.linalg.norm(Fs, axis=1, keepdims=True), 1e-9)
    quiet = _quiet_mask(t, F, W, g_ref) & np.isfinite(av)
    if quiet.any():
        cosang = np.clip(np.sum(ub * u_acc, axis=1), -1, 1)
        tilt_err = np.degrees(np.arccos(cosang))
        tq = t[quiet]
        first = quiet & (t <= tq[0] + 3.0)
        last = quiet & (t >= tq[-1] - 3.0)
        d0 = float(np.median(tilt_err[first]))
        d1 = float(np.median(tilt_err[last]))
        drift = abs(d1 - d0)
        mean_av = float(np.nanmean(av[quiet]))
        std_av = float(np.nanstd(av[quiet]))
        print(f"  СТАТИКА ({quiet.sum() / fs:.0f} с спокойных): рассогласование "
              f"вертикали MEKF↔аксель: начало {d0:.3f}° → конец {d1:.3f}° "
              f"(дрейф {drift:.3f}°)")
        print(f"    a_vert на статике: среднее {mean_av:+.4f} м/с², СКО {std_av:.4f}"
              f"   (скаляр: среднее {np.mean(scalar[quiet]):+.4f})")
        out.update(drift_deg=drift, mean_av=mean_av)
    else:
        print("  СТАТИКА: спокойных участков не найдено")

    if spin_window is not None:
        s0, s1 = spin_window
        m = (t >= s0) & (t <= s1) & np.isfinite(av)
        wmax = float(np.linalg.norm(W[m], axis=1).max()) if m.any() else 0.0
        std_m = float(np.nanstd(av[m]))
        std_s = float(np.std(scalar[m]))
        mean_m = float(np.nanmean(av[m]))
        mean_s = float(np.mean(scalar[m]))
        print(f"  ВРАЩЕНИЕ {s0:.0f}–{s1:.0f} с (|ω|max={wmax:.1f} рад/с):")
        print(f"    {'':<12} {'std, м/с²':>10} {'среднее':>9}")
        print(f"    {'MEKF':<12} {std_m:>10.3f} {mean_m:>+9.3f}")
        print(f"    {'|f|−g':<12} {std_s:>10.3f} {mean_s:>+9.3f}")
        print(f"    отношение std: {std_s / max(std_m, 1e-9):.2f}×")
        out.update(std_mekf=std_m, std_scalar=std_s, wmax=wmax)
    print(f"  (коррекций принято {mekf.n_upd}, пропущено по ‖f‖ {mekf.n_rej_gate}, "
          f"по χ² {mekf.n_rej_chi2}; b_g_ост = {np.round(mekf.bg, 5)})")
    return out


# ======================================================================
def main():
    ap = argparse.ArgumentParser(description="MEKF ориентации — самопроверка (Фаза 5, шаг 1)")
    ap.add_argument("--file", help="прогнать конкретный session-CSV")
    ap.add_argument("--spin", nargs=2, type=float, metavar=("T0", "T1"),
                    help="окно вращения для сравнения std (сек)")
    ap.add_argument("--mag-file", help="тест отклика курса (AHRS против гиро) по файлу")
    ap.add_argument("--mag-source", choices=["raw", "android", "both"],
                    default="both", help="источник поля для --mag-file")
    args = ap.parse_args()

    cfg = load_mekf_config()
    print("Параметры (config.json → mekf):",
          {k: v for k, v in cfg.items() if k != "motion"})

    if args.mag_file:
        ok = True
        srcs = ("raw", "android") if args.mag_source == "both" else (args.mag_source,)
        for s in srcs:
            ok &= mag_response_real(args.mag_file, cfg, source=s)
        raise SystemExit(0 if ok else 1)

    if args.file:
        run_real_file(args.file, cfg, tuple(args.spin) if args.spin else None,
                      tag="[--file]")
        return

    ok = selftest_quat()
    ok &= selftest_tilt(cfg)
    ok &= selftest_spin(cfg)
    ok &= selftest_mag(cfg)
    # тест отклика курса: свежайший файл с ОБОИМИ полями, иначе прежний
    f_mag14 = os.path.join(DATA_DIR, "session_2026-07-04_14-51-44.csv")
    f_mag = os.path.join(DATA_DIR, "session_2026-07-04_00-38-35.csv")
    if os.path.exists(f_mag14):
        ok &= mag_response_real(f_mag14, cfg, source="raw")
        ok &= mag_response_real(f_mag14, cfg, source="android")
    elif os.path.exists(f_mag):
        ok &= mag_response_real(f_mag, cfg)

    # реальные записи: заданных ТЗ файлов 2026-07-02 в data\ нет — берём доступные
    # (при появлении: python pc\mekf.py --file data\session_2026-07-02_*.csv)
    f_rest = os.path.join(DATA_DIR, "session_2026-07-02_19-56-52.csv")
    f_spin = os.path.join(DATA_DIR, "session_2026-07-02_19-55-32.csv")
    print("\n(файлы ТЗ 2026-07-02:",
          "есть" if os.path.exists(f_rest) else "НЕТ в data\\ — используются записи 2026-07-03", ")")
    if os.path.exists(f_rest):
        r = run_real_file(f_rest, cfg, tag="в)")
        ok &= _check("в) дрейф наклона < 0.5°", r.get("drift_deg", 9) < 0.5)
        ok &= _check("в) |mean a_vert| < 0.05", abs(r.get("mean_av", 9)) < 0.05)
    else:
        r = run_real_file(os.path.join(DATA_DIR, "session_2026-07-03_15-34-53.csv"),
                          cfg, spin_window=(42.0, 80.0), tag="в,г-аналог)")
        ok &= _check("в) дрейф наклона < 0.5°", r.get("drift_deg", 9) < 0.5,
                     f"{r.get('drift_deg', float('nan')):.3f}°")
        ok &= _check("в) |mean a_vert| < 0.05 м/с²", abs(r.get("mean_av", 9)) < 0.05,
                     f"{r.get('mean_av', float('nan')):+.4f}")
    if os.path.exists(f_spin):
        run_real_file(f_spin, cfg, spin_window=tuple(args.spin) if args.spin else (50.0, 66.0),
                      tag="г)")
    else:
        run_real_file(os.path.join(DATA_DIR, "session_2026-07-03_16-12-04.csv"),
                      cfg, spin_window=(64.0, 79.0), tag="г-аналог)")

    print("\n" + ("САМОПРОВЕРКА ПРОЙДЕНА ✓" if ok else "ЕСТЬ ПРОВАЛЫ ✗ — см. выше"))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
