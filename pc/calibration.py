# -*- coding: utf-8 -*-
"""
calibration.py
==============
КАЛИБРОВКА 3-осевых датчиков (магнитометр, акселерометр) подгонкой ЭЛЛИПСОИДА.

ЗАЧЕМ (простыми словами)
------------------------
Если медленно повернуть телефон во все стороны, показания магнитометра (или
акселерометра в покое) должны лечь на ШАР: длина вектора всегда одна и та же
(|B| для магнита, g для ускорения), меняется только направление.

На деле из-за искажений точки ложатся не на шар, а на наклонный сплющенный
ЭЛЛИПСОИД, ещё и сдвинутый от нуля:
  • сдвиг центра          = hard-iron (постоянная помеха / смещение нуля), вектор V;
  • растяжение/наклон     = soft-iron (искажение масштаба по осям), матрица.

Калибровка = найти этот эллипсоид и «вернуть» его в шар:
      калиброванная точка = M · (сырая − V)
где V — центр эллипсоида, M — корректирующая матрица (soft-iron).

МАТЕМАТИКА (без хаков)
----------------------
1) Подгоняем ОБЩИЙ КВАДРИК (уравнение эллипсоида в пространстве) методом
   наименьших квадратов:
        A x² + B y² + C z² + 2D xy + 2E xz + 2F yz + 2G x + 2H y + 2I z = 1
   Это линейная задача по 9 коэффициентам → одна строка numpy.lstsq.
   В матричном виде:  pᵀ·Aₘ·p + nᵀ·p = 1,
        Aₘ = [[A,D,E],[D,B,F],[E,F,C]],   n = [2G, 2H, 2I].

2) Центр (hard-iron):   V = −0.5 · Aₘ⁻¹ · n   (там, где градиент квадрика = 0).

3) Сдвигаем в центр: для q = p − V уравнение становится  qᵀ·Aₘ·q = d,
   где d = 1 − 0.5·nᵀ·V. Делаем собственное разложение Aₘ = U·Λ·Uᵀ.
   Полуоси эллипсоида = sqrt(d/λᵢ). Корректирующая матрица, превращающая
   эллипсоид в шар радиуса r:
        M = r · U · sqrt(Λ/d) · Uᵀ   (= r · sqrt(Aₘ/d)).
   Тогда |M·q| = r для всех точек эллипсоида.

4) Радиус-эталон r: для акселерометра = локальное g; для магнитометра =
   средняя длина |сырая − V|.

Остаточная ошибка = насколько калиброванные точки реально легли на шар радиуса r
(СКО отклонения их длины от r).
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class CalibResult:
    """Результат калибровки одного датчика."""
    V: np.ndarray            # (3,)   центр / hard-iron смещение
    M_corr: np.ndarray       # (3,3)  корректирующая матрица soft-iron
    radius: float            # целевой радиус шара (g или |B|)
    axes: np.ndarray         # (3,)   полуоси найденного эллипсоида (в исходных единицах)
    scales: np.ndarray       # (3,)   масштабные коэффициенты = radius / полуось
    evecs: np.ndarray        # (3,3)  ориентация эллипсоида (собственные векторы)
    residual_abs: float      # остаточная ошибка, в единицах датчика
    residual_rel: float      # она же в долях (доля от радиуса)
    n_points: int            # число точек
    raw: np.ndarray          # (N,3)  сырые точки
    calibrated: np.ndarray   # (N,3)  калиброванные точки
    raw_radii_minmax: tuple  # (min, max) длины сырых векторов от центра
    model: str = "ellipsoid"      # "ellipsoid" (полный) или "diagonal" (смещение+масштаб)
    mean_raw_radius: float = 0.0  # средняя длина |сырая − V| (натуральный радиус данных)
    # --- поля отбраковки выбросов (заполняет calibrate_robust) ---
    n_total: int = 0              # сколько точек было ДО отбраковки
    n_dropped: int = 0            # сколько точек отброшено как выбросы
    residual_before_rel: float = 0.0  # остаток ДО отбраковки (доля от радиуса)
    robust: bool = False          # была ли отбраковка выбросов


def fit_ellipsoid(points: np.ndarray):
    """
    Подгонка общего квадрика к облаку точек (метод наименьших квадратов).
    Возвращает (Aₘ 3x3, n 3,) из уравнения pᵀ·Aₘ·p + nᵀ·p = 1.
    """
    P = np.asarray(points, dtype=float)
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    # столбцы под коэффициенты [A, B, C, D, E, F, G, H, I]
    Dmat = np.column_stack([
        x * x, y * y, z * z,
        2 * x * y, 2 * x * z, 2 * y * z,
        2 * x, 2 * y, 2 * z,
    ])
    rhs = np.ones(P.shape[0])
    coef, *_ = np.linalg.lstsq(Dmat, rhs, rcond=None)
    A, B, C, D, E, F, G, H, I = coef
    Am = np.array([[A, D, E],
                   [D, B, F],
                   [E, F, C]], dtype=float)
    n = np.array([2 * G, 2 * H, 2 * I], dtype=float)
    return Am, n


def calibrate(points: np.ndarray, target_radius: float | None = None) -> CalibResult:
    """
    Главная функция: по облаку сырых точек вернуть центр V и матрицу M_corr.

    target_radius — желаемый радиус шара:
        акселерометр → локальное g (например, 9.81);
        магнитометр  → None (тогда берём среднюю длину |сырая − V|).
    """
    P = np.asarray(points, dtype=float)
    if P.shape[0] < 9:
        raise ValueError("Нужно минимум 9 точек для подгонки эллипсоида")

    # --- ПРЕДЦЕНТРИРОВАНИЕ + масштаб (для устойчивости вычислений) ---
    # ВАЖНО: сначала вычитаем среднее (грубый центр облака), и только потом
    # подгоняем квадрик. На S23 сырое смещение магнитометра ~1230 мкТл (внутренний
    # магнит, виден лишь в uncalibrated). Если НЕ вычесть его перед подгонкой,
    # расчёт численно разваливается: огромный сдвиг (≫ радиуса) забивает значащие
    # разряды и y²/z²/перекрёстные члены тонут в шуме. Решаем у нуля → возвращаем центр.
    c0 = P.mean(axis=0)
    Pc = P - c0
    s = float(np.sqrt(np.mean(np.sum(Pc * Pc, axis=1))))
    if s == 0.0:
        raise ValueError("Все точки совпали — нечего калибровать")
    Ps = Pc / s

    # --- подгонка эллипсоида в центрированно-масштабированном пространстве ---
    Am, n = fit_ellipsoid(Ps)
    Vs = -0.5 * np.linalg.solve(Am, n)          # центр (в этом пространстве)
    d = 1.0 - 0.5 * float(n @ Vs)               # qᵀ·Aₘ·q = d
    evals, evecs = np.linalg.eigh(Am)           # Aₘ симметрична → eigh
    scaled = evals / d                          # для эллипсоида все > 0
    if np.any(scaled <= 0):
        raise ValueError(
            "Подгонка не дала корректный эллипсоид — мало точек или они в одной "
            "плоскости. Поверните датчик во все стороны (шарик во все углы).")

    # центр в исходных единицах (возвращаем вычтенное среднее)
    V = c0 + s * Vs

    # целевой радиус
    if target_radius is None:
        target_radius = float(np.mean(np.linalg.norm(P - V, axis=1)))
    target_radius = float(target_radius)

    # матрица, переводящая (Ps − Vs) в ЕДИНИЧНЫЙ шар:
    M_corr_unit = evecs @ np.diag(np.sqrt(scaled)) @ evecs.T
    # перевод в исходные единицы и масштаб до целевого радиуса
    M_corr = (target_radius / s) * M_corr_unit

    # калиброванные точки: c = M_corr · (raw − V)
    calibrated = (P - V) @ M_corr.T
    radii = np.linalg.norm(calibrated, axis=1)
    residual_abs = float(np.sqrt(np.mean((radii - target_radius) ** 2)))
    residual_rel = residual_abs / target_radius

    # полуоси эллипсоида и масштабные коэффициенты (в исходных единицах)
    axes = s / np.sqrt(scaled)
    scales = target_radius / axes

    raw_r = np.linalg.norm(P - V, axis=1)

    return CalibResult(
        V=V, M_corr=M_corr, radius=target_radius,
        axes=axes, scales=scales, evecs=evecs,
        residual_abs=residual_abs, residual_rel=residual_rel,
        n_points=P.shape[0], raw=P, calibrated=calibrated,
        raw_radii_minmax=(float(raw_r.min()), float(raw_r.max())),
        model="ellipsoid", mean_raw_radius=float(raw_r.mean()),
    )


def calibrate_diagonal(points: np.ndarray, target_radius: float | None = None) -> CalibResult:
    """
    Калибровка по модели «СМЕЩЕНИЕ + МАСШТАБ ПО ОСЯМ» (диагональная, без перекрёстных
    членов). Подходит для АКСЕЛЕРОМЕТРА: устойчивее на 6–12 точках, чем полный
    эллипсоид.

    Подгоняем осе-ориентированный эллипсоид:
        A·x² + B·y² + C·z² + G·x + H·y + I·z = 1   (6 коэффициентов, без xy/xz/yz)
    Центр  Vx = −G/(2A),  Vy = −H/(2B),  Vz = −I/(2C).
    Полуоси a_i = sqrt(d / коэф_i),  где d = 1 + G²/(4A) + H²/(4B) + I²/(4C).
    Масштаб по оси  s_i = r / a_i,  матрица M = diag(s_x, s_y, s_z).
    Калиброванная точка = M·(сырая − V).
    """
    P = np.asarray(points, dtype=float)
    if P.shape[0] < 6:
        raise ValueError("Нужно минимум 6 точек для диагональной калибровки")

    # предцентрирование + масштаб для устойчивости (см. подробно в calibrate())
    c0 = P.mean(axis=0)
    Pc = P - c0
    s = float(np.sqrt(np.mean(np.sum(Pc * Pc, axis=1))))
    if s == 0.0:
        raise ValueError("Все точки совпали — нечего калибровать")
    Ps = Pc / s
    x, y, z = Ps[:, 0], Ps[:, 1], Ps[:, 2]

    # подгонка осе-ориентированного эллипсоида (6 столбцов, без перекрёстных)
    Dmat = np.column_stack([x * x, y * y, z * z, x, y, z])
    coef, *_ = np.linalg.lstsq(Dmat, np.ones(P.shape[0]), rcond=None)
    A, B, C, G, H, I = coef
    if A <= 0 or B <= 0 or C <= 0:
        raise ValueError("Подгонка не дала эллипсоид — мало точек или они в одной плоскости. "
                         "Снимайте телефон в разных положениях (грани, наклоны).")

    Vs = np.array([-G / (2 * A), -H / (2 * B), -I / (2 * C)])
    d = 1.0 + G * G / (4 * A) + H * H / (4 * B) + I * I / (4 * C)
    if d <= 0:
        raise ValueError("Некорректная подгонка акселерометра (d<=0)")

    V = c0 + s * Vs                             # центр в исходных единицах (вернули среднее)
    axes = s * np.sqrt(d / np.array([A, B, C]))  # полуоси в исходных единицах

    if target_radius is None:
        target_radius = float(np.mean(np.linalg.norm(P - V, axis=1)))
    target_radius = float(target_radius)

    scales = target_radius / axes               # масштаб по каждой оси
    M_corr = np.diag(scales)                     # ДИАГОНАЛЬНАЯ матрица (без перекрёстных)

    calibrated = (P - V) @ M_corr.T
    radii = np.linalg.norm(calibrated, axis=1)
    residual_abs = float(np.sqrt(np.mean((radii - target_radius) ** 2)))
    residual_rel = residual_abs / target_radius
    raw_r = np.linalg.norm(P - V, axis=1)

    return CalibResult(
        V=V, M_corr=M_corr, radius=target_radius,
        axes=axes, scales=scales, evecs=np.eye(3),
        residual_abs=residual_abs, residual_rel=residual_rel,
        n_points=P.shape[0], raw=P, calibrated=calibrated,
        raw_radii_minmax=(float(raw_r.min()), float(raw_r.max())),
        model="diagonal", mean_raw_radius=float(raw_r.mean()),
    )


def calibrate_robust(points: np.ndarray, target_radius: float | None = None,
                     model: str = "ellipsoid", sigma_k: float = 2.5,
                     iters: int = 3, min_keep_frac: float = 0.6) -> CalibResult:
    """
    Калибровка с ОТБРАКОВКОЙ ВЫБРОСОВ (RANSAC-стиль). Свою калибровку НЕ выбрасываем —
    лишь чистим её от «плохих» замеров.

    Зачем: одиночные грубые точки (рывок руки, всплеск помехи рядом с металлом)
    тянут эллипсоид на себя и завышают остаток. Поэтому:
      1) подгоняем эллипсоид по ВСЕМ точкам;
      2) для каждой точки считаем отклонение |калиброванная| − радиус;
      3) отбрасываем точки с |отклонением| > sigma_k·σ (σ — СКО отклонений);
      4) пересчитываем эллипсоид по оставшимся; повторяем до `iters` раз.

    Возвращает CalibResult по «чистым» точкам, с заполненными:
      residual_before_rel — остаток ДО отбраковки;
      n_total, n_dropped  — сколько было и сколько отброшено;
      robust = True.
    """
    P = np.asarray(points, dtype=float)
    fit = calibrate if model == "ellipsoid" else calibrate_diagonal
    min_pts = 9 if model == "ellipsoid" else 6
    n_total = P.shape[0]

    res = fit(P, target_radius)               # первая подгонка по всем точкам
    residual_before = res.residual_rel
    idx = np.arange(n_total)                   # индексы текущих «чистых» точек

    for _ in range(max(0, int(iters))):
        radii = np.linalg.norm(res.calibrated, axis=1)   # длины калиброванных (для P[idx])
        dev = radii - res.radius
        sigma = float(np.std(dev))
        if sigma <= 1e-12:
            break                              # идеально легли — отбрасывать нечего
        keep = np.abs(dev) <= sigma_k * sigma
        n_keep = int(keep.sum())
        if n_keep == keep.size:
            break                              # выбросов нет — сошлись
        # не отбрасываем слишком много: оставляем ≥ min_pts и ≥ доли от всех точек
        if n_keep < max(min_pts, int(min_keep_frac * n_total)):
            break
        idx = idx[keep]
        res = fit(P[idx], target_radius)       # пересчёт по оставшимся

    res.n_total = n_total
    res.n_dropped = n_total - idx.size
    res.residual_before_rel = residual_before
    res.robust = True
    return res


# ----------------------------------------------------------------------
# САМОПРОВЕРКА НА СИНТЕТИКЕ: python calibration.py
# Берём идеальный шар → искажаем ИЗВЕСТНЫМИ V и матрицей + шум →
# проверяем, что калибровка их восстановила.
# ----------------------------------------------------------------------
def make_synthetic(n=600, radius=50.0, V_true=(12.0, -5.0, 8.0),
                   W=None, noise=0.5, seed=0):
    """Идеальный шар, искажённый известными soft-iron (W) и hard-iron (V_true)."""
    rng = np.random.default_rng(seed)
    if W is None:
        # известное искажение soft-iron (симметричное, не единичное)
        W = np.array([[1.10, 0.05, -0.03],
                      [0.05, 0.80, 0.04],
                      [-0.03, 0.04, 1.25]])
    dirs = rng.normal(size=(n, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)   # равномерно по сфере
    true = dirs * radius                                  # идеальный шар
    raw = true @ W.T + np.asarray(V_true, float)          # исказили + сместили
    raw += rng.normal(scale=noise, size=raw.shape)        # шум датчика
    return raw, np.asarray(V_true, float), W, radius


if __name__ == "__main__":
    print("=" * 60)
    print("САМОПРОВЕРКА КАЛИБРОВКИ НА СИНТЕТИКЕ")
    print("=" * 60)

    raw, V_true, W, R = make_synthetic()
    res = calibrate(raw, target_radius=R)

    print(f"Точек: {res.n_points}")
    print(f"Радиусы СЫРЫХ точек (от центра): "
          f"{res.raw_radii_minmax[0]:.2f} … {res.raw_radii_minmax[1]:.2f} "
          f"(должны были бы быть {R:.1f}, но разъехались из-за искажений)")
    print()
    print(f"Центр V истинный   : [{V_true[0]:+.3f} {V_true[1]:+.3f} {V_true[2]:+.3f}]")
    print(f"Центр V найденный  : [{res.V[0]:+.3f} {res.V[1]:+.3f} {res.V[2]:+.3f}]")
    print(f"Ошибка центра      : {np.linalg.norm(res.V - V_true):.4f}")
    print()
    print(f"Целевой радиус     : {res.radius:.3f}")
    print(f"Остаток после калибровки: {res.residual_abs:.4f} "
          f"({res.residual_rel * 100:.2f}%)  ← насколько точки легли на шар")
    print()
    # soft-iron восстанавливается с точностью до ПОВОРОТА (шар симметричен),
    # поэтому проверяем, что M_corr·W — ортогональная матрица (≈ единичная по модулю).
    OW = res.M_corr @ W
    should_be_I = OW @ OW.T
    err_orth = np.max(np.abs(should_be_I - np.eye(3)))
    print("Проверка soft-iron (M_corr·W должна быть ортогональной → (M·W)(M·W)ᵀ ≈ I):")
    print(np.array2string(should_be_I, formatter={'float_kind': lambda v: f'{v:+.3f}'}))
    print(f"Макс. отклонение от единичной: {err_orth:.4f}")
    print()

    ok = (np.linalg.norm(res.V - V_true) < 0.5
          and res.residual_rel < 0.02
          and err_orth < 0.05)
    print("РЕЗУЛЬТАТ:", "✓ калибровка восстановила искажения" if ok
          else "✗ что-то не так")

    # ------------------------------------------------------------------
    # ТЕСТ 2: большой hard-iron (как внутренний магнит S23 ≈1230) + выбросы.
    # Проверяем, что (а) предцентрирование не даёт расчёту развалиться,
    # (б) отбраковка выбросов роняет остаток.
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("ТЕСТ 2: большое смещение ≈1230 + выбросы → предцентр + отбраковка")
    print("=" * 60)
    raw2, V2_true, W2, R2 = make_synthetic(
        n=400, radius=50.0, V_true=(1230.0, -40.0, 25.0), noise=0.6, seed=3)
    # добавляем ~5% грубых выбросов (рывки/помехи)
    rng = np.random.default_rng(7)
    n_out = max(1, int(0.05 * raw2.shape[0]))
    out_idx = rng.choice(raw2.shape[0], size=n_out, replace=False)
    raw2[out_idx] += rng.normal(scale=60.0, size=(n_out, 3))

    res2 = calibrate_robust(raw2, target_radius=R2, model="ellipsoid")
    print(f"Точек: {res2.n_total},  отброшено выбросов: {res2.n_dropped} "
          f"({res2.n_dropped / res2.n_total * 100:.1f}%)")
    print(f"Центр V истинный : [{V2_true[0]:+.2f} {V2_true[1]:+.2f} {V2_true[2]:+.2f}]")
    print(f"Центр V найденный: [{res2.V[0]:+.2f} {res2.V[1]:+.2f} {res2.V[2]:+.2f}]")
    print(f"Ошибка центра    : {np.linalg.norm(res2.V - V2_true):.3f}")
    print(f"Остаток ДО отбраковки : {res2.residual_before_rel * 100:.2f}%")
    print(f"Остаток ПОСЛЕ         : {res2.residual_rel * 100:.2f}%  "
          f"({'упал ✓' if res2.residual_rel < res2.residual_before_rel else 'не упал ✗'})")
    ok2 = (np.linalg.norm(res2.V - V2_true) < 3.0
           and res2.residual_rel <= res2.residual_before_rel)
    print("РЕЗУЛЬТАТ ТЕСТА 2:",
          "✓ предцентрирование и отбраковка работают" if ok2 else "✗ что-то не так")
