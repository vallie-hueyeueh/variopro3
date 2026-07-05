# -*- coding: utf-8 -*-
"""
baro_inertial_vario.py
======================
ЯДРО ВАРИОМЕТРА — баро-инерциальный фильтр Калмана (вертикальный канал).

Это проверенная (как во всех реальных вариометрах: LX, XCTracer, Syride)
архитектура. Она решает ровно твою проблему: значения НЕ уплывают, когда
телефон лежит на столе, и при этом вариометр почти не шумит и реагирует
быстро (малая задержка).

ПОЧЕМУ ОНА РАБОТАЕТ
-------------------
Состояние фильтра: x = [h, v, b]
    h — высота (м)              <- это то, что показывает баро (но баро шумит)
    v — вертикальная скорость   <- ЭТО И ЕСТЬ ВАРИОМЕТР (м/с)
    b — смещение акселерометра  <- фильтр сам его оценивает и вычитает

Барометр даёт АБСОЛЮТНУЮ высоту: шумную, но без накопления ошибки (не уплывает).
Акселерометр даёт ускорение: малошумное на коротком интервале, но если его
просто интегрировать — скорость уходит в бесконечность из-за смещения нуля
(классическая проблема инерциалки).

Фильтр объединяет их: баро держит долгосрочную правду (не даёт уплыть),
акселерометр даёт быструю реакцию (малую задержку). Третья переменная b
«ловит» смещение акселерометра: если телефон стоит, а акселерометр врёт,
что есть ускорение — фильтр понимает это по баро и загоняет ошибку в b.
Поэтому на столе вариометр показывает ~0 и не уплывает.

ВХОД:
    a_world_vertical — вертикальное ускорение в МИРОВОЙ системе координат,
                       уже без g (м/с^2). Его даёт блок ориентации
                       (Madgwick/Mahony/EKF по гиро+аксель+магнитометр),
                       который проецирует ускорение на вертикаль и вычитает 9.81.
    h_baro           — высота по барометру (м), пересчитанная из давления.

ВЫХОД:
    h — сглаженная высота (м)
    v — вертикальная скорость = ВАРИОМЕТР (м/с)
"""

import numpy as np


class BaroInertialVario:
    def __init__(
        self,
        dt: float = 0.02,          # шаг по времени, с (0.02 = 50 Гц)
        sigma_accel: float = 0.30, # СКО ускорения, м/с^2 (шум аксель + ошибка проекции g)
        sigma_baro: float = 0.15,  # СКО измерения высоты баро, м
        sigma_bias: float = 0.003, # скорость "блуждания" смещения аксель, м/с^2 / sqrt(с)
        h0: float = 0.0,           # начальная высота
    ):
        self.dt = float(dt)
        self.sigma_accel = float(sigma_accel)
        self.sigma_baro = float(sigma_baro)
        self.sigma_bias = float(sigma_bias)

        # ПЕРЕМЕННОЕ доверие акселерометру (задаёт vario_app по детектору движения):
        #   accel_trust — множитель на sigma_accel: 1.0 в покое, k_dyn (~5) в динамике.
        #     Больше множитель → больше Q → фильтр меньше верит акселю, опирается на баро.
        #     Именно ПЕРЕМЕННЫЙ (не вкл/выкл): в Фазе 5 его поведёт адаптивная логика.
        #   bias_frozen — True в движении: оценка нуля акселя b НЕ обновляется
        #     (иначе мусор вращения въедается в b, и вариометр врёт и ПОСЛЕ остановки).
        self.accel_trust = 1.0
        self.bias_frozen = False

        # ПАКЕТ 15, блок А — три рычага честности баро-обновления (ставит vario_app):
        #   huber_k     — робастный баро (Хьюбер): при |e| > k·√S дисперсия R
        #     раздувается ×(|e|/(k√S))² — выброс аэродинамики порта (реально до
        #     ±8 м на качании!) перестаёт таскать состояние; честные обновления
        #     не трогаются. 0 = выключено.
        #   R_baro_mult — временный множитель R (детектор КАЧАНИЯ: баро на
        #     размахивании телефона «дышит» портом — на время качания R×K_osc).
        #   zupt_r      — дисперсия ZUPT-измерения v=0 (м/с)²; None = ZUPT нет.
        #     Ставится ТОЛЬКО при строгом покое; применяется внутри step()
        #     ПОСЛЕ баро-коррекции (и до записи RTS — сглаживатель консистентен).
        self.huber_k = 2.0
        self.huber_recover_sec = 2.0  # клип дольше → одно обновление с штатным R
        self.R_baro_mult = 1.0
        self.zupt_r = None
        self.n_huber = 0              # сколько обновлений раздуто Хьюбером
        self.n_huber_recover = 0      # сколько раз «перезахватили» баро после клипа
        self.n_zupt = 0               # сколько ZUPT-обновлений сделано
        self._huber_run = 0.0         # секунд подряд в клипе (для перезахвата)

        # вектор состояния [h, v, b]
        self.x = np.array([[h0], [0.0], [0.0]], dtype=float)

        # ковариация состояния (начальная неопределённость)
        self.P = np.diag([1.0, 1.0, 1.0]).astype(float)

        # ДЛЯ АДАПТИВНЫХ R/Q (пакет 13, блок З.1): последняя инновация баро y
        # и её ковариация S = H·P·Hᵀ + R — по ним считаются NIS и оценка R
        self.last_innov = 0.0
        self.last_S = 1.0
        # ДЛЯ RTS-СГЛАЖИВАТЕЛЯ (блок З.2): если record — список, step() пишет
        # (dt, x_pred, P_pred, x_post, P_post) на каждом шаге (только файловый
        # пакетный прогон; в реальном времени None — накладных расходов нет)
        self.record = None

        # измерение: видим только высоту h. R/H/I от dt НЕ зависят и создаются
        # ОДИН РАЗ (пакет 14, блок Ж): раньше R пересоздавалась в _build_matrices
        # при каждом изменении dt — а dt реального потока дрожит каждый сэмпл,
        # и адаптивный R̂ (З.1) молча затирался паспортным значением.
        self.H = np.array([[1.0, 0.0, 0.0]])
        self.R = np.array([[self.sigma_baro ** 2]])
        self.I = np.eye(3)

        self._build_matrices()

    def _build_matrices(self):
        dt = self.dt
        # модель перехода: h += v*dt - 0.5*b*dt^2 ; v += -b*dt ; b = b
        # (ускорение a входит как управление через B)
        self.F = np.array([
            [1.0, dt, -0.5 * dt * dt],
            [0.0, 1.0, -dt],
            [0.0, 0.0, 1.0],
        ])
        self.B = np.array([
            [0.5 * dt * dt],
            [dt],
            [0.0],
        ])
        # шум процесса: от шума ускорения (через B) + блуждание смещения.
        # Q_acc пересчитывается на каждом predict (accel_trust меняется на лету),
        # поэтому храним заготовки: B·Bᵀ и блок блуждания bias.
        self._BBt = self.B @ self.B.T
        self._Q_bias = np.zeros((3, 3))
        self._Q_bias[2, 2] = (self.sigma_bias ** 2) * dt

    def predict(self, a_world_vertical: float):
        """Шаг предсказания по ускорению (управление)."""
        u = np.array([[float(a_world_vertical)]])
        self.x = self.F @ self.x + self.B @ u
        # Q собираем здесь: недоверие акселю = (sigma_accel·accel_trust)²;
        # при замороженном bias его блуждание не растёт (b удерживается)
        q_acc = (self.sigma_accel * float(self.accel_trust)) ** 2
        Q = self._BBt * q_acc
        if not self.bias_frozen:
            Q = Q + self._Q_bias
        self.P = self.F @ self.P @ self.F.T + Q

    def update(self, h_baro: float):
        """Шаг коррекции по высоте барометра.

        Пакет 15, блок А: эффективная дисперсия R_eff = R · R_baro_mult
        (детектор качания) и робастный Хьюбер поверх: |e| > k·√S →
        R_eff ×(|e|/(k√S))² — выброс баро (аэродинамика порта) гасится,
        не таская состояние; штатные обновления не меняются вовсе."""
        z = np.array([[float(h_baro)]])
        y = z - self.H @ self.x                 # невязка
        e = float(y[0, 0])
        hph = float(self.P[0, 0])               # H·P·Hᵀ при H=[1,0,0]
        R_eff = float(self.R[0, 0]) * float(self.R_baro_mult)
        S = hph + R_eff
        # Хьюбер НЕ работает поверх множителя качания (R_baro_mult>1): на
        # качании рассогласование с баро — затяжное (дрейф интеграции акселя),
        # и робастное раздувание R запускает положительную обратную связь
        # «фильтр отстёгивается от баро» (найдено на 23-21-13: v убегал до
        # +4 м/с). Качание уже демпфировано множителем k_baro.
        if self.huber_k > 0.0 and self.R_baro_mult == 1.0:
            ks = self.huber_k * (S ** 0.5)
            if abs(e) > ks:
                self._huber_run += self.dt
                if self._huber_run >= self.huber_recover_sec > 0:
                    # клип дольше huber_recover_sec — это не выброс, а реальный
                    # уход баро (смена давления): принимаем со штатным R,
                    # фильтр «перезахватывает» барометр
                    self._huber_run = 0.0
                    self.n_huber_recover += 1
                else:
                    R_eff = R_eff * (e / ks) ** 2   # = R·(|e|/(k·√S))²
                    S = hph + R_eff
                    self.n_huber += 1
            else:
                self._huber_run = 0.0
        self.last_innov = e                     # для адаптивных R/Q (блок З.1)
        self.last_S = S                         # эффективная S (после Хьюбера)
        Rm = np.array([[R_eff]])
        K = self.P @ self.H.T / S               # усиление Калмана (S — скаляр)
        if self.bias_frozen:
            K[2, 0] = 0.0                       # ДВИЖЕНИЕ: оценку нуля b не трогаем
        self.x = self.x + K @ y
        # форма Джозефа: корректная ковариация при ЛЮБОМ K (в т.ч. с занулённой строкой)
        A = self.I - K @ self.H
        self.P = A @ self.P @ A.T + K @ Rm @ K.T

    def update_zupt(self, r_zupt: float):
        """ZUPT (пакет 15, А.1): измерение СКОРОСТИ v = 0 при строгом покое.
        H_z = [0,1,0], R = r_zupt (обычно (0.02 м/с)²), форма Джозефа.
        Вызывается только когда телефон действительно неподвижен (условие
        держит vario_app) — «прибивает» дрожание вариометра к нулю, не мешая
        высоте следовать за баро."""
        y = -float(self.x[1, 0])                # 0 − v
        S = float(self.P[1, 1]) + float(r_zupt)
        K = self.P[:, 1:2] / S                  # P·Hᵀ/S, H=[0,1,0]
        if self.bias_frozen:
            K[2, 0] = 0.0
        self.x = self.x + K * y
        Hz = np.array([[0.0, 1.0, 0.0]])
        A = self.I - K @ Hz
        self.P = A @ self.P @ A.T + (K @ K.T) * float(r_zupt)
        self.n_zupt += 1

    def step(self, a_world_vertical: float, h_baro: float, dt: float | None = None):
        """Один полный шаг: предсказание + коррекция. Возвращает (высота, вариометр).

        dt — фактический шаг времени, с (реальные записи телефона идут ~400+ Гц,
        а не 50 Гц по умолчанию). Не задан — используется прежний self.dt."""
        if dt is not None and abs(dt - self.dt) > 1e-9:
            self.dt = float(dt)
            self._build_matrices()
        self.predict(a_world_vertical)
        if self.record is not None:
            xp, Pp = self.x.copy(), self.P.copy()
        self.update(h_baro)
        if self.zupt_r is not None:             # строгий покой: v = 0 (А.1)
            self.update_zupt(self.zupt_r)
        if self.record is not None:
            self.record.append((self.dt, xp, Pp, self.x.copy(), self.P.copy()))
        return float(self.x[0, 0]), float(self.x[1, 0])

    @property
    def altitude(self) -> float:
        return float(self.x[0, 0])

    @property
    def vario(self) -> float:
        return float(self.x[1, 0])

    @property
    def accel_bias(self) -> float:
        return float(self.x[2, 0])


# ----------------------------------------------------------------------
# САМОПРОВЕРКА: запусти этот файл напрямую (python baro_inertial_vario.py)
# и увидишь график, который доказывает, что фильтр работает.
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(42)
    dt = 0.02
    T = 90.0
    n = int(T / dt)
    t = np.arange(n) * dt

    # --- СИНТЕТИЧЕСКАЯ "ПРАВДА" (что происходит на самом деле) ---
    # 0-20 c: стоим на столе (v=0)  | 20-50 c: набор +1.5 м/с (термик)
    # 50-70 c: горизонт            | 70-90 c: снижение -2.0 м/с
    v_true = np.zeros(n)
    v_true[(t >= 20) & (t < 50)] = 1.5
    v_true[(t >= 70)] = -2.0
    h_true = np.cumsum(v_true) * dt
    a_true = np.gradient(v_true, dt)  # истинное ускорение

    # --- ИМИТАЦИЯ ДАТЧИКОВ S23 ---
    accel_bias_true = 0.25        # реальное смещение аксель, м/с^2 (его и "не видно")
    a_meas = a_true + accel_bias_true + rng.normal(0, 0.30, n)   # аксель шумит + смещён
    h_baro = h_true + rng.normal(0, 0.15, n)                     # баро шумит

    # --- 1) НАИВНО: вариометр = производная баро (так делают "в лоб") ---
    vario_naive_baro = np.gradient(h_baro, dt)

    # --- 2) НАИВНО: вариометр = интеграл ускорения (уплывает!) ---
    vario_naive_accel = np.cumsum(a_meas) * dt

    # --- 3) НАШ ФИЛЬТР ---
    f = BaroInertialVario(dt=dt, sigma_accel=0.30, sigma_baro=0.15, sigma_bias=0.003)
    h_kf = np.zeros(n)
    v_kf = np.zeros(n)
    b_kf = np.zeros(n)
    for i in range(n):
        h_kf[i], v_kf[i] = f.step(a_meas[i], h_baro[i])
        b_kf[i] = f.accel_bias

    # --- МЕТРИКА: СКО ошибки вариометра, когда стоим на столе (0-20 c) ---
    mask_rest = t < 20
    rms_naive = np.sqrt(np.mean(vario_naive_baro[mask_rest] ** 2))
    rms_kf = np.sqrt(np.mean(v_kf[mask_rest] ** 2))
    print(f"СКО шума вариометра НА СТОЛЕ (0-20 c):")
    print(f"  наивный (производная баро): {rms_naive:6.3f} м/с")
    print(f"  фильтр Калмана:             {rms_kf:6.3f} м/с")
    print(f"  улучшение: x{rms_naive / max(rms_kf, 1e-9):.1f}")
    print(f"Оценённое смещение аксель к концу: {b_kf[-1]:.3f} м/с^2 (истинное {accel_bias_true})")

    # --- ГРАФИКИ (научный стиль) ---
    plt.rcParams.update({"font.size": 10, "axes.grid": True,
                         "grid.alpha": 0.3, "figure.dpi": 110})
    fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    ax[0].plot(t, h_baro, color="0.7", lw=0.8, label="Баро (сырое, шумит)")
    ax[0].plot(t, h_kf, "C0", lw=1.8, label="Фильтр Калмана")
    ax[0].plot(t, h_true, "k--", lw=1.2, label="Истинная высота")
    ax[0].set_ylabel("Высота, м")
    ax[0].set_title("Баро-инерциальный вариометр — проверка на синтетике")
    ax[0].legend(loc="upper left", framealpha=0.9)

    ax[1].plot(t, vario_naive_baro, color="0.7", lw=0.7,
               label=f"Наивно: производная баро (СКО на столе {rms_naive:.2f} м/с)")
    ax[1].plot(t, v_kf, "C3", lw=1.8,
               label=f"Фильтр Калмана (СКО на столе {rms_kf:.2f} м/с)")
    ax[1].plot(t, v_true, "k--", lw=1.2, label="Истинный вариометр")
    ax[1].set_ylabel("Вариометр, м/с")
    ax[1].set_ylim(-4, 4)
    ax[1].legend(loc="upper left", framealpha=0.9)

    ax[2].plot(t, vario_naive_accel, "C1", lw=1.2,
               label="Наивно: интеграл ускорения — УПЛЫВАЕТ")
    ax[2].plot(t, v_true, "k--", lw=1.2, label="Истинный вариометр")
    ax[2].set_ylabel("Вариометр, м/с")
    ax[2].set_xlabel("Время, с")
    ax[2].legend(loc="upper left", framealpha=0.9)

    plt.tight_layout()
    plt.savefig("vario_proof.png", dpi=130, bbox_inches="tight")
    print("\nГрафик сохранён: vario_proof.png")
