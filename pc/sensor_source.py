# -*- coding: utf-8 -*-
"""
sensor_source.py
================
АБСТРАКЦИЯ ИСТОЧНИКА ДАННЫХ для вариометра.

ЗАЧЕМ ЭТО НУЖНО (простыми словами)
----------------------------------
Программе-вариометру всё равно, ОТКУДА пришли числа: из симуляции, из
записанного файла или (в будущем) с телефона по Bluetooth. Важно лишь, чтобы
любой источник выдавал ОДИНАКОВЫЙ "кубик данных" — один замер (Sample):

    Sample = (время_сек, вертикальное_ускорение, высота_по_барометру)

Эти три числа — ровно то, что нужно фильтру Калмана (ядру вариометра).
Поэтому окно программы работает с любым источником одинаково, а добавить
новый источник = написать ещё один маленький класс. Ничего другого менять
не придётся.

ТРИ ИСТОЧНИКА
-------------
  • SimSource       — синтетика (та же сцена, что в самопроверке фильтра):
                      стол → термик → горизонт → снижение.
  • CsvSource       — читает запись из файла data/*.csv.
  • BluetoothSource — ЗАДЕЛ НА ФАЗУ 2 (пока не используется): классический
                      Bluetooth SPP, телефон-сервер, ПК читает поток через
                      pyserial по COM-порту. Формат строки тот же, что у CSV,
                      поэтому код разбора строки общий.

ЖИЗНЕННЫЙ ЦИКЛ любого источника:
    src.open()                  # подготовиться (открыть файл / порт / сгенерировать)
    while True:
        s = src.read_sample()   # взять следующий замер (может блокировать)
        if s is None:           # None = данные кончились
            break
        ... используем s ...
    src.close()                 # закрыть/освободить ресурсы
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
# ПРОГРЕВ numpy.random на импорте модуля: в numpy 2.x он ленивый, и первый
# np.random.default_rng() из ПОТОКА ЧТЕНИЯ (SimSource.open в SourceWorker)
# запускает импорт-каскад под хуком PySide (shibokensupport) — тот читает
# исходники через inspect на каждый модуль, блокируя поток на секунды.
import numpy.random  # noqa: F401  (side effect: импорт в главном потоке)


# ----------------------------------------------------------------------
# ОДИН ЗАМЕР ДАТЧИКОВ
# ----------------------------------------------------------------------
@dataclass
class Sample:
    """Один "кубик" данных — то, что выдаёт любой источник за один шаг."""
    t: float                  # время в секундах (с начала записи/запуска)
    a_world_vertical: float   # вертикальное ускорение в мировой СК, без g, м/с^2
    h_baro: float             # высота по барометру, м
    accel3: tuple | None = None  # сырое ускорение (ax,ay,az) — для компаса, если есть
    mag3: tuple | None = None    # магнитное поле (mx,my,mz) — для компаса, если есть
    mag_raw: bool = False        # True = mag3 СЫРОЕ (uncalib) → к курсу применить калибровку прибора
    gyro3: tuple | None = None   # угловая скорость (gx,gy,gz), рад/с — для детектора движения
    imu_raw: bool = False        # True = a_world_vertical НЕ задан источником: приёмник сам
                                 # получает вертикальное ускорение из accel3 (сырой IMU телефона)
    mag3a: tuple | None = None   # Android-КАЛИБРОВАННОЕ поле (mxa,mya,mza), протокол v4;
                                 # None в старых файлах/потоках v3 — компас падает на mag3


# ----------------------------------------------------------------------
# ПЕЙСЕР РЕАЛЬНОГО ВРЕМЕНИ
# ----------------------------------------------------------------------
class _RealtimePacer:
    """
    Помощник, который "притормаживает" выдачу замеров, чтобы данные шли как в
    реальном полёте (а не мгновенно за доли секунды).

    Идея: запоминаем момент старта по настенным часам и время первого замера.
    Дальше для каждого замера ждём ровно столько, чтобы расстояние между
    замерами по настенным часам совпало с расстоянием по времени данных
    (с учётом множителя скорости speed).

    Если программа где-то задержалась и мы "отстали" — НЕ спим (догоняем).
    Именно поэтому при больших speed выдача идёт пачками, но в среднем
    скорость соблюдается даже на Windows с грубым таймером сна.
    """

    def __init__(self, speed: float = 1.0):
        self.speed = max(float(speed), 1e-6)  # во сколько раз быстрее реального времени
        self._t0_wall: Optional[float] = None  # момент старта по настенным часам
        self._t0_sample: Optional[float] = None  # время первого замера

    def reset(self) -> None:
        """Сбросить отсчёт (вызывается при каждом новом open)."""
        self._t0_wall = None
        self._t0_sample = None

    def wait_for(self, t_sample: float) -> None:
        """Подождать до "положенного" момента выдачи замера со временем t_sample."""
        now = time.perf_counter()
        if self._t0_wall is None:
            # первый замер: просто запоминаем точку отсчёта и не ждём
            self._t0_wall = now
            self._t0_sample = t_sample
            return
        # к какому моменту настенных часов "положено" отдать этот замер
        target = self._t0_wall + (t_sample - self._t0_sample) / self.speed
        delay = target - now
        # спим маленькими порциями, чтобы программу можно было быстро остановить
        while delay > 0:
            time.sleep(min(delay, 0.05))
            delay = target - time.perf_counter()


# ----------------------------------------------------------------------
# БАЗОВЫЙ КЛАСС ИСТОЧНИКА (общий интерфейс для всех)
# ----------------------------------------------------------------------
class SensorSource(ABC):
    """Базовый класс. Любой источник обязан уметь open / read_sample / close."""

    name: str = "источник"  # человекочитаемое имя (показывается в окне)
    live: bool = False      # True = живой поток: None из read_sample означает
                            # «данных сейчас нет» (не конец) — читатель продолжает

    @abstractmethod
    def open(self) -> None:
        """Подготовить источник к чтению."""

    @abstractmethod
    def read_sample(self) -> Optional[Sample]:
        """Вернуть следующий замер или None, если данные закончились."""

    @abstractmethod
    def close(self) -> None:
        """Освободить ресурсы (закрыть файл/порт)."""

    def __iter__(self):
        """Удобство: по источнику можно пройтись циклом for (в скриптах, не в окне)."""
        self.open()
        try:
            while True:
                s = self.read_sample()
                if s is None:
                    break
                yield s
        finally:
            self.close()


# ----------------------------------------------------------------------
# СИНТЕТИЧЕСКАЯ СЦЕНА (та же, что в самопроверке фильтра)
# ----------------------------------------------------------------------
def generate_synthetic(dt: float = 0.02, T: float = 90.0, seed: int = 42):
    """
    Сгенерировать синтетический "полёт" — точь-в-точь как в самопроверке
    baro_inertial_vario.py:
        0-20 c  — лежим на столе (скорость 0)
        20-50 c — набор высоты +1.5 м/с (термик)
        50-70 c — горизонтальный полёт
        70-90 c — снижение -2.0 м/с

    Возвращает массивы: (t, a_meas, h_baro, v_true, h_true)
      a_meas — ускорение датчика: истинное + смещение нуля + шум
      h_baro — высота баро: истинная + шум
      v_true, h_true — "правда" (для сравнения/отладки, в полёте её не знаем)
    """
    rng = np.random.default_rng(seed)
    n = int(T / dt)
    t = np.arange(n) * dt

    # истинная вертикальная скорость по сценарию
    v_true = np.zeros(n)
    v_true[(t >= 20) & (t < 50)] = 1.5
    v_true[(t >= 70)] = -2.0
    h_true = np.cumsum(v_true) * dt          # истинная высота = интеграл скорости
    a_true = np.gradient(v_true, dt)         # истинное ускорение = производная скорости

    # имитация реальных датчиков Samsung S23
    accel_bias_true = 0.25                    # реальное смещение нуля акселерометра, м/с^2
    a_meas = a_true + accel_bias_true + rng.normal(0, 0.30, n)  # аксель: смещён и шумит
    h_baro = h_true + rng.normal(0, 0.15, n)                    # баро: шумит, но не уплывает

    return t, a_meas, h_baro, v_true, h_true


def synth_compass(t: float):
    """
    Синтетика для КОМПАСА в ОСЯХ ANDROID (x — вправо, y — к верхнему краю экрана,
    z — из экрана; в покое az = +g). Телефон медленно вращается по курсу
    (yaw ~12°/с) с постоянным наклоном — видно работу наклон-компенсации.
    Возвращает (accel3, mag3) в системе телефона — как реальные датчики.
    """
    yaw = np.radians((t * 12.0) % 360.0)   # курс (по часовой от севера)
    pitch = np.radians(18.0)               # наклон вокруг оси x телефона
    roll = np.radians(8.0)                 # наклон вокруг оси y телефона
    g = 9.81
    H, Z = 24.0, 42.0                      # поле: горизонталь на север + ВНИЗ, мкТл (demo)
    # матрица телефон→мир (базис E,N,U): yaw вокруг вертикали, затем наклоны
    cy, sy = np.cos(-yaw), np.sin(-yaw)    # по часовой (вид сверху) = −yaw в математике
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    Ry = np.array([[cr, 0, sr], [0, 1, 0], [-sr, 0, cr]])
    W = Rz @ Rx @ Ry                       # телефон → мир
    a = W.T @ np.array([0.0, 0.0, g])      # реакция опоры: вверх (U)
    m = W.T @ np.array([0.0, H, -Z])       # поле: на север + вниз (сев. полушарие)
    return (float(a[0]), float(a[1]), float(a[2])), (float(m[0]), float(m[1]), float(m[2]))


# ----------------------------------------------------------------------
# ИСТОЧНИК 1: СИМУЛЯЦИЯ
# ----------------------------------------------------------------------
class SimSource(SensorSource):
    """
    Синтетический источник. Генерирует сцену из generate_synthetic и отдаёт
    её замер за замером в реальном времени (через пейсер).

    loop=True  — по окончании сцены начинать заново (бесконечный живой поток),
                 время продолжает расти монотонно.
    speed      — множитель скорости (1.0 = реальное время, 20.0 = в 20 раз быстрее).
    """

    name = "Симуляция"

    def __init__(self, dt: float = 0.02, T: float = 90.0, seed: int = 42,
                 speed: float = 1.0, loop: bool = True):
        self.dt = float(dt)
        self.T = float(T)
        self.seed = int(seed)
        self.loop = bool(loop)
        self.pacer = _RealtimePacer(speed)
        self._t = self._a = self._h = None
        self._i = 0
        self._n = 0
        self._loops = 0

    def open(self) -> None:
        # генерируем всю сцену заранее (она маленькая)
        self._t, self._a, self._h, _, _ = generate_synthetic(self.dt, self.T, self.seed)
        self._n = len(self._t)
        self._i = 0
        self._loops = 0
        self.pacer.reset()

    def read_sample(self) -> Optional[Sample]:
        if self._i >= self._n:
            if not self.loop:
                return None            # сцена кончилась и зацикливание выключено
            self._i = 0                # начать сцену заново
            self._loops += 1
        idx = self._i
        # время растёт монотонно даже при зацикливании (чтобы график не "прыгал" назад)
        t = self._loops * self.T + float(self._t[idx])
        self.pacer.wait_for(t)         # выдержать темп реального времени
        acc3, mg3 = synth_compass(t)   # синтетика для компаса (курс крутится)
        s = Sample(t, float(self._a[idx]), float(self._h[idx]), accel3=acc3, mag3=mg3)
        self._i += 1
        return s

    def close(self) -> None:
        pass  # ресурсов нет — закрывать нечего


# ----------------------------------------------------------------------
# РАЗБОР СТРОКИ ДАННЫХ (общий для CSV и Bluetooth)
# ----------------------------------------------------------------------
def parse_sample_line(line: str, sep: str = ",") -> Optional[Sample]:
    """
    Превратить текстовую строку 't,a_world_vertical,h_baro' в Sample.

    Возвращает None, если строку нельзя разобрать (например, это строка-заголовок
    't,a_world_vertical,h_baro' или пустая строка). Так один и тот же код подходит
    и для чтения файла, и для приёма строк по Bluetooth.
    """
    parts = line.strip().split(sep)
    if len(parts) < 3:
        return None
    try:
        return Sample(float(parts[0]), float(parts[1]), float(parts[2]))
    except ValueError:
        return None  # не числа (скорее всего, заголовок) — просто пропускаем


# ----------------------------------------------------------------------
# ИСТОЧНИК 2: CSV-ФАЙЛ
# ----------------------------------------------------------------------
class CsvSource(SensorSource):
    """
    Читает запись из CSV-файла со столбцами: t, a_world_vertical, h_baro.
    Строка-заголовок (если есть) распознаётся и пропускается автоматически.

    realtime=True — воспроизводить запись в темпе реального времени (как полёт);
                    иначе замеры отдаются максимально быстро.
    speed         — множитель скорости при realtime.
    """

    name = "CSV-файл"

    def __init__(self, path: str, speed: float = 1.0, realtime: bool = True):
        self.path = str(path)
        self.realtime = bool(realtime)
        self.pacer = _RealtimePacer(speed)
        self._rows: list[Sample] = []
        self._i = 0

    def open(self) -> None:
        # файлы Фазы 0 небольшие — читаем сразу целиком в память
        self._rows = []
        import csv as _csv
        with open(self.path, "r", encoding="utf-8", newline="") as fh:
            rows = list(_csv.reader(fh))
        if not rows:
            raise ValueError(f"В файле нет данных: {self.path}")
        header = [c.strip() for c in rows[0]]
        is_session = all(c in header for c in ("t", "ax", "ay", "az", "mx", "my", "mz"))
        if is_session:
            # SESSION CSV телефона: сырой IMU. accel+mag — компасу и детектору движения,
            # gyro — детектору движения; вертикальное ускорение для фильтра приёмник
            # (vario_app) вычислит сам из accel3 (imu_raw=True). Высота — из altitude.
            ix = {name: header.index(name) for name in header}
            has_alt = "altitude" in ix
            # v4: Android-калиброванное поле в ДОПОЛНИТЕЛЬНЫХ колонках mxa,mya,mza
            # (mx,my,mz остаются сырыми); старые файлы без них читаются как раньше
            has_ma = all(c in ix for c in ("mxa", "mya", "mza"))

            def num(row, name):
                try:
                    return float(row[ix[name]])
                except (ValueError, IndexError, KeyError):
                    return 0.0
            for row in rows[1:]:
                if len(row) < len(header):
                    continue
                a3 = (num(row, "ax"), num(row, "ay"), num(row, "az"))
                m3 = (num(row, "mx"), num(row, "my"), num(row, "mz"))
                g3 = (num(row, "gx"), num(row, "gy"), num(row, "gz"))
                ma3 = ((num(row, "mxa"), num(row, "mya"), num(row, "mza"))
                       if has_ma else None)
                h = num(row, "altitude") if has_alt else 0.0
                self._rows.append(Sample(num(row, "t"), 0.0, h,
                                         accel3=a3, mag3=m3, mag_raw=True,
                                         gyro3=g3, imu_raw=True, mag3a=ma3))
        else:
            # обычный 3-колоночный CSV вариометра: t, a_world_vertical, h_baro
            for row in rows:
                s = parse_sample_line(",".join(row))
                if s is not None:
                    self._rows.append(s)
        if not self._rows:
            raise ValueError(f"В файле нет данных: {self.path}")
        # сдвигаем время так, чтобы запись начиналась с нуля (для графика по оси X)
        t0 = self._rows[0].t
        for s in self._rows:
            s.t -= t0
        self._i = 0
        self.pacer.reset()

    def read_sample(self) -> Optional[Sample]:
        if self._i >= len(self._rows):
            return None
        s = self._rows[self._i]
        self._i += 1
        if self.realtime:
            self.pacer.wait_for(s.t)
        # КОМПАС: CSV вариометра содержит только (t, a_world_vertical, h_baro) —
        # колонок компаса (accel+mag) в нём нет. Чтобы стрелка компаса работала так
        # же, как в «Симуляции», даём ей синтетику synth_compass(t) — тот же источник,
        # что использует SimSource. Если в строке уже есть реальные accel/mag — не трогаем.
        if s.accel3 is None or s.mag3 is None:
            s.accel3, s.mag3 = synth_compass(s.t)
        return s

    def close(self) -> None:
        self._rows = []


# ----------------------------------------------------------------------
# ИСТОЧНИК 3: BLUETOOTH SPP  —  ЗАДЕЛ НА ФАЗУ 2 (пока не используется)
# ----------------------------------------------------------------------
class BluetoothSource(SensorSource):
    """
    ФАЗА 2. Классический Bluetooth SPP (Serial Port Profile):
        • телефон = сервер (отдаёт поток данных),
        • ПК = клиент: после сопряжения в Windows появляется виртуальный
          COM-порт (например, COM5), и мы читаем его через pyserial.

    Телефон шлёт строки в том же формате, что и CSV:
        't,a_world_vertical,h_baro\\n'
    Поэтому разбор строки — общий (parse_sample_line). Темп задаёт сам телефон,
    данные приходят в реальном времени, поэтому пейсер тут НЕ нужен.

    Этот класс уже написан, но в Фазе 0 не вызывается — он показывает, что
    архитектура к Bluetooth готова: добавить его в окно = одна строчка.
    """

    name = "Bluetooth (SPP)"

    def __init__(self, port: str, baudrate: int = 115200, timeout: float = 1.0):
        self.port = str(port)         # имя COM-порта, например "COM5"
        self.baudrate = int(baudrate) # скорость (для SPP число условное, но нужно задать)
        self.timeout = float(timeout) # таймаут чтения строки, сек
        self._ser = None

    def open(self) -> None:
        import serial  # импорт внутри метода: зависимость не мешает Фазе 0
        self._ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)

    def read_sample(self) -> Optional[Sample]:
        # readline() блокирует до прихода строки или до таймаута
        raw = self._ser.readline()
        if not raw:
            return None  # таймаут / соединение закрылось
        line = raw.decode("utf-8", errors="ignore")
        return parse_sample_line(line)

    def close(self) -> None:
        if self._ser is not None:
            self._ser.close()
            self._ser = None


# ----------------------------------------------------------------------
# ИСТОЧНИК 4: ЖИВОЙ ПОТОК (Фаза 3) — симулятор по TCP или Bluetooth COM-порт
# ----------------------------------------------------------------------
class StreamSource(SensorSource):
    """
    Живой поток по протоколу docs/stream_protocol.md через pyserial serial_for_url:
        симулятор            → url = "socket://127.0.0.1:5555"
        настоящий Bluetooth  → url = "COM5"           (тот же код!)

    Особенности:
      • читается в отдельном потоке (SourceWorker), readline с таймаутом 1 с —
        окно не блокируется;
      • битые/рваные строки пропускаются;
      • обрыв связи НЕ роняет пульт: источник сам переподключается (status
        показывает «нет связи (переподключение…)»);
      • собирает метрики связи для вкладки «Связь/Задержка»: события приёма
        (время приёма, t_send, seq), счётчики принятых/потерянных (по дыркам seq).
    """

    name = "Поток (Bluetooth/симулятор)"
    live = True

    def __init__(self, url: str, baudrate: int = 115200):
        self.url = str(url)
        self.baudrate = int(baudrate)     # для COM-порта; socket:// его игнорирует
        self._ser = None
        self._closed = False
        self._silent = 0                  # подряд пустых чтений (детект «тихого» обрыва)
        self._rxbuf = bytearray()         # приёмный буфер (строки режем сами — быстрее readline)
        self._t0 = None                   # для сдвига времени данных к нулю
        self._last_raw_t = None           # последний t из потока (детект рестарта)
        self._last_out_t = 0.0            # последний выданный t (монотонность оси X)
        # --- статус и метрики связи (читает вкладка «Связь/Задержка») ---
        self.status = "ожидание"
        self.recv_events = deque(maxlen=4000)   # (время приёма time.time(), t_send, seq)
        self.received = 0                 # принятых пакетов
        self.lost = 0                     # потерянных (по дыркам в seq)
        self.last_seq = None
        self.nominal_hz = None            # номинальная частота из HELLO (rate=…)
        self.mag_source = None            # из HELLO v4: mag_source=… (настройка телефона)
        self.model = None                 # из HELLO v4: model=… (модель устройства)
        # SENSORS (пакет 15, З.1): метаданные датчиков телефона (после HELLO)
        self.sensors = {}                 # ключ acc/gyro/mag/maga/baro/temp → dict
        self.sensors_wall = None          # когда получены (для «Связи»)
        self.temp_last = None             # TEMP (З.3): {"t","c","wall"} | None
        self.last_data_wall = None        # время последнего ПАКЕТА ДАННЫХ (индикатор тишины)
        # --- честные часы: PING/PONG (NTP-схема по тому же каналу) ---
        self.pongs = deque(maxlen=16)     # (rtt_s, offset_s) по последним обменам
        self._ping_id = 0
        self._last_ping = 0.0
        # --- GPS-строки телефона (только показ; в фильтр не идёт) ---
        self.gps_last = None              # {"t","lat","lon","alt","acc","wall"}
        # --- менеджер записей (LIST/GET/DEL, протокол v3) ---
        self.file_list = None             # [{"name","size","mtime"}] после LIST
        self.file_list_wall = None        # когда список получен
        self._files_pending = None        # сколько строк FILE ещё ждём
        self._files_acc = []
        self._dl = None                   # текущий приём: {"name","size","buf"}
        self.download_progress = None     # (получено_байт, всего, имя)
        self.download_result = None       # {"name","data",...} | {"error": str}
        self.del_result = None            # ("OK", "") / ("ERR", причина)

    # ---- подключение ----
    def _try_connect(self, initial: bool = False) -> bool:
        import serial
        if initial:
            self.status = "подключение…"
        try:
            self._ser = serial.serial_for_url(self.url, baudrate=self.baudrate,
                                              timeout=1.0)
            self._silent = 0
            self._rxbuf.clear()
            self.status = "подключено"
            return True
        except Exception:
            self._ser = None
            self.status = "нет связи (переподключение…)"
            return False

    def _drop_connection(self):
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass
        self._ser = None
        self.status = "нет связи (переподключение…)"

    def open(self) -> None:
        self._closed = False
        self._try_connect(initial=True)   # не вышло — read_sample продолжит попытки

    def _next_line(self):
        """Достать одну строку из приёмного буфера; при необходимости дочитать
        ПАЧКУ байтов. Схема «1 блокирующий байт + мгновенный остаток»: read(1)
        ждёт данные (до таймаута 1 с), затем с timeout=0 забирается ВСЁ, что уже
        пришло. Так и поток 100 Гц не ждёт, и большие передачи (GET файла) идут
        на полной скорости — in_waiting у socket-обработчика pyserial всегда «1»,
        полагаться на него нельзя. None = строки пока нет / тишина / обрыв."""
        i = self._rxbuf.find(b"\n")
        if i < 0:
            try:
                chunk = self._ser.read(1)          # ждём первый байт (timeout 1 с)
                if chunk:
                    self._ser.timeout = 0          # и мгновенно всё, что накопилось
                    try:
                        chunk += self._ser.read(65536)
                    finally:
                        self._ser.timeout = 1.0
            except Exception:
                self._drop_connection()
                return None
            if not chunk:
                # тишина: таймаут ИЛИ «тихий» обрыв (сокет закрыт). После ~3 с
                # подряд без данных считаем связь потерянной и переподключаемся.
                self._silent += 1
                if self._silent >= 3:
                    self._drop_connection()
                return None
            self._silent = 0
            self._rxbuf += chunk
            i = self._rxbuf.find(b"\n")
            if i < 0:
                return None                        # строка ещё не докатилась
        line = bytes(self._rxbuf[:i])
        del self._rxbuf[:i + 1]
        return line

    # ---- менеджер записей: команды ПК → телефон (протокол v3) ----
    def _send_cmd(self, cmd: str) -> bool:
        """Отправить команду по каналу. False, если связи нет."""
        if self._ser is None:
            return False
        try:
            self._ser.write((cmd + "\n").encode("utf-8"))
            return True
        except Exception:
            self._drop_connection()
            return False

    def request_list(self) -> bool:
        """LIST → телефон пришлёт FILES/FILE…; результат появится в self.file_list."""
        self.file_list = None
        self._files_pending = None
        self._files_acc = []
        return self._send_cmd("LIST")

    def request_get(self, name: str) -> bool:
        """GET,<имя> → приём в self.download_result (прогресс в download_progress)."""
        self._dl = None
        self.download_progress = None
        self.download_result = None
        return self._send_cmd(f"GET,{name}")

    def request_del(self, name: str) -> bool:
        """DEL,<имя> → ответ в self.del_result."""
        self.del_result = None
        return self._send_cmd(f"DEL,{name}")

    def _handle_service_line(self, p) -> bool:
        """Служебные строки менеджера/GPS/SENSORS/TEMP. True = обработана."""
        tag = p[0]
        if tag == "SENSORS" and len(p) >= 7:
            # SENSORS,<ключ>,<name>,<vendor>,<resolution>,<maxRange>,<minDelayUs>
            # (пакет 15, З.1; запятые в name/vendor телефон заменил на ';')
            self.sensors[p[1]] = {"name": p[2], "vendor": p[3],
                                  "resolution": p[4], "max_range": p[5],
                                  "min_delay_us": p[6]}
            self.sensors_wall = time.time()
            return True
        if tag == "TEMP" and len(p) >= 3:
            # TEMP,<t данных>,<°C> — раз в ~5 с (З.3); только лог/показ
            try:
                self.temp_last = {"t": float(p[1]), "c": float(p[2]),
                                  "wall": time.time()}
            except ValueError:
                pass
            return True
        if tag == "GPS" and len(p) >= 6:
            try:
                self.gps_last = {"t": float(p[1]), "lat": float(p[2]),
                                 "lon": float(p[3]), "alt": float(p[4]),
                                 "acc": float(p[5]), "wall": time.time()}
            except ValueError:
                pass
            return True
        if tag == "FILES" and len(p) >= 2:
            try:
                self._files_pending = int(p[1])
            except ValueError:
                self._files_pending = 0
            self._files_acc = []
            if self._files_pending == 0:
                self.file_list = []
                self.file_list_wall = time.time()
            return True
        if tag == "FILE" and len(p) >= 4 and self._files_pending is not None:
            try:
                self._files_acc.append({"name": p[1], "size": int(p[2]),
                                        "mtime": int(p[3])})
            except ValueError:
                pass
            if len(self._files_acc) >= self._files_pending:
                self.file_list = list(self._files_acc)
                self.file_list_wall = time.time()
                self._files_pending = None
            return True
        if tag == "FILESTART" and len(p) >= 3:
            try:
                self._dl = {"name": p[1], "size": int(p[2]), "buf": bytearray()}
                self.download_progress = (0, self._dl["size"], p[1])
            except ValueError:
                self._dl = None
            return True
        if tag == "B64" and len(p) >= 2 and self._dl is not None:
            import base64
            try:
                self._dl["buf"] += base64.b64decode(p[1])
                self.download_progress = (len(self._dl["buf"]),
                                          self._dl["size"], self._dl["name"])
            except Exception:
                pass
            return True
        if tag == "FILEEND" and self._dl is not None:
            import zlib
            data = bytes(self._dl["buf"])
            crc_ok = None
            try:
                crc_ok = (zlib.crc32(data) & 0xFFFFFFFF) == int(p[1])
            except (ValueError, IndexError):
                pass
            self.download_result = {"name": self._dl["name"], "data": data,
                                    "size_ok": len(data) == self._dl["size"],
                                    "crc_ok": bool(crc_ok)}
            self._dl = None
            self.download_progress = None
            return True
        if tag == "OK":
            self.del_result = ("OK", "")
            return True
        if tag == "ERR":
            err = ",".join(p[1:]) if len(p) > 1 else "ошибка"
            if self._dl is not None or self.download_progress is not None:
                self.download_result = {"error": err}
                self._dl = None
                self.download_progress = None
            else:
                self.del_result = ("ERR", err)
            return True
        return False

    # ---- служебное: PING раз в ~2 с (честные часы, раздел PING/PONG протокола) ----
    def _maybe_ping(self):
        now = time.time()
        if now - self._last_ping < 2.0:
            return
        self._last_ping = now
        self._ping_id += 1
        try:
            self._ser.write(f"PING,{self._ping_id},{now:.6f}\n".encode("utf-8"))
        except Exception:
            self._drop_connection()

    # ---- чтение ----
    def read_sample(self) -> Optional[Sample]:
        if self._closed:
            return None                       # остановлено пользователем — конец
        if self._ser is None:
            time.sleep(0.5)                   # пауза между попытками переподключения
            self._try_connect()
            return None                       # live: читатель просто продолжит
        self._maybe_ping()
        raw = self._next_line()
        if raw is None:
            return None
        line = raw.decode("utf-8", errors="ignore").strip()
        if not line:
            return None
        if line.startswith("HELLO"):
            # рукопожатие: номинальная частота rate=…; v4 добавляет mag_source=
            # (настройка «Источник магнитометра» телефона) и model= (Build.MODEL,
            # пробелы заменены '_'). Неизвестные токены игнорируются (правило).
            for tok in line.split():
                if tok.startswith("rate="):
                    try:
                        self.nominal_hz = float(tok[5:])
                    except ValueError:
                        pass
                elif tok.startswith("mag_source="):
                    self.mag_source = tok[11:]
                elif tok.startswith("model="):
                    self.model = tok[6:].replace("_", " ")
            return None
        p = line.split(",")
        if self._handle_service_line(p):
            return None                       # GPS/менеджер записей — не данные
        if p[0] == "PONG" and len(p) >= 4:
            # PONG,<id>,<t_pc_эхо>,<t_отправителя> → RTT и смещение часов
            try:
                t_echo = float(p[2])
                t_their = float(p[3])
            except ValueError:
                return None
            t_rx = time.time()
            rtt = t_rx - t_echo
            if 0.0 <= rtt < 10.0:
                offset = t_their - (t_echo + rtt / 2.0)
                self.pongs.append((rtt, offset))
            return None
        if len(p) < 14:
            return None                       # рваная/служебная строка — пропускаем
        try:
            seq = int(p[0])
            t_send = float(p[1])
            t = float(p[2])
            ax, ay, az = float(p[3]), float(p[4]), float(p[5])
            gx, gy, gz = float(p[6]), float(p[7]), float(p[8])
            mx, my, mz = float(p[9]), float(p[10]), float(p[11])
            alt = float(p[13])
        except ValueError:
            return None                       # битые числа — пропускаем
        ma3 = None
        if len(p) >= 17:                      # v4: Android-калиброванное поле
            try:
                ma3 = (float(p[14]), float(p[15]), float(p[16]))
            except ValueError:
                ma3 = None
        # --- метрики связи ---
        now = time.time()
        self.recv_events.append((now, t_send, seq))
        self.received += 1
        self.last_data_wall = now
        if self.last_seq is not None and seq > self.last_seq + 1:
            self.lost += seq - self.last_seq - 1
        self.last_seq = seq
        # --- монотонная ось времени (переподключение начинает файл заново) ---
        if self._t0 is None or (self._last_raw_t is not None
                                and t < self._last_raw_t - 1.0):
            self._t0 = t - (self._last_out_t + 0.02 if self._last_raw_t is not None else 0.0)
        self._last_raw_t = t
        self._last_out_t = t - self._t0
        return Sample(self._last_out_t, 0.0, alt,
                      accel3=(ax, ay, az), mag3=(mx, my, mz), mag_raw=True,
                      gyro3=(gx, gy, gz), imu_raw=True, mag3a=ma3)

    def link_metrics(self) -> dict:
        """
        ЕДИНЫЙ расчёт метрик связи (используют обе вкладки — числа одинаковые):
          fact_hz     — фактическая частота пакетов (окно 2 с);
          loss_pct    — потери в скользящем окне ~10 с (по дыркам seq);
          quality_pct — качество связи = min(1, факт/номинал)·(1−потери)·100, или None;
          rtt_ms, offset_s — медианы по PING/PONG (None, пока обменов < 4);
          delay_ms, delay_std_ms — честная задержка = приём − (t_send − offset), окно 2 с;
          silence_s   — секунд с последнего пакета данных (None = данных не было).
        """
        now = time.time()
        # копия под защитой (Д.2): recv_events наполняет ПОТОК ЧТЕНИЯ, а сюда
        # заходит GUI-поток; итерация deque во время append может кинуть
        # RuntimeError — тогда честно отдаём прошлую картину (пустую)
        try:
            events = list(self.recv_events)
        except RuntimeError:
            events = []
        win2 = [(w, ts, sq) for (w, ts, sq) in events if now - w <= 2.0]
        fact_hz = len(win2) / 2.0
        # потери в окне ~10 с: хвост ПОСЛЕДНЕЙ сессии (seq монотонен), дырки = потери
        tail = []
        prev_seq = None
        for (w, ts, sq) in reversed(events):
            if now - w > 10.0 or (prev_seq is not None and sq > prev_seq):
                break                          # вышли из окна или из текущей сессии
            tail.append(sq)
            prev_seq = sq
        loss_pct = None
        if len(tail) >= 2:
            expected = max(tail) - min(tail) + 1
            loss_pct = max(0.0, 100.0 * (1.0 - len(tail) / expected))
        # тишина и качество
        silence_s = (now - self.last_data_wall) if self.last_data_wall else None
        quality_pct = None
        if (self.nominal_hz and silence_s is not None and silence_s <= 1.5
                and fact_hz > 0):
            q = min(1.0, fact_hz / self.nominal_hz)
            if loss_pct is not None:
                q *= (1.0 - loss_pct / 100.0)
            quality_pct = max(0.0, min(100.0, q * 100.0))
        # часы: медианы по последним обменам PING/PONG
        rtt_ms = offset_s = None
        if len(self.pongs) >= 4:
            rr = sorted(r for (r, _) in self.pongs)
            oo = sorted(o for (_, o) in self.pongs)
            rtt_ms = rr[len(rr) // 2] * 1000.0
            offset_s = oo[len(oo) // 2]
        # честная задержка по окну 2 с (нужен offset)
        delay_ms = delay_std_ms = None
        if offset_s is not None and win2:
            ds = [(w - (ts - offset_s)) * 1000.0 for (w, ts, _) in win2]
            m = sum(ds) / len(ds)
            delay_ms = m
            delay_std_ms = (sum((x - m) ** 2 for x in ds) / len(ds)) ** 0.5
        return {"fact_hz": fact_hz, "loss_pct": loss_pct, "quality_pct": quality_pct,
                "rtt_ms": rtt_ms, "offset_s": offset_s,
                "delay_ms": delay_ms, "delay_std_ms": delay_std_ms,
                "silence_s": silence_s}

    def close(self) -> None:
        self._closed = True
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass
            self._ser = None
        self.status = "остановлено"


# ----------------------------------------------------------------------
# Маленькая самопроверка модуля: python sensor_source.py
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # быстро прогоняем симуляцию на максимальной скорости и печатаем первые замеры
    src = SimSource(speed=1000.0, loop=False)
    src.open()
    print("Первые 3 замера симуляции (t, a, h_baro):")
    for _ in range(3):
        s = src.read_sample()
        print(f"  t={s.t:5.2f} c   a={s.a_world_vertical:+.3f} м/с^2   h={s.h_baro:+.3f} м")
    src.close()
    print("OK: источник данных работает.")
