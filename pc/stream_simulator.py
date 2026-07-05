# -*- coding: utf-8 -*-
"""
stream_simulator.py
===================
СИМУЛЯТОР ТЕЛЕФОНА для Фазы 3: слушает TCP 127.0.0.1:5555 и по подключению
проигрывает session-CSV в реальном темпе (по колонке t) в формате протокола
docs/stream_protocol.md. Пульт подключается через socket://127.0.0.1:5555 —
тем же кодом, каким позже будет читать настоящий Bluetooth COM-порт.

Запуск:
    python pc\\stream_simulator.py --file data\\session_*.csv
        --delay 120   постоянная задержка канала, мс
        --jitter 30   дрожание задержки (равномерно ±jitter), мс
        --loss 2      потери пакетов, % (дырки в seq)
        --rate 100    прореживание строк до ~Гц (0 = полный темп записи)

По концу файла соединение закрывается; новый коннект — проигрывание с начала.
Ctrl+C — выход.
"""

from __future__ import annotations

import argparse
import base64
import csv
import glob
import os
import random
import socket
import threading
import time
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_FILE = os.path.join(ROOT, "data", "session_2026-07-02_19-56-52.csv")

COLS = "seq,t_send,t,ax,ay,az,gx,gy,gz,mx,my,mz,pressure,altitude"

NEED = ("t", "ax", "ay", "az", "gx", "gy", "gz",
        "mx", "my", "mz", "pressure", "altitude")


def load_rows(path: str, rate_hz: float):
    """Прочитать session-CSV и (опц.) проредить строки до ~rate_hz по колонке t.
    Возвращает (rows, has_ma): v4-файлы дают ещё и mxa,mya,mza (Android-поле) —
    симулятор шлёт их, как настоящий телефон (пакет 14, Б.2)."""
    with open(path, "r", encoding="utf-8", newline="") as fh:
        rd = csv.reader(fh)
        rows = list(rd)
    header = [c.strip() for c in rows[0]]
    ix = {}
    for c in NEED:
        if c not in header:
            raise SystemExit(f"В файле нет колонки {c!r} — нужен session-CSV")
        ix[c] = header.index(c)
    has_ma = all(c in header for c in ("mxa", "mya", "mza"))
    cols = list(NEED) + (["mxa", "mya", "mza"] if has_ma else [])
    if has_ma:
        for c in ("mxa", "mya", "mza"):
            ix[c] = header.index(c)
    out = []
    min_dt = (1.0 / rate_hz) if rate_hz and rate_hz > 0 else 0.0
    last_t = None
    for row in rows[1:]:
        if len(row) < len(header):
            continue
        try:
            vals = [float(row[ix[c]]) for c in cols]
        except ValueError:
            continue
        t = vals[0]
        if last_t is not None and min_dt > 0 and (t - last_t) < min_dt:
            continue                      # прореживание IMU-строк до ~rate Гц
        last_t = t
        out.append(vals)
    if not out:
        raise SystemExit("В файле нет пригодных строк")
    return out, has_ma


def _safe_name(name: str) -> str | None:
    """Только имя файла, без путей/подъёмов — иначе None."""
    name = name.strip()
    if not name or "/" in name or "\\" in name or ".." in name:
        return None
    return name


def command_responder(conn: socket.socket, send_lock: threading.Lock,
                      clock_offset: float, stop: threading.Event,
                      files_dir: str, pause: threading.Event,
                      get_kbps: float = 0.0):
    """Читает входящие строки от ПК: PING → PONG немедленно; LIST/GET/DEL —
    менеджер записей (v3). На время GET ставит pause (главный цикл данных ждёт).
    get_kbps > 0 — дросселировать передачу файла (эмуляция скорости Bluetooth,
    чтобы пауза потока длилась как в жизни). Неизвестное — игнор."""
    buf = b""

    def send(s: str):
        with send_lock:
            conn.sendall(s.encode("utf-8"))

    try:
        while not stop.is_set():
            chunk = conn.recv(4096)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                p = line.decode("utf-8", "ignore").strip().split(",")
                if not p:
                    continue
                if p[0] == "PING" and len(p) >= 3:
                    t_mine = time.time() + clock_offset
                    send(f"PONG,{p[1]},{p[2]},{t_mine:.6f}\n")
                elif p[0] == "LIST":
                    files = sorted(glob.glob(os.path.join(files_dir, "*.csv"))
                                   + glob.glob(os.path.join(files_dir, "*.json")))
                    send(f"FILES,{len(files)}\n")
                    for fp in files:
                        st = os.stat(fp)
                        send(f"FILE,{os.path.basename(fp)},{st.st_size},{int(st.st_mtime)}\n")
                elif p[0] == "GET" and len(p) >= 2:
                    name = _safe_name(p[1])
                    fp = os.path.join(files_dir, name) if name else None
                    if not fp or not os.path.isfile(fp):
                        send("ERR,нет такого файла\n")
                        continue
                    pause.set()                    # данные молчат на время передачи
                    try:
                        with open(fp, "rb") as fh:
                            data = fh.read()
                        send(f"FILESTART,{name},{len(data)}\n")
                        for i in range(0, len(data), 3072):
                            b64 = base64.b64encode(data[i:i + 3072]).decode("ascii")
                            send(f"B64,{b64}\n")
                            if get_kbps > 0:
                                time.sleep(3072.0 / (get_kbps * 1024.0))
                        send(f"FILEEND,{zlib.crc32(data) & 0xFFFFFFFF}\n")
                    finally:
                        pause.clear()
                elif p[0] == "DEL" and len(p) >= 2:
                    name = _safe_name(p[1])
                    fp = os.path.join(files_dir, name) if name else None
                    if not fp or not os.path.isfile(fp):
                        send("ERR,нет такого файла\n")
                    else:
                        try:
                            os.remove(fp)
                            send("OK,DEL\n")
                        except OSError as e:
                            send(f"ERR,{e}\n")
                # другие типы строк молча игнорируем (правило протокола)
    except (ConnectionError, OSError):
        pass


def serve_client(conn: socket.socket, rows, delay_s: float, jitter_s: float,
                 loss_frac: float, nominal_hz: float, clock_offset: float,
                 files_dir: str, get_kbps: float = 0.0):
    """Проиграть файл одному клиенту в реальном темпе по t.

    Задержка канала — КОНВЕЙЕРНАЯ (как в жизни): каждый пакет доезжает на
    delay±jitter позже момента отправки, но темп потока сохраняется (пакеты
    «в пути» параллельно). t_send — часы отправителя в момент отправки
    (сдвинуты на clock_offset — эмуляция чужих часов). Раз в ~1 с шлётся GPS.
    На время передачи файла (GET) поток данных ПРИОСТАНАВЛИВАЕТСЯ (pause).
    """
    send_lock = threading.Lock()
    stop = threading.Event()
    pause = threading.Event()
    # v4, если файл несёт Android-поле (пакет 14, Б.2) — как настоящий телефон
    has_ma = rows and len(rows[0]) >= 15
    if has_ma:
        hello = (f"HELLO variopro-stream 4 rate={nominal_hz:.0f} "
                 f"model=simulator cols={COLS},mxa,mya,mza\n")
    else:
        hello = (f"HELLO variopro-stream 3 rate={nominal_hz:.0f} cols={COLS}\n")
    # SENSORS после HELLO (пакет 15, З.1) — эмуляция датчиков S23, чтобы
    # сквозная проверка приоров работала и без телефона
    hello += (
        "SENSORS,acc,LSM6DSO Accelerometer,STMicroelectronics,0.0024,78.4532,2404\n"
        "SENSORS,gyro,LSM6DSO Gyroscope,STMicroelectronics,0.000122,34.9066,2404\n"
        "SENSORS,mag,AK09918 Magnetometer Uncalibrated,AKM,0.0625,4912.0,10000\n"
        "SENSORS,maga,AK09918 Magnetometer,AKM,0.0625,4912.0,10000\n"
        "SENSORS,baro,LPS22HH Barometer,STMicroelectronics,0.0018,1260.0,40000\n"
        "SENSORS,temp,-,-,-,-,-\n")
    with send_lock:
        conn.sendall(hello.encode("utf-8"))
    # HELLO повторяется раз в ~5 с: pyserial при open() чистит входной буфер,
    # и первый HELLO может погибнуть в этой гонке (медленное подключение) —
    # без номинала пульт не может посчитать «Качество связи». Повтор безопасен:
    # протокол разрешает игнорировать/переобрабатывать любые строки.
    last_hello = time.perf_counter()
    # параллельный ответчик: PING/PONG + менеджер записей LIST/GET/DEL
    pr = threading.Thread(target=command_responder,
                          args=(conn, send_lock, clock_offset, stop, files_dir,
                                pause, get_kbps),
                          daemon=True)
    pr.start()
    t0 = rows[0][0]
    start = time.perf_counter()           # темп по монотонным часам
    wall0 = time.time() + clock_offset    # часы отправителя на старте (со сдвигом)
    sent = dropped = 0
    last_gps_t = -10.0
    pause_skipping = False
    try:
        for seq, vals in enumerate(rows):
            # пауза на время передачи файла (GET) — КАК НА ТЕЛЕФОНЕ: сэмплы за
            # время паузы ПРОПУСКАЮТСЯ (дыра по t, seq у телефона не тратится —
            # здесь seq тратится, что эквивалентно потерям; StreamingService
            # в onSensorChanged делает `if (sendingFile) return`)
            if pause.is_set():
                while pause.is_set() and not stop.is_set():
                    time.sleep(0.05)
                pause_skipping = True
            t_rel = vals[0] - t0
            t_send = wall0 + t_rel        # «телефон отправил» строго в свой темп
            d = max(0.0, delay_s + random.uniform(-jitter_s, jitter_s))
            write_due = start + t_rel + d # момент «доезда» до приёмника
            now = time.perf_counter()
            if pause_skipping:
                if write_due < now:
                    continue              # сэмпл «прошёл» во время паузы — дыра в t
                pause_skipping = False
            if write_due > now:
                time.sleep(write_due - now)
            if time.perf_counter() - last_hello >= 5.0:   # повтор HELLO (см. выше)
                last_hello = time.perf_counter()
                with send_lock:
                    conn.sendall(hello.encode("utf-8"))
            # GPS раз в ~1 с (ПРИМЕРНЫЕ координаты — центр Санкт-Петербурга;
            # реальное место задаётся пользователем, здесь только демо-поток)
            if t_rel - last_gps_t >= 1.0:
                last_gps_t = t_rel
                gps = (f"GPS,{vals[0]:.4f},{59.9386 + 1e-5 * t_rel:.6f},"
                       f"{30.3141:.6f},{vals[11]:.1f},{3.5:.1f}\n")
                with send_lock:
                    conn.sendall(gps.encode("utf-8"))
            if random.random() < loss_frac:   # потеря: seq расходуется, пакет не уходит
                dropped += 1
                continue
            line = (f"{seq},{t_send:.6f},{vals[0]:.4f},"
                    f"{vals[1]:.6f},{vals[2]:.6f},{vals[3]:.6f},"
                    f"{vals[4]:.6f},{vals[5]:.6f},{vals[6]:.6f},"
                    f"{vals[7]:.4f},{vals[8]:.4f},{vals[9]:.4f},"
                    f"{vals[10]:.4f},{vals[11]:.4f}")
            if len(vals) >= 15:           # v4: Android-поле mxa,mya,mza из файла
                line += f",{vals[12]:.4f},{vals[13]:.4f},{vals[14]:.4f}"
            line += "\n"
            with send_lock:
                conn.sendall(line.encode("utf-8"))
            sent += 1
    finally:
        stop.set()
    print(f"  файл проигран: отправлено {sent}, «потеряно» {dropped}")


def main():
    ap = argparse.ArgumentParser(description="Симулятор телефона (живой поток, Фаза 3)")
    ap.add_argument("--file", default=DEFAULT_FILE, help="session-CSV для проигрывания")
    ap.add_argument("--delay", type=float, default=0.0, help="задержка канала, мс")
    ap.add_argument("--jitter", type=float, default=0.0, help="дрожание задержки ±мс")
    ap.add_argument("--loss", type=float, default=0.0, help="потери пакетов, %%")
    ap.add_argument("--rate", type=float, default=100.0,
                    help="прореживание до ~Гц (0 = полный темп записи)")
    ap.add_argument("--clock-offset", type=float, default=0.0,
                    help="искусственный сдвиг часов отправителя, с (проверка NTP-поправки)")
    ap.add_argument("--get-kbps", type=float, default=0.0,
                    help="скорость передачи файла GET, КБ/с (0 = мгновенно; "
                         "~25 = как Bluetooth, пауза потока как в жизни)")
    ap.add_argument("--files-dir", default=os.path.join(ROOT, "data"),
                    help="папка «записей телефона» для LIST/GET/DEL (по умолчанию data\\)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5555)
    args = ap.parse_args()

    rows, has_ma = load_rows(args.file, args.rate)
    dur = rows[-1][0] - rows[0][0]
    nominal_hz = len(rows) / max(dur, 1e-9)   # НОМИНАЛ потока — объявляется в HELLO
    print(f"Файл: {os.path.basename(args.file)}  строк {len(rows)} "
          f"(номинал ~{nominal_hz:.0f} Гц), длительность {dur:.1f} с"
          + (", ОБА маг-поля (v4)" if has_ma else " (v3, без mxa)"))
    print(f"Канал: задержка {args.delay:.0f}±{args.jitter:.0f} мс, потери {args.loss:.1f}%, "
          f"сдвиг часов {args.clock_offset:+.2f} с")

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(1)
    print(f"Слушаю {args.host}:{args.port} — в пульте источник "
          f"«Поток», URL socket://{args.host}:{args.port}. Ctrl+C — выход.")
    try:
        while True:
            conn, addr = srv.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print(f"Клиент подключился: {addr}")
            try:
                serve_client(conn, rows, args.delay / 1000.0,
                             args.jitter / 1000.0, args.loss / 100.0,
                             nominal_hz, args.clock_offset, args.files_dir,
                             args.get_kbps)
            except (ConnectionError, OSError):
                print("  клиент отключился")
            finally:
                try:
                    conn.close()
                except OSError:
                    pass
            print("Жду следующее подключение…")
    except KeyboardInterrupt:
        print("\nВыход.")
    finally:
        srv.close()


if __name__ == "__main__":
    main()
