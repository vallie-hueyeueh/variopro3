# -*- coding: utf-8 -*-
"""Блок А: живой тракт конец-в-конец. Подключается к симулятору телефона
(StreamSource — тот же код, что в пульте), гонит замеры через реплику
пайплайна и по пути дёргает GET (пауза потока = дыра в t, как на телефоне).

Запустить симулятор:  python pc\stream_simulator.py --file data\session_2026-07-04_00-38-35.csv --port 5560
Затем:                python scripts\pack13_a_live.py --url socket://127.0.0.1:5560 --stop-after 130 --get-at 78 --get-name session_2026-07-03_16-12-04.csv
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pack13_common import load_cfg, PipelineReplica, baro_speed, episodes  # noqa: E402
from sensor_source import StreamSource                                     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="socket://127.0.0.1:5560")
    ap.add_argument("--stop-after", type=float, default=130.0, help="стоп по t данных")
    ap.add_argument("--get-at", type=float, default=-1.0, help="дёрнуть GET на этом t")
    ap.add_argument("--get-name", default="session_2026-07-03_16-12-04.csv")
    ap.add_argument("--mode", default="mekf")
    ap.add_argument("--max-wall", type=float, default=400.0)
    a = ap.parse_args()

    cfg = load_cfg()
    p = PipelineReplica(a.mode, cfg)
    src = StreamSource(a.url)
    src.open()
    T, VF, HH, RST, AA = [], [], [], [], []
    got_fired = False
    t_last_data = None
    wall0 = time.time()
    dl_note = ""
    while time.time() - wall0 < a.max_wall:
        s = src.read_sample()
        if s is None:
            if src.download_result is not None and not dl_note:
                r = src.download_result
                dl_note = (f"GET завершён: {r.get('name')} "
                           f"{len(r.get('data', b''))} Б, crc_ok={r.get('crc_ok')}"
                           if "error" not in r else f"GET ошибка: {r['error']}")
                print(f"  [live] {dl_note}")
            continue
        r = p.step(s.t, s.accel3, s.gyro3, s.h_baro)
        t_last_data = s.t
        if not got_fired and a.get_at > 0 and s.t >= a.get_at:
            got_fired = True
            ok = src.request_get(a.get_name)
            print(f"  [live] t={s.t:.1f} с: GET,{a.get_name} → {'отправлен' if ok else 'нет связи'}")
        if r is None:
            continue
        T.append(r["t"]); VF.append(r["vf"]); HH.append(r["h"])
        RST.append(1 if r["rest"] else 0); AA.append(r["a"])
        if r["t"] >= a.stop_after:
            break
    src.close()
    T, VF, HH, RST, AA = map(np.asarray, (T, VF, HH, RST, AA))
    print(f"\nживой тракт ({a.mode}): принято {len(T)} замеров, t до {T[-1]:.1f} с, "
          f"потери {src.lost}, дыры dt>0.5с: "
          f"{int((np.diff(T) > 0.5).sum())} (макс {np.diff(T).max():.2f} с)")
    vb = baro_speed(T, HH)
    rest = (RST == 1) & (np.abs(np.nan_to_num(vb, nan=9)) < 0.10)
    bad = rest & (np.abs(VF) > 1.0)
    print(f"эпизоды |vario|>1 в Покое: {episodes(T, bad)}")
    print(f"vario в Покое: mean {VF[rest].mean():+.3f}, min {VF[rest].min():+.3f}, "
          f"max {VF[rest].max():+.3f}; a_vert {AA[rest].mean():+.3f}±{AA[rest].std():.3f}")
    if p.mekf is not None:
        mk = p.mekf
        print(f"MEKF: упд {mk.n_upd}, χ² {mk.n_rej_chi2}, ‖f‖ {mk.n_rej_gate}, "
              f"восстановлений {mk.n_recover}, дыр {mk.n_hole_infl}")
    print(f"watchdog: {p.wd_events if p.wd_events else 'не срабатывал'}")


if __name__ == "__main__":
    main()
