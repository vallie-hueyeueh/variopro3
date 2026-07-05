# -*- coding: utf-8 -*-
"""Блок А.4: сработал бы watchdog ДО фикса MEKF? Реконструируем до-фиксный MEKF
(без проекции инновации, без восстановления, без раздувания на дырах) прямо из
исходника pc/mekf.py и гоняем сценарий дыры 75–95 с. Затем — проверка ложных
срабатываний watchdog ПОСЛЕ фикса на чистых файлах и на дырах.

python scripts\pack13_a_watchdog.py
"""
import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pack13_common import ROOT, load_cfg, read_session, PipelineReplica  # noqa: E402
import mekf as mekf_new                                                  # noqa: E402


def build_prefix_mekf_module():
    """Собрать модуль «MEKF до фикса»: вырезать проекцию инновации на касательную
    плоскость (строки cy…) — восстановление/дыры отключим параметрами."""
    src = open(os.path.join(ROOT, "pc", "mekf.py"), encoding="utf-8").read()
    a = src.index("cy = y0 * u0")
    b = src.index("y2 -= cy * u2") + len("y2 -= cy * u2")
    cut = src[a:b]
    assert cut.count("\n") == 3, "разметка проекции изменилась"
    src_old = src.replace(cut, "cy = 0.0  # ДО ФИКСА: без проекции")
    mod = types.ModuleType("mekf_prefix")
    mod.__file__ = "mekf_prefix(reconstructed)"
    sys.modules["mekf_prefix"] = mod
    exec(compile(src_old, "mekf_prefix", "exec"), mod.__dict__)
    return mod


def run_gap(mekf_factory, label):
    cfg = load_cfg()
    t, A, G, H = read_session(os.path.join(ROOT, "data",
                                           "session_2026-07-04_00-38-35.csv"))
    t, A, G, H = t[::4], A[::4], G[::4], H[::4]      # ~104 Гц как поток
    keep = ~((t >= 75.0) & (t <= 95.0))
    p = PipelineReplica("mekf", cfg, mekf_factory=mekf_factory)
    vfs = []
    for i in np.where(keep)[0]:
        r = p.step(t[i], A[i], G[i], H[i])
        if r is not None:
            vfs.append((r["t"], r["vf"]))
    vfs = np.array(vfs)
    print(f"\n== {label}: дыра 75–95 с, поток ~104 Гц ==")
    print(f"  min vario {vfs[:,1].min():+.2f}, max {vfs[:,1].max():+.2f} м/с")
    if p.wd_events:
        print("  WATCHDOG СРАБОТАЛ: "
              + "; ".join(f"t={t0:.1f} с (расхождение {d:+.2f} м/с)"
                          for t0, d in p.wd_events))
    else:
        print("  watchdog не сработал")
    return p.wd_events


def run_clean(path, label):
    cfg = load_cfg()
    t, A, G, H = read_session(path)
    p = PipelineReplica("mekf", cfg)
    for i in range(len(t)):
        p.step(t[i], A[i], G[i], H[i])
    print(f"  {label}: ложных срабатываний watchdog: {len(p.wd_events)}"
          + (f" {p.wd_events}" if p.wd_events else ""))


if __name__ == "__main__":
    old = build_prefix_mekf_module()

    def old_factory(g_ref):
        cfg = old.load_mekf_config()
        cfg["chi2_recover_sec"] = 0.0
        cfg["hole_infl_deg"] = 0.0
        return old.make_mekf(cfg, g_ref)

    run_gap(old_factory, "ДО ФИКСА (реконструкция)")
    run_gap(None, "ПОСЛЕ ФИКСА")
    print("\n== ложные срабатывания на чистых файлах (после фикса) ==")
    for n in ("session_2026-07-04_00-38-35.csv", "session_2026-07-03_16-12-04.csv",
              "session_2026-07-03_15-34-53.csv"):
        run_clean(os.path.join(ROOT, "data", n), n)
