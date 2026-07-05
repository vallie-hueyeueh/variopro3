# -*- coding: utf-8 -*-
"""Пакет 14, Г.2 — приёмка синтеза звука ЧИСЛОМ (без аудио-устройства):
колбэк зовётся напрямую, выход анализируется.

Активный профиль (xctracer (1).txt): пороги climbOn 0.1 / sinkOn −0.7.
Приёмка:
  v=+2.0  → 675 Гц, цикл 400 мс, звучит 200 мс (скважность 50);
  v=+0.05 → тишина (ниже порога 0.1);
  v=−1.0  → непрерывный 440 Гц;
  v=−0.55 → тишина (между sink_on и climb_on).
Плюс: внутри бипа частота ПОСТОЯННА при смене цели на лету (латч на старте
бипа) и фронты без щелчков (макс скачок амплитуды между сэмплами мал).
Запуск: PYTHONIOENCODING=utf-8 python scripts/pack14_g2_sound.py
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pc"))

# sound_app тянет vario_app (Qt) только ради save/load_config — офлайн это ок
from sound_app import ToneProfile, VarioTonePlayer  # noqa: E402

PROFILE = os.path.join(ROOT, "data", "sound_profiles", "xctracer (1).txt")
FS = VarioTonePlayer.FS
BLOCK = VarioTonePlayer.BLOCK


def render(player, seconds):
    """Прогнать колбэк на seconds секунд, вернуть массив сэмплов."""
    n_blocks = int(seconds * FS / BLOCK)
    out = np.zeros((n_blocks * BLOCK, 1), dtype=np.float32)
    for b in range(n_blocks):
        player._callback(out[b * BLOCK:(b + 1) * BLOCK], BLOCK, None, None)
    return out[:, 0]


def on_mask(sig, win_ms=4.0, thr=0.03):
    """Маска «звучит»: RMS в окошках win_ms выше порога."""
    w = int(FS * win_ms / 1000.0)
    n = len(sig) // w
    rms = np.sqrt(np.mean(sig[:n * w].reshape(n, w) ** 2, axis=1))
    return rms > thr, w


def measure_beeps(sig):
    """(частота Гц, цикл мс, звучит мс) по устойчивой части сигнала."""
    m, w = on_mask(sig)
    # фронты вкл
    rises = np.where(np.diff(m.astype(int)) > 0)[0] + 1
    falls = np.where(np.diff(m.astype(int)) < 0)[0] + 1
    if len(rises) < 3:
        return None
    cyc = np.median(np.diff(rises)) * w / FS * 1000.0
    # длительность звучания: от каждого подъёма до следующего спада
    ons = []
    for r in rises[:-1]:
        f = falls[falls > r]
        if len(f):
            ons.append((f[0] - r) * w / FS * 1000.0)
    on_ms = float(np.median(ons)) if ons else float("nan")
    # частота: нуль-пересечения вверх в середине одного бипа
    r = rises[1] * w
    f = falls[falls > rises[1]][0] * w
    core = sig[r + int(0.010 * FS):f - int(0.010 * FS)]
    if len(core) < 100:
        return None
    zc = np.where((core[:-1] <= 0) & (core[1:] > 0))[0]
    freq = (len(zc) - 1) / ((zc[-1] - zc[0]) / FS) if len(zc) > 2 else float("nan")
    return freq, cyc, on_ms


def measure_cont(sig):
    """Частота непрерывного тона по устойчивой части."""
    core = sig[len(sig) // 2:]
    m, _ = on_mask(core)
    if m.mean() < 0.98:
        return None
    zc = np.where((core[:-1] <= 0) & (core[1:] > 0))[0]
    return (len(zc) - 1) / ((zc[-1] - zc[0]) / FS)


def check(name, cond, detail=""):
    print(f"  {'✓' if cond else '✗'} {name}{('  — ' + detail) if detail else ''}")
    return bool(cond)


def main():
    text = open(PROFILE, encoding="utf-8").read()
    prof = ToneProfile(text, os.path.basename(PROFILE))
    print(f"профиль: {prof.name}, точек {len(prof.points)}, "
          f"служебных строк {prof.unknown_lines}")
    pl = VarioTonePlayer()
    pl._enabled = True
    pl._volume = 0.8
    # пакет 15 (В.4): по умолчанию теперь МЕАНДР (как прибор) — его волна
    # прыгает ±A по построению; проверка «фронтов без щелчков» — инвариант
    # ОГИБАЮЩЕЙ и меряется на синусе (меандр отдельно проверяет pack15_v_sound)
    pl.set_waveform("sine")
    pl.set_profile(prof)
    c_on, c_off, s_on, s_off = prof.default_thresholds()
    pl.set_thresholds(c_on, c_off, s_on, s_off)
    ok = True

    # --- v = +2.0: 675 Гц / цикл 400 мс / звучит 200 мс ---
    pl.set_vario(2.0)
    sig = render(pl, 3.0)
    r = measure_beeps(sig)
    assert r is not None, "бипы не обнаружены"
    freq, cyc, on_ms = r
    ok &= check("v=+2.0 → 675 Гц", abs(freq - 675) < 5, f"измерено {freq:.1f} Гц")
    ok &= check("v=+2.0 → цикл 400 мс", abs(cyc - 400) < 15, f"{cyc:.0f} мс")
    ok &= check("v=+2.0 → звучит 200 мс", abs(on_ms - 200) < 20, f"{on_ms:.0f} мс")

    # --- щелчки: максимум скачка амплитуды соседних сэмплов ---
    dmax = float(np.max(np.abs(np.diff(sig))))
    # для 675 Гц на 44.1 кГц плавный синус сам даёт до 2π·675/44100·A ≈ 0.096·A
    ok &= check("фронты сглажены (нет щелчков)", dmax < 0.13,
                f"макс скачок {dmax:.3f} (чистый синус ~{0.096 * 0.8:.3f})")

    # --- частота внутри бипа ПОСТОЯННА при смене цели на лету (латч) ---
    pl.set_vario(None)
    render(pl, 0.5)                        # тишина: следующий бип начнётся заново
    pl.set_vario(2.0)                      # 675 Гц
    part1 = render(pl, 0.25)               # первый бип точно начался
    pl.set_vario(4.0)                      # цель изменилась (v=4 → 812.5 Гц)
    part2 = render(pl, 1.2)
    sig2 = np.concatenate([part1, part2])

    def freq_of(seg):
        zc = np.where((seg[:-1] <= 0) & (seg[1:] > 0))[0]
        if len(zc) < 3:
            return float("nan")
        return (len(zc) - 1) / ((zc[-1] - zc[0]) / FS)

    m, w = on_mask(sig2)
    rises = np.where(np.diff(m.astype(int)) > 0)[0] + 1
    falls = np.where(np.diff(m.astype(int)) < 0)[0] + 1
    if m[0]:
        rises = np.concatenate([[0], rises])
    beat_freqs = []
    in_beep_dev = 0.0
    for r in rises:
        f_after = falls[falls > r]
        if not len(f_after):
            continue
        a, b = r * w + int(0.012 * FS), f_after[0] * w - int(0.012 * FS)
        if b - a < int(0.06 * FS):
            continue
        mid = (a + b) // 2
        f1, f2 = freq_of(sig2[a:mid]), freq_of(sig2[mid:b])
        if np.isfinite(f1) and np.isfinite(f2):
            beat_freqs.append(0.5 * (f1 + f2))
            in_beep_dev = max(in_beep_dev, abs(f1 - f2))
    seen675 = any(abs(f - 675) < 10 for f in beat_freqs)
    seen812 = any(abs(f - 812.5) < 10 for f in beat_freqs)
    between = any(700 < f < 790 for f in beat_freqs)
    ok &= check("частота внутри бипа постоянна (латч на старте)",
                in_beep_dev < 8 and seen675 and seen812 and not between,
                f"макс расхождение половинок {in_beep_dev:.1f} Гц; бипы: "
                + ", ".join(f"{f:.0f}" for f in beat_freqs))

    # --- v = +0.05: тишина (порог 0.1) ---
    pl.set_vario(None)
    render(pl, 0.3)
    pl.set_vario(0.05)
    sig = render(pl, 1.0)
    ok &= check("v=+0.05 → тишина (порог 0.1)",
                float(np.max(np.abs(sig[FS // 10:]))) < 1e-3,
                f"макс |амплитуда| {np.max(np.abs(sig[FS // 10:])):.4f}")

    # --- v = −1.0: непрерывный 440 Гц ---
    pl.set_vario(-1.0)
    sig = render(pl, 2.0)
    f = measure_cont(sig)
    ok &= check("v=−1.0 → непрерывный 440 Гц", f is not None and abs(f - 440) < 4,
                f"измерено {f if f else float('nan'):.1f} Гц")

    # --- v = −0.55: тишина (между sink_on −0.7 и climb_on 0.1) ---
    pl.set_vario(None)
    render(pl, 0.3)
    pl.set_vario(-0.55)
    sig = render(pl, 1.0)
    ok &= check("v=−0.55 → тишина (между порогами)",
                float(np.max(np.abs(sig[FS // 10:]))) < 1e-3,
                f"макс |амплитуда| {np.max(np.abs(sig[FS // 10:])):.4f}")

    # --- гистерезис: звучало на −1.0 → на −0.65 продолжает (off −0.6) ---
    pl.set_vario(-1.0)
    render(pl, 0.5)
    pl.set_vario(-0.65)
    sig = render(pl, 0.5)
    ok &= check("гистерезис: −0.65 после −1.0 ещё звучит (off −0.6)",
                float(np.max(np.abs(sig))) > 0.1)
    pl.set_vario(-0.55)
    render(pl, 0.2)
    sig = render(pl, 0.5)
    ok &= check("… а на −0.55 гаснет", float(np.max(np.abs(sig))) < 1e-3)

    print("Г.2 ПРОЙДЕН ✓" if ok else "Г.2 ПРОВАЛЕН ✗")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
