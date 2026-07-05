# -*- coding: utf-8 -*-
r"""Пакет 15, блок В.3 — диагностика звука ДО правок и приёмка ПОСЛЕ.

Рендерит data\sound_profiles\debug_v+2.wav и debug_v-1.wav (по 3 с, активный
профиль и пороги из config, НАСТОЯЩИЙ аудио-колбэк VarioTonePlayer._callback —
без звуковой карты), затем проверяет числом:
  • v=+2.0 → пачки НУЖНОЙ частоты по ~цикл·скважность мс с паузами (по профилю);
  • v=−1.0 → непрерывный тон нужной частоты (без пауз после атаки);
  • спектр: доминирующая частота каждой пачки;
  • дребезг у порога: серия v = climb_on ± 0.03 (0.5 Гц) — переключений газа
    должно быть мало (гистерезис), не десятки;
  • форма волны (пакет 15, В.4): square/sine — печатается фактическая.

Запуск: PYTHONIOENCODING=utf-8 python scripts/pack15_v_sound.py [--wave sine]
"""
import argparse
import os
import sys
import wave

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pc"))

import numpy as np  # noqa: E402


def check(name, cond, detail=""):
    print(f"  {'✓' if cond else '✗'} {name}{('  — ' + detail) if detail else ''}")
    return bool(cond)


def render(player, v, seconds=3.0):
    """Прогнать НАСТОЯЩИЙ колбэк плеера и вернуть сигнал float32."""
    player.set_vario(v)
    n_blocks = int(seconds * player.FS / player.BLOCK) + 1
    out = np.zeros((player.BLOCK, 1), dtype=np.float32)
    chunks = []
    for _ in range(n_blocks):
        player._callback(out, player.BLOCK, None, None)
        chunks.append(out[:, 0].copy())
    return np.concatenate(chunks)[: int(seconds * player.FS)]


def save_wav(path, sig, fs):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(fs)
        w.writeframes((np.clip(sig, -1, 1) * 32767).astype("<i2").tobytes())


def bursts(sig, fs, thr=0.01, min_gap_ms=20):
    """Границы пачек по огибающей: список (t0, t1) в секундах."""
    env = np.abs(sig)
    k = int(0.002 * fs)
    env = np.convolve(env, np.ones(k) / k, mode="same")
    on = env > thr
    edges = np.flatnonzero(np.diff(on.astype(int)))
    if on[0]:
        edges = np.r_[0, edges]
    if on[-1]:
        edges = np.r_[edges, len(on) - 1]
    pairs = [(edges[i] / fs, edges[i + 1] / fs) for i in range(0, len(edges) - 1, 2)]
    # склеить пачки, разделённые «паузой» короче min_gap_ms (дрожание огибающей)
    out = []
    for p in pairs:
        if out and (p[0] - out[-1][1]) * 1000.0 < min_gap_ms:
            out[-1] = (out[-1][0], p[1])
        else:
            out.append(p)
    return out


def dom_freq(sig, fs):
    """Доминирующая частота куска (парабол. уточнение пика БПФ)."""
    if len(sig) < 256:
        return 0.0
    w = sig * np.hanning(len(sig))
    sp = np.abs(np.fft.rfft(w))
    i = int(np.argmax(sp[1:])) + 1
    if 1 <= i < len(sp) - 1:
        a, b, c = sp[i - 1], sp[i], sp[i + 1]
        d = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0.0
    else:
        d = 0.0
    return (i + d) * fs / len(sig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", choices=["square", "sine", "config"], default="config")
    args = ap.parse_args()

    import sound_app
    sound_app.save_config = lambda *a, **k: None
    from vario_app import load_config
    cfg_all = load_config()
    snd = cfg_all.get("sound", {}) if isinstance(cfg_all.get("sound"), dict) else {}
    prof_name = snd.get("profile", sound_app.DEFAULT_PROFILE_NAME)
    prof_path = os.path.join(sound_app.PROFILES_DIR, prof_name)
    text = open(prof_path, "r", encoding="utf-8").read()
    prof = sound_app.ToneProfile(text, prof_name)
    assert prof.valid, "активный профиль без таблицы tone="

    player = sound_app.VarioTonePlayer()
    player.set_profile(prof)
    if args.wave != "config":
        player.set_waveform(args.wave)
    p_c, p_c_off, p_s, p_s_off = prof.default_thresholds()
    c_on = snd.get("climb_on", p_c)
    s_on = snd.get("sink_on", p_s)
    c_gap = (p_c - p_c_off) if (p_c is not None and p_c_off is not None) else 0.05
    s_gap = (p_s_off - p_s) if (p_s is not None and p_s_off is not None) else 0.10
    c_off = (c_on - max(c_gap, 0.0)) if c_on is not None else None
    s_off = (s_on + max(s_gap, 0.0)) if s_on is not None else None
    player.set_thresholds(c_on, c_off, s_on, s_off)
    player._enabled = True
    player.set_volume(0.9)
    fs = player.FS
    wave_now = getattr(player, "waveform", "sine (нет поля — код до пакета 15)")
    print(f"профиль: {prof_name}; пороги: climb_on={c_on} (off {c_off}), "
          f"sink_on={s_on} (off {s_off}); форма волны: {wave_now}")

    ok = True
    # --- v=+2.0: пачки ---
    f_exp, cyc_exp, duty_exp = prof.lookup(2.0)
    on_exp = cyc_exp * duty_exp / 100.0
    sig = render(player, +2.0, 3.0)
    save_wav(os.path.join(sound_app.PROFILES_DIR, "debug_v+2.wav"), sig, fs)
    bl = bursts(sig, fs)
    inner = bl[1:-1] if len(bl) > 2 else bl     # крайние могут быть обрезаны
    durs = [(b - a) * 1000.0 for (a, b) in inner]
    gaps = [(inner[i + 1][0] - inner[i][1]) * 1000.0 for i in range(len(inner) - 1)]
    freqs = [dom_freq(sig[int(a * fs):int(b * fs)], fs) for (a, b) in inner]
    print(f"\nv=+2.0 (ожидание: {f_exp:.0f} Гц, цикл {cyc_exp:.0f} мс, "
          f"звучит {on_exp:.0f} мс): пачек {len(bl)}")
    if durs:
        print(f"  длит. пачек {np.mean(durs):.0f}±{np.std(durs):.0f} мс, "
              f"паузы {np.mean(gaps):.0f}±{np.std(gaps):.0f} мс, "
              f"частоты {min(freqs):.1f}–{max(freqs):.1f} Гц")
    ok &= check("пачек за 3 с ≈ 3с/цикл", abs(len(bl) - 3000.0 / cyc_exp) <= 1.5,
                f"{len(bl)} против {3000.0 / cyc_exp:.1f}")
    ok &= check("длительность пачки = цикл·скважность ±15 мс",
                bool(durs) and abs(np.mean(durs) - on_exp) <= 15,
                f"{np.mean(durs):.0f} против {on_exp:.0f} мс")
    ok &= check("частота пачек верна ±2 Гц",
                bool(freqs) and max(abs(f - f_exp) for f in freqs) <= 2.0,
                f"{np.mean(freqs):.1f} против {f_exp:.1f} Гц")

    # --- v=−1.0: непрерывный ---
    f_exp1, cyc1, duty1 = prof.lookup(-1.0)
    sig1 = render(player, -1.0, 3.0)
    save_wav(os.path.join(sound_app.PROFILES_DIR, "debug_v-1.wav"), sig1, fs)
    b1 = bursts(sig1, fs)
    f1 = dom_freq(sig1[fs // 2:], fs)
    cont = len(b1) == 1 and (b1[0][1] - b1[0][0]) > 2.5
    print(f"\nv=−1.0 (ожидание: непрерывный {f_exp1:.0f} Гц; скважность профиля "
          f"{duty1:.0f}): пачек {len(b1)}, частота {f1:.1f} Гц")
    ok &= check("непрерывный тон (одна «пачка» ≥2.5 с)", cont,
                f"{len(b1)} пачек")
    ok &= check("частота верна ±2 Гц", abs(f1 - f_exp1) <= 2.0,
                f"{f1:.1f} против {f_exp1:.1f}")

    # --- дребезг у порога: v колеблется вокруг climb_on ±0.03 (0.5 Гц), 10 с ---
    if c_on is not None:
        player.set_vario(None)
        # прогнать колбэк, чтобы дойти до тишины
        render(player, None, 0.3)
        gate_changes = 0
        gate_prev = None
        for i in range(300):                     # 10 с по ~30 Гц
            v = c_on + 0.03 * np.sin(2 * np.pi * 0.5 * i / 30.0)
            player.set_vario(float(v))
            g = player._gate_on
            if gate_prev is not None and g != gate_prev:
                gate_changes += 1
            gate_prev = g
        # синус 0.5 Гц за 10 с пересекает ВКЛ-порог 5 раз вверх; выключение —
        # ниже off (= on − гистерезис 0.05 > амплитуды 0.03) → газ должен
        # включиться ОДИН раз и не выключаться
        print(f"\nдребезг у порога: переключений газа {gate_changes} "
              f"(без гистерезиса было бы ~10)")
        ok &= check("гистерезис держит газ (переключений ≤ 2)", gate_changes <= 2,
                    f"{gate_changes}")

    print("\nВ.3 ПРОЙДЕН ✓" if ok else "\nВ.3: ЕСТЬ ПРОВАЛЫ ✗")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
