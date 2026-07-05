# -*- coding: utf-8 -*-
"""
make_sample_data.py
===================
Создаёт ПРИМЕР ЗАПИСИ полёта в файл data/sample_flight.csv.

Зачем: чтобы режим «CSV-файл» в программе можно было проверить сразу, не имея
телефона и реальных записей. Данные берём из той же синтетической сцены, что и
симуляция (стол → термик → горизонт → снижение).

Формат файла (те же столбцы, что потом пришлёт телефон по Bluetooth):
    t,a_world_vertical,h_baro

Запуск:
    python pc/make_sample_data.py
"""

import os
import csv

from sensor_source import generate_synthetic

# корень проекта = папка над pc/; демо-файлы живут в data/samples/
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
SAMPLES_DIR = os.path.join(DATA_DIR, "samples")


def main():
    # сгенерировать сцену (t — время, a — ускорение датчика, h — высота баро)
    t, a, h, _, _ = generate_synthetic(dt=0.02, T=90.0, seed=42)

    os.makedirs(SAMPLES_DIR, exist_ok=True)
    path = os.path.join(SAMPLES_DIR, "sample_flight.csv")

    # записать в CSV со строкой-заголовком
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["t", "a_world_vertical", "h_baro"])  # заголовок
        for i in range(len(t)):
            # округляем для компактности файла
            writer.writerow([f"{t[i]:.3f}", f"{a[i]:.5f}", f"{h[i]:.5f}"])

    print(f"Готово: создан пример записи ({len(t)} строк)")
    print(f"  {path}")


if __name__ == "__main__":
    main()
