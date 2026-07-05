package com.neuroskydynamics.variopro

import android.content.Context
import org.json.JSONObject
import java.io.File

/**
 * Помощник по ФАЙЛАМ калибровки. Телефон только СОБИРАЕТ и СОХРАНЯЕТ сырьё;
 * вычисление и применение калибровки делает ПК-пульт. Поэтому здесь только:
 *   • разбор файла для показа статуса (bias, число точек, дата);
 *   • список сохранённых файлов (для «выбрать другую»).
 *
 * Формат файла — см. docs/calib_format.md.
 */
object CalibrationStore {

    /** Краткое содержимое файла калибровки (для показа на экране). */
    class Data(
        val gyroBias: FloatArray?,
        val accelCount: Int,
        val magCount: Int,
        val created: String,
        val name: String
    )

    /** Список сохранённых файлов калибровки (новые сверху). */
    fun listFiles(ctx: Context): List<File> {
        val dir = ctx.getExternalFilesDir(null) ?: ctx.filesDir
        val arr = dir.listFiles { f -> f.name.startsWith("calib_") && f.name.endsWith(".json") }
            ?: return emptyList()
        return arr.sortedByDescending { it.lastModified() }
    }

    /** Разобрать файл калибровки для показа статуса. null — если не читается. */
    fun parse(file: File): Data? {
        return try {
            val obj = JSONObject(file.readText())
            val gb = obj.optJSONArray("gyro_bias")
            val gyroBias = if (gb != null && gb.length() >= 3)
                floatArrayOf(gb.getDouble(0).toFloat(), gb.getDouble(1).toFloat(), gb.getDouble(2).toFloat())
            else null
            val accelCount = obj.optJSONArray("accel_points")?.length() ?: 0
            val magCount = obj.optJSONArray("mag_stream")?.length() ?: 0
            Data(gyroBias, accelCount, magCount, obj.optString("created", ""), file.name)
        } catch (e: Exception) {
            null
        }
    }
}
