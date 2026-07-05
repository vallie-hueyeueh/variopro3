package com.neuroskydynamics.variopro

import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.graphics.Color
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import com.google.android.gms.location.FusedLocationProviderClient
import com.google.android.gms.location.LocationCallback
import com.google.android.gms.location.LocationRequest
import com.google.android.gms.location.LocationResult
import com.google.android.gms.location.LocationServices
import com.google.android.gms.location.Priority
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.cos
import kotlin.math.min
import kotlin.math.sqrt

/**
 * VarioPro — Фаза 2: экран КАЛИБРОВКИ датчиков.
 *
 * Ведёт пользователя по трём шагам и складывает результат в один файл,
 * который потом загрузит ПК-пульт (раздел «Калибровка»):
 *   1) ГИРОСКОП     — телефон неподвижен 3 c → среднее (bias) и СКО (правило сигм);
 *   2) АКСЕЛЕРОМЕТР — N статичных положений (минимум 6), каждое проверяем на
 *                     неподвижность и близость модуля к g; плюс проверка, что
 *                     положения РАЗНЫЕ (охватывают 3D);
 *   3) МАГНИТОМЕТР  — поток mx,my,mz + одновременно gx,gy,gz (для будущего EKF).
 *
 * Формат файла описан в docs/calib_format.md.
 */
class CalibrationActivity : AppCompatActivity(), SensorEventListener {

    // --- пороги (правило сигм и проверки), легко настраиваются ---
    private val GYRO_CAPTURE_MS = 3000L      // длительность замера гироскопа
    private val ACCEL_CAPTURE_MS = 1000L     // длительность замера одного положения
    private val GYRO_STILL_STD = 0.03f       // макс. СКО гироскопа «в покое», рад/с
    private val ACCEL_STILL_STD = 0.20f      // макс. СКО акселерометра «в покое», м/с²
    private val ACCEL_G = 9.81f              // эталон |ускорения| для статики
    private val ACCEL_G_TOL = 0.6f           // допуск модуля от g, м/с²
    private val ORIENT_LAMBDA_MIN = 0.08     // порог «охвата 3D» (мин. собств. число)

    // --- датчики ---
    private lateinit var sensorManager: SensorManager
    private var accelSensor: Sensor? = null
    private var gyroSensor: Sensor? = null
    private var magSensor: Sensor? = null        // TYPE_MAGNETIC_FIELD_UNCALIBRATED (наш сырой)
    private var magCalSensor: Sensor? = null     // TYPE_MAGNETIC_FIELD (Android-калиброванный) — для сравнения

    // --- последние значения с датчиков ---
    private val accelV = FloatArray(3)
    private val gyroV = FloatArray(3)
    private val magV = FloatArray(3)             // сырое поле (uncalib), values[0..2]
    private val magBias = FloatArray(3)          // смещение от Android (uncalib values[3..5])
    private var magCalB = 0.0                     // |B| по Android-калиброванному датчику

    // --- режим текущего замера ---
    private enum class Mode { NONE, GYRO, ACCEL, MAG }
    private var mode = Mode.NONE

    // --- буферы текущего замера ---
    private val gyroSamples = ArrayList<FloatArray>()
    private val accelSamples = ArrayList<FloatArray>()
    private var magStartNs = 0L

    // --- РЕЗУЛЬТАТЫ калибровки ---
    private var gyroBias: FloatArray? = null
    private var gyroBiasStd: FloatArray? = null
    private val accelPoints = ArrayList<FloatArray>()  // статичные точки (ax,ay,az)
    private val magStream = ArrayList<DoubleArray>()   // [t,mx,my,mz,gx,gy,gz]
    private var accelLambdaMin = 0.0                   // охват 3D последней проверки
    private var allowAccelContinue = false            // пользователь настоял продолжить

    // --- виджеты ---
    private lateinit var btnGyro: Button
    private lateinit var tvGyroStatus: TextView
    private lateinit var etN: EditText
    private lateinit var btnAccelCapture: Button
    private lateinit var tvAccelCount: TextView
    private lateinit var tvAccelStatus: TextView
    private lateinit var btnAccelContinue: Button
    private lateinit var btnMagStart: Button
    private lateinit var btnMagStop: Button
    private lateinit var tvMagStatus: TextView
    private lateinit var btnSave: Button
    private lateinit var tvSaveStatus: TextView

    // --- блок «Текущая калибровка» (только управление файлами) ---
    private lateinit var tvCurrentCalib: TextView
    private lateinit var btnDeleteCalib: Button
    private lateinit var btnChooseCalib: Button
    private lateinit var btnRecapture: Button
    private var currentFile: File? = null   // файл, показанный в блоке «Текущая калибровка»

    // --- живой модуль |B| и его стабильность (последние ~3 с) ---
    private lateinit var tvMagB: TextView
    private lateinit var tvMagStab: TextView
    private val magBuf = ArrayDeque<DoubleArray>()   // [tNs, |B|] за последние ~3 с

    // живые значения акселерометра (видны ВСЕГДА, в т.ч. после набора N точек)
    private lateinit var tvAccelLive: TextView

    // --- GPS (координаты для ПК-пульта) ---
    private lateinit var tvGps: TextView
    private lateinit var btnGps: Button
    private var fusedClient: FusedLocationProviderClient? = null
    private var gpsLat: Double? = null
    private var gpsLon: Double? = null
    private var gpsAlt: Double? = null
    private val LOC_PERM_CODE = 1001
    private val locationCallback = object : LocationCallback() {
        override fun onLocationResult(result: LocationResult) {
            val loc = result.lastLocation ?: return
            gpsLat = loc.latitude
            gpsLon = loc.longitude
            gpsAlt = if (loc.hasAltitude()) loc.altitude else null
            updateGpsText()
        }
    }

    private val ui = Handler(Looper.getMainLooper())
    // обновление счётчика магнитометра во время записи
    private val magCounter = object : Runnable {
        override fun run() {
            if (mode == Mode.MAG) {
                setStatus(tvMagStatus, "● Идёт запись магнитометра: ${magStream.size} точек", Color.RED)
                ui.postDelayed(this, 200L)
            }
        }
    }
    // живой |B| и индикатор стабильности (всё время, пока экран открыт)
    private val magLive = object : Runnable {
        override fun run() {
            updateMagLive()
            updateAccelLive()
            ui.postDelayed(this, 150L)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_calibration)
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(bars.left, bars.top, bars.right, bars.bottom)
            insets
        }

        btnGyro = findViewById(R.id.btnGyro)
        tvGyroStatus = findViewById(R.id.tvGyroStatus)
        etN = findViewById(R.id.etN)
        btnAccelCapture = findViewById(R.id.btnAccelCapture)
        tvAccelCount = findViewById(R.id.tvAccelCount)
        tvAccelStatus = findViewById(R.id.tvAccelStatus)
        btnAccelContinue = findViewById(R.id.btnAccelContinue)
        btnMagStart = findViewById(R.id.btnMagStart)
        btnMagStop = findViewById(R.id.btnMagStop)
        tvMagStatus = findViewById(R.id.tvMagStatus)
        btnSave = findViewById(R.id.btnSave)
        tvSaveStatus = findViewById(R.id.tvSaveStatus)

        btnGyro.setOnClickListener { startGyro() }
        btnAccelCapture.setOnClickListener { startAccel() }
        btnAccelContinue.setOnClickListener { onAccelContinue() }
        btnMagStart.setOnClickListener { startMag() }
        btnMagStop.setOnClickListener { stopMag() }
        btnSave.setOnClickListener { saveFile() }

        // блок «Текущая калибровка» (только управление файлами)
        tvCurrentCalib = findViewById(R.id.tvCurrentCalib)
        btnDeleteCalib = findViewById(R.id.btnDeleteCalib)
        btnChooseCalib = findViewById(R.id.btnChooseCalib)
        btnRecapture = findViewById(R.id.btnRecapture)
        btnDeleteCalib.setOnClickListener { deleteCurrentCalib() }
        btnChooseCalib.setOnClickListener { chooseOtherCalib() }
        btnRecapture.setOnClickListener { recaptureReset() }

        // живой модуль поля и стабильность
        tvMagB = findViewById(R.id.tvMagB)
        tvMagStab = findViewById(R.id.tvMagStab)
        tvAccelLive = findViewById(R.id.tvAccelLive)

        // GPS
        tvGps = findViewById(R.id.tvGps)
        btnGps = findViewById(R.id.btnGps)
        btnGps.setOnClickListener { ensureLocation() }
        fusedClient = LocationServices.getFusedLocationProviderClient(this)

        sensorManager = getSystemService(SensorManager::class.java)
        accelSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        gyroSensor = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
        // САМЫЙ сырой магнитометр: Android НЕ вычитает свою динамическую калибровку
        magSensor = sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD_UNCALIBRATED)
        // Android-калиброванный (для сравнения «сырое vs Android»)
        magCalSensor = sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD)

        updateAccelCount()
        updateButtons()
    }

    override fun onResume() {
        super.onResume()
        accelSensor?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_FASTEST) }
        gyroSensor?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_FASTEST) }
        magSensor?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_FASTEST) }
        magCalSensor?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_FASTEST) }
        // показать последнюю сохранённую калибровку (а не пустой экран)
        refreshCurrentCalibration()
        ui.post(magLive)   // живой |B|, стабильность, живые ax/ay/az
        ensureLocation()   // GPS-координаты
    }

    override fun onPause() {
        super.onPause()
        // если шла запись магнитометра — корректно остановим
        if (mode == Mode.MAG) stopMag()
        sensorManager.unregisterListener(this)
        ui.removeCallbacks(magCounter)
        ui.removeCallbacks(magLive)
        fusedClient?.removeLocationUpdates(locationCallback)
    }

    // ---- приём данных датчиков (главный поток) ----
    override fun onSensorChanged(event: SensorEvent) {
        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> {
                System.arraycopy(event.values, 0, accelV, 0, 3)
                if (mode == Mode.ACCEL) accelSamples.add(accelV.copyOf())
            }
            Sensor.TYPE_GYROSCOPE -> {
                System.arraycopy(event.values, 0, gyroV, 0, 3)
                if (mode == Mode.GYRO) gyroSamples.add(gyroV.copyOf())
            }
            Sensor.TYPE_MAGNETIC_FIELD_UNCALIBRATED -> {
                // values[0..2] — поле БЕЗ вычета hard-iron самим Android (наше сырое)
                System.arraycopy(event.values, 0, magV, 0, 3)
                // values[3..5] — смещение (bias), которое сообщает сам Android
                if (event.values.size >= 6) System.arraycopy(event.values, 3, magBias, 0, 3)
                // буфер |B| за последние ~3 с (для живого модуля и стабильности)
                val b = kotlin.math.sqrt(
                    (magV[0] * magV[0] + magV[1] * magV[1] + magV[2] * magV[2]).toDouble())
                magBuf.addLast(doubleArrayOf(event.timestamp.toDouble(), b))
                while (magBuf.isNotEmpty() && event.timestamp - magBuf.first()[0] > 3e9) {
                    magBuf.removeFirst()
                }
                if (mode == Mode.MAG) {
                    if (magStartNs == 0L) magStartNs = event.timestamp
                    val t = (event.timestamp - magStartNs) / 1e9
                    // пишем сырое поле + гироскоп + смещение Android (bx,by,bz)
                    magStream.add(doubleArrayOf(
                        t,
                        magV[0].toDouble(), magV[1].toDouble(), magV[2].toDouble(),
                        gyroV[0].toDouble(), gyroV[1].toDouble(), gyroV[2].toDouble(),
                        magBias[0].toDouble(), magBias[1].toDouble(), magBias[2].toDouble()
                    ))
                }
            }
            Sensor.TYPE_MAGNETIC_FIELD -> {
                // Android-калиброванное поле — берём только его модуль для сравнения
                magCalB = kotlin.math.sqrt(
                    (event.values[0] * event.values[0] +
                     event.values[1] * event.values[1] +
                     event.values[2] * event.values[2]).toDouble())
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) { /* не используем */ }

    // ==================================================================
    // 1) ГИРОСКОП
    // ==================================================================
    private fun startGyro() {
        if (mode != Mode.NONE) return
        if (gyroSensor == null) {
            setStatus(tvGyroStatus, "Гироскоп не найден", Color.RED); return
        }
        mode = Mode.GYRO
        gyroSamples.clear()
        updateButtons()
        setStatus(tvGyroStatus, "Идёт калибровка… держите телефон НЕПОДВИЖНО (3 c)", Color.DKGRAY)
        ui.postDelayed({ finishGyro() }, GYRO_CAPTURE_MS)
    }

    private fun finishGyro() {
        mode = Mode.NONE
        updateButtons()
        if (gyroSamples.size < 10) {
            setStatus(tvGyroStatus, "Слишком мало данных, повторите", Color.RED); return
        }
        val (mean, std) = meanStd(gyroSamples)
        val maxStd = maxOf(std[0], std[1], std[2])
        // ПРАВИЛО СИГМ: если разброс (СКО) выше порога — было движение
        if (maxStd > GYRO_STILL_STD) {
            setStatus(tvGyroStatus,
                "Обнаружено движение, положите телефон неподвижно и повторите\n" +
                "(СКО=${f3(maxStd)} > ${f3(GYRO_STILL_STD)} рад/с)", Color.RED)
            return
        }
        gyroBias = mean
        gyroBiasStd = std
        setStatus(tvGyroStatus,
            "✓ Гироскоп откалиброван\n" +
            "  bias = [${f4(mean[0])} ${f4(mean[1])} ${f4(mean[2])}] рад/с\n" +
            "  СКО  = [${f4(std[0])} ${f4(std[1])} ${f4(std[2])}]",
            Color.rgb(0x2c, 0x7a, 0x2c))
    }

    // ==================================================================
    // 2) АКСЕЛЕРОМЕТР (статичные положения)
    // ==================================================================
    private fun startAccel() {
        if (mode != Mode.NONE) return
        if (accelSensor == null) {
            setStatus(tvAccelStatus, "Акселерометр не найден", Color.RED); return
        }
        mode = Mode.ACCEL
        accelSamples.clear()
        updateButtons()
        setStatus(tvAccelStatus, "Держите телефон неподвижно (1 c)…", Color.DKGRAY)
        ui.postDelayed({ finishAccel() }, ACCEL_CAPTURE_MS)
    }

    private fun finishAccel() {
        mode = Mode.NONE
        updateButtons()
        if (accelSamples.size < 10) {
            setStatus(tvAccelStatus, "Слишком мало данных, переснимите", Color.RED); return
        }
        val (mean, std) = meanStd(accelSamples)
        val maxStd = maxOf(std[0], std[1], std[2])
        val magnitude = sqrt(mean[0] * mean[0] + mean[1] * mean[1] + mean[2] * mean[2])

        // (а) неподвижность — правило сигм; (б) модуль близок к g
        if (maxStd > ACCEL_STILL_STD) {
            setStatus(tvAccelStatus,
                "Тряска/движение, переснимите положение (СКО=${f3(maxStd)})", Color.RED)
            return
        }
        if (abs(magnitude - ACCEL_G) > ACCEL_G_TOL) {
            setStatus(tvAccelStatus,
                "Тряска/движение, переснимите положение (|a|=${f2(magnitude)}, далёк от g)", Color.RED)
            return
        }

        accelPoints.add(mean)
        updateAccelCount()
        setStatus(tvAccelStatus,
            "✓ Положение принято: a=[${f2(mean[0])} ${f2(mean[1])} ${f2(mean[2])}]",
            Color.rgb(0x2c, 0x7a, 0x2c))

        // как только набрали N — проверяем РАЗНООБРАЗИЕ ориентаций
        if (accelPoints.size >= targetN()) checkAccelDiversity()
    }

    /**
     * Проверка охвата 3D. Пишет в tvAccelCount (счётчик + охват) — НЕ трогает
     * tvAccelStatus, чтобы строка «✓ Положение принято» оставалась видна
     * одновременно с охватом и живыми осями.
     */
    private fun checkAccelDiversity() {
        accelLambdaMin = orientationLambdaMin(accelPoints)
        val n = accelPoints.size
        if (accelLambdaMin < ORIENT_LAMBDA_MIN && !allowAccelContinue) {
            setStatus(tvAccelCount,
                "Снято $n из ${targetN()}\n" +
                "Снимите телефон в разных положениях (грани, наклоны)\n" +
                "(охват 3D мал: λmin=${f3(accelLambdaMin.toFloat())})", Color.rgb(0xC0, 0x70, 0x10))
            btnAccelContinue.visibility = android.view.View.VISIBLE
        } else {
            btnAccelContinue.visibility = android.view.View.GONE
            setStatus(tvAccelCount,
                "готово $n положений, охват 3D ок (λmin=${f3(accelLambdaMin.toFloat())})",
                Color.rgb(0x2c, 0x7a, 0x2c))
        }
    }

    private fun onAccelContinue() {
        allowAccelContinue = true
        checkAccelDiversity()   // покажет «готово … охват ок» и спрячет кнопку
    }

    private fun updateAccelCount() {
        tvAccelCount.text = "Снято ${accelPoints.size} из ${targetN()}"
    }

    private fun targetN(): Int {
        val v = etN.text.toString().toIntOrNull() ?: 6
        return maxOf(6, v)
    }

    // ==================================================================
    // 3) МАГНИТОМЕТР (вращение)
    // ==================================================================
    private fun startMag() {
        if (mode != Mode.NONE) return
        if (magSensor == null) {
            setStatus(tvMagStatus, "Магнитометр не найден", Color.RED); return
        }
        mode = Mode.MAG
        magStream.clear()
        magStartNs = 0L
        updateButtons()
        ui.post(magCounter)
    }

    private fun stopMag() {
        if (mode != Mode.MAG) return
        mode = Mode.NONE
        ui.removeCallbacks(magCounter)
        updateButtons()
        setStatus(tvMagStatus, "✓ Магнитометр: собрано ${magStream.size} точек",
            Color.rgb(0x2c, 0x7a, 0x2c))
    }

    // ==================================================================
    // ТЕКУЩАЯ КАЛИБРОВКА (постоянство + управление файлами)
    // ==================================================================

    /** Показать последнюю сохранённую калибровку (самую свежую файл). */
    private fun refreshCurrentCalibration() {
        currentFile = CalibrationStore.listFiles(this).firstOrNull()
        showCurrentStatus()
    }

    /** Показать статус текущего файла калибровки (или «нет»). */
    private fun showCurrentStatus() {
        val f = currentFile
        if (f == null || !f.exists()) {
            setStatus(tvCurrentCalib,
                "Нет сохранённых калибровок.\nСнимите данные ниже и нажмите «Сохранить».", Color.GRAY)
            btnDeleteCalib.isEnabled = false
            return
        }
        val d = CalibrationStore.parse(f)
        val gyroTxt = if (d?.gyroBias != null)
            "есть [${f4(d.gyroBias[0])} ${f4(d.gyroBias[1])} ${f4(d.gyroBias[2])}]" else "нет"
        setStatus(tvCurrentCalib,
            "Файл: ${f.name}\n" +
            "Дата: ${if (d?.created.isNullOrEmpty()) "—" else d!!.created}\n" +
            "Гироскоп bias: $gyroTxt\n" +
            "Акселерометр: ${d?.accelCount ?: 0} точек\n" +
            "Магнитометр: ${d?.magCount ?: 0} точек\n" +
            "(саму калибровку вычисляет ПК-пульт по этому файлу)",
            Color.DKGRAY)
        btnDeleteCalib.isEnabled = true
    }

    /** «Удалить» — удалить текущий файл калибровки. */
    private fun deleteCurrentCalib() {
        val f = currentFile ?: return
        AlertDialog.Builder(this)
            .setTitle("Удалить калибровку?")
            .setMessage(f.name)
            .setNegativeButton("Отмена", null)
            .setPositiveButton("Удалить") { _, _ ->
                f.delete()
                refreshCurrentCalibration()
            }
            .show()
    }

    /** Живой модуль |B| и индикатор стабильности (разброс за ~3 с). */
    private fun updateMagLive() {
        if (magSensor == null) {
            tvMagB.text = "|B|: магнитометр не найден"
            return
        }
        if (magBuf.isEmpty()) {
            tvMagB.text = "|B| = …  мкТл"
            return
        }
        val last = magBuf.last()[1]
        // ДВА значения рядом: сырое (uncalib) и Android-калиброванное
        tvMagB.text = String.format(
            Locale.US, "сырое (uncalib): %.1f мкТл    Android-калибр.: %.1f мкТл", last, magCalB)
        if (magBuf.size < 5) {
            setStatus(tvMagStab, "разброс: набор данных…", Color.GRAY)
            return
        }
        var sum = 0.0; var sum2 = 0.0; val n = magBuf.size
        for (e in magBuf) { sum += e[1]; sum2 += e[1] * e[1] }
        val mean = sum / n
        val variance = (sum2 / n - mean * mean).coerceAtLeast(0.0)
        val spread = if (mean > 1e-6) Math.sqrt(variance) / mean * 100.0 else 0.0
        val (txt, color) = when {
            spread < 3.0 -> "разброс: ${"%.1f".format(spread)}%  (чисто)" to Color.rgb(0x2c, 0x7a, 0x2c)
            spread < 8.0 -> "разброс: ${"%.1f".format(spread)}%  (так себе)" to Color.rgb(0xC0, 0x90, 0x10)
            else -> "разброс: ${"%.1f".format(spread)}%  (металл рядом?)" to Color.RED
        }
        setStatus(tvMagStab, txt, color)
    }

    /** Живые значения акселерометра (показываем ВСЕГДА, в т.ч. после набора N точек). */
    private fun updateAccelLive() {
        if (accelSensor == null) {
            tvAccelLive.text = "акселерометр не найден"
            return
        }
        tvAccelLive.text = String.format(
            Locale.US, "ax=%+.2f  ay=%+.2f  az=%+.2f м/с²", accelV[0], accelV[1], accelV[2])
    }

    // ==================================================================
    // GPS (FusedLocationProvider) — координаты для ПК-пульта
    // ==================================================================
    private fun ensureLocation() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION)
            == PackageManager.PERMISSION_GRANTED) {
            startLocationUpdates()
        } else {
            ActivityCompat.requestPermissions(
                this, arrayOf(Manifest.permission.ACCESS_FINE_LOCATION), LOC_PERM_CODE)
        }
    }

    @SuppressLint("MissingPermission")  // разрешение проверяем в ensureLocation()
    private fun startLocationUpdates() {
        val req = LocationRequest.Builder(Priority.PRIORITY_HIGH_ACCURACY, 2000L).build()
        try {
            fusedClient?.requestLocationUpdates(req, locationCallback, Looper.getMainLooper())
            if (gpsLat == null) tvGps.text = "GPS: получаю координаты…"
        } catch (e: SecurityException) {
            tvGps.text = "GPS: нет разрешения"
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == LOC_PERM_CODE) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                startLocationUpdates()
            } else {
                tvGps.text = "GPS: разрешение на геолокацию не выдано"
            }
        }
    }

    private fun updateGpsText() {
        val la = gpsLat
        val lo = gpsLon
        if (la == null || lo == null) {
            tvGps.text = "GPS: ещё нет данных"
            return
        }
        val altTxt = gpsAlt?.let { String.format(Locale.US, "%.1f", it) } ?: "—"
        tvGps.text = String.format(Locale.US, "GPS: φ=%.5f  λ=%.5f  h=%s м", la, lo, altTxt)
    }

    /** «Выбрать другую» — список сохранённых файлов калибровки. */
    private fun chooseOtherCalib() {
        val files = CalibrationStore.listFiles(this)
        if (files.isEmpty()) {
            setStatus(tvCurrentCalib, "Сохранённых калибровок нет", Color.GRAY); return
        }
        val names = files.map { it.name }.toTypedArray()
        AlertDialog.Builder(this)
            .setTitle("Выберите калибровку")
            .setItems(names) { _, which ->
                currentFile = files[which]
                showCurrentStatus()
            }
            .show()
    }

    /** «Снять заново» — сбросить буферы съёма для нового цикла (3 блока ниже). */
    private fun recaptureReset() {
        if (mode != Mode.NONE) return
        gyroBias = null
        gyroBiasStd = null
        accelPoints.clear()
        magStream.clear()
        allowAccelContinue = false
        accelLambdaMin = 0.0
        btnAccelContinue.visibility = android.view.View.GONE
        setStatus(tvGyroStatus, "Гироскоп: не откалиброван", Color.GRAY)
        setStatus(tvAccelStatus, "", Color.GRAY)
        setStatus(tvMagStatus, "Магнитометр: 0 точек", Color.GRAY)
        updateAccelCount()
        setStatus(tvSaveStatus, "Новый цикл: пройдите три шага ниже и сохраните.", Color.DKGRAY)
    }

    // ==================================================================
    // СОХРАНЕНИЕ
    // ==================================================================
    private fun saveFile() {
        if (mode != Mode.NONE) {
            setStatus(tvSaveStatus, "Сначала остановите текущий замер", Color.RED); return
        }
        if (gyroBias == null && accelPoints.isEmpty() && magStream.isEmpty()) {
            setStatus(tvSaveStatus, "Нечего сохранять — сделайте хотя бы один шаг", Color.RED); return
        }
        setStatus(tvSaveStatus, "Сохранение…", Color.DKGRAY)
        val json = buildJson()  // строим в главном потоке (быстро), пишем в фоне
        Thread {
            val dir: File = getExternalFilesDir(null) ?: filesDir
            val name = "calib_" + SimpleDateFormat("yyyy-MM-dd_HH-mm-ss", Locale.US).format(Date()) + ".json"
            val file = File(dir, name)
            try {
                file.writeText(json)
                runOnUiThread {
                    setStatus(tvSaveStatus,
                        "✓ Сохранено: $name\n  папка приложения (как достать — см. docs)",
                        Color.rgb(0x2c, 0x7a, 0x2c))
                    // показать свежий файл в блоке «Текущая калибровка» (ещё не применён)
                    currentFile = file
                    showCurrentStatus()
                }
            } catch (e: Exception) {
                runOnUiThread { setStatus(tvSaveStatus, "Ошибка сохранения: ${e.message}", Color.RED) }
            }
        }.start()
    }

    /** Собрать JSON в формате, который понимает ПК-пульт (см. docs/calib_format.md). */
    private fun buildJson(): String {
        val created = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US).format(Date())
        val sb = StringBuilder()
        sb.append("{\n")
        sb.append("  \"format\": \"variopro_calib\",\n")
        sb.append("  \"version\": 1,\n")
        sb.append("  \"created\": \"$created\",\n")
        sb.append("  \"device\": \"Samsung S23\",\n")
        sb.append("  \"accel_g\": $ACCEL_G,\n")
        sb.append("  \"units\": { \"accel\": \"m/s^2\", \"gyro\": \"rad/s\", \"mag\": \"uT\", \"t\": \"s\" },\n")
        sb.append("  \"mag_source\": \"TYPE_MAGNETIC_FIELD_UNCALIBRATED\",\n")
        // GPS-координаты (если получены) — для варианта «из GPS» на ПК
        val la = gpsLat; val lo = gpsLon
        if (la != null && lo != null) {
            val altStr = gpsAlt?.let { String.format(Locale.US, "%.2f", it) } ?: "null"
            sb.append("  \"gps\": { \"lat\": ${String.format(Locale.US, "%.6f", la)}, " +
                    "\"lon\": ${String.format(Locale.US, "%.6f", lo)}, \"alt\": $altStr },\n")
        } else {
            sb.append("  \"gps\": null,\n")
        }

        // гироскоп
        val gb = gyroBias
        if (gb != null) {
            val gs = gyroBiasStd!!
            sb.append("  \"gyro_bias\": [${f6(gb[0])}, ${f6(gb[1])}, ${f6(gb[2])}],\n")
            sb.append("  \"gyro_bias_std\": [${f6(gs[0])}, ${f6(gs[1])}, ${f6(gs[2])}],\n")
        } else {
            sb.append("  \"gyro_bias\": null,\n")
        }

        // акселерометр (точки для эллипсоида)
        sb.append("  \"accel_orientation_lambda_min\": ${f4(accelLambdaMin.toFloat())},\n")
        sb.append("  \"accel_points\": [")
        for (i in accelPoints.indices) {
            val p = accelPoints[i]
            if (i > 0) sb.append(", ")
            sb.append("[${f6(p[0])}, ${f6(p[1])}, ${f6(p[2])}]")
        }
        sb.append("],\n")

        // магнитометр(сырое) + гироскоп + смещение Android (bx,by,bz)
        sb.append("  \"mag_stream_columns\": [\"t\", \"mx\", \"my\", \"mz\", " +
                "\"gx\", \"gy\", \"gz\", \"bx\", \"by\", \"bz\"],\n")
        sb.append("  \"mag_stream\": [\n")
        for (i in magStream.indices) {
            val r = magStream[i]
            sb.append("    [${fd4(r[0])}, ${fd4(r[1])}, ${fd4(r[2])}, ${fd4(r[3])}, " +
                    "${fd6(r[4])}, ${fd6(r[5])}, ${fd6(r[6])}, " +
                    "${fd4(r[7])}, ${fd4(r[8])}, ${fd4(r[9])}]")
            sb.append(if (i < magStream.size - 1) ",\n" else "\n")
        }
        sb.append("  ]\n")
        sb.append("}\n")
        return sb.toString()
    }

    // ==================================================================
    // ВСПОМОГАТЕЛЬНОЕ
    // ==================================================================

    /** Среднее и СКО (std) по каждой оси для списка 3-векторов. */
    private fun meanStd(samples: List<FloatArray>): Pair<FloatArray, FloatArray> {
        val n = samples.size
        val mean = FloatArray(3)
        for (s in samples) for (k in 0..2) mean[k] += s[k]
        for (k in 0..2) mean[k] /= n
        val std = FloatArray(3)
        for (s in samples) for (k in 0..2) {
            val d = s[k] - mean[k]; std[k] += d * d
        }
        for (k in 0..2) std[k] = sqrt(std[k] / n)
        return mean to std
    }

    /**
     * Охват 3D: минимальное собственное число ковариации единичных направлений
     * гравитации. ~0.33 = равномерно во все стороны; ~0 = все в одной плоскости/кучей.
     */
    private fun orientationLambdaMin(points: List<FloatArray>): Double {
        var c00 = 0.0; var c11 = 0.0; var c22 = 0.0
        var c01 = 0.0; var c02 = 0.0; var c12 = 0.0
        var cnt = 0
        for (p in points) {
            val n = sqrt((p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).toDouble())
            if (n < 1e-6) continue
            val x = p[0] / n; val y = p[1] / n; val z = p[2] / n
            c00 += x * x; c11 += y * y; c22 += z * z
            c01 += x * y; c02 += x * z; c12 += y * z
            cnt++
        }
        if (cnt == 0) return 0.0
        c00 /= cnt; c11 /= cnt; c22 /= cnt; c01 /= cnt; c02 /= cnt; c12 /= cnt
        return minEigSym3(c00, c11, c22, c01, c02, c12)
    }

    /** Минимальное собственное число симметричной матрицы 3x3 (замкнутая формула). */
    private fun minEigSym3(a00: Double, a11: Double, a22: Double,
                           a01: Double, a02: Double, a12: Double): Double {
        val p1 = a01 * a01 + a02 * a02 + a12 * a12
        if (p1 == 0.0) return min(a00, min(a11, a22))   // уже диагональная
        val q = (a00 + a11 + a22) / 3.0
        val p2 = (a00 - q) * (a00 - q) + (a11 - q) * (a11 - q) + (a22 - q) * (a22 - q) + 2 * p1
        val p = sqrt(p2 / 6.0)
        val b00 = (a00 - q) / p; val b11 = (a11 - q) / p; val b22 = (a22 - q) / p
        val b01 = a01 / p; val b02 = a02 / p; val b12 = a12 / p
        val detB = b00 * (b11 * b22 - b12 * b12) -
                   b01 * (b01 * b22 - b12 * b02) +
                   b02 * (b01 * b12 - b11 * b02)
        var r = detB / 2.0
        if (r < -1.0) r = -1.0 else if (r > 1.0) r = 1.0
        val phi = acos(r) / 3.0
        val eig1 = q + 2 * p * cos(phi)
        val eig3 = q + 2 * p * cos(phi + 2.0 * PI / 3.0)
        val eig2 = 3 * q - eig1 - eig3
        return min(eig1, min(eig2, eig3))
    }

    /** Кнопки активны/неактивны в зависимости от текущего замера. */
    private fun updateButtons() {
        val idle = mode == Mode.NONE
        btnGyro.isEnabled = idle
        btnAccelCapture.isEnabled = idle
        btnMagStart.isEnabled = idle
        btnMagStop.isEnabled = (mode == Mode.MAG)
        btnSave.isEnabled = idle
    }

    private fun setStatus(tv: TextView, text: String, color: Int) {
        tv.setTextColor(color)
        tv.text = text
    }

    // форматирование чисел
    private fun f2(x: Float) = String.format(Locale.US, "%.2f", x)
    private fun f3(x: Float) = String.format(Locale.US, "%.3f", x)
    private fun f4(x: Float) = String.format(Locale.US, "%.4f", x)
    private fun f6(x: Float) = String.format(Locale.US, "%.6f", x)
    private fun fd4(x: Double) = String.format(Locale.US, "%.4f", x)
    private fun fd6(x: Double) = String.format(Locale.US, "%.6f", x)
}
