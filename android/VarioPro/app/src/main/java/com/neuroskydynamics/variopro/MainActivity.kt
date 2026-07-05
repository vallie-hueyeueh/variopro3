package com.neuroskydynamics.variopro

import android.Manifest
import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothManager
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Color
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.location.Location
import android.location.LocationManager
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.os.Looper
import android.os.SystemClock
import android.provider.Settings
import android.widget.Button
import android.widget.TextView
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.location.LocationManagerCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import com.google.android.gms.location.FusedLocationProviderClient
import com.google.android.gms.location.LocationServices
import com.google.android.gms.location.Priority
import com.google.android.gms.tasks.CancellationTokenSource
import java.io.BufferedWriter
import java.io.File
import java.io.FileWriter
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * VarioPro — Фаза 1, шаг 2.
 * Читает 4 датчика телефона, показывает их на экране И умеет записывать сессию
 * в CSV-файл (кнопки «Запись» / «Стоп»). Bluetooth и фильтр пока не подключаем.
 *
 * Класс реализует SensorEventListener — Android присылает сюда показания датчиков
 * в метод onSensorChanged().
 */
class MainActivity : AppCompatActivity(), SensorEventListener {

    // "Диспетчер датчиков" Android — через него находим датчики и подписываемся на них.
    private lateinit var sensorManager: SensorManager

    // Сами датчики. null = такого датчика в телефоне нет.
    private var accelSensor: Sensor? = null   // акселерометр
    private var gyroSensor: Sensor? = null    // гироскоп
    private var magSensor: Sensor? = null     // магнитометр СЫРОЙ (uncalibrated)
    private var magCalSensor: Sensor? = null  // магнитометр Android-КАЛИБРОВАННЫЙ (v4)
    private var pressSensor: Sensor? = null   // барометр (давление)

    // Последние пришедшие значения (их показываем на экране и пишем в CSV).
    private val accelVals = FloatArray(3)     // ax, ay, az  (м/с²)
    private val gyroVals = FloatArray(3)      // gx, gy, gz  (рад/с)
    private val magVals = FloatArray(3)       // mx, my, mz  (мкТл, сырое)
    private val magCalVals = FloatArray(3)    // mxa, mya, mza (мкТл, Android-калиброванное)
    private var pressVal = 0f                  // давление    (гПа)

    // Пришли ли уже первые данные (чтобы до первого замера писать "ожидание…").
    private var gotAccel = false
    private var gotGyro = false
    private var gotMag = false
    private var gotPress = false

    // Счётчики событий — по ним измеряем ФАКТИЧЕСКУЮ частоту (Гц) каждого датчика.
    private var accelCount = 0
    private var gyroCount = 0
    private var magCount = 0
    private var pressCount = 0
    private var hzWindowStartNs = 0L          // начало окна измерения частоты
    private var accelHz = 0f
    private var gyroHz = 0f
    private var magHz = 0f
    private var pressHz = 0f

    // --- Экранные поля ---
    private lateinit var tvLive: TextView
    private lateinit var tvRecStatus: TextView
    private lateinit var btnRecord: Button
    private lateinit var btnStop: Button

    // --- Bluetooth-стриминг (Фаза 3B) ---
    private lateinit var btnStreamToggle: Button
    private lateinit var tvStreamStatus: TextView
    private val BT_PERM_CODE = 2001

    // --- GPS-блок: статус + разовый запрос позиции ---
    private lateinit var tvGpsStatus: TextView
    private lateinit var btnGpsEnable: Button
    private lateinit var btnGpsRefresh: Button
    private val GPS_PERM_CODE = 2002
    private var fusedClient: FusedLocationProviderClient? = null
    private var lastFix: Location? = null          // последний полученный фикс
    private var lastFixWallMs = 0L                 // когда получен (для возраста)
    private var gpsRequesting = false              // идёт разовый запрос

    // --- Запись CSV ---
    private lateinit var writerThread: HandlerThread  // отдельный поток для записи на диск
    private lateinit var writerHandler: Handler       // очередь задач записи
    private var writer: BufferedWriter? = null        // сам файл (трогаем только из writerThread)
    private var pendingSinceFlush = 0                 // счётчик строк до сброса на диск (writerThread)
    private var isRecording = false                   // идёт ли запись
    private var recStartNs = 0L                        // время первого замера (для t от нуля)
    private var rowCount = 0                           // сколько строк записано
    private var lastT = 0f                             // время последней строки, с
    private var currentFileName: String? = null        // имя текущего файла
    private var currentFilePath: String? = null        // полный путь файла

    // Таймер: обновляем ЭКРАН ~7 раз в секунду (а не на каждое событие датчика).
    private val ui = Handler(Looper.getMainLooper())
    private val refreshMs = 150L
    private val refresher = object : Runnable {
        override fun run() {
            updateHz()                       // пересчитать частоты
            tvLive.text = buildLiveText()    // обновить цифры на экране
            if (isRecording) updateRecStatus()  // обновить строку записи (секунды/строки)
            updateStreamStatus()             // строка Bluetooth-стриминга
            updateGpsStatus()                // строка GPS (статус всегда видно)
            ui.postDelayed(this, refreshMs)  // повторить через 150 мс
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // ПРОВЕРКА ДАТЧИКОВ НА СТАРТЕ (пакет 13, блок Е.2): барометр, акселерометр
        // и гироскоп ОБЯЗАТЕЛЬНЫ (без них вариометр не работает); магнитометр
        // желателен (без него — нет компаса, остальное работает).
        sensorManager = getSystemService(SensorManager::class.java)
        accelSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        gyroSensor = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
        magSensor = sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD_UNCALIBRATED)
        magCalSensor = sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD)
        pressSensor = sensorManager.getDefaultSensor(Sensor.TYPE_PRESSURE)
        val missing = ArrayList<String>()
        if (pressSensor == null) missing.add("барометр")
        if (accelSensor == null) missing.add("акселерометр")
        if (gyroSensor == null) missing.add("гироскоп")
        if (missing.isNotEmpty()) {
            showUnsupportedScreen(missing)
            return
        }

        enableEdgeToEdge()
        setContentView(R.layout.activity_main)
        // отступ под системные панели (часы/навигация), как было в шаблоне
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(bars.left, bars.top, bars.right, bars.bottom)
            insets
        }

        tvLive = findViewById(R.id.tvLive)
        tvRecStatus = findViewById(R.id.tvRecStatus)
        btnRecord = findViewById(R.id.btnRecord)
        btnStop = findViewById(R.id.btnStop)
        btnRecord.setOnClickListener { startRecording() }
        btnStop.setOnClickListener { stopRecording() }

        // кнопки перехода: экран калибровки и экран файлов
        findViewById<Button>(R.id.btnOpenCalibration).setOnClickListener {
            startActivity(Intent(this, CalibrationActivity::class.java))
        }
        findViewById<Button>(R.id.btnOpenFiles).setOnClickListener {
            startActivity(Intent(this, FilesActivity::class.java))
        }

        // Bluetooth-стриминг (Фаза 3B)
        btnStreamToggle = findViewById(R.id.btnStreamToggle)
        tvStreamStatus = findViewById(R.id.tvStreamStatus)
        btnStreamToggle.setOnClickListener { toggleStreaming() }

        // GPS-блок: «Включить GPS» → системные настройки геолокации;
        // «Обновить» → разовый запрос позиции (спросит разрешение, если надо)
        tvGpsStatus = findViewById(R.id.tvGpsStatus)
        btnGpsEnable = findViewById(R.id.btnGpsEnable)
        btnGpsRefresh = findViewById(R.id.btnGpsRefresh)
        fusedClient = LocationServices.getFusedLocationProviderClient(this)
        btnGpsEnable.setOnClickListener {
            try {
                startActivity(Intent(Settings.ACTION_LOCATION_SOURCE_SETTINGS))
            } catch (e: Exception) {
                tvGpsStatus.text = "не удалось открыть настройки геолокации"
            }
        }
        btnGpsRefresh.setOnClickListener { requestGpsFix() }

        // Кнопка «Источник магнитометра» УБРАНА (пакет 14, Б.2): телефон всегда
        // пишет и шлёт ОБА поля (сырое mx..mz и Android-калиброванное mxa..mza);
        // что использует компас — выбирается на ПК (вкладка «Калибровка»).

        // отдельный поток для записи файла (чтобы не тормозить экран)
        writerThread = HandlerThread("csv-writer").apply { start() }
        writerHandler = Handler(writerThread.looper)

        // Характеристики датчиков (один раз — они не меняются); датчики найдены
        // выше, до проверки Е.2. СЫРОЙ магнитометр (uncalibrated) остаётся
        // основным для ЗАПИСИ (пригоден для калибровки эллипсоида);
        // TYPE_MAGNETIC_FIELD пишется РЯДОМ в колонки mxa..mza (протокол v4).
        findViewById<TextView>(R.id.tvSensorInfo).text = buildSensorInfo()
    }

    /** Экран «Устройство не подходит» (блок Е.2): нет обязательного датчика. */
    private fun showUnsupportedScreen(missing: List<String>) {
        val pad = (16 * resources.displayMetrics.density).toInt()
        val tv = TextView(this)
        tv.setPadding(pad, pad * 2, pad, pad)
        tv.textSize = 18f
        tv.text = buildString {
            append("Устройство не подходит: нет — ${missing.joinToString(", ")}.\n\n")
            append("VarioPro3 требует: барометр, акселерометр, гироскоп ")
            append("(магнитометр — желателен, для компаса).\n\n")
            append("Модель: ${Build.MODEL}\n\nНайдено на этом устройстве:\n")
            append(buildSensorInfo())
        }
        val sv = android.widget.ScrollView(this)
        sv.addView(tv)
        setContentView(sv)
    }

    companion object {
        const val PREFS = "variopro"
        // настройка mag_source убрана (пакет 14, Б.2): оба поля идут всегда,
        // выбор источника для компаса делает ПК
    }

    /**
     * onResume — экран активен. ЗДЕСЬ подписываемся на датчики,
     * чтобы они работали только когда экран виден.
     */
    override fun onResume() {
        super.onResume()
        if (!::tvLive.isInitialized) return   // экран «не подходит» (Е.2): датчиков нет
        accelSensor?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_FASTEST) }
        gyroSensor?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_FASTEST) }
        magSensor?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_FASTEST) }
        magCalSensor?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_FASTEST) }
        pressSensor?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_FASTEST) }

        accelCount = 0; gyroCount = 0; magCount = 0; pressCount = 0
        hzWindowStartNs = System.nanoTime()
        ui.post(refresher)
    }

    /**
     * onPause — экран ушёл на задний план. ЗДЕСЬ отписываемся от датчиков и
     * останавливаем таймер (иначе зря садится батарея). Если шла запись —
     * аккуратно её завершаем (в фоне писать нельзя без foreground-сервиса).
     */
    override fun onPause() {
        super.onPause()
        if (isRecording) stopRecording()        // закрыть файл, не оставлять "висящим"
        sensorManager.unregisterListener(this)   // отписка сразу от всех датчиков
        ui.removeCallbacks(refresher)            // остановить обновление экрана
    }

    /** Освобождаем поток записи при полном закрытии приложения. */
    override fun onDestroy() {
        super.onDestroy()
        writerThread.quitSafely()
    }

    /** Сюда Android присылает новое показание датчика. Делаем минимум — запоминаем (и пишем строку CSV на событии акселерометра). */
    override fun onSensorChanged(event: SensorEvent) {
        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> {
                System.arraycopy(event.values, 0, accelVals, 0, 3); gotAccel = true; accelCount++
                // CSV пишем по событию АКСЕЛЕРОМЕТРА (он самый частый);
                // остальные датчики берём последними известными значениями.
                if (isRecording) writeCsvRow(event.timestamp)
            }
            Sensor.TYPE_GYROSCOPE -> {
                System.arraycopy(event.values, 0, gyroVals, 0, 3); gotGyro = true; gyroCount++
            }
            Sensor.TYPE_MAGNETIC_FIELD_UNCALIBRATED -> {
                // values[0..2] — сырое поле (без вычета hard-iron самим Android)
                System.arraycopy(event.values, 0, magVals, 0, 3); gotMag = true; magCount++
            }
            Sensor.TYPE_MAGNETIC_FIELD -> {
                // Android-калиброванное поле (ОС уже сняла железо) — колонки mxa..mza (v4)
                System.arraycopy(event.values, 0, magCalVals, 0, 3)
            }
            Sensor.TYPE_PRESSURE -> {
                pressVal = event.values[0]; gotPress = true; pressCount++
            }
        }
    }

    /** Точность датчика поменялась — для нашей задачи не нужно. */
    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) { /* не используем */ }

    // ==================================================================
    // ЗАПИСЬ CSV
    // ==================================================================

    /** Нажата «Запись»: создаём файл с датой-временем в папке приложения и пишем заголовок. */
    private fun startRecording() {
        if (isRecording) return
        if (accelSensor == null) {
            tvRecStatus.text = "Нет акселерометра — записывать нечего"
            return
        }

        // папка приложения во внешней памяти (разрешения не нужны); запасной вариант — внутренняя память
        val dir: File = getExternalFilesDir(null) ?: filesDir
        val stamp = SimpleDateFormat("yyyy-MM-dd_HH-mm-ss", Locale.US).format(Date())
        val name = "session_$stamp.csv"
        val file = File(dir, name)

        // сбросить счётчики записи
        recStartNs = 0L
        rowCount = 0
        lastT = 0f
        currentFileName = name
        currentFilePath = file.absolutePath
        isRecording = true

        // открыть файл и записать заголовок — в потоке записи
        writerHandler.post {
            try {
                pendingSinceFlush = 0
                val w = BufferedWriter(FileWriter(file))
                // v4: mx..mz — СЫРОЕ поле (для калибровки эллипсоида, как раньше),
                // mxa..mza — Android-калиброванное (для компаса). Старые читатели
                // лишние колонки игнорируют.
                w.write("t,ax,ay,az,gx,gy,gz,mx,my,mz,pressure,altitude,mxa,mya,mza\n")
                writer = w
            } catch (e: IOException) {
                writer = null
                // сообщить об ошибке в главном потоке
                ui.post {
                    isRecording = false
                    setRecButtons(recording = false)
                    tvRecStatus.text = "Ошибка записи: ${e.message}"
                }
            }
        }

        setRecButtons(recording = true)
        updateRecStatus()
    }

    /** Нажата «Стоп» (или ушли с экрана): сбрасываем буфер на диск и закрываем файл. */
    private fun stopRecording() {
        if (!isRecording) return
        isRecording = false

        writerHandler.post {
            try {
                writer?.flush()
                writer?.close()
            } catch (e: IOException) {
                // игнорируем — файл и так уже почти весь записан
            } finally {
                writer = null
            }
        }

        setRecButtons(recording = false)
        tvRecStatus.setTextColor(Color.DKGRAY)
        tvRecStatus.text =
            "Сохранено: $currentFileName\n" +
            "  строк: $rowCount   время: ${String.format(Locale.US, "%.1f", lastT)} с\n" +
            "  папка приложения (как достать — см. docs/hardware.md)"
    }

    /** Сформировать одну строку CSV и отправить её в поток записи. Вызывается из onSensorChanged (главный поток). */
    private fun writeCsvRow(eventTimestampNs: Long) {
        if (recStartNs == 0L) recStartNs = eventTimestampNs       // первый замер = t 0
        val t = (eventTimestampNs - recStartNs) / 1e9            // секунды от старта записи

        // высоту считаем из давления только если барометр уже дал данные (иначе была бы чепуха)
        val alt = if (gotPress)
            SensorManager.getAltitude(SensorManager.PRESSURE_STANDARD_ATMOSPHERE, pressVal)
        else 0f

        val line = String.format(
            Locale.US,
            "%.4f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
            t,
            accelVals[0], accelVals[1], accelVals[2],
            gyroVals[0], gyroVals[1], gyroVals[2],
            magVals[0], magVals[1], magVals[2],
            pressVal, alt,
            magCalVals[0], magCalVals[1], magCalVals[2]
        )

        rowCount++
        lastT = t.toFloat()

        // саму запись на диск делаем в фоновом потоке
        writerHandler.post {
            try {
                writer?.write(line)
                if (++pendingSinceFlush >= 200) {   // периодически сбрасываем буфер на диск (на случай сбоя)
                    writer?.flush()
                    pendingSinceFlush = 0
                }
            } catch (e: IOException) {
                // строку пропускаем; общий процесс не роняем
            }
        }
    }

    /** Кнопки: при записи активна «Стоп», иначе «Запись». */
    private fun setRecButtons(recording: Boolean) {
        btnRecord.isEnabled = !recording
        btnStop.isEnabled = recording
    }

    /** Строка состояния во время записи (тикает по таймеру). */
    private fun updateRecStatus() {
        tvRecStatus.setTextColor(Color.RED)
        tvRecStatus.text =
            "● ИДЁТ ЗАПИСЬ   ${String.format(Locale.US, "%.1f", lastT)} с   $rowCount строк\n" +
            "файл: $currentFileName"
    }

    // ==================================================================
    // BLUETOOTH-СТРИМИНГ (Фаза 3B): кнопка, разрешения, статус
    // ==================================================================

    /** Кнопка «Включить/Выключить стриминг». */
    private fun toggleStreaming() {
        if (StreamingService.running) {
            stopService(Intent(this, StreamingService::class.java))
            return
        }
        if (!ensureBtPermissions()) return        // спросим и продолжим в колбэке
        startStreamingChecked()
    }

    /** Разрешения стриминга: BLUETOOTH_CONNECT (12+) и POST_NOTIFICATIONS (13+). */
    private fun ensureBtPermissions(): Boolean {
        val need = ArrayList<String>()
        if (Build.VERSION.SDK_INT >= 31 &&
            ContextCompat.checkSelfPermission(this, Manifest.permission.BLUETOOTH_CONNECT)
            != PackageManager.PERMISSION_GRANTED) {
            need.add(Manifest.permission.BLUETOOTH_CONNECT)
        }
        if (Build.VERSION.SDK_INT >= 33 &&
            ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS)
            != PackageManager.PERMISSION_GRANTED) {
            need.add(Manifest.permission.POST_NOTIFICATIONS)
        }
        if (need.isEmpty()) return true
        ActivityCompat.requestPermissions(this, need.toTypedArray(), BT_PERM_CODE)
        return false
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == GPS_PERM_CODE) {
            // разрешение на геолокацию: если дали — сразу делаем разовый запрос
            if (hasLocationPermission()) requestGpsFix()
            else tvGpsStatus.text = "нет разрешения на геолокацию"
            return
        }
        if (requestCode != BT_PERM_CODE) return
        // BLUETOOTH_CONNECT обязателен; уведомления — желательны, но не блокируют
        val btOk = Build.VERSION.SDK_INT < 31 ||
                ContextCompat.checkSelfPermission(this, Manifest.permission.BLUETOOTH_CONNECT) ==
                PackageManager.PERMISSION_GRANTED
        if (btOk) startStreamingChecked()
        else tvStreamStatus.text = "нет разрешения Bluetooth («Устройства поблизости»)"
    }

    /** Bluetooth включён? Если нет — предложить включить; затем стартовать сервис. */
    private fun startStreamingChecked() {
        val adapter = getSystemService(BluetoothManager::class.java)?.adapter
        if (adapter == null) {
            tvStreamStatus.text = "Bluetooth недоступен на этом устройстве"
            return
        }
        if (!adapter.isEnabled) {
            try {
                startActivity(Intent(BluetoothAdapter.ACTION_REQUEST_ENABLE))
            } catch (e: SecurityException) {
                tvStreamStatus.text = "включите Bluetooth и нажмите ещё раз"
            }
            tvStreamStatus.text = "включите Bluetooth и нажмите ещё раз"
            return
        }
        startForegroundService(Intent(this, StreamingService::class.java))
    }

    /** Строка статуса стриминга (тикает по общему таймеру refresher). */
    private fun updateStreamStatus() {
        btnStreamToggle.text =
            if (StreamingService.running) "📡 Выключить стриминг" else "📡 Включить стриминг"
        if (!StreamingService.running) {
            if (tvStreamStatus.text.startsWith("нет разрешения") ||
                tvStreamStatus.text.startsWith("включите Bluetooth") ||
                tvStreamStatus.text.startsWith("Bluetooth недоступен")) return
            tvStreamStatus.setTextColor(Color.DKGRAY)
            tvStreamStatus.text = "выключено"
            return
        }
        val cl = StreamingService.clientName
        if (cl == null) {
            tvStreamStatus.setTextColor(Color.rgb(0xC0, 0x90, 0x10))
            tvStreamStatus.text = "ожидание подключения (служба «VarioPro3» видна ПК)"
        } else {
            tvStreamStatus.setTextColor(Color.rgb(0x2c, 0x7a, 0x2c))
            tvStreamStatus.text = String.format(
                Locale.US, "подключён: %s\nотправлено %d строк, ~%.0f Гц",
                cl, StreamingService.sentLines, StreamingService.rateHz)
        }
    }

    // ==================================================================
    // GPS-БЛОК: статус всегда виден, «Обновить» — разовый запрос позиции
    // ==================================================================

    private fun hasLocationPermission(): Boolean =
        ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) ==
                PackageManager.PERMISSION_GRANTED

    private fun isSystemLocationOn(): Boolean {
        val lm = getSystemService(LocationManager::class.java) ?: return false
        return LocationManagerCompat.isLocationEnabled(lm)
    }

    /** «Обновить»: разовый запрос позиции. Нет разрешения — сначала спросим его. */
    private fun requestGpsFix() {
        if (!isSystemLocationOn()) {
            tvGpsStatus.text = "GPS выключен системно — нажми «Включить GPS»"
            return
        }
        if (!hasLocationPermission()) {
            ActivityCompat.requestPermissions(
                this, arrayOf(Manifest.permission.ACCESS_FINE_LOCATION), GPS_PERM_CODE)
            return
        }
        gpsRequesting = true
        try {
            fusedClient?.getCurrentLocation(
                Priority.PRIORITY_HIGH_ACCURACY, CancellationTokenSource().token)
                ?.addOnSuccessListener { loc ->
                    gpsRequesting = false
                    if (loc != null) {
                        lastFix = loc
                        lastFixWallMs = SystemClock.elapsedRealtime()
                    } else {
                        tvGpsStatus.text = "позиция не получена (нет сигнала?) — повтори"
                    }
                }
                ?.addOnFailureListener {
                    gpsRequesting = false
                    tvGpsStatus.text = "ошибка запроса позиции: ${it.message}"
                }
        } catch (e: SecurityException) {
            gpsRequesting = false
            tvGpsStatus.text = "нет разрешения на геолокацию"
        }
    }

    /** Строка статуса GPS (тикает по общему таймеру refresher). Кнопки без
     *  дублирования: «Включить GPS» активна только когда геолокация выключена
     *  системно, «Обновить» — когда включена. */
    private fun updateGpsStatus() {
        val sysOn = isSystemLocationOn()
        btnGpsEnable.isEnabled = !sysOn
        btnGpsRefresh.isEnabled = sysOn
        if (!sysOn) {
            tvGpsStatus.setTextColor(Color.rgb(0xc0, 0x39, 0x2b))
            tvGpsStatus.text = "выключен системно («Включить GPS» → настройки)"
            return
        }
        if (!hasLocationPermission()) {
            tvGpsStatus.setTextColor(Color.rgb(0xC0, 0x90, 0x10))
            tvGpsStatus.text = "нет разрешения — нажми «Обновить», приложение спросит"
            return
        }
        val fix = lastFix
        val head = if (gpsRequesting) "активен, запрашиваю позицию…" else "активен"
        if (fix == null) {
            tvGpsStatus.setTextColor(Color.DKGRAY)
            tvGpsStatus.text = "$head\nфикса ещё нет — нажми «Обновить»"
            return
        }
        val age = (SystemClock.elapsedRealtime() - lastFixWallMs) / 1000
        val alt = if (fix.hasAltitude())
            String.format(Locale.US, "%.1f", fix.altitude) else "—"
        val acc = if (fix.hasAccuracy())
            String.format(Locale.US, "±%.1f", fix.accuracy) else ""
        tvGpsStatus.setTextColor(Color.rgb(0x2c, 0x7a, 0x2c))
        tvGpsStatus.text = String.format(Locale.US,
            "%s\nh = %s %s м   φ=%.5f λ=%.5f   %d с назад",
            head, alt, acc, fix.latitude, fix.longitude, age)
    }

    // ==================================================================
    // ЧАСТОТА И ТЕКСТ НА ЭКРАНЕ
    // ==================================================================

    /** Раз в ~секунду считаем фактическую частоту = число событий / прошедшее время. */
    private fun updateHz() {
        val now = System.nanoTime()
        val dt = (now - hzWindowStartNs) / 1e9
        if (dt >= 1.0) {
            accelHz = (accelCount / dt).toFloat(); accelCount = 0
            gyroHz = (gyroCount / dt).toFloat(); gyroCount = 0
            magHz = (magCount / dt).toFloat(); magCount = 0
            pressHz = (pressCount / dt).toFloat(); pressCount = 0
            hzWindowStartNs = now
        }
    }

    /** Собираем текст с характеристиками датчиков. */
    private fun buildSensorInfo(): String {
        val sb = StringBuilder()
        sb.append("Модель: ${Build.MODEL} (${Build.MANUFACTURER})\n\n")
        if (magSensor == null && magCalSensor == null) {
            sb.append("⚠ Магнитометра нет — работаем БЕЗ КОМПАСА\n")
            sb.append("  (вариометр и запись работают полностью)\n\n")
        }
        sb.append(describe(accelSensor, "Акселерометр (TYPE_ACCELEROMETER)"))
        sb.append(describe(gyroSensor, "Гироскоп (TYPE_GYROSCOPE)"))
        sb.append(describe(magSensor, "Магнитометр (TYPE_MAGNETIC_FIELD_UNCALIBRATED)"))
        sb.append(describe(magCalSensor, "Магнитометр Android-калибр. (TYPE_MAGNETIC_FIELD)"))
        sb.append(describe(pressSensor, "Барометр (TYPE_PRESSURE)"))
        // пакет 15 (З.3): честная проверка датчика температуры среды — на S23
        // его через публичный API обычно нет («не найден» = TEMP в поток не идёт)
        sb.append(describe(
            sensorManager.getDefaultSensor(Sensor.TYPE_AMBIENT_TEMPERATURE),
            "Температура среды (TYPE_AMBIENT_TEMPERATURE)"))
        return sb.toString().trimEnd()
    }

    /** Характеристики одного датчика (как их отдаёт Android из даташита). */
    private fun describe(s: Sensor?, title: String): String {
        if (s == null) return "$title:\n  не найден\n\n"
        val maxHz = if (s.minDelay > 0) 1_000_000.0 / s.minDelay else 0.0
        val maxHzStr =
            if (maxHz > 0) String.format(Locale.US, " (макс %.0f Гц)", maxHz) else " (по изменению)"
        val sb = StringBuilder()
        sb.append("$title:\n")
        sb.append("  name       : ${s.name}\n")
        sb.append("  vendor     : ${s.vendor}\n")
        sb.append("  resolution : ${s.resolution}\n")
        sb.append("  maxRange   : ${s.maximumRange}\n")
        sb.append("  minDelay   : ${s.minDelay} мкс$maxHzStr\n")
        sb.append("  power      : ${s.power} мА\n\n")
        return sb.toString()
    }

    /** Собираем текст с живыми значениями датчиков. */
    private fun buildLiveText(): String {
        val sb = StringBuilder()

        sb.append("АКСЕЛЕРОМЕТР  ")
        when {
            accelSensor == null -> sb.append("— не найден\n\n")
            !gotAccel -> sb.append("(ожидание…)\n\n")
            else -> {
                sb.append("f=${hz(accelHz)}\n")
                sb.append(" ax = ${v(accelVals[0])} м/с²\n")
                sb.append(" ay = ${v(accelVals[1])} м/с²\n")
                sb.append(" az = ${v(accelVals[2])} м/с²\n\n")
            }
        }

        sb.append("ГИРОСКОП  ")
        when {
            gyroSensor == null -> sb.append("— не найден\n\n")
            !gotGyro -> sb.append("(ожидание…)\n\n")
            else -> {
                sb.append("f=${hz(gyroHz)}\n")
                sb.append(" gx = ${v(gyroVals[0])} рад/с\n")
                sb.append(" gy = ${v(gyroVals[1])} рад/с\n")
                sb.append(" gz = ${v(gyroVals[2])} рад/с\n\n")
            }
        }

        sb.append("МАГНИТОМЕТР (сырое, uncalib)  ")
        when {
            magSensor == null -> sb.append("— не найден (без компаса)\n\n")
            !gotMag -> sb.append("(ожидание…)\n\n")
            else -> {
                sb.append("f=${hz(magHz)}\n")
                sb.append(" mx = ${v(magVals[0])} мкТл\n")
                sb.append(" my = ${v(magVals[1])} мкТл\n")
                sb.append(" mz = ${v(magVals[2])} мкТл\n")
                if (magCalSensor != null) {
                    sb.append(" Android-калибр. (mxa,mya,mza):\n")
                    sb.append(" ${v(magCalVals[0])} ${v(magCalVals[1])} ${v(magCalVals[2])} мкТл\n\n")
                } else sb.append("\n")
            }
        }

        sb.append("БАРОМЕТР  ")
        when {
            pressSensor == null -> sb.append("— не найден\n")
            !gotPress -> sb.append("(ожидание…)\n")
            else -> {
                val alt = SensorManager.getAltitude(
                    SensorManager.PRESSURE_STANDARD_ATMOSPHERE, pressVal
                )
                sb.append("f=${hz(pressHz)}\n")
                sb.append(" давление = ${String.format(Locale.US, "%.2f", pressVal)} гПа\n")
                sb.append(" высота   = ${String.format(Locale.US, "%+.1f", alt)} м  (QNH 1013.25)\n")
            }
        }

        return sb.toString().trimEnd()
    }

    /** Формат числа: знак + 3 знака после запятой, ширина 8 (для ровных столбцов). */
    private fun v(x: Float) = String.format(Locale.US, "%+8.3f", x)

    /** Формат частоты: целое число + "Гц". */
    private fun hz(x: Float) = String.format(Locale.US, "%.0f Гц", x)
}
