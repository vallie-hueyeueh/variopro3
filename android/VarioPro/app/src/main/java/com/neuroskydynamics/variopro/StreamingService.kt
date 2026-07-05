package com.neuroskydynamics.variopro

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.bluetooth.BluetoothManager
import android.bluetooth.BluetoothServerSocket
import android.bluetooth.BluetoothSocket
import android.content.Intent
import android.content.pm.ServiceInfo
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.Manifest
import android.content.pm.PackageManager
import android.os.Handler
import android.os.HandlerThread
import android.os.IBinder
import android.os.Looper
import android.os.SystemClock
import android.util.Base64
import androidx.core.app.NotificationCompat
import androidx.core.app.ServiceCompat
import androidx.core.content.ContextCompat
import com.google.android.gms.location.FusedLocationProviderClient
import com.google.android.gms.location.LocationCallback
import com.google.android.gms.location.LocationRequest
import com.google.android.gms.location.LocationResult
import com.google.android.gms.location.LocationServices
import com.google.android.gms.location.Priority
import java.io.BufferedOutputStream
import java.io.File
import java.io.IOException
import java.util.Locale
import java.util.UUID
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.TimeUnit
import java.util.zip.CRC32

/**
 * VarioPro — Фаза 3, шаг 3B: BLUETOOTH SPP-СТРИМИНГ (телефон = СЕРВЕР).
 *
 * Foreground-сервис (живёт при погашенном экране):
 *   • поднимает RFCOMM-сервер listenUsingRfcommWithServiceRecord("VarioPro3", SPP UUID);
 *     служба «VarioPro3» видна Windows ТОЛЬКО пока стриминг включён;
 *   • по подключению шлёт протокол docs/stream_protocol.md:
 *       HELLO …\n  затем строки  seq,t_send,t,ax,ay,az,gx,gy,gz,mx,my,mz,pressure,altitude\n
 *     t_send = System.currentTimeMillis()/1000.0 (часы телефона в момент отправки);
 *   • сам подписывается на датчики (не зависит от экрана приложения); локальная
 *     запись CSV в MainActivity работает ОДНОВРЕМЕННО и остаётся «истиной»;
 *   • IMU прореживается до ~100 Гц (каждый DECIMATE-й сэмпл акселерометра);
 *   • после разрыва снова слушает; при переподключении seq и t идут с нуля
 *     (ПК-читатель это умеет — ось времени останется монотонной).
 *
 * Статус для UI — через companion-поля (@Volatile): MainActivity читает их
 * своим таймером (без биндинга и бродкастов — просто и надёжно).
 */
class StreamingService : Service(), SensorEventListener {

    companion object {
        val SPP_UUID: UUID = UUID.fromString("00001101-0000-1000-8000-00805F9B34FB")
        const val SERVICE_NAME = "VarioPro3"      // имя SDP-службы (видно в Windows)
        const val CHANNEL_ID = "variopro_stream"
        const val NOTIF_ID = 42

        /** Прореживание IMU: слать каждый N-й сэмпл акселерометра.
         *  Аксель на S23 ~417 Гц → 417/4 ≈ 104 Гц. SPP тянет это с запасом. */
        const val DECIMATE = 4

        /** Номинальная частота потока (объявляется в HELLO rate=…): ~417/DECIMATE. */
        const val NOMINAL_HZ = 104

        /** Колонки v4: mx..mz — СЫРОЕ поле (для калибровки, как раньше),
         *  mxa..mza — Android-калиброванное (для компаса). Старые ПК-читатели
         *  лишние поля игнорируют (правило протокола). */
        const val COLS = "seq,t_send,t,ax,ay,az,gx,gy,gz,mx,my,mz,pressure,altitude,mxa,mya,mza"

        /** HELLO v4: + model (модель устройства, пробелы → '_'). Токен
         *  mag_source убран (пакет 14, Б.2): телефон всегда шлёт ОБА поля,
         *  источник для компаса выбирает ПК; старые ПК-читатели отсутствие
         *  токена игнорируют (правило протокола). */
        fun hello(): String =
            "HELLO variopro-stream 4 rate=$NOMINAL_HZ " +
                    "model=${android.os.Build.MODEL.replace(' ', '_')} cols=$COLS\n"

        /** SENSORS (пакет 15, З.1): метаданные датчиков для ПК — по строке на
         *  датчик, сразу ПОСЛЕ каждого HELLO (HELLO повторяется ~5 с — повтор
         *  SENSORS безопасен, протокол разрешает переобработку строк).
         *  Формат: SENSORS,<ключ>,<name>,<vendor>,<resolution>,<maxRange>,<minDelayUs>
         *  Запятые в name/vendor заменяются на ';' (поля не ломают CSV). */
        private fun sensorLine(key: String, s: Sensor?): String {
            if (s == null) return "SENSORS,$key,-,-,-,-,-\n"
            fun clean(x: String) = x.replace(',', ';')
            return "SENSORS,$key,${clean(s.name)},${clean(s.vendor)}," +
                    "${s.resolution},${s.maximumRange},${s.minDelay}\n"
        }

        // --- живой статус для главного экрана ---
        @Volatile var running = false             // сервис работает (кнопка «вкл»)
        @Volatile var status = "выключено"        // человекочитаемое состояние
        @Volatile var clientName: String? = null  // имя подключённого ПК (или null)
        @Volatile var sentLines = 0L              // отправлено строк за это подключение
        @Volatile var rateHz = 0f                 // реальная частота отправки, Гц
    }

    // --- датчики (свои подписки, независимо от MainActivity) ---
    private lateinit var sensorManager: SensorManager
    private var accelSensor: Sensor? = null
    private var gyroSensor: Sensor? = null
    private var magSensor: Sensor? = null         // UNCALIBRATED — сырое поле, как в CSV
    private var magCalSensor: Sensor? = null      // TYPE_MAGNETIC_FIELD — колонки mxa..mza (v4)
    private var pressSensor: Sensor? = null
    private var tempSensor: Sensor? = null        // TYPE_AMBIENT_TEMPERATURE (пакет 15, З.3;
                                                  // на S23 обычно ОТСУТСТВУЕТ — тогда TEMP не шлётся)
    private val accelV = FloatArray(3)
    private val gyroV = FloatArray(3)
    private val magV = FloatArray(3)
    private val magCalV = FloatArray(3)
    private var pressVal = 0f
    private var gotPress = false
    private var tempVal = Float.NaN               // последняя температура, °C
    private var lastTempSentNs = 0L               // TEMP шлём раз в ~5 с
    private var accelCount = 0                    // для прореживания (каждый DECIMATE-й)
    private var sensorThread: HandlerThread? = null

    // --- поток-сервер и очередь строк на отправку ---
    @Volatile private var alive = false           // сервис жив (для циклов)
    @Volatile private var connected = false       // клиент подключён (формировать строки)
    @Volatile private var sendingFile = false     // идёт GET: поток данных приостановлен

    // --- GPS: последний фикс (шлётся строкой GPS,… раз в ~1 с) ---
    private var fusedClient: FusedLocationProviderClient? = null
    @Volatile private var gpsLat = Double.NaN
    @Volatile private var gpsLon = Double.NaN
    @Volatile private var gpsAlt = Double.NaN
    @Volatile private var gpsAcc = Double.NaN
    private var lastGpsSentNs = 0L
    private val locationCallback = object : LocationCallback() {
        override fun onLocationResult(result: LocationResult) {
            val loc = result.lastLocation ?: return
            gpsLat = loc.latitude
            gpsLon = loc.longitude
            gpsAlt = if (loc.hasAltitude()) loc.altitude else Double.NaN
            gpsAcc = if (loc.hasAccuracy()) loc.accuracy.toDouble() else Double.NaN
        }
    }
    private var seq = 0L
    private var startNs = 0L                      // t данных = (timestamp − startNs)/1e9
    private val queue = LinkedBlockingQueue<String>(4096)
    private var serverThread: Thread? = null
    @Volatile private var serverSock: BluetoothServerSocket? = null
    @Volatile private var clientSock: BluetoothSocket? = null

    // частота отправки: счётчик в окне ~2 с
    private var rateCount = 0
    private var rateWindowStartNs = 0L

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        sensorManager = getSystemService(SensorManager::class.java)
        accelSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        gyroSensor = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
        magSensor = sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD_UNCALIBRATED)
        magCalSensor = sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD)
        pressSensor = sensorManager.getDefaultSensor(Sensor.TYPE_PRESSURE)
        // Температура окружающей среды (пакет 15, З.3): на S23 датчика через
        // публичный API обычно НЕТ (getDefaultSensor вернёт null) — тогда
        // строки TEMP просто не шлются; на устройствах с датчиком — раз в ~5 с
        tempSensor = sensorManager.getDefaultSensor(Sensor.TYPE_AMBIENT_TEMPERATURE)
    }

    /** Блок строк SENSORS (З.1) — сразу после HELLO. */
    private fun sensorsBlock(): String =
        sensorLine("acc", accelSensor) +
                sensorLine("gyro", gyroSensor) +
                sensorLine("mag", magSensor) +
                sensorLine("maga", magCalSensor) +
                sensorLine("baro", pressSensor) +
                sensorLine("temp", tempSensor)

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (running) return START_STICKY          // уже работаем
        running = true
        alive = true
        sentLines = 0
        rateHz = 0f
        status = "запуск…"

        createChannel()
        // foreground с типом connectedDevice (обязателен для targetSdk 34+)
        ServiceCompat.startForeground(
            this, NOTIF_ID, buildNotification("ожидание подключения"),
            ServiceInfo.FOREGROUND_SERVICE_TYPE_CONNECTED_DEVICE)

        // датчики — на своём HandlerThread (не грузим главный поток)
        sensorThread = HandlerThread("stream-sensors").apply { start() }
        val h = Handler(sensorThread!!.looper)
        accelSensor?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_FASTEST, h) }
        gyroSensor?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_FASTEST, h) }
        magSensor?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_FASTEST, h) }
        magCalSensor?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_FASTEST, h) }
        pressSensor?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_FASTEST, h) }
        tempSensor?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_NORMAL, h) }

        // GPS раз в ~1 с (если разрешение на геолокацию выдано; иначе просто без GPS)
        fusedClient = LocationServices.getFusedLocationProviderClient(this)
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION)
            == PackageManager.PERMISSION_GRANTED) {
            try {
                val req = LocationRequest.Builder(Priority.PRIORITY_HIGH_ACCURACY, 1000L).build()
                fusedClient?.requestLocationUpdates(req, locationCallback, Looper.getMainLooper())
            } catch (e: SecurityException) { /* без GPS */ }
        }

        serverThread = Thread({ serverLoop() }, "stream-server").apply {
            isDaemon = true
            start()
        }
        return START_STICKY
    }

    // ------------------------------------------------------------------
    // RFCOMM-СЕРВЕР: слушаем → шлём → при разрыве снова слушаем
    // ------------------------------------------------------------------
    private fun serverLoop() {
        val adapter = getSystemService(BluetoothManager::class.java)?.adapter
        if (adapter == null) {
            status = "Bluetooth недоступен"
            return
        }
        while (alive) {
            var server: BluetoothServerSocket? = null
            var sock: BluetoothSocket? = null
            try {
                status = "ожидание подключения"
                clientName = null
                updateNotification("ожидание подключения")
                server = adapter.listenUsingRfcommWithServiceRecord(SERVICE_NAME, SPP_UUID)
                serverSock = server
                sock = server.accept()            // блокирует до подключения ПК
                clientSock = sock
                server.close()                    // одно подключение за раз
                serverSock = null

                // новое подключение = новая сессия: seq и t с нуля, очередь чистая
                queue.clear()
                seq = 0
                startNs = 0L
                sentLines = 0
                rateCount = 0
                rateWindowStartNs = 0L
                val name = try {
                    sock.remoteDevice?.name ?: sock.remoteDevice?.address ?: "?"
                } catch (e: SecurityException) { "?" }
                clientName = name
                status = "подключён: $name"
                updateNotification("подключён: $name")

                // HELLO v4 (без mag_source — пакет 14, Б.2: оба поля идут всегда)
                // + SENSORS (пакет 15, З.1): метаданные датчиков сразу после HELLO
                val helloLine = hello() + sensorsBlock()
                val out = BufferedOutputStream(sock.outputStream, 8192)
                val writeLock = Object()          // запись из двух потоков: данные + PONG
                synchronized(writeLock) {
                    out.write(helloLine.toByteArray(Charsets.UTF_8))
                    out.flush()
                }
                connected = true

                // ЧИТАТЕЛЬ входящих строк от ПК: PING → PONG немедленно (часы);
                // LIST/GET/DEL — менеджер записей (протокол v3). Неизвестное — игнор.
                val inStream = sock.inputStream
                val reader = Thread({
                    fun send(s: String) {
                        synchronized(writeLock) {
                            out.write(s.toByteArray(Charsets.UTF_8))
                            out.flush()
                        }
                    }
                    fun safeName(n: String): String? {
                        val t = n.trim()
                        if (t.isEmpty() || t.contains("/") || t.contains("\\")
                            || t.contains("..")) return null
                        return t
                    }
                    val dir: File = getExternalFilesDir(null) ?: filesDir
                    try {
                        val br = inStream.bufferedReader(Charsets.UTF_8)
                        while (alive) {
                            val ln = br.readLine() ?: break
                            val p = ln.trim().split(",")
                            when {
                                p.size >= 3 && p[0] == "PING" -> {
                                    val tMine = System.currentTimeMillis() / 1000.0
                                    send(String.format(Locale.US,
                                        "PONG,%s,%s,%.6f\n", p[1], p[2], tMine))
                                }
                                p[0] == "LIST" -> {
                                    val files = (dir.listFiles() ?: arrayOf())
                                        .filter { it.isFile && (it.name.endsWith(".csv")
                                                || it.name.endsWith(".json")) }
                                        .sortedBy { it.name }
                                    send("FILES,${files.size}\n")
                                    for (f in files) {
                                        // время СОЗДАНИЯ (пакет 13, Д.1): lastModified
                                        // у записи = КОНЕЦ записи; создание честнее
                                        // (если ФС не хранит creationTime — Android
                                        // сам вернёт lastModified)
                                        val created = try {
                                            java.nio.file.Files.readAttributes(
                                                f.toPath(),
                                                java.nio.file.attribute.BasicFileAttributes::class.java
                                            ).creationTime().toMillis() / 1000
                                        } catch (e: Exception) {
                                            f.lastModified() / 1000
                                        }
                                        send("FILE,${f.name},${f.length()},$created\n")
                                    }
                                }
                                p.size >= 2 && p[0] == "GET" -> {
                                    val name = safeName(p[1])
                                    val f = if (name != null) File(dir, name) else null
                                    if (f == null || !f.isFile) {
                                        send("ERR,нет такого файла\n")
                                    } else {
                                        sendingFile = true   // поток данных на паузу
                                        try {
                                            val data = f.readBytes()
                                            send("FILESTART,${f.name},${data.size}\n")
                                            var i = 0
                                            while (i < data.size) {
                                                val n = minOf(3072, data.size - i)
                                                val b64 = Base64.encodeToString(
                                                    data, i, n, Base64.NO_WRAP)
                                                send("B64,$b64\n")
                                                i += n
                                            }
                                            val crc = CRC32()
                                            crc.update(data)
                                            send("FILEEND,${crc.value}\n")
                                        } catch (e: Exception) {
                                            send("ERR,${e.message ?: "чтение файла"}\n")
                                        } finally {
                                            sendingFile = false
                                        }
                                    }
                                }
                                p.size >= 2 && p[0] == "DEL" -> {
                                    val name = safeName(p[1])
                                    val f = if (name != null) File(dir, name) else null
                                    when {
                                        f == null || !f.isFile -> send("ERR,нет такого файла\n")
                                        f.delete() -> send("OK,DEL\n")
                                        else -> send("ERR,не удалось удалить\n")
                                    }
                                }
                                // прочее молча игнорируем
                            }
                        }
                    } catch (e: IOException) {
                        // разрыв — писательский цикл заметит сам
                    }
                }, "stream-cmd").apply { isDaemon = true; start() }

                // писательский цикл: берём строки из очереди, flush когда очередь пуста.
                // HELLO повторяем раз в ~5 с: приёмник (pyserial) при открытии порта
                // чистит входной буфер, и первый HELLO может погибнуть в этой гонке —
                // без него ПК не знает номинал частоты (не считает «Качество связи»).
                // Повтор безопасен: протокол игнорирует/переобрабатывает любые строки.
                var lastHello = SystemClock.elapsedRealtime()
                while (alive) {
                    if (SystemClock.elapsedRealtime() - lastHello >= 5000) {
                        lastHello = SystemClock.elapsedRealtime()
                        synchronized(writeLock) {
                            out.write(helloLine.toByteArray(Charsets.UTF_8))
                            out.flush()
                        }
                    }
                    val line = queue.poll(500, TimeUnit.MILLISECONDS)
                    if (line == null) {
                        synchronized(writeLock) { out.flush() }   // тихо — доталкиваем буфер
                        continue
                    }
                    synchronized(writeLock) {
                        out.write(line.toByteArray(Charsets.UTF_8))
                        if (queue.isEmpty()) out.flush()
                    }
                    sentLines++
                }
            } catch (e: IOException) {
                // разрыв связи или закрытие сокета при выключении — норма
            } catch (e: SecurityException) {
                status = "нет разрешения Bluetooth"
                break
            } catch (e: InterruptedException) {
                break
            } finally {
                connected = false
                try { sock?.close() } catch (e: IOException) {}
                try { server?.close() } catch (e: IOException) {}
                clientSock = null
                serverSock = null
            }
            if (alive) SystemClock.sleep(300)     // пауза и снова слушаем
        }
    }

    // ------------------------------------------------------------------
    // ДАТЧИКИ: строка на каждый DECIMATE-й сэмпл акселерометра
    // ------------------------------------------------------------------
    override fun onSensorChanged(event: SensorEvent) {
        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> {
                System.arraycopy(event.values, 0, accelV, 0, 3)
                if (!connected) return            // без клиента строки не формируем
                if (sendingFile) return           // идёт GET: данные на паузе (seq не тратится)
                accelCount++
                if (accelCount % DECIMATE != 0) return
                if (startNs == 0L) startNs = event.timestamp
                val t = (event.timestamp - startNs) / 1e9
                val tSend = System.currentTimeMillis() / 1000.0   // часы телефона
                val alt = if (gotPress)
                    SensorManager.getAltitude(SensorManager.PRESSURE_STANDARD_ATMOSPHERE, pressVal)
                else 0f
                val line = String.format(Locale.US,
                    "%d,%.6f,%.4f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                    seq, tSend, t,
                    accelV[0], accelV[1], accelV[2],
                    gyroV[0], gyroV[1], gyroV[2],
                    magV[0], magV[1], magV[2],
                    pressVal, alt,
                    magCalV[0], magCalV[1], magCalV[2])
                seq++                              // seq расходуется всегда:
                if (!queue.offer(line)) {          // очередь полна (клиент завис) —
                    queue.poll()                   // выбрасываем старую строку:
                    queue.offer(line)              // дырка в seq = честная «потеря»
                }
                // реальная частота (окно ~2 с)
                rateCount++
                if (rateWindowStartNs == 0L) rateWindowStartNs = event.timestamp
                val dt = (event.timestamp - rateWindowStartNs) / 1e9
                if (dt >= 2.0) {
                    rateHz = (rateCount / dt).toFloat()
                    rateCount = 0
                    rateWindowStartNs = event.timestamp
                }
                // GPS-строка раз в ~1 с (тем же каналом; в фильтр ПК не идёт — показ)
                if (!gpsLat.isNaN() && event.timestamp - lastGpsSentNs >= 1_000_000_000L) {
                    lastGpsSentNs = event.timestamp
                    queue.offer(String.format(Locale.US,
                        "GPS,%.4f,%.6f,%.6f,%.1f,%.1f\n",
                        t, gpsLat, gpsLon,
                        if (gpsAlt.isNaN()) 0.0 else gpsAlt,
                        if (gpsAcc.isNaN()) 0.0 else gpsAcc))
                }
                // TEMP раз в ~5 с (пакет 15, З.3) — только если датчик есть и
                // дал данные; ПК логирует/показывает, в фильтр НЕ вводит
                if (!tempVal.isNaN() && event.timestamp - lastTempSentNs >= 5_000_000_000L) {
                    lastTempSentNs = event.timestamp
                    queue.offer(String.format(Locale.US, "TEMP,%.4f,%.1f\n", t, tempVal))
                }
            }
            Sensor.TYPE_GYROSCOPE ->
                System.arraycopy(event.values, 0, gyroV, 0, 3)
            Sensor.TYPE_MAGNETIC_FIELD_UNCALIBRATED ->
                System.arraycopy(event.values, 0, magV, 0, 3)   // values[0..2] = сырое поле
            Sensor.TYPE_MAGNETIC_FIELD ->
                System.arraycopy(event.values, 0, magCalV, 0, 3)  // Android-калиброванное (v4)
            Sensor.TYPE_PRESSURE -> {
                pressVal = event.values[0]
                gotPress = true
            }
            Sensor.TYPE_AMBIENT_TEMPERATURE ->
                tempVal = event.values[0]         // °C (З.3; на S23 не приходит)
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) { /* не используем */ }

    // ------------------------------------------------------------------
    // ВЫКЛЮЧЕНИЕ
    // ------------------------------------------------------------------
    override fun onDestroy() {
        alive = false
        connected = false
        // закрыть сокеты — это выбьет accept()/write() из блокировки
        try { serverSock?.close() } catch (e: IOException) {}
        try { clientSock?.close() } catch (e: IOException) {}
        serverThread?.interrupt()
        sensorManager.unregisterListener(this)
        sensorThread?.quitSafely()
        fusedClient?.removeLocationUpdates(locationCallback)
        running = false
        status = "выключено"
        clientName = null
        rateHz = 0f
        super.onDestroy()
    }

    // ------------------------------------------------------------------
    // УВЕДОМЛЕНИЕ foreground-сервиса
    // ------------------------------------------------------------------
    private fun createChannel() {
        val ch = NotificationChannel(
            CHANNEL_ID, "Bluetooth-стриминг",
            NotificationManager.IMPORTANCE_LOW)   // тихое, без звука
        ch.description = "Передача данных датчиков на ПК"
        getSystemService(NotificationManager::class.java).createNotificationChannel(ch)
    }

    private fun buildNotification(text: String): Notification =
        NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.stat_sys_data_bluetooth)
            .setContentTitle("VarioPro: стриминг на ПК")
            .setContentText(text)
            .setOngoing(true)
            .build()

    private fun updateNotification(text: String) {
        getSystemService(NotificationManager::class.java)
            .notify(NOTIF_ID, buildNotification(text))
    }
}
