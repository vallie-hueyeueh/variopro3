package com.neuroskydynamics.variopro

import android.content.Intent
import android.os.Bundle
import android.widget.Button
import android.widget.ListView
import android.widget.SimpleAdapter
import android.widget.TextView
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.FileProvider
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Экран «ФАЙЛЫ»: записи датчиков (session_*.csv) и калибровки (calib_*.json)
 * из папки приложения. Дата и размер у каждого файла; сортировка по дате или
 * размеру (кнопка-переключатель); удаление с подтверждением; «Поделиться» —
 * системное окно share (файл уходит через FileProvider: почта, телега, диск…).
 *
 * Файлы можно забирать и по Bluetooth с ПК (вкладка «Записи» пульта) — этот
 * экран нужен как запасной путь и для чистки памяти прямо на телефоне.
 */
class FilesActivity : AppCompatActivity() {

    private lateinit var list: ListView
    private lateinit var tvStatus: TextView
    private lateinit var btnSort: Button
    private lateinit var btnShare: Button
    private lateinit var btnDelete: Button

    private var files: List<File> = emptyList()   // в порядке показа
    private var sortByDate = true                  // true: по дате, false: по размеру
    private var selected = -1                      // выбранная строка (-1 = нет)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_files)
        // отступ под статус-бар/вырез камеры (блок Е.1) — как на других экранах
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.filesRoot)) { v, insets ->
            val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(bars.left, bars.top, bars.right, bars.bottom)
            insets
        }
        title = "Файлы записей"

        list = findViewById(R.id.listFiles)
        tvStatus = findViewById(R.id.tvFilesStatus)
        btnSort = findViewById(R.id.btnSort)
        btnShare = findViewById(R.id.btnShare)
        btnDelete = findViewById(R.id.btnDelete)

        list.choiceMode = ListView.CHOICE_MODE_SINGLE
        list.setOnItemClickListener { _, _, pos, _ ->
            selected = pos
            updateButtons()
        }
        btnSort.setOnClickListener {
            sortByDate = !sortByDate
            refresh()
        }
        btnShare.setOnClickListener { shareSelected() }
        btnDelete.setOnClickListener { deleteSelected() }
    }

    override fun onResume() {
        super.onResume()
        refresh()
    }

    /** Перечитать папку приложения и показать список. */
    private fun refresh() {
        val dir: File = getExternalFilesDir(null) ?: filesDir
        val all = (dir.listFiles() ?: arrayOf())
            .filter {
                it.isFile && ((it.name.startsWith("session_") && it.name.endsWith(".csv"))
                        || (it.name.startsWith("calib_") && it.name.endsWith(".json")))
            }
        files = if (sortByDate) all.sortedByDescending { it.lastModified() }
                else all.sortedByDescending { it.length() }

        val fmt = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US)
        val rows = files.map {
            mapOf(
                "name" to it.name,
                "info" to "${fmt.format(Date(it.lastModified()))}    ${fmtSize(it.length())}"
            )
        }
        list.adapter = SimpleAdapter(
            this, rows, android.R.layout.simple_list_item_activated_2,
            arrayOf("name", "info"), intArrayOf(android.R.id.text1, android.R.id.text2))

        selected = -1
        list.clearChoices()
        btnSort.text = if (sortByDate) "Сортировка: по дате ▾" else "Сортировка: по размеру ▾"
        tvStatus.text = if (files.isEmpty())
            "Файлов пока нет. Запись создаёт session_*.csv, калибровка — calib_*.json."
        else
            "${files.size} файл(ов). Нажми на файл, затем «Поделиться» или «Удалить»."
        updateButtons()
    }

    private fun updateButtons() {
        val has = selected in files.indices
        btnShare.isEnabled = has
        btnDelete.isEnabled = has
    }

    private fun fmtSize(n: Long): String = when {
        n >= 1 shl 20 -> String.format(Locale.US, "%.1f МБ", n / 1048576.0)
        n >= 1 shl 10 -> String.format(Locale.US, "%.0f КБ", n / 1024.0)
        else -> "$n Б"
    }

    /** «Поделиться»: системное окно share через FileProvider (почта/мессенджер/диск). */
    private fun shareSelected() {
        val f = files.getOrNull(selected) ?: return
        try {
            val uri = FileProvider.getUriForFile(
                this, "$packageName.fileprovider", f)
            val send = Intent(Intent.ACTION_SEND).apply {
                type = if (f.name.endsWith(".json")) "application/json" else "text/csv"
                putExtra(Intent.EXTRA_STREAM, uri)
                putExtra(Intent.EXTRA_SUBJECT, f.name)
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            }
            startActivity(Intent.createChooser(send, "Отправить ${f.name}"))
        } catch (e: Exception) {
            tvStatus.text = "Не удалось поделиться: ${e.message}"
        }
    }

    /** «Удалить»: с подтверждением, файл удаляется безвозвратно. */
    private fun deleteSelected() {
        val f = files.getOrNull(selected) ?: return
        AlertDialog.Builder(this)
            .setTitle("Удалить файл?")
            .setMessage("Удалить ${f.name} с телефона безвозвратно?")
            .setPositiveButton("Удалить") { _, _ ->
                if (f.delete()) {
                    tvStatus.text = "Удалён: ${f.name}"
                    refresh()
                } else {
                    tvStatus.text = "Не удалось удалить ${f.name}"
                }
            }
            .setNegativeButton("Отмена", null)
            .show()
    }
}
