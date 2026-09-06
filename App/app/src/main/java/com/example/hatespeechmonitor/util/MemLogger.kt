package com.example.hatespeechmonitor.util

import android.content.Context
import android.os.Debug
import java.io.File

/**
 * Lightweight memory profiler for Test 2 (system aspects). Appends a timestamped
 * sample to filesDir/mem_profile.csv at tagged points (app start, model loads, each
 * training epoch), so we can plot RAM-vs-time and see the model-load spike vs. the
 * (cheap) per-epoch training cost. Pull the CSV with:
 *   adb exec-out run-as com.example.hatespeechmonitor cat files/mem_profile.csv > mem_profile.csv
 */
object MemLogger {
    private var file: File? = null
    private var t0 = 0L

    /** Start a fresh profile file. Call once, early in Application.onCreate(). */
    fun init(context: Context) {
        file = File(context.filesDir, "mem_profile.csv")
            .apply { writeText("tag,t_ms,cpu_ms,java_heap_kb,native_heap_kb,total_pss_kb\n") }
        t0 = System.currentTimeMillis()
    }

    /** Record one memory + CPU sample under [tag]. No-op if [init] wasn't called. */
    @Synchronized
    fun sample(tag: String) {
        val f = file ?: return
        val rt = Runtime.getRuntime()
        val cpuMs = android.os.Process.getElapsedCpuTime()   // process CPU time (all threads), ms
        val javaKb = (rt.totalMemory() - rt.freeMemory()) / 1024
        val nativeKb = Debug.getNativeHeapAllocatedSize() / 1024
        val pssKb = Debug.getPss().toLong()   // total app RAM footprint (PSS), KB
        val wall = System.currentTimeMillis() - t0
        f.appendText("$tag,$wall,$cpuMs,$javaKb,$nativeKb,$pssKb\n")
    }
}
