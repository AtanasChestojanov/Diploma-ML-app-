package com.example.hatespeechmonitor.personalization

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Persists the trainable head's weights blob (raw little-endian float32) to a file in
 * app-internal storage.
 */
class HeadWeightsStorage(private val file: File) {

    val path: String get() = file.absolutePath
    fun exists(): Boolean = file.exists()
    fun sizeBytes(): Long = if (file.exists()) file.length() else 0L

    suspend fun save(weights: FloatArray) = withContext(Dispatchers.IO) {
        val bb = ByteBuffer.allocate(weights.size * 4).order(ByteOrder.LITTLE_ENDIAN)
        bb.asFloatBuffer().put(weights)
        file.parentFile?.mkdirs()
        // write-and-rename for atomic replacement
        val tmp = File(file.parentFile, file.name + ".tmp")
        tmp.writeBytes(bb.array())
        if (!tmp.renameTo(file)) {
            file.delete()
            require(tmp.renameTo(file)) { "Could not rename $tmp -> $file" }
        }
    }

    suspend fun load(expectedSize: Int): FloatArray? = withContext(Dispatchers.IO) {
        if (!file.exists()) return@withContext null
        val bytes = file.readBytes()
        if (bytes.size != expectedSize * 4) return@withContext null
        val fb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        FloatArray(expectedSize).also { fb.get(it) }
    }

    suspend fun delete(): Boolean = withContext(Dispatchers.IO) {
        if (file.exists()) file.delete() else true
    }
}
