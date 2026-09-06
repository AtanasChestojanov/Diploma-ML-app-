package com.example.hatespeechmonitor.personalization

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.FloatBuffer
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class HeadTrainer(
    context: Context,
    modelAssetName: String = MODEL_ASSET,
    val numThreads: Int = DEFAULT_NUM_THREADS,
) : AutoCloseable {

    private val mappedModel: MappedByteBuffer
    private val interpreter: Interpreter
    val modelSizeBytes: Long
    val embeddingDim: Int
    val numClasses: Int
    val weightsSize: Int

    private val lock = Any()

    @Volatile private var closed = false

    init {
        val afd = context.assets.openFd(modelAssetName)
        modelSizeBytes = afd.declaredLength
        mappedModel = FileInputStream(afd.fileDescriptor).use { fis ->
            fis.channel.map(FileChannel.MapMode.READ_ONLY, afd.startOffset, afd.declaredLength)
        }
        afd.close()

        val options = Interpreter.Options().apply { setNumThreads(numThreads) }
        interpreter = Interpreter(mappedModel, options)

        val keys = interpreter.signatureKeys.toSet()
        for (k in REQUIRED_SIGNATURES) {
            require(k in keys) { "HeadTrainer: missing signature '$k'. Available: $keys" }
        }

        // shape contract (the static-dim columns are read from .shape(); the dynamic batch
        // dim is whatever runSignature will set per call)
        val inferXShape = interpreter.getInputTensorFromSignature("x", SIG_INFER).shape()
        require(inferXShape.size == 2) {
            "HeadTrainer: infer.x shape ${inferXShape.toList()} expected rank 2"
        }
        embeddingDim = inferXShape[1]

        val probsShape = interpreter.getOutputTensorFromSignature("probabilities", SIG_INFER).shape()
        require(probsShape.size == 2) {
            "HeadTrainer: infer.probabilities shape ${probsShape.toList()} expected rank 2"
        }
        numClasses = probsShape[1]
        require(numClasses == EXPECTED_NUM_CLASSES) {
            "HeadTrainer: numClasses=$numClasses, expected $EXPECTED_NUM_CLASSES"
        }

        val savedShape = interpreter.getOutputTensorFromSignature("weights", SIG_SAVE).shape()
        require(savedShape.size == 1) {
            "HeadTrainer: save.weights shape ${savedShape.toList()} expected rank 1"
        }
        weightsSize = savedShape[0]
    }

    /** [embeddings] is `[N, embeddingDim]`. Returns `[N, numClasses]` softmax probabilities. */
    suspend fun infer(embeddings: Array<FloatArray>): Array<FloatArray> = withContext(Dispatchers.Default) {
        check(!closed) { "HeadTrainer is closed" }
        validateBatch(embeddings)
        val n = embeddings.size
        synchronized(lock) {
            check(!closed) { "HeadTrainer is closed" }
            val output = Array(n) { FloatArray(numClasses) }
            interpreter.runSignature(
                mapOf<String, Any>("x" to embeddings),
                mapOf<String, Any>("probabilities" to output),
                SIG_INFER,
            )
            output
        }
    }

    suspend fun trainStep(embeddings: Array<FloatArray>, labels: IntArray): Float = withContext(Dispatchers.Default) {
        check(!closed) { "HeadTrainer is closed" }
        validateBatch(embeddings)
        require(labels.size == embeddings.size) {
            "trainStep: labels.size=${labels.size} != embeddings.size=${embeddings.size}"
        }
        val loss = FloatBuffer.allocate(1)
        synchronized(lock) {
            check(!closed) { "HeadTrainer is closed" }
            interpreter.runSignature(
                mapOf<String, Any>("x" to embeddings, "y" to labels),
                mapOf<String, Any>("loss" to loss),
                SIG_TRAIN,
            )
        }
        loss.get(0)
    }

    /** Snapshot the head's trainable weights as a flat float vector of length [weightsSize]. */
    suspend fun saveWeights(): FloatArray = withContext(Dispatchers.Default) {
        check(!closed) { "HeadTrainer is closed" }
        val out = FloatArray(weightsSize)
        synchronized(lock) {
            check(!closed) { "HeadTrainer is closed" }
            interpreter.runSignature(
                mapOf<String, Any>("unused" to floatArrayOf(0f)),
                mapOf<String, Any>("weights" to out),
                SIG_SAVE,
            )
        }
        out
    }

    /**
     * Load a previously-saved weights blob.
     */
    suspend fun restoreWeights(weights: FloatArray) = withContext(Dispatchers.Default) {
        check(!closed) { "HeadTrainer is closed" }
        require(weights.size == weightsSize) {
            "restoreWeights: got ${weights.size} floats, expected $weightsSize"
        }
        val restoredSink = ByteBuffer.allocate(1)
        synchronized(lock) {
            check(!closed) { "HeadTrainer is closed" }
            interpreter.runSignature(
                mapOf<String, Any>("weights" to weights),
                mapOf<String, Any>("restored" to restoredSink),
                SIG_RESTORE,
            )
        }
    }

    private fun validateBatch(embeddings: Array<FloatArray>) {
        require(embeddings.isNotEmpty()) { "Empty batch" }
        require(embeddings.all { it.size == embeddingDim }) {
            "All embeddings must be of length $embeddingDim"
        }
    }

    override fun close() {
        synchronized(lock) {
            if (closed) return
            closed = true
            interpreter.close()
        }
    }

    companion object {
        const val MODEL_ASSET = "mobilebert_head_trainer.tflite"
        const val DEFAULT_NUM_THREADS = 2
        const val EXPECTED_NUM_CLASSES = 2

        const val SIG_INFER = "infer"
        const val SIG_TRAIN = "train"
        const val SIG_SAVE = "save"
        const val SIG_RESTORE = "restore"

        private val REQUIRED_SIGNATURES = setOf(SIG_INFER, SIG_TRAIN, SIG_SAVE, SIG_RESTORE)
    }
}
