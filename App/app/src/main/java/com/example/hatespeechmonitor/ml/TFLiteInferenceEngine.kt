package com.example.hatespeechmonitor.ml

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class TFLiteInferenceEngine(
    context: Context,
    private val tokenizer: Tokenizer,
    modelAssetName: String = MODEL_ASSET,
    override val numThreads: Int = DEFAULT_NUM_THREADS,
) : InferenceEngine {

    private val mappedModel: MappedByteBuffer
    private val interpreter: Interpreter
    override val modelSizeBytes: Long
    override val sequenceLength: Int

    private val inputIdsIndex: Int
    private val attentionMaskIndex: Int
    private val tokenTypeIdsIndex: Int

    private val lock = Any()

    @Volatile
    private var hasRunOnce = false

    @Volatile
    private var closed = false

    init {
        val afd = context.assets.openFd(modelAssetName)
        modelSizeBytes = afd.declaredLength
        mappedModel = FileInputStream(afd.fileDescriptor).use { fis ->
            fis.channel.map(FileChannel.MapMode.READ_ONLY, afd.startOffset, afd.declaredLength)
        }
        afd.close()

        val options = Interpreter.Options().apply { setNumThreads(numThreads) }
        interpreter = Interpreter(mappedModel, options)

        inputIdsIndex = resolveInputIndex("input_ids")
        attentionMaskIndex = resolveInputIndex("attention_mask")
        tokenTypeIdsIndex = resolveInputIndex("token_type_ids")

        val seqShape = interpreter.getInputTensor(inputIdsIndex).shape()
        require(seqShape.size == 2 && seqShape[0] == 1) {
            "Unexpected input_ids shape: ${seqShape.toList()}"
        }
        sequenceLength = seqShape[1]
        require(sequenceLength == tokenizer.maxLen) {
            "Model sequence length ($sequenceLength) does not match tokenizer.maxLen (${tokenizer.maxLen})"
        }

        val outShape = interpreter.getOutputTensor(0).shape()
        require(outShape.contentEquals(intArrayOf(1, 2))) {
            "Unexpected output shape: ${outShape.toList()} (expected [1, 2])"
        }
    }

    private fun resolveInputIndex(needle: String): Int {
        val n = interpreter.inputTensorCount
        val matches = (0 until n).filter { i ->
            interpreter.getInputTensor(i).name().contains(needle)
        }
        check(matches.size == 1) {
            val names = (0 until n).joinToString { interpreter.getInputTensor(it).name() }
            "Expected exactly one input tensor name containing '$needle', got ${matches.size}. " +
                "Available: [$names]"
        }
        return matches.single()
    }

    override suspend fun classify(text: String): ClassificationResult = withContext(Dispatchers.Default) {
        check(!closed) { "Engine is closed" }
        val encoded = tokenizer.encode(text)

        val inputs = arrayOfNulls<Any>(interpreter.inputTensorCount)
        inputs[inputIdsIndex] = arrayOf(encoded.inputIds)
        inputs[attentionMaskIndex] = arrayOf(encoded.attentionMask)
        inputs[tokenTypeIdsIndex] = arrayOf(encoded.tokenTypeIds)

        val output = Array(1) { FloatArray(2) }
        val outputs = HashMap<Int, Any>(1).apply { put(0, output) }

        val cold: Boolean
        val latencyMs: Long
        synchronized(lock) {
            check(!closed) { "Engine is closed" }
            cold = !hasRunOnce
            val t0 = System.nanoTime()
            interpreter.runForMultipleInputsOutputs(inputs, outputs)
            latencyMs = (System.nanoTime() - t0) / 1_000_000L
            hasRunOnce = true
        }

        ClassificationResult(
            pNothate = output[0][0],
            pHate = output[0][1],
            latencyMs = latencyMs,
            isCold = cold,
        )
    }

    override fun close() {
        synchronized(lock) {
            if (closed) return
            closed = true
            interpreter.close()
        }
    }

    companion object {
        const val MODEL_ASSET = "mobilebert_dynamic_range.tflite"
        const val DEFAULT_NUM_THREADS = 4
    }
}
