package com.example.hatespeechmonitor.personalization

import android.content.Context
import com.example.hatespeechmonitor.ml.ClassificationResult
import com.example.hatespeechmonitor.ml.EncoderEngine
import com.example.hatespeechmonitor.ml.Tokenizer
import com.example.hatespeechmonitor.ml.modelIndex
import com.example.hatespeechmonitor.util.MemLogger
import java.io.File
import java.util.Random

/**
 * Two-model implementation of [PersonalizationEngine]:
 *  - [encoder] = frozen MobileBERT encoder (base TFLite runtime), text → 512-d.
 *  - [head] = trainable head (Flex runtime), 512-d → softmax over {nothate, hate}.
 *
 * Use [create] to construct — the factory snapshots the head's factory weights (for cheap
 * [resetToFactory]) and restores any previously-persisted personalization blob.
 */
class TFLitePersonalizationEngine private constructor(
    private val encoder: EncoderEngine,
    private val head: HeadTrainer,
    private val storage: HeadWeightsStorage,
    private val factoryWeights: FloatArray,
) : PersonalizationEngine {

    override val encoderModelSizeBytes: Long = encoder.modelSizeBytes
    override val headModelSizeBytes: Long = head.modelSizeBytes
    override val modelSizeBytes: Long = encoderModelSizeBytes + headModelSizeBytes
    override val embeddingDim: Int = encoder.embeddingDim
    override val hasPersistedWeights: Boolean get() = storage.exists()

    override suspend fun classify(text: String): ClassificationResult {
        val enc = encoder.encode(text)
        val tHead = System.nanoTime()
        val probs = head.infer(arrayOf(enc.embedding))
        val headMs = (System.nanoTime() - tHead) / 1_000_000L
        return ClassificationResult(
            pNothate = probs[0][0],
            pHate = probs[0][1],
            latencyMs = enc.latencyMs + headMs,
            isCold = enc.isCold,
            encoderLatencyMs = enc.latencyMs,
            headLatencyMs = headMs,
        )
    }

    override suspend fun embed(text: String): FloatArray = encoder.encode(text).embedding

    override suspend fun train(
        samples: List<FeedbackSample>,
        epochs: Int,
        batchSize: Int,
    ): TrainResult {
        require(samples.isNotEmpty()) { "train: no feedback samples" }
        require(epochs >= 1) { "epochs must be >= 1" }
        require(batchSize >= 1) { "batchSize must be >= 1" }

        // 1. Populate any missing embeddings (cache misses) and time it.
        val embedStart = System.nanoTime()
        val populated = samples.map { sample ->
            sample.embedding ?: encoder.encode(sample.text).embedding
        }
        val embeddingMillis = (System.nanoTime() - embedStart) / 1_000_000L
        MemLogger.sample("embed_done")

        require(populated.all { it.size == embeddingDim }) {
            "All embeddings must have length $embeddingDim"
        }

        val n = populated.size
        val xs: Array<FloatArray> = Array(n) { populated[it] }
        val ys: IntArray = IntArray(n) { samples[it].userLabel.modelIndex }

        // 2. Train: epochs × mini-batches, shuffling indices each epoch with a
        //    FIXED-SEED RNG so runs are reproducible (see TRAIN_SEED / shuffleInPlace).
        val indices = IntArray(n) { it }
        val rng = Random(TRAIN_SEED)
        var initialLoss = 0f
        var finalLoss = 0f
        val trainStart = System.nanoTime()
        for (epoch in 0 until epochs) {
            shuffleInPlace(indices, rng)
            var lossSum = 0f
            var batches = 0
            var i = 0
            while (i < n) {
                val end = minOf(i + batchSize, n)
                val bs = end - i
                val bx = Array(bs) { xs[indices[i + it]] }
                val by = IntArray(bs) { ys[indices[i + it]] }
                lossSum += head.trainStep(bx, by)
                batches += 1
                i = end
            }
            val epochAvg = if (batches > 0) lossSum / batches else 0f
            if (epoch == 0) initialLoss = epochAvg
            if (epoch == epochs - 1) finalLoss = epochAvg
            MemLogger.sample("epoch_${epoch + 1}")
        }
        val trainingMillis = (System.nanoTime() - trainStart) / 1_000_000L

        return TrainResult(
            epochs = epochs,
            samplesUsed = n,
            finalLoss = finalLoss,
            initialLoss = initialLoss,
            embeddingMillis = embeddingMillis,
            trainingMillis = trainingMillis,
        )
    }

    override suspend fun save() {
        val current = head.saveWeights()
        storage.save(current)
    }

    override suspend fun resetToFactory() {
        head.restoreWeights(factoryWeights)
        storage.delete()
    }

    override fun close() {
        runCatching { encoder.close() }
        runCatching { head.close() }
    }

    private fun shuffleInPlace(a: IntArray, rng: Random) {
        // Fisher–Yates with a fixed-seed RNG (TRAIN_SEED) so on-device training is
        // reproducible run-to-run. Previously used the unseeded global Math.random(),
        // which — together with dropout in the head trainer — made identical setups
        // differ by up to ~0.2 F1. With dropout now off (training=False in the head
        // trainer export) and this seeded, personalization is deterministic.
        for (i in a.size - 1 downTo 1) {
            val j = rng.nextInt(i + 1)
            val tmp = a[i]; a[i] = a[j]; a[j] = tmp
        }
    }

    companion object {
        const val DEFAULT_WEIGHTS_FILE = "head_weights.bin"

        /** Fixed seed for the per-epoch shuffle → reproducible on-device training. */
        const val TRAIN_SEED = 42L

        /**
         * Construct an engine, snapshot the head's factory weights, and restore any
         * previously-persisted personalization blob.
         */
        suspend fun create(
            context: Context,
            tokenizer: Tokenizer,
            persistFileName: String = DEFAULT_WEIGHTS_FILE,
        ): TFLitePersonalizationEngine {
            val encoder = EncoderEngine(context, tokenizer)
            MemLogger.sample("encoder_loaded")
            val head = HeadTrainer(context)
            MemLogger.sample("head_loaded")
            val storage = HeadWeightsStorage(File(context.filesDir, persistFileName))

            // Snapshot factory weights immediately so resetToFactory is cheap.
            val factory = head.saveWeights()

            // Re-apply any previously persisted personalization.
            val persisted = storage.load(head.weightsSize)
            if (persisted != null) head.restoreWeights(persisted)

            return TFLitePersonalizationEngine(encoder, head, storage, factory)
        }
    }
}
