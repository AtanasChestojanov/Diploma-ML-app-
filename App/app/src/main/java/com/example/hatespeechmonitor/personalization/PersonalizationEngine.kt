package com.example.hatespeechmonitor.personalization

import com.example.hatespeechmonitor.ml.ClassificationResult

/**
 * Two-model on-device personalization: a frozen encoder (text → 512-d embedding) plus a
 * trainable classifier head (embedding → softmax over {nothate, hate}).
 */
interface PersonalizationEngine : AutoCloseable {

    /** Combined size of the encoder + head models in bytes. */
    val modelSizeBytes: Long

    /** Encoder model size only (for the thesis model-size table). */
    val encoderModelSizeBytes: Long

    /** Head trainer model size only. */
    val headModelSizeBytes: Long

    /** Embedding dimensionality between the encoder and the head. */
    val embeddingDim: Int

    /** True if a personalized weights blob is currently persisted to disk. */
    val hasPersistedWeights: Boolean

    /** Tokenize → encoder → head.infer → probabilities → result (with split latencies). */
    suspend fun classify(text: String): ClassificationResult

    /** Run the frozen encoder only; cache the result on a [FeedbackSample]. */
    suspend fun embed(text: String): FloatArray

    suspend fun train(
        samples: List<FeedbackSample>,
        epochs: Int = DEFAULT_EPOCHS,
        batchSize: Int = DEFAULT_BATCH_SIZE,
    ): TrainResult

    /** Persist the current trainable head's weights to app storage. */
    suspend fun save()

    /**
     * Revert the trainable head to its factory (offline-trained) weights and delete any
     * persisted blob.
     */
    suspend fun resetToFactory()

    override fun close()

    companion object {
        const val EMBEDDING_DIM = 512
        const val DEFAULT_EPOCHS = 3
        const val DEFAULT_BATCH_SIZE = 8

        /** Baked into the head_trainer's `train` signature at export. */
        const val ON_DEVICE_LR = 0.05f
    }
}

data class TrainResult(
    val epochs: Int,
    val samplesUsed: Int,
    /** Loss averaged over the last epoch's mini-batches. */
    val finalLoss: Float,
    /** Loss of the first epoch (averaged over its batches). Useful "did it learn?" signal. */
    val initialLoss: Float,
    /** Wall-clock spent running the encoder on samples whose embedding was not cached. */
    val embeddingMillis: Long,
    /** Wall-clock spent in the head trainer's `train` signature (epochs × batches). */
    val trainingMillis: Long,
)
