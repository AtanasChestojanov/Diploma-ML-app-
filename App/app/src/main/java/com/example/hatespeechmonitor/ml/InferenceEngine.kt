package com.example.hatespeechmonitor.ml


interface InferenceEngine : AutoCloseable {
    /** Size of the underlying model file in bytes (for the thesis size table). */
    val modelSizeBytes: Long

    /** Sequence length the model was exported with (must match the tokenizer). */
    val sequenceLength: Int

    /** Number of interpreter threads in use. */
    val numThreads: Int

    suspend fun classify(text: String): ClassificationResult

    override fun close()
}
