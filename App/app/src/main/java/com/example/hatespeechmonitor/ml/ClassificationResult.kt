package com.example.hatespeechmonitor.ml

enum class Label { HATE, NOTHATE }

/** Class index used by the trained model: 0 = nothate, 1 = hate. */
val Label.modelIndex: Int get() = when (this) {
    Label.NOTHATE -> 0
    Label.HATE -> 1
}

fun labelFromModelIndex(index: Int): Label = when (index) {
    0 -> Label.NOTHATE
    1 -> Label.HATE
    else -> error("Unknown class index: $index")
}

data class ClassificationResult(
    val pHate: Float,
    val pNothate: Float,
    val latencyMs: Long,
    val isCold: Boolean,
    /** Encoder-leg wall clock; non-null only for the two-model personalization path. */
    val encoderLatencyMs: Long? = null,
    /** Head-leg wall clock; non-null only for the two-model personalization path. */
    val headLatencyMs: Long? = null,
) {
    /** Argmax label. Equivalent to [labelAt] with threshold 0.5f. */
    val argmaxLabel: Label get() = if (pHate >= pNothate) Label.HATE else Label.NOTHATE

    /** Apply a tunable decision threshold to P(hate). */
    fun labelAt(threshold: Float): Label =
        if (pHate >= threshold) Label.HATE else Label.NOTHATE
}
