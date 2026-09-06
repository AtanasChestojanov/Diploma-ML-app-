package com.example.hatespeechmonitor.personalization

import com.example.hatespeechmonitor.ml.Label

data class FeedbackSample(
    val text: String,
    val userLabel: Label,
    val embedding: FloatArray? = null,
    val timestampMillis: Long = System.currentTimeMillis(),
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is FeedbackSample) return false
        return text == other.text &&
            userLabel == other.userLabel &&
            timestampMillis == other.timestampMillis &&
            (embedding?.contentEquals(other.embedding) ?: (other.embedding == null))
    }

    override fun hashCode(): Int {
        var r = text.hashCode()
        r = 31 * r + userLabel.hashCode()
        r = 31 * r + timestampMillis.hashCode()
        r = 31 * r + (embedding?.contentHashCode() ?: 0)
        return r
    }
}
