package com.example.hatespeechmonitor.ml

data class TokenizerOutput(
    val inputIds: IntArray,
    val attentionMask: IntArray,
    val tokenTypeIds: IntArray,
    val tokens: List<String>,
) {
    val nonPadCount: Int get() = attentionMask.sum()

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is TokenizerOutput) return false
        return inputIds.contentEquals(other.inputIds) &&
            attentionMask.contentEquals(other.attentionMask) &&
            tokenTypeIds.contentEquals(other.tokenTypeIds) &&
            tokens == other.tokens
    }

    override fun hashCode(): Int {
        var r = inputIds.contentHashCode()
        r = 31 * r + attentionMask.contentHashCode()
        r = 31 * r + tokenTypeIds.contentHashCode()
        r = 31 * r + tokens.hashCode()
        return r
    }
}

interface Tokenizer {
    val maxLen: Int
    fun encode(text: String): TokenizerOutput
}
