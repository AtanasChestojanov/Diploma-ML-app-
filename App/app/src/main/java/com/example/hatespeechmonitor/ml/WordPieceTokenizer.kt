package com.example.hatespeechmonitor.ml

import java.text.Normalizer

/**
 * Reproduces the HuggingFace BERT uncased tokenizer (BasicTokenizer + WordPiece) used by
 * `google/mobilebert-uncased`. Pipeline per token: clean control chars, surround CJK with spaces,
 * whitespace-split; then lowercase, strip NFD combining marks, split on punctuation; finally
 * greedy longest-match WordPiece with `##` continuation prefix. Sequence is wrapped as
 * `[CLS] ... [SEP]`, right-padded with `[PAD]` to [maxLen], and truncated to [maxLen].
 *
 * Verified against `tokenization_examples.json`.
 */
class WordPieceTokenizer(
    private val vocab: Map<String, Int>,
    override val maxLen: Int = 64,
    private val doLowerCase: Boolean = true,
    private val maxInputCharsPerWord: Int = 100,
) : Tokenizer {

    private val padId = vocab.getValue("[PAD]")
    private val clsId = vocab.getValue("[CLS]")
    private val sepId = vocab.getValue("[SEP]")
    private val unkId = vocab.getValue("[UNK]")

    override fun encode(text: String): TokenizerOutput {
        val basic = basicTokenize(text)
        val wp = ArrayList<String>(basic.size)
        for (t in basic) wp.addAll(wordPiece(t))

        val maxBody = maxLen - 2
        val body: List<String> = if (wp.size > maxBody) wp.subList(0, maxBody) else wp

        val tokens = ArrayList<String>(maxLen)
        tokens.add("[CLS]")
        tokens.addAll(body)
        tokens.add("[SEP]")

        val inputIds = IntArray(maxLen) { padId }
        val attention = IntArray(maxLen)
        for (i in tokens.indices) {
            inputIds[i] = vocab[tokens[i]] ?: unkId
            attention[i] = 1
        }
        // pad the displayable tokens list to maxLen for parity with the JSON contract
        val padded = ArrayList<String>(maxLen).apply {
            addAll(tokens)
            while (size < maxLen) add("[PAD]")
        }
        return TokenizerOutput(
            inputIds = inputIds,
            attentionMask = attention,
            tokenTypeIds = IntArray(maxLen),
            tokens = padded,
        )
    }

    // ---------- basic tokenizer ----------

    private fun basicTokenize(text: String): List<String> {
        val cleaned = cleanText(text)
        val cjkSpaced = tokenizeChineseChars(cleaned)
        val origTokens = whitespaceSplit(cjkSpaced)
        val out = ArrayList<String>(origTokens.size * 2)
        for (token in origTokens) {
            var t = token
            if (doLowerCase) {
                t = t.lowercase()
                t = stripAccents(t)
            }
            for (piece in splitOnPunctuation(t)) {
                if (piece.isNotEmpty()) out.add(piece)
            }
        }
        return out
    }

    private fun cleanText(text: String): String {
        val sb = StringBuilder(text.length)
        var i = 0
        while (i < text.length) {
            val cp = text.codePointAt(i)
            val count = Character.charCount(cp)
            when {
                cp == 0 || cp == 0xFFFD || isControl(cp) -> { /* drop */ }
                isWhitespace(cp) -> sb.append(' ')
                else -> sb.appendCodePoint(cp)
            }
            i += count
        }
        return sb.toString()
    }

    private fun stripAccents(text: String): String {
        val nfd = Normalizer.normalize(text, Normalizer.Form.NFD)
        val sb = StringBuilder(nfd.length)
        var i = 0
        while (i < nfd.length) {
            val cp = nfd.codePointAt(i)
            val count = Character.charCount(cp)
            if (Character.getType(cp) != Character.NON_SPACING_MARK.toInt()) {
                sb.appendCodePoint(cp)
            }
            i += count
        }
        return sb.toString()
    }

    private fun tokenizeChineseChars(text: String): String {
        val sb = StringBuilder(text.length)
        var i = 0
        while (i < text.length) {
            val cp = text.codePointAt(i)
            val count = Character.charCount(cp)
            if (isChineseChar(cp)) {
                sb.append(' ').appendCodePoint(cp).append(' ')
            } else {
                sb.appendCodePoint(cp)
            }
            i += count
        }
        return sb.toString()
    }

    private fun splitOnPunctuation(text: String): List<String> {
        val out = ArrayList<StringBuilder>()
        var startNewWord = true
        var i = 0
        while (i < text.length) {
            val cp = text.codePointAt(i)
            val count = Character.charCount(cp)
            if (isPunctuation(cp)) {
                out.add(StringBuilder().appendCodePoint(cp))
                startNewWord = true
            } else {
                if (startNewWord) out.add(StringBuilder())
                startNewWord = false
                out.last().appendCodePoint(cp)
            }
            i += count
        }
        return out.map { it.toString() }
    }

    private fun whitespaceSplit(text: String): List<String> {
        val trimmed = text.trim()
        if (trimmed.isEmpty()) return emptyList()
        return trimmed.split(Regex("\\s+"))
    }

    // ---------- WordPiece ----------

    private fun wordPiece(token: String): List<String> {
        if (token.length > maxInputCharsPerWord) return listOf("[UNK]")
        val out = ArrayList<String>()
        var start = 0
        val n = token.length
        while (start < n) {
            var end = n
            var curSubstr: String? = null
            while (start < end) {
                val piece = if (start == 0) token.substring(start, end) else "##" + token.substring(start, end)
                if (piece in vocab) {
                    curSubstr = piece
                    break
                }
                end -= 1
            }
            if (curSubstr == null) return listOf("[UNK]")
            out.add(curSubstr)
            start = end
        }
        return out
    }

    // ---------- character predicates (mirror HuggingFace tokenization.py) ----------

    private fun isWhitespace(cp: Int): Boolean {
        if (cp == ' '.code || cp == '\t'.code || cp == '\n'.code || cp == '\r'.code) return true
        return Character.getType(cp) == Character.SPACE_SEPARATOR.toInt()
    }

    private fun isControl(cp: Int): Boolean {
        if (cp == '\t'.code || cp == '\n'.code || cp == '\r'.code) return false
        val cat = Character.getType(cp)
        return cat == Character.CONTROL.toInt() ||
            cat == Character.FORMAT.toInt() ||
            cat == Character.SURROGATE.toInt() ||
            cat == Character.PRIVATE_USE.toInt() ||
            cat == Character.UNASSIGNED.toInt()
    }

    private fun isPunctuation(cp: Int): Boolean {
        if (cp in 33..47 || cp in 58..64 || cp in 91..96 || cp in 123..126) return true
        val cat = Character.getType(cp)
        return cat == Character.CONNECTOR_PUNCTUATION.toInt() ||
            cat == Character.DASH_PUNCTUATION.toInt() ||
            cat == Character.START_PUNCTUATION.toInt() ||
            cat == Character.END_PUNCTUATION.toInt() ||
            cat == Character.INITIAL_QUOTE_PUNCTUATION.toInt() ||
            cat == Character.FINAL_QUOTE_PUNCTUATION.toInt() ||
            cat == Character.OTHER_PUNCTUATION.toInt()
    }

    private fun isChineseChar(cp: Int): Boolean {
        return (cp in 0x4E00..0x9FFF) ||
            (cp in 0x3400..0x4DBF) ||
            (cp in 0x20000..0x2A6DF) ||
            (cp in 0x2A700..0x2B73F) ||
            (cp in 0x2B740..0x2B81F) ||
            (cp in 0x2B820..0x2CEAF) ||
            (cp in 0xF900..0xFAFF) ||
            (cp in 0x2F800..0x2FA1F)
    }
}
