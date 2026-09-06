package com.example.hatespeechmonitor.ml

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Verifies that the Kotlin tokenizer reproduces, byte-for-byte, the input_ids /
 * attention_mask / token_type_ids that the offline (HuggingFace) tokenizer produced
 * for each text in `assets/tokenization_examples.json`. Any mismatch is a hard failure.
 */
@RunWith(AndroidJUnit4::class)
class WordPieceTokenizerTest {

    private lateinit var tokenizer: WordPieceTokenizer
    private lateinit var examples: JSONArray
    private var contractMaxLen: Int = 64

    @Before
    fun setUp() {
        val ctx = InstrumentationRegistry.getInstrumentation().context
            ?: InstrumentationRegistry.getInstrumentation().targetContext

        // vocab.txt lives in the *app under test* assets, not the test apk.
        val targetCtx = InstrumentationRegistry.getInstrumentation().targetContext
        val vocab = VocabLoader.loadFromAssets(targetCtx, "vocab.txt")

        val rawJson = targetCtx.assets.open("tokenization_examples.json")
            .bufferedReader(Charsets.UTF_8)
            .use { it.readText() }
        val root = JSONObject(rawJson)
        val contract = root.getJSONObject("contract")
        contractMaxLen = contract.getInt("max_len")
        examples = root.getJSONArray("examples")

        tokenizer = WordPieceTokenizer(vocab, maxLen = contractMaxLen)
        // silence unused-var warning on ctx in some Kotlin versions
        check(ctx === ctx)
    }

    @Test
    fun reproducesEveryExample() {
        assertEquals("contract max_len must be 64", 64, contractMaxLen)
        var failures = 0
        val report = StringBuilder()
        for (i in 0 until examples.length()) {
            val ex = examples.getJSONObject(i)
            val text = ex.getString("text")
            val expectedIds = ex.getJSONArray("input_ids").toIntArray()
            val expectedMask = ex.getJSONArray("attention_mask").toIntArray()
            val expectedTypes = ex.getJSONArray("token_type_ids").toIntArray()
            val expectedNonPad = ex.getInt("non_pad_count")

            val out = tokenizer.encode(text)

            try {
                assertEquals(
                    "input_ids length (example $i: \"$text\")",
                    contractMaxLen,
                    out.inputIds.size,
                )
                assertArrayEquals(
                    "input_ids mismatch (example $i: \"$text\")\n  expected=${expectedIds.toList()}\n  actual  =${out.inputIds.toList()}",
                    expectedIds,
                    out.inputIds,
                )
                assertArrayEquals(
                    "attention_mask mismatch (example $i: \"$text\")",
                    expectedMask,
                    out.attentionMask,
                )
                assertArrayEquals(
                    "token_type_ids mismatch (example $i: \"$text\")",
                    expectedTypes,
                    out.tokenTypeIds,
                )
                assertEquals(
                    "non_pad_count mismatch (example $i: \"$text\")",
                    expectedNonPad,
                    out.nonPadCount,
                )
            } catch (e: AssertionError) {
                failures += 1
                report.append("FAIL #$i: ${e.message}\n")
            }
        }
        if (failures > 0) {
            throw AssertionError("$failures/${examples.length()} examples failed:\n$report")
        }
    }

    @Test
    fun tokenTypeIdsAreAllZeros() {
        for (i in 0 until examples.length()) {
            val text = examples.getJSONObject(i).getString("text")
            val out = tokenizer.encode(text)
            for ((j, v) in out.tokenTypeIds.withIndex()) {
                assertEquals("token_type_ids[$j] for \"$text\"", 0, v)
            }
        }
    }

    private fun JSONArray.toIntArray(): IntArray = IntArray(length()) { getInt(it) }
}
