package com.example.hatespeechmonitor.ml

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import kotlin.math.sqrt

@RunWith(AndroidJUnit4::class)
class EncoderEngineTest {

    private lateinit var engine: EncoderEngine

    @Before
    fun setUp() {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val vocab = VocabLoader.loadFromAssets(ctx)
        val tokenizer = WordPieceTokenizer(vocab, maxLen = 64)
        engine = EncoderEngine(ctx, tokenizer)
    }

    @After
    fun tearDown() {
        engine.close()
    }

    @Test
    fun outputShapeAndFinite() = runBlocking {
        assertEquals(64, engine.sequenceLength)
        assertEquals(512, engine.embeddingDim)

        val r = engine.encode("hello world")
        assertEquals("embedding length", 512, r.embedding.size)
        assertTrue("cold on first call", r.isCold)
        assertTrue("latency >= 0", r.latencyMs >= 0)
        for ((i, v) in r.embedding.withIndex()) {
            assertTrue("embedding[$i] = $v is not finite", v.isFinite())
        }
    }

    @Test
    fun deterministic() = runBlocking {
        val a = engine.encode("the quick brown fox jumps over the lazy dog").embedding
        val b = engine.encode("the quick brown fox jumps over the lazy dog").embedding
        val diff = l2(a, b)
        assertTrue("encoder is deterministic: same text → same embedding (diff=$diff)", diff < 1e-3f)
    }

    @Test
    fun distinctTextsDiffer() = runBlocking {
        val a = engine.encode("I love everyone and wish them well").embedding
        val b = engine.encode("kill all those people, they are subhuman").embedding
        val diff = l2(a, b)
        assertTrue("distinct texts should produce distinct embeddings (l2=$diff)", diff > 0.1f)
    }

    @Test
    fun warmAfterCold() = runBlocking {
        val first = engine.encode("warmup text")
        assertTrue(first.isCold)
        val second = engine.encode("warmup text")
        assertTrue("second call must not be cold", !second.isCold)
    }

    private fun l2(a: FloatArray, b: FloatArray): Float {
        require(a.size == b.size)
        var s = 0.0
        for (i in a.indices) {
            val d = (a[i] - b[i]).toDouble()
            s += d * d
        }
        return sqrt(s).toFloat()
    }
}
