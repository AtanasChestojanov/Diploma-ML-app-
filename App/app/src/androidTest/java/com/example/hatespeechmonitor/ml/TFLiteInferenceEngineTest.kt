package com.example.hatespeechmonitor.ml

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class TFLiteInferenceEngineTest {

    private lateinit var engine: TFLiteInferenceEngine

    @Before
    fun setUp() {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val vocab = VocabLoader.loadFromAssets(ctx)
        val tokenizer = WordPieceTokenizer(vocab, maxLen = 64)
        engine = TFLiteInferenceEngine(ctx, tokenizer)
    }

    @After
    fun tearDown() {
        engine.close()
    }

    @Test
    fun outputShapeAndProbabilitiesAreValid() = runBlocking {
        assertEquals("sequence length", 64, engine.sequenceLength)
        assertTrue("model size > 1 MB", engine.modelSizeBytes > 1_000_000)

        val r = engine.classify("I love everyone in the world")
        assertTrue("cold on first call", r.isCold)
        assertTrue("pNothate in [0,1]", r.pNothate in -1e-4f..1.0001f)
        assertTrue("pHate in [0,1]", r.pHate in -1e-4f..1.0001f)
        assertEquals("probabilities sum to ~1", 1.0f, r.pHate + r.pNothate, 1e-2f)
        assertTrue("latency >= 0", r.latencyMs >= 0)
    }

    @Test
    fun warmAfterCold() = runBlocking {
        val cold = engine.classify("the quick brown fox jumps over the lazy dog")
        assertTrue(cold.isCold)
        val warm = engine.classify("the quick brown fox jumps over the lazy dog")
        assertFalse("second call must not be cold", warm.isCold)
        assertEquals("determinism: pHate", cold.pHate, warm.pHate, 1e-5f)
        assertEquals("determinism: pNothate", cold.pNothate, warm.pNothate, 1e-5f)
    }

    @Test
    fun differentInputsProduceDifferentOutputs() = runBlocking {
        val a = engine.classify("I love my family and everyone deserves respect")
        val b = engine.classify("kill them all, they are subhuman")
        val diff = kotlin.math.abs(a.pHate - b.pHate)
        assertTrue(
            "two clearly different texts should differ in P(hate); a=${a.pHate} b=${b.pHate}",
            diff > 0.05f,
        )
    }
}
