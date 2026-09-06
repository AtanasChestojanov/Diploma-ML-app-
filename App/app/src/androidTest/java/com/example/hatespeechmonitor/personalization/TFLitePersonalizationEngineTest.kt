package com.example.hatespeechmonitor.personalization

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.example.hatespeechmonitor.ml.Label
import com.example.hatespeechmonitor.ml.VocabLoader
import com.example.hatespeechmonitor.ml.WordPieceTokenizer
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import kotlin.math.abs

@RunWith(AndroidJUnit4::class)
class TFLitePersonalizationEngineTest {

    private lateinit var engine: TFLitePersonalizationEngine
    private val testFile = "head_weights_test_${System.currentTimeMillis()}.bin"

    @Before
    fun setUp() {
        runBlocking {
            val ctx = InstrumentationRegistry.getInstrumentation().targetContext
            File(ctx.filesDir, testFile).delete()
            val vocab = VocabLoader.loadFromAssets(ctx)
            val tokenizer = WordPieceTokenizer(vocab, maxLen = 64)
            engine = TFLitePersonalizationEngine.create(ctx, tokenizer, persistFileName = testFile)
        }
    }

    @After
    fun tearDown() {
        runBlocking {
            engine.close()
            val ctx = InstrumentationRegistry.getInstrumentation().targetContext
            File(ctx.filesDir, testFile).delete()
        }
    }

    @Test
    fun classifyReportsSplitLatencies() = runBlocking {
        val r = engine.classify("hello world")
        assertEquals("p sums to ~1", 1.0f, r.pHate + r.pNothate, 1e-2f)
        assertNotNull("encoderLatencyMs populated", r.encoderLatencyMs)
        assertNotNull("headLatencyMs populated", r.headLatencyMs)
        assertEquals(
            "total latency = encoder + head",
            r.latencyMs, (r.encoderLatencyMs ?: 0L) + (r.headLatencyMs ?: 0L),
        )
        assertTrue("cold on first call", r.isCold)
    }

    @Test
    fun feedbackShiftsPredictionsTowardUserLabel() = runBlocking {
        // 1. Baseline: classify a borderline-ish text and capture pHate.
        val text = "people sometimes do things that I do not like"
        val baseline = engine.classify(text)
        val baselinePHate = baseline.pHate

        // 2. Pick a user label OPPOSITE to the model's current prediction so the shift
        //    must move pHate by a clearly measurable amount.
        val userLabel = if (baselinePHate >= 0.5f) Label.NOTHATE else Label.HATE

        // 3. Embed once, reuse across samples (this is what the FeedbackSample cache exists for).
        val embedding = engine.embed(text)
        val samples = List(8) {
            FeedbackSample(text = text, userLabel = userLabel, embedding = embedding)
        }

        // 4. Train for several epochs.
        val result = engine.train(samples, epochs = 10, batchSize = 8)
        assertEquals(10, result.epochs)
        assertEquals(8, result.samplesUsed)
        assertTrue("trainingMillis > 0", result.trainingMillis > 0)
        // embedding wall-clock is 0 here because we passed cached embeddings on every sample.
        assertEquals("no missing embeddings = no encoder calls", 0L, result.embeddingMillis)
        assertTrue("loss should decrease: init=${result.initialLoss} final=${result.finalLoss}",
            result.finalLoss < result.initialLoss)

        // 5. Re-classify; pHate must have moved toward the user label.
        val after = engine.classify(text)
        val shift = after.pHate - baselinePHate
        val expectedSign = if (userLabel == Label.HATE) 1 else -1
        assertTrue(
            "pHate must shift toward user label '$userLabel': " +
                "baseline=$baselinePHate after=${after.pHate} shift=$shift",
            (if (expectedSign > 0) shift > 0.05f else shift < -0.05f),
        )
    }

    @Test
    fun saveAndResetToFactory() = runBlocking {
        // 1. Capture a "factory" prediction.
        val text = "the quick brown fox jumps over the lazy dog"
        val factoryProb = engine.classify(text).pHate

        // 2. Train to move weights away from factory.
        val embedding = engine.embed(text)
        engine.train(
            List(8) { FeedbackSample(text, Label.HATE, embedding) },
            epochs = 10, batchSize = 8,
        )
        val trainedProb = engine.classify(text).pHate
        assertTrue(
            "trained predictions should differ from factory: factory=$factoryProb trained=$trainedProb",
            abs(trainedProb - factoryProb) > 0.05f,
        )

        // 3. Persist.
        assertTrue("hasPersistedWeights = false before save", !engine.hasPersistedWeights)
        engine.save()
        assertTrue("hasPersistedWeights = true after save", engine.hasPersistedWeights)

        // 4. Reset to factory wipes both the head and the persisted blob.
        engine.resetToFactory()
        assertTrue("hasPersistedWeights = false after reset", !engine.hasPersistedWeights)
        val resetProb = engine.classify(text).pHate
        assertEquals(
            "reset must restore factory predictions",
            factoryProb, resetProb, 1e-4f,
        )
    }

    @Test
    fun persistsAcrossEngineRecreate() = runBlocking {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val text = "totally arbitrary sentence for this test"

        // 1. Train and save.
        val embedding = engine.embed(text)
        engine.train(
            List(8) { FeedbackSample(text, Label.HATE, embedding) },
            epochs = 10, batchSize = 8,
        )
        val trainedProb = engine.classify(text).pHate
        engine.save()
        engine.close()

        // 2. Recreate the engine; it should restore the persisted weights at construction.
        val vocab = VocabLoader.loadFromAssets(ctx)
        val tokenizer = WordPieceTokenizer(vocab, maxLen = 64)
        engine = TFLitePersonalizationEngine.create(ctx, tokenizer, persistFileName = testFile)

        assertTrue("hasPersistedWeights after restart", engine.hasPersistedWeights)
        val restoredProb = engine.classify(text).pHate
        assertEquals(
            "recreated engine should reproduce trained predictions",
            trainedProb, restoredProb, 1e-4f,
        )
    }
}
