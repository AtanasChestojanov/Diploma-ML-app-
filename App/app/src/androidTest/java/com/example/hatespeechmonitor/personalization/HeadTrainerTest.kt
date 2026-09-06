package com.example.hatespeechmonitor.personalization

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import kotlin.math.abs
import kotlin.random.Random

@RunWith(AndroidJUnit4::class)
class HeadTrainerTest {

    private lateinit var head: HeadTrainer

    @Before
    fun setUp() {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        head = HeadTrainer(ctx)
    }

    @After
    fun tearDown() {
        head.close()
    }

    @Test
    fun inferShapeAndSoftmax() = runBlocking {
        assertEquals(512, head.embeddingDim)
        assertEquals(2, head.numClasses)

        val rng = Random(seed = 1234L)
        val n = 5
        val x = Array(n) { FloatArray(512) { rng.nextFloat() * 2f - 1f } }

        val probs = head.infer(x)
        assertEquals("batch dim", n, probs.size)
        for ((i, row) in probs.withIndex()) {
            assertEquals("row $i width", 2, row.size)
            assertTrue("row $i: p0 in [0,1] = ${row[0]}", row[0] in -1e-4f..1.0001f)
            assertTrue("row $i: p1 in [0,1] = ${row[1]}", row[1] in -1e-4f..1.0001f)
            assertEquals("row $i sums to ~1", 1.0f, row[0] + row[1], 1e-2f)
        }
    }

    @Test
    fun trainLossDecreases() = runBlocking {
        val rng = Random(seed = 42L)
        val n = 8
        val x = Array(n) { FloatArray(512) { rng.nextFloat() * 2f - 1f } }
        // mixed labels so the head has something to learn
        val y = IntArray(n) { if (it < n / 2) 0 else 1 }

        val losses = ArrayList<Float>(50)
        repeat(50) { losses.add(head.trainStep(x, y)) }
        for ((i, l) in losses.withIndex()) {
            assertTrue("loss[$i]=$l is finite", l.isFinite())
        }
        val firstAvg = losses.subList(0, 5).average()
        val lastAvg = losses.subList(45, 50).average()
        assertTrue(
            "loss should decrease over 50 steps: first5avg=$firstAvg last5avg=$lastAvg",
            lastAvg < firstAvg * 0.9,
        )
    }

    @Test
    fun saveRestoreRoundTrip() = runBlocking {
        val rng = Random(seed = 7L)
        val n = 6
        val x = Array(n) { FloatArray(512) { rng.nextFloat() * 2f - 1f } }
        val y = IntArray(n) { it % 2 }

        // 1. Move away from factory weights so we have something to round-trip.
        repeat(10) { head.trainStep(x, y) }
        val predsA = head.infer(x).deepCopy()

        // 2. Snapshot.
        val weights = head.saveWeights()
        assertEquals("weights length", head.weightsSize, weights.size)
        for ((i, w) in weights.withIndex()) {
            assertTrue("weights[$i]=$w finite", w.isFinite())
        }

        // 3. Mutate the head (train more) — predictions should drift.
        repeat(15) { head.trainStep(x, y) }
        val predsB = head.infer(x)
        val driftedAtLeastOne = (0 until n).any { i ->
            abs(predsB[i][1] - predsA[i][1]) > 1e-3f
        }
        assertTrue("further training should change predictions on at least one row", driftedAtLeastOne)

        // 4. Restore (no return value; success = no exception thrown).
        head.restoreWeights(weights)

        // 5. After restore, predictions should match the snapshot within tolerance.
        val predsC = head.infer(x)
        for (i in 0 until n) {
            assertEquals(
                "restore p(hate) row $i: A=${predsA[i][1]} C=${predsC[i][1]}",
                predsA[i][1], predsC[i][1], 1e-4f,
            )
            assertEquals(
                "restore p(nothate) row $i: A=${predsA[i][0]} C=${predsC[i][0]}",
                predsA[i][0], predsC[i][0], 1e-4f,
            )
        }
    }

    @Test
    fun batchSizeChangeIsHandled() = runBlocking {
        val rng = Random(seed = 99L)
        val x1 = Array(1) { FloatArray(512) { rng.nextFloat() } }
        val x4 = Array(4) { FloatArray(512) { rng.nextFloat() } }
        val x8 = Array(8) { FloatArray(512) { rng.nextFloat() } }
        // varying batch sizes must not blow up the SignatureRunner resize cache
        assertEquals(1, head.infer(x1).size)
        assertEquals(4, head.infer(x4).size)
        assertEquals(8, head.infer(x8).size)
        assertEquals(4, head.infer(x4).size) // back to a previously-seen N
    }
}

private fun Array<FloatArray>.deepCopy(): Array<FloatArray> = Array(size) { i -> this[i].copyOf() }
