package com.example.hatespeechmonitor.ml

data class BenchmarkResult(
    val coldMs: Long?,
    val warmRunsMs: List<Long>,
) {
    val warmCount: Int get() = warmRunsMs.size
    val warmMeanMs: Double get() = if (warmRunsMs.isEmpty()) 0.0 else warmRunsMs.average()
    val warmMinMs: Long get() = warmRunsMs.minOrNull() ?: 0
    val warmMaxMs: Long get() = warmRunsMs.maxOrNull() ?: 0
    val warmP95Ms: Long get() {
        if (warmRunsMs.isEmpty()) return 0
        val sorted = warmRunsMs.sorted()
        val idx = ((sorted.size - 1) * 0.95).toInt()
        return sorted[idx]
    }
}

class Benchmark(private val engine: InferenceEngine) {
    suspend fun run(text: String, warmRuns: Int): BenchmarkResult {
        require(warmRuns >= 1) { "warmRuns must be >= 1" }
        val first = engine.classify(text)
        val cold = if (first.isCold) first.latencyMs else null
        val warm = ArrayList<Long>(warmRuns + (if (cold == null) 1 else 0))
        if (cold == null) warm.add(first.latencyMs)
        repeat(warmRuns) {
            warm.add(engine.classify(text).latencyMs)
        }
        return BenchmarkResult(cold, warm)
    }
}
