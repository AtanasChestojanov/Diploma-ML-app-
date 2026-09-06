package com.example.hatespeechmonitor.ml

import org.json.JSONArray
import org.json.JSONObject

/**
 * One row of a batch-test export: the input text plus the predictions of both the base
 * (fused MobileBERT) engine and the personalized (encoder + trainable head) engine.
 */
data class BatchTestEntry(
    val text: String,
    val basePHate: Float,
    val basePNothate: Float,
    val baseLatencyMs: Long,
    val personalizedPHate: Float,
    val personalizedPNothate: Float,
    val personalizedLatencyMs: Long,
    val encoderLatencyMs: Long,
    val headLatencyMs: Long,
)

/** Parses the input file the user picks for a batch test (sentences only — no labels). */
object BatchInputParser {

    fun parse(content: String): List<String> {
        val stripped = content.removePrefix("\uFEFF").trim()
        if (stripped.isEmpty()) return emptyList()
        return when {
            stripped.startsWith("[") -> parseJsonArray(stripped)
            stripped.startsWith("{") -> parseJsonObject(stripped)
            else -> parseLines(stripped)
        }.also {
            require(it.isNotEmpty()) { "No sentences found in the file." }
        }
    }

    private fun parseJsonArray(s: String): List<String> {
        val arr = JSONArray(s)
        return (0 until arr.length()).mapNotNull { i ->
            when (val v = arr.opt(i)) {
                is String -> v.trim().takeIf { it.isNotEmpty() }
                is JSONObject -> v.optString("text", "").trim().takeIf { it.isNotEmpty() }
                else -> null
            }
        }
    }

    private fun parseJsonObject(s: String): List<String> {
        val obj = JSONObject(s)
        for (key in listOf("sentences", "texts", "samples", "tests", "inputs")) {
            if (obj.has(key)) {
                val arr = obj.getJSONArray(key)
                return (0 until arr.length()).mapNotNull { i ->
                    when (val v = arr.opt(i)) {
                        is String -> v.trim().takeIf { it.isNotEmpty() }
                        is JSONObject -> v.optString("text", "").trim().takeIf { it.isNotEmpty() }
                        else -> null
                    }
                }
            }
        }
        error("JSON object must contain an array under `sentences`, `texts`, `samples`, `tests`, or `inputs`.")
    }

    private fun parseLines(s: String): List<String> =
        s.lineSequence()
            .map { it.trim() }
            .filter { it.isNotEmpty() && !it.startsWith("#") }
            .toList()
}

/** Serializes batch-test results to a JSON document suitable for thesis analysis. */
object BatchResultWriter {

    fun toJson(
        results: List<BatchTestEntry>,
        decisionThreshold: Float,
        exportedAtMillis: Long,
        modelLabels: Map<String, String> = mapOf(
            "base" to "mobilebert_dynamic_range.tflite",
            "encoder" to "mobilebert_encoder.tflite",
            "head" to "mobilebert_head_trainer.tflite",
        ),
    ): String {
        val root = JSONObject()
        root.put("exportedAtMillis", exportedAtMillis)
        root.put("decisionThreshold", decisionThreshold.toDouble())
        root.put("models", JSONObject(modelLabels.mapValues { it.value as Any }))

        var agreeCount = 0
        val rows = JSONArray()
        for (r in results) {
            val baseLabel = if (r.basePHate >= decisionThreshold) "HATE" else "NOTHATE"
            val perLabel = if (r.personalizedPHate >= decisionThreshold) "HATE" else "NOTHATE"
            val agree = baseLabel == perLabel
            if (agree) agreeCount += 1

            val row = JSONObject()
            row.put("text", r.text)

            val base = JSONObject()
            base.put("pHate", round4(r.basePHate))
            base.put("pNothate", round4(r.basePNothate))
            base.put("label", baseLabel)
            base.put("latencyMs", r.baseLatencyMs)

            val per = JSONObject()
            per.put("pHate", round4(r.personalizedPHate))
            per.put("pNothate", round4(r.personalizedPNothate))
            per.put("label", perLabel)
            per.put("latencyMs", r.personalizedLatencyMs)
            per.put("encoderLatencyMs", r.encoderLatencyMs)
            per.put("headLatencyMs", r.headLatencyMs)

            row.put("base", base)
            row.put("personalized", per)
            row.put("agree", agree)
            row.put("deltaPHate", round4(r.personalizedPHate - r.basePHate))

            rows.put(row)
        }

        val summary = JSONObject()
        summary.put("count", results.size)
        summary.put("agreeCount", agreeCount)
        summary.put("disagreeCount", results.size - agreeCount)
        if (results.isNotEmpty()) {
            summary.put("meanBaseLatencyMs", round4(results.map { it.baseLatencyMs }.average().toFloat()))
            summary.put("meanPersonalizedLatencyMs",
                round4(results.map { it.personalizedLatencyMs }.average().toFloat()))
            summary.put("meanEncoderLatencyMs", round4(results.map { it.encoderLatencyMs }.average().toFloat()))
            summary.put("meanHeadLatencyMs", round4(results.map { it.headLatencyMs }.average().toFloat()))
        }
        root.put("summary", summary)
        root.put("results", rows)

        return root.toString(2)
    }

    private fun round4(v: Float): Double {
        return Math.round(v.toDouble() * 10_000.0) / 10_000.0
    }
}