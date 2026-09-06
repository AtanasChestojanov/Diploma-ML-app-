package com.example.hatespeechmonitor.personalization

import com.example.hatespeechmonitor.ml.Label
import org.json.JSONArray
import org.json.JSONObject

data class RawSample(val text: String, val label: Label)

object SampleImporter {

    fun parse(content: String): List<RawSample> {
        val stripped = content.removePrefix("\uFEFF").trim()
        if (stripped.isEmpty()) return emptyList()
        return when {
            stripped.startsWith("{") -> parseJsonObject(stripped)
            stripped.startsWith("[") -> parseJsonArray(stripped)
            else -> parseLines(stripped)
        }
    }

    private fun parseJsonObject(s: String): List<RawSample> {
        val obj = JSONObject(s)
        val out = ArrayList<RawSample>()
        // hate
        for (key in listOf("hate", "HATE", "Hate")) {
            if (obj.has(key)) {
                val arr = obj.getJSONArray(key)
                for (i in 0 until arr.length()) {
                    val text = arr.getString(i).trim()
                    if (text.isNotEmpty()) out.add(RawSample(text, Label.HATE))
                }
            }
        }
        // nothate
        for (key in listOf("nothate", "not_hate", "not-hate", "NOTHATE", "notHate")) {
            if (obj.has(key)) {
                val arr = obj.getJSONArray(key)
                for (i in 0 until arr.length()) {
                    val text = arr.getString(i).trim()
                    if (text.isNotEmpty()) out.add(RawSample(text, Label.NOTHATE))
                }
            }
        }
        require(out.isNotEmpty()) {
            "JSON object must contain `hate` and/or `nothate` arrays of strings."
        }
        return out
    }

    private fun parseJsonArray(s: String): List<RawSample> {
        val arr = JSONArray(s)
        val out = ArrayList<RawSample>(arr.length())
        for (i in 0 until arr.length()) {
            val obj = arr.getJSONObject(i)
            val text = obj.getString("text").trim()
            val label = parseLabel(obj.getString("label"))
                ?: error("Item $i: unknown label '${obj.optString("label")}'")
            if (text.isNotEmpty()) out.add(RawSample(text, label))
        }
        return out
    }

    private fun parseLines(s: String): List<RawSample> {
        val out = ArrayList<RawSample>()
        s.lineSequence().forEachIndexed { idx, rawLine ->
            val line = rawLine.trim()
            if (line.isEmpty() || line.startsWith("#")) return@forEachIndexed
            // header row
            if (idx == 0 && (line.equals("label,text", true) || line.equals("label\ttext", true))) {
                return@forEachIndexed
            }
            // JSONL
            if (line.startsWith("{")) {
                val obj = runCatching { JSONObject(line) }
                    .getOrElse { error("Line ${idx + 1}: bad JSON — ${it.message}") }
                val text = obj.getString("text").trim()
                val label = parseLabel(obj.getString("label"))
                    ?: error("Line ${idx + 1}: unknown label '${obj.optString("label")}'")
                if (text.isNotEmpty()) out.add(RawSample(text, label))
                return@forEachIndexed
            }
            // delimited: split on FIRST tab; if none, on FIRST comma
            val splitChar = if ('\t' in line) '\t' else ','
            val split = line.indexOf(splitChar)
            require(split > 0) {
                "Line ${idx + 1}: expected `label${if (splitChar == '\t') "<TAB>" else ","}text`, got: $line"
            }
            val labelStr = line.substring(0, split).trim().trim('"', '\'')
            val text = line.substring(split + 1).trim().trim('"', '\'')
            val label = parseLabel(labelStr) ?: error("Line ${idx + 1}: unknown label '$labelStr'")
            if (text.isNotEmpty()) out.add(RawSample(text, label))
        }
        require(out.isNotEmpty()) { "No samples found in file." }
        return out
    }

    private fun parseLabel(s: String): Label? = when (s.lowercase().trim()) {
        "hate", "h", "1", "true" -> Label.HATE
        "nothate", "not_hate", "not-hate", "notHate".lowercase(), "n", "0", "false" -> Label.NOTHATE
        else -> null
    }
}
