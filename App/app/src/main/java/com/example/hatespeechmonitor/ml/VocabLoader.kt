package com.example.hatespeechmonitor.ml

import android.content.Context

object VocabLoader {

    fun loadFromAssets(context: Context, assetName: String = "vocab.txt"): Map<String, Int> {
        val map = HashMap<String, Int>(35_000)
        context.assets.open(assetName).bufferedReader(Charsets.UTF_8).useLines { lines ->
            for ((index, raw) in lines.withIndex()) {
                val line = if (index == 0 && raw.isNotEmpty() && raw[0] == '\uFEFF') raw.substring(1) else raw
                map[line] = index
            }
        }
        require(map["[PAD]"] == 0) { "vocab.txt malformed: [PAD] is not at line 0" }
        require(map["[UNK]"] == 100) { "vocab.txt malformed: [UNK] is not at line 100" }
        require(map["[CLS]"] == 101) { "vocab.txt malformed: [CLS] is not at line 101" }
        require(map["[SEP]"] == 102) { "vocab.txt malformed: [SEP] is not at line 102" }
        return map
    }
}
