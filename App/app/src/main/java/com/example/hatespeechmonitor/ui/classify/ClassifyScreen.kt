package com.example.hatespeechmonitor.ui.classify

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AssistChip
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ElevatedCard
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.hatespeechmonitor.ml.BatchTestEntry
import com.example.hatespeechmonitor.ml.BenchmarkResult
import com.example.hatespeechmonitor.ml.ClassificationResult
import com.example.hatespeechmonitor.ml.Label
import kotlinx.coroutines.delay
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

@Composable
fun ClassifyScreen(
    contentPadding: PaddingValues,
    vm: ClassifyViewModel = viewModel(factory = ClassifyViewModel.Factory),
) {
    val ui by vm.ui.collectAsState()
    val threshold by vm.threshold.collectAsState()

    LaunchedEffect(ui.feedbackMessage) {
        if (ui.feedbackMessage != null) {
            delay(2_500)
            vm.dismissFeedbackMessage()
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(contentPadding)
            .padding(horizontal = 16.dp)
            .verticalScroll(rememberScrollState()),
    ) {
        Spacer(Modifier.height(12.dp))
        Text("Classify text", style = MaterialTheme.typography.headlineSmall)
        Spacer(Modifier.height(12.dp))

        OutlinedTextField(
            value = ui.text,
            onValueChange = vm::onTextChange,
            modifier = Modifier
                .fillMaxWidth()
                .height(140.dp),
            label = { Text("Message") },
            placeholder = { Text("Type or paste a message to classify…") },
            singleLine = false,
        )
        Spacer(Modifier.height(12.dp))
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Button(
                onClick = vm::classify,
                enabled = !ui.classifying && !ui.benchmarking && ui.text.isNotBlank(),
                modifier = Modifier.weight(1f),
            ) {
                if (ui.classifying) CircularProgressIndicator(Modifier.height(18.dp), strokeWidth = 2.dp)
                else Text("Classify")
            }
            OutlinedButton(
                onClick = vm::clearResult,
                enabled = ui.personalizedResult != null || ui.baseResult != null || ui.benchmark != null,
            ) { Text("Clear") }
        }

        ui.error?.let { msg ->
            Spacer(Modifier.height(8.dp))
            Text(
                "Error: $msg",
                color = MaterialTheme.colorScheme.error,
                style = MaterialTheme.typography.bodyMedium,
            )
        }

        AnimatedVisibility(visible = ui.personalizedResult != null) {
            ui.personalizedResult?.let { ResultCard("Personalized (encoder + trainable head)", it, threshold, primary = true) }
        }
        AnimatedVisibility(visible = ui.baseResult != null) {
            ui.baseResult?.let { ResultCard("Base (fused MobileBERT, non-personalized)", it, threshold, primary = false) }
        }

        AnimatedVisibility(visible = ui.personalizedResult != null) {
            FeedbackBar(
                feedbackMessage = ui.feedbackMessage,
                onMark = vm::submitFeedback,
            )
        }

        Spacer(Modifier.height(16.dp))
        HorizontalDivider()
        Spacer(Modifier.height(16.dp))
        BatchTestSection(
            testing = ui.batchTesting,
            done = ui.batchDone,
            total = ui.batchTotal,
            progress = ui.batchProgress,
            results = ui.batchResults,
            savedTo = ui.batchSavedTo,
            threshold = threshold,
            onPick = vm::runBatchTest,
            onSave = vm::saveBatchResults,
            onClear = vm::clearBatchResults,
        )

        Spacer(Modifier.height(16.dp))
        HorizontalDivider()
        Spacer(Modifier.height(16.dp))
        BenchmarkSection(
            running = ui.benchmarking,
            result = ui.benchmark,
            onRun = vm::runBenchmark,
        )
        Spacer(Modifier.height(24.dp))
    }
}

@Composable
private fun BatchTestSection(
    testing: Boolean,
    done: Int,
    total: Int,
    progress: Float,
    results: List<BatchTestEntry>?,
    savedTo: String?,
    threshold: Float,
    onPick: (android.net.Uri) -> Unit,
    onSave: (android.net.Uri) -> Unit,
    onClear: () -> Unit,
) {
    var showFormatHelp by remember { mutableStateOf(false) }

    val inputLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument(),
    ) { uri -> uri?.let(onPick) }

    val saveLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.CreateDocument("application/json"),
    ) { uri -> uri?.let(onSave) }

    Column {
        Text("Batch test", style = MaterialTheme.typography.titleMedium)
        Spacer(Modifier.height(4.dp))
        Text(
            "Pick a JSON/text file of sentences (no labels needed). The app runs each sentence " +
                "through both the base and personalized engines and exports the results as JSON for analysis.",
            style = MaterialTheme.typography.bodySmall,
        )
        Spacer(Modifier.height(8.dp))
        Row(
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Button(
                onClick = { inputLauncher.launch(arrayOf("*/*")) },
                enabled = !testing,
                modifier = Modifier.weight(1f),
            ) {
                if (testing) CircularProgressIndicator(Modifier.height(18.dp), strokeWidth = 2.dp)
                else Text("Load + test sentences")
            }
            OutlinedButton(
                onClick = onClear,
                enabled = !testing && (results != null),
            ) { Text("Clear") }
        }
        if (testing && total > 0) {
            Spacer(Modifier.height(8.dp))
            LinearProgressIndicator(
                progress = { progress },
                modifier = Modifier.fillMaxWidth(),
            )
            Text("Classifying $done / $total…", style = MaterialTheme.typography.bodySmall)
        }
        AnimatedVisibility(visible = results != null) {
            results?.let { BatchResultSummaryCard(it, threshold, savedTo) {
                val ts = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
                saveLauncher.launch("test_results_$ts.json")
            } }
        }
        Spacer(Modifier.height(4.dp))
        androidx.compose.material3.TextButton(onClick = { showFormatHelp = !showFormatHelp }) {
            Text(if (showFormatHelp) "Hide supported formats" else "Show supported formats")
        }
        if (showFormatHelp) BatchInputFormatHelp()
    }
}

@Composable
private fun BatchResultSummaryCard(
    results: List<BatchTestEntry>,
    threshold: Float,
    savedTo: String?,
    onSave: () -> Unit,
) {
    val n = results.size
    val agree = results.count {
        (it.basePHate >= threshold) == (it.personalizedPHate >= threshold)
    }
    val disagreeShift = results
        .filter { (it.basePHate >= threshold) != (it.personalizedPHate >= threshold) }
        .map { it.personalizedPHate - it.basePHate }
    val baseHate = results.count { it.basePHate >= threshold }
    val perHate = results.count { it.personalizedPHate >= threshold }

    Spacer(Modifier.height(12.dp))
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text("Results", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.SemiBold)
            Spacer(Modifier.height(6.dp))
            Text("Sentences tested: $n")
            Text("Base labels HATE: $baseHate / $n")
            Text("Personalized labels HATE: $perHate / $n")
            Text("Agreement: $agree / $n (${if (n == 0) 0 else 100 * agree / n}%)")
            if (disagreeShift.isNotEmpty()) {
                val mean = disagreeShift.average()
                Text(
                    "On disagreements, personalized P(hate) shifts by mean ${"%+.3f".format(mean)} vs base",
                    style = MaterialTheme.typography.bodySmall,
                )
            }
            Spacer(Modifier.height(8.dp))
            Button(onClick = onSave, modifier = Modifier.fillMaxWidth()) {
                Text("Save results JSON…")
            }
            savedTo?.let {
                Spacer(Modifier.height(6.dp))
                Text(
                    "Saved to: $it",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.primary,
                )
            }
        }
    }
}

@Composable
private fun BatchInputFormatHelp() {
    Spacer(Modifier.height(4.dp))
    Column {
        Text(
            "Pick any text file. Format is auto-detected — no labels needed, only sentences.",
            style = MaterialTheme.typography.bodySmall,
        )
        Spacer(Modifier.height(6.dp))
        Text("JSON array:", style = MaterialTheme.typography.labelSmall, fontWeight = FontWeight.SemiBold)
        Text("""["sentence one","sentence two", ...]""", style = MaterialTheme.typography.bodySmall)
        Spacer(Modifier.height(6.dp))
        Text("JSON object:", style = MaterialTheme.typography.labelSmall, fontWeight = FontWeight.SemiBold)
        Text("""{"sentences":["...","..."]}""", style = MaterialTheme.typography.bodySmall)
        Spacer(Modifier.height(6.dp))
        Text("Plain text:", style = MaterialTheme.typography.labelSmall, fontWeight = FontWeight.SemiBold)
        Text("one sentence per line; blank lines and lines starting with # are ignored.",
            style = MaterialTheme.typography.bodySmall)
    }
}

@Composable
private fun ResultCard(
    title: String,
    result: ClassificationResult,
    threshold: Float,
    primary: Boolean,
) {
    val label = result.labelAt(threshold)
    val labelColor = if (label == Label.HATE) Color(0xFFD32F2F) else Color(0xFF2E7D32)
    Spacer(Modifier.height(12.dp))
    val cardModifier = Modifier.fillMaxWidth()
    if (primary) {
        ElevatedCard(modifier = cardModifier) { ResultCardContent(title, result, label, labelColor, threshold) }
    } else {
        Card(modifier = cardModifier, colors = CardDefaults.cardColors()) {
            ResultCardContent(title, result, label, labelColor, threshold)
        }
    }
}

@Composable
private fun ResultCardContent(
    title: String,
    result: ClassificationResult,
    label: Label,
    labelColor: Color,
    threshold: Float,
) {
    Column(modifier = Modifier.padding(16.dp)) {
        Text(title, style = MaterialTheme.typography.labelMedium)
        Spacer(Modifier.height(6.dp))
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(
                text = label.name,
                style = MaterialTheme.typography.headlineMedium,
                color = labelColor,
                fontWeight = FontWeight.Bold,
            )
            Spacer(Modifier.weight(1f))
            if (result.isCold) AssistChip(onClick = {}, label = { Text("cold") })
        }
        Spacer(Modifier.height(8.dp))
        Text("P(hate) = ${"%.4f".format(result.pHate)}")
        Text("P(nothate) = ${"%.4f".format(result.pNothate)}")
        Spacer(Modifier.height(6.dp))
        LinearProgressIndicator(
            progress = { result.pHate.coerceIn(0f, 1f) },
            modifier = Modifier.fillMaxWidth(),
        )
        Spacer(Modifier.height(8.dp))
        val splitNote = if (result.encoderLatencyMs != null && result.headLatencyMs != null) {
            " (encoder ${result.encoderLatencyMs} ms + head ${result.headLatencyMs} ms)"
        } else ""
        Text(
            "Latency: ${result.latencyMs} ms$splitNote" +
                if (result.isCold) " — cold, includes tensor allocation" else "",
            style = MaterialTheme.typography.bodySmall,
        )
        Text(
            "Threshold ${"%.2f".format(threshold)} → ${if (result.pHate >= threshold) "HATE" else "NOTHATE"}",
            style = MaterialTheme.typography.bodySmall,
        )
    }
}

@Composable
private fun FeedbackBar(
    feedbackMessage: String?,
    onMark: (Label) -> Unit,
) {
    Spacer(Modifier.height(12.dp))
    Column {
        Text("Feedback", style = MaterialTheme.typography.titleSmall)
        Spacer(Modifier.height(2.dp))
        Text(
            "Tell the on-device learner the correct label for this text. " +
                "Feedback feeds the next train run on the Train tab.",
            style = MaterialTheme.typography.bodySmall,
        )
        Spacer(Modifier.height(8.dp))
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            OutlinedButton(onClick = { onMark(Label.HATE) }, modifier = Modifier.weight(1f)) {
                Text("Mark as HATE")
            }
            OutlinedButton(onClick = { onMark(Label.NOTHATE) }, modifier = Modifier.weight(1f)) {
                Text("Mark as NOT HATE")
            }
        }
        AnimatedVisibility(visible = feedbackMessage != null) {
            Text(
                feedbackMessage.orEmpty(),
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.primary,
                modifier = Modifier.padding(top = 6.dp),
            )
        }
    }
}

@Composable
private fun BenchmarkSection(
    running: Boolean,
    result: BenchmarkResult?,
    onRun: (Int) -> Unit,
) {
    var nWarm by remember { mutableStateOf(50f) }
    Column {
        Text("Benchmark (base engine)", style = MaterialTheme.typography.titleMedium)
        Spacer(Modifier.height(4.dp))
        Text(
            "Runs the current text (or a default sentence) repeatedly on the fused base " +
                "model. The first call is cold (includes tensor allocation); the remaining are warm.",
            style = MaterialTheme.typography.bodySmall,
        )
        Spacer(Modifier.height(8.dp))
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("Warm runs: ${nWarm.toInt()}", modifier = Modifier.weight(1f))
        }
        Slider(
            value = nWarm,
            onValueChange = { nWarm = it },
            valueRange = 10f..500f,
            steps = 48,
            enabled = !running,
        )
        Spacer(Modifier.height(8.dp))
        Button(
            onClick = { onRun(nWarm.toInt()) },
            enabled = !running,
        ) {
            if (running) CircularProgressIndicator(Modifier.height(18.dp), strokeWidth = 2.dp)
            else Text("Run benchmark")
        }
        AnimatedVisibility(visible = result != null) {
            result?.let { BenchmarkResultCard(it) }
        }
    }
}

@Composable
private fun BenchmarkResultCard(result: BenchmarkResult) {
    Spacer(Modifier.height(12.dp))
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text("Benchmark result", style = MaterialTheme.typography.titleSmall)
            Spacer(Modifier.height(8.dp))
            result.coldMs?.let {
                Text("cold: $it ms")
            } ?: Text("cold: (engine was already warm)", style = MaterialTheme.typography.bodySmall)
            Text("warm n = ${result.warmCount}")
            Text("warm mean = ${"%.2f".format(result.warmMeanMs)} ms")
            Text("warm p95  = ${result.warmP95Ms} ms")
            Text("warm min  = ${result.warmMinMs} ms")
            Text("warm max  = ${result.warmMaxMs} ms")
        }
    }
}
