package com.example.hatespeechmonitor.ui.train

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
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
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.hatespeechmonitor.personalization.PersonalizationEngine
import com.example.hatespeechmonitor.personalization.TrainResult
import kotlinx.coroutines.delay

@Composable
fun TrainScreen(
    contentPadding: PaddingValues,
    vm: TrainViewModel = viewModel(factory = TrainViewModel.Factory),
) {
    val ui by vm.ui.collectAsState()
    var showFormatHelp by remember { mutableStateOf(false) }

    val importLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument(),
    ) { uri -> uri?.let(vm::importFromUri) }

    LaunchedEffect(ui.message) {
        if (ui.message != null) {
            delay(2_500)
            vm.dismissMessage()
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
        Text("Personalize on-device", style = MaterialTheme.typography.headlineSmall)
        Spacer(Modifier.height(4.dp))
        Text(
            "Train a small classifier head on this device from your feedback. The frozen " +
                "MobileBERT encoder produces embeddings; only the head's weights change. " +
                "Learning rate is fixed at ${PersonalizationEngine.ON_DEVICE_LR} (baked into the model at export).",
            style = MaterialTheme.typography.bodySmall,
        )
        Spacer(Modifier.height(16.dp))

        if (!ui.engineReady) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                CircularProgressIndicator(Modifier.height(16.dp), strokeWidth = 2.dp)
                Text("  Loading encoder + head trainer…")
            }
            Spacer(Modifier.height(12.dp))
        }

        // Feedback corpus + import
        Card(modifier = Modifier.fillMaxWidth()) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text("Feedback corpus", style = MaterialTheme.typography.titleMedium)
                Spacer(Modifier.height(8.dp))
                Text("Total samples: ${ui.sampleCount}")
                Row(verticalAlignment = Alignment.CenterVertically) {
                    AssistChip(onClick = {}, label = { Text("HATE: ${ui.hateCount}") })
                    Text("  ", style = MaterialTheme.typography.bodySmall)
                    AssistChip(onClick = {}, label = { Text("NOTHATE: ${ui.nothateCount}") })
                }
                Spacer(Modifier.height(12.dp))

                // Import row
                Row(
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Button(
                        onClick = { importLauncher.launch(arrayOf("*/*")) },
                        enabled = ui.engineReady && !ui.importing && !ui.training,
                        modifier = Modifier.weight(1f),
                    ) {
                        if (ui.importing) CircularProgressIndicator(Modifier.height(18.dp), strokeWidth = 2.dp)
                        else Text("Import samples from file")
                    }
                    OutlinedButton(
                        onClick = vm::clearFeedback,
                        enabled = ui.sampleCount > 0 && !ui.training && !ui.importing,
                    ) { Text("Clear") }
                }
                if (ui.importing && ui.importTotal > 0) {
                    Spacer(Modifier.height(8.dp))
                    LinearProgressIndicator(
                        progress = { ui.importProgress },
                        modifier = Modifier.fillMaxWidth(),
                    )
                    Text(
                        "Embedding ${ui.importDone} / ${ui.importTotal}…",
                        style = MaterialTheme.typography.bodySmall,
                    )
                }
                TextButton(onClick = { showFormatHelp = !showFormatHelp }) {
                    Text(if (showFormatHelp) "Hide supported formats" else "Show supported formats")
                }
                if (showFormatHelp) FormatHelp()
            }
        }
        Spacer(Modifier.height(16.dp))

        // Training controls
        Text("Training", style = MaterialTheme.typography.titleMedium)
        Spacer(Modifier.height(8.dp))
        Text("Epochs: ${ui.epochs}")
        Slider(
            value = ui.epochs.toFloat(),
            onValueChange = { vm.setEpochs(it.toInt()) },
            valueRange = 1f..20f,
            steps = 18,
            enabled = !ui.training,
        )
        Text("Batch size: ${ui.batchSize}")
        Slider(
            value = ui.batchSize.toFloat(),
            onValueChange = { vm.setBatchSize(it.toInt()) },
            valueRange = 1f..32f,
            steps = 30,
            enabled = !ui.training,
        )
        Spacer(Modifier.height(8.dp))
        Button(
            onClick = vm::train,
            enabled = ui.engineReady && !ui.training && ui.sampleCount > 0,
            modifier = Modifier.fillMaxWidth(),
        ) {
            if (ui.training) CircularProgressIndicator(Modifier.height(18.dp), strokeWidth = 2.dp)
            else Text("Train now")
        }

        ui.lastResult?.let { result ->
            Spacer(Modifier.height(12.dp))
            TrainResultCard(result)
        }

        Spacer(Modifier.height(16.dp))
        HorizontalDivider()
        Spacer(Modifier.height(16.dp))

        Text("Personalization state", style = MaterialTheme.typography.titleMedium)
        Spacer(Modifier.height(4.dp))
        Text(
            if (ui.hasPersistedWeights) "Personalized weights are persisted on this device."
            else "No personalized weights persisted (using factory weights).",
            style = MaterialTheme.typography.bodySmall,
        )
        Spacer(Modifier.height(8.dp))
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(
                onClick = vm::save,
                enabled = ui.engineReady && !ui.saving && !ui.resetting,
                modifier = Modifier.weight(1f),
            ) {
                if (ui.saving) CircularProgressIndicator(Modifier.height(18.dp), strokeWidth = 2.dp)
                else Text("Save weights")
            }
            OutlinedButton(
                onClick = vm::resetToFactory,
                enabled = ui.engineReady && !ui.saving && !ui.resetting,
                modifier = Modifier.weight(1f),
            ) {
                if (ui.resetting) CircularProgressIndicator(Modifier.height(18.dp), strokeWidth = 2.dp)
                else Text("Reset to factory")
            }
        }

        ui.message?.let {
            Spacer(Modifier.height(8.dp))
            Text(it, color = MaterialTheme.colorScheme.primary, style = MaterialTheme.typography.bodySmall)
        }
        ui.error?.let {
            Spacer(Modifier.height(8.dp))
            Text("Error: $it", color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodyMedium)
        }
        Spacer(Modifier.height(24.dp))
    }
}

@Composable
private fun FormatHelp() {
    Spacer(Modifier.height(4.dp))
    Column {
        Text(
            "Pick any text file. Format is auto-detected; labels are case-insensitive " +
                "(hate/h/1 → HATE; nothate/not_hate/n/0 → NOTHATE).",
            style = MaterialTheme.typography.bodySmall,
        )
        Spacer(Modifier.height(8.dp))
        Text("JSON object:", style = MaterialTheme.typography.labelSmall, fontWeight = FontWeight.SemiBold)
        Text(
            """{"hate":["...","..."],"nothate":["...","..."]}""",
            style = MaterialTheme.typography.bodySmall,
        )
        Spacer(Modifier.height(6.dp))
        Text("JSONL (one object per line):", style = MaterialTheme.typography.labelSmall, fontWeight = FontWeight.SemiBold)
        Text(
            """{"label":"hate","text":"..."}""",
            style = MaterialTheme.typography.bodySmall,
        )
        Spacer(Modifier.height(6.dp))
        Text("CSV / TSV (header optional):", style = MaterialTheme.typography.labelSmall, fontWeight = FontWeight.SemiBold)
        Text(
            "label,text\nhate,I dislike them all\nnothate,have a nice day",
            style = MaterialTheme.typography.bodySmall,
        )
        Spacer(Modifier.height(6.dp))
        Text(
            "Text may contain commas — only the FIRST delimiter is the split point. " +
                "Lines starting with # are ignored.",
            style = MaterialTheme.typography.bodySmall,
        )
    }
}

@Composable
private fun TrainResultCard(result: TrainResult) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text("Last training run", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.SemiBold)
            Spacer(Modifier.height(6.dp))
            Text("Epochs: ${result.epochs}")
            Text("Samples: ${result.samplesUsed}")
            Text("Initial loss → final loss: ${"%.4f".format(result.initialLoss)} → ${"%.4f".format(result.finalLoss)}")
            Text("Embedding time: ${result.embeddingMillis} ms")
            Text("Training time (head only): ${result.trainingMillis} ms")
        }
    }
}
