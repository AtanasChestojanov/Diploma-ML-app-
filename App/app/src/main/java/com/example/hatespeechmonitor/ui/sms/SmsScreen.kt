package com.example.hatespeechmonitor.ui.sms

import android.Manifest
import android.content.pm.PackageManager
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
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.AssistChip
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.hatespeechmonitor.ml.Label
import com.example.hatespeechmonitor.sms.ClassifiedSms
import kotlinx.coroutines.delay
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

private val SMS_PERMS = arrayOf(
    Manifest.permission.RECEIVE_SMS,
    Manifest.permission.READ_SMS,
)

@Composable
fun SmsScreen(
    contentPadding: PaddingValues,
    vm: SmsViewModel = viewModel(factory = SmsViewModel.Factory),
) {
    val context = LocalContext.current
    val ui by vm.ui.collectAsState()
    val threshold by vm.threshold.collectAsState()

    LaunchedEffect(ui.feedbackMessage) {
        if (ui.feedbackMessage != null) {
            delay(2_500)
            vm.dismissFeedbackMessage()
        }
    }

    val launcher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestMultiplePermissions(),
    ) { grants ->
        val granted = SMS_PERMS.all { grants[it] == true }
        vm.onPermissionsResult(granted)
    }

    LaunchedEffect(Unit) {
        val granted = SMS_PERMS.all {
            ContextCompat.checkSelfPermission(context, it) == PackageManager.PERMISSION_GRANTED
        }
        vm.onPermissionsResult(granted)
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(contentPadding)
            .padding(horizontal = 16.dp),
    ) {
        Spacer(Modifier.height(12.dp))
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("SMS monitoring", style = MaterialTheme.typography.headlineSmall)
            Spacer(Modifier.weight(1f))
            if (ui.permissionGranted) {
                OutlinedButton(
                    onClick = vm::refresh,
                    enabled = !ui.loading,
                ) { Text("Refresh") }
            }
        }
        Spacer(Modifier.height(4.dp))
        Text(
            "Classified via the personalized engine. Feedback here trains your on-device head.",
            style = MaterialTheme.typography.bodySmall,
        )
        Spacer(Modifier.height(8.dp))

        if (!ui.permissionGranted) {
            PermissionPrompt(onRequest = { launcher.launch(SMS_PERMS) })
            return@Column
        }

        if (ui.loading) {
            LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
            Spacer(Modifier.height(8.dp))
        }
        ui.error?.let { msg ->
            Text(
                "Error: $msg",
                color = MaterialTheme.colorScheme.error,
                style = MaterialTheme.typography.bodyMedium,
            )
            Spacer(Modifier.height(8.dp))
        }
        ui.feedbackMessage?.let {
            Text(it, color = MaterialTheme.colorScheme.primary, style = MaterialTheme.typography.bodySmall)
            Spacer(Modifier.height(4.dp))
        }

        if (ui.items.isEmpty() && !ui.loading) {
            Text(
                "No SMS messages in the inbox yet. Incoming messages will appear here automatically.",
                style = MaterialTheme.typography.bodyMedium,
            )
        } else {
            val dateFmt = remember { SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.getDefault()) }
            LazyColumn(
                verticalArrangement = Arrangement.spacedBy(8.dp),
                modifier = Modifier.fillMaxSize(),
            ) {
                items(
                    ui.items,
                    key = { it.message.timestampMillis.toString() + it.message.address },
                ) { item ->
                    SmsItemCard(
                        item = item,
                        threshold = threshold,
                        dateFmt = dateFmt,
                        onMark = { label -> vm.submitFeedback(item.message, label) },
                    )
                }
            }
        }
    }
}

@Composable
private fun PermissionPrompt(onRequest: () -> Unit) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text("SMS permissions required", style = MaterialTheme.typography.titleMedium)
            Spacer(Modifier.height(8.dp))
            Text(
                "This screen needs RECEIVE_SMS (to classify incoming messages as they arrive) " +
                    "and READ_SMS (to load and classify recent inbox messages). " +
                    "If you deny either, the rest of the app keeps working in manual-only mode.",
                style = MaterialTheme.typography.bodyMedium,
            )
            Spacer(Modifier.height(12.dp))
            Button(onClick = onRequest) { Text("Grant permissions") }
        }
    }
}

@Composable
private fun SmsItemCard(
    item: ClassifiedSms,
    threshold: Float,
    dateFmt: SimpleDateFormat,
    onMark: (Label) -> Unit,
) {
    val result = item.result
    val label = result?.labelAt(threshold)
    val labelColor = when (label) {
        Label.HATE -> Color(0xFFD32F2F)
        Label.NOTHATE -> Color(0xFF2E7D32)
        null -> MaterialTheme.colorScheme.onSurfaceVariant
    }
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(12.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(
                    text = item.message.address,
                    style = MaterialTheme.typography.titleSmall,
                    fontWeight = FontWeight.SemiBold,
                )
                Spacer(Modifier.weight(1f))
                Text(
                    text = dateFmt.format(Date(item.message.timestampMillis)),
                    style = MaterialTheme.typography.bodySmall,
                )
            }
            Spacer(Modifier.height(4.dp))
            Text(item.message.body, style = MaterialTheme.typography.bodyMedium)
            Spacer(Modifier.height(6.dp))
            if (result == null) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    CircularProgressIndicator(modifier = Modifier.height(14.dp), strokeWidth = 2.dp)
                    Text("  classifying…", style = MaterialTheme.typography.bodySmall)
                }
            } else {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    AssistChip(
                        onClick = {},
                        label = {
                            Text(
                                label?.name ?: "?",
                                color = labelColor,
                                fontWeight = FontWeight.Bold,
                            )
                        },
                    )
                    Spacer(Modifier.weight(1f))
                    Text(
                        "P(hate) = ${"%.3f".format(result.pHate)}",
                        style = MaterialTheme.typography.bodySmall,
                    )
                }
                Spacer(Modifier.height(4.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                    TextButton(onClick = { onMark(Label.HATE) }) { Text("Mark HATE") }
                    TextButton(onClick = { onMark(Label.NOTHATE) }) { Text("Mark NOT HATE") }
                }
            }
        }
    }
}
