package com.example.hatespeechmonitor.sms

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.provider.Telephony
import android.util.Log
import com.example.hatespeechmonitor.HateSpeechApplication
import kotlinx.coroutines.launch

class SmsReceiver : BroadcastReceiver() {

    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action != Telephony.Sms.Intents.SMS_RECEIVED_ACTION) return
        val app = context.applicationContext as? HateSpeechApplication ?: return

        val messages = Telephony.Sms.Intents.getMessagesFromIntent(intent) ?: return
        if (messages.isEmpty()) return

        // multipart SMS arrives as several PDUs from the same sender; join their bodies.
        val body = messages.joinToString(separator = "") { it.messageBody.orEmpty() }
        if (body.isBlank()) return
        val address = messages.first().originatingAddress ?: "(unknown)"
        val msg = SmsMessage(address = address, body = body, timestampMillis = System.currentTimeMillis())

        // BroadcastReceiver.onReceive runs on the main thread; goAsync keeps the process alive
        // long enough for the suspend classify() call to complete.
        val pending = goAsync()
        app.applicationScope.launch {
            try {
                val engine = app.personalizationEngine.await()
                val result = engine.classify(body)
                app.smsRepository.recordIncoming(msg, result)
            } catch (t: Throwable) {
                Log.e(TAG, "Failed to classify incoming SMS", t)
            } finally {
                pending.finish()
            }
        }
    }

    companion object {
        private const val TAG = "SmsReceiver"
    }
}
