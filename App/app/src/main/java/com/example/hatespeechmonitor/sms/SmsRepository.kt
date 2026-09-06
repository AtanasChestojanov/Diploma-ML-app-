package com.example.hatespeechmonitor.sms

import android.content.Context
import android.provider.Telephony
import com.example.hatespeechmonitor.ml.ClassificationResult
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.withContext

/**
 * Reads from the system SMS inbox and broadcasts incoming-SMS classifications.
 *
 * - [loadRecent] queries `content://sms/inbox` via [Telephony.Sms.Inbox] (requires READ_SMS).
 * - [recordIncoming] is called from [SmsReceiver] after classifying a freshly received SMS;
 *   it emits to [incoming], which the SMS screen collects to prepend live arrivals to the list.
 */
class SmsRepository(private val context: Context) {

    private val _incoming = MutableSharedFlow<ClassifiedSms>(extraBufferCapacity = 16)
    val incoming: SharedFlow<ClassifiedSms> = _incoming.asSharedFlow()

    suspend fun recordIncoming(message: SmsMessage, result: ClassificationResult) {
        _incoming.emit(ClassifiedSms(message, result))
    }

    suspend fun loadRecent(limit: Int = 30): List<SmsMessage> = withContext(Dispatchers.IO) {
        val cols = arrayOf(
            Telephony.Sms.ADDRESS,
            Telephony.Sms.BODY,
            Telephony.Sms.DATE,
        )
        val list = ArrayList<SmsMessage>(limit)
        context.contentResolver.query(
            Telephony.Sms.Inbox.CONTENT_URI,
            cols,
            null,
            null,
            "${Telephony.Sms.DATE} DESC",
        )?.use { c ->
            val addrIdx = c.getColumnIndexOrThrow(Telephony.Sms.ADDRESS)
            val bodyIdx = c.getColumnIndexOrThrow(Telephony.Sms.BODY)
            val dateIdx = c.getColumnIndexOrThrow(Telephony.Sms.DATE)
            while (c.moveToNext() && list.size < limit) {
                list.add(
                    SmsMessage(
                        address = c.getString(addrIdx) ?: "(unknown)",
                        body = c.getString(bodyIdx).orEmpty(),
                        timestampMillis = c.getLong(dateIdx),
                    )
                )
            }
        }
        list
    }
}
