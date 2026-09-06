package com.example.hatespeechmonitor.sms

import com.example.hatespeechmonitor.ml.ClassificationResult

data class SmsMessage(
    val address: String,
    val body: String,
    val timestampMillis: Long,
)

data class ClassifiedSms(
    val message: SmsMessage,
    val result: ClassificationResult?,
)
