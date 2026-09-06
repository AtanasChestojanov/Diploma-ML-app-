package com.example.hatespeechmonitor

import android.app.Application
import com.example.hatespeechmonitor.data.SettingsRepository
import com.example.hatespeechmonitor.ml.InferenceEngine
import com.example.hatespeechmonitor.ml.TFLiteInferenceEngine
import com.example.hatespeechmonitor.ml.Tokenizer
import com.example.hatespeechmonitor.ml.VocabLoader
import com.example.hatespeechmonitor.ml.WordPieceTokenizer
import com.example.hatespeechmonitor.personalization.FeedbackStore
import com.example.hatespeechmonitor.personalization.InMemoryFeedbackStore
import com.example.hatespeechmonitor.personalization.PersonalizationEngine
import com.example.hatespeechmonitor.personalization.TFLitePersonalizationEngine
import com.example.hatespeechmonitor.sms.SmsRepository
import com.example.hatespeechmonitor.util.MemLogger
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Deferred
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async

/**
 * Process-wide container for the single [InferenceEngine] (the TFLite Interpreter
 * is expensive to instantiate, mmaps a 26 MB model, and is not thread-safe — one
 * instance, serialized access) and the [SettingsRepository].
 */
class HateSpeechApplication : Application() {

    val tokenizer: Tokenizer by lazy {
        WordPieceTokenizer(VocabLoader.loadFromAssets(this), maxLen = 64)
    }

    val inferenceEngine: InferenceEngine by lazy {
        TFLiteInferenceEngine(this, tokenizer)
    }

    val settingsRepository: SettingsRepository by lazy {
        SettingsRepository(this)
    }

    val smsRepository: SmsRepository by lazy {
        SmsRepository(this)
    }

    /**
     * Stub feedback store for the future personalization phase (see
     * [com.example.hatespeechmonitor.personalization.PersonalizationEngine]). Process-lifetime
     * only for now; swap in a persistent implementation when that phase lands without changing
     * any caller.
     */
    val feedbackStore: FeedbackStore by lazy { InMemoryFeedbackStore() }

    /** Process-lifetime scope used by [com.example.hatespeechmonitor.sms.SmsReceiver]. */
    val applicationScope: CoroutineScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)

    /**
     * The on-device personalization path (encoder + trainable head). Construction is async
     * because it mmaps two TFLite models and restores any persisted weights blob from disk.
     * Callers `await()` from a coroutine; subsequent awaits return the same instance.
     */
    val personalizationEngine: Deferred<PersonalizationEngine> by lazy {
        applicationScope.async {
            TFLitePersonalizationEngine.create(this@HateSpeechApplication, tokenizer)
        }
    }

    override fun onCreate() {
        super.onCreate()
        MemLogger.init(this)
        MemLogger.sample("app_start")
        // Kick the personalization engine load early so it's warm by the time the user
        // first navigates to Classify / SMS. Lazy is started by touching the property.
        personalizationEngine
    }
}
