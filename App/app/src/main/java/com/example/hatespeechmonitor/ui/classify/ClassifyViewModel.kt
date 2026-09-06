package com.example.hatespeechmonitor.ui.classify

import android.content.Context
import android.net.Uri
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.ViewModelProvider.AndroidViewModelFactory.Companion.APPLICATION_KEY
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.initializer
import androidx.lifecycle.viewmodel.viewModelFactory
import com.example.hatespeechmonitor.HateSpeechApplication
import com.example.hatespeechmonitor.data.SettingsRepository
import com.example.hatespeechmonitor.ml.BatchInputParser
import com.example.hatespeechmonitor.ml.BatchResultWriter
import com.example.hatespeechmonitor.ml.BatchTestEntry
import com.example.hatespeechmonitor.ml.Benchmark
import com.example.hatespeechmonitor.ml.BenchmarkResult
import com.example.hatespeechmonitor.ml.ClassificationResult
import com.example.hatespeechmonitor.ml.InferenceEngine
import com.example.hatespeechmonitor.ml.Label
import com.example.hatespeechmonitor.personalization.FeedbackSample
import com.example.hatespeechmonitor.personalization.FeedbackStore
import com.example.hatespeechmonitor.personalization.PersonalizationEngine
import kotlinx.coroutines.Deferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

data class ClassifyUiState(
    val text: String = "",
    val classifying: Boolean = false,
    val benchmarking: Boolean = false,
    val personalizedResult: ClassificationResult? = null,
    val baseResult: ClassificationResult? = null,
    val benchmark: BenchmarkResult? = null,
    val error: String? = null,
    val feedbackMessage: String? = null,
    // Batch test state
    val batchTesting: Boolean = false,
    val batchDone: Int = 0,
    val batchTotal: Int = 0,
    val batchResults: List<BatchTestEntry>? = null,
    val batchSavedTo: String? = null,
) {
    val batchProgress: Float get() = if (batchTotal == 0) 0f else batchDone.toFloat() / batchTotal
}

class ClassifyViewModel(
    private val appContext: Context,
    private val baseEngine: InferenceEngine,
    private val personalizationEngineDeferred: Deferred<PersonalizationEngine>,
    private val feedbackStore: FeedbackStore,
    settings: SettingsRepository,
) : ViewModel() {

    private val _ui = MutableStateFlow(ClassifyUiState())
    val ui: StateFlow<ClassifyUiState> = _ui.asStateFlow()

    val threshold: StateFlow<Float> = settings.decisionThreshold
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), SettingsRepository.DEFAULT_THRESHOLD)

    fun onTextChange(text: String) {
        _ui.update { it.copy(text = text, error = null) }
    }

    fun classify() {
        val text = _ui.value.text.trim()
        if (text.isEmpty()) return
        viewModelScope.launch {
            _ui.update {
                it.copy(classifying = true, error = null, personalizedResult = null, baseResult = null)
            }
            runCatching {
                val pEngine = personalizationEngineDeferred.await()
                coroutineScope {
                    val basePromise = async { baseEngine.classify(text) }
                    val personalizedPromise = async { pEngine.classify(text) }
                    personalizedPromise.await() to basePromise.await()
                }
            }.onSuccess { (personalized, base) ->
                _ui.update {
                    it.copy(classifying = false, personalizedResult = personalized, baseResult = base)
                }
            }.onFailure { e ->
                _ui.update { it.copy(classifying = false, error = e.message ?: e::class.java.simpleName) }
            }
        }
    }

    fun submitFeedback(label: Label) {
        val text = _ui.value.text.trim()
        if (text.isEmpty()) return
        viewModelScope.launch {
            runCatching {
                val pEngine = personalizationEngineDeferred.await()
                val embedding = pEngine.embed(text)
                feedbackStore.add(FeedbackSample(text = text, userLabel = label, embedding = embedding))
            }.onSuccess {
                _ui.update { it.copy(feedbackMessage = "Recorded as ${label.name}") }
            }.onFailure { e ->
                _ui.update { it.copy(error = e.message ?: e::class.java.simpleName) }
            }
        }
    }

    fun dismissFeedbackMessage() {
        _ui.update { it.copy(feedbackMessage = null) }
    }

    fun runBenchmark(warmRuns: Int) {
        val text = _ui.value.text.trim().ifEmpty { DEFAULT_BENCH_TEXT }
        viewModelScope.launch {
            _ui.update { it.copy(benchmarking = true, benchmark = null, error = null) }
            runCatching { Benchmark(baseEngine).run(text, warmRuns) }
                .onSuccess { b -> _ui.update { it.copy(benchmarking = false, benchmark = b) } }
                .onFailure { e ->
                    _ui.update { it.copy(benchmarking = false, error = e.message ?: e::class.java.simpleName) }
                }
        }
    }

    fun clearResult() {
        _ui.update { it.copy(personalizedResult = null, baseResult = null, benchmark = null) }
    }

    // ---------------- Batch test ----------------

    fun runBatchTest(uri: Uri) {
        viewModelScope.launch {
            _ui.update {
                it.copy(
                    batchTesting = true,
                    batchDone = 0,
                    batchTotal = 0,
                    batchResults = null,
                    batchSavedTo = null,
                    error = null,
                )
            }
            runCatching {
                val content = withContext(Dispatchers.IO) {
                    appContext.contentResolver.openInputStream(uri)
                        ?.bufferedReader(Charsets.UTF_8)?.use { it.readText() }
                        ?: error("Could not open the selected file.")
                }
                val sentences = BatchInputParser.parse(content)
                _ui.update { it.copy(batchTotal = sentences.size) }

                val pEngine = personalizationEngineDeferred.await()
                val results = ArrayList<BatchTestEntry>(sentences.size)
                for ((i, text) in sentences.withIndex()) {
                    val (base, per) = coroutineScope {
                        val baseAsync = async { baseEngine.classify(text) }
                        val perAsync = async { pEngine.classify(text) }
                        baseAsync.await() to perAsync.await()
                    }
                    results.add(
                        BatchTestEntry(
                            text = text,
                            basePHate = base.pHate,
                            basePNothate = base.pNothate,
                            baseLatencyMs = base.latencyMs,
                            personalizedPHate = per.pHate,
                            personalizedPNothate = per.pNothate,
                            personalizedLatencyMs = per.latencyMs,
                            encoderLatencyMs = per.encoderLatencyMs ?: 0L,
                            headLatencyMs = per.headLatencyMs ?: 0L,
                        )
                    )
                    _ui.update { it.copy(batchDone = i + 1) }
                }
                results
            }.onSuccess { results ->
                _ui.update { it.copy(batchTesting = false, batchResults = results) }
            }.onFailure { e ->
                _ui.update { it.copy(batchTesting = false, error = e.message ?: e::class.java.simpleName) }
            }
        }
    }

    fun saveBatchResults(uri: Uri) {
        val results = _ui.value.batchResults ?: return
        viewModelScope.launch {
            runCatching {
                val json = BatchResultWriter.toJson(
                    results = results,
                    decisionThreshold = threshold.value,
                    exportedAtMillis = System.currentTimeMillis(),
                )
                withContext(Dispatchers.IO) {
                    appContext.contentResolver.openOutputStream(uri)
                        ?.bufferedWriter(Charsets.UTF_8)?.use { it.write(json) }
                        ?: error("Could not open output stream.")
                }
                uri
            }.onSuccess { savedUri ->
                _ui.update { it.copy(batchSavedTo = savedUri.toString()) }
            }.onFailure { e ->
                _ui.update { it.copy(error = e.message ?: e::class.java.simpleName) }
            }
        }
    }

    fun clearBatchResults() {
        _ui.update { it.copy(batchResults = null, batchSavedTo = null, batchDone = 0, batchTotal = 0) }
    }

    companion object {
        private const val DEFAULT_BENCH_TEXT = "the quick brown fox jumps over the lazy dog"

        val Factory: ViewModelProvider.Factory = viewModelFactory {
            initializer {
                val app = this[APPLICATION_KEY] as HateSpeechApplication
                ClassifyViewModel(
                    appContext = app.applicationContext,
                    baseEngine = app.inferenceEngine,
                    personalizationEngineDeferred = app.personalizationEngine,
                    feedbackStore = app.feedbackStore,
                    settings = app.settingsRepository,
                )
            }
        }
    }
}