package com.example.hatespeechmonitor.ui.train

import android.content.Context
import android.net.Uri
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.ViewModelProvider.AndroidViewModelFactory.Companion.APPLICATION_KEY
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.initializer
import androidx.lifecycle.viewmodel.viewModelFactory
import com.example.hatespeechmonitor.HateSpeechApplication
import com.example.hatespeechmonitor.ml.Label
import com.example.hatespeechmonitor.personalization.FeedbackSample
import com.example.hatespeechmonitor.personalization.FeedbackStore
import com.example.hatespeechmonitor.personalization.PersonalizationEngine
import com.example.hatespeechmonitor.personalization.SampleImporter
import com.example.hatespeechmonitor.personalization.TrainResult
import kotlinx.coroutines.Deferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

data class TrainUiState(
    val samples: List<FeedbackSample> = emptyList(),
    val epochs: Int = PersonalizationEngine.DEFAULT_EPOCHS,
    val batchSize: Int = PersonalizationEngine.DEFAULT_BATCH_SIZE,
    val training: Boolean = false,
    val saving: Boolean = false,
    val resetting: Boolean = false,
    val importing: Boolean = false,
    val importTotal: Int = 0,
    val importDone: Int = 0,
    val lastResult: TrainResult? = null,
    val hasPersistedWeights: Boolean = false,
    val engineReady: Boolean = false,
    val error: String? = null,
    val message: String? = null,
) {
    val sampleCount: Int get() = samples.size
    val hateCount: Int get() = samples.count { it.userLabel == Label.HATE }
    val nothateCount: Int get() = samples.count { it.userLabel == Label.NOTHATE }
    val importProgress: Float get() = if (importTotal == 0) 0f else importDone.toFloat() / importTotal
}

class TrainViewModel(
    private val appContext: Context,
    private val personalizationEngineDeferred: Deferred<PersonalizationEngine>,
    private val feedbackStore: FeedbackStore,
) : ViewModel() {

    private val _ui = MutableStateFlow(TrainUiState())
    val ui: StateFlow<TrainUiState> = _ui.asStateFlow()

    init {
        viewModelScope.launch {
            feedbackStore.samples.collect { list ->
                _ui.update { it.copy(samples = list) }
            }
        }
        viewModelScope.launch {
            val engine = personalizationEngineDeferred.await()
            _ui.update { it.copy(engineReady = true, hasPersistedWeights = engine.hasPersistedWeights) }
        }
    }

    fun setEpochs(v: Int) {
        _ui.update { it.copy(epochs = v.coerceIn(1, 20)) }
    }

    fun setBatchSize(v: Int) {
        _ui.update { it.copy(batchSize = v.coerceIn(1, 32)) }
    }

    fun train() {
        viewModelScope.launch {
            val samples = feedbackStore.all()
            if (samples.isEmpty()) {
                _ui.update { it.copy(error = "No feedback samples yet. Mark some texts on Classify / SMS, or import a file.") }
                return@launch
            }
            _ui.update { it.copy(training = true, error = null, message = null) }
            runCatching {
                val engine = personalizationEngineDeferred.await()
                engine.train(samples, epochs = _ui.value.epochs, batchSize = _ui.value.batchSize)
            }.onSuccess { result ->
                _ui.update {
                    it.copy(
                        training = false,
                        lastResult = result,
                        message = "Trained ${result.epochs} epochs on ${result.samplesUsed} samples",
                    )
                }
            }.onFailure { e ->
                _ui.update { it.copy(training = false, error = e.message ?: e::class.java.simpleName) }
            }
        }
    }

    fun save() {
        viewModelScope.launch {
            _ui.update { it.copy(saving = true, error = null, message = null) }
            runCatching {
                val engine = personalizationEngineDeferred.await()
                engine.save()
                engine
            }.onSuccess { engine ->
                _ui.update {
                    it.copy(
                        saving = false,
                        hasPersistedWeights = engine.hasPersistedWeights,
                        message = "Personalization saved to disk",
                    )
                }
            }.onFailure { e ->
                _ui.update { it.copy(saving = false, error = e.message ?: e::class.java.simpleName) }
            }
        }
    }

    fun resetToFactory() {
        viewModelScope.launch {
            _ui.update { it.copy(resetting = true, error = null, message = null) }
            runCatching {
                val engine = personalizationEngineDeferred.await()
                engine.resetToFactory()
                engine
            }.onSuccess { engine ->
                _ui.update {
                    it.copy(
                        resetting = false,
                        hasPersistedWeights = engine.hasPersistedWeights,
                        lastResult = null,
                        message = "Head reverted to factory weights",
                    )
                }
            }.onFailure { e ->
                _ui.update { it.copy(resetting = false, error = e.message ?: e::class.java.simpleName) }
            }
        }
    }

    fun clearFeedback() {
        viewModelScope.launch {
            feedbackStore.clear()
            _ui.update { it.copy(message = "Feedback cleared") }
        }
    }

    fun importFromUri(uri: Uri) {
        viewModelScope.launch {
            _ui.update {
                it.copy(importing = true, importDone = 0, importTotal = 0, error = null, message = null)
            }
            runCatching {
                val content = withContext(Dispatchers.IO) {
                    appContext.contentResolver.openInputStream(uri)
                        ?.bufferedReader(Charsets.UTF_8)
                        ?.use { r -> r.readText() }
                        ?: error("Could not open the selected file.")
                }
                val raw = SampleImporter.parse(content)
                require(raw.isNotEmpty()) { "No samples found in the file." }
                _ui.update { it.copy(importTotal = raw.size) }

                val engine = personalizationEngineDeferred.await()
                for ((i, sample) in raw.withIndex()) {
                    val embedding = engine.embed(sample.text)
                    feedbackStore.add(
                        FeedbackSample(text = sample.text, userLabel = sample.label, embedding = embedding),
                    )
                    _ui.update { it.copy(importDone = i + 1) }
                }
                raw
            }.onSuccess { raw ->
                val h = raw.count { it.label == Label.HATE }
                val n = raw.count { it.label == Label.NOTHATE }
                _ui.update {
                    it.copy(
                        importing = false,
                        message = "Imported ${raw.size} samples ($h HATE, $n NOTHATE)",
                    )
                }
            }.onFailure { e ->
                _ui.update { it.copy(importing = false, error = e.message ?: e::class.java.simpleName) }
            }
        }
    }

    fun dismissMessage() {
        _ui.update { it.copy(message = null, error = null) }
    }

    companion object {
        val Factory: ViewModelProvider.Factory = viewModelFactory {
            initializer {
                val app = this[APPLICATION_KEY] as HateSpeechApplication
                TrainViewModel(
                    appContext = app.applicationContext,
                    personalizationEngineDeferred = app.personalizationEngine,
                    feedbackStore = app.feedbackStore,
                )
            }
        }
    }
}
