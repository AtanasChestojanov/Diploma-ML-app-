package com.example.hatespeechmonitor.ui.sms

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.ViewModelProvider.AndroidViewModelFactory.Companion.APPLICATION_KEY
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.initializer
import androidx.lifecycle.viewmodel.viewModelFactory
import com.example.hatespeechmonitor.HateSpeechApplication
import com.example.hatespeechmonitor.data.SettingsRepository
import com.example.hatespeechmonitor.ml.Label
import com.example.hatespeechmonitor.personalization.FeedbackSample
import com.example.hatespeechmonitor.personalization.FeedbackStore
import com.example.hatespeechmonitor.personalization.PersonalizationEngine
import com.example.hatespeechmonitor.sms.ClassifiedSms
import com.example.hatespeechmonitor.sms.SmsMessage
import com.example.hatespeechmonitor.sms.SmsRepository
import kotlinx.coroutines.Deferred
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

data class SmsUiState(
    val items: List<ClassifiedSms> = emptyList(),
    val loading: Boolean = false,
    val permissionGranted: Boolean = false,
    val error: String? = null,
    val feedbackMessage: String? = null,
)

class SmsViewModel(
    private val personalizationEngineDeferred: Deferred<PersonalizationEngine>,
    private val repo: SmsRepository,
    private val feedbackStore: FeedbackStore,
    settings: SettingsRepository,
) : ViewModel() {

    private val _ui = MutableStateFlow(SmsUiState())
    val ui: StateFlow<SmsUiState> = _ui.asStateFlow()

    val threshold: StateFlow<Float> = settings.decisionThreshold
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), SettingsRepository.DEFAULT_THRESHOLD)

    private var loadJob: Job? = null

    init {
        viewModelScope.launch {
            repo.incoming.collect { incoming ->
                _ui.update { state -> state.copy(items = listOf(incoming) + state.items) }
            }
        }
    }

    fun onPermissionsResult(granted: Boolean) {
        _ui.update { it.copy(permissionGranted = granted) }
        if (granted) refresh()
    }

    fun refresh() {
        loadJob?.cancel()
        loadJob = viewModelScope.launch {
            _ui.update { it.copy(loading = true, error = null) }
            runCatching {
                val recent = repo.loadRecent(limit = 30)
                _ui.update { it.copy(items = recent.map { m -> ClassifiedSms(m, null) }) }
                val engine = personalizationEngineDeferred.await()
                recent.forEach { msg ->
                    val result = engine.classify(msg.body)
                    _ui.update { state ->
                        val updated = state.items.toMutableList()
                        val pos = updated.indexOfFirst { it.message === msg || (it.result == null && it.message == msg) }
                        if (pos >= 0) updated[pos] = ClassifiedSms(msg, result)
                        state.copy(items = updated)
                    }
                }
            }.onFailure { e ->
                _ui.update { it.copy(error = e.message ?: e::class.java.simpleName) }
            }
            _ui.update { it.copy(loading = false) }
        }
    }

    fun submitFeedback(message: SmsMessage, label: Label) {
        viewModelScope.launch {
            runCatching {
                val engine = personalizationEngineDeferred.await()
                val embedding = engine.embed(message.body)
                feedbackStore.add(FeedbackSample(text = message.body, userLabel = label, embedding = embedding))
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

    companion object {
        val Factory: ViewModelProvider.Factory = viewModelFactory {
            initializer {
                val app = this[APPLICATION_KEY] as HateSpeechApplication
                SmsViewModel(
                    personalizationEngineDeferred = app.personalizationEngine,
                    repo = app.smsRepository,
                    feedbackStore = app.feedbackStore,
                    settings = app.settingsRepository,
                )
            }
        }
    }
}
