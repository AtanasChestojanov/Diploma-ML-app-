package com.example.hatespeechmonitor.ui.settings

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.ViewModelProvider.AndroidViewModelFactory.Companion.APPLICATION_KEY
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.initializer
import androidx.lifecycle.viewmodel.viewModelFactory
import com.example.hatespeechmonitor.HateSpeechApplication
import com.example.hatespeechmonitor.data.SettingsRepository
import com.example.hatespeechmonitor.ml.InferenceEngine
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch

data class ModelInfo(
    val assetName: String,
    val sizeBytes: Long,
    val sequenceLength: Int,
    val numThreads: Int,
)

class SettingsViewModel(
    private val settings: SettingsRepository,
    engine: InferenceEngine,
) : ViewModel() {

    val threshold: StateFlow<Float> = settings.decisionThreshold
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), SettingsRepository.DEFAULT_THRESHOLD)

    val modelInfo: ModelInfo = ModelInfo(
        assetName = "mobilebert_dynamic_range.tflite",
        sizeBytes = engine.modelSizeBytes,
        sequenceLength = engine.sequenceLength,
        numThreads = engine.numThreads,
    )

    fun setThreshold(value: Float) {
        viewModelScope.launch { settings.setDecisionThreshold(value) }
    }

    fun resetThreshold() {
        viewModelScope.launch { settings.setDecisionThreshold(SettingsRepository.DEFAULT_THRESHOLD) }
    }

    companion object {
        val Factory: ViewModelProvider.Factory = viewModelFactory {
            initializer {
                val app = this[APPLICATION_KEY] as HateSpeechApplication
                SettingsViewModel(app.settingsRepository, app.inferenceEngine)
            }
        }
    }
}
