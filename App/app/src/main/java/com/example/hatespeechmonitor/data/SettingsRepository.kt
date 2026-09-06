package com.example.hatespeechmonitor.data

import android.content.Context
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.emptyPreferences
import androidx.datastore.preferences.core.floatPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.map
import java.io.IOException

private val Context.settingsDataStore: DataStore<Preferences> by preferencesDataStore(name = "settings")

class SettingsRepository(context: Context) {

    private val ds = context.applicationContext.settingsDataStore

    val decisionThreshold: Flow<Float> = ds.data
        .catch { e -> if (e is IOException) emit(emptyPreferences()) else throw e }
        .map { it[KEY_THRESHOLD] ?: DEFAULT_THRESHOLD }

    suspend fun setDecisionThreshold(value: Float) {
        ds.edit { it[KEY_THRESHOLD] = value.coerceIn(0f, 1f) }
    }

    companion object {
        const val DEFAULT_THRESHOLD = 0.5f
        private val KEY_THRESHOLD = floatPreferencesKey("decision_threshold")
    }
}
