package com.example.hatespeechmonitor.personalization

import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

interface FeedbackStore {
    val samples: Flow<List<FeedbackSample>>
    suspend fun add(sample: FeedbackSample)
    suspend fun all(): List<FeedbackSample>
    suspend fun count(): Int
    suspend fun clear()
}

class InMemoryFeedbackStore : FeedbackStore {
    private val mutex = Mutex()
    private val backing = ArrayList<FeedbackSample>()
    private val state = MutableStateFlow<List<FeedbackSample>>(emptyList())

    override val samples: Flow<List<FeedbackSample>> = state.asStateFlow()

    override suspend fun add(sample: FeedbackSample) = mutex.withLock {
        backing.add(sample)
        state.value = backing.toList()
    }

    override suspend fun all(): List<FeedbackSample> = mutex.withLock { backing.toList() }

    override suspend fun count(): Int = mutex.withLock { backing.size }

    override suspend fun clear() = mutex.withLock {
        backing.clear()
        state.value = emptyList()
    }
}
