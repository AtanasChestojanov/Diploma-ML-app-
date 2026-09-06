package com.example.hatespeechmonitor.ui

import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Build
import androidx.compose.material.icons.filled.Edit
import androidx.compose.material.icons.filled.Email
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.Icon
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.navigation.NavDestination.Companion.hierarchy
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import com.example.hatespeechmonitor.ui.classify.ClassifyScreen
import com.example.hatespeechmonitor.ui.settings.SettingsScreen
import com.example.hatespeechmonitor.ui.sms.SmsScreen
import com.example.hatespeechmonitor.ui.train.TrainScreen

private enum class Tab(val route: String, val label: String, val icon: ImageVector) {
    Classify("classify", "Classify", Icons.Filled.Edit),
    Sms("sms", "SMS", Icons.Filled.Email),
    Train("train", "Train", Icons.Filled.Build),
    Settings("settings", "Settings", Icons.Filled.Settings),
}

@Composable
fun AppNav() {
    val nav = rememberNavController()
    val backStack by nav.currentBackStackEntryAsState()
    val currentRoute = backStack?.destination?.route

    Scaffold(
        modifier = Modifier.fillMaxSize(),
        bottomBar = {
            NavigationBar {
                for (tab in Tab.values()) {
                    val selected = backStack?.destination?.hierarchy?.any { it.route == tab.route } == true ||
                        currentRoute == tab.route
                    NavigationBarItem(
                        selected = selected,
                        onClick = {
                            if (currentRoute != tab.route) {
                                nav.navigate(tab.route) {
                                    popUpTo(nav.graph.startDestinationId) { saveState = true }
                                    launchSingleTop = true
                                    restoreState = true
                                }
                            }
                        },
                        icon = { Icon(tab.icon, contentDescription = tab.label) },
                        label = { Text(tab.label) },
                    )
                }
            }
        },
    ) { padding ->
        NavHost(navController = nav, startDestination = Tab.Classify.route) {
            composable(Tab.Classify.route) { ClassifyScreen(contentPadding = padding) }
            composable(Tab.Sms.route) { SmsScreen(contentPadding = padding) }
            composable(Tab.Train.route) { TrainScreen(contentPadding = padding) }
            composable(Tab.Settings.route) { SettingsScreen(contentPadding = padding) }
        }
    }
}
