package com.tfliteexample.smartlens

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.expandVertically
import androidx.compose.animation.shrinkVertically
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.lifecycle.viewmodel.compose.viewModel
import java.util.concurrent.Executors

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    SmartLensScreen()
                }
            }
        }
    }
}

@Composable
fun SmartLensScreen(viewModel: SmartLensViewModel = viewModel()) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current

    val detectionResults by viewModel.detectionResults.collectAsState()
    val inferenceTime by viewModel.inferenceTime.collectAsState()
    val isGpuEnabled by viewModel.isGpuEnabled.collectAsState()
    val isInitialized by viewModel.isInitialized.collectAsState()
    val errorMessage by viewModel.errorMessage.collectAsState()
    val imageWidth by viewModel.imageWidth.collectAsState()
    val imageHeight by viewModel.imageHeight.collectAsState()

    // Fine-tuning settings state
    val minConfidenceThreshold by viewModel.minConfidenceThreshold.collectAsState()
    val paddingMarginPercent by viewModel.paddingMarginPercent.collectAsState()
    val isSingleImageMode by viewModel.isSingleImageMode.collectAsState()
    val maxDetections by viewModel.maxDetections.collectAsState()

    var showSettingsPanel by remember { mutableStateOf(false) }

    var hasCameraPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(
                context,
                Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED
        )
    }

    val permissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        hasCameraPermission = isGranted
    }

    LaunchedEffect(Unit) {
        if (!hasCameraPermission) {
            permissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    if (!hasCameraPermission) {
        Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Text("📷", fontSize = 48.sp)
                Spacer(modifier = Modifier.height(16.dp))
                Text("Camera permission is required")
                Spacer(modifier = Modifier.height(8.dp))
                Button(onClick = { permissionLauncher.launch(Manifest.permission.CAMERA) }) {
                    Text("Grant Permission")
                }
            }
        }
        return
    }

    val executor = remember { Executors.newSingleThreadExecutor() }
    val previewView = remember { PreviewView(context) }

    LaunchedEffect(isInitialized) {
        if (!isInitialized) return@LaunchedEffect

        val cameraProvider = ProcessCameraProvider.getInstance(context).get()

        val preview = Preview.Builder().build().also {
            it.setSurfaceProvider(previewView.surfaceProvider)
        }

        val imageAnalysis = ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
            .build()
            .also { analysis ->
                analysis.setAnalyzer(executor) { imageProxy ->
                    viewModel.processImageProxy(imageProxy, imageProxy.imageInfo.rotationDegrees)
                }
            }

        try {
            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                lifecycleOwner,
                CameraSelector.DEFAULT_BACK_CAMERA,
                preview,
                imageAnalysis
            )
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    Box(modifier = Modifier.fillMaxSize()) {
        // Camera preview with FILL_CENTER scale type
        AndroidView(
            modifier = Modifier.fillMaxSize(),
            factory = {
                previewView.apply {
                    scaleType = PreviewView.ScaleType.FILL_CENTER
                }
            }
        )

        // Bounding Box Overlay
        if (imageWidth > 0 && imageHeight > 0) {
            DetectionOverlay(
                results = detectionResults,
                imageWidth = imageWidth,
                imageHeight = imageHeight,
                modifier = Modifier.fillMaxSize()
            )
        }

        // Top Action Bar: Settings Toggle Button
        IconButton(
            onClick = { showSettingsPanel = !showSettingsPanel },
            modifier = Modifier
                .align(Alignment.TopEnd)
                .padding(16.dp)
                .background(Color.Black.copy(alpha = 0.6f), RoundedCornerShape(50))
        ) {
            Icon(
                imageVector = Icons.Default.Settings,
                contentDescription = "Accuracy Settings",
                tint = if (showSettingsPanel) MaterialTheme.colorScheme.primary else Color.White
            )
        }

        // Bottom Info Bar + Expandable Fine-Tuning Panel
        Column(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .fillMaxWidth()
                .background(Color.Black.copy(alpha = 0.8f))
                .padding(16.dp)
        ) {
            // Expandable Accuracy Tuning Settings Panel
            AnimatedVisibility(
                visible = showSettingsPanel,
                enter = expandVertically(),
                exit = shrinkVertically()
            ) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(bottom = 12.dp)
                        .background(Color.White.copy(alpha = 0.1f), RoundedCornerShape(12.dp))
                        .padding(12.dp)
                ) {
                    Text(
                        "⚙️ Accuracy & Detection Tuning",
                        color = Color.White,
                        fontSize = 14.sp,
                        style = MaterialTheme.typography.titleMedium
                    )

                    Spacer(modifier = Modifier.height(8.dp))

                    // Min Confidence Threshold Slider
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween
                    ) {
                        Text("Min Confidence Threshold:", color = Color.White, fontSize = 12.sp)
                        Text("${(minConfidenceThreshold * 100).toInt()}%", color = Color.Green, fontSize = 12.sp)
                    }
                    Slider(
                        value = minConfidenceThreshold,
                        onValueChange = { viewModel.setMinConfidenceThreshold(it) },
                        valueRange = 0.10f..0.90f
                    )

                    // Bounding Box Padding Margin Slider
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween
                    ) {
                        Text("Box Padding Margin:", color = Color.White, fontSize = 12.sp)
                        Text("${(paddingMarginPercent * 100).toInt()}%", color = Color.Yellow, fontSize = 12.sp)
                    }
                    Slider(
                        value = paddingMarginPercent,
                        onValueChange = { viewModel.setPaddingMarginPercent(it) },
                        valueRange = 0.00f..0.30f
                    )

                    // Single Image High-Res Detector Mode Switch
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Column {
                            Text("High-Res Single Image Mode", color = Color.White, fontSize = 12.sp)
                            Text(
                                if (isSingleImageMode) "Higher resolution detector per frame" else "Fast stream tracking mode",
                                color = Color.Gray,
                                fontSize = 10.sp
                            )
                        }
                        Switch(
                            checked = isSingleImageMode,
                            onCheckedChange = { viewModel.setSingleImageMode(it) }
                        )
                    }

                    Spacer(modifier = Modifier.height(8.dp))

                    // Max Objects Count Slider
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween
                    ) {
                        Text("Max Objects Detected:", color = Color.White, fontSize = 12.sp)
                        Text("$maxDetections", color = Color.Cyan, fontSize = 12.sp)
                    }
                    Slider(
                        value = maxDetections.toFloat(),
                        onValueChange = { viewModel.setMaxDetections(it.toInt()) },
                        valueRange = 1f..5f,
                        steps = 3
                    )
                }
            }

            if (!isInitialized) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier.padding(bottom = 8.dp)
                ) {
                    CircularProgressIndicator(
                        modifier = Modifier.size(16.dp),
                        strokeWidth = 2.dp,
                        color = Color.White
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = "Initializing ML models via Google Play Services...",
                        color = Color.White,
                        fontSize = 13.sp
                    )
                }
            }

            errorMessage?.let {
                Text(
                    text = it,
                    color = Color.Red,
                    fontSize = 12.sp,
                    modifier = Modifier.padding(bottom = 4.dp)
                )
            }

            // Main Status Row
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text("GPU", color = Color.White, fontSize = 14.sp)
                    Spacer(modifier = Modifier.width(8.dp))
                    Switch(
                        checked = isGpuEnabled,
                        onCheckedChange = { viewModel.toggleGpu() },
                        enabled = isInitialized
                    )
                }

                Text(
                    text = "Inference: ${inferenceTime}ms (${if (isGpuEnabled) "GPU" else "CPU"})",
                    color = Color.White,
                    fontSize = 14.sp
                )
            }

            // Object count status
            if (detectionResults.isNotEmpty()) {
                Text(
                    text = "${detectionResults.size} object(s) detected",
                    color = Color.White.copy(alpha = 0.7f),
                    fontSize = 12.sp,
                    modifier = Modifier.padding(top = 4.dp)
                )
            }
        }
    }
}
