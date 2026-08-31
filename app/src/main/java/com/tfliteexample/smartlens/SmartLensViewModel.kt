package com.tfliteexample.smartlens

import android.annotation.SuppressLint
import android.app.Application
import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.RectF
import androidx.camera.core.ImageProxy
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.google.android.gms.tasks.Tasks
import com.google.android.gms.tflite.client.TfLiteInitializationOptions
import com.google.android.gms.tflite.gpu.support.TfLiteGpu
import com.google.android.gms.tflite.java.TfLite
import com.google.mlkit.vision.common.InputImage
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.tasks.await
import kotlinx.coroutines.withContext
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer

class SmartLensViewModel(application: Application) : AndroidViewModel(application) {

    data class DetectionResult(
        val boundingBox: RectF,
        val label: String,
        val confidence: Float,
        val mlKitLabel: String
    )

    private val _detectionResults = MutableStateFlow<List<DetectionResult>>(emptyList())
    val detectionResults: StateFlow<List<DetectionResult>> = _detectionResults.asStateFlow()

    private val _inferenceTime = MutableStateFlow(0L)
    val inferenceTime: StateFlow<Long> = _inferenceTime.asStateFlow()

    private val _isGpuEnabled = MutableStateFlow(false)
    val isGpuEnabled: StateFlow<Boolean> = _isGpuEnabled.asStateFlow()

    private val _isInitialized = MutableStateFlow(false)
    val isInitialized: StateFlow<Boolean> = _isInitialized.asStateFlow()

    private val _errorMessage = MutableStateFlow<String?>(null)
    val errorMessage: StateFlow<String?> = _errorMessage.asStateFlow()

    private val _imageWidth = MutableStateFlow(1)
    val imageWidth: StateFlow<Int> = _imageWidth.asStateFlow()

    private val _imageHeight = MutableStateFlow(1)
    val imageHeight: StateFlow<Int> = _imageHeight.asStateFlow()

    private var objectDetectorHelper: ObjectDetectorHelper? = null
    private var imageClassifierHelper: ImageClassifierHelper? = null

    private var isProcessing = false

    init {
        initializeModels()
    }

    private fun initializeModels() {
        viewModelScope.launch(Dispatchers.IO) {
            try {
                val gpuAvailableTask = TfLiteGpu.isGpuDelegateAvailable(getApplication())
                val isGpuAvailable = gpuAvailableTask.await()
                
                val initOptionsBuilder = TfLiteInitializationOptions.builder()
                if (isGpuAvailable) {
                    initOptionsBuilder.setEnableGpuDelegateSupport(true)
                }

                TfLite.initialize(getApplication(), initOptionsBuilder.build()).await()
                
                objectDetectorHelper = ObjectDetectorHelper()
                imageClassifierHelper = ImageClassifierHelper(getApplication())
                
                _isInitialized.value = true
            } catch (e: Exception) {
                _errorMessage.value = "Failed to initialize models: ${e.message}"
            }
        }
    }

    fun toggleGpu() {
        if (!_isInitialized.value) return
        
        viewModelScope.launch(Dispatchers.IO) {
            val newState = !_isGpuEnabled.value
            try {
                imageClassifierHelper?.toggleGpu(newState)
                _isGpuEnabled.value = newState
            } catch (e: Exception) {
                _errorMessage.value = "Failed to toggle GPU: ${e.message}"
            }
        }
    }

    @SuppressLint("UnsafeOptInUsageError")
    fun processImageProxy(imageProxy: ImageProxy, imageRotation: Int) {
        if (!_isInitialized.value || isProcessing) {
            imageProxy.close()
            return
        }

        isProcessing = true
        _imageWidth.value = imageProxy.width
        _imageHeight.value = imageProxy.height

        val mediaImage = imageProxy.image
        if (mediaImage != null) {
            val inputImage = InputImage.fromMediaImage(mediaImage, imageRotation)
            val bitmap = imageProxy.toBitmap()

            viewModelScope.launch(Dispatchers.IO) {
                try {
                    val detectedObjects = objectDetectorHelper?.detect(inputImage) ?: emptyList()
                    val results = mutableListOf<DetectionResult>()
                    var totalInferenceTime = 0L

                    for (detectedObject in detectedObjects) {
                        val boundingBox = detectedObject.boundingBox
                        val mlKitLabel = detectedObject.labels.firstOrNull()?.text ?: "Unknown"

                        if (bitmap != null) {
                            val safeRect = Rect(
                                Math.max(0, boundingBox.left),
                                Math.max(0, boundingBox.top),
                                Math.min(bitmap.width, boundingBox.right),
                                Math.min(bitmap.height, boundingBox.bottom)
                            )
                            if (safeRect.width() > 0 && safeRect.height() > 0) {
                                val croppedBitmap = Bitmap.createBitmap(
                                    bitmap,
                                    safeRect.left,
                                    safeRect.top,
                                    safeRect.width(),
                                    safeRect.height()
                                )

                                val classificationResult = imageClassifierHelper?.classify(croppedBitmap)
                                croppedBitmap.recycle()

                                if (classificationResult != null) {
                                    totalInferenceTime += classificationResult.inferenceTimeMs
                                    val topResult = classificationResult.topResults.firstOrNull()
                                    if (topResult != null) {
                                        results.add(
                                            DetectionResult(
                                                boundingBox = RectF(boundingBox),
                                                label = topResult.first,
                                                confidence = topResult.second,
                                                mlKitLabel = mlKitLabel
                                            )
                                        )
                                    }
                                }
                            }
                        }
                    }

                    _detectionResults.value = results
                    _inferenceTime.value = totalInferenceTime
                } catch (e: Exception) {
                    _errorMessage.value = "Processing error: ${e.message}"
                } finally {
                    imageProxy.close()
                    isProcessing = false
                }
            }
        } else {
            imageProxy.close()
            isProcessing = false
        }
    }

    override fun onCleared() {
        super.onCleared()
        objectDetectorHelper?.close()
        imageClassifierHelper?.close()
    }
}
