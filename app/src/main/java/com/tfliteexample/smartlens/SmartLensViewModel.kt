package com.tfliteexample.smartlens

import android.annotation.SuppressLint
import android.app.Application
import android.graphics.Bitmap
import android.graphics.Rect
import android.graphics.RectF
import androidx.camera.core.ImageProxy
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.google.android.gms.tflite.client.TfLiteInitializationOptions
import com.google.android.gms.tflite.gpu.support.TfLiteGpu
import com.google.android.gms.tflite.java.TfLite
import com.google.mlkit.vision.common.InputImage
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.tasks.await
import java.util.concurrent.Executors

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

    // ── Tunable Settings State ──────────────────────────────────────────
    private val _minConfidenceThreshold = MutableStateFlow(0.40f)
    val minConfidenceThreshold: StateFlow<Float> = _minConfidenceThreshold.asStateFlow()

    private val _paddingMarginPercent = MutableStateFlow(0.12f)
    val paddingMarginPercent: StateFlow<Float> = _paddingMarginPercent.asStateFlow()

    private val _isSingleImageMode = MutableStateFlow(false)
    val isSingleImageMode: StateFlow<Boolean> = _isSingleImageMode.asStateFlow()

    private val _maxDetections = MutableStateFlow(3)
    val maxDetections: StateFlow<Int> = _maxDetections.asStateFlow()

    private var objectDetectorHelper: ObjectDetectorHelper? = null
    private var imageClassifierHelper: ImageClassifierHelper? = null

    // Single-threaded executor & dispatcher to guarantee all TFLite GPU delegate operations
    // (creation, inference, toggle, cleanup) run on the EXACT same thread.
    private val mlExecutor = Executors.newSingleThreadExecutor()
    private val mlDispatcher = mlExecutor.asCoroutineDispatcher()
    private val mlMutex = Mutex()

    @Volatile
    private var isProcessing = false

    init {
        initializeModels()
    }

    private fun initializeModels() {
        viewModelScope.launch(mlDispatcher) {
            mlMutex.withLock {
                try {
                    // Check GPU delegate availability and request GPU support initialization in GMS
                    val gpuAvailableTask = TfLiteGpu.isGpuDelegateAvailable(getApplication())
                    val isGpuAvailable = gpuAvailableTask.await()

                    val initOptionsBuilder = TfLiteInitializationOptions.builder()
                    if (isGpuAvailable) {
                        initOptionsBuilder.setEnableGpuDelegateSupport(true)
                    }

                    TfLite.initialize(getApplication(), initOptionsBuilder.build()).await()

                    objectDetectorHelper = ObjectDetectorHelper()

                    // Launch with CPU engine by default for fast & reliable app startup
                    imageClassifierHelper = ImageClassifierHelper(getApplication())

                    _isInitialized.value = true
                } catch (e: Exception) {
                    _errorMessage.value = "Failed to initialize models: ${e.message}"
                }
            }
        }
    }

    fun setMinConfidenceThreshold(value: Float) {
        _minConfidenceThreshold.value = value
    }

    fun setPaddingMarginPercent(value: Float) {
        _paddingMarginPercent.value = value
    }

    fun setSingleImageMode(value: Boolean) {
        _isSingleImageMode.value = value
        viewModelScope.launch(mlDispatcher) {
            mlMutex.withLock {
                objectDetectorHelper?.setMode(value)
            }
        }
    }

    fun setMaxDetections(value: Int) {
        _maxDetections.value = value
    }

    fun toggleGpu() {
        if (!_isInitialized.value) return

        viewModelScope.launch(mlDispatcher) {
            mlMutex.withLock {
                val newState = !_isGpuEnabled.value
                try {
                    imageClassifierHelper?.toggleGpu(newState)
                    _isGpuEnabled.value = newState
                    _errorMessage.value = null
                } catch (e: Exception) {
                    _errorMessage.value = "Failed to toggle GPU: ${e.message}"
                }
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

        // Correctly swap width and height in portrait mode (90° or 270°) so bounding box scaling matches rotation
        val isRotated = imageRotation == 90 || imageRotation == 270
        val effectiveWidth = if (isRotated) imageProxy.height else imageProxy.width
        val effectiveHeight = if (isRotated) imageProxy.width else imageProxy.height

        _imageWidth.value = effectiveWidth
        _imageHeight.value = effectiveHeight

        val mediaImage = imageProxy.image
        if (mediaImage != null) {
            val inputImage = InputImage.fromMediaImage(mediaImage, imageRotation)
            val bitmap = imageProxy.toBitmap()

            val currentMinConf = _minConfidenceThreshold.value
            val currentMargin = _paddingMarginPercent.value
            val currentMaxCount = _maxDetections.value

            viewModelScope.launch(mlDispatcher) {
                try {
                    mlMutex.withLock {
                        val rawObjects = objectDetectorHelper?.detect(inputImage) ?: emptyList()

                        // Filter out small or ghost detections (< 40x40 px)
                        val minArea = 40 * 40
                        val detectedObjects = rawObjects
                            .filter { obj ->
                                val rect = obj.boundingBox
                                rect.width() * rect.height() >= minArea
                            }
                            .take(currentMaxCount)

                        val results = mutableListOf<DetectionResult>()
                        var totalInferenceTime = 0L

                        for (detectedObject in detectedObjects) {
                            val rawBox = detectedObject.boundingBox
                            val mlKitLabel = detectedObject.labels.firstOrNull()?.text ?: "Unknown"

                            if (bitmap != null) {
                                // Dynamic box margin padding (0% to 30%) around detected object
                                val padX = (rawBox.width() * currentMargin).toInt()
                                val padY = (rawBox.height() * currentMargin).toInt()

                                val safeRect = Rect(
                                    Math.max(0, rawBox.left - padX),
                                    Math.max(0, rawBox.top - padY),
                                    Math.min(bitmap.width, rawBox.right + padX),
                                    Math.min(bitmap.height, rawBox.bottom + padY)
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
                                        // Filter results by user's minimum confidence threshold
                                        if (topResult != null && topResult.second >= currentMinConf) {
                                            results.add(
                                                DetectionResult(
                                                    boundingBox = RectF(safeRect),
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
                    }
                } catch (e: Exception) {
                    _errorMessage.value = "Processing error: ${e.message}"
                } finally {
                    bitmap?.recycle()
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
        viewModelScope.launch(mlDispatcher) {
            mlMutex.withLock {
                objectDetectorHelper?.close()
                imageClassifierHelper?.close()
            }
            mlExecutor.shutdown()
        }
    }
}
