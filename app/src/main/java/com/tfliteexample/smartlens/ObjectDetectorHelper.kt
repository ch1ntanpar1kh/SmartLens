package com.tfliteexample.smartlens

import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.objects.DetectedObject
import com.google.mlkit.vision.objects.ObjectDetection
import com.google.mlkit.vision.objects.ObjectDetector
import com.google.mlkit.vision.objects.defaults.ObjectDetectorOptions
import kotlinx.coroutines.tasks.await

class ObjectDetectorHelper {

    private var currentMode = ObjectDetectorOptions.STREAM_MODE
    private var objectDetector: ObjectDetector = createDetector(currentMode)

    private fun createDetector(mode: Int): ObjectDetector {
        val options = ObjectDetectorOptions.Builder()
            .setDetectorMode(mode)
            .enableMultipleObjects()
            .enableClassification()
            .build()
        return ObjectDetection.getClient(options)
    }

    fun setMode(singleImageMode: Boolean) {
        val newMode = if (singleImageMode) {
            ObjectDetectorOptions.SINGLE_IMAGE_MODE
        } else {
            ObjectDetectorOptions.STREAM_MODE
        }

        if (newMode != currentMode) {
            currentMode = newMode
            objectDetector.close()
            objectDetector = createDetector(currentMode)
        }
    }

    suspend fun detect(image: InputImage): List<DetectedObject> {
        return try {
            objectDetector.process(image).await()
        } catch (e: Exception) {
            emptyList()
        }
    }

    fun close() {
        objectDetector.close()
    }
}
