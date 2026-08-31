package com.tfliteexample.smartlens

import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.objects.DetectedObject
import com.google.mlkit.vision.objects.ObjectDetection
import com.google.mlkit.vision.objects.defaults.ObjectDetectorOptions
import kotlinx.coroutines.tasks.await

class ObjectDetectorHelper {

    private val options = ObjectDetectorOptions.Builder()
        .setDetectorMode(ObjectDetectorOptions.STREAM_MODE)
        .enableMultipleObjects()
        .enableClassification()
        .build()

    private val objectDetector = ObjectDetection.getClient(options)

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
