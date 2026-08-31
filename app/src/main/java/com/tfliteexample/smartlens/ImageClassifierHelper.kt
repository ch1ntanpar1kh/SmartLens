package com.tfliteexample.smartlens

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.InterpreterApi
import org.tensorflow.lite.gpu.GpuDelegateFactory
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class ImageClassifierHelper(private val context: Context) {

    data class ClassificationResult(
        val topResults: List<Pair<String, Float>>,
        val inferenceTimeMs: Long
    )

    private var interpreter: InterpreterApi? = null
    private val labels = mutableListOf<String>()

    private val imageSize = 224
    private val numClasses = 1001

    init {
        loadLabels()
        setupInterpreter(false)
    }

    private fun loadLabels() {
        try {
            val inputStream = context.assets.open("labels.txt")
            inputStream.bufferedReader().useLines { lines ->
                labels.addAll(lines)
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    private fun loadModelFile(): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd("mobilenet_v2_1.0_224_quant.tflite")
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        return fileChannel.map(
            FileChannel.MapMode.READ_ONLY,
            fileDescriptor.startOffset,
            fileDescriptor.declaredLength
        )
    }

    private fun setupInterpreter(useGpu: Boolean) {
        close()

        val options = InterpreterApi.Options()
            .setRuntime(InterpreterApi.Options.TfLiteRuntime.FROM_SYSTEM_ONLY)

        if (useGpu) {
            options.addDelegateFactory(GpuDelegateFactory())
        } else {
            options.setNumThreads(4)
        }

        val model = loadModelFile()
        interpreter = InterpreterApi.create(model, options)
    }

    fun toggleGpu(useGpu: Boolean) {
        setupInterpreter(useGpu)
    }

    fun classify(bitmap: Bitmap): ClassificationResult {
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, imageSize, imageSize, true)
        val inputBuffer = convertBitmapToByteBuffer(resizedBitmap)

        // Output format for uint8 quantized model
        val outputArray = Array(1) { ByteArray(numClasses) }

        val startTime = System.nanoTime()
        interpreter?.run(inputBuffer, outputArray)
        val endTime = System.nanoTime()

        val probabilities = outputArray[0].map { (it.toInt() and 0xFF) / 255.0f }

        val topResults = probabilities.mapIndexed { index, confidence ->
            Pair(labels.getOrElse(index) { "Unknown" }, confidence)
        }
        .filter { it.first != "background" }
        .sortedByDescending { it.second }
        .take(3)

        return ClassificationResult(
            topResults = topResults,
            inferenceTimeMs = (endTime - startTime) / 1000000
        )
    }

    private fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val byteBuffer = ByteBuffer.allocateDirect(1 * imageSize * imageSize * 3)
        byteBuffer.order(ByteOrder.nativeOrder())

        val intValues = IntArray(imageSize * imageSize)
        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)

        var pixel = 0
        for (i in 0 until imageSize) {
            for (j in 0 until imageSize) {
                val value = intValues[pixel++]
                byteBuffer.put(((value shr 16) and 0xFF).toByte())
                byteBuffer.put(((value shr 8) and 0xFF).toByte())
                byteBuffer.put((value and 0xFF).toByte())
            }
        }

        return byteBuffer
    }

    fun close() {
        interpreter?.close()
        interpreter = null
    }
}
