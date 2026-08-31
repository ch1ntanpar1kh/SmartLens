package com.tfliteexample.smartlens

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
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
    private var numClasses = 1001

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
        // Upgrade to EfficientNet-Lite0 int8 quantized model (5.2 MB) for significantly higher accuracy
        val fileDescriptor = context.assets.openFd("efficientnet_lite0_int8.tflite")
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
        val newInterpreter = InterpreterApi.create(model, options)

        // Dynamically inspect model output tensor shape
        try {
            val outputShape = newInterpreter.getOutputTensor(0).shape()
            if (outputShape.size >= 2) {
                numClasses = outputShape[1]
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }

        interpreter = newInterpreter
    }

    fun toggleGpu(useGpu: Boolean) {
        setupInterpreter(useGpu)
    }

    fun classify(bitmap: Bitmap): ClassificationResult {
        // Preserves aspect ratio using letterboxing with a neutral gray background
        val letterboxedBitmap = letterboxBitmap(bitmap, imageSize)
        val inputBuffer = convertBitmapToByteBuffer(letterboxedBitmap)
        letterboxedBitmap.recycle()

        // Output format for int8 quantized model
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

    /**
     * Resizes [source] bitmap into a [targetSize] x [targetSize] canvas while preserving original aspect ratio.
     * Empty space is letterboxed with a neutral gray (RGB 128,128,128) color.
     */
    private fun letterboxBitmap(source: Bitmap, targetSize: Int): Bitmap {
        val result = Bitmap.createBitmap(targetSize, targetSize, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(result)

        // Fill background with neutral gray (128, 128, 128)
        canvas.drawColor(Color.rgb(128, 128, 128))

        val srcWidth = source.width.toFloat()
        val srcHeight = source.height.toFloat()

        val scale = Math.min(targetSize / srcWidth, targetSize / srcHeight)
        val scaledWidth = srcWidth * scale
        val scaledHeight = srcHeight * scale

        val left = (targetSize - scaledWidth) / 2f
        val top = (targetSize - scaledHeight) / 2f

        val destRect = RectF(left, top, left + scaledWidth, top + scaledHeight)
        val paint = Paint(Paint.FILTER_BITMAP_FLAG)
        canvas.drawBitmap(source, null, destRect, paint)

        return result
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
