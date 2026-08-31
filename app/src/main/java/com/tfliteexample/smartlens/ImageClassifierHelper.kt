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

    private val imageSize = 640
    private var isBchw = false

    init {
        loadLabels()
        setupInterpreter(false)
    }

    private fun loadLabels() {
        try {
            val inputStream = context.assets.open("coco_labels.txt")
            inputStream.bufferedReader().useLines { lines ->
                labels.addAll(lines)
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    private fun loadModelFile(): MappedByteBuffer {
        // Load YOLOv8-nano model exported via LiteRT-Torch
        val fileDescriptor = context.assets.openFd("yolov8n.tflite")
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

        try {
            val inputShape = newInterpreter.getInputTensor(0).shape()
            // Check if input is BCHW [1, 3, 640, 640] or BHWC [1, 640, 640, 3]
            isBchw = inputShape.size >= 4 && inputShape[1] == 3
        } catch (e: Exception) {
            e.printStackTrace()
        }

        interpreter = newInterpreter
    }

    fun toggleGpu(useGpu: Boolean) {
        setupInterpreter(useGpu)
    }

    fun classify(bitmap: Bitmap): ClassificationResult {
        val letterboxedBitmap = letterboxBitmap(bitmap, imageSize)
        val inputBuffer = convertBitmapToByteBuffer(letterboxedBitmap)
        letterboxedBitmap.recycle()

        // Output shape for YOLOv8n: [1, 84, 8400]
        val outputArray = Array(1) { Array(84) { FloatArray(8400) } }

        val startTime = System.nanoTime()
        interpreter?.run(inputBuffer, outputArray)
        val endTime = System.nanoTime()

        // Parse YOLOv8 outputs: 80 COCO classes probabilities (indices 4..83 across 8400 anchors)
        val classScores = FloatArray(labels.size)

        val output = outputArray[0]
        for (col in 0 until 8400) {
            for (cls in 0 until Math.min(80, labels.size)) {
                val score = output[4 + cls][col]
                if (score > classScores[cls]) {
                    classScores[cls] = score
                }
            }
        }

        val topResults = classScores.mapIndexed { index, confidence ->
            Pair(labels.getOrElse(index) { "Unknown" }, confidence)
        }
        .sortedByDescending { it.second }
        .take(3)

        return ClassificationResult(
            topResults = topResults,
            inferenceTimeMs = (endTime - startTime) / 1000000
        )
    }

    private fun letterboxBitmap(source: Bitmap, targetSize: Int): Bitmap {
        val result = Bitmap.createBitmap(targetSize, targetSize, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(result)
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
        val byteBuffer = ByteBuffer.allocateDirect(4 * 1 * imageSize * imageSize * 3) // 4 bytes per float
        byteBuffer.order(ByteOrder.nativeOrder())

        val intValues = IntArray(imageSize * imageSize)
        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)

        if (isBchw) {
            // [1, 3, 640, 640] format
            val rBuffer = FloatArray(imageSize * imageSize)
            val gBuffer = FloatArray(imageSize * imageSize)
            val bBuffer = FloatArray(imageSize * imageSize)

            var pixel = 0
            for (i in 0 until imageSize * imageSize) {
                val value = intValues[pixel++]
                rBuffer[i] = ((value shr 16) and 0xFF) / 255.0f
                gBuffer[i] = ((value shr 8) and 0xFF) / 255.0f
                bBuffer[i] = (value and 0xFF) / 255.0f
            }

            for (f in rBuffer) byteBuffer.putFloat(f)
            for (f in gBuffer) byteBuffer.putFloat(f)
            for (f in bBuffer) byteBuffer.putFloat(f)
        } else {
            // [1, 640, 640, 3] format
            var pixel = 0
            for (i in 0 until imageSize) {
                for (j in 0 until imageSize) {
                    val value = intValues[pixel++]
                    byteBuffer.putFloat(((value shr 16) and 0xFF) / 255.0f)
                    byteBuffer.putFloat(((value shr 8) and 0xFF) / 255.0f)
                    byteBuffer.putFloat((value and 0xFF) / 255.0f)
                }
            }
        }

        return byteBuffer
    }

    fun close() {
        interpreter?.close()
        interpreter = null
    }
}
