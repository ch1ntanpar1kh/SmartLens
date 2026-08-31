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
        setupInterpreter(useGpu = false)
    }

    private fun loadLabels() {
        runCatching {
            context.assets.open("coco_labels.txt").bufferedReader().useLines { lines ->
                labels.addAll(lines)
            }
        }
    }

    private fun loadModelFile(): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd("yolov8n.tflite")
        FileInputStream(fileDescriptor.fileDescriptor).use { inputStream ->
            return inputStream.channel.map(
                FileChannel.MapMode.READ_ONLY,
                fileDescriptor.startOffset,
                fileDescriptor.declaredLength
            )
        }
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

        val newInterpreter = InterpreterApi.create(loadModelFile(), options)

        runCatching {
            val inputShape = newInterpreter.getInputTensor(0).shape()
            isBchw = inputShape.size >= 4 && inputShape[1] == 3
        }

        interpreter = newInterpreter
    }

    fun toggleGpu(useGpu: Boolean) = setupInterpreter(useGpu)

    fun classify(bitmap: Bitmap): ClassificationResult {
        val letterboxedBitmap = letterboxBitmap(bitmap, imageSize)
        val inputBuffer = convertBitmapToByteBuffer(letterboxedBitmap)
        letterboxedBitmap.recycle()

        // Output shape for YOLOv8n: [1, 84, 8400]
        val outputArray = Array(1) { Array(84) { FloatArray(8400) } }

        val startTime = System.nanoTime()
        interpreter?.run(inputBuffer, outputArray)
        val endTime = System.nanoTime()

        // Parse YOLOv8 outputs across 8,400 anchors for 80 COCO classes
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
            inferenceTimeMs = (endTime - startTime) / 1_000_000
        )
    }

    private fun letterboxBitmap(source: Bitmap, targetSize: Int): Bitmap {
        val result = Bitmap.createBitmap(targetSize, targetSize, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(result)
        canvas.drawColor(Color.rgb(128, 128, 128))

        val scale = Math.min(targetSize / source.width.toFloat(), targetSize / source.height.toFloat())
        val scaledWidth = source.width * scale
        val scaledHeight = source.height * scale

        val left = (targetSize - scaledWidth) / 2f
        val top = (targetSize - scaledHeight) / 2f

        val destRect = RectF(left, top, left + scaledWidth, top + scaledHeight)
        canvas.drawBitmap(source, null, destRect, Paint(Paint.FILTER_BITMAP_FLAG))

        return result
    }

    private fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3)
        byteBuffer.order(ByteOrder.nativeOrder())

        val intValues = IntArray(imageSize * imageSize)
        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)

        if (isBchw) {
            val r = FloatArray(imageSize * imageSize)
            val g = FloatArray(imageSize * imageSize)
            val b = FloatArray(imageSize * imageSize)
            for (i in intValues.indices) {
                val v = intValues[i]
                r[i] = ((v shr 16) and 0xFF) / 255.0f
                g[i] = ((v shr 8) and 0xFF) / 255.0f
                b[i] = (v and 0xFF) / 255.0f
            }
            for (f in r) byteBuffer.putFloat(f)
            for (f in g) byteBuffer.putFloat(f)
            for (f in b) byteBuffer.putFloat(f)
        } else {
            for (v in intValues) {
                byteBuffer.putFloat(((v shr 16) and 0xFF) / 255.0f)
                byteBuffer.putFloat(((v shr 8) and 0xFF) / 255.0f)
                byteBuffer.putFloat((v and 0xFF) / 255.0f)
            }
        }

        return byteBuffer
    }

    fun close() {
        interpreter?.close()
        interpreter = null
    }
}
