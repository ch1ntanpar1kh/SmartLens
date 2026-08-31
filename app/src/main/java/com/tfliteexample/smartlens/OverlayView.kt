package com.tfliteexample.smartlens

import android.graphics.Paint
import android.graphics.Typeface
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.unit.dp

@Composable
fun DetectionOverlay(
    results: List<SmartLensViewModel.DetectionResult>,
    imageWidth: Int,
    imageHeight: Int,
    modifier: Modifier = Modifier
) {
    if (results.isEmpty()) return

    Canvas(modifier = modifier.fillMaxSize()) {
        val canvasWidth = size.width
        val canvasHeight = size.height

        val scaleX = canvasWidth / imageWidth.toFloat()
        val scaleY = canvasHeight / imageHeight.toFloat()

        for (result in results) {
            val left = result.boundingBox.left * scaleX
            val top = result.boundingBox.top * scaleY
            val right = result.boundingBox.right * scaleX
            val bottom = result.boundingBox.bottom * scaleY

            val confidenceColor = when {
                result.confidence > 0.8f -> Color.Green
                result.confidence > 0.5f -> Color.Yellow
                else -> Color.Red
            }

            // Draw bounding box
            drawRoundRect(
                color = confidenceColor,
                topLeft = Offset(left, top),
                size = Size(right - left, bottom - top),
                cornerRadius = CornerRadius(8f, 8f),
                style = Stroke(width = 3.dp.toPx())
            )

            // Draw label background + text
            val text = "${result.label} ${(result.confidence * 100).toInt()}%"
            val textPaint = Paint().apply {
                color = android.graphics.Color.BLACK
                textSize = 14.dp.toPx()
                typeface = Typeface.DEFAULT_BOLD
            }
            val textWidth = textPaint.measureText(text)
            val textHeight = textPaint.descent() - textPaint.ascent()

            drawRect(
                color = confidenceColor,
                topLeft = Offset(left, top - textHeight - 12f),
                size = Size(textWidth + 16f, textHeight + 12f)
            )

            drawContext.canvas.nativeCanvas.drawText(
                text,
                left + 8f,
                top - 8f,
                textPaint
            )
        }
    }
}
