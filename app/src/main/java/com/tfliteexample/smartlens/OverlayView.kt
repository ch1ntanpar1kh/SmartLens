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
    if (results.isEmpty() || imageWidth <= 0 || imageHeight <= 0) return

    Canvas(modifier = modifier.fillMaxSize()) {
        val canvasWidth = size.width
        val canvasHeight = size.height

        // CameraX PreviewView uses ScaleType.FILL_CENTER (center crop to fill screen).
        // Uniform scale and offset alignment ensures bounding boxes match the rendered camera feed pixel-for-pixel.
        val scale = Math.max(canvasWidth / imageWidth.toFloat(), canvasHeight / imageHeight.toFloat())
        val offsetX = (canvasWidth - imageWidth * scale) / 2f
        val offsetY = (canvasHeight - imageHeight * scale) / 2f

        for (result in results) {
            val left = result.boundingBox.left * scale + offsetX
            val top = result.boundingBox.top * scale + offsetY
            val right = result.boundingBox.right * scale + offsetX
            val bottom = result.boundingBox.bottom * scale + offsetY

            val confidenceColor = when {
                result.confidence > 0.7f -> Color.Green
                result.confidence > 0.4f -> Color.Yellow
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

            val textBackgroundTop = Math.max(0f, top - textHeight - 12f)

            drawRect(
                color = confidenceColor,
                topLeft = Offset(left, textBackgroundTop),
                size = Size(textWidth + 16f, textHeight + 12f)
            )

            drawContext.canvas.nativeCanvas.drawText(
                text,
                left + 8f,
                textBackgroundTop + textHeight,
                textPaint
            )
        }
    }
}
