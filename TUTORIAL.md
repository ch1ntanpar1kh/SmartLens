# Step-by-Step Beginner Tutorial: Building an On-Device Vision App with ML Kit & LiteRT via Google Play Services (GMS Core)

This tutorial guides you through building a real-time, high-accuracy object detection and classification Android app from scratch. You will learn how to pair **ML Kit Object Detection** with a custom **YOLOv8-nano LiteRT (TensorFlow Lite) model**, both delivered through **Google Play Services (GMS Core)** for a lightweight, GPU-accelerated footprint.

---

## 🌟 1. Why Deploy via Google Play Services (GMS Core)?

Traditional mobile ML apps bundle native C++ inference engines (`libtensorflowlite_jni.so`) and heavy pre-trained models directly inside the APK. This inflates APK sizes to **50–150 MB** and forces developer updates whenever underlying ML runtimes improve.

By offloading runtime delivery to **Google Play Services (GMS Core)**, your app achieves a **"Zero-Weight ML Engine"** architecture:

```
  ┌────────────────────────────────────────────────────────┐
  │ Your Android App (~5 MB APK)                           │
  │  ├── UI & Application Logic (Jetpack Compose)          │
  │  ├── CameraX Engine                                    │
  │  ├── LiteRT Model File (yolov8n.tflite ~12 MB)         │
  │  └── Thin GMS Client Libraries (~200 KB)               │
  └──────────────────────────┬─────────────────────────────┘
                             │ Calls thin APIs
                             ▼
  ┌────────────────────────────────────────────────────────┐
  │ Google Play Services (Pre-installed on device)          │
  │  ├── Shared ML Kit Object Detection Engine & Models    │
  │  ├── Shared LiteRT C++ Interpreter Engine              │
  │  └── Dynamic GPU Acceleration Drivers (Vulkan/OpenCL)  │
  └──────────────────────────?─────────────────────────────┘
```

### Key Deployment Benefits

| Feature | Standard Bundled ML App | **GMS Core Delivered App** |
|---|---|---|
| **APK Footprint** | ~50 MB – 150 MB | **~5 MB** (App code + model asset only) |
| **C++ Inference Engine** | Bundled inside your APK | Shared GMS system module |
| **Runtime Updates** | Requires manual app update in Play Store | Google updates ML engine & security patches automatically |
| **GPU Delegate Compatibility** | Developers manage native GPU binaries | GMS dynamically handles Qualcomm Adreno / Exynos / Tensor OpenCL/Vulkan drivers |
| **Cold-Start Delay** | None, but heavy download | Zero cold-start delay using `<meta-data>` install-time pre-fetch |

---

## 🧠 2. Model Selection & Source

### Why Standard ImageNet Models Fail for Everyday Apps
Most tutorials default to **MobileNet V2/V3** trained on the **ImageNet (ILSVRC)** dataset. While popular, ImageNet was designed for 2012 academic research and contains **1,000 hyper-specific classes**:
- 🐶 **120+ categories are specific dog breeds** (*Norfolk Terrier, Pembroke Welsh Corgi, Samoyed*)
- 🐟 **60+ categories are specific fish & birds** (*Tench, Goldfinch, Brambling*)
- ❌ **Missing everyday items**: ImageNet has **NO** generic class for *smartphone*, *desk pen*, *wallet*, *credit card*, *keys*, or *remote control*. When pointed at a smartphone, ImageNet models guess *"iPod"* or *"cellular telephone"*.

### Why YOLOv8-nano on COCO 80 is the Ideal Choice
We select **YOLOv8-nano**, trained on the **COCO dataset**, which focuses on **80 common real-world items**:

```
📱 Cell Phone      💻 Laptop       ☕ Cup / Mug       🍾 Bottle
🪑 Chair           🛋️ Couch        📺 TV              ⌨️ Keyboard
🎒 Backpack        🔑 Book         ✂️ Scissors        🐕 Dog / Cat
🚗 Car / Bike      🥪 Sandwich     🍕 Pizza           🖱️ Mouse / Remote
```

### Model Source & Conversion with LiteRT-Torch
The PyTorch model (`yolov8n.pt`) is converted directly into a LiteRT `.tflite` model using **LiteRT-Torch 0.9.1**:

```python
from ultralytics import YOLO

# Load standard PyTorch model
model = YOLO('yolov8n.pt')

# Export to unified Google LiteRT format
model.export(format='tflite') 
# Output: yolov8n.tflite (12.2 MB)
```

---

## 🛠️ Step 1: Project Setup & GMS Dependencies

### 1. `app/build.gradle.kts`
Add thin client dependencies for ML Kit and LiteRT GMS runtime:

```kotlin
dependencies {
    // ── ML Kit Object Detection via GMS ───────────────────────────
    implementation("com.google.mlkit:object-detection:17.0.2")

    // ── LiteRT (TFLite) Runtime & GPU Delegate via GMS Core ───────
    implementation("com.google.android.gms:play-services-tflite-java:16.4.0")
    implementation("com.google.android.gms:play-services-tflite-gpu:16.4.0")
    implementation("com.google.android.gms:play-services-tflite-support:16.4.0")

    // ── CameraX & Jetpack Compose ─────────────────────────────────
    val cameraVersion = "1.4.1"
    implementation("androidx.camera:camera-camera2:$cameraVersion")
    implementation("androidx.camera:camera-lifecycle:$cameraVersion")
    implementation("androidx.camera:camera-view:$cameraVersion")
    implementation(platform("androidx.compose:compose-bom:2024.12.01"))
    implementation("androidx.compose.material3:material3")
}
```

### 2. `AndroidManifest.xml`
Request Google Play Services to pre-download the ML Kit Object Detection model at app install time so there is zero latency on first run:

```xml
<manifest xmlns:android="http://schemas.android.com/apk/res/android">
    <uses-permission android:name="android.permission.CAMERA" />
    <uses-feature android:name="android.hardware.camera" android:required="true" />

    <application ...>
        <!-- Auto-download ML Kit Object Detection model on app install -->
        <meta-data
            android:name="com.google.mlkit.vision.DEPENDENCIES"
            android:value="object_detection" />
    </application>
</manifest>
```

---

## 🎨 Step 2: Image Pre-Processing Deep Dive

Raw camera frames cannot be passed directly into vision models. Pre-processing transforms camera crops into the exact tensor format expected by YOLOv8.

```
Raw Crop ──► 12% Margin Padding ──► Aspect-Ratio Letterbox (640x640) ──► Float32 RGB Buffer
```

### A. Bounding Box Margin Padding (12%)
ML Kit bounding boxes crop tightly around the object. Tight cropping cuts off contextual edges (e.g., mug handles, bottle caps). We expand the crop region by **12%** on all sides:

```kotlin
val padX = (rawBox.width() * 0.12f).toInt()
val padY = (rawBox.height() * 0.12f).toInt()

val safeRect = Rect(
    Math.max(0, rawBox.left - padX),
    Math.max(0, rawBox.top - padY),
    Math.min(bitmap.width, rawBox.right + padX),
    Math.min(bitmap.height, rawBox.bottom + padY)
)
```

### B. Aspect-Ratio Preserving Letterboxing
Standard `Bitmap.createScaledBitmap(..., 640, 640)` squashes non-square objects (e.g., tall bottles become short and wide), ruining shape features. 

**Letterboxing** resizes the image while preserving aspect ratio, centering it on a `640×640` canvas filled with neutral gray `RGB(128, 128, 128)`:

```kotlin
private fun letterboxBitmap(source: Bitmap, targetSize: Int): Bitmap {
    val result = Bitmap.createBitmap(targetSize, targetSize, Bitmap.Config.ARGB_8888)
    val canvas = Canvas(result)

    // Neutral gray background (128, 128, 128)
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
```

### C. Normalization & Channel Format (BCHW vs BHWC)
YOLOv8 requires input normalized to Float32 values between `[0.0, 1.0]`. We inspect model input shape dynamically to support both **BHWC** `[1, 640, 640, 3]` and **BCHW** `[1, 3, 640, 640]` layouts:

```kotlin
private fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
    val byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3)
    byteBuffer.order(ByteOrder.nativeOrder())

    val intValues = IntArray(imageSize * imageSize)
    bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)

    if (isBchw) { // Planar [1, 3, 640, 640]
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
    } else { // Interleaved [1, 640, 640, 3]
        for (v in intValues) {
            byteBuffer.putFloat(((v shr 16) and 0xFF) / 255.0f)
            byteBuffer.putFloat(((v shr 8) and 0xFF) / 255.0f)
            byteBuffer.putFloat((v and 0xFF) / 255.0f)
        }
    }
    return byteBuffer
}
```

---

## ⚡ Step 3: Thread-Safe GMS Execution & GPU Acceleration

GPU drivers (Qualcomm Adreno OpenCL/Vulkan) **require the LiteRT GPU delegate to be created, executed, and closed on the exact same background thread**. Running LiteRT across generic coroutine pools (`Dispatchers.IO`) causes GPU driver hangs.

### Thread-Safety Pattern

```kotlin
class SmartLensViewModel(application: Application) : AndroidViewModel(application) {

    private val mlExecutor = Executors.newSingleThreadExecutor()
    private val mlDispatcher = mlExecutor.asCoroutineDispatcher()
    private val mlMutex = Mutex()

    private fun initializeModels() {
        viewModelScope.launch(mlDispatcher) {
            mlMutex.withLock {
                val isGpuAvailable = TfLiteGpu.isGpuDelegateAvailable(getApplication()).await()
                val initOptions = TfLiteInitializationOptions.builder()
                    .setEnableGpuDelegateSupport(isGpuAvailable)
                    .build()

                TfLite.initialize(getApplication(), initOptions).await()
                imageClassifierHelper = ImageClassifierHelper(getApplication())
                _isInitialized.value = true
            }
        }
    }

    fun toggleGpu() {
        viewModelScope.launch(mlDispatcher) {
            mlMutex.withLock {
                imageClassifierHelper?.toggleGpu(!_isGpuEnabled.value)
                _isGpuEnabled.value = !_isGpuEnabled.value
            }
        }
    }
}
```

---

## 📊 Step 4: Image Post-Processing & YOLOv8 Output Parsing

### Understanding YOLOv8 Output Matrix
YOLOv8 returns a 3D float array with shape **`[1, 84, 8400]`**:
- **`8,400`**: Spatial candidate anchor boxes across the 640×640 frame.
- **`84`**: Channels per anchor:
  - Rows 0, 1, 2, 3: Bounding box center & size `(cx, cy, w, h)`
  - Rows 4 to 83: 80 COCO class probability scores (`0.0` to `1.0`)

```
Output Array: [1][84][8400]
 ┌────────────────────────────────────────────────────────────┐
 │ Row 0: cx                                                  │
 │ Row 1: cy                                                  │
 │ Row 2: w                                                   │
 │ Row 3: h                                                   │
 │ Row 4: "person" score                                      │
 │ Row 5: "bicycle" score                                     │
 │ ...                                                        │
 │ Row 83: "toothbrush" score                                 │
 └────────────────────────────────────────────────────────────┘
   Col 0      Col 1      Col 2   ...  Col 8399
```

### Parsing Class Confidence Scores

```kotlin
fun classify(bitmap: Bitmap): ClassificationResult {
    val letterboxedBitmap = letterboxBitmap(bitmap, imageSize)
    val inputBuffer = convertBitmapToByteBuffer(letterboxedBitmap)
    
    val outputArray = Array(1) { Array(84) { FloatArray(8400) } }

    val startTime = System.nanoTime()
    interpreter?.run(inputBuffer, outputArray)
    val endTime = System.nanoTime()

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

    return ClassificationResult(topResults, (endTime - startTime) / 1_000_000)
}
```

---

## 🧹 Step 5: Code Optimization, Modular Composables & Asset Minimization

To maintain a clean codebase and minimize APK size:

1. **Asset Minimization (-5.2 MB)**: Clean out legacy unused models and label files from `app/src/main/assets/`, leaving only `yolov8n.tflite` (12.2 MB) and `coco_labels.txt`.
2. **Modular Composables**: In `MainActivity.kt`, break large inline screens into reusable private composables (`TuningSettingsPanel`, `PermissionRequestContent`).
3. **Helper Method Extraction**: In `SmartLensViewModel.kt`, extract crop & classification logic into `cropAndClassify()` and frame inference into `processFrameInference()`.
4. **Expression Bodies**: Use Kotlin single-expression functions (`runCatching`) in helper classes like `ObjectDetectorHelper.kt`.

---

## 🏁 Summary Checklist

- [x] Use `com.google.mlkit:object-detection` and `play-services-tflite-java` for GMS delivery (keep APK size ~5 MB).
- [x] Select a practical model trained on COCO 80 items (YOLOv8) over 1,000-class academic ImageNet models.
- [x] Add **12% padding margin** around bounding box crops.
- [x] Apply **aspect-ratio letterboxing** with neutral gray fill before resizing to 640×640.
- [x] Enforce a **single-threaded executor + Mutex** for all LiteRT interpreter & GPU delegate operations.
- [x] Clean up asset directory (-5.2 MB) and modularize UI into reusable composables.
