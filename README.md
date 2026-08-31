# Smart Lens 🔍

Real-time Android object detection and classification app powered by **YOLOv8-nano**, **ML Kit**, **LiteRT (TensorFlow Lite)** via **Google Play Services (GMS)**, **CameraX**, and **Jetpack Compose**.

> 📚 **Beginner Step-by-Step Guide**: Read the comprehensive [**TUTORIAL.md**](TUTORIAL.md) for full architecture details, GMS Core deployment benefits, pre-processing math, YOLOv8 output matrix parsing, and GPU thread-safety patterns.

## Features

- 🎯 **YOLOv8-nano LiteRT Classification**: Converted via **LiteRT-Torch 0.9.1** to classify 80 everyday COCO objects (*cell phone, laptop, coffee mug, chair, bottle, backpack, etc.*) instead of legacy academic ImageNet classes.
- 👁️ **Real-Time Object Detection**: Uses ML Kit Object Detection delivered via Google Play Services to locate bounding boxes in the live camera feed.
- ⚙️ **In-App Accuracy Tuning Panel**: Interactive settings drawer (⚙️ icon) to adjust:
  - **Min Confidence Threshold** (10% to 90%)
  - **Box Padding Margin** (0% to 30% padding)
  - **Detector Mode** (Fast `STREAM_MODE` vs High-Res `SINGLE_IMAGE_MODE`)
  - **Max Objects Count** (1 to 5 objects)
- 📐 **Pixel-Perfect Alignment & Letterboxing**:
  - Aspect-ratio letterboxing prevents object distortion.
  - Uniform CameraX `FILL_CENTER` scaling & rotation math ensures bounding boxes align accurately in portrait mode.
- ⚡ **Thread-Safe GPU Acceleration**:
  - Dynamic runtime toggle between CPU (multithreaded) and GPU (`GpuDelegateFactory`) via Google Play Services.
  - Dedicated single-threaded executor + Mutex synchronization guarantees GPU context stability on mobile SoCs (e.g. Qualcomm Snapdragon Adreno GPU).
- 📦 **Minimal APK Footprint**: ML Kit and LiteRT runtimes are delivered dynamically via Google Play Services, avoiding heavy bundled native binaries.

## 📦 Why Deployment via Google Play Services (GMS) Matters

Instead of bundling heavy C++ native libraries (`libtensorflowlite_jni.so`) inside your APK, Smart Lens leverages Google Play Services for dynamic runtime delivery:

| Feature | Standard Bundled TFLite | **Smart Lens (GMS Delivered)** |
|---|---|---|
| **APK Size** | ~30 MB – 80 MB | **~5 MB** |
| **TFLite Engine** | Bundled inside your APK | Loaded dynamically from Google Play Services |
| **GPU Drivers** | Compiled into app | Provided by device GMS system updates |
| **Updates** | Requires app update in Play Store | Google updates ML runtime automatically in background |

### Key Benefits

1. **Smaller Downloads**: Keeps the initial APK download size tiny for users.
2. **Automatic Runtime & Security Updates**: Google Play Services updates the underlying C++ inference engines and security fixes without needing a new app release.
3. **Hardware Acceleration Compatibility**: GMS dynamically manages GPU delegate compatibility (Vulkan/OpenCL) across different SoC chipsets (e.g. Snapdragon, Exynos, Tensor).

## Architecture & Tech Stack

- **UI**: Jetpack Compose + Material 3
- **Camera**: CameraX (`camera-camera2`, `camera-lifecycle`, `camera-view`)
- **Object Detection**: `com.google.mlkit:object-detection`
- **LiteRT Engine**: `com.google.android.gms:play-services-tflite-java`
- **LiteRT GPU Delegate**: `com.google.android.gms:play-services-tflite-gpu`
- **Classifier Model**: `yolov8n.tflite` (12.2 MB, LiteRT-Torch 0.9.1)
- **Language**: Kotlin 2.0 + Coroutines + StateFlow

## Setup & Running

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Ch1ntanpar1kh/SmartLens.git
   ```
2. **Open in Android Studio**.
3. **Connect an Android device** (e.g., Samsung S23) with USB Debugging enabled.
4. **Build & Run**: Click **Run ▶** in Android Studio.

## License

MIT License
