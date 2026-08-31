# Smart Lens 🔍

Real-time Android object detection and fine-grained classification app built with **ML Kit**, **LiteRT (TensorFlow Lite)** via **Google Play Services (GMS)**, **CameraX**, and **Jetpack Compose**.

## Features

- 👁️ **Real-Time Object Detection**: Uses ML Kit Object Detection delivered via Google Play Services to identify and bound objects in live camera stream.
- 🏷️ **Fine-Grained Classification**: Uses quantized MobileNet V2 `.tflite` model running via LiteRT `InterpreterApi` for 1,000 ImageNet categories.
- ⚡ **GPU Acceleration**: Dynamic runtime toggle between CPU (multithreaded) and GPU (`GpuDelegateFactory`) via Google Play Services.
- 📦 **Minimal APK Footprint**: Both ML Kit and LiteRT runtimes are delivered dynamically via Google Play Services, avoiding heavy bundled binaries.
- 🎨 **Modern Jetpack Compose UI**: CameraX preview with custom Canvas overlay for real-time bounding boxes and confidence scores.

## Architecture & Tech Stack

- **UI**: Jetpack Compose + Material 3
- **Camera**: CameraX (`camera-camera2`, `camera-lifecycle`, `camera-view`)
- **Object Detection**: `com.google.mlkit:object-detection`
- **LiteRT Engine**: `com.google.android.gms:play-services-tflite-java`
- **LiteRT GPU Delegate**: `com.google.android.gms:play-services-tflite-gpu`
- **Language**: Kotlin 2.0 + Coroutines + StateFlow

## Setup & Running

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Ch1ntanpar1kh/SmartLens.git
   ```
2. **Open in Android Studio** (Koa or newer recommended).
3. **Connect an Android device** (e.g. Samsung S23) with USB Debugging enabled.
4. **Build & Run**: Click **Run ▶** in Android Studio.

## License

MIT License
