FocusPulse
Real-time fatigue and attention monitoring using webcam-based computer vision.

What it does
FocusPulse uses a webcam to monitor facial and eye-related signals and estimate fatigue in real time.
The system processes facial information and provides a fatigue indication to help identify reduced attention or
possible fatigue.

How it works
Webcam
  |
  v
Face Detection / Facial Landmarks
  |
  v
Eye & Blink Analysis
  |
  v
Fatigue / Attention Estimation
  |
  v
Real-Time Output

Technology Stack
• Python
• OpenCV
• MediaPipe
• SciPy
• Webcam-based computer vision

Qualcomm AI Hub / Snapdragon Validation
A selected face-processing model, FaceMap 3DMM, was compiled and profiled on a real Snapdragon device
using Qualcomm AI Hub.

Verified profiling result
• Device: Samsung Galaxy S25
• Runtime: TFLite
• Estimated inference time: ~0.2 ms
• Peak memory: 38 MiB
• Total operations: 38
• Compute units: NPU: 38 operations, GPU: 0, CPU: 0
This demonstrates that the selected model configuration can be accelerated using the Snapdragon NPU.
The Qualcomm AI Hub profiling validates the selected model configuration. It does not mean that the complete
Python application is currently running end-to-end on the NPU.

Current Limitations
• The current Python application is not an end-to-end NPU deployment.
• AI Hub profiling was performed on a selected face-processing model rather than the complete application.
• The profiled FaceMap 3DMM model is not identical to the application's 468-point MediaPipe Face Mesh
pipeline.
• Fatigue estimation can be affected by lighting, camera position, head movement and individual differences.
• Further optimization is required for complete on-device NPU deployment.

Future Scope
• Integrate an optimized Qualcomm AI Hub model into the complete pipeline.
• Deploy the application on Snapdragon hardware.
• Improve fatigue estimation using temporal analysis.
• Add personalized fatigue thresholds.
• Optimize power and latency for continuous monitoring.

Conclusion
FocusPulse demonstrates a computer-vision approach for real-time fatigue and attention monitoring, together
with Qualcomm AI Hub validation of a selected face-processing model on Snapdragon hardware.
The project provides a foundation for future NPU-accelerated, low-latency fatigue monitoring applications.
