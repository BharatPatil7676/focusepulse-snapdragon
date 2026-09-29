# FocusPulse: Offline Fatigue Monitor for Study Sessions

Built for the **Snapdragon® AI Lab Build & Present Challenge** (Qualcomm x Unstop).

FocusPulse uses an ordinary webcam to notice when a student is getting mentally tired during long study sessions, and suggests a break before focus collapses. It reads the pulse from tiny colour changes in the skin (rPPG), tracks heart-rate variability (HRV) and blinking, and combines them into a 0-100 fatigue score. Everything runs locally. No video or health data leaves the laptop.

![Calm state](images/calm.png)
![Break alert](images/alert.png)

> Screenshots: calm state (low score) and the "take a break" alert.

## The problem

Students preparing for exams and placements study for hours and rarely notice fatigue building. They push until focus drops. Wearables can measure this, but most students don't own one. Almost every student has a webcam.

## How it works

1. **Face tracking:** MediaPipe Face Mesh (468 landmarks) finds the forehead and eyes on every frame.
2. **Pulse (rPPG):** the average forehead colour per frame is turned into a pulse signal with the POS (Plane-Orthogonal-to-Skin) algorithm.
3. **Filtering:** a Butterworth bandpass filter (0.75-2.5 Hz, i.e. 45-150 bpm) keeps only the heartbeat band.
4. **HRV:** peaks give inter-beat intervals, and their spread (SDNN) is the variability measure. Lower HRV compared to your own baseline points to rising fatigue.
5. **Blinks:** Eye Aspect Ratio (EAR) from the eye landmarks gives eye-closure fraction. Longer closures point to drowsiness.
6. **Fusion:** a transparent rule-based score. Up to 50 points from HRV drop against a personal baseline (first 45 seconds), and up to 50 from eye closure. An alert appears when the score gets high.

The score is rule-based on purpose, so every point of it can be explained.

## Why Snapdragon

Fatigue monitoring is an always-on background job that runs for hours, which is exactly where a low-power NPU beats a CPU that would drain the battery. It is also privacy-sensitive (face video plus heart data), so it should never go to the cloud.

The two neural networks in the per-frame loop are the face detector and the 468-point face landmark model. We profiled both on a **cloud-hosted Snapdragon X Elite** through Qualcomm AI Hub:

| Model | Est. inference time | Ops on NPU | Ops on CPU/GPU |
|---|---|---|---|
| Face detector | 0.7 ms | 145 / 145 | 0 |
| Face landmarks (468 points) | 0.3 ms | 105 / 105 | 0 |

- Together that is about 1 ms of a 33 ms frame budget at 30 fps (about 3%), leaving the rest for the pulse and HRV math.
- Accuracy check: NPU outputs matched the reference model (PSNR 67-138 dB, against the 30 dB "good" guideline).
- Full details and job links: [benchmarks/RESULTS.md](benchmarks/RESULTS.md)

## Honest scope

- The times are **estimates from Qualcomm AI Hub profiling** on a cloud-hosted Snapdragon X Elite device, not measurements from our own laptop.
- The demo app in this repo currently runs MediaPipe on the **CPU**. The NPU part is a validated deployment path with real profiling data. Running the full app on a Snapdragon laptop through ONNX Runtime with the QNN provider is the next step.
- FocusPulse is a **study-fatigue indicator, not a medical device**. rPPG accuracy depends on lighting, distance and movement, and the thresholds were tuned on one person.

## Run it

Requires Python 3.10-3.12 (MediaPipe does not support newer versions yet) and a webcam.

```bash
pip install -r requirements.txt
python app.py
```

Sit facing a light source, stay still for the first 10 seconds. The first 45 seconds calibrate your personal baseline. Press `q` to quit.

## Roadmap

- Run the full pipeline on a Snapdragon X laptop via ONNX Runtime QNN and measure real latency and power
- Streamlit dashboard with a live pulse waveform
- Small classifier trained on self-rated fatigue, exported to ONNX
- Validation against a reference pulse sensor

## Author

Bharat Patil (GitHub: [BharatPatil7676](https://github.com/BharatPatil7676)), ECE student.
