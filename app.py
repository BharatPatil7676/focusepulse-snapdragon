"""
FocusPulse — Non-contact fatigue monitor for study sessions.

Pipeline:
  1. Webcam frame -> MediaPipe Face Mesh landmarks
  2. Forehead ROI -> average RGB per frame -> buffered signal
  3. POS algorithm -> extract blood volume pulse (rPPG) from RGB signal
  4. Bandpass filter (0.75-2.5 Hz) -> isolate heartbeat frequency band
  5. Peak detection -> Inter-Beat Intervals -> HRV (SDNN)
  6. Eye Aspect Ratio (EAR) from eye landmarks -> blink rate / closure duration
  7. Fuse HRV drop + blink slowdown -> fatigue score -> on-screen alert

Run:
    python app.py
Press 'q' to quit.
"""

import time
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
from scipy.signal import butter, filtfilt, find_peaks

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FPS_ASSUMED = 30                 # used for filter design; actual fps is measured too
BUFFER_SECONDS = 10              # rPPG signal window for pulse extraction
BUFFER_LEN = FPS_ASSUMED * BUFFER_SECONDS
HRV_WINDOW_SECONDS = 90          # rolling window for HRV/fatigue trend
LOW_HZ, HIGH_HZ = 0.75, 2.5      # 45-150 bpm band
EAR_BLINK_THRESHOLD = 0.21       # eye aspect ratio below this = eye closed
CALIBRATION_SECONDS = 45         # first N seconds used as personal baseline

mp_face_mesh = mp.solutions.face_mesh

# Landmark indices (MediaPipe Face Mesh, 468-point model)
FOREHEAD_IDX = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323]
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]


# ---------------------------------------------------------------------------
# rPPG: POS algorithm
# ---------------------------------------------------------------------------
def pos_algorithm(rgb_signal: np.ndarray) -> np.ndarray:
    """
    Plane-Orthogonal-to-Skin (POS) method.
    rgb_signal: shape (N, 3), columns = [R, G, B] averaged over ROI per frame.
    Returns a 1D pulse signal.
    """
    mean_rgb = np.mean(rgb_signal, axis=0)
    mean_rgb[mean_rgb == 0] = 1e-6
    normalized = rgb_signal / mean_rgb  # temporal normalization

    S1 = normalized[:, 1] - normalized[:, 2]                     # G - B
    S2 = normalized[:, 1] + normalized[:, 2] - 2 * normalized[:, 0]  # G + B - 2R

    std1, std2 = np.std(S1), np.std(S2)
    alpha = std1 / std2 if std2 != 0 else 0
    pulse = S1 + alpha * S2
    return pulse - np.mean(pulse)


def bandpass_filter(signal: np.ndarray, fs: float, low=LOW_HZ, high=HIGH_HZ) -> np.ndarray:
    nyq = 0.5 * fs
    low_n, high_n = low / nyq, min(high / nyq, 0.99)
    if low_n <= 0 or low_n >= high_n:
        return signal
    b, a = butter(2, [low_n, high_n], btype="band")
    return filtfilt(b, a, signal)


def estimate_hr_and_hrv(pulse: np.ndarray, fs: float):
    """Return (heart_rate_bpm, sdnn_ms) from a filtered pulse waveform."""
    min_distance = int(fs * 60 / 150)  # cap at 150 bpm
    peaks, _ = find_peaks(pulse, distance=max(min_distance, 1))
    if len(peaks) < 3:
        return None, None

    ibi_samples = np.diff(peaks)              # inter-beat intervals, in samples
    ibi_ms = (ibi_samples / fs) * 1000.0       # convert to milliseconds
    ibi_ms = ibi_ms[(ibi_ms > 400) & (ibi_ms < 1500)]  # keep physiological range (40-150bpm)
    if len(ibi_ms) < 2:
        return None, None

    hr_bpm = 60000.0 / np.mean(ibi_ms)
    sdnn = np.std(ibi_ms)  # HRV metric: standard deviation of NN intervals
    return hr_bpm, sdnn


# ---------------------------------------------------------------------------
# Blink / EAR
# ---------------------------------------------------------------------------
def eye_aspect_ratio(landmarks, idx, w, h):
    pts = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in idx])
    p1, p2, p3, p4, p5, p6 = pts
    vertical1 = np.linalg.norm(p2 - p6)
    vertical2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)
    if horizontal == 0:
        return 0.3
    return (vertical1 + vertical2) / (2.0 * horizontal)


# ---------------------------------------------------------------------------
# Fatigue fusion
# ---------------------------------------------------------------------------
class FatigueScorer:
    def __init__(self):
        self.baseline_sdnn = None
        self.baseline_blink_rate = None
        self.calibrated = False

    def calibrate(self, sdnn_history, blink_rate):
        if sdnn_history:
            self.baseline_sdnn = float(np.median(sdnn_history))
        self.baseline_blink_rate = blink_rate
        self.calibrated = True

    def score(self, current_sdnn, current_blink_rate, avg_ear_closed_frac):
        """
        Returns a 0-100 fatigue score.
        Heuristic, rule-based — documented and explainable, not a black box.
        """
        if not self.calibrated or current_sdnn is None:
            return 0.0

        # HRV drop relative to personal baseline -> higher fatigue
        hrv_ratio = current_sdnn / max(self.baseline_sdnn, 1e-6)
        hrv_component = np.clip((1.0 - hrv_ratio), 0, 1) * 50  # up to 50 points

        # Blink rate slowdown / longer closures -> higher fatigue
        blink_component = np.clip(avg_ear_closed_frac * 4, 0, 1) * 50  # up to 50 points

        return float(np.clip(hrv_component + blink_component, 0, 100))


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check camera permissions/index.")

    rgb_buffer = deque(maxlen=BUFFER_LEN)
    timestamps = deque(maxlen=BUFFER_LEN)
    sdnn_history = deque(maxlen=20)
    ear_history = deque(maxlen=FPS_ASSUMED * 5)  # last ~5s for closure fraction

    scorer = FatigueScorer()
    start_time = time.time()
    last_hr, last_sdnn = None, None

    with mp_face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as face_mesh:

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark

                # --- Forehead ROI -> mean RGB ---
                pts = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in FOREHEAD_IDX], dtype=np.int32)
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillConvexPoly(mask, cv2.convexHull(pts), 255)
                mean_color = cv2.mean(frame, mask=mask)[:3]  # BGR
                rgb_buffer.append([mean_color[2], mean_color[1], mean_color[0]])  # store as R,G,B
                timestamps.append(time.time())

                # --- Blink / EAR ---
                ear_l = eye_aspect_ratio(landmarks, LEFT_EYE_IDX, w, h)
                ear_r = eye_aspect_ratio(landmarks, RIGHT_EYE_IDX, w, h)
                ear = (ear_l + ear_r) / 2.0
                ear_history.append(ear < EAR_BLINK_THRESHOLD)

                cv2.polylines(frame, [cv2.convexHull(pts)], True, (0, 255, 0), 1)

                # --- Once we have enough buffer, run rPPG ---
                if len(rgb_buffer) == BUFFER_LEN:
                    fs = BUFFER_LEN / max(timestamps[-1] - timestamps[0], 1e-6)
                    pulse_raw = pos_algorithm(np.array(rgb_buffer))
                    pulse_filtered = bandpass_filter(pulse_raw, fs)
                    hr, sdnn = estimate_hr_and_hrv(pulse_filtered, fs)

                    if hr is not None:
                        last_hr, last_sdnn = hr, sdnn
                        sdnn_history.append(sdnn)

                elapsed = time.time() - start_time
                closed_frac = float(np.mean(ear_history)) if ear_history else 0.0

                if not scorer.calibrated and elapsed > CALIBRATION_SECONDS and sdnn_history:
                    scorer.calibrate(list(sdnn_history), closed_frac)

                fatigue = scorer.score(last_sdnn, None, closed_frac)

                # --- Overlay ---
                y = 30
                for text in [
                    f"HR: {last_hr:.1f} bpm" if last_hr else "HR: calibrating...",
                    f"HRV (SDNN): {last_sdnn:.1f} ms" if last_sdnn else "HRV: calibrating...",
                    f"Blink closure frac: {closed_frac:.2f}",
                    f"Fatigue score: {fatigue:.0f}/100" if scorer.calibrated else "Calibrating baseline...",
                ]:
                    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    y += 28

                if scorer.calibrated and fatigue > 60:
                    cv2.putText(frame, "TAKE A BREAK", (10, y + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            cv2.imshow("FocusPulse - Fatigue Monitor", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
