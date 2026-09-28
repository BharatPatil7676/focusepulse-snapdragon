# FocusPulse — Non-Contact Fatigue Monitor for Study Sessions

Built for the Qualcomm Snapdragon AI Lab Build & Present Challenge.

Reads your invisible facial pulse (rPPG) and blink pattern through the webcam
to detect mental fatigue during long study sessions, fully offline — designed
to run efficiently on Snapdragon's Hexagon NPU for all-day background use.

## Day-by-day build plan

**Day 1 — Get the pipeline visibly working**
- `pip install -r requirements.txt`
- `python app.py`
- Sit still, good lighting, face the camera. Watch the HR number — it should
  land in a believable range (60-100 bpm at rest). If it's wild/unstable:
  - Improve lighting (avoid backlight)
  - Hold very still for the first 10 seconds
  - Check the forehead ROI overlay is actually on your forehead

**Day 2 — Validate HRV makes sense**
- Let it run for the full 45s calibration + a few minutes after.
- Try one relaxed session and one where you do mental math / stay stressed —
  SDNN should trend down in the stressed session. Note real numbers, don't
  fabricate results in the pitch.

**Day 3 — Blink detection + fusion tuning**
- Confirm blink detection triggers when you actually blink (watch the
  `closed_frac` number rise briefly).
- Tune `EAR_BLINK_THRESHOLD` and the fatigue score weighting in
  `FatigueScorer.score()` against how you actually feel during a real
  session — this is where you make the heuristic defensible, not arbitrary.

**Day 4 — Train a small classifier (optional but strengthens the pitch)**
- Log (sdnn, blink_closed_frac, hour_of_session) -> self-rated fatigue
  (1-5) across a few of your own study sessions.
- Train a small RandomForestClassifier (scikit-learn) on this data.
- Export to ONNX with `skl2onnx` or `onnxmltools`.

**Day 5 — Run it through Qualcomm AI Hub / QNN**
- Sign up at https://aihub.qualcomm.com, get an API token.
- Use `qai_hub.submit_compile_job()` targeting a Snapdragon X Elite device
  to compile your ONNX model — this gives you real profiling numbers to cite.
- On the actual Snapdragon laptop, run inference through `onnxruntime` with
  `QNNExecutionProvider`, falling back to `DmlExecutionProvider` then
  `CPUExecutionProvider` if QNN isn't available — same pattern used by other
  strong submissions in this challenge.
- Record real latency/power numbers if you can measure them; if you can't
  measure power directly, cite Qualcomm's published Hexagon NPU TOPS/power
  figures and be explicit in your pitch about what you measured vs. what
  you're citing.

**Day 6 — Streamlit dashboard (nicer demo than a raw OpenCV window)**
- Wrap the same pipeline in a `streamlit run dashboard.py` app with a live
  pulse waveform chart and fatigue score gauge — much better for a judged
  demo than a terminal window.

**Day 7 — Deliverables**
- README (this file, expanded with your real results)
- Short project description (PDF)
- Pitch deck (PPT) — lead with the problem (burnout during exam prep),
  show the live demo, show your real benchmark numbers, end with the
  Snapdragon NPU efficiency case (continuous background monitoring needs
  low power — this is exactly what the NPU is for).
- Clean GitHub repo with this code, tests if you have time, and the docs above.

## Honesty checklist before you submit
- [ ] Every "on-device" claim is something you actually ran, not assumed
- [ ] NPU benchmark numbers are labeled clearly as measured vs. cited
- [ ] The app is framed as a fatigue *indicator*, not a medical/diagnostic tool
- [ ] Calibration behavior (45s baseline) is explained in the pitch, not hidden
