# Copilot Instructions for DogDayDiffuser

## Big picture architecture
- This is a real-time OpenCV pipeline (not generative AI). Keep per-frame work cheap and vectorized.
- Main control flow lives in `main.py` (`DogDayDiffuser.run()`):
  1) camera frame capture, 2) face detection every N frames, 3) optional audio features, 4) effect `apply()`, 5) render + key handling.
- Effects are plug-in style classes registered in `effects/__init__.py` (`EFFECTS`, `EFFECT_ORDER`).
- Frame contract across modules: BGR `np.uint8` arrays at internal resolution (default 320x240).

## Core component boundaries
- `camera.py`: owns `cv2.VideoCapture`, returns resized frames or `None` on read failure.
- `face_detection.py`: detector abstraction + factory (`build_detector`) with graceful OpenVINO -> Haar fallback.
- `audio_reactivity.py`: optional `sounddevice` input; `AudioReactor.get_features()` returns smoothed `AudioFeatures` snapshot.
- `renderer.py`: upscales display (2x), overlays FPS/effect name, handles fullscreen/window lifecycle.

## Project-specific coding patterns
- Favor graceful degradation over hard failure for optional features:
  - Audio import/stream failures disable audio and continue.
  - OpenVINO failures warn and fall back to Haar detector.
- Avoid Python pixel loops in effects. Use NumPy + OpenCV ops (`cv2.remap`, `cv2.warpAffine`, `cv2.addWeighted`, etc.).
- Cache expensive per-frame maps/buffers and rebuild only on meaningful parameter changes:
  - e.g. `KaleidoscopeEffect._rebuild_maps()`, `WarpEffect._rebuild_maps()`, `FeedbackEffect._buffer`.
- When adding effect controls, wire keyboard behavior in `DogDayDiffuser._handle_key()` and keep shared knobs compatible (`trail_decay`, `warp_amount`).

## Adding or changing effects
- New effects should subclass `effects/base.py` `BaseEffect` and implement:
  `apply(frame, face: Optional[FaceInfo], audio: Optional[AudioFeatures]) -> np.ndarray`.
- Register new effect names/classes in `effects/__init__.py` so CLI + key cycling can discover them.
- Respect optional inputs (`face`/`audio` may be `None`) and return a valid BGR `uint8` frame.

## Runtime workflows (no formal test suite)
- Create env + install deps: `pip install -r requirements.txt`
- Run app: `python3 main.py`
- Useful manual checks while running:
  - switch effects (`1`-`4`, `space`),
  - toggle face/audio (`d`, `a`),
  - tune parameters (`+/-`, `w/s`),
  - exit (`q` or `Esc`).
- Common startup failure is camera open error (`Camera.open()`); verify device index (`--camera`) and webcam availability.

## Config and integration points
- Config precedence in `config.py`: defaults < JSON file (`--config`) < explicit CLI args.
- Optional dependencies are intentionally commented in `requirements.txt`; do not force-enable unless requested.
- OpenVINO expects model path `models/face-detection-retail-0004.xml` unless overridden in detector factory usage.
