"""
ANTIGRAVITY ASL Interpreter — Optimized Real-Time Detection Engine
==================================================================
Performance:  TFLite inference + async processing + lightweight HUD
Accuracy:     Tuned thresholds, proper smoothing, full vocabulary
Design:       Clean glassmorphism HUD with minimal overdraw
"""

import cv2
import mediapipe as mp
import numpy as np
import threading
import subprocess
from collections import deque, Counter
import time
import os
import json

# ─── Paths ────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
MODELS_DIR = os.path.join(PROJECT_DIR, 'models')
TFLITE_PATH = os.path.join(MODELS_DIR, 'landmark_model.tflite')
KERAS_PATH = os.path.join(MODELS_DIR, 'landmark_model.keras')
LABEL_MAP_PATH = os.path.join(MODELS_DIR, 'label_map.json')

# ─── Label Loading ────────────────────────────────────────────────────
def load_labels():
    """Load the label map with proper error handling."""
    try:
        with open(LABEL_MAP_PATH, 'r') as f:
            labels = json.load(f)['labels']
        print(f"[Labels] Loaded {len(labels)} signs: {', '.join(labels)}")
        return labels
    except FileNotFoundError:
        print(f"[Error] {LABEL_MAP_PATH} not found. Run training first.")
        raise SystemExit(1)
    except (json.JSONDecodeError, KeyError) as e:
        print(f"[Error] Malformed label_map.json: {e}")
        raise SystemExit(1)

LABELS = load_labels()
L_ALPHABET = [l for l in LABELS if len(l) == 1]
L_WORDS = [l for l in LABELS if len(l) > 1]

# ─── Theme ────────────────────────────────────────────────────────────
# BGR tuples
CLR_ACCENT    = (255, 180, 0)     # Cyan-blue accent
CLR_GOLD      = (0, 200, 255)     # Warm gold
CLR_WHITE     = (240, 240, 240)
CLR_DIM       = (140, 140, 140)
CLR_SUCCESS   = (100, 255, 180)
CLR_PANEL_BG  = (20, 20, 25)
CLR_BORDER    = (60, 60, 65)


class CameraStream:
    """Threaded camera capture — always serves the latest frame with zero buffer lag."""

    def __init__(self, src=0):
        print(f"[Camera] Initializing source {src}...")
        # Use CAP_DSHOW on Windows for faster initialization, but fall back if it fails
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            print("[Camera] CAP_DSHOW failed, trying default backend...")
            self.cap = cv2.VideoCapture(src)
        
        if not self.cap.isOpened():
            print("[CRITICAL] Camera could not be opened.")
            self._ret = False
            self._frame = None
            self._running = False
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._ret = False
        self._frame = None
        self._lock = threading.Lock()
        self._running = True

        self._thread = threading.Thread(target=self._update, name="CameraThread", daemon=True)
        self._thread.start()
        print("[Camera] Thread started (warmup in background).")

    def _update(self):
        # Quick warmup in background to avoid blocking main thread (eventlet loop)
        print("[Camera] Warming up sensors in background...")
        for _ in range(5):
            if not self._running: break
            self.cap.read()
            time.sleep(0.01)

        while self._running:
            ret, frame = self.cap.read()
            with self._lock:
                self._ret = ret
                self._frame = frame

    def read(self):
        with self._lock:
            return self._ret, self._frame.copy() if self._frame is not None else None

    @property
    def is_opened(self):
        return self.cap.isOpened()

    def release(self):
        self._running = False
        self._thread.join(timeout=1)
        self.cap.release()


class InferenceEngine:
    """Handles model loading and prediction, preferring TFLite for speed."""

    def __init__(self):
        self._use_tflite = False
        self._interpreter = None
        self._keras_model = None
        self._input_details = None
        self._output_details = None
        self._load()

    def _load(self):
        # Prefer TFLite for ~5-10x faster inference
        if os.path.exists(TFLITE_PATH):
            try:
                import tensorflow as tf
                self._interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
                self._interpreter.allocate_tensors()
                self._input_details = self._interpreter.get_input_details()
                self._output_details = self._interpreter.get_output_details()
                self._use_tflite = True
                input_shape = self._input_details[0]['shape']
                print(f"[Engine] TFLite model loaded (input shape: {input_shape})")
                return
            except Exception as e:
                print(f"[Engine] TFLite load failed ({e}), falling back to Keras")

        # Fallback to Keras
        if os.path.exists(KERAS_PATH):
            try:
                import tensorflow as tf
                self._keras_model = tf.keras.models.load_model(KERAS_PATH)
                input_shape = self._keras_model.input_shape
                print(f"[Engine] Keras model loaded (input shape: {input_shape})")
            except Exception as e:
                print(f"[Error] Cannot load any model: {e}")
                raise SystemExit(1)
        else:
            print("[Error] No model found. Run training first.")
            raise SystemExit(1)

    @property
    def input_size(self):
        """Return the expected input feature count."""
        if self._use_tflite:
            return self._input_details[0]['shape'][-1]
        return self._keras_model.input_shape[-1]

    def predict(self, features: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """Run inference and return softmax probabilities, with optional category masking."""
        if self._use_tflite:
            input_data = features.astype(np.float32).reshape(1, -1)
            self._interpreter.set_tensor(self._input_details[0]['index'], input_data)
            self._interpreter.invoke()
            probs = self._interpreter.get_tensor(self._output_details[0]['index'])[0]
        else:
            probs = self._keras_model.predict(features.reshape(1, -1), verbose=0)[0]

        if mask is not None:
            # Apply mask to probabilities (set excluded categories to ~0)
            probs = probs * mask
            # Re-normalize if sum > 0
            s = np.sum(probs)
            if s > 0:
                probs = probs / s
                
        return probs


class ASLInterpreter:
    """
    Real-time ASL detection with async inference and clean HUD.

    Hotkeys:
        Q — Quit
        M — Toggle mode (Alphabet / Words / All)
        C — Clear sentence log
    """

    # ── Mode cycle ──
    MODES = ["ALL", "WORDS", "ALPHABET"]

    def __init__(self):
        # MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.35,
            min_tracking_confidence=0.35,
        )

        # Inference engine (TFLite preferred)
        self.engine = InferenceEngine()
        self._feature_size = self.engine.input_size  # 42 or 63

        # State
        self.mode_idx = 0
        self.mode = self.MODES[self.mode_idx]
        self.history = deque(maxlen=10)
        self.sentence = []
        self.last_spoken_time = 0
        self.confidence_threshold = 0.30

        # Landmark smoothing (Dynamic EMA for stability)
        self.prev_lms = {}
        self.min_alpha = 0.08 # Ultra-stable for still hands
        self.max_alpha = 0.60 # Responsive for movement

        # Async inference
        self._latest_result = {}  # hand_idx -> (label, conf)
        self._inference_lock = threading.Lock()

        # FPS tracking
        self._frame_times = deque(maxlen=30)
        self._last_frame_time = time.perf_counter()

        # TTS cooldown
        self._speak_cooldown = 1.8  # seconds between TTS triggers
        
        # Category Masks for Strict Filtering
        self._alphabet_mask = np.array([1.0 if len(l) == 1 else 0.0 for l in LABELS], dtype=np.float32)
        self._words_mask    = np.array([1.0 if len(l) > 1 else 0.0 for l in LABELS], dtype=np.float32)

        # Mode button (top-right corner)
        self._btn_w = 130
        self._btn_h = 32
        self._btn_margin = 12

        self._setup_speech()
        print("[System] ASL Interpreter initialized.")

    def _setup_speech(self):
        """No-op placeholder; speech is fire-and-forget via PowerShell."""
        pass

    def speak(self, text: str):
        """Non-blocking text-to-speech via PowerShell."""
        def _tts():
            try:
                safe_text = text.replace('"', "'").replace("_", " ")
                cmd = (
                    'Add-Type -AssemblyName System.Speech; '
                    f'(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak("{safe_text}")'
                )
                subprocess.Popen(
                    ['powershell', '-Command', cmd],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except OSError:
                pass  # PowerShell not available (non-Windows)
        threading.Thread(target=_tts, daemon=True).start()

    def toggle_mode(self):
        """Cycle through detection modes."""
        self.mode_idx = (self.mode_idx + 1) % len(self.MODES)
        self.mode = self.MODES[self.mode_idx]
        self.history.clear()
        print(f"[Mode] Switched to: {self.mode}")

    # ── Landmark Processing ───────────────────────────────────────────

    def _extract_features(self, hand_landmarks) -> np.ndarray:
        """Extract wrist-relative landmark features (supports 42 or 63 dims)."""
        bx = hand_landmarks.landmark[0].x
        by = hand_landmarks.landmark[0].y
        bz = hand_landmarks.landmark[0].z

        if self._feature_size >= 63:
            # Full 3D features
            feats = []
            for lm in hand_landmarks.landmark:
                feats.extend([lm.x - bx, lm.y - by, lm.z - bz])
            return np.array(feats[:self._feature_size], dtype=np.float32)
        else:
            # 2D features (x, y only)
            feats = []
            for lm in hand_landmarks.landmark:
                feats.extend([lm.x - bx, lm.y - by])
            return np.array(feats[:self._feature_size], dtype=np.float32)

    def _smooth_landmarks(self, hand_label: str, current: np.ndarray) -> np.ndarray:
        """Dynamic EMA smoothing: use handedness to maintain state and prevent flickering."""
        prev = self.prev_lms.get(hand_label)
        if prev is None or prev.shape != current.shape:
            self.prev_lms[hand_label] = current
            return current
        
        # Movement velocity
        avg_vel = np.mean(np.abs(current - prev))
        # Adaptive responsiveness
        alpha = np.clip(avg_vel * 20.0, self.min_alpha, self.max_alpha)
        
        smoothed = prev * (1 - alpha) + current * alpha
        self.prev_lms[hand_label] = smoothed
        return smoothed

    def _is_valid_label(self, label: str) -> bool:
        """Check if a label passes the current mode filter."""
        if self.mode == "ALL":
            return True
        if self.mode == "ALPHABET":
            return label in L_ALPHABET
        if self.mode == "WORDS":
            return label in L_WORDS
        return True

    # ── HUD Drawing ───────────────────────────────────────────────────

    def _draw_header(self, frame, fw, fh):
        """Draw a clean mode toggle button in the top-right corner."""
        bw, bh, margin = self._btn_w, self._btn_h, self._btn_margin
        bx1 = fw - bw - margin
        by1 = margin
        bx2 = fw - margin
        by2 = margin + bh

        # Semi-transparent button background
        roi = frame[by1:by2, bx1:bx2]
        if roi.size > 0:
            overlay = np.full_like(roi, (30, 30, 35), dtype=np.uint8)
            cv2.addWeighted(roi, 0.3, overlay, 0.7, 0, dst=roi)
            frame[by1:by2, bx1:bx2] = roi

        # Button border
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (200, 200, 200), 1, cv2.LINE_AA)

        # Button text
        label = self.mode.capitalize()
        cv2.putText(frame, label, (bx1 + 12, by2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_footer(self, frame, fw, fh):
        """No footer — clean view."""
        pass

    def _draw_hand_hud(self, frame, hand_landmarks, label, confidence, top_3, fw, fh):
        """Exact reference style: red bbox, red text, white bones, pink dots."""
        # Colors (BGR)
        RED = (0, 0, 255)            # Pure red for bbox + text
        CYAN = (255, 255, 0)         # Debug color
        BONE_WHITE = (255, 255, 255) # White skeleton lines
        DOT_PINK = (150, 150, 255)   # Small pinkish-red joint dots
        DOT_OUTLINE = (255, 255, 255)# White outline ring

        # Pre-compute all joint pixel positions
        pts = [(int(lm.x * fw), int(lm.y * fh)) for lm in hand_landmarks.landmark]

        # Bounding box
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        pad = 20
        x1 = max(0, min(xs) - pad)
        y1 = max(0, min(ys) - pad)
        x2 = min(fw, max(xs) + pad)
        y2 = min(fh, max(ys) + pad)

        # Red bounding rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), RED, 2, cv2.LINE_AA)

        # Text: Confidence + Prediction (red, above the box)
        display_label = label.replace("_", " ")
        conf_text = f"Confidence: {confidence * 100:.2f}%"
        pred_text = f"Prediction: {display_label}"

        tx = x1
        conf_y = max(16, y1 - 26)
        pred_y = max(36, y1 - 4)

        cv2.putText(frame, conf_text, (tx, conf_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2, cv2.LINE_AA)
        cv2.putText(frame, pred_text, (tx, pred_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2, cv2.LINE_AA)

        # DEBUG: Top 3 candidates (subtle cyan)
        debug_y = y2 + 15
        for i, (l, c) in enumerate(top_3):
            if debug_y > fh - 10: break
            txt = f"[{i+1}] {l}: {c:.1%}"
            cv2.putText(frame, txt, (x1, debug_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, CYAN, 1, cv2.LINE_AA)
            debug_y += 14

        # Skeleton: thin white bones
        for conn in self.mp_hands.HAND_CONNECTIONS:
            cv2.line(frame, pts[conn[0]], pts[conn[1]], BONE_WHITE, 2, cv2.LINE_AA)

        # Joints: small pink-red dots with white ring
        for jx, jy in pts:
            cv2.circle(frame, (jx, jy), 4, DOT_OUTLINE, -1, cv2.LINE_AA)
            cv2.circle(frame, (jx, jy), 3, DOT_PINK, -1, cv2.LINE_AA)

    def _calculate_fps(self) -> float:
        now = time.perf_counter()
        self._frame_times.append(now - self._last_frame_time)
        self._last_frame_time = now
        if len(self._frame_times) < 2:
            return 0.0
        avg = sum(self._frame_times) / len(self._frame_times)
        return 1.0 / avg if avg > 0 else 0.0

    # ── Core Loop ─────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame: detect hands, run inference, render HUD."""
        # Standardize lighting slightly without heavy artifacts
        frame = cv2.convertScaleAbs(frame, alpha=1.1, beta=5) 
        
        # Lightweight sharpening (Unsharp mask with small radius)
        # removes camera "fuzziness" without introducing artifacts
        blur = cv2.GaussianBlur(frame, (0, 0), 3)
        frame = cv2.addWeighted(frame, 1.3, blur, -0.3, 0)

        fh, fw = frame.shape[:2]
        display = frame.copy()

        # MediaPipe detection
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        frame_labels = []

        if results.multi_hand_landmarks:
            # Deduplicate: only keep the "best" Left hand and "best" Right hand
            best_hands = {} # label -> (hand_lms, raw_feats, handedness_score)
            
            for idx, hand_lms in enumerate(results.multi_hand_landmarks):
                # Identify hand (Left/Right)
                hand_label = "Unknown"
                score = 0
                if results.multi_handedness:
                    cls = results.multi_handedness[idx].classification[0]
                    hand_label = cls.label
                    score = cls.score
                
                raw_feats = self._extract_features(hand_lms)
                
                # If we haven't seen this hand label, or this one is better:
                if hand_label not in best_hands or score > best_hands[hand_label][2]:
                    best_hands[hand_label] = (hand_lms, raw_feats, score)

            # Process only the best unique hands
            for hand_label, (hand_lms, raw_feats, _) in best_hands.items():
                # Smooth using persistent hand label
                feats = self._smooth_landmarks(hand_label, raw_feats)

                # Determine Mask
                active_mask = None
                if self.mode == "ALPHABET": active_mask = self._alphabet_mask
                elif self.mode == "WORDS": active_mask = self._words_mask
                
                # Inference
                probs = self.engine.predict(feats, mask=active_mask)
                top_indices = np.argsort(probs)[-3:][::-1]
                top_3 = [(LABELS[i], float(probs[i])) for i in top_indices if i < len(LABELS)]
                
                top_idx = top_indices[0]
                conf = float(probs[top_idx])
                label = LABELS[top_idx]

                # Mode filter (already masked, but good for stability) + confidence gate
                if conf > self.confidence_threshold:
                    frame_labels.append(label)
                    self._draw_hand_hud(display, hand_lms, label, conf, top_3, fw, fh)
                else:
                    self._draw_hand_hud(display, hand_lms, "Low Conf", conf, top_3, fw, fh)

        # Temporal stability
        if frame_labels:
            dominant = Counter(frame_labels).most_common(1)[0][0]
            self.history.append(dominant)
        else:
            # Decay — push None so stale predictions don't persist forever
            self.history.append(None)

        # Stable sign detection: require agreement in history (buffer of 10)
        non_none = [h for h in self.history if h is not None]
        if len(non_none) >= 6:
            counts = Counter(non_none)
            stable_label, freq = counts.most_common(1)[0]
            agreement = freq / len(self.history)

            # Consensus: require 60% agreement for rock-solid stability
            if agreement >= 0.6:
                now = time.time()
                # Allow repeat signs after cooldown
                last_in_sentence = self.sentence[-1] if self.sentence else None
                if now - self.last_spoken_time > self._speak_cooldown:
                    if stable_label != last_in_sentence:
                        self.sentence.append(stable_label)
                        self.speak(stable_label)
                        self.last_spoken_time = now
                        self.history.clear()
                        # Keep sentence manageable
                        if len(self.sentence) > 10:
                            self.sentence = self.sentence[-8:]

        # Render HUD
        self._draw_header(display, fw, fh)
        self._draw_footer(display, fw, fh)

        return display

    def _on_mouse(self, event, mx, my, flags, param):
        """Handle mouse clicks on the mode button."""
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        fw = param.get('fw', 640)
        bw, bh, margin = self._btn_w, self._btn_h, self._btn_margin
        bx1 = fw - bw - margin
        by1 = margin
        bx2 = fw - margin
        by2 = margin + bh
        if bx1 <= mx <= bx2 and by1 <= my <= by2:
            self.toggle_mode()

    def run(self):
        """Main capture loop with threaded camera for zero latency."""
        cam = CameraStream(0)
        if not cam.is_opened:
            print("[Error] Cannot access camera.")
            return

        win_name = "ASL DETECTION"
        cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)

        # Read one frame to get dimensions for mouse callback
        ret, first = cam.read()
        fw = first.shape[1] if ret and first is not None else 640
        cv2.setMouseCallback(win_name, self._on_mouse, {'fw': fw})

        print("[System] Camera active. Press Q to quit, M to toggle mode, C to clear.")

        while True:
            ret, frame = cam.read()
            if not ret or frame is None:
                continue

            frame = cv2.flip(frame, 1)
            display = self.process_frame(frame)

            cv2.imshow(win_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('m'):
                self.toggle_mode()
            elif key == ord('c'):
                self.sentence.clear()
                self.history.clear()
                print("[System] Sentence cleared.")

        cam.release()
        cv2.destroyAllWindows()
        self.hands.close()
        print("[System] Interpreter closed.")


if __name__ == "__main__":
    interpreter = ASLInterpreter()
    interpreter.run()
