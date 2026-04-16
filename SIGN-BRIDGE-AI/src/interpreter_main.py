import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np
import pyttsx3
import speech_recognition as sr
import os
import threading
from collections import deque, Counter
import time

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(os.path.dirname(BASE_DIR), "models")
TFLITE_PATH = os.path.join(MODELS_DIR, "landmark_model.tflite")
LANDMARK_MODEL_PATH = os.path.join(MODELS_DIR, "landmark_model.keras")
LABEL_MAP_PATH = os.path.join(MODELS_DIR, "label_map.json")

# Intelligent Label Loading
try:
    import json

    with open(LABEL_MAP_PATH, "r") as f:
        LABELS = json.load(f)["labels"]
    print(f"[Sync] Loaded {len(LABELS)} signs from synchronized map.")
except:
    print("[Warning] label_map.json missing. Using defaults.")
    LABELS = list("ABCDEFGHIKLMNOPQRSUVWYZ") + [
        "hello",
        "goodbye",
        "thanks",
        "yes",
        "no",
        "please",
        "I_Love_You",
        "Sorry",
        "Maybe",
    ]


SIGN_DB_PATH = "data/signs_database"

# --- UI COLORS (Premium Aesthetic) ---
COLOR_BG = (20, 20, 20)
COLOR_ACCENT = (147, 51, 234)  # Purple (BGR: 234, 51, 147)
COLOR_TEXT = (255, 255, 255)
COLOR_SUCCESS = (0, 255, 127)  # Emerald Green


class ASLInterpreter:
    def __init__(self):
        # 1. Initialize Hand Tracking
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8,
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # 2. Load Landmark-based TFLite Model
        self._use_tflite = False
        self._interpreter = None
        self._input_details = None
        self._output_details = None
        self._feature_size = 42

        try:
            if os.path.exists(TFLITE_PATH):
                self._interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
                self._interpreter.allocate_tensors()
                self._input_details = self._interpreter.get_input_details()
                self._output_details = self._interpreter.get_output_details()
                self._use_tflite = True
                self._feature_size = self._input_details[0]["shape"][-1]
                print(
                    f"[Core] TFLite model loaded (input shape: {self._input_details[0]['shape']})"
                )
            elif os.path.exists(LANDMARK_MODEL_PATH):
                self.model = tf.keras.models.load_model(LANDMARK_MODEL_PATH)
                self._feature_size = self.model.input_shape[-1]
                print(
                    f"[Core] Keras model loaded (input shape: {self.model.input_shape})"
                )
            else:
                print("[Error] No model found in models/ directory")
                self.model = None
        except Exception as e:
            print(f"[Error] Failed to load model: {e}")
            self.model = None
            self._interpreter = None

        # 3. Initialize Speech & TTS
        self.engine = pyttsx3.init()
        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone()

        # 4. State Management
        self.prediction_history = deque(maxlen=10)
        self.last_sign = ""
        self.spoken_text = ""
        self.caption_text = "Waiting for input..."
        self.is_listening = True
        self.last_vocal_time = 0
        self.vocation_cooldown = 2.0

        # 5. Start Background Threads
        self.voice_thread = threading.Thread(
            target=self._listen_for_speech, daemon=True
        )
        self.voice_thread.start()

    def _listen_for_speech(self):
        """Background thread to transcribe speech (Speech -> Sign/Text)."""
        print("[Interpreter] Speech listener active.")
        while self.is_listening:
            try:
                with self.mic as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    audio = self.recognizer.listen(
                        source, timeout=2, phrase_time_limit=5
                    )

                    text = self.recognizer.recognize_google(audio)
                    if text:
                        print(f"[Speech] Heard: {text}")
                        # Display Text at Bottom
                        self.caption_text = f"Spoken: {text}"

                        # Trigger Voice to Sign (if matching keyword)
                        for word in LABELS:
                            if word.lower() in text.lower():
                                self.caption_text = f"Sign Translated: [{word.upper()}]"
                                # Note: In a professional app, we'd trigger a 3D animation here.
                                # For now, we update the caption prominently.
            except (sr.WaitTimeoutError, sr.UnknownValueError):
                pass
            except Exception as e:
                print(f"[Speech Error] {e}")

    def speak(self, text):
        """Vocalize the detected sign."""

        def _speak_thread():
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except:
                pass

        threading.Thread(target=_speak_thread, daemon=True).start()

    def draw_overlays(self, frame, hand_found=False):
        """Draw a professional UI over the frame."""
        h, w, _ = frame.shape

        # Overlay: Header (Sign -> Text)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 60), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        cv2.putText(
            frame,
            "SIGN -> SPEECH TRANSLATION",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            COLOR_SUCCESS,
            2,
        )

        if self.last_sign:
            cv2.putText(
                frame,
                f"SIGN DETECTED: {self.last_sign.upper()}",
                (w // 2, 35),
                cv2.FONT_HERSHEY_DUPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

        # Overlay: Footer (Speech -> Sign)
        overlay_foot = frame.copy()
        cv2.rectangle(overlay_foot, (0, h - 70), (w, h), (40, 40, 40), -1)
        cv2.addWeighted(overlay_foot, 0.6, frame, 0.4, 0, frame)

        # Captioning
        cv2.putText(
            frame,
            "SPEECH CAPTIONS",
            (20, h - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )
        cv2.putText(
            frame,
            self.caption_text,
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
        )

        return frame

    def run(self):
        cap = cv2.VideoCapture(0)
        print("[Interpreter] Application running. Press 'Q' to quit.")

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            h, w, c = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 1. Processing Hand Sign
            results = self.hands.process(rgb_frame)
            hand_found = False

            if results.multi_hand_landmarks:
                hand_found = True
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )

                    # Extract landmark features (wrist-relative)
                    bx = hand_landmarks.landmark[0].x
                    by = hand_landmarks.landmark[0].y
                    bz = hand_landmarks.landmark[0].z

                    feats = []
                    for lm in hand_landmarks.landmark:
                        feats.extend([lm.x - bx, lm.y - by])
                    feats = np.array(feats[: self._feature_size], dtype=np.float32)

                    if feats.shape[0] == self._feature_size:
                        # Run inference
                        try:
                            if self._use_tflite and self._interpreter:
                                input_data = feats.reshape(1, -1).astype(np.float32)
                                self._interpreter.set_tensor(
                                    self._input_details[0]["index"], input_data
                                )
                                self._interpreter.invoke()
                                probs = self._interpreter.get_tensor(
                                    self._output_details[0]["index"]
                                )[0]
                            elif self.model:
                                probs = self.model.predict(
                                    feats.reshape(1, -1), verbose=0
                                )[0]
                            else:
                                continue

                            top_idx = np.argmax(probs)
                            confidence = probs[top_idx]

                            if confidence > 0.7:
                                label = LABELS[top_idx]
                                self.prediction_history.append(label)

                                # Smoothing
                                if len(self.prediction_history) == 10:
                                    counts = Counter(self.prediction_history)
                                    smooth_sign, freq = counts.most_common(1)[0]

                                    if freq > 7:
                                        display_label = smooth_sign.replace("_", " ")

                                        # Trigger Vocalization
                                        current_time = time.time()
                                        if (
                                            smooth_sign != self.last_sign
                                            or current_time - self.last_vocal_time
                                            > self.vocation_cooldown
                                        ):
                                            self.speak(display_label)
                                            self.last_vocal_time = current_time
                                            self.last_sign = smooth_sign
                        except Exception as e:
                            pass

            # 2. Render UI
            frame = self.draw_overlays(frame, hand_found)

            cv2.imshow("Antigravity Real-Time ASL Interpreter", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.is_listening = False
                break

        cap.release()
        cv2.destroyAllWindows()
        print("[Interpreter] Application closed.")


if __name__ == "__main__":
    interpreter = ASLInterpreter()
    interpreter.run()
