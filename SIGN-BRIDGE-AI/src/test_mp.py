import mediapipe as mp
print(f"Mediapipe version: {mp.__version__}")
try:
    print(f"Solutions via mp.solutions: {mp.solutions}")
except AttributeError as e:
    print(f"Failed mp.solutions: {e}")

try:
    from mediapipe.python.solutions import hands
    print(f"Hands via from mediapipe.python.solutions import hands: {hands}")
except ImportError as e:
    print(f"Failed from mediapipe.python.solutions: {e}")

try:
    import mediapipe.solutions.hands as hands2
    print(f"Hands via import mediapipe.solutions.hands: {hands2}")
except ImportError as e:
    print(f"Failed import mediapipe.solutions.hands: {e}")
