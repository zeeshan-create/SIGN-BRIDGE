import sys
import os

print(f"Current working directory: {os.getcwd()}")
src_path = os.path.join(os.getcwd(), 'src')
sys.path.append(src_path)
print(f"Added to path: {src_path}")

try:
    import real_time_detection
    print(f"Successfully imported real_time_detection from {real_time_detection.__file__}")
    from real_time_detection import ASLInterpreter
    print("Successfully imported ASLInterpreter")
except Exception as e:
    print(f"Failed to import: {e}")
    import traceback
    traceback.print_exc()
