import sys
import os
# Add 'src' to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from modules.STT_Module.stt_engine import STTEngine

def test_voice():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    model_path = os.path.join(project_root, "models", "vosk-model")
    
    stt = STTEngine(model_path)
    print("Speak into your mic to test...")
    for text in stt.listen():
        print(f"Detected: {text}")
        if text: break # Stop after first successful detection

if __name__ == "__main__":
    test_voice()