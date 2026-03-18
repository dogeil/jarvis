import multiprocessing
import uvicorn
import time
import os
import sys

# Ensure the 'src' directory is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from modules.Hand_Module.hand_engine import HandEngine
from modules.STT_Module.stt_engine import STTEngine
from modules.TTS_Module.tts_engine import TTSEngine

def run_hand_module():
    print("[Launcher] Starting Hand Module...")
    root_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(root_dir, "models") 
    try:
        hand_module = HandEngine(model_dir=model_path)
        hand_module.start()
    except Exception as e:
        print(f"[Hand Module Error] {e}")

def run_stt_module():
    print("[Launcher] Starting STT Module...")
    root_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(root_dir, "models", "vosk-model") 
    try:
        stt_engine = STTEngine(model_path=model_path)
        for text in stt_engine.listen():
            if text:
                print(f"[STT User Said]: {text}")
    except Exception as e:
        print(f"[STT Error] {e}")

def run_server():
    print("[Launcher] Starting FastAPI Brain...")
    # FastAPI app is defined in src/main.py
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)

def say_greeting():
    """Simple startup greeting"""
    try:
        tts = TTSEngine()
        tts.speak("All systems are online. JARVIS is ready.")
    except Exception as e:
        print(f"[TTS Greeting Error] {e}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    print("--- INITIALIZING JARVIS CORE ---")

    # Define the processes
    hand_process = multiprocessing.Process(target=run_hand_module, name="Jarvis-Hand")
    stt_process = multiprocessing.Process(target=run_stt_module, name="Jarvis-STT")
    server_process = multiprocessing.Process(target=run_server, name="Jarvis-API")

    # Start all systems
    hand_process.start()
    stt_process.start()
    server_process.start()

    # Give them a second to initialize before speaking
    time.sleep(2)
    say_greeting()

    try:
        while True:
            time.sleep(1)
            if not all([hand_process.is_alive(), stt_process.is_alive(), server_process.is_alive()]):
                print("[Launcher] A critical process died. Shutting down...")
                break
    except KeyboardInterrupt:
        print("[Launcher] Manual shutdown initiated.")
    finally:
        hand_process.terminate()
        stt_process.terminate()
        server_process.terminate()
        print("[Launcher] All systems offline.")