import multiprocessing
import uvicorn
import time
import os
import sys

# Ensure the 'src' directory is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from modules.Hand_Module.hand_engine import HandEngine
from modules.STT_Module.stt_engine import STTEngine

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
    # Ensure this matches your Vosk folder name
    model_path = os.path.join(root_dir, "models", "vosk-model") 

    if not os.path.exists(model_path):
        print(f"[STT Error] Model not found at {model_path}")
        return

    try:
        stt_engine = STTEngine(model_path=model_path)
        print("[STT] Listening for commands...")
        for text in stt_engine.listen():
            if text:
                print(f"[STT User Said]: {text}")
                # Future: Send this 'text' to your FastAPI or a Command Handler
    except Exception as e:
        print(f"[STT Error] {e}")

def run_server():
    print("[Launcher] Starting FastAPI Brain...")
    # FastAPI app is defined in src/main.py
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)

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

    try:
        while True:
            time.sleep(1)
            # Monitor if any core process has died
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