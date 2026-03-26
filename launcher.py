import multiprocessing
import threading
import uvicorn
import time
import os
import sys
import cv2

# Ensure the 'src' directory is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from src.modules.STT_Module.stt_engine import STTEngine
from src.modules.TTS_Module.tts_engine import TTSEngine
from src.modules.SFX_Module.sfx_engine import SFXEngine
from src.modules.Vision_Module.vision_module import VisionModule
from src.modules.Vision_Module.engines.hand_engine import HandEngine


def _is_camera_available(index: int = 0) -> bool:
    """Best-effort camera probe used at launcher startup."""
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    try:
        if not cap.isOpened():
            return False
        ret, frame = cap.read()
        return bool(ret and frame is not None)
    finally:
        cap.release()


def _ask_start_without_vision() -> bool:
    """
    Ask the user whether to continue without the vision module.
    Returns True to continue, False to close launcher.
    """
    print("\n[Warning] No camera/video input device detected.")
    print("[Warning] Vision module cannot start.")
    print("[Warning] Continue running Jarvis without vision? (y/n)")
    while True:
        try:
            answer = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("[Launcher] Please answer with 'y' or 'n'.")


def run_vision_module(
    hand_command_queue: "multiprocessing.Queue",
    sfx_command_queue: "multiprocessing.Queue",
):
    print("[Launcher] Starting Vision Module...")
    root_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(root_dir, "models") 
    try:
        hand_engine = HandEngine(model_dir=model_path)
        vision_module = VisionModule(engines=[hand_engine], sfx_command_queue=sfx_command_queue)
        vision_module.start(hand_command_queue)
    except Exception as e:
        print(f"[Vision Module Error] {e}")

def run_stt_module():
    print("[Launcher] Starting STT Module (voice)...")
    root_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(root_dir, "models", "vosk-model") 

    def voice_loop() -> None:
        try:
            stt_engine = STTEngine(model_path=model_path)
            for text in stt_engine.listen():
                if text:
                    print(f"[STT User Said]: {text}")
        except Exception as e:
            print(f"[STT Error] {e}")

    voice_loop()


def run_server():
    print("[Launcher] Starting FastAPI Brain...")
    # FastAPI app is defined in src/main.py
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)

def run_sfx_module(sfx_command_queue: "multiprocessing.Queue"):
    print("[Launcher] Starting SFX Module (sound effects)...")
    root_dir = os.path.dirname(os.path.abspath(__file__))
    sounds_dir = os.path.join(root_dir, "sounds")
    engine = SFXEngine(sounds_dir=sounds_dir)

    print(f"[SFX] Using sounds directory: {sounds_dir}")
    while True:
        cmd = sfx_command_queue.get()
        if not cmd:
            continue

        action = cmd[0]
        if action == "play" and len(cmd) >= 2:
            engine.play(cmd[1])
        elif action == "stop":
            engine.stop()
        elif action in {"quit", "exit"}:
            break

def say_greeting():
    """Simple startup greeting"""
    try:
        tts = TTSEngine()
        tts.speak("All systems are online. JARVIS is ready.")
    except Exception as e:
        print(f"[TTS Greeting Error] {e}")


def console_loop(hand_command_queue: "multiprocessing.Queue", sfx_command_queue: "multiprocessing.Queue"):
    """Read commands from the main console."""
    print("[Console] Type commands here. Use /quit to stop Jarvis.")
    print("[Console] Resolution: /res <width> <height>, /default")
    print("[Console] Sound FX: /sfx <file.wav> (absolute path or file under ./sounds)")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Console] Input closed.")
            break

        if not line:
            continue
        normalized = line.strip()
        lower = normalized.lower()
        if lower in {"/quit", "/exit"}:
            print("[Console] Stopping Jarvis by user request.")
            break

        print(f"[Console User Typed]: {line}")

        # Support resolution changes while Jarvis is running.
        # Examples: `/res 640 480`, `res 640 480`, `/default`, `default`
        cmdline = normalized
        if cmdline.startswith("/"):
            cmdline = cmdline[1:].lstrip()
        parts = cmdline.split()
        if not parts:
            continue

        cmd = parts[0].lower()
        if cmd == "res":
            if len(parts) != 3:
                print("[Console] Usage: /res <width> <height>")
                continue
            try:
                w = int(parts[1])
                h = int(parts[2])
            except ValueError:
                print("[Console] Width/height must be integers.")
                continue
            if w <= 0 or h <= 0:
                print("[Console] Width/height must be positive.")
                continue
            hand_command_queue.put(("res", w, h))
            continue

        if cmd in {"default", "reset"}:
            hand_command_queue.put(("default",))
            continue

        if cmd == "sfx":
            if len(parts) != 2:
                print("[Console] Usage: /sfx <file.wav>")
                continue
            sfx_command_queue.put(("play", parts[1]))
            continue

        if cmd in {"sfxstop", "stop-sfx", "sfx_stop"}:
            sfx_command_queue.put(("stop",))
            continue

if __name__ == "__main__":
    multiprocessing.freeze_support()
    print("--- INITIALIZING JARVIS CORE ---")

    vision_enabled = True
    if not _is_camera_available():
        continue_without_vision = _ask_start_without_vision()
        if continue_without_vision:
            vision_enabled = False
        else:
            print("[Launcher] Startup canceled by user (no camera available).")
            sys.exit(0)

    # Define the processes
    hand_command_queue = multiprocessing.Queue()
    sfx_command_queue = multiprocessing.Queue()
    vision_process = None
    if vision_enabled:
        vision_process = multiprocessing.Process(
            target=run_vision_module,
            args=(hand_command_queue, sfx_command_queue),
            name="Jarvis-Vision",
        )
    stt_process = multiprocessing.Process(target=run_stt_module, name="Jarvis-STT")
    server_process = multiprocessing.Process(target=run_server, name="Jarvis-API")
    sfx_process = multiprocessing.Process(
        target=run_sfx_module,
        args=(sfx_command_queue,),
        name="Jarvis-SFX",
    )

    # Start all systems
    if vision_process is not None:
        vision_process.start()
    else:
        print("[Launcher] Vision module disabled. Running in voice/API mode.")
    stt_process.start()
    server_process.start()
    sfx_process.start()

    shutdown_event = threading.Event()

    def _console_worker() -> None:
        try:
            console_loop(hand_command_queue, sfx_command_queue)
        finally:
            shutdown_event.set()

    console_thread = threading.Thread(
        target=_console_worker, name="Jarvis-Console", daemon=True
    )
    console_thread.start()

    # Give them a second to initialize before speaking
    time.sleep(2)
    say_greeting()

    try:
        while not shutdown_event.is_set():
            # If ESC is pressed in the Hand window, Hand module exits; we treat that
            # as a full Jarvis shutdown trigger.
            critical_alive = [stt_process.is_alive(), server_process.is_alive(), sfx_process.is_alive()]
            if vision_process is not None:
                critical_alive.append(vision_process.is_alive())

            if not all(critical_alive):
                print("[Launcher] A critical process died. Shutting down...")
                break
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("[Launcher] Manual shutdown initiated.")
    finally:
        shutdown_event.set()
        if vision_process is not None and vision_process.is_alive():
            vision_process.terminate()
        stt_process.terminate()
        server_process.terminate()
        sfx_process.terminate()
        print("[Launcher] All systems offline.")