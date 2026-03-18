import sys
from typing import Optional

import pyttsx3


class TTSEngine:
    def __init__(self, rate: int = 180, volume: float = 1.0, driver: Optional[str] = None):
        """
        Initialize a cross-platform TTS engine.

        - On Windows, default to 'sapi5' (if available).
        - On other platforms, let pyttsx3 pick the default driver.
        """
        if driver is None and sys.platform.startswith("win"):
            driver = "sapi5"

        try:
            self.engine = pyttsx3.init(driver)
        except Exception as e:
            # In CI or unsupported environments, fall back to a no-op engine.
            print(f"[TTS Warning] pyttsx3 initialization failed ({e}). Falling back to silent mode.")
            self.engine = None
            return

        self.engine.setProperty("rate", rate)
        self.engine.setProperty("volume", volume)

        voices = self.engine.getProperty("voices")
        if voices and len(voices) > 2:
            self.engine.setProperty("voice", voices[2].id)
        elif voices:
            self.engine.setProperty("voice", voices[0].id)
        else:
            print("[TTS Warning] No voices found on this system.")

    def speak(self, text: str) -> None:
        """
        Converts text to audible speech.
        In environments where the TTS engine could not be initialized,
        this degrades to a simple console print.
        """
        print(f"[JARVIS]: {text}")
        if not getattr(self, "engine", None):
            return
        self.engine.say(text)
        self.engine.runAndWait()