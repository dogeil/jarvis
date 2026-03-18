import pyttsx3

class TTSEngine:
    def __init__(self, rate=180, volume=1.0):
        # Initialize with SAPI5 for Windows stability
        self.engine = pyttsx3.init('sapi5')
        self.engine.setProperty('rate', rate)
        self.engine.setProperty('volume', volume)
        
        # Get all installed voices
        voices = self.engine.getProperty('voices')
        
        # Directly use the first voice in the list as requested
        if voices:
            self.engine.setProperty('voice', voices[2].id)
        else:
            print("[TTS Warning] No voices found on this system.")

    def speak(self, text):
        """Converts text to audible speech."""
        print(f"[JARVIS]: {text}")
        self.engine.say(text)
        self.engine.runAndWait()