import sys
import os

# Add 'src' to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from modules.TTS_Module.tts_engine import TTSEngine

def test_speech():
    # Check if a message was passed as an argument
    if len(sys.argv) > 1:
        # Join all arguments into a single string (e.g., "Hello World")
        message = " ".join(sys.argv[1:])
    else:
        # Default message if no argument is provided
        message = "No text provided. System check complete."

    jarvis_voice = TTSEngine()
    jarvis_voice.speak(message)

if __name__ == "__main__":
    test_speech()