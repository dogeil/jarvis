from modules.TTS_Module.tts_engine import TTSEngine

def test_speech():
    # Keep CI non-interactive: ensure we can construct the engine
    # without raising on missing platform-specific backends.
    engine = TTSEngine()
    assert engine is not None