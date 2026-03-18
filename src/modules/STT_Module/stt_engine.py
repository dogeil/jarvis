import queue
import json
from vosk import Model, KaldiRecognizer

class STTEngine:
    def __init__(self, model_path):
        self.model = Model(model_path)
        self.rec = KaldiRecognizer(self.model, 16000)
        self.audio_queue = queue.Queue()

    def _callback(self, indata, frames, time, status):
        """Captures audio chunks and puts them in the queue."""
        self.audio_queue.put(bytes(indata))

    def listen(self):
        """Generator that yields text when speech is recognized."""
        # Import sounddevice lazily so this module can be imported in environments
        # that don't have PortAudio installed (e.g., GitHub Actions runners).
        try:
            import sounddevice as sd  # type: ignore
        except OSError as e:
            raise RuntimeError(
                "Audio backend unavailable (PortAudio missing). "
                "Install PortAudio (system) to use STT listening."
            ) from e

        with sd.RawInputStream(
            samplerate=16000,
            blocksize=8000,
            dtype="int16",
            channels=1,
            callback=self._callback,
        ):
            print("Listening...")
            while True:
                data = self.audio_queue.get()
                if self.rec.AcceptWaveform(data):
                    result = json.loads(self.rec.Result())
                    yield result.get("text", "")