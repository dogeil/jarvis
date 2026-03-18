import queue
import json
import sounddevice as sd
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
        with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                               channels=1, callback=self._callback):
            print("Listening...")
            while True:
                data = self.audio_queue.get()
                if self.rec.AcceptWaveform(data):
                    result = json.loads(self.rec.Result())
                    yield result.get("text", "")