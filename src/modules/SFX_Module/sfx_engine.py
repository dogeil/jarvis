import os
from typing import Optional


try:
    import winsound  # Windows built-in audio playback for simple wav files
except Exception:  # pragma: no cover
    winsound = None


class SFXEngine:
    """
    Minimal sound effects engine intended for quick "fire and forget" playback.

    Setup expectation:
    - Put `.wav` files somewhere you can reference (absolute path, or under `sounds_dir`).
    """

    def __init__(self, sounds_dir: str):
        self.sounds_dir = sounds_dir

    def resolve_sound_path(self, sound: str) -> Optional[str]:
        """
        Resolve a sound identifier to an on-disk file path.

        - If `sound` is an absolute path, verify it exists and return it.
        - Otherwise try `<sounds_dir>/<sound>` and `./<sound>`.
        """
        if not sound:
            return None

        if os.path.isabs(sound):
            return sound if os.path.exists(sound) else None

        candidate = os.path.join(self.sounds_dir, sound)
        if os.path.exists(candidate):
            return candidate

        # Convenience: allow passing `boom` instead of `boom.wav`
        if os.path.splitext(sound)[1] == "":
            candidate_wav = candidate + ".wav"
            if os.path.exists(candidate_wav):
                return candidate_wav

        candidate = os.path.abspath(sound)
        if os.path.exists(candidate):
            return candidate

        return None

    def _is_riff_wave(self, path: str) -> bool:
        """
        winsound can reliably play RIFF/WAVE PCM WAV files.
        Many files named `.wav` are actually other encodings (e.g. MP3).
        """
        try:
            with open(path, "rb") as f:
                header = f.read(12)
            if len(header) < 12:
                return False
            riff, _, wave, _ = header[0:4], header[4:8], header[8:12], None
            return riff == b"RIFF" and wave == b"WAVE"
        except Exception:
            return False

    def play(self, sound: str) -> None:
        if winsound is None:
            print("[SFX Warning] winsound unavailable; cannot play audio.")
            return

        path = self.resolve_sound_path(sound)
        if not path:
            print(f"[SFX Error] Sound not found: {sound}")
            return

        if not self._is_riff_wave(path):
            print(
                f"[SFX Error] {path} is not a RIFF/WAVE WAV file "
                f"(winsound only supports WAV). Re-export as PCM WAV."
            )
            return

        # Stop any currently playing sound to reduce overlap.
        try:
            winsound.PlaySound(None, winsound.SND_PURGE)
            winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
        except Exception as e:  # pragma: no cover
            print(f"[SFX Error] Failed to play sound ({e}).")

    def stop(self) -> None:
        if winsound is None:
            return
        try:
            winsound.PlaySound(None, winsound.SND_PURGE)
        except Exception:  # pragma: no cover
            pass

