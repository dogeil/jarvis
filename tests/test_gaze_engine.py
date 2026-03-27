import unittest
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from modules.Vision_Module.engines.gaze_engine import GazeEngine


class _DummyFaceEngine:
    _last_primary_bbox = None


class TestGazeEngine(unittest.TestCase):
    def test_engine_initialization(self):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        model_path = os.path.join(project_root, "models", "face", "face_landmarker.task")
        engine = GazeEngine(model_path=model_path, face_engine=_DummyFaceEngine(), gaze_every_n_frames=5)
        self.assertIsNotNone(engine)


if __name__ == "__main__":
    unittest.main()

