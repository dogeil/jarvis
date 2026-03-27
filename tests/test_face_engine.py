import unittest
import os
import sys

# Add 'src' to path (matches existing tests style)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from modules.Vision_Module.engines.face_engine import FaceEngine


class TestFaceEngine(unittest.TestCase):
    def test_engine_initialization(self):
        """Check if FaceEngine can initialize without camera access."""
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        model_dir = os.path.join(project_root, "models", "face")
        engine = FaceEngine(model_dir=model_dir)
        self.assertIsNotNone(engine)


if __name__ == "__main__":
    unittest.main()

