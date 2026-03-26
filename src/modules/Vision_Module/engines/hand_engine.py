import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
from typing import List, Set


class HandEngine:
    """
    Hand vision engine.

    Responsibilities:
    - Runs MediaPipe hand detection / gesture recognition on the provided frame.
    - Draws hand skeleton + label overlay on the frame.
    - Returns the set of custom labels detected for the VisionModule to aggregate.

    Note: Action routing happens in VisionModule (not here).
    """

    def __init__(self, model_dir: str):
        # Original Skeleton Connections
        self.PALM = ((0, 1), (0, 5), (9, 13), (13, 17), (5, 9), (0, 17))
        self.THUMB = ((1, 2), (2, 3), (3, 4))
        self.INDEX = ((5, 6), (6, 7), (7, 8))
        self.MIDDLE = ((9, 10), (10, 11), (11, 12))
        self.RING = ((13, 14), (14, 15), (15, 16))
        self.PINKY = ((17, 18), (18, 19), (19, 20))
        self.CONNECTIONS = self.PALM + self.THUMB + self.INDEX + self.MIDDLE + self.RING + self.PINKY

        # MediaPipe Setup
        hand_model = os.path.join(model_dir, "hand_landmarker.task")
        gesture_model = os.path.join(model_dir, "gesture_recognizer.task")

        self.detector = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=hand_model),
                num_hands=2,
                running_mode=vision.RunningMode.IMAGE,
            )
        )
        self.recognizer = vision.GestureRecognizer.create_from_options(
            vision.GestureRecognizerOptions(
                base_options=python.BaseOptions(model_asset_path=gesture_model),
                num_hands=2,
                running_mode=vision.RunningMode.IMAGE,
            )
        )

    # --- Geometry helpers ---
    def _dist(self, lm, a: int, b: int) -> float:
        dx, dy = lm[a].x - lm[b].x, lm[a].y - lm[b].y
        return (dx * dx + dy * dy) ** 0.5

    def _finger_up(self, lm, tip: int, pip: int) -> bool:
        return lm[tip].y < lm[pip].y - 0.02

    def _thumb_closed(self, lm) -> bool:
        palm_size = self._dist(lm, 0, 9) + 1e-6
        min_thumb_to_palm = min(self._dist(lm, 4, i) for i in (0, 5, 9, 13, 17))
        return (min_thumb_to_palm < 0.70 * palm_size) and (self._dist(lm, 4, 2) < 0.80 * palm_size)

    def _count_raised(self, lm) -> int:
        return sum(
            [
                self._finger_up(lm, 8, 6),
                self._finger_up(lm, 12, 10),
                self._finger_up(lm, 16, 14),
                self._finger_up(lm, 20, 18),
            ]
        )

    def get_custom_labels(self, lm, is_open: bool, is_thumb: bool) -> List[str]:
        raised = self._count_raised(lm)
        labels: List[str] = []
        if (not is_open) and (raised == 4) and self._thumb_closed(lm):
            labels.append("FOUR_FINGERS")
        if raised == 3:
            labels.append("THREE_FINGERS")
        elif raised == 2:
            labels.append("TWO_FINGERS")
        elif (raised == 1) and (not is_thumb):
            # Distinguish "middle finger only" from generic "one finger"
            middle_up = self._finger_up(lm, 12, 10)
            if middle_up:
                labels.append("MIDDLE_FINGER_UP")
            else:
                labels.append("ONE_FINGER")
        return labels

    def custom_labels_four_to_one(self, lm, is_open_palm: bool, is_thumb_up: bool) -> List[str]:
        # Adapter helper to keep compatibility with older naming.
        return self.get_custom_labels(lm, is_open=is_open_palm, is_thumb=is_thumb_up)

    def process(self, frame_bgr, mp_image) -> Set[str]:
        """
        Process one frame.

        Returns:
            Set[str]: custom labels detected in this frame.
        """
        h, w, _ = frame_bgr.shape
        custom_labels_detected: Set[str] = set()

        detection_result = self.detector.detect(mp_image)
        recognition_result = self.recognizer.recognize(mp_image)

        # Draw per-hand skeleton + compute labels for that hand.
        for hand_idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
            top_canned_name = ""
            if recognition_result.gestures and hand_idx < len(recognition_result.gestures):
                if recognition_result.gestures[hand_idx]:
                    top_canned_name = getattr(
                        recognition_result.gestures[hand_idx][0], "category_name", ""
                    )

            if top_canned_name == "Open_Palm":
                custom_labels_detected.add("FIVE_FINGERS")

            for label in self.custom_labels_four_to_one(
                hand_landmarks,
                is_open_palm=(top_canned_name == "Open_Palm"),
                is_thumb_up=(top_canned_name == "Thumb_Up"),
            ):
                custom_labels_detected.add(label)

            # Draw Skeleton Connections
            points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
            for s, e in self.CONNECTIONS:
                cv2.line(frame_bgr, points[s], points[e], (0, 255, 0), 2)
            for pt in points:
                cv2.circle(frame_bgr, pt, 4, (0, 0, 255), -1)

        # UI Overlay (hand-specific)
        canned_name = "None"
        if recognition_result.gestures and len(recognition_result.gestures) > 0:
            canned_name = getattr(recognition_result.gestures[0][0], "category_name", "None")
        cv2.putText(
            frame_bgr,
            f"Canned: {canned_name}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 255, 0),
            3,
        )

        y_offset = 80
        for label in [
            "FIVE_FINGERS",
            "FOUR_FINGERS",
            "THREE_FINGERS",
            "TWO_FINGERS",
            "ONE_FINGER",
            "MIDDLE_FINGER_UP",
        ]:
            if label in custom_labels_detected:
                cv2.putText(
                    frame_bgr,
                    f"Custom: {label}",
                    (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 255),
                    2,
                )
                y_offset += 40

        return custom_labels_detected

