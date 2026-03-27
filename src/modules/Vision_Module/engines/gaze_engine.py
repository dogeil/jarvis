import os
import time
from typing import Optional, Set, Tuple

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class GazeEngine:
    """
    Primary-face eye-gaze estimation using MediaPipe FaceLandmarker iris landmarks.

    - Uses the primary face bbox produced by FaceEngine (passed in at init).
    - Runs every N frames (configurable) and caches last gaze direction in between.
    - Emits labels: GAZE_LEFT/RIGHT/UP/DOWN/CENTER and GAZE_ACTIVE.
    """

    # FaceMesh landmark indices (MediaPipe)
    _LEFT_EYE_OUTER = 33
    _LEFT_EYE_INNER = 133
    _LEFT_EYE_TOP = 159
    _LEFT_EYE_BOTTOM = 145
    _RIGHT_EYE_OUTER = 263
    _RIGHT_EYE_INNER = 362
    _RIGHT_EYE_TOP = 386
    _RIGHT_EYE_BOTTOM = 374

    # Iris landmark indices (requires model with iris)
    _LEFT_IRIS = [468, 469, 470, 471, 472]
    _RIGHT_IRIS = [473, 474, 475, 476, 477]

    def __init__(
        self,
        model_path: str,
        face_engine,
        gaze_every_n_frames: int = 5,
        grace_window_sec: float = 0.5,
        crop_margin: float = 0.35,
        ema_alpha: float = 0.35,
        switch_hold_frames: int = 2,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing FaceLandmarker model: {model_path}")

        self._face_engine = face_engine
        self._gaze_every_n_frames = max(1, int(gaze_every_n_frames))
        self._grace_window_sec = float(grace_window_sec)
        self._crop_margin = float(crop_margin)
        self._ema_alpha = float(ema_alpha)
        self._switch_hold_frames = max(1, int(switch_hold_frames))

        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

        self._frame_idx = 0
        self._last_seen_ts: Optional[float] = None
        self._last_gaze_label: str = "GAZE_CENTER"
        self._last_gaze_xy: Optional[Tuple[float, float]] = None
        self._pending_gaze_label: Optional[str] = None
        self._pending_count: int = 0
        self._last_left_iris_abs: Optional[Tuple[int, int]] = None
        self._last_right_iris_abs: Optional[Tuple[int, int]] = None

    def set_gaze_every_n_frames(self, n: int) -> None:
        self._gaze_every_n_frames = max(1, int(n))

    def _crop_from_bbox(self, frame_bgr, bbox_xywh: Tuple[int, int, int, int]):
        h, w, _ = frame_bgr.shape
        x, y, bw, bh = bbox_xywh
        cx = x + bw / 2.0
        cy = y + bh / 2.0
        bw2 = bw * (1.0 + self._crop_margin)
        bh2 = bh * (1.0 + self._crop_margin)
        x1 = int(max(0, cx - bw2 / 2.0))
        y1 = int(max(0, cy - bh2 / 2.0))
        x2 = int(min(w, cx + bw2 / 2.0))
        y2 = int(min(h, cy + bh2 / 2.0))
        if x2 <= x1 or y2 <= y1:
            return None
        return frame_bgr[y1:y2, x1:x2], (x1, y1, x2, y2)

    def _lm_xy(self, lm, idx: int, width: int, height: int) -> Tuple[float, float]:
        p = lm[idx]
        return float(p.x) * width, float(p.y) * height

    def _iris_center_xy(self, lm, indices, width: int, height: int) -> Tuple[float, float]:
        xs, ys = [], []
        for idx in indices:
            x, y = self._lm_xy(lm, idx, width, height)
            xs.append(x)
            ys.append(y)
        return float(np.mean(xs)), float(np.mean(ys))

    def _eye_norm_xy(self, lm, left_idx, right_idx, top_idx, bottom_idx, iris_indices, width, height):
        lx, ly = self._lm_xy(lm, left_idx, width, height)
        rx, ry = self._lm_xy(lm, right_idx, width, height)
        tx, ty = self._lm_xy(lm, top_idx, width, height)
        bx, by = self._lm_xy(lm, bottom_idx, width, height)
        ix, iy = self._iris_center_xy(lm, iris_indices, width, height)

        denom_x = max(1e-6, (rx - lx))
        denom_y = max(1e-6, (by - ty))
        nx = (ix - lx) / denom_x
        ny = (iy - ty) / denom_y
        return float(nx), float(ny), (ix, iy), (lx, ly, rx, ry, tx, ty, bx, by)

    def _classify(self, nx: float, ny: float) -> str:
        # Add a center deadzone to reduce jitter around boundaries.
        if 0.46 <= nx <= 0.54 and 0.46 <= ny <= 0.54:
            return "GAZE_CENTER"
        if nx < 0.42:
            return "GAZE_LEFT"
        if nx > 0.58:
            return "GAZE_RIGHT"
        if ny < 0.42:
            return "GAZE_UP"
        if ny > 0.58:
            return "GAZE_DOWN"
        return "GAZE_CENTER"

    def process(self, frame_bgr, mp_image=None) -> Set[str]:
        self._frame_idx += 1
        now = time.time()

        # Pull primary face bbox from FaceEngine
        bbox = getattr(self._face_engine, "_last_primary_bbox", None)
        labels: Set[str] = set()

        if bbox is None:
            # allow a short grace window to keep last label stable
            if self._last_seen_ts is not None and (now - self._last_seen_ts) <= self._grace_window_sec:
                labels.add("GAZE_ACTIVE")
                labels.add(self._last_gaze_label)
            return labels

        # Only run the landmarker every N frames; reuse cached result in between.
        should_run = (self._frame_idx % self._gaze_every_n_frames) == 0 or self._last_seen_ts is None
        crop = self._crop_from_bbox(frame_bgr, bbox)
        if crop is None:
            return labels
        crop_bgr, (x1, y1, x2, y2) = crop

        if should_run:
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)

            result = self._landmarker.detect(mp_img)
            if not result.face_landmarks:
                return labels

            lm = result.face_landmarks[0]
            ch, cw, _ = crop_bgr.shape

            lnx, lny, (lix, liy), _ = self._eye_norm_xy(
                lm,
                self._LEFT_EYE_OUTER,
                self._LEFT_EYE_INNER,
                self._LEFT_EYE_TOP,
                self._LEFT_EYE_BOTTOM,
                self._LEFT_IRIS,
                cw,
                ch,
            )
            rnx, rny, (rix, riy), _ = self._eye_norm_xy(
                lm,
                self._RIGHT_EYE_INNER,
                self._RIGHT_EYE_OUTER,
                self._RIGHT_EYE_TOP,
                self._RIGHT_EYE_BOTTOM,
                self._RIGHT_IRIS,
                cw,
                ch,
            )

            nx = (lnx + rnx) / 2.0
            ny = (lny + rny) / 2.0
            if self._last_gaze_xy is None:
                smx, smy = nx, ny
            else:
                px, py = self._last_gaze_xy
                smx = (self._ema_alpha * nx) + ((1.0 - self._ema_alpha) * px)
                smy = (self._ema_alpha * ny) + ((1.0 - self._ema_alpha) * py)
            self._last_gaze_xy = (smx, smy)

            candidate = self._classify(smx, smy)
            if candidate != self._last_gaze_label:
                if candidate == self._pending_gaze_label:
                    self._pending_count += 1
                else:
                    self._pending_gaze_label = candidate
                    self._pending_count = 1
                if self._pending_count >= self._switch_hold_frames:
                    self._last_gaze_label = candidate
                    self._pending_gaze_label = None
                    self._pending_count = 0
            else:
                self._pending_gaze_label = None
                self._pending_count = 0

            self._last_seen_ts = now

            # Cache iris points so overlay remains stable between inference frames.
            self._last_left_iris_abs = (int(x1 + lix), int(y1 + liy))
            self._last_right_iris_abs = (int(x1 + rix), int(y1 + riy))

        # Overlay direction (cached)
        if self._last_left_iris_abs is not None:
            cv2.circle(frame_bgr, self._last_left_iris_abs, 2, (0, 255, 255), -1)
        if self._last_right_iris_abs is not None:
            cv2.circle(frame_bgr, self._last_right_iris_abs, 2, (0, 255, 255), -1)

        cv2.putText(
            frame_bgr,
            self._last_gaze_label,
            (max(10, x1), max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

        labels.add("GAZE_ACTIVE")
        labels.add(self._last_gaze_label)
        return labels

