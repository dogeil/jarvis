import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import os
import queue as queue_mod
from typing import List, Optional, Set


from .gesture_logic import GestureProcessor


class VisionModule:
    """
    Vision orchestration module.

    Responsibilities:
    - Owns the OpenCV capture loop and frame timing.
    - Aggregates labels from attached vision engines.
    - Runs gesture/action matching (GestureProcessor) across aggregated labels.
    - Routes actions to other modules (SFX, exit).
    """

    def __init__(self, engines: List[object], sfx_command_queue=None):
        self.engines = engines
        self.sfx_command_queue = sfx_command_queue
        self.processor = GestureProcessor()

    def start(self, command_queue=None):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        default_width = 640
        default_height = 480
        current_width = default_width
        current_height = default_height
        requested_width = default_width
        requested_height = default_height

        # Lowering resolution slightly ensures the CPU can handle API + Vision
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, default_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, default_height)

        prev_time = 0
        try:
            while cap.isOpened():
                # Apply any requested resolution changes from the launcher.
                if command_queue is not None:
                    try:
                        # Drain the queue and keep the most recent request.
                        while True:
                            cmd = command_queue.get_nowait()
                            if not cmd:
                                continue
                            if cmd[0] == "res":
                                requested_width, requested_height = int(cmd[1]), int(cmd[2])
                            elif cmd[0] == "default":
                                requested_width, requested_height = default_width, default_height
                            elif cmd[0] == "face_n":
                                n = int(cmd[1])
                                for engine in self.engines:
                                    setter = getattr(engine, "set_recognize_every_n_frames", None)
                                    if callable(setter):
                                        setter(n)
                                print(f"[Vision] Face recognition cadence set to every {n} frames.")
                            elif cmd[0] == "face_reload":
                                for engine in self.engines:
                                    reload_fn = getattr(engine, "reload_face_models", None)
                                    if callable(reload_fn):
                                        reload_fn()
                                print("[Vision] Face models/gallery reloaded.")
                            elif cmd[0] == "gaze_n":
                                n = int(cmd[1])
                                for engine in self.engines:
                                    setter = getattr(engine, "set_gaze_every_n_frames", None)
                                    if callable(setter):
                                        setter(n)
                                print(f"[Vision] Gaze cadence set to every {n} frames.")
                            elif cmd[0] in {"quit", "exit"}:
                                cap.release()
                                cv2.destroyAllWindows()
                                return
                    except queue_mod.Empty:
                        pass

                if requested_width != current_width or requested_height != current_height:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, requested_width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, requested_height)
                    current_width, current_height = requested_width, requested_height

                ret, frame = cap.read()
                if not ret or frame is None:
                    continue

                # FPS Calculation
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
                prev_time = curr_time

                # Convert to RGB for MediaPipe
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

                # Run all engines and aggregate labels.
                custom_labels_detected: Set[str] = set()
                for engine in self.engines:
                    labels = engine.process(frame, mp_image)
                    custom_labels_detected.update(labels)

                # Vision-level overlay
                h, w, _ = frame.shape
                cv2.putText(
                    frame,
                    f"FPS: {int(fps)}",
                    (w - 120, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Res(frame): {w}x{h}",
                    (w - 260, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

                # Action matching
                action = self.processor.process_frame(None, list(custom_labels_detected))
                if action:
                    print(f"Sequence Triggered: {action}")
                    if action == ("EXIT_JARVIS",):
                        print("[Vision Module] Exit gesture detected. Shutting down Jarvis...")
                        break
                    if self.sfx_command_queue is not None and action[0] == "SFX_PLAY":
                        self.sfx_command_queue.put(("play", action[1]))

                cv2.imshow("Jarvis Vision Module", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

