"""
Violence + Smoking + Fire/Smoke Detection  -- OPTIMIZED  (+ Person Counter)
----------------------------------------------------------------------------
Key accuracy improvements:
  - Scene validation: dark/blank/blurry frames are skipped (not inferred)
  - Raised thresholds with runtime calibration guidance
  - Violence model uses rolling mean of last N predictions (smoother)
  - Smoking model uses a dedicated YOLOv5 checkpoint for cigarette detection
  - api.py / server.py share identical thresholds via config block

Changes:
  - Violence detection only triggers when >= 2 persons are in frame
  - Smoking detection uses the provided YOLOv5 custom weights
"""

import cv2
import numpy as np
import argparse
import time
import os
import threading
import queue
from collections import deque
from datetime import datetime
from typing import Any

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    import mediapipe as mp
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False

HERE = os.path.dirname(os.path.abspath(__file__))


def _env_or_first_existing(env_name: str, *fallbacks: str) -> str:
    explicit = os.environ.get(env_name, "").strip()
    if explicit:
        return explicit
    for candidate in fallbacks:
        if candidate and os.path.exists(candidate):
            return candidate
    for candidate in fallbacks:
        if candidate:
            return candidate
    return ""


# ------------------------------------------------------------------------------
# CONFIGURATION  (edit here; api.py / server.py import from this module)
# ------------------------------------------------------------------------------
VIOLENCE_H5_PATH       = _env_or_first_existing("PLITHOS_VIOLENCE_MODEL", os.path.join(HERE, "modelnew.h5"), "modelnew.h5")
VIOLENCE_ONNX_PATH     = _env_or_first_existing("PLITHOS_VIOLENCE_ONNX", os.path.join(HERE, "modelnew.onnx"), "modelnew.onnx")
SMOKING_MODEL_PATH     = _env_or_first_existing("PLITHOS_SMOKING_MODEL", os.path.join(HERE, "weights (1).pt"), r"C:\Users\yousi\Downloads\weights (1).pt")
SMOKING_VERIFY_MODEL_PATH = _env_or_first_existing("PLITHOS_SMOKING_VERIFY_MODEL", "")
FIRE_MODEL_PATH        = _env_or_first_existing("PLITHOS_FIRE_MODEL", os.path.join(HERE, "Fire_best.pt"), "Fire_best.pt")

SMOKING_CLASS_NAMES = {0: "cigarette"}
SMOKING_ALLOWED_NAMES = {
    "cigarette", "cigar", "smoking", "smoke", "vape", "vaping",
    "e-cigarette", "ecigarette",
}
SMOKING_VERIFY_THRESHOLD = 0.25
SMOKING_VERIFY_OVERLAP = 0.25
SMOKING_PRIMARY_FALLBACK_CONF = 0.40
SMOKING_REQUIRE_PERSON = True
SMOKING_MIN_PERSONS = 1
SMOKING_PERSON_BOX_EXPAND = 0.20
SMOKING_PERSON_MIN_OVERLAP = 0.15

IMG_SIZE            = (128, 128)
SEQUENCE_LEN        = 1
VIOLENCE_CLASS      = 1

# ── Accuracy-critical thresholds ─────────────────────────────────────────────
VIOLENCE_THRESHOLD  = 0.30
VIOLENCE_SMOOTH     = 5
SMOKING_THRESHOLD   = 0.40
FIRE_THRESHOLD      = 0.55

# Consecutive-frame gates (extra guard on top of rolling mean)
VIOLENCE_CONSEC     = 3
SMOKING_CONSEC      = 2
FIRE_CONSEC         = 2

YOLO_IMGSZ          = 320

# Scene validation
MIN_BRIGHTNESS      = 20
MIN_VARIANCE        = 50
MIN_SHARPNESS       = 20

PERSON_CLASS_ID     = 0
PERSON_THRESHOLD    = 0.40
PERSON_MODEL_PATH   = _env_or_first_existing("PLITHOS_PERSON_MODEL", os.path.join(HERE, "yolov8n.pt"), "")

# ── Violence person gate ──────────────────────────────────────────────────────
# Violence detection will not fire unless at least this many persons are visible.
# A lone person or empty scene cannot constitute a crowd fight.
VIOLENCE_MIN_PERSONS = 2
# After the crowd threshold is reached, keep violence detection suppressed for
# this many continuous seconds before allowing alerts to fire.
VIOLENCE_PERSON_GRACE_SEC = 5.0

# Smoking detector compatibility aliases

ENABLE_FIRE         = True

SAVE_VIDEO          = True
LOG_FILE            = "detections.log"
MOTION_THRESHOLD    = 10
MAX_QUEUE_DEPTH     = 2
FPS_CAP             = 30

MODE                = "both"

# Backward-compatible aliases for the rest of the module
LITTER_MODEL_PATH = SMOKING_MODEL_PATH
LITTER_VERIFY_MODEL_PATH = SMOKING_VERIFY_MODEL_PATH
LITTER_CLASS_NAMES = SMOKING_CLASS_NAMES
LITTER_PLAUSIBLE_CLASSES = SMOKING_ALLOWED_NAMES
LITTER_VERIFY_THRESHOLD = SMOKING_VERIFY_THRESHOLD
LITTER_VERIFY_OVERLAP = SMOKING_VERIFY_OVERLAP
LITTER_PRIMARY_FALLBACK_CONF = SMOKING_PRIMARY_FALLBACK_CONF
LITTER_THRESHOLD = SMOKING_THRESHOLD
LITTER_CONSEC = SMOKING_CONSEC
# ------------------------------------------------------------------------------


# ==============================================================================
#  SCENE VALIDATION
# ==============================================================================

def is_valid_frame(frame: np.ndarray) -> tuple[bool, str]:
    """
    Returns (True, "") if the frame is suitable for inference.
    Returns (False, reason) if it should be skipped.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if MIN_BRIGHTNESS > 0:
        mean_brightness = float(gray.mean())
        if mean_brightness < MIN_BRIGHTNESS:
            return False, f"dark ({mean_brightness:.1f}<{MIN_BRIGHTNESS})"

    if MIN_VARIANCE > 0:
        variance = float(gray.var())
        if variance < MIN_VARIANCE:
            return False, f"blank ({variance:.1f}<{MIN_VARIANCE})"

    if MIN_SHARPNESS > 0:
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if lap_var < MIN_SHARPNESS:
            return False, f"blurry ({lap_var:.1f}<{MIN_SHARPNESS})"

    return True, ""


# ==============================================================================
#  HELPERS
# ==============================================================================

def _ort_providers():
    import onnxruntime as ort
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _use_half() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _infer_onnx_imgsz(model_path: str):
    if not str(model_path).lower().endswith(".onnx"):
        return None
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        shape = session.get_inputs()[0].shape
        if len(shape) >= 4:
            h, w = shape[2], shape[3]
            if isinstance(h, int) and isinstance(w, int) and h > 0 and w > 0:
                return h if h == w else (h, w)
    except Exception as exc:
        print(f"[WARN] Could not inspect ONNX input size for {model_path}: {exc}")
    return None


class LiveLitterDetector:
    """
    Legacy detector wrapper kept only for compatibility with older code paths.

    This version is designed for streaming use:
      - state is kept per camera/session
      - motion and background subtraction work across frames
      - event cooldown avoids duplicate alerts
      - short alert hold keeps the UI stable after a detection
    """

    def __init__(
        self,
        confidence_threshold: float = 0.50,
        yolo_model: str = "yolo26n.pt",
        use_pose: bool = True,
        yolo_imgsz: int = 320,
    ):
        self.confidence_threshold = max(float(confidence_threshold), MIN_EVENT_SCORE)
        self.yolo_model_path = yolo_model
        self.yolo_imgsz = int(yolo_imgsz)
        self.yolo = YOLO(yolo_model) if YOLO_AVAILABLE and yolo_model else None

        self.use_pose = bool(use_pose and MP_AVAILABLE)
        self.mp_pose = None
        self.pose_model = None
        if self.use_pose:
            try:
                self.mp_pose = mp.solutions.pose
                self.pose_model = self.mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=0,
                    min_detection_confidence=0.30,
                    min_tracking_confidence=0.30,
                )
            except Exception:
                self.use_pose = False
                self.mp_pose = None
                self.pose_model = None

        self.reset()

    def reset(self):
        self.bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=35, detectShadows=False
        )
        self.prev_gray = None
        self.prev_trash_score = 0.0
        self.frames_seen = 0
        self.last_event_t = -999.0
        self.active_until = -999.0
        self.last_detections: list[dict[str, Any]] = []
        self.last_confidence = 0.0
        self.last_signals: list[str] = []

    def set_confidence_threshold(self, threshold: float):
        self.confidence_threshold = max(float(threshold), MIN_EVENT_SCORE)

    def analyse_frame(self, frame_bgr: np.ndarray, now: float | None = None):
        if now is None:
            now = time.monotonic()

        self.frames_seen += 1

        h, w = frame_bgr.shape[:2]
        zones = self._compute_zones(w, h)

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        yolo_persons, yolo_litter, yolo_vehicles, yolo_blocked = self._yolo_detect(frame_bgr, zones, w, h)
        in_car = bool(yolo_vehicles) or self._is_car_scene(gray, w, h)
        litter_confirmed = bool(yolo_litter)
        blocked_in_ground = bool(yolo_blocked)

        arm_score = self._signal_arm(hsv, yolo_persons, zones)
        trash_score = self._signal_trash(gray, yolo_litter, zones, h)
        motion_score = self._signal_motion(gray, zones) if self.prev_gray is not None else 0.0
        bg_score = self._signal_bg_subtraction(frame_bgr, zones) if self.frames_seen > BG_WARMUP_FRAMES else 0.0
        pose_score, pose_loc = self._signal_pose(rgb, w, h, in_car, yolo_persons)
        yolo_score, yolo_loc = self._signal_yolo_combined(yolo_persons, yolo_litter, w, h, in_car)

        person_present = arm_score > 0.1 or bool(yolo_persons)

        final_score = 0.0
        active_signals: list[str] = []
        event_loc = None

        if arm_score > 0.2 and trash_score > 0.2:
            combined = arm_score * WEIGHT_ARM + trash_score * WEIGHT_TRASH
            if motion_score > 0.15:
                combined += motion_score * WEIGHT_MOTION
            final_score = max(final_score, combined)
            active_signals.append("arm+trash")
            event_loc = zones["event_loc"]

        if arm_score > 0.3 and motion_score > 0.3 and person_present:
            score = (arm_score * 0.5 + motion_score * 0.5) * 0.88
            final_score = max(final_score, score)
            active_signals.append("arm+motion")
            event_loc = event_loc or zones["window_center"]

        if bg_score > 0.3 and person_present:
            score = bg_score * 0.82
            final_score = max(final_score, score)
            active_signals.append("new_object")

        if pose_score > 0.35:
            final_score = max(final_score, pose_score)
            active_signals.append("pose_gesture")
            if pose_loc:
                event_loc = pose_loc

        if yolo_score > 0.35:
            final_score = max(final_score, yolo_score)
            active_signals.append("yolo_detection")
            if yolo_loc:
                event_loc = yolo_loc

        trash_delta = trash_score - self.prev_trash_score
        if trash_delta > 0.30 and person_present:
            score = min(0.88, 0.50 + trash_delta * 0.8)
            final_score = max(final_score, score)
            active_signals.append("trash_appeared")
            event_loc = event_loc or zones["ground_center"]

        if len(active_signals) >= 3:
            final_score = min(0.99, final_score * 1.20)
        elif len(active_signals) == 2:
            final_score = min(0.99, final_score * 1.10)

        strong_cv_case = (
            in_car
            and arm_score >= 0.35
            and trash_score >= 0.35
            and motion_score >= 0.20
            and (bg_score >= 0.08 or pose_score >= 0.35)
        )

        event_allowed = (litter_confirmed or strong_cv_case) and not (blocked_in_ground and not litter_confirmed)

        detections = self._build_detections(
            yolo_litter=yolo_litter,
        )

        event_triggered = (
            final_score >= self.confidence_threshold
            and bool(active_signals)
            and event_allowed
            and (now - self.last_event_t) >= EVENT_COOLDOWN_S
        )

        if event_triggered:
            self.last_event_t = now
            self.active_until = now + ALERT_HOLD_S
            self.last_detections = detections
            self.last_confidence = final_score
            self.last_signals = list(active_signals)
            is_littering = True
            out_detections = detections
            out_confidence = final_score
        elif now < self.active_until and self.last_detections:
            is_littering = True
            out_detections = self.last_detections
            out_confidence = self.last_confidence
        else:
            is_littering = False
            out_detections = []
            out_confidence = final_score

        meta = {
            "confidence": round(float(out_confidence), 4),
            "signals": list(active_signals if event_triggered else self.last_signals if is_littering else []),
            "in_car": bool(in_car),
            "litter_confirmed": litter_confirmed,
            "blocked_in_ground": blocked_in_ground,
        }

        self.prev_gray = gray
        self.prev_trash_score = trash_score
        return is_littering, out_detections, meta

    def _signal_arm(self, hsv, yolo_persons, zones) -> float:
        wx1, wy1, wx2, wy2 = zones["window"]
        window_hsv = hsv[wy1:wy2, wx1:wx2]

        mask1 = cv2.inRange(window_hsv, SKIN_HSV_LOWER, SKIN_HSV_UPPER)
        mask2 = cv2.inRange(window_hsv, SKIN_HSV_LOWER2, SKIN_HSV_UPPER2)
        skin_mask = cv2.bitwise_or(mask1, mask2)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_DILATE, kernel)

        skin_pixels = int(np.sum(skin_mask > 0))
        if skin_pixels < SKIN_PIXEL_THRESHOLD:
            base_score = 0.0
        else:
            base_score = min(
                1.0,
                (skin_pixels - SKIN_PIXEL_THRESHOLD) / (SKIN_PIXEL_THRESHOLD * 4.0) + 0.30,
            )

        if yolo_persons and base_score > 0:
            base_score = min(1.0, base_score * 1.20)
        return round(base_score, 3)

    def _signal_trash(self, gray, yolo_litter, zones, frame_h: int) -> float:
        gx1, gy1, gx2, gy2 = zones["ground"]
        ground_gray = gray[gy1:gy2, gx1:gx2]

        bright_mask = (ground_gray > TRASH_BRIGHT_THRESHOLD).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel)

        bright_pixels = int(np.sum(bright_mask > 0))
        if bright_pixels < TRASH_PIXEL_THRESHOLD:
            base_score = 0.0
        else:
            base_score = min(
                1.0,
                (bright_pixels - TRASH_PIXEL_THRESHOLD) / (TRASH_PIXEL_THRESHOLD * 5.0) + 0.30,
            )

        if yolo_litter:
            for litter_obj in yolo_litter:
                y_center = (litter_obj["bbox"][1] + litter_obj["bbox"][3]) / 2
                if y_center > frame_h * 0.50:
                    base_score = min(1.0, base_score + litter_obj["conf"] * 0.25)
                    break

        return round(base_score, 3)

    def _signal_motion(self, gray, zones) -> float:
        wx1, wy1, wx2, wy2 = zones["window"]
        curr_roi = gray[wy1:wy2, wx1:wx2]
        prev_roi = self.prev_gray[wy1:wy2, wx1:wx2]

        diff = cv2.absdiff(curr_roi, prev_roi)
        _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        motion_pixels = int(np.sum(thresh > 0))
        if motion_pixels < MOTION_PIXEL_THRESHOLD:
            return 0.0
        return round(
            min(
                1.0,
                (motion_pixels - MOTION_PIXEL_THRESHOLD) / (MOTION_PIXEL_THRESHOLD * 4.0) + 0.25,
            ),
            3,
        )

    def _signal_bg_subtraction(self, frame_bgr, zones) -> float:
        fg = self.bg_sub.apply(frame_bgr)
        gx1, gy1, gx2, gy2 = zones["ground"]
        ground_fg = fg[gy1:gy2, gx1:gx2]

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        ground_fg = cv2.morphologyEx(ground_fg, cv2.MORPH_OPEN, kernel)
        ground_fg = cv2.morphologyEx(ground_fg, cv2.MORPH_DILATE, kernel)

        fg_pixels = int(np.sum(ground_fg > 0))
        zone_area = max(1, (gx2 - gx1) * (gy2 - gy1))
        fg_ratio = fg_pixels / zone_area

        if fg_ratio < 0.01 or fg_ratio > 0.40:
            return 0.0
        return round(min(0.85, fg_ratio * 4.0), 3)

    def _signal_pose(self, rgb, frame_w, frame_h, in_car, yolo_persons):
        if not self.pose_model or not yolo_persons:
            return 0.0, None

        try:
            res = self.pose_model.process(rgb)
            if not res.pose_landmarks:
                return 0.0, None

            lm = res.pose_landmarks.landmark
            mp_pose = self.mp_pose

            rw = lm[mp_pose.PoseLandmark.RIGHT_WRIST]
            lw = lm[mp_pose.PoseLandmark.LEFT_WRIST]
            rs = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            ls = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]

            if in_car:
                best = 0.0
                best_loc = None
                for wrist, shoulder in ((rw, rs), (lw, ls)):
                    if wrist.visibility < 0.25:
                        continue
                    near_edge = wrist.x < 0.22 or wrist.x > 0.78
                    stretch = abs(wrist.x - shoulder.x)
                    if near_edge and stretch > 0.08:
                        score = min(0.92, stretch * wrist.visibility * 4.5)
                        if score > best:
                            best = score
                            best_loc = (int(wrist.x * frame_w), int(wrist.y * frame_h))
                return best, best_loc

            rk = lm[mp_pose.PoseLandmark.RIGHT_KNEE]
            lk = lm[mp_pose.PoseLandmark.LEFT_KNEE]
            rh = lm[mp_pose.PoseLandmark.RIGHT_HIP]
            lh = lm[mp_pose.PoseLandmark.LEFT_HIP]
            knee_y = (rk.y + lk.y) / 2
            hip_y = (rh.y + lh.y) / 2
            wrist_y = min(rw.y, lw.y)
            score = max(0.0, wrist_y - knee_y) * 0.55 + max(0.0, wrist_y - hip_y - 0.02) * 0.45
            if score < 0.03:
                return 0.0, None
            active = rw if rw.y > lw.y else lw
            return min(0.92, score * 5.5), (int(active.x * frame_w), int(active.y * frame_h))
        except Exception:
            return 0.0, None

    def _signal_yolo_combined(self, persons, litter, frame_w, frame_h, in_car):
        if not litter or not persons:
            return 0.0, None

        best = 0.0
        best_loc = None
        for litter_obj in litter:
            lx = (litter_obj["bbox"][0] + litter_obj["bbox"][2]) // 2
            ly = (litter_obj["bbox"][1] + litter_obj["bbox"][3]) // 2
            for person in persons:
                px1, py1, px2, py2 = person["bbox"]
                in_zone = (px1 - 150) <= lx <= (px2 + 150) and (py1 - 60) <= ly <= (py2 + 120)
                if not in_zone:
                    continue

                if in_car:
                    edge = lx / max(frame_w, 1) < 0.32 or lx / max(frame_w, 1) > 0.68
                    score = litter_obj["conf"] * person["conf"] * (0.92 if edge else 0.70)
                else:
                    lower = ly > frame_h * 0.38
                    score = litter_obj["conf"] * person["conf"] * (0.92 if lower else 0.72)

                if score > best:
                    best = score
                    best_loc = (lx, ly)

        return round(best, 3), best_loc

    def _build_detections(self, yolo_litter):
        detections: list[dict[str, Any]] = []

        if yolo_litter:
            for litter_obj in sorted(yolo_litter, key=lambda item: item["conf"], reverse=True)[:3]:
                detections.append({
                    "bbox": tuple(litter_obj["bbox"]),
                    "conf": float(litter_obj["conf"]),
                    "label": "smoking",
                })
        return detections

    def _compute_zones(self, frame_w: int, frame_h: int):
        if frame_h > frame_w:
            window = (int(frame_w * 0.15), int(frame_h * 0.25), int(frame_w * 0.85), int(frame_h * 0.55))
            ground = (int(frame_w * 0.05), int(frame_h * 0.55), int(frame_w * 0.75), int(frame_h * 0.78))
        else:
            window = (int(frame_w * 0.10), int(frame_h * 0.20), int(frame_w * 0.70), int(frame_h * 0.70))
            ground = (int(frame_w * 0.05), int(frame_h * 0.55), int(frame_w * 0.65), int(frame_h * 0.92))

        window_center = ((window[0] + window[2]) // 2, (window[1] + window[3]) // 2)
        ground_center = ((ground[0] + ground[2]) // 2, (ground[1] + ground[3]) // 2)
        event_loc = (window_center[0], (window[3] + ground[1]) // 2)
        return {
            "window": window,
            "ground": ground,
            "window_center": window_center,
            "ground_center": ground_center,
            "event_loc": event_loc,
        }

    def _is_car_scene(self, gray, frame_w: int, frame_h: int) -> bool:
        edges = cv2.Canny(gray, 35, 110)
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=55,
            minLineLength=int(frame_w * 0.10),
            maxLineGap=20,
        )
        if lines is None:
            return False
        return sum(1 for line in lines if abs(line[0][3] - line[0][1]) < 20) >= 3

    def _bbox_center(self, bbox):
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    def _bbox_in_zone(self, bbox, zone):
        cx, cy = self._bbox_center(bbox)
        x1, y1, x2, y2 = zone
        return x1 <= cx <= x2 and y1 <= cy <= y2

    def _bbox_area_ratio(self, bbox, frame_w: int, frame_h: int) -> float:
        x1, y1, x2, y2 = bbox
        area = max(0, x2 - x1) * max(0, y2 - y1)
        return area / max(1, frame_w * frame_h)

    def _yolo_detect(self, frame_bgr, zones, frame_w: int, frame_h: int):
        if not self.yolo:
            return [], [], [], []

        try:
            results = self.yolo.predict(frame_bgr, verbose=False, conf=0.25, imgsz=self.yolo_imgsz)[0]
            persons = []
            litter = []
            vehicles = []
            blocked = []
            for box in results.boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                bbox = [x1, y1, x2, y2]
                name = str(results.names.get(cls, str(cls))).lower()
                det = {"cls": cls, "conf": conf, "bbox": [x1, y1, x2, y2], "name": name}
                if cls == 0:
                    persons.append(det)
                elif cls in VEHICLE_IDS:
                    vehicles.append(det)
                elif (
                    name in LITTER_ALLOWED_NAMES
                    and conf >= LITTER_MIN_CONF
                    and self._bbox_in_zone(bbox, zones["ground"])
                    and self._bbox_area_ratio(bbox, frame_w, frame_h) >= LITTER_MIN_BOX_AREA
                ):
                    litter.append(det)
                elif (
                    name in LITTER_BLOCKED_NAMES
                    and self._bbox_in_zone(bbox, zones["ground"])
                ):
                    blocked.append(det)
            return persons, litter, vehicles, blocked
        except Exception:
            return [], [], [], []


class ViolencePersonGate:
    """
    Tracks how long the scene has continuously contained enough people for
    violence detection to be allowed.

    If the crowd drops below the minimum, the timer resets immediately.
    """

    def __init__(self, min_persons: int, grace_period_sec: float):
        self.min_persons = int(min_persons)
        self.grace_period_sec = max(0.0, float(grace_period_sec))
        self._crowd_since = None

    def update(self, person_count: int, now: float | None = None) -> tuple[bool, float]:
        if now is None:
            now = time.monotonic()

        if person_count >= self.min_persons:
            if self._crowd_since is None:
                self._crowd_since = now
            elapsed = max(0.0, now - self._crowd_since)
            remaining = max(0.0, self.grace_period_sec - elapsed)
            return remaining <= 0.0, remaining

        self._crowd_since = None
        return False, self.grace_period_sec

    def reset(self):
        self._crowd_since = None


# ==============================================================================
#  ONE-TIME CONVERSION
# ==============================================================================

def convert_h5_to_onnx(h5_path: str, onnx_path: str):
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    try:
        import tf2onnx
    except ImportError:
        raise ImportError("Run:  pip install tf2onnx onnx")

    print(f"[CONVERT] Loading {h5_path} ...")
    from tensorflow.keras.utils import custom_object_scope

    class FixedDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
        def __init__(self, *args, **kwargs):
            kwargs.pop('groups', None)
            super().__init__(*args, **kwargs)

    with custom_object_scope({'DepthwiseConv2D': FixedDepthwiseConv2D}):
        model = tf.keras.models.load_model(h5_path)

    input_sig = [tf.TensorSpec(model.inputs[0].shape, tf.float32, name="input")]
    print(f"[CONVERT] Converting to ONNX -> {onnx_path} ...")
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=input_sig, opset=13)

    import onnx
    onnx.save(model_proto, onnx_path)
    print(f"[CONVERT] Saved to: {onnx_path}")


# ==============================================================================
#  MODEL LOADERS
# ==============================================================================

def load_violence_model(h5_path: str, onnx_path: str):
    if os.path.exists(onnx_path):
        print(f"[INFO] Loading violence model (ONNX): {onnx_path}")
        import onnxruntime as ort
        sess       = ort.InferenceSession(onnx_path, providers=_ort_providers())
        input_name = sess.get_inputs()[0].name
        print(f"[INFO] Violence ONNX providers: {sess.get_providers()}")
        return ("onnx", sess, input_name)
    else:
        print(f"[INFO] ONNX not found — using Keras: {h5_path}")
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")
        from tensorflow.keras.utils import custom_object_scope

        class FixedDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
            def __init__(self, *args, **kwargs):
                kwargs.pop('groups', None)
                super().__init__(*args, **kwargs)

        with custom_object_scope({'DepthwiseConv2D': FixedDepthwiseConv2D}):
            model = tf.keras.models.load_model(h5_path)

        print(f"[INFO] Keras model input: {model.input_shape}")
        return ("keras", model, None)


def load_litter_model(path: str, verify_path: str | None = None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Smoking model not found: '{path}'")

    print(f"[INFO] Loading smoking model: {path}")
    backend = "ultralytics"
    if str(path).lower().endswith(".pt"):
        import torch
        import yolov5

        original_torch_load = torch.load

        def _legacy_torch_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return original_torch_load(*args, **kwargs)

        torch.load = _legacy_torch_load
        try:
            primary = yolov5.load(path, device="cuda" if _use_half() else "cpu")
        finally:
            torch.load = original_torch_load
        setattr(primary, "_plithos_imgsz", YOLO_IMGSZ)
        backend = "yolov5"
        print(f"[INFO] Smoking model classes: {getattr(primary, 'names', {})}")
    else:
        from ultralytics import YOLO

        primary = YOLO(path, task="detect")
        primary_imgsz = _infer_onnx_imgsz(path)
        if primary_imgsz is not None:
            setattr(primary, "_plithos_imgsz", primary_imgsz)
            print(f"[INFO] Smoking model input size: {primary_imgsz}")

    verifier = None

    if verify_path:
        if not os.path.exists(verify_path):
            raise FileNotFoundError(f"Smoking verifier model not found: '{verify_path}'")
        print(f"[INFO] Loading smoking verifier model: {verify_path}")
        from ultralytics import YOLO
        verifier = YOLO(verify_path, task="detect")
        verifier_imgsz = _infer_onnx_imgsz(verify_path)
        if verifier_imgsz is not None:
            setattr(verifier, "_plithos_imgsz", verifier_imgsz)
            print(f"[INFO] Smoking verifier input size: {verifier_imgsz}")

    return {
        "primary": primary,
        "verifier": verifier,
        "primary_path": path,
        "verify_path": verify_path,
        "backend": backend,
    }


def create_litter_detector(l_model_config):
    return l_model_config


def _normalise_model_names(model) -> dict[int, str]:
    names = getattr(model, "names", {}) or {}
    if isinstance(names, list):
        return {idx: str(name).lower() for idx, name in enumerate(names)}
    return {int(idx): str(name).lower() for idx, name in names.items()}


def _is_custom_litter_model(model) -> bool:
    names = set(_normalise_model_names(model).values())
    if not names:
        return False
    if any(name in names for name in SMOKING_ALLOWED_NAMES):
        return True
    return len(names) == 1 and any(
        "cigarette" in name or "smok" in name or "vape" in name
        for name in names
    )


def _yolo_result_boxes(model, frame: np.ndarray, conf: float):
    imgsz = getattr(model, "_plithos_imgsz", YOLO_IMGSZ)
    results = model.predict(frame, conf=conf, imgsz=imgsz,
                            verbose=False, half=_use_half())
    if not results:
        return []
    result = results[0]
    if result.boxes is None:
        return []

    names = _normalise_model_names(model)
    detections = []
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id = int(box.cls[0])
        detections.append({
            "bbox": (x1, y1, x2, y2),
            "conf": float(box.conf[0]),
            "cls": cls_id,
            "name": names.get(cls_id, str(cls_id)),
        })
    return detections


def _yolov5_result_boxes(model, frame: np.ndarray, conf: float):
    imgsz = getattr(model, "_plithos_imgsz", YOLO_IMGSZ)
    if hasattr(model, "conf"):
        model.conf = conf
    results = model(frame, size=imgsz)
    detections = []
    xyxy = getattr(results, "xyxy", None) or []
    if not xyxy:
        return detections

    names = getattr(model, "names", {}) or {}
    if isinstance(names, list):
        names = {idx: str(name).lower() for idx, name in enumerate(names)}
    else:
        names = {int(idx): str(name).lower() for idx, name in names.items()}

    for row in xyxy[0].tolist():
        x1, y1, x2, y2, score, cls_id = row[:6]
        cls_id = int(cls_id)
        detections.append({
            "bbox": (int(x1), int(y1), int(x2), int(y2)),
            "conf": float(score),
            "cls": cls_id,
            "name": names.get(cls_id, str(cls_id)),
        })
    return detections


def _intersection_over_smaller(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    smaller = max(1, min(area_a, area_b))
    return inter_area / smaller


def _box_center(box) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _expand_box(box, frame_shape, ratio: float):
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = box
    pad_x = int(max(2.0, (x2 - x1) * ratio))
    pad_y = int(max(2.0, (y2 - y1) * ratio))
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(w - 1, x2 + pad_x),
        min(h - 1, y2 + pad_y),
    )


def _smoking_matches_person(smoking_box, person_detections, frame_shape) -> bool:
    sx, sy = _box_center(smoking_box)
    for person in person_detections:
        expanded = _expand_box(person["bbox"], frame_shape, SMOKING_PERSON_BOX_EXPAND)
        px1, py1, px2, py2 = expanded
        if px1 <= sx <= px2 and py1 <= sy <= py2:
            return True
        if _intersection_over_smaller(smoking_box, expanded) >= SMOKING_PERSON_MIN_OVERLAP:
            return True
    return False


def enforce_smoking_person_gate(smoking_detections, person_detections, frame_shape):
    if not smoking_detections:
        return []
    if not SMOKING_REQUIRE_PERSON:
        return [dict(det) for det in smoking_detections]
    if len(person_detections or []) < SMOKING_MIN_PERSONS:
        return []

    accepted = []
    for det in smoking_detections:
        if _smoking_matches_person(det["bbox"], person_detections, frame_shape):
            filtered = dict(det)
            filtered["person_matched"] = True
            accepted.append(filtered)
    return accepted


def _extract_primary_litter_detections(model, frame: np.ndarray, *, backend: str = "ultralytics"):
    if backend == "yolov5":
        detections = _yolov5_result_boxes(model, frame, conf=LITTER_THRESHOLD)
    else:
        detections = _yolo_result_boxes(model, frame, conf=LITTER_VERIFY_THRESHOLD)
    if _is_custom_litter_model(model):
        return [
            {
                "bbox": det["bbox"],
                "conf": det["conf"],
                "label": det["name"],
                "primary_class": det["name"],
            }
            for det in detections
        ]

    return [
        {
            "bbox": det["bbox"],
            "conf": det["conf"],
            "label": det["name"],
            "primary_class": det["name"],
        }
        for det in detections
        if det["name"] in LITTER_PLAUSIBLE_CLASSES
    ]


def _extract_verifier_hits(model, frame: np.ndarray):
    return [
        det for det in _yolo_result_boxes(model, frame, conf=LITTER_VERIFY_THRESHOLD)
        if det["name"] in LITTER_PLAUSIBLE_CLASSES
    ]


def load_fire_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fire model not found: '{path}'")
    print(f"[INFO] Loading fire/smoke model: {path}")
    from ultralytics import YOLO
    mdl = YOLO(path)
    print(f"[INFO] Fire model classes: {mdl.names}")
    return mdl


def load_person_model():
    if PERSON_MODEL_PATH and os.path.exists(PERSON_MODEL_PATH):
        from ultralytics import YOLO
        return YOLO(PERSON_MODEL_PATH)
    print("[INFO] Person counter: loading yolov8n.pt")
    from ultralytics import YOLO
    return YOLO("yolov8n.pt")


# ==============================================================================
#  INFERENCE
# ==============================================================================

def preprocess_frame_violence(frame: np.ndarray) -> np.ndarray:
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, IMG_SIZE)
    return resized.astype(np.float32) / 255.0


def predict_violence(v_model_bundle, frame_buffer: deque,
                     conf_history: deque) -> tuple[bool, float]:
    """
    Returns (is_violent, smoothed_confidence).
    conf_history is a deque(maxlen=VIOLENCE_SMOOTH) maintained by the caller.
    NOTE: Person-count gate is applied by the caller (InferenceWorker / InferenceSession).
    """
    kind, model, input_name = v_model_bundle
    frames = np.array(list(frame_buffer))

    if kind == "onnx":
        inp   = frames[-1][np.newaxis]
        preds = model.run(None, {input_name: inp})[0][0]
    else:
        rank = len(model.input_shape)
        if rank == 4:
            inp = frames[-1][np.newaxis]
        else:
            seq = frames[-SEQUENCE_LEN:]
            if len(seq) < SEQUENCE_LEN:
                pad = np.zeros((SEQUENCE_LEN - len(seq), *seq.shape[1:]), dtype=np.float32)
                seq = np.concatenate([pad, seq], axis=0)
            inp = seq[np.newaxis]
        preds = model.predict(inp, verbose=0)[0]

    raw_conf = float(preds[0] if preds.shape[0] == 1 else preds[VIOLENCE_CLASS])
    conf_history.append(raw_conf)

    smoothed   = float(np.mean(conf_history))
    is_violent = smoothed >= VIOLENCE_THRESHOLD
    return is_violent, smoothed

def predict_litter(l_model, frame: np.ndarray):
    primary = (l_model or {}).get("primary")
    verifier = (l_model or {}).get("verifier")
    backend = str((l_model or {}).get("backend") or "ultralytics")
    if primary is None:
        return False, []

    primary_detections = _extract_primary_litter_detections(primary, frame, backend=backend)
    if not primary_detections:
        return False, []

    if backend == "yolov5":
        accepted = [
            {
                "bbox": det["bbox"],
                "conf": det["conf"],
                "label": det.get("label", det.get("primary_class", "cigarette")),
                "verified": True,
                "primary_class": det.get("primary_class", det.get("label", "cigarette")),
            }
            for det in primary_detections
            if det["conf"] >= LITTER_THRESHOLD
        ]
        return bool(accepted), accepted

    verifier_hits = _extract_verifier_hits(verifier, frame) if verifier is not None else []
    accepted = []

    for det in primary_detections:
        verified = any(
            _intersection_over_smaller(det["bbox"], hit["bbox"]) >= LITTER_VERIFY_OVERLAP
            for hit in verifier_hits
        )
        if verified or not verifier_hits or det["conf"] >= LITTER_PRIMARY_FALLBACK_CONF:
            accepted.append({
                "bbox": det["bbox"],
                "conf": det["conf"],
                "label": det["label"],
                "verified": verified,
                "primary_class": det.get("primary_class", det["label"]),
            })

    return bool(accepted), accepted


def load_smoking_model(path: str, verify_path: str | None = None):
    return load_litter_model(path, verify_path)


def predict_smoking(l_model, frame: np.ndarray, person_detections=None):
    is_smoking, smoking_detections = predict_litter(l_model, frame)
    if not is_smoking:
        return False, []
    filtered = enforce_smoking_person_gate(smoking_detections, person_detections or [], frame.shape)
    return bool(filtered), filtered


def predict_fire(f_model, frame: np.ndarray):
    results    = f_model.predict(frame, conf=FIRE_THRESHOLD, imgsz=YOLO_IMGSZ,
                                 verbose=False, half=_use_half())
    detections = []
    is_fire    = False
    is_smoke   = False
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf_val        = float(box.conf[0])
            cls_id          = int(box.cls[0])
            label           = f_model.names.get(cls_id, str(cls_id)).lower()
            detections.append({'bbox': (x1, y1, x2, y2), 'conf': conf_val, 'label': label})
            if 'fire' in label:
                is_fire  = True
            elif 'smoke' in label:
                is_smoke = True
            else:
                is_fire  = True
    return is_fire, is_smoke, detections


def predict_persons(p_model, frame: np.ndarray):
    results = p_model.predict(frame, conf=PERSON_THRESHOLD, imgsz=YOLO_IMGSZ,
                              classes=[PERSON_CLASS_ID], verbose=False, half=_use_half())
    persons = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            if int(box.cls[0]) != PERSON_CLASS_ID:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf_val        = float(box.conf[0])
            persons.append({'bbox': (x1, y1, x2, y2), 'conf': conf_val})
    return len(persons), persons


def has_motion(prev_gray, curr_gray) -> bool:
    if prev_gray is None:
        return True
    return float(cv2.absdiff(prev_gray, curr_gray).mean()) > MOTION_THRESHOLD


# ==============================================================================
#  DRAWING
# ==============================================================================

def draw_overlay(frame,
                 is_violent, v_conf,
                 is_littering, litter_detections,
                 is_fire, is_smoke, fire_detections,
                 person_count, person_detections,
                 fps_display,
                 frame_skipped=False):
    h, w = frame.shape[:2]

    run_v = MODE in ("both", "violence")
    run_l = MODE in ("both", "litter", "smoking")
    run_f = ENABLE_FIRE

    if run_f and (is_fire or is_smoke):
        if is_fire and is_smoke:
            color = (0, 60, 255);  label = "FIRE + SMOKE DETECTED"
        elif is_fire:
            color = (0, 60, 255);  label = "FIRE DETECTED"
        else:
            color = (0, 120, 200); label = "SMOKE DETECTED"
    elif run_v and is_violent and run_l and is_littering:
        color = (0, 0, 210);       label = "WARNING  VIOLENCE + SMOKING"
    elif run_v and is_violent:
        color = (0, 0, 210);       label = "WARNING  VIOLENCE DETECTED"
    elif run_l and is_littering:
        color = (0, 140, 255);     label = "WARNING  SMOKING DETECTED"
    else:
        color = (30, 180, 30);     label = "All Clear"

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 58), color, -1)
    cv2.addWeighted(overlay, 0.50, frame, 0.50, 0, frame)
    cv2.putText(frame, label, (14, 40),
                cv2.FONT_HERSHEY_DUPLEX, 0.95, (255, 255, 255), 2, cv2.LINE_AA)

    skip_tag = "  [SKIP]" if frame_skipped else ""
    if run_v:
        right_txt = f"Fight:{v_conf*100:.0f}%  {fps_display:.0f}fps{skip_tag}"
    else:
        right_txt = f"{fps_display:.0f}fps{skip_tag}"
    (tw, _), _ = cv2.getTextSize(right_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.putText(frame, right_txt, (w - tw - 14, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    active_modes = []
    if run_v: active_modes.append("VIOLENCE")
    if run_l: active_modes.append("SMOKING")
    if run_f: active_modes.append("FIRE")
    mode_label = "MODE: " + "+".join(active_modes)
    (mw, mh), _ = cv2.getTextSize(mode_label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.rectangle(frame, (w - mw - 16, 62), (w - 4, 62 + mh + 8), (40, 40, 40), -1)
    cv2.putText(frame, mode_label, (w - mw - 10, 62 + mh + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    if run_v:
        v_color = (0, 0, 210) if is_violent else (30, 180, 30)
        cv2.rectangle(frame, (0, 0), (int(w * v_conf), 6), v_color, -1)
        cv2.rectangle(frame, (0, 0), (w, 6), (80, 80, 80), 1)

    if run_f:
        if is_fire:
            pulse = frame.copy()
            cv2.rectangle(pulse, (0, 0), (w, h), (0, 0, 180), -1)
            cv2.addWeighted(pulse, 0.08, frame, 0.92, 0, frame)
        for det in fire_detections:
            x1, y1, x2, y2 = det['bbox']
            lbl             = det['label']
            conf_val        = det['conf']
            box_color = (0, 60, 255) if 'fire' in lbl else (160, 120, 60)
            tag = f"{lbl} {conf_val*100:.0f}%"
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            (bw, bh), bl = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            ty1 = max(y1 - bh - bl - 4, 0)
            cv2.rectangle(frame, (x1, ty1), (x1 + bw + 6, y1 + bl), box_color, -1)
            cv2.putText(frame, tag, (x1 + 3, y1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        if fire_detections:
            fire_cnt  = sum(1 for d in fire_detections if 'fire'  in d['label'])
            smoke_cnt = sum(1 for d in fire_detections if 'smoke' in d['label'])
            other_cnt = len(fire_detections) - fire_cnt - smoke_cnt
            parts = []
            if fire_cnt:  parts.append(f"Fire:{fire_cnt}")
            if smoke_cnt: parts.append(f"Smoke:{smoke_cnt}")
            if other_cnt: parts.append(f"Other:{other_cnt}")
            cv2.putText(frame, "  ".join(parts), (14, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 100, 255), 2, cv2.LINE_AA)

    if run_l:
        for det in litter_detections:
            x1, y1, x2, y2 = det['bbox']
            tag = f"{det['label']} {det['conf']*100:.0f}%"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 140, 255), 2)
            (lw, lh), bl = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            ty1 = max(y1 - lh - bl - 4, 0)
            cv2.rectangle(frame, (x1, ty1), (x1 + lw + 6, y1 + bl), (0, 140, 255), -1)
            cv2.putText(frame, tag, (x1 + 3, y1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        if litter_detections:
            cv2.putText(frame, f"Smoking objects: {len(litter_detections)}", (14, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2, cv2.LINE_AA)

    for p in person_detections:
        x1, y1, x2, y2 = p['bbox']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
        ptag = f"person {p['conf']*100:.0f}%"
        (plw, plh), pbl = cv2.getTextSize(ptag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        pty1 = max(y1 - plh - pbl - 4, 0)
        cv2.rectangle(frame, (x1, pty1), (x1 + plw + 6, y1 + pbl), (255, 200, 0), -1)
        cv2.putText(frame, ptag, (x1 + 3, y1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)

    badge_w, badge_h = 110, 60
    bx1 = w - badge_w - 10
    by1 = h - badge_h - 10
    bx2, by2 = w - 10, h - 10
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (50, 50, 50), -1)
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 200, 0), 2)
    badge_lbl = "PERSONS"
    (slw, slh), _ = cv2.getTextSize(badge_lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    cv2.putText(frame, badge_lbl, (bx1 + (badge_w - slw) // 2, by1 + slh + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 200, 0), 1, cv2.LINE_AA)
    count_str = str(person_count)
    (nw, nh), _ = cv2.getTextSize(count_str, cv2.FONT_HERSHEY_DUPLEX, 1.5, 3)
    cv2.putText(frame, count_str, (bx1 + (badge_w - nw) // 2, by2 - 10),
                cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)

    if run_f:
        fb_w, fb_h = 130, 60
        fbx1, fby1 = 10, h - fb_h - 10
        fbx2, fby2 = fbx1 + fb_w, fby1 + fb_h
        fb_bg     = (0, 30, 140) if is_fire else (30, 60, 100) if is_smoke else (50, 50, 50)
        fb_border = (0, 60, 255) if is_fire else (160, 120, 60) if is_smoke else (120, 120, 120)
        cv2.rectangle(frame, (fbx1, fby1), (fbx2, fby2), fb_bg, -1)
        cv2.rectangle(frame, (fbx1, fby1), (fbx2, fby2), fb_border, 2)
        fb_lbl = "FIRE / SMOKE"
        (flw, flh), _ = cv2.getTextSize(fb_lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
        cv2.putText(frame, fb_lbl, (fbx1 + (fb_w - flw) // 2, fby1 + flh + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, fb_border, 1, cv2.LINE_AA)
        if is_fire and is_smoke: status_txt = "FIRE+SMOKE"
        elif is_fire:            status_txt = "  FIRE"
        elif is_smoke:           status_txt = " SMOKE"
        else:                    status_txt = " CLEAR"
        (stw, sth), _ = cv2.getTextSize(status_txt, cv2.FONT_HERSHEY_DUPLEX, 0.75, 2)
        cv2.putText(frame, status_txt, (fbx1 + (fb_w - stw) // 2, fby2 - 10),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(frame, datetime.now().strftime("%Y-%m-%d  %H:%M:%S"), (14, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 210, 210), 1, cv2.LINE_AA)
    return frame


# ==============================================================================
#  INFERENCE WORKER
# ==============================================================================

class InferenceWorker(threading.Thread):
    def __init__(self, v_model_bundle, l_model, f_model, p_model, in_q, out_q, log_f):
        super().__init__(daemon=True)
        self.v_model   = v_model_bundle
        self.l_model   = l_model
        self.f_model   = f_model
        self.p_model   = p_model
        self.in_q      = in_q
        self.out_q     = out_q
        self.log_f     = log_f
        self.frame_buf   = deque(maxlen=max(SEQUENCE_LEN, 1))
        self.conf_hist   = deque(maxlen=VIOLENCE_SMOOTH)
        self.running     = True
        self._v_streak   = 0
        self._l_streak   = 0
        self._f_streak   = 0
        self._v_active   = False
        self._l_active   = False
        self._f_active   = False
        self._violence_person_gate = ViolencePersonGate(
            VIOLENCE_MIN_PERSONS, VIOLENCE_PERSON_GRACE_SEC)

    def _smooth(self, raw_flag, streak_attr, active_attr, consec):
        streak = getattr(self, streak_attr)
        active = getattr(self, active_attr)
        if raw_flag:
            streak += 1
        else:
            streak  = 0
            active  = False
        if streak >= consec:
            active = True
        setattr(self, streak_attr, streak)
        setattr(self, active_attr, active)
        return active

    def run(self):
        run_v = MODE in ("both", "violence")
        run_l = MODE in ("both", "litter", "smoking")
        run_f = ENABLE_FIRE and self.f_model is not None

        while self.running:
            try:
                frame, frame_count = self.in_q.get(timeout=1.0)
            except queue.Empty:
                continue

            # ── Scene validation ──────────────────────────────────────────
            valid, reason = is_valid_frame(frame)
            if not valid:
                self.out_q.put((False, 0.0,
                                False, [],
                                False, False, [],
                                0, [],
                                True))   # frame_skipped=True
                self.in_q.task_done()
                continue

            v_res = [False, 0.0]
            l_res = [False, []]
            f_res = [False, False, []]
            p_res = [0, []]

            threads = []

            if run_v:
                self.frame_buf.append(preprocess_frame_violence(frame))
                ch = self.conf_hist
                def run_vf(buf=self.frame_buf, ch=ch):
                    v_res[0], v_res[1] = predict_violence(self.v_model, buf, ch)
                threads.append(threading.Thread(target=run_vf))

            if run_f:
                def run_ff(fr=frame):
                    f_res[0], f_res[1], f_res[2] = predict_fire(self.f_model, fr)
                threads.append(threading.Thread(target=run_ff))

            def run_pf(fr=frame):
                p_res[0], p_res[1] = predict_persons(self.p_model, fr)
            threads.append(threading.Thread(target=run_pf))

            for t in threads: t.start()
            for t in threads: t.join()

            is_violent, v_conf        = v_res
            is_fire, is_smoke, fire_d = f_res
            person_count, person_d    = p_res

            if run_l:
                is_littering, litter_d = predict_smoking(self.l_model, frame, person_d)
            else:
                is_littering, litter_d = False, []

            # ── Violence person-gate ──────────────────────────────────────
            # Require at least VIOLENCE_MIN_PERSONS people in frame before
            # violence can be flagged, and keep a short grace period before
            # enabling detection so transient passers-by do not immediately
            # arm the violence model. If the crowd drops below the threshold,
            # the timer and streak reset.
            if run_v:
                gate_ready, _ = self._violence_person_gate.update(person_count)
            else:
                gate_ready = False

            if run_v and not gate_ready:
                is_violent       = False
                v_conf           = 0.0
                self._v_streak   = 0
                self._v_active   = False

            # Consecutive-frame temporal smoothing
            is_violent   = self._smooth(is_violent,   '_v_streak', '_v_active', VIOLENCE_CONSEC)
            is_littering = self._smooth(is_littering, '_l_streak', '_l_active', LITTER_CONSEC)
            _fire_raw    = is_fire or is_smoke
            _fire_smooth = self._smooth(_fire_raw,    '_f_streak', '_f_active', FIRE_CONSEC)
            if not _fire_smooth:
                is_fire = is_smoke = False
                fire_d  = []

            ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            if run_v and is_violent:
                msg = f"[ALERT] Violence  | frame {frame_count:6d} | conf {v_conf:.3f} | persons {person_count} | {ts}"
                print(msg)
                if self.log_f: self.log_f.write(msg + "\n"); self.log_f.flush()
            if run_l and is_littering:
                msg = f"[ALERT] Smoking | frame {frame_count:6d} | objects {len(litter_d)} | persons {person_count} | {ts}"
                print(msg)
                if self.log_f: self.log_f.write(msg + "\n"); self.log_f.flush()
            if run_f and (is_fire or is_smoke):
                kind_str = ("FIRE+SMOKE" if (is_fire and is_smoke)
                            else "FIRE" if is_fire else "SMOKE")
                msg = f"[ALERT] {kind_str:<10} | frame {frame_count:6d} | objects {len(fire_d)} | {ts}"
                print(msg)
                if self.log_f: self.log_f.write(msg + "\n"); self.log_f.flush()

            if self.out_q.full():
                try: self.out_q.get_nowait()
                except queue.Empty: pass
            self.out_q.put((is_violent, v_conf,
                            is_littering, litter_d,
                            is_fire, is_smoke, fire_d,
                            person_count, person_d,
                            False))   # frame_skipped=False
            self.in_q.task_done()


# ==============================================================================
#  MAIN LOOP
# ==============================================================================

def open_log(path):
    if not path: return None
    f = open(path, "a", encoding="utf-8")
    f.write(f"\n{'='*60}\nSession started: {datetime.now()}  "
            f"[mode={MODE}  fire={'ON' if ENABLE_FIRE else 'OFF'}]\n{'='*60}\n")
    return f


def warmup(v_model, l_model, f_model, p_model):
    print("[INFO] Warming up models ...")
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy[:] = 128
    dummy_hist = deque(maxlen=VIOLENCE_SMOOTH)
    if v_model is not None:
        buf = deque([preprocess_frame_violence(dummy)], maxlen=1)
        predict_violence(v_model, buf, dummy_hist)
    if l_model is not None:
        predict_litter(l_model, dummy)
    if f_model is not None:
        predict_fire(f_model, dummy)
    if p_model is not None:
        predict_persons(p_model, dummy)
    print("[INFO] Warmup done.")


def run(source):
    run_v = MODE in ("both", "violence")
    run_l = MODE in ("both", "litter", "smoking")

    print(f"[INFO] Mode: {MODE.upper()}  |  "
          f"Violence={'ON' if run_v else 'OFF'}  |  "
          f"Smoking={'ON' if run_l else 'OFF'}  |  "
          f"Fire/Smoke={'ON' if ENABLE_FIRE else 'OFF'}  |  "
          f"PersonCounter=ON")
    print(f"[INFO] Thresholds: violence={VIOLENCE_THRESHOLD}  "
          f"smoking={LITTER_THRESHOLD}  fire={FIRE_THRESHOLD}")
    print(f"[INFO] Scene validation: brightness>={MIN_BRIGHTNESS}  "
          f"variance>={MIN_VARIANCE}  sharpness>={MIN_SHARPNESS}")
    print(f"[INFO] Violence min persons: {VIOLENCE_MIN_PERSONS}")
    print(f"[INFO] Violence grace period: {VIOLENCE_PERSON_GRACE_SEC:.1f}s")
    print(f"[INFO] Smoking primary model: {LITTER_MODEL_PATH}")
    if LITTER_VERIFY_MODEL_PATH:
        print(f"[INFO] Smoking verifier model: {LITTER_VERIFY_MODEL_PATH}")
    else:
        print("[INFO] Smoking verifier model: disabled")
    print(f"[INFO] Smoking classes: {sorted(LITTER_PLAUSIBLE_CLASSES)}")

    v_model = load_violence_model(VIOLENCE_H5_PATH, VIOLENCE_ONNX_PATH) if run_v else None
    l_model = load_litter_model(LITTER_MODEL_PATH, LITTER_VERIFY_MODEL_PATH) if run_l else None
    f_model = load_fire_model(FIRE_MODEL_PATH)                           if ENABLE_FIRE else None
    p_model = load_person_model()

    warmup(v_model, l_model, f_model, p_model)

    src = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {source}")

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Source: {source}  {width}x{height} @ {fps:.1f}fps")

    writer = None
    if SAVE_VIDEO:
        out_path = f"output_{MODE}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        writer   = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        print(f"[INFO] Saving to: {out_path}")

    log_f  = open_log(LOG_FILE)
    in_q   = queue.Queue(maxsize=MAX_QUEUE_DEPTH)
    out_q  = queue.Queue(maxsize=MAX_QUEUE_DEPTH)
    worker = InferenceWorker(v_model, l_model, f_model, p_model, in_q, out_q, log_f)
    worker.start()

    frame_count    = 0
    prev_gray      = None
    is_violent     = False;  v_conf      = 0.0
    is_littering   = False;  litter_dets = []
    is_fire        = False;  is_smoke    = False;  fire_dets = []
    person_count   = 0;      person_dets = []
    fps_display    = 0.0
    frame_skipped  = False
    t_fps          = time.time();  fps_counter = 0
    violence_count = 0;  smoking_count = 0;  fire_count = 0;  smoke_count = 0
    t_start        = time.time()
    frame_interval = (1.0 / FPS_CAP) if FPS_CAP > 0 else 0.0
    t_last_frame   = 0.0

    win_title = "Detector [VIOLENCE+SMOKING+FIRE]  (Q = quit)"
    cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)
    print(f"[INFO] FPS cap: {FPS_CAP if FPS_CAP > 0 else 'disabled'}")
    print("[INFO] Running — press  Q  to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of stream.")
            break

        if frame_interval > 0:
            now = time.time()
            sleep_for = frame_interval - (now - t_last_frame)
            if sleep_for > 0:
                time.sleep(sleep_for)
        t_last_frame = time.time()

        frame_count += 1
        fps_counter += 1
        if fps_counter >= 30:
            fps_display = fps_counter / (time.time() - t_fps)
            t_fps       = time.time()
            fps_counter = 0

        curr_gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (160, 120))
        if has_motion(prev_gray, curr_gray):
            prev_gray = curr_gray
            if not in_q.full():
                in_q.put_nowait((frame.copy(), frame_count))

        try:
            result = out_q.get_nowait()
            if len(result) == 10:
                (is_violent, v_conf,
                 is_littering, litter_dets,
                 is_fire, is_smoke, fire_dets,
                 person_count, person_dets,
                 frame_skipped) = result
            else:
                (is_violent, v_conf,
                 is_littering, litter_dets,
                 is_fire, is_smoke, fire_dets,
                 person_count, person_dets) = result
                frame_skipped = False

            if is_violent:   violence_count += 1
            if is_littering: smoking_count  += 1
            if is_fire:      fire_count      += 1
            if is_smoke:     smoke_count     += 1
        except queue.Empty:
            pass

        annotated = draw_overlay(
            frame,
            is_violent, v_conf,
            is_littering, litter_dets,
            is_fire, is_smoke, fire_dets,
            person_count, person_dets,
            fps_display,
            frame_skipped=frame_skipped,
        )
        if writer: writer.write(annotated)
        cv2.imshow(win_title, annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[INFO] User quit.")
            break

    worker.running = False
    worker.join(timeout=3)
    elapsed = time.time() - t_start
    cap.release()
    if writer: writer.release()
    cv2.destroyAllWindows()

    if log_f:
        log_f.write(
            f"\nSession ended     : {datetime.now()}\n"
            f"Mode              : {MODE}\n"
            f"Frames processed  : {frame_count}\n"
            f"Violence alerts   : {violence_count}\n"
            f"Smoking alerts    : {smoking_count}\n"
            f"Fire alerts       : {fire_count}\n"
            f"Smoke alerts      : {smoke_count}\n"
            f"Elapsed           : {elapsed:.1f}s\n"
        )
        log_f.close()

    print(f"\n[SUMMARY] Mode:{MODE} | Frames:{frame_count} | "
          f"Violence:{violence_count} | Smoking:{smoking_count} | "
          f"Fire:{fire_count} | Smoke:{smoke_count} | Time:{elapsed:.1f}s")


# ------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Violence + Smoking + Fire/Smoke Detection - Optimized")
    parser.add_argument("--source",              default="0")
    parser.add_argument(
        "--mode",
        default="both",
        metavar="{both,violence,smoking}",
        help="Detection mode to run",
    )
    parser.add_argument("--violence-model",      default=VIOLENCE_H5_PATH)
    parser.add_argument("--violence-onnx",       default=VIOLENCE_ONNX_PATH)
    parser.add_argument("--smoking-model",       dest="smoking_model", default=SMOKING_MODEL_PATH)
    parser.add_argument("--smoking-verify-model", dest="smoking_verify_model", default=SMOKING_VERIFY_MODEL_PATH)
    parser.add_argument("--litter-model",        dest="smoking_model", help=argparse.SUPPRESS)
    parser.add_argument("--litter-verify-model", dest="smoking_verify_model", help=argparse.SUPPRESS)
    parser.add_argument("--fire-model",          default=FIRE_MODEL_PATH)
    parser.add_argument("--no-fire",             action="store_true")
    parser.add_argument("--fire-threshold",      type=float, default=FIRE_THRESHOLD)
    parser.add_argument("--person-model",        default=PERSON_MODEL_PATH)
    parser.add_argument("--person-class-id",     type=int,   default=PERSON_CLASS_ID)
    parser.add_argument("--person-threshold",    type=float, default=PERSON_THRESHOLD)
    parser.add_argument("--violence-threshold",  type=float, default=VIOLENCE_THRESHOLD)
    parser.add_argument("--smoking-threshold",   dest="smoking_threshold", type=float, default=SMOKING_THRESHOLD)
    parser.add_argument("--smoking-verify-threshold",
                        dest="smoking_verify_threshold", type=float, default=SMOKING_VERIFY_THRESHOLD)
    parser.add_argument("--smoking-verify-overlap",
                        dest="smoking_verify_overlap", type=float, default=SMOKING_VERIFY_OVERLAP)
    parser.add_argument("--smoking-primary-fallback-conf",
                        dest="smoking_primary_fallback_conf", type=float, default=SMOKING_PRIMARY_FALLBACK_CONF)
    parser.add_argument("--litter-threshold",    dest="smoking_threshold", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--litter-verify-threshold",
                        dest="smoking_verify_threshold", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--litter-verify-overlap",
                        dest="smoking_verify_overlap", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--litter-primary-fallback-conf",
                        dest="smoking_primary_fallback_conf", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--violence-min-persons",type=int,   default=VIOLENCE_MIN_PERSONS)
    parser.add_argument("--violence-person-grace", type=float,
                        default=VIOLENCE_PERSON_GRACE_SEC)
    parser.add_argument("--yolo-size",           type=int,   default=YOLO_IMGSZ)
    parser.add_argument("--motion-threshold",    type=int,   default=MOTION_THRESHOLD)
    parser.add_argument("--fps-cap",             type=int,   default=FPS_CAP)
    parser.add_argument("--min-brightness",      type=int,   default=MIN_BRIGHTNESS)
    parser.add_argument("--min-variance",        type=int,   default=MIN_VARIANCE)
    parser.add_argument("--min-sharpness",       type=int,   default=MIN_SHARPNESS)
    parser.add_argument("--violence-smooth",     type=int,   default=VIOLENCE_SMOOTH)
    parser.add_argument("--convert-onnx",        action="store_true")
    args = parser.parse_args()

    mode_value = str(args.mode).strip().lower()
    if mode_value == "litter":
        mode_value = "smoking"
    if mode_value not in {"both", "violence", "smoking"}:
        parser.error("argument --mode: choose from both, violence, or smoking")

    MODE                 = mode_value
    VIOLENCE_H5_PATH     = args.violence_model
    VIOLENCE_ONNX_PATH   = args.violence_onnx
    SMOKING_MODEL_PATH   = args.smoking_model
    SMOKING_VERIFY_MODEL_PATH = args.smoking_verify_model
    LITTER_MODEL_PATH    = SMOKING_MODEL_PATH
    LITTER_VERIFY_MODEL_PATH = SMOKING_VERIFY_MODEL_PATH
    FIRE_MODEL_PATH      = args.fire_model
    ENABLE_FIRE          = not args.no_fire
    FIRE_THRESHOLD       = args.fire_threshold
    PERSON_MODEL_PATH    = args.person_model
    PERSON_CLASS_ID      = args.person_class_id
    PERSON_THRESHOLD     = args.person_threshold
    VIOLENCE_THRESHOLD   = args.violence_threshold
    SMOKING_THRESHOLD    = args.smoking_threshold
    SMOKING_VERIFY_THRESHOLD = args.smoking_verify_threshold
    SMOKING_VERIFY_OVERLAP = args.smoking_verify_overlap
    SMOKING_PRIMARY_FALLBACK_CONF = args.smoking_primary_fallback_conf
    LITTER_THRESHOLD     = SMOKING_THRESHOLD
    LITTER_VERIFY_THRESHOLD = SMOKING_VERIFY_THRESHOLD
    LITTER_VERIFY_OVERLAP = SMOKING_VERIFY_OVERLAP
    LITTER_PRIMARY_FALLBACK_CONF = SMOKING_PRIMARY_FALLBACK_CONF
    VIOLENCE_MIN_PERSONS = args.violence_min_persons
    VIOLENCE_PERSON_GRACE_SEC = args.violence_person_grace
    YOLO_IMGSZ           = args.yolo_size
    MOTION_THRESHOLD     = args.motion_threshold
    FPS_CAP              = args.fps_cap
    MIN_BRIGHTNESS       = args.min_brightness
    MIN_VARIANCE         = args.min_variance
    MIN_SHARPNESS        = args.min_sharpness
    VIOLENCE_SMOOTH      = args.violence_smooth

    if args.convert_onnx:
        convert_h5_to_onnx(VIOLENCE_H5_PATH, VIOLENCE_ONNX_PATH)
    else:
        run(args.source)
